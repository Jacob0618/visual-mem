"""Gemini client used by every experiment in this project.

- generate(): text/image generation with a persistent sqlite cache (identical request -> identical stored response,
  so re-running an experiment never re-bills and is bit-for-bit reproducible), retries, and per-call usage logging.
- embed(): gemini-embedding-2 (multimodal) / gemini-embedding-001 embeddings, cached per item.
- pmap(): thread-pool map with progress.
- spend(): USD estimate from logged usage (prices verified on ai.google.dev/gemini-api/docs/pricing, 2026-10-26).

Images are referenced by path; cache keys use the sha256 of the image bytes + requested media resolution.
Large requests (> ~15 MB inline) automatically upload images through the Files API (URIs cached for 46 h).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keys import gemini_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

PRICES = {  # USD per 1M tokens (standard tier, through 2026-12-31)
    "gemini-3.8-flash": {"in": 0.75, "cached": 0.075, "out": 3.75},
    "gemini-2.5-flash-lite": {"in": 0.10, "cached": 0.01, "out": 0.40},
    "gemini-2.5-flash": {"in": 0.30, "cached": 0.03, "out": 2.50},
    "gemini-3.5-flash-lite": {"in": 0.30, "cached": 0.03, "out": 2.50},
    "gemini-3.1-flash-lite": {"in": 0.25, "cached": 0.025, "out": 1.50},
    "gemma-4-31b-it": {"in": 0.0, "cached": 0.0, "out": 0.0},  # free of charge (rate-limited)
    "gemini-embedding-2": {"text": 0.20, "image": 0.45},
    "gemini-embedding-001": {"text": 0.15},
}
DEFAULT_MODEL = "gemini-3.8-flash"
FRESH_TAGS: list = []   # tags whose calls bypass the local cache (cost measurement of API-side prefix caching)
EMB_MODEL = "gemini-embedding-2"

_client = None
_client_lock = threading.Lock()


HTTP_TIMEOUT_S = 240


def client():
    global _client
    with _client_lock:
        if _client is None:
            from google import genai
            from google.genai import types
            _client = genai.Client(api_key=gemini_key(), http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_S * 1000))
        return _client


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hkey(obj) -> str:
    return _sha(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode())


# --------------------------------------------------------------------------- images
_img_sha_cache: dict = {}
_img_lock = threading.Lock()


@dataclass(frozen=True)
class Img:
    """An image part. res in {None, 'low', 'medium', 'high', 'ultra_high'} (per-part media resolution)."""
    path: str
    res: Optional[str] = None

    @property
    def sha(self) -> str:
        with _img_lock:
            s = _img_sha_cache.get(self.path)
        if s is None:
            s = _sha(Path(self.path).read_bytes())
            with _img_lock:
                _img_sha_cache[self.path] = s
        return s

    @property
    def mime(self) -> str:
        ext = Path(self.path).suffix.lower()
        return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext, "image/png")

    def key(self):
        return {"img": self.sha, "res": self.res}


def _img_bytes_for_api(img: Img) -> tuple[bytes, str]:
    """PNG/JPEG pass through; other formats are converted to PNG (embedding API accepts PNG/JPEG only)."""
    data = Path(img.path).read_bytes()
    if img.mime in ("image/png", "image/jpeg"):
        return data, img.mime
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue(), "image/png"


# Files API (for requests whose inline payload would exceed the 20 MB limit)
_FILES_DB = CACHE_DIR / "files_api.json"
_files_lock = threading.Lock()
_files_mem: dict = {}
_sha_locks: dict = {}


def _file_uri(img: Img) -> tuple[str, str]:
    sha = img.sha
    with _files_lock:
        if not _files_mem and _FILES_DB.exists():
            _files_mem.update(json.loads(_FILES_DB.read_text()))
        hit = _files_mem.get(sha)
        if hit and time.time() - hit["t"] < 46 * 3600:
            return hit["uri"], hit["mime"]
        lk = _sha_locks.setdefault(sha, threading.Lock())
    with lk:  # one upload per image even under concurrency
        with _files_lock:
            hit = _files_mem.get(sha)
            if hit and time.time() - hit["t"] < 46 * 3600:
                return hit["uri"], hit["mime"]
        return _upload(img)


def ensure_uploaded(paths, workers: int = 16):
    """Pre-upload images to the Files API (one-time per 46 h) so generation requests only carry URIs."""
    pmap(lambda p: _file_uri(Img(p)), sorted(set(paths)), workers=workers, desc="upload")


def _upload(img: Img) -> tuple[str, str]:
    data, mime = _img_bytes_for_api(img)
    for attempt in range(6):
        try:
            f = client().files.upload(file=io.BytesIO(data), config={"mime_type": mime})
            break
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt + random.random())
    while getattr(f, "state", None) is not None and str(f.state).endswith("PROCESSING"):
        time.sleep(1)
        f = client().files.get(name=f.name)
    with _files_lock:
        _files_mem[img.sha] = {"uri": f.uri, "mime": mime, "t": time.time()}
        _FILES_DB.write_text(json.dumps(_files_mem))
    return f.uri, mime


_RES_LEVEL = {"low": "MEDIA_RESOLUTION_LOW", "medium": "MEDIA_RESOLUTION_MEDIUM", "high": "MEDIA_RESOLUTION_HIGH",
              "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH"}


def _to_parts(contents: Sequence[Union[str, Img]], use_files: bool):
    from google.genai import types
    parts = []
    for c in contents:
        if isinstance(c, str):
            parts.append(types.Part.from_text(text=c))
        elif isinstance(c, Img):
            mr = types.PartMediaResolution(level=_RES_LEVEL[c.res]) if c.res else None
            if use_files:
                uri, mime = _file_uri(c)
                p = types.Part(file_data=types.FileData(file_uri=uri, mime_type=mime), media_resolution=mr)
            else:
                data, mime = _img_bytes_for_api(c)
                p = types.Part(inline_data=types.Blob(data=data, mime_type=mime), media_resolution=mr)
            parts.append(p)
        else:
            raise TypeError(type(c))
    return parts


# --------------------------------------------------------------------------- sqlite cache
class _KV:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.con = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, tag TEXT, t REAL)")
        self.con.commit()

    def get(self, k):
        with self.lock:
            r = self.con.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def set(self, k, v, tag=None):
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?,?)", (k, json.dumps(v), tag, time.time()))
            self.con.commit()


class _EmbKV:
    def __init__(self, path: Path):
        self.lock = threading.Lock()
        self.con = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB, ntext INTEGER, nimg INTEGER, t REAL)")
        self.con.commit()

    def get_many(self, keys):
        out = {}
        with self.lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = "SELECT k, v FROM emb WHERE k IN (%s)" % ",".join("?" * len(chunk))
                for k, v in self.con.execute(q, chunk):
                    out[k] = np.frombuffer(v, dtype=np.float16).astype(np.float32)
        return out

    def set_many(self, rows):
        with self.lock:
            self.con.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?,?,?)",
                                 [(k, v.astype(np.float16).tobytes(), nt, ni, time.time()) for k, v, nt, ni in rows])
            self.con.commit()


_gen_cache = _KV(CACHE_DIR / "gen.sqlite")
_emb_cache = _EmbKV(CACHE_DIR / "emb.sqlite")

_usage_lock = threading.Lock()
USAGE_LOG = CACHE_DIR / "usage.jsonl"


def _log_usage(rec):
    with _usage_lock:
        with open(USAGE_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- generation
def _retryable(e: Exception) -> bool:
    s = str(e)
    return any(x in s for x in ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE",
                                "INTERNAL", "timed out", "Timeout", "timeout", "Connection", "Server disconnected",
                                "RemoteProtocol", "ReadError", "WriteError")) or type(e).__name__ in (
        "ReadTimeout", "WriteTimeout", "ConnectTimeout", "PoolTimeout", "ReadError", "ConnectError", "RemoteProtocolError")


def _is_25(model: str) -> bool:
    return model.startswith("gemini-2.5")


def generate(contents: Union[str, Sequence[Union[str, Img]]], *, model: str = DEFAULT_MODEL, system: Optional[str] = None,
             temperature: float = 0.0, thinking: Optional[Union[str, int]] = "low", max_output_tokens: Optional[int] = None,
             json_schema: Optional[dict] = None, tag: Optional[str] = None, seed: Optional[int] = None,
             retries: int = 8, use_cache: bool = True, media_resolution: Optional[str] = None,
             cache_prefix: Optional[str] = None) -> dict:
    """Returns {'text', 'usage': {...}, 'finish', 'cached': bool}. Never raises on safety blocks (text='').
    cache_prefix: a static TEXT prefix (e.g. a user's memory index) that is placed, together with the system
    instruction, in an explicit API context cache (created once per (model, system, prefix), reused until it expires);
    the request then sends only `contents`. Text-only prefix; the response is equivalent to sending prefix + contents.
    Gemini 3.x: thinking = level str ('low'), per-image resolution via Img.res.
    Gemini 2.5: thinking = int budget (0 = off) or None; resolution is per REQUEST only (verified: per-part is
    ignored), so Img.res values are lifted into a global media_resolution and must agree."""
    from google.genai import types
    if FRESH_TAGS and tag and any(tag.startswith(t) for t in FRESH_TAGS):
        use_cache = False   # measurement runs: bypass the local response cache (the API's own prefix cache still applies)
    if isinstance(contents, str):
        contents = [contents]
    if _is_25(model):
        if isinstance(thinking, str):
            thinking = 0
        rs = [c.res for c in contents if isinstance(c, Img) and c.res]
        if rs:  # one resolution per request: use the majority (history images; a question image may differ)
            media_resolution = media_resolution or max(set(rs), key=rs.count)
            contents = [Img(c.path) if isinstance(c, Img) else c for c in contents]
    ckey_obj = {"model": model, "system": system, "temperature": temperature, "thinking": thinking,
                "max_out": max_output_tokens, "schema": json_schema, "seed": seed,
                "contents": ([cache_prefix] if cache_prefix else []) + [c if isinstance(c, str) else c.key() for c in contents]}
    if media_resolution:
        ckey_obj["media_resolution"] = media_resolution
    ck = _hkey(ckey_obj)
    if use_cache:
        hit = _gen_cache.get(ck)
        if hit is not None:
            hit["cached"] = True
            return hit
    n_img = sum(isinstance(c, Img) for c in contents)
    use_files = n_img > 0  # images always go through the Files API (uploaded once, referenced by URI)
    cfg = dict(temperature=temperature)
    if cache_prefix:
        cfg["cached_content"] = explicit_cache(model, system, cache_prefix)
    elif system:
        cfg["system_instruction"] = system
    if _is_25(model):
        if thinking is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=int(thinking))
        if media_resolution:
            cfg["media_resolution"] = _RES_LEVEL[media_resolution]
    elif isinstance(thinking, int) and not isinstance(thinking, bool):   # Gemini 3.x with an explicit token budget
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking)
    elif thinking:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
    if max_output_tokens:
        cfg["max_output_tokens"] = max_output_tokens
    if json_schema:
        cfg["response_mime_type"] = "application/json"
        cfg["response_json_schema"] = json_schema
    if seed is not None:
        cfg["seed"] = seed
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = client().models.generate_content(model=model, contents=_to_parts(contents, use_files),
                                                 config=types.GenerateContentConfig(**cfg))
            dt = time.time() - t0
            u = r.usage_metadata
            usage = {"prompt": u.prompt_token_count or 0, "cached": u.cached_content_token_count or 0,
                     "out": u.candidates_token_count or 0, "thoughts": u.thoughts_token_count or 0,
                     "img_tokens": sum(d.token_count or 0 for d in (u.prompt_tokens_details or []) if str(d.modality).endswith("IMAGE"))}
            try:
                text = r.text or ""
            except Exception:
                text = ""
            finish = str(r.candidates[0].finish_reason) if r.candidates else "NO_CANDIDATE"
            out = {"text": text, "usage": usage, "finish": finish, "latency_s": round(dt, 2), "n_img": n_img,
                   "model_version": getattr(r, "model_version", None)}
            _log_usage({"model": model, "tag": tag, **usage, "t": time.time()})
            if use_cache:
                _gen_cache.set(ck, out, tag)
            out["cached"] = False
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            s = str(e)
            if cache_prefix and ("CachedContent" in s or "cached content" in s.lower()) and attempt < retries - 1:
                with _xcache_lock:   # expired explicit cache -> recreate
                    _xcache.pop(_hkey({"m": model, "s": system, "p": cache_prefix}), None)
                cfg["cached_content"] = explicit_cache(model, system, cache_prefix)
                continue
            if use_files and ("PERMISSION_DENIED" in s or "not exist" in s or "NOT_FOUND" in s) and attempt < retries - 1:
                with _files_lock:  # stale/expired Files API URIs -> force re-upload
                    for c in contents:
                        if isinstance(c, Img):
                            _files_mem.pop(c.sha, None)
                continue
            if not _retryable(e) or attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt) + random.random() * 2)
    raise last  # pragma: no cover


# --------------------------------------------------------------------------- explicit context caching
_xcache: Dict[str, tuple] = {}
_xcache_lock = threading.Lock()
XCACHE_TTL_S = 3600
XCACHE_STORAGE_PER_MTOK_HOUR = {"gemini-3.8-flash": 0.50, "gemini-2.5-flash-lite": 1.00, "gemini-2.5-flash": 1.00}


def explicit_cache(model: str, system: Optional[str], prefix: str) -> str:
    """Create (once) an explicit context cache holding the system instruction and a static text prefix. The creation
    is logged as a usage record (tag 'xcache:create') with the prefix tokens and the storage cost for the TTL."""
    from google.genai import types
    k = _hkey({"m": model, "s": system, "p": prefix})
    with _xcache_lock:
        hit = _xcache.get(k)
        if hit and hit[1] > time.time() + 60:
            return hit[0]
        cfg = {"contents": [types.Content(role="user", parts=[types.Part.from_text(text=prefix)])],
               "ttl": f"{XCACHE_TTL_S}s"}
        if system:
            cfg["system_instruction"] = system
        c = client().caches.create(model=model, config=types.CreateCachedContentConfig(**cfg))
        ntok = getattr(getattr(c, "usage_metadata", None), "total_token_count", None) or 0
        storage = ntok / 1e6 * XCACHE_STORAGE_PER_MTOK_HOUR.get(model, 1.0) * XCACHE_TTL_S / 3600
        _log_usage({"model": model, "tag": "xcache:create", "prompt": ntok, "cached": 0, "out": 0, "thoughts": 0,
                    "img_tokens": 0, "storage_usd": storage, "t": time.time()})
        _xcache[k] = (c.name, time.time() + XCACHE_TTL_S)
        return c.name


# --------------------------------------------------------------------------- embeddings
DOC_FMT = "title: none | text: {}"
QA_QUERY_FMT = "task: question answering | query: {}"

EmbItem = Union[str, Img, Sequence[Union[str, Img]]]  # a list = interleaved parts -> ONE aggregated embedding


def _emb_item_key(model, dim, item) -> str:
    if isinstance(item, (str, Img)):
        item = [item]
    return _hkey({"m": model, "d": dim, "parts": [p if isinstance(p, str) else p.sha for p in item]})


def _embed_call(model: str, dim: Optional[int], items: List[List[Union[str, Img]]]) -> List[np.ndarray]:
    from google.genai import types
    contents = []
    for it in items:
        parts = []
        for p in it:
            if isinstance(p, str):
                parts.append(types.Part.from_text(text=p))
            else:
                data, mime = _img_bytes_for_api(p)
                parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        contents.append(types.Content(parts=parts))
    cfg = types.EmbedContentConfig(output_dimensionality=dim) if dim else None
    for attempt in range(8):
        try:
            r = client().models.embed_content(model=model, contents=contents, config=cfg)
            vs = [np.asarray(e.values, dtype=np.float32) for e in r.embeddings]
            assert len(vs) == len(items), (len(vs), len(items))
            return vs
        except Exception as e:  # noqa: BLE001
            if not _retryable(e) or attempt == 7:
                raise
            time.sleep(min(60, 2 ** attempt) + random.random() * 2)


def embed(items: Sequence[EmbItem], *, model: str = EMB_MODEL, dim: Optional[int] = None, workers: int = 8,
          batch_text: int = 50, progress: bool = False) -> np.ndarray:
    """Embeds each item separately (an item may be a list of parts -> one aggregated embedding). L2-normalized."""
    norm = [[it] if isinstance(it, (str, Img)) else list(it) for it in items]
    keys = [_emb_item_key(model, dim, it) for it in norm]
    have = _emb_cache.get_many(list(set(keys)))
    todo_idx = {}
    for i, k in enumerate(keys):
        if k not in have and k not in todo_idx:
            todo_idx[k] = i
    batches, cur, cur_img, cur_mb = [], [], 0, 0.0
    for k, i in todo_idx.items():
        it = norm[i]
        ni = sum(isinstance(p, Img) for p in it)
        mb = sum(Path(p.path).stat().st_size for p in it if isinstance(p, Img)) / 1e6
        if cur and (len(cur) >= batch_text or cur_img + ni > 6 or cur_mb + mb > 12):
            batches.append(cur)
            cur, cur_img, cur_mb = [], 0, 0.0
        cur.append((k, it))
        cur_img += ni
        cur_mb += mb
    if cur:
        batches.append(cur)

    def run(batch):
        vs = _embed_call(model, dim, [it for _, it in batch])
        rows = []
        for (k, it), v in zip(batch, vs):
            nt = sum(len(p) for p in it if isinstance(p, str))
            ni = sum(isinstance(p, Img) for p in it)
            rows.append((k, v, nt, ni))
        _emb_cache.set_many(rows)
        _log_usage({"model": model, "tag": "embed", "chars": sum(r[2] for r in rows), "images": sum(r[3] for r in rows),
                    "n": len(rows), "t": time.time()})
        return {k: v for k, v, _, _ in rows}

    if batches:
        for d in pmap(run, batches, workers=workers, desc=f"embed {len(todo_idx)}" if progress else None):
            have.update(d)
    M = np.stack([have[k] for k in keys]).astype(np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M


# --------------------------------------------------------------------------- utils
def pmap(fn, xs: Iterable, workers: int = 16, desc: Optional[str] = None) -> list:
    xs = list(xs)
    out = [None] * len(xs)
    if not xs:
        return out
    bar = None
    if desc:
        from tqdm import tqdm
        bar = tqdm(total=len(xs), desc=desc, mininterval=5, file=sys.stderr)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(xs)}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    return out


def spend(since: float = 0.0, tag_prefix: Optional[str] = None) -> dict:
    """USD estimate from usage.jsonl (only real API calls are logged; cache hits are free)."""
    tot = {}
    if not USAGE_LOG.exists():
        return tot
    for line in open(USAGE_LOG):
        r = json.loads(line)
        if r["t"] < since or (tag_prefix and not str(r.get("tag") or "").startswith(tag_prefix)):
            continue
        m = r["model"]
        p = PRICES.get(m, {})
        if r.get("tag") == "embed":
            usd = r["chars"] / 4 / 1e6 * p.get("text", 0) + r["images"] * 258 / 1e6 * p.get("image", 0)
        else:
            usd = ((r["prompt"] - r["cached"]) * p.get("in", 0) + r["cached"] * p.get("cached", 0)
                   + (r["out"] + r["thoughts"]) * p.get("out", 0)) / 1e6
        tot[m] = tot.get(m, 0) + usd
    tot["TOTAL"] = sum(tot.values())
    return {k: round(v, 4) for k, v in tot.items()}


if __name__ == "__main__":
    r = generate("Say OK.", tag="selftest")
    print(r["text"], r["usage"], r["cached"])
    v = embed(["hello", "world"], dim=768)
    print(v.shape, float(v[0] @ v[1]))
    print(spend())


# --------------------------------------------------------------------------- OpenRouter (cross-family models, text only)
OR_PRICES = {  # USD per 1M tokens, from https://openrouter.ai/api/v1/models (2026-10-27)
    "qwen/qwen3-235b-a22b-2507": {"in": 0.09, "out": 0.55},
    "qwen/qwen3-vl-235b-a22b-instruct": {"in": 0.21, "out": 1.90},
    "openai/gpt-5-mini": {"in": 0.25, "out": 2.00},
    "openai/gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "openai/gpt-4.1": {"in": 2.00, "out": 8.00},
    "qwen/qwen3-235b-a22b-thinking-2507": {"in": 0.45, "out": 3.50},
    "deepseek/deepseek-chat-v3.1": {"in": 0.25, "out": 0.95},
    "openai/gpt-5": {"in": 1.25, "out": 10.00},
}


def generate_or(contents: Union[str, Sequence[str]], *, model: str, system: Optional[str] = None,
                temperature: float = 0.0, max_output_tokens: int = 2048, json_mode: bool = False,
                tag: Optional[str] = None, retries: int = 8, use_cache: bool = True, seed: Optional[int] = None) -> dict:
    """Text-only chat completion through OpenRouter, cached in the same store as generate()."""
    import requests
    from keys import openrouter_key
    text = contents if isinstance(contents, str) else "".join(c for c in contents if isinstance(c, str))
    ck = _hkey({"provider": "openrouter", "model": model, "system": system, "temperature": temperature,
                "max_out": max_output_tokens, "json": json_mode, "seed": seed, "contents": [text]})
    if use_cache:
        hit = _gen_cache.get(ck)
        if hit is not None:
            hit["cached"] = True
            return hit
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    body = {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_output_tokens,
            "usage": {"include": True}}
    if seed is not None:
        body["seed"] = seed
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=HTTP_TIMEOUT_S,
                              headers={"Authorization": f"Bearer {openrouter_key()}"})
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code} {r.text[:300]}")
            d = r.json()
            if "choices" not in d:
                raise RuntimeError(f"no choices: {str(d)[:300]}")
            u = d.get("usage", {}) or {}
            usage = {"prompt": u.get("prompt_tokens", 0) or 0, "cached": 0, "out": u.get("completion_tokens", 0) or 0,
                     "thoughts": 0, "img_tokens": 0, "cost_usd": u.get("cost")}
            text_out = (d["choices"][0]["message"].get("content") or "")
            out = {"text": text_out, "usage": usage, "finish": d["choices"][0].get("finish_reason"),
                   "latency_s": round(time.time() - t0, 2), "n_img": 0, "model_version": d.get("model")}
            _log_usage({"model": model, "tag": tag, **usage, "t": time.time()})
            if use_cache:
                _gen_cache.set(ck, out, tag)
            out["cached"] = False
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(60, 2 ** attempt) + random.random() * 2)
    raise RuntimeError(f"OpenRouter call failed after {retries} attempts: {last}")


def or_spend() -> dict:
    tot = {}
    if USAGE_LOG.exists():
        for line in open(USAGE_LOG):
            r = json.loads(line)
            m = r.get("model", "")
            if m in OR_PRICES:
                c = r.get("cost_usd")
                if c is None:
                    c = (r["prompt"] * OR_PRICES[m]["in"] + r["out"] * OR_PRICES[m]["out"]) / 1e6
                tot[m] = tot.get(m, 0) + c
    tot["TOTAL"] = sum(tot.values())
    return {k: round(v, 4) for k, v in tot.items()}
