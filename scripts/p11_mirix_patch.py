"""Patches for running MIRIX 0.1.5 (pip) with current Gemini models; imported by p11_mirix.py before mirix is used.
1. text-embedding-004 is no longer served -> gemini-embedding-001 (3072-d; MIRIX pads vectors to its maximum width).
2. gemini-3.8-flash is added to MIRIX's list of Gemini models.
3. Every generateContent request is cached by content, sent with low thinking for gemini-3.x, and its usage logged."""
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = {"v": "bl:mirix"}
_lock = threading.Lock()


def apply():
    import requests                     # MIRIX posts without a timeout; a dropped connection then hangs forever
    if not getattr(requests.post, "_p11", False):
        _post = requests.post

        def post(*a, **k):
            k.setdefault("timeout", 300)
            return _post(*a, **k)
        post._p11 = True
        requests.post = post
    from google import genai             # embeddings and image uploads go through google-genai, which has no default
    from google.genai import types       # timeout either
    if not getattr(genai.Client.__init__, "_p11", False):
        _init = genai.Client.__init__

        def init(self, *a, **k):
            if k.get("http_options") is None:
                k["http_options"] = types.HttpOptions(timeout=300_000)
            _init(self, *a, **k)
        init._p11 = True
        genai.Client.__init__ = init
    from mirix.schemas import embedding_config as ec
    from mirix.agent import app_constants as ac
    from mirix.llm_api import google_ai_client as gc
    orig_default = ec.EmbeddingConfig.default_config.__func__

    def default_config(cls, model_name=None, provider=None):
        if model_name in (None, "text-embedding-004") and provider in (None, "google_ai"):
            return cls(embedding_model="gemini-embedding-001", embedding_endpoint_type="google_ai",
                       embedding_endpoint="https://generativelanguage.googleapis.com", embedding_dim=3072,
                       embedding_chunk_size=2048)
        return orig_default(cls, model_name, provider)
    ec.EmbeddingConfig.default_config = classmethod(default_config)
    for m in ("gemini-3.8-flash",):
        if m not in ac.GEMINI_MODELS:
            ac.GEMINI_MODELS.append(m)
    db = sqlite3.connect(str(ROOT / "cache/mirix_requests.sqlite"), check_same_thread=False, timeout=60)
    db.execute("CREATE TABLE IF NOT EXISTS r (k TEXT PRIMARY KEY, v TEXT)")
    db.commit()
    orig_request = gc.GoogleAIClient.request

    def request(self, request_data):
        model = self.llm_config.model
        if model.startswith("gemini-3"):
            gen = request_data.setdefault("generation_config", request_data.get("generationConfig", {}))
            gen.setdefault("thinkingConfig", {"thinkingLevel": "low"})
        key = hashlib.sha256((model + json.dumps(request_data, sort_keys=True, default=str)).encode()).hexdigest()
        with _lock:
            row = db.execute("SELECT v FROM r WHERE k=?", (key,)).fetchone()
        replay = row is not None
        if replay:
            resp = json.loads(row[0])
        else:
            for attempt in range(8):                      # the network drops TLS connections under load
                try:
                    resp = orig_request(self, request_data)
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 7:
                        raise
                    time.sleep(min(60, 2 ** attempt * 2))
        if not replay:
            with _lock:
                db.execute("INSERT OR REPLACE INTO r VALUES (?, ?)", (key, json.dumps(resp)))
                db.commit()
        u = resp.get("usageMetadata", {}) if isinstance(resp, dict) else {}
        rec = {"model": model, "tag": TAG["v"], "prompt": u.get("promptTokenCount", 0), "cached": u.get("cachedContentTokenCount", 0),
               "out": u.get("candidatesTokenCount", 0), "thoughts": u.get("thoughtsTokenCount", 0), "img_tokens": 0,
               "t": time.time(), "oai": True, "pid": os.getpid()}
        with _lock:
            with open(ROOT / "cache/usage_bl.jsonl", "a") as f:
                f.write(json.dumps({**rec, "replay": replay}) + "\n")
            if not replay:
                with open(ROOT / "cache/usage.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
        return resp
    gc.GoogleAIClient.request = request
