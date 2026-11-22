"""Phase-2 multimodal memory methods (all built on the raw-round store of src/mm.py).

Selectors (which rounds to load):
  rag:K            cosine top-K over a round key (baseline)
  gx:k0:j:K        top-k0 hits + caption-space nearest neighbours of the hits, up to K rounds (set expansion, no LLM)
  toc:N            LLM reads a compact table of contents (one line per round) and picks <= N rounds
  tocu:N:k         union of toc:N and rag:k
Readers (how loaded rounds are read):
  read:<res>       one call over the loaded rounds (raw images at <res>)
  read+toc:<res>   same, plus the full table of contents as global context
  zoom[+toc]:<res> two passes: pass 1 may request up to 3 zooms (image id + optional box); pass 2 sees the crops
  mr[+toc]:<res>   map-reduce: one question-aware extraction call per loaded image at <res>, then a text-only answer
Every call's usage is returned so costs are totals over all stages.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm  # noqa: E402
from data_mm import QA, Episode, Round  # noqa: E402
from gem import CACHE_DIR, Img, generate, pmap  # noqa: E402

def llm(prompt_parts, model, system, thinking, schema=None, tag=None, max_output_tokens=None, temperature=0.0, seed=None,
        cache_prefix=None):
    """Gemini (generate, JSON schema) or a cross-family OpenRouter model (generate_or, JSON mode) for text prompts.
    cache_prefix (Gemini only): static text prefix served from an explicit context cache."""
    if "/" in model:
        from gem import generate_or
        txt = "".join(p for p in prompt_parts if isinstance(p, str))
        sys_ = (system or "") + ("\nReturn only valid JSON." if schema else "")
        # reasoning models spend output tokens on thinking: with 2,048 tokens gpt-5-mini returned no JSON for 22% of the
        # MemEye selections (phase 17), so they get a larger limit
        _mo = max_output_tokens or (8192 if ("gpt-5" in model or "thinking" in model) else 2048)
        return generate_or(txt, model=model, system=sys_, temperature=temperature, json_mode=bool(schema), tag=tag,
                           max_output_tokens=_mo, seed=seed)
    kw = {} if (temperature == 0.0 and seed is None) else {"temperature": temperature, "seed": seed}
    if cache_prefix:
        kw["cache_prefix"] = cache_prefix
    return generate(prompt_parts, model=model, system=system, thinking=thinking, json_schema=schema, tag=tag,
                    max_output_tokens=max_output_tokens or MAX_OUT, **kw)


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return {}
    return {}


CROPS = CACHE_DIR / "crops"
CROPS.mkdir(exist_ok=True)
MAX_OUT = 2048  # cap every phase-2 generation (a zoom call once looped to 65k output tokens)
SEED = None  # decoding seed for the ANSWER calls only (selection calls stay identical -> cached)


def _trunc(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


# ----------------------------------------------------------------------------- table of contents
def labels(ep: Episode) -> Dict[str, str]:
    """image label (as rendered in prompts) -> image path"""
    out = {}
    for r in ep.rounds:
        for j, p in enumerate(r.images):
            out[mm._image_label(r, j)] = p
    for lab, p in ep.persona_images:
        out[lab] = p
    return out


TOC_MODE = "full"  # ablations: "text" (no image captions), "cap" (captions only, no dialogue text)
XCACHE = False     # Phase 6: serve the static index prefix from an explicit API context cache (persistent memory)
# Phase 7: index compression (all default off = the frozen index)
NO_ASSIST = False  # drop the assistant snippet
SHORT_IDS = False  # rounds are listed as r<k> (position in the history) under one header per session
TELEGRAPH = False  # strip function words from the user snippet before truncation (keeps names, numbers, labels)
MASK_P = 0.0       # Phase 8 (visibility manipulation): mask each question content word in EVERY index line with prob. p
_TG_STOP = set("""a an the and or but so of to in on at for with by from as is are was were be been being am do does did
have has had i me my mine we our ours you your yours he him his she her it its they them their this that these those there
here what which who whom when where why how can could would should will shall may might must just really very quite also
too then than about into over under again further once only own same such both each few more most other some any all
not no nor if because while until up down out off s t don doesn didn isn wasn aren weren hasn haven hadn won wouldn
couldn shouldn let im ive id ill youre thats theres its""".split())
SNIPPET = 200      # characters of the user message kept in an index line (frozen on MemEye; "QSS-long" uses 600)
COORDS = False     # "narrative coordinates": session / day / round ordinals made explicit in every index line
_coord_cache: Dict[str, Dict[str, str]] = {}


def coords(ep: Episode) -> Dict[str, str]:
    """rid -> 'session 9 of 12 · day 9 of 12 · round 6 of 8 (last session)' (deterministic, no LLM)."""
    if ep.name not in _coord_cache:
        sids = list(dict.fromkeys(r.sid for r in ep.rounds))
        dates = sorted({r.date for r in ep.rounds})
        per = collections.Counter(r.sid for r in ep.rounds)
        seen, out = collections.Counter(), {}
        for r in ep.rounds:
            seen[r.sid] += 1
            si = sids.index(r.sid) + 1
            tag = " (first session)" if si == 1 else (" (last session)" if si == len(sids) else "")
            out[r.rid] = (f"session {si} of {len(sids)}{tag} · day {dates.index(r.date) + 1} of {len(dates)} · "
                          f"round {seen[r.sid]} of {per[r.sid]}")
        _coord_cache[ep.name] = out
    return _coord_cache[ep.name]


SNIPPET_MODE = "head"   # "head": first SNIPPET chars; "query": head + the window that best matches the question
_TL = threading.local()
_STOP = set("a an the and or but of to in on at for with by from as is are was were be been being do does did have has had "
            "i me my we our you your he she it its they them their this that these those what which who whom when where why "
            "how all any each how much many more most some such no not only than too very can will just should now about "
            "answer question date exact only e g".split())


def _words(t: str):
    return [w for w in re.findall(r"[a-z0-9$]+", t.lower()) if w not in _STOP and len(w) > 1]


def query_snippet(text: str, question: str, head: int = 100, win: int = 200) -> str:
    """First `head` chars + the `win`-char window with the largest overlap with the question's content words
    (lexical, no API call). Lets an index line expose evidence that sits deep inside a long message."""
    if len(text) <= head + win:
        return text
    q = set(_words(question))
    best, best_i = -1, head
    for i in range(head, max(head + 1, len(text) - win + 1), 40):
        sc = len(q & set(_words(text[i:i + win])))
        if sc > best:
            best, best_i = sc, i
    return text[:head] + " … " + text[best_i:best_i + win] + (" …" if best_i + win < len(text) else "")


def mask_question_words(text: str, question: Optional[str], p: float, salt: str) -> str:
    """Visibility manipulation: replace each occurrence of a content word of the question by '[...]' with probability p
    (deterministic per (question, round, word)). Applied to every round alike, so the gold set does not leak."""
    if not question or p <= 0:
        return text
    qw = set(_words(question.split("\n", 1)[-1]))
    if not qw:
        return text
    import hashlib

    def rep(m):
        w = m.group(0)
        if w.lower() not in qw:
            return w
        h = int(hashlib.md5(f"{question}|{salt}|{w.lower()}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "[...]" if h < p else w
    return re.sub(r"[A-Za-z0-9$]+", rep, text)


def telegraph(text: str) -> str:
    """Rule-based compression: drop function words and filler punctuation, keep content words, names, numbers."""
    toks = re.findall(r"[$€£]?\d[\d.,:/%]*|[A-Za-z][A-Za-z0-9'’\-]*|[^\sA-Za-z0-9]", text)
    out = [t for t in toks if not (t.lower().replace("'", "").replace("’", "") in _TG_STOP or t in ",;:!?\"()[]—–-…")]
    return " ".join(out)


_SESS_ORD: Dict[str, Dict[str, int]] = {}


def session_label(ep: Episode, sid: str) -> str:
    """Short session label for compact indexes: the benchmark's own label when it is short and informative (e.g.
    'S3', 'D9' from 'BRAND_S3' / 'HEALTH_D9'), otherwise the session's ordinal position ('session 7')."""
    tail = sid.split("_")[-1]
    if len(tail) <= 6 and re.search(r"\d", tail) and not re.fullmatch(r"[0-9a-f]{6,}", tail):
        return tail
    if ep.name not in _SESS_ORD:
        order = {}
        for r in ep.rounds:
            order.setdefault(r.sid, len(order) + 1)
        _SESS_ORD[ep.name] = order
    return f"session {_SESS_ORD[ep.name][sid]}"


def short_id(ep: Episode, r: Round) -> str:
    return f"r{ep.rid2idx[r.rid]}"


def toc_line(r: Round, short_caps: Dict[str, str], ep: Optional[Episode] = None) -> str:
    if SHORT_IDS and ep is not None:
        s = f"{short_id(ep, r)}"
    else:
        s = (f"[{r.rid} | {coords(ep)[r.rid].split(' · ')[-1]}]" if (COORDS and ep is not None) else f"[{r.rid} | {r.date}]")
    if TOC_MODE != "cap":
        qq = getattr(_TL, "question", None)
        if SNIPPET_MODE == "gist" and _gist_key(r) in GISTS:
            s += f" U: {_trunc(r.user, 100)} | GIST: {GISTS[_gist_key(r)]}"
        else:
            ut = telegraph(r.user) if TELEGRAPH else r.user
            s += f" U: {query_snippet(ut, qq) if (SNIPPET_MODE == 'query' and qq) else _trunc(ut, SNIPPET)}"
    for j, p in enumerate(r.images):
        lab = f"img{j + 1}" if SHORT_IDS else mm._image_label(r, j)
        s += f" | IMG {lab}" + (f": {_trunc(short_caps[p], 220)}" if TOC_MODE != "text" else "")
    if r.assistant and TOC_MODE != "cap" and not NO_ASSIST:
        s += f" | A: {_trunc(r.assistant, 100)}"
    if MASK_P > 0:
        s = mask_question_words(s, getattr(_TL, "mask_q", None), MASK_P, r.rid)
    if USE_ALIASES and ep is not None and ALIASES.get(ep.name, {}).get(r.rid):
        s += " | aka: " + "; ".join(ALIASES[ep.name][r.rid])
    return s


def session_header(ep: Episode, sid: str) -> str:
    """Narrative frame of a session: ordinal position, day, date and how the user opened it (deterministic)."""
    rs = [r for r in ep.rounds if r.sid == sid]
    c = coords(ep)[rs[0].rid].split(" · round")[0]
    return f"== {sid} | {c} | {rs[0].date} | opens with: \"{_trunc(rs[0].user, 160)}\" =="


def render_lines(ep: Episode, idxs: Sequence[int], short_caps: Dict[str, str]) -> str:
    """Index lines for the given rounds (chronological); with COORDS, grouped under session headers."""
    out, last = [], None
    for i in sorted(idxs):
        r = ep.rounds[i]
        if SHORT_IDS and not COORDS and r.sid != last:
            out.append(f"== {session_label(ep, r.sid)} | {r.date} ==")
            last = r.sid
        if COORDS and r.sid != last:
            out.append(session_header(ep, r.sid))
            last = r.sid
        out.append(toc_line(r, short_caps, ep))
    return "\n".join(out)


def toc(ep: Episode, short_caps: Dict[str, str]) -> str:
    return render_lines(ep, range(len(ep.rounds)), short_caps)


# Phase 16: image-anchored index (IAC). One header per session (handle s<k>, the session's own label, date, size, key
# words of its text-only rounds) and one line per image round (handle r<k>, the user's words, each image's description).
INDEX_MODE = "lines"   # "lines" = one line per round (the frozen index) | "iac"
IAC_KW = 12            # key words per session header: the rarest within the history, shown in reading order
IAC_B = 20             # user words on an image round's line (telegraphed, first occurrences)
_IAC: Dict[tuple, tuple] = {}


def _wkey(w: str) -> str:
    return re.sub(r"[^\w$€£%.-]", "", w.lower()).strip(".-")


def iac_index(ep: Episode, short_caps: Dict[str, str], ids: str = "handles") -> tuple:
    """-> (index text, {session handle: [round idx]}); deterministic, no model calls.
    ids="handles": r<k> / s<k> (selector); ids="rids": the benchmark's round and session ids (reader context)."""
    key = (ep.name, id(short_caps), IAC_KW, IAC_B, TOC_MODE, ids)
    if key in _IAC:
        return _IAC[key]
    df = collections.Counter()
    for r in ep.rounds:
        df.update({_wkey(w) for w in telegraph(r.user).split() if _wkey(w)})
    idf = {k: math.log((len(ep.rounds) + 1) / (c + 0.5)) for k, c in df.items()}
    sess: Dict[str, List[int]] = {}
    for i, r in enumerate(ep.rounds):
        sess.setdefault(r.sid, []).append(i)
    lines, handles = [], {}
    for si, (sid, idxs) in enumerate(sess.items(), 1):
        handles[f"s{si}"] = idxs
        cand: Dict[str, str] = {}
        for i in idxs:
            if not ep.rounds[i].images:
                for w in telegraph(ep.rounds[i].user).split():
                    k = _wkey(w)
                    if k and k not in cand:
                        cand[k] = w
        top = set(sorted(cand, key=lambda k: -idf.get(k, 0.0))[:IAC_KW])
        words = " ".join(w for k, w in cand.items() if k in top)
        sh = f"s{si} [{session_label(ep, sid)}]" if ids == "handles" else sid
        lines.append(f"== {sh} {ep.rounds[idxs[0]].date} · {len(idxs)} rounds" + (f" · {words}" if words else "") + " ==")
        for i in idxs:
            r = ep.rounds[i]
            if not r.images:
                continue
            ws, seen = [], set()
            for w in telegraph(r.user).split():
                k = _wkey(w)
                if k and k not in seen:
                    seen.add(k)
                    ws.append(w)
                if len(ws) >= IAC_B:
                    break
            line = (f"r{i}" if ids == "handles" else f"[{r.rid}]") + f" U: {' '.join(ws)}"
            for j, p in enumerate(r.images):
                lab = f"img{j + 1}" if ids == "handles" else mm._image_label(r, j)
                line += f" | IMG {lab}" + (f": {_trunc(short_caps[p], 220)}" if TOC_MODE != "text" else "")
            lines.append(line)
    _IAC[key] = ("\n".join(lines), handles)
    return _IAC[key]


TOC_SELECT_SYS = (
    "You are the retrieval step of a personal assistant's long-term memory. You get a compact index of the user's "
    "entire conversation history (one line per round; images are summarised in a few words) and a question. Choose "
    "the rounds that must be opened in full (with their original images) to answer the question.\n"
    "- Include EVERY round that may hold needed evidence, especially ALL members of any group the question refers to "
    "(e.g. all ads of a brand, every scene of an episode, all versions of a screen over time), not just the best "
    "match.\n- Include rounds that define labels the question uses (episode/arc names, 'first', 'later', "
    "'updated').\n- Image summaries are lossy: when unsure whether an image is relevant, include it.\n"
    "- Order by importance. Return JSON {\"rounds\": [round ids]} with at most {N} ids.")
_SEL_SCHEMA = {"type": "object", "properties": {"rounds": {"type": "array", "items": {"type": "string"}}},
               "required": ["rounds"]}
IAC_SELECT_SYS = (
    "You are the retrieval step of a personal assistant's long-term memory. You get a compact index of the user's "
    "entire conversation history and a question. The index has one header per session (handle s<k>, the session's "
    "own label, its date and number of rounds, and key words from its rounds without images) and, under it, one line "
    "per round that contains images (handle r<k>, the user's words and a short description of each image). Choose "
    "what must be opened in full (with the original text and images) to answer the question: give r<k> to open an "
    "image round, and give s<k> to open every round of a session only when needed evidence may be in its rounds "
    "without images.\n"
    "- Include EVERY round or session that may hold needed evidence, especially ALL members of any group the question "
    "refers to (e.g. all ads of a brand, every scene of an episode, all versions of a screen over time), not just the "
    "best match.\n- Include rounds or sessions that define labels the question uses (episode/arc names, 'first', "
    "'later', 'updated').\n- Image descriptions are lossy: when unsure whether an image is relevant, include it.\n"
    "- Order by importance. Return JSON {\"rounds\": [ids]} with at most {N} ids.")
TOC_SELECT_CF_SYS = TOC_SELECT_SYS.replace(
    "- Order by importance. Return JSON {\"rounds\": [round ids]} with at most {N} ids.",
    "- First write the groups the question refers to (a short description and how many members you expect), then "
    "list the round ids of every member of every group, then any other needed rounds. Return JSON {\"groups\": "
    "[{\"description\": str, \"expected\": int}], \"rounds\": [round ids]} with at most {N} ids.")
_SEL_CF_SCHEMA = {"type": "object", "properties": {
    "groups": {"type": "array", "items": {"type": "object", "properties": {"description": {"type": "string"},
                                                                          "expected": {"type": "integer"}}}},
    "rounds": {"type": "array", "items": {"type": "string"}}}, "required": ["groups", "rounds"],
    "propertyOrdering": ["groups", "rounds"]}


def toc_select(ep: Episode, qa: QA, short_caps, N: int, reader: str, thinking,
               subset: Optional[Sequence[int]] = None, temperature: float = 0.0, seed=None,
               system: Optional[str] = None, schema: Optional[dict] = None) -> tuple[List[int], dict]:
    """reader = the SELECTOR model here (may differ from the answering model).
    subset: only show these rounds (e.g. a cosine prefilter), in chronological order."""
    _TL.question = qa.question if SNIPPET_MODE == "query" else None
    _TL.mask_q = qa.question if MASK_P > 0 else None
    iac = INDEX_MODE == "iac" and subset is None
    if iac:
        body, handles = iac_index(ep, short_caps)
        head = f"=== MEMORY INDEX ({len(ep.rounds)} rounds in {len(handles)} sessions) ==="
        system = system or IAC_SELECT_SYS
    elif subset is None:
        body, head = (toc(ep, short_caps) if SNIPPET_MODE in ("head", "gist") else render_lines(ep, range(len(ep.rounds)), short_caps)), \
            f"=== MEMORY INDEX ({len(ep.rounds)} rounds) ==="
    else:
        body = render_lines(ep, subset, short_caps)
        head = (f"=== MEMORY INDEX: {len(subset)} candidate rounds (pre-filtered by similarity from {len(ep.rounds)} "
                f"rounds; chronological) ===")
    demo = DEMOS.get(qa.task, "") if USE_DEMOS else ""
    if XCACHE and subset is None and "/" not in reader:   # static per-user prefix from an explicit context cache
        pre = f"{demo}{head}\n{body}\n=== END INDEX ===\n"
        prompt = [f"{usage_block(qa)}\nQuestion: {qa.question}"]
    else:
        pre = None
        prompt = [f"{demo}{head}\n{body}\n=== END INDEX ===\n{usage_block(qa)}\nQuestion: {qa.question}"]
    if qa.question_image:
        prompt += ["\n[Image attached to the question]", Img(qa.question_image, "medium")]
    r = llm(prompt, reader, (system or TOC_SELECT_SYS).replace("{N}", str(N)), thinking, schema=schema or _SEL_SCHEMA,
            tag="p2:toc_select" + (":xcache" if pre else ""), temperature=temperature, seed=seed, cache_prefix=pre)
    r2i = ep.rid2idx
    rids = [str(x) for x in (parse_json(r["text"]).get("rounds") or []) if isinstance(x, (str, int))]
    if iac:
        rids = rids or re.findall(r"\b[rs]\d+\b", r["text"] or "")
        seen, ids = set(), []
        for x in rids:
            m = re.fullmatch(r"([rs])(\d+)", x.strip().strip("[]").lower())
            if not m:
                continue
            new = handles.get(x.strip().lower(), []) if m.group(1) == "s" else (
                [int(m.group(2))] if int(m.group(2)) < len(ep.rounds) else [])
            for j in new:
                if j not in seen:
                    seen.add(j)
                    ids.append(j)
        return ids[:N], r["usage"]
    if not rids:
        rids = re.findall(r"r\d+" if SHORT_IDS else r"[A-Za-z0-9_\-]+:[A-Za-z0-9_\-]+", r["text"] or "")
    seen, ids = set(), []
    for x in rids:
        x = x.strip().strip("[]")
        if SHORT_IDS:
            m = re.fullmatch(r"r?(\d+)", x)
            j = int(m.group(1)) if m else None
            if j is not None and 0 <= j < len(ep.rounds) and j not in seen:
                seen.add(j)
                ids.append(j)
            continue
        if x in r2i and r2i[x] not in seen:
            seen.add(r2i[x])
            ids.append(r2i[x])
    return ids[:N], r["usage"]


SCAN_SYS = (
    "You are filtering a user's conversation history for a question. You see a CHUNK of the history (one line per "
    "round; images are summarised in a few words). List the round ids in this chunk that may contain evidence needed "
    "to answer the question, or that define labels the question uses (names of episodes, series, versions, 'first', "
    "'later'). Be inclusive: if a round plausibly belongs to a group the question refers to, include it. Return JSON "
    "{\"rounds\": [ids]} (empty list if none).")


def scan_select(ep: Episode, qa: QA, short_caps, N: int, chunk: int, model: str, thinking,
                order_scores: Optional[np.ndarray] = None) -> tuple[List[int], list]:
    """Exhaustive chunked relevance scan: one small call per `chunk` rounds; union capped at N
    (if more than N are marked, keep the N with the highest cosine score)."""
    lines = [toc_line(r, short_caps, ep) for r in ep.rounds]
    r2i = ep.rid2idx

    def one(start):
        body = "\n".join(lines[start:start + chunk])
        p = [f"Question: {qa.question}\n\n=== CHUNK: rounds {start + 1}-{min(start + chunk, len(lines))} of "
             f"{len(lines)} ===\n{body}\n=== END CHUNK ==="]
        g = generate(p, model=model, system=SCAN_SYS, thinking=thinking, json_schema=_SEL_SCHEMA,
                     tag="p2:scan", max_output_tokens=MAX_OUT)
        try:
            ids = [r2i[x] for x in json.loads(g["text"])["rounds"] if isinstance(x, str) and x in r2i]
        except Exception:  # noqa: BLE001
            ids = []
        return [i for i in ids if start <= i < start + chunk], g["usage"]

    outs = pmap(one, list(range(0, len(lines), chunk)), workers=8)
    sel = sorted({i for ids, _ in outs for i in ids})
    if len(sel) > N and order_scores is not None:
        sel = sorted(sel, key=lambda i: -order_scores[i])[:N]
    return sel[:N], [u for _, u in outs]


# ----------------------------------------------------------------------------- ranking control (RankGPT-style)
RANK_SYS = (
    "You are the retrieval step of a personal assistant's long-term memory. You get candidate rounds from the user's "
    "conversation history (one line per round; images are summarised in a few words) and a question. Rank ALL candidate "
    "rounds from most to least useful for answering the question. Return JSON {\"ranking\": [all round ids, best first]}.")
_RANK_SCHEMA = {"type": "object", "properties": {"ranking": {"type": "array", "items": {"type": "string"}}},
                "required": ["ranking"]}
# Phase 8 (review): ranking control with the SAME recall-oriented instructions as TOC_SELECT_SYS; only the output format
# differs (a complete ranking instead of an unordered set of model-chosen size).
RANK_RECALL_SYS = (
    "You are the retrieval step of a personal assistant's long-term memory. You get candidate rounds from the user's "
    "conversation history (one line per round; images are summarised in a few words) and a question. Rank ALL candidate "
    "rounds by how likely they must be opened in full (with their original images) to answer the question.\n"
    "- Rank first EVERY round that may hold needed evidence, especially ALL members of any group the question refers to "
    "(e.g. all ads of a brand, every scene of an episode, all versions of a screen over time), not just the best "
    "match.\n- Rank high the rounds that define labels the question uses (episode/arc names, 'first', 'later', "
    "'updated').\n- Image summaries are lossy: when unsure whether an image is relevant, rank it high.\n"
    "Return JSON {\"ranking\": [all round ids, most needed first]}.")


def rank_select(ep: Episode, qa: QA, short_caps, subset: Sequence[int], model: str, thinking,
                recall: bool = False) -> tuple[List[int], dict]:
    """Same model, same candidates and index lines as ptoc selection, but a RANKING instruction (no set semantics).
    recall=True uses the recall-oriented instructions of the set selector (RANK_RECALL_SYS)."""
    body = render_lines(ep, subset, short_caps)
    prompt = [f"=== CANDIDATE ROUNDS ({len(subset)}, chronological) ===\n{body}\n=== END ===\n\nQuestion: {qa.question}"]
    r = generate(prompt, model=model, system=RANK_RECALL_SYS if recall else RANK_SYS, thinking=thinking,
                 json_schema=_RANK_SCHEMA, tag="p8:rank_recall" if recall else "p3:rank", max_output_tokens=4096)
    r2i = ep.rid2idx
    try:
        raw = [str(x).strip() for x in json.loads(r["text"])["ranking"] if isinstance(x, (str, int))]
    except Exception:  # noqa: BLE001
        raw = []
    if SHORT_IDS:   # compact index: rounds are listed as r<k> (position in the history)
        ids = [int(m.group(1)) for m in (re.fullmatch(r"r?(\d+)", x) for x in raw) if m and int(m.group(1)) < len(ep.rounds)]
    else:
        ids = [r2i[x] for x in raw if x in r2i]
    ids = list(dict.fromkeys(i for i in ids if i in set(subset)))
    rest = [i for i in subset if i not in ids]   # unranked candidates keep their cosine order at the end
    return ids + rest, r["usage"]


# ----------------------------------------------------------------------------- dynamic-cutoff reranking baseline
GRADE_SYS = (
    "You are the retrieval step of a personal assistant's long-term memory. You get candidate rounds from the user's "
    "conversation history (one line per round; images are summarised in a few words) and a question. Rate EVERY "
    "candidate round for how useful it is for answering the question: 3 = contains needed evidence, 2 = probably "
    "relevant, 1 = possibly relevant, 0 = not relevant. Return JSON {\"scores\": {round_id: 0-3, ...}} covering all "
    "candidates.")
_GRADE_SCHEMA = {"type": "object", "properties": {"scores": {"type": "object", "additionalProperties": {"type": "integer"}}},
                 "required": ["scores"]}


def grade_select(ep: Episode, qa: QA, short_caps, subset: Sequence[int], model: str, thinking,
                 threshold: int = 2) -> tuple[List[int], dict, Dict[int, int]]:
    """Model reranking with a DYNAMIC cut-off: graded relevance (0-3) for every candidate in one listwise call; keep
    candidates graded >= threshold. Same candidates and index lines as QSS, but a relevance (not set) instruction."""
    body = render_lines(ep, subset, short_caps)
    prompt = [f"=== CANDIDATE ROUNDS ({len(subset)}, chronological) ===\n{body}\n=== END ===\n\nQuestion: {qa.question}"]
    r = llm(prompt, model, GRADE_SYS, thinking, schema=_GRADE_SCHEMA if "/" not in model else {"json": 1},
            tag="p4:grade", max_output_tokens=8192)
    r2i = ep.rid2idx
    sc = {}
    for k, v in (parse_json(r["text"]).get("scores") or {}).items():
        if k in r2i and r2i[k] in set(subset):
            try:
                sc[r2i[k]] = int(v)
            except Exception:  # noqa: BLE001
                pass
    keep = sorted([i for i, v in sc.items() if v >= threshold], key=lambda i: (-sc[i], i))
    return keep, r["usage"], sc


# ----------------------------------------------------------------------------- MemLens official judge
_ML_JUDGE = None


def memlens_judge(qa: QA, pred: str, model: str = "qwen/qwen3-vl-235b-a22b-instruct") -> dict:
    """Official MemLens judge prompt (judge_prompts/, binary 0/1); default judge model = Qwen3-VL-235B (official family)."""
    global _ML_JUDGE
    if _ML_JUDGE is None:
        import importlib
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/MemLens"))
        _ML_JUDGE = importlib.import_module("judge_prompts")
    qtype, sub = qa.qtype.split("|")
    question = qa.question.split("\n", 1)[1] if qa.question.startswith("Question date:") else qa.question
    task = _ML_JUDGE.get_task_key(qtype, sub, qa.answer)
    pred = pred if len(pred) <= 2000 else pred[:2000] + " ...[truncated]"
    prompt = _ML_JUDGE.build_judge_prompt(task, question, qa.answer, pred)
    if isinstance(prompt, (list, tuple)):
        prompt = "\n\n".join(str(x) for x in prompt)
    r = llm([prompt], model, None, "low", tag="p4:mljudge", max_output_tokens=1024)
    m = re.findall(r'"answer_score"\s*:\s*([01])', r["text"] or "")
    return {"score": float(m[-1]) if m else 0.0, "reason": (r["text"] or "")[-300:], "judge": model}


# ----------------------------------------------------------------------------- cheap-model set selection
def text_scores(ix: mm.RoundIndex, text: str, key: str) -> np.ndarray:
    from gem import embed
    return ix.M[key] @ embed([mm.QA_QUERY_FMT.format(text)])[0]


PRED_SYS = (
    "You help a memory system find evidence. A question asks about a user's past conversations, in which the user "
    "shared many images. List the GROUPS of past items that must be found to answer it. Write each group as a short "
    "membership description that a single conversation round either satisfies or not (e.g. 'a Pepsi advertisement "
    "image', 'a frame from Episode 2 of the cartoon', 'a screenshot of the glucose dashboard taken on the busy day', "
    "'the round where the first main-bench photo was shared'). Add a group for every reference the question needs "
    "resolved. If the question states or implies how many items a group has, give that number, else null. "
    "Return JSON {\"groups\": [{\"description\": str, \"count\": int or null}]} with 1-4 groups.")
_PRED_SCHEMA = {"type": "object", "properties": {"groups": {"type": "array", "items": {"type": "object", "properties": {
    "description": {"type": "string"}, "count": {"type": "integer"}}, "required": ["description"]}}},
    "required": ["groups"]}
MEMBER_SYS = (
    "You check group membership for a memory system. You get a question, a group description, and candidate rounds "
    "from the user's history (one line per round; images summarised in a few words). Return EVERY candidate round "
    "that belongs to the group; judge each line on its own, using both the user's words and the image summary. "
    "Return JSON {\"rounds\": [round ids]} (empty list if none).")


def psc_select(ep: Episode, qa: QA, short_caps, ix: mm.RoundIndex, key: str, C: int, N: int, model: str,
               thinking) -> tuple[List[int], list, dict]:
    """Predicate-guided Set Completion (cheap model):
    1) parse the question into membership predicates (groups) with optional counts,
    2) per predicate, build a candidate pool = cosine top-C for the predicate text ∪ top-C/2 for the question,
    3) verify membership listwise inside each pool, 4) union the members (capped at N, cosine order)."""
    usages = []
    q_img = ["\n[Image attached to the question]", Img(qa.question_image, "medium")] if qa.question_image else []
    g = generate([f"Question: {qa.question}"] + q_img, model=model, system=PRED_SYS, thinking=thinking,
                 json_schema=_PRED_SCHEMA, tag="p3:psc_pred", max_output_tokens=MAX_OUT)
    usages.append(g["usage"])
    try:
        groups = [x for x in json.loads(g["text"])["groups"] if isinstance(x, dict) and x.get("description")][:4]
    except Exception:  # noqa: BLE001
        groups = []
    if not groups:
        groups = [{"description": qa.question, "count": None}]
    sq = ix.scores(qa, key)
    q_top = [int(i) for i in np.argsort(-sq)[: max(1, C // 2)]]
    r2i = ep.rid2idx

    def one(gr):
        sp = text_scores(ix, gr["description"], key)
        pool = list(dict.fromkeys([int(i) for i in np.argsort(-sp)[:C]] + q_top))
        lines = render_lines(ep, pool, short_caps)
        cnt = f" (the question implies about {gr['count']} such items)" if gr.get("count") else ""
        p = [f"Question: {qa.question}\nGroup: {gr['description']}{cnt}\n\n=== CANDIDATE ROUNDS (chronological) ===\n"
             f"{lines}\n=== END ==="]
        r = generate(p, model=model, system=MEMBER_SYS, thinking=thinking, json_schema=_SEL_SCHEMA,
                     tag="p3:psc_member", max_output_tokens=MAX_OUT)
        try:
            ids = [r2i[x] for x in json.loads(r["text"])["rounds"] if isinstance(x, str) and x in r2i]
        except Exception:  # noqa: BLE001
            ids = []
        return [i for i in ids if i in set(pool)], r["usage"]

    outs = pmap(one, groups, workers=4)
    sel = []
    for ids, u in outs:
        usages.append(u)
        for i in ids:
            if i not in sel:
                sel.append(i)
    if len(sel) > N:
        sel = sorted(sel, key=lambda i: -sq[i])[:N]
    return sel, usages, {"groups": groups}


def toc_select_sampled(ep: Episode, qa: QA, short_caps, N: int, model: str, thinking, K: int, min_votes: int = 1,
                       temperature: float = 1.0, subset=None) -> tuple[List[int], list]:
    """Self-consistency over K temperature samples of toc_select; keep rounds chosen by >= min_votes samples."""
    votes, usages = collections.Counter(), []

    def one(k):
        return toc_select(ep, qa, short_caps, N, model, thinking, subset=subset, temperature=temperature, seed=k)

    for ids, u in pmap(one, list(range(K)), workers=K):
        usages.append(u)
        votes.update(set(ids))
    sel = [i for i, v in votes.most_common() if v >= min_votes]
    return sel[:N], usages


# ----------------------------------------------------------------------------- in-context distillation (teacher -> cheap selector)
DEMOS: Dict[str, str] = {}  # task name -> demonstration block built from OTHER tasks (leave-one-task-out)
USE_DEMOS = False


def build_demos(eps: Sequence[Episode], short_caps, teacher: Dict[str, dict], n_demo: int = 3, seed: int = 0,
                max_sessions: int = 3) -> Dict[str, str]:
    """For every task, pick n_demo teacher selections from n_demo DIFFERENT other tasks (complete teacher sets with >=2
    gold rounds, preferring questions the cheap selector got wrong) and render each as: index excerpt (whole sessions
    that contain the selected rounds, with session headers) + question + the teacher's selection."""
    import random
    global COORDS
    by_task = collections.defaultdict(list)
    ep_of = {}
    for e in eps:
        for qa in e.qas:
            t = teacher.get(qa.qid)
            if t and t.get("full") == 1.0 and t.get("n_gold", 0) >= 2:
                by_task[qa.task].append((t.get("weak_full", 1.0), qa, e))
            ep_of[qa.task] = e
    rng = random.Random(seed)
    out = {}
    for task in ep_of:
        others = [t for t in by_task if t != task]
        rng.shuffle(others)
        blocks = []
        for t in others[:n_demo]:
            cands = sorted(by_task[t], key=lambda x: (x[0], x[1].qid))  # weak-selector failures first
            _, qa, e = cands[0]
            sel = [e.rid2idx[x] for x in teacher[qa.qid]["ids"] if x in e.rid2idx]
            sids = list(dict.fromkeys(e.rounds[i].sid for i in sorted(sel)))[:max_sessions]
            idxs = [i for i, r in enumerate(e.rounds) if r.sid in sids]
            old, COORDS = COORDS, True
            body = render_lines(e, idxs, short_caps)
            COORDS = old
            chosen = [e.rounds[i].rid for i in sel if e.rounds[i].sid in sids]
            blocks.append(f"--- Example (another user's history, excerpt) ---\n{body}\nQuestion: {qa.question}\n"
                          f"Rounds to open: {json.dumps(chosen)}")
        out[task] = ("Examples of good selections (note how every member of the referenced group is included and how "
                     "labels such as day numbers, 'first/final' or the situation of a session are resolved):\n\n"
                     + "\n\n".join(blocks) + "\n\n=== NOW THE ACTUAL TASK ===\n")
    return out


# ----------------------------------------------------------------------------- usage write-back (resolved references)
USAGE: Dict[str, List[dict]] = {}  # qid -> visible past resolutions [{"question", "rids"}] (built by the runner)
USE_USAGE = False


def usage_block(qa: QA) -> str:
    past = USAGE.get(qa.qid, []) if USE_USAGE else []
    if not past:
        return ""
    lines = [f"- \"{_trunc(p['question'], 220)}\" -> {json.dumps(p['rids'])}" for p in past]
    return ("\n=== RESOLVED EARLIER QUESTIONS from this user (the rounds that were opened to answer them; the labels they "
            "use may refer to the same items as the current question) ===\n" + "\n".join(lines) + "\n=== END ===\n")


def build_usage(eps: Sequence[Episode], resolutions: Dict[str, List[str]], mode: str = "loo", seed: int = 0,
                max_sim: float = 0.9, max_items: int = 40, frac: float = 1.0) -> Dict[str, List[dict]]:
    """Past resolutions visible to each question. mode 'loo': all other questions of the same history;
    'online': only questions that come earlier in a random order. Paraphrase guard: past questions whose embedding
    cosine to the current question exceeds max_sim are dropped (they would leak the evidence set)."""
    import random
    from gem import embed
    out = {}
    for e in eps:
        qs = [qa for qa in e.qas if qa.qid in resolutions]
        if not qs:
            continue
        V = embed([mm.QA_QUERY_FMT.format(qa.question) for qa in qs])
        S = V @ V.T
        order = list(range(len(qs)))
        random.Random(seed).shuffle(order)
        pos = {qs[i].qid: p for p, i in enumerate(order)}
        written = {qs[i].qid for i in order if random.Random(f"{seed}:{qs[i].qid}").random() < frac}
        for a, qa in enumerate(qs):
            cand = []
            for b, qb in enumerate(qs):
                if b == a or S[a, b] > max_sim:
                    continue
                if mode == "online" and pos[qb.qid] > pos[qa.qid]:
                    continue
                if qb.qid not in written:
                    continue
                cand.append((S[a, b], qb))
            cand.sort(key=lambda x: -x[0])
            out[qa.qid] = [{"question": qb.question, "rids": resolutions[qb.qid]} for _, qb in cand[:max_items]]
    return out


# ----------------------------------------------------------------------------- write-time alias enrichment
ALIAS_SYS = (
    "You maintain the index of a personal assistant's long-term memory. You see the user's whole conversation history "
    "(sessions with their position, day and opening line; one line per round; images summarised briefly). For EVERY "
    "round that shares an image, write the short phrases the user might later use to refer to that round or its image "
    "(e.g. 'the D9 summary card', 'the baseline trend screen', 'the busy-day dashboard', 'the second Pepsi ad', "
    "'checkpoint 3 of the night route', 'the first living-room photo'): use day/session numbers, order words "
    "(first/later/final), the situation of the session, and what the image shows. 2-5 phrases per round. "
    "Return JSON {\"aliases\": [{\"round\": id, \"phrases\": [str]}]}.")
_ALIAS_SCHEMA = {"type": "object", "properties": {"aliases": {"type": "array", "items": {"type": "object", "properties": {
    "round": {"type": "string"}, "phrases": {"type": "array", "items": {"type": "string"}}}, "required": ["round", "phrases"]}}},
    "required": ["aliases"]}
ALIASES: Dict[str, Dict[str, List[str]]] = {}  # episode -> rid -> phrases (filled by build_aliases)
USE_ALIASES = False


def build_aliases(ep: Episode, short_caps, model: str = "gemini-3.8-flash", chunk: int = 60) -> tuple[Dict[str, List[str]], list]:
    """One write-time pass (per chunk of image rounds, whole-history index as context). Amortised over all questions."""
    global COORDS
    old, COORDS = COORDS, True
    full = toc(ep, short_caps)
    COORDS = old
    img_rounds = [r.rid for r in ep.rounds if r.images]
    out, usages = {}, []

    def one(start):
        ids = img_rounds[start:start + chunk]
        g = generate([f"=== HISTORY ({len(ep.rounds)} rounds) ===\n{full}\n=== END ===\n\nWrite aliases for these rounds: "
                      + ", ".join(ids)], model=model, system=ALIAS_SYS, thinking="low", json_schema=_ALIAS_SCHEMA,
                     tag="p3:aliases", max_output_tokens=16384)
        try:
            return {x["round"]: [str(p)[:80] for p in x.get("phrases", [])][:5] for x in json.loads(g["text"])["aliases"]
                    if isinstance(x, dict) and x.get("round") in ids}, g["usage"]
        except Exception:  # noqa: BLE001
            return {}, g["usage"]

    for d, u in pmap(one, list(range(0, len(img_rounds), chunk)), workers=4):
        out.update(d)
        usages.append(u)
    return out, usages


# ----------------------------------------------------------------------------- write-time group index
GROUP_SYS = (
    "You organise a user's conversation history into a navigation index. You get the whole history as one line per "
    "round (images summarised in a few words). Write groups that a future question might refer to as a set: e.g. all "
    "items of one brand/series/person/place/object, all scenes of an episode or arc, all versions of the same screen "
    "or document over time, all photos of the same room or route, all rounds of one game. Each group has a short "
    "descriptive name and the list of member round ids; a round may belong to several groups; groups may be large. "
    "Cover every round that shows or discusses a distinct item. Return JSON {\"groups\": [{\"name\": str, "
    "\"rounds\": [ids]}]}.")
_GROUP_SCHEMA = {"type": "object", "properties": {"groups": {"type": "array", "items": {"type": "object", "properties": {
    "name": {"type": "string"}, "rounds": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "rounds"]}}},
    "required": ["groups"]}


def build_groups(ep: Episode, short_caps, model: str = "gemini-3.8-flash") -> tuple[list, dict]:
    """One write-time call per history (amortised over all future questions)."""
    r = generate([f"=== HISTORY INDEX ({len(ep.rounds)} rounds) ===\n{toc(ep, short_caps)}\n=== END ==="],
                 model=model, system=GROUP_SYS, thinking="low", json_schema=_GROUP_SCHEMA, tag="p2:groups",
                 max_output_tokens=32768)
    r2i = ep.rid2idx
    try:
        gs = json.loads(r["text"])["groups"]
    except Exception:  # noqa: BLE001
        gs = []
    out = []
    for g in gs:
        ids = sorted({r2i[x] for x in g.get("rounds", []) if x in r2i})
        if ids:
            out.append({"name": str(g.get("name", ""))[:120], "rounds": ids})
    return out, r["usage"]


GSEL_SYS = (
    "You are the retrieval step of a personal assistant's memory. The history has been organised into named groups "
    "(with their sizes and date ranges). Choose the groups whose rounds must be opened to answer the question; pick "
    "every group the question refers to (and groups that define labels it uses). Return JSON {\"groups\": [group "
    "numbers]}.")
_GSEL_SCHEMA = {"type": "object", "properties": {"groups": {"type": "array", "items": {"type": "integer"}}},
                "required": ["groups"]}


def group_select(ep: Episode, qa: QA, groups: list, N: int, model: str, thinking,
                 fallback: Optional[List[int]] = None) -> tuple[List[int], dict]:
    lines = []
    for gi, g in enumerate(groups):
        d0, d1 = ep.rounds[g["rounds"][0]].date, ep.rounds[g["rounds"][-1]].date
        lines.append(f"{gi}. {g['name']} ({len(g['rounds'])} rounds, {d0}..{d1})")
    p = [f"=== GROUPS ===\n" + "\n".join(lines) + f"\n=== END ===\n\nQuestion: {qa.question}"]
    r = generate(p, model=model, system=GSEL_SYS, thinking=thinking, json_schema=_GSEL_SCHEMA, tag="p2:gsel",
                 max_output_tokens=MAX_OUT)
    try:
        gids = [int(x) for x in json.loads(r["text"])["groups"] if 0 <= int(x) < len(groups)]
    except Exception:  # noqa: BLE001
        gids = []
    ids = []
    for gi in gids:
        for i in groups[gi]["rounds"]:
            if i not in ids:
                ids.append(i)
    for i in fallback or []:  # cosine top hits are always added (cheap safety net)
        if i not in ids:
            ids.append(i)
    return ids[:N], r["usage"]


# ----------------------------------------------------------------------------- graph set expansion (no LLM)
def graph_expand(ix: mm.RoundIndex, qa: QA, key: str, k0: int, j: int, K: int) -> List[int]:
    s = ix.scores(qa, key)
    order = [int(i) for i in np.argsort(-s)]
    sel = order[:k0]
    M = ix.M[key]
    for h in list(sel):
        if len(sel) >= K:
            break
        nb = np.argsort(-(M @ M[h]))
        added = 0
        for n in nb:
            n = int(n)
            if n not in sel:
                sel.append(n)
                added += 1
            if added >= j or len(sel) >= K:
                break
    for i in order:  # fill remaining budget with the next cosine ranks
        if len(sel) >= K:
            break
        if i not in sel:
            sel.append(i)
    return sel[:K]


# ----------------------------------------------------------------------------- reading
def _context_parts(ep: Episode, idxs: Sequence[int], res: str, caps, with_toc: bool, rep: str = "raw") -> list:
    parts = []
    if with_toc and INDEX_MODE == "iac":
        txt, hs = iac_index(ep, caps["short"], ids="rids")
        parts.append(f"=== INDEX OF THE WHOLE HISTORY ({len(ep.rounds)} rounds in {len(hs)} sessions; session headers "
                     f"list key words of rounds without images; image summaries are lossy, the opened rounds below show "
                     f"the original images) ===\n{txt}\n=== END INDEX ===\n")
    elif with_toc:
        parts.append(f"=== INDEX OF THE WHOLE HISTORY ({len(ep.rounds)} rounds; image summaries are lossy, the "
                     f"opened rounds below show the original images) ===\n{toc(ep, caps['short'])}\n=== END INDEX ===\n")
    rr = {"raw": f"raw:{res}", "both": f"both:dense:{res}", "cap": "cap:dense", "none": "none"}[rep]
    return parts + mm.build_prompt(ep, _NoQ(), idxs, rr, caps)  # tail (dummy question) cut by _strip_q


class _NoQ:
    question = "\x00"
    question_image = None


def _question_parts(qa: QA) -> list:
    out = ["\nQuestion: " + qa.question]
    if qa.question_image:
        out += ["\n[Image attached to the question]", Img(qa.question_image, "high")]
    for lab, path in (getattr(qa, "option_images", None) or []):
        out += [f"\n[Option {lab}]", Img(path, "low")]
    return out


def _strip_q(parts: list) -> list:
    """build_prompt(... _NoQ) ends with '...Question: \\x00\\nAnswer:'; cut that tail off."""
    out = list(parts)
    if out and isinstance(out[-1], str) and "\x00" in out[-1]:
        out[-1] = out[-1].split("\n=== END OF HISTORY ===")[0] + "\n=== END OF HISTORY ===\n"
    return out


def read(ep, qa, idxs, res, caps, reader, thinking, with_toc=False, tag="p2:read", rep="raw"):
    ctx = _strip_q(_context_parts(ep, idxs, res, caps, with_toc, rep))
    pre = None
    if XCACHE and with_toc and ctx and isinstance(ctx[0], str):   # the index block is static per user -> explicit cache
        pre, ctx = ctx[0], ctx[1:]
    parts = mm._merge(ctx + _question_parts(qa) + ["\nAnswer:"])
    r = generate(parts, model=reader, system=mm.system_prompt(ep.bench), thinking=thinking, tag=tag + (":xcache" if pre else ""),
                 max_output_tokens=MAX_OUT, seed=SEED, cache_prefix=pre)
    return r["text"].strip(), [r["usage"]]


def hint_read(ep, qa, hint_idxs, res, caps, reader, thinking, tag="p2:hint"):
    """Full history (all rounds) + the selector's picks as a hint placed right before the question."""
    all_idx = list(range(len(ep.rounds)))
    rids = ", ".join(ep.rounds[i].rid for i in sorted(hint_idxs))
    hint = (f"\n[Memory-retrieval hint: rounds pre-selected as relevant to this question (may be incomplete): {rids}. "
            f"Check all of them carefully, and use the rest of the history as needed.]")
    parts = mm._merge(_strip_q(_context_parts(ep, all_idx, res, caps, False)) + [hint] + _question_parts(qa)
                      + ["\nAnswer:"])
    r = generate(parts, model=reader, system=mm.system_prompt(ep.bench), thinking=thinking, tag=tag,
                 max_output_tokens=MAX_OUT)
    return r["text"].strip(), [r["usage"]]


ZOOM_INSTR = (
    "\n\nBefore answering, decide whether you need a closer look. The images above are shown at REDUCED resolution, so "
    "small text, small symbols, exact colours, counts and fine positions may be unreliable. You may request up to 3 "
    "zoomed views. Each zoom names an image ID exactly as shown in brackets and a box [ymin, xmin, ymax, xmax] "
    "(integers 0-1000, relative to that image) around the region that holds the needed detail (use [0,0,1000,1000] "
    "for the whole image). Whenever the answer depends on visual detail in specific images, zoom into them. Also give "
    "your best provisional answer.\nReturn JSON {\"zooms\": [{\"image\": id, \"box\": [ymin,xmin,ymax,xmax]}], "
    "\"answer\": \"...\"}.")
ZOOM_INSTR_FORCED = ZOOM_INSTR.replace("You may request up to 3", "You MUST request between 1 and 3").replace(
    "Whenever the answer depends on visual detail in specific images, zoom into them.",
    "Zoom into the image(s) and region(s) most likely to contain the answer.")
_ZOOM_SCHEMA = {"type": "object", "properties": {
    "zooms": {"type": "array", "items": {"type": "object", "properties": {
        "image": {"type": "string"}, "box": {"type": "array", "items": {"type": "integer"}}}, "required": ["image"]}},
    "answer": {"type": "string"}}, "required": ["zooms", "answer"]}


def crop(path: str, box: Optional[list], min_side: int = 768) -> str:
    """Crop (0-1000 normalised [ymin,xmin,ymax,xmax], 6% padding) and upscale so the long side >= min_side."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    if box and len(box) == 4:
        y0, x0, y1, x1 = [max(0, min(1000, int(v))) for v in box]
        if y1 <= y0 or x1 <= x0:
            y0, x0, y1, x1 = 0, 0, 1000, 1000
    else:
        y0, x0, y1, x1 = 0, 0, 1000, 1000
    py, px = 0.06 * (y1 - y0), 0.06 * (x1 - x0)
    X0, Y0 = max(0, int((x0 - px) / 1000 * W)), max(0, int((y0 - py) / 1000 * H))
    X1, Y1 = min(W, int((x1 + px) / 1000 * W) + 1), min(H, int((y1 + py) / 1000 * H) + 1)
    c = im.crop((X0, Y0, max(X1, X0 + 8), max(Y1, Y0 + 8)))
    s = min_side / max(c.size)
    if s > 1:
        c = c.resize((round(c.size[0] * s), round(c.size[1] * s)), Image.LANCZOS)
    key = hashlib.sha256(f"{path}|{X0},{Y0},{X1},{Y1}|{min_side}".encode()).hexdigest()[:24]
    dst = CROPS / f"{key}.png"
    if not dst.exists():
        c.save(dst, format="PNG")
    return str(dst)


def zoom_read(ep, qa, idxs, res, caps, reader, thinking, with_toc=False, zoom_res="high", tag="p2:zoom",
              forced=False):
    ctx = _strip_q(_context_parts(ep, idxs, res, caps, with_toc)) + _question_parts(qa)
    r1 = generate(mm._merge(ctx + [ZOOM_INSTR_FORCED if forced else ZOOM_INSTR]), model=reader, system=mm.system_prompt(ep.bench),
                  thinking=thinking, json_schema=_ZOOM_SCHEMA, tag=tag + ":p1", max_output_tokens=MAX_OUT)
    usages = [r1["usage"]]
    try:
        o = json.loads(r1["text"])
    except Exception:  # noqa: BLE001
        return (r1["text"] or "").strip(), usages, []
    lab2p = labels(ep)
    shown = {mm._image_label(ep.rounds[i], j) for i in idxs for j in range(len(ep.rounds[i].images))}
    shown |= {lab for lab, _ in ep.persona_images}
    zooms = [z for z in (o.get("zooms") or []) if isinstance(z, dict) and z.get("image") in lab2p
             and z.get("image") in shown][:3]
    if not zooms:
        return str(o.get("answer", "")).strip(), usages, []
    zparts = ["\n\n=== ZOOMED VIEWS YOU REQUESTED ==="]
    for z in zooms:
        p = crop(lab2p[z["image"]], z.get("box"))
        zparts += [f"\n[Zoom of image {z['image']}, box {z.get('box')}]", Img(p, zoom_res)]
    # gemini-2.5: one resolution per request -> zooms go in at the context resolution (crops still add pixels)
    if reader.startswith("gemini-2.5"):
        zparts = [Img(x.path, res) if isinstance(x, Img) else x for x in zparts]
    r2 = generate(mm._merge(ctx + zparts + ["\n\nNow answer the question.\nAnswer:"]), model=reader,
                  system=mm.system_prompt(ep.bench), thinking=thinking, tag=tag + ":p2", max_output_tokens=MAX_OUT)
    usages.append(r2["usage"])
    return r2["text"].strip(), usages, zooms


ZOOM_REQ = (
    "\n\nDo NOT answer yet. The images above are shown at REDUCED resolution, so small text, small symbols, exact "
    "colours, counts and fine positions may be unreliable. Decide whether a closer look would help answer the "
    "question. You may request up to 3 zoomed views; each names an image ID exactly as shown in brackets and a box "
    "[ymin, xmin, ymax, xmax] (integers 0-1000, relative to that image) around the region holding the needed detail "
    "([0,0,1000,1000] = whole image). Return JSON {\"zooms\": [{\"image\": id, \"box\": [ymin,xmin,ymax,xmax]}]} — an "
    "empty list if no zoom is needed.")
ZOOM_REQ_FORCED = ZOOM_REQ.replace("Decide whether a closer look would help answer the question. You may request up "
                                   "to 3", "Request between 1 and 3").replace(" — an empty list if no zoom is needed", "")
_ZREQ_SCHEMA = {"type": "object", "properties": {"zooms": {"type": "array", "items": {"type": "object", "properties": {
    "image": {"type": "string"}, "box": {"type": "array", "items": {"type": "integer"}}}, "required": ["image"]}}},
    "required": ["zooms"]}


def _zoom_answer(ep, qa, idxs, res, caps, reader, thinking, with_toc, zooms, zoom_res, tag):
    """pass 2: plain answer prompt + the requested crops (identical to read() when zooms is empty)."""
    if not zooms:
        return read(ep, qa, idxs, res, caps, reader, thinking, with_toc, tag=tag + ":plain")
    lab2p = labels(ep)
    zparts = ["\n\n=== ZOOMED VIEWS (higher detail of regions of the images above) ==="]
    for z in zooms:
        p = crop(lab2p[z["image"]], z.get("box"))
        zparts += [f"\n[Zoom of image {z['image']}, box {z.get('box')}]",
                   Img(p, res if reader.startswith("gemini-2.5") else zoom_res)]
    parts = mm._merge(_strip_q(_context_parts(ep, idxs, res, caps, with_toc)) + _question_parts(qa) + zparts
                      + ["\nAnswer:"])
    r = generate(parts, model=reader, system=mm.system_prompt(ep.bench), thinking=thinking, tag=tag + ":p2",
                 max_output_tokens=MAX_OUT)
    return r["text"].strip(), [r["usage"]]


def zoom2_read(ep, qa, idxs, res, caps, reader, thinking, with_toc=False, forced=False, zoom_res="high",
               tag="p2:zoom2"):
    """Reader-driven zoom, strictly additive: pass 1 returns only zoom requests; pass 2 is the baseline prompt plus
    the crops (or exactly the baseline call when nothing is requested)."""
    ctx = mm._merge(_strip_q(_context_parts(ep, idxs, res, caps, with_toc)) + _question_parts(qa)
                    + [ZOOM_REQ_FORCED if forced else ZOOM_REQ])
    r1 = generate(ctx, model=reader, system=mm.system_prompt(ep.bench), thinking=thinking, json_schema=_ZREQ_SCHEMA,
                  tag=tag + ":p1", max_output_tokens=MAX_OUT)
    shown = {mm._image_label(ep.rounds[i], j) for i in idxs for j in range(len(ep.rounds[i].images))}
    shown |= {lab for lab, _ in ep.persona_images}
    try:
        zooms = [z for z in json.loads(r1["text"]).get("zooms", []) if isinstance(z, dict) and z.get("image") in shown][:3]
    except Exception:  # noqa: BLE001
        zooms = []
    pred, us = _zoom_answer(ep, qa, idxs, res, caps, reader, thinking, with_toc, zooms, zoom_res, tag)
    return pred, [r1["usage"]] + us, zooms


def _render_inplace(ep, idxs, res, caps, reader, zmap, with_toc):
    """History where each requested image is upgraded IN PLACE: gemini-3.x -> that image at HIGH resolution (+ the
    requested crop right after it); gemini-2.5 (one resolution per request) -> the crop right after it."""
    parts = []
    if with_toc:
        parts.append(f"=== INDEX OF THE WHOLE HISTORY ({len(ep.rounds)} rounds) ===\n{toc(ep, caps['short'])}\n"
                     f"=== END INDEX ===\n")
    parts.append(f"User profile: {json.dumps(ep.profile, ensure_ascii=False)[:1500]}\n")
    if ep.persona_images:
        parts.append("\nPeople who appear in the user's saved chats (reference photos):")
        parts += mm._render_images([lab for lab, _ in ep.persona_images], [p for _, p in ep.persona_images],
                                   "raw", None, res, caps)
        parts.append("\n")
    parts.append(f"\n=== CONVERSATION HISTORY ({len(idxs)} of {len(ep.rounds)} rounds, chronological) ===")
    is25 = reader.startswith("gemini-2.5")
    for i in sorted(idxs):
        r = ep.rounds[i]
        parts.append(f"\n[{r.sid} | {r.date} | round {r.rid}]\nUser: {r.user}")
        for j, p in enumerate(r.images):
            lab = mm._image_label(r, j)
            z = zmap.get(lab)
            parts += [f"\n[Image {lab}]", Img(p, "high" if (z is not None and not is25) else res)]
            if z is not None and z.get("box") and list(z["box"]) != [0, 0, 1000, 1000]:
                parts += [f"\n[Zoomed region of image {lab}, box {z['box']}]",
                          Img(crop(p, z["box"]), res if is25 else "high")]
        parts.append(f"\nAssistant: {r.assistant}\n")
    parts.append("\n=== END OF HISTORY ===\n")
    return parts


def zoom3_read(ep, qa, idxs, res, caps, reader, thinking, with_toc=False, forced=False, tag="p2:zoom3"):
    """Reader-driven zoom with IN-PLACE upgrade (pass 1 identical to zoom2)."""
    ctx = mm._merge(_strip_q(_context_parts(ep, idxs, res, caps, with_toc)) + _question_parts(qa)
                    + [ZOOM_REQ_FORCED if forced else ZOOM_REQ])
    r1 = generate(ctx, model=reader, system=mm.system_prompt(ep.bench), thinking=thinking, json_schema=_ZREQ_SCHEMA,
                  tag=tag + ":p1", max_output_tokens=MAX_OUT)
    shown = {mm._image_label(ep.rounds[i], j) for i in idxs for j in range(len(ep.rounds[i].images))}
    try:
        zooms = [z for z in json.loads(r1["text"]).get("zooms", []) if isinstance(z, dict) and z.get("image") in shown][:3]
    except Exception:  # noqa: BLE001
        zooms = []
    if not zooms:
        pred, us = read(ep, qa, idxs, res, caps, reader, thinking, with_toc, tag=tag + ":plain")
        return pred, [r1["usage"]] + us, []
    zmap = {z["image"]: z for z in zooms}
    parts = mm._merge(_render_inplace(ep, idxs, res, caps, reader, zmap, with_toc) + _question_parts(qa) + ["\nAnswer:"])
    r = generate(parts, model=reader, system=mm.system_prompt(ep.bench), thinking=thinking, tag=tag + ":p2",
                 max_output_tokens=MAX_OUT)
    return r["text"].strip(), [r1["usage"], r["usage"]], zooms


def blind_zoom_read(ep, qa, idxs, res, caps, reader, thinking, ix, key, with_toc=False, zoom_res="high",
                    tag="p2:zoomb"):
    """Control for reader-driven zoom: same extra budget (<=3 crops) but chosen WITHOUT the reader:
    the most query-similar loaded image as a 2x2 grid of tiles -> 3 tiles with the most content (top-left, top-right,
    bottom-left are taken in order; fixed rule)."""
    s = ix.scores(qa, key)
    cand = [i for i in sorted(idxs, key=lambda i: -s[i]) if ep.rounds[i].images]
    if not cand:
        return read(ep, qa, idxs, res, caps, reader, thinking, with_toc, tag=tag + ":plain") + ([],)
    lab = mm._image_label(ep.rounds[cand[0]], 0)
    zooms = [{"image": lab, "box": b} for b in ([0, 0, 520, 520], [0, 480, 520, 1000], [480, 0, 1000, 520])]
    pred, us = _zoom_answer(ep, qa, idxs, res, caps, reader, thinking, with_toc, zooms, zoom_res, tag)
    return pred, us, zooms


MAP_PROMPT = (
    "You are helping answer a question about a user's past conversation. Below is ONE round of that conversation "
    "(with its date and any image the user shared) and the question. Look carefully at the image at full detail and "
    "write a short factual note (<= 60 words) with everything in THIS round that could help answer the question: "
    "exact colours, counts, visible text, positions, identities, states. If nothing here is relevant, reply exactly "
    "IRRELEVANT.\n\nQuestion: {q}\n")


def map_reduce(ep, qa, idxs, res, caps, reader, thinking, with_toc=False, tag="p2:mr"):
    usages = []

    def one(i):
        r = ep.rounds[i]
        if not r.images:
            return i, None, None
        parts = [MAP_PROMPT.format(q=qa.question)] + mm.render_round(r, "raw", None, res, caps)
        if qa.question_image:
            parts += ["\n[Image attached to the question]", Img(qa.question_image, res)]
        g = generate(mm._merge(parts), model=reader, thinking=thinking, tag=tag + ":map", max_output_tokens=MAX_OUT)
        return i, g["text"].strip(), g["usage"]

    notes = {}
    for i, n, u in pmap(one, sorted(idxs), workers=8):
        if u:
            usages.append(u)
        if n and not n.upper().startswith("IRRELEVANT"):
            notes[i] = n
    parts = []
    if with_toc:
        parts.append(f"=== INDEX OF THE WHOLE HISTORY ({len(ep.rounds)} rounds) ===\n{toc(ep, caps['short'])}\n"
                     f"=== END INDEX ===\n")
    parts.append(f"\n=== OPENED ROUNDS ({len(idxs)}) — images were inspected one by one at full detail and replaced by "
                 f"question-specific notes ===")
    for i in sorted(idxs):
        r = ep.rounds[i]
        s = f"\n[{r.sid} | {r.date} | round {r.rid}]\nUser: {r.user}"
        for j in range(len(r.images)):
            s += f"\n[Image {mm._image_label(r, j)}] note: " + (notes.get(i, "nothing relevant to this question"))
        parts.append(s + f"\nAssistant: {r.assistant}\n")
    parts += _question_parts(qa) + ["\nAnswer:"]
    g = generate(mm._merge(parts), model=reader, system=mm.system_prompt(ep.bench), thinking=thinking,
                 tag=tag + ":reduce", max_output_tokens=MAX_OUT)
    usages.append(g["usage"])
    return g["text"].strip(), usages, len(notes)


# ----------------------------------------------------------------------------- Phase 5: EVA (Expand, Verify, Aggregate)
EVA_MAP_SYS = (
    "You check ONE session of a user's past conversation with an assistant, including the images the user shared, for "
    "evidence that helps answer a question about the user's history. Read every message and look carefully at every "
    "image (receipts, screenshots, photos, charts): copy exact numbers, names, labels, and dates.")
EVA_MAP_INSTR = (
    "\nReturn JSON {{\"relevant\": true|false, \"facts\": [\"<round id>: <one concrete fact>\", ...]}}.\n"
    "Set relevant=true if this session contains ANY information that could be part of the answer: one of several items "
    "that the question totals, counts, compares, lists, or orders; an earlier or later state of something the question "
    "asks about; or a date or time needed for a computation. Each fact must be self-contained (what, which item, exact "
    "value, when) and cite its round id. For every image that relates to the question, add a fact stating what the image "
    "shows that matters (objects, visible text, numbers, colours, positions). If nothing is relevant, return "
    "relevant=false and an empty list.")
_EVA_SCHEMA = {"type": "object", "properties": {"relevant": {"type": "boolean"},
                                                "facts": {"type": "array", "items": {"type": "string"}}},
               "required": ["relevant", "facts"]}
EVA_MAP_MODEL = "gemini-2.5-flash-lite"
EVA_MAX_FACTS = 12


def session_units(ep: Episode, idxs: Sequence[int]) -> List[List[int]]:
    """Expand candidate rounds to whole sessions (all rounds of each touched session), in chronological order."""
    want = {ep.rounds[i].sid for i in idxs}
    units: Dict[str, List[int]] = {}
    for j, r in enumerate(ep.rounds):
        if r.sid in want:
            units.setdefault(r.sid, []).append(j)
    return sorted(units.values(), key=lambda u: u[0])


def eva_map(ep: Episode, qa: QA, unit: Sequence[int], res: str, caps, model: str = None, thinking=0,
            tag: str = "p5:eva_map") -> tuple:
    model = model or EVA_MAP_MODEL
    r0 = ep.rounds[unit[0]]
    what = "SESSION" if len(unit) > 1 else "PART"
    parts = [f"=== ONE {what} OF THE USER'S PAST CONVERSATION (session {r0.sid}, date {r0.date}) ==="]
    for j in unit:
        parts += mm.render_round(ep.rounds[j], "raw", None, res, caps)
    parts.append("=== END OF SESSION ===\n")
    parts += _question_parts(qa) + [EVA_MAP_INSTR.format()]
    g = generate(mm._merge(parts), model=model, system=EVA_MAP_SYS, thinking=thinking, json_schema=_EVA_SCHEMA,
                 tag=tag, max_output_tokens=MAX_OUT)
    o = parse_json(g["text"])
    facts = [str(f) for f in (o.get("facts") or []) if str(f).strip()][:EVA_MAX_FACTS]
    return bool(o.get("relevant")) and bool(facts), facts, g["usage"]


EVA_REDUCE_HEAD = (
    "=== NOTES FROM YOUR PAST SESSIONS WITH THE USER ===\n"
    "Each session of the history that may bear on the question was read in full, including its images, and the facts "
    "relevant to the question were written down below (chronological, with session dates and round ids). The same item "
    "can appear in several rounds of one session, for example a mention in text and a receipt or photo of it; count such "
    "an item once. When the question totals, counts, compares, or orders items, go through every note.\n")


EVA_GUIDE_HEAD = (
    "=== NOTES (a reading aid) ===\n"
    "Each retrieved round below was first checked on its own, including its images, and the facts relevant to the question "
    "were written down here (with session dates and round ids). Use the notes to make sure you consider every relevant "
    "item, then verify them against the original rounds that follow. The same item can appear in several rounds of one "
    "session (for example a mention in text and a receipt or photo of it); count such an item once.\n")


def eva_read(ep: Episode, qa: QA, idxs: Sequence[int], res: str, caps, reader: str, thinking,
             tag: str = "p5:eva", map_model: str = None, with_raw: bool = False, fallback_idxs: Sequence[int] = (),
             unit: str = "session", all_raw: bool = False, raw_idxs: Optional[Sequence[int]] = None):
    """EVA: expand candidates to units (whole sessions, or single rounds), verify + extract per unit with a cheap model
    (raw text + images), then aggregate: the reader answers from the extracted notes (with_raw: plus raw units)."""
    units = session_units(ep, idxs) if unit == "session" else [[i] for i in sorted(idxs)]
    outs = pmap(lambda u: (u,) + eva_map(ep, qa, u, res, caps, model=map_model, tag=tag + ":map"), units, workers=4)
    usages = []
    rel = [(u, facts) for u, ok, facts, _ in outs if ok]
    info = {"n_units": len(units), "n_relevant_units": len(rel), "n_facts": sum(len(f) for _, f in rel),
            "relevant_rounds": [ep.rounds[j].rid for u, _ in rel for j in u], "map_usages": [o[3] for o in outs],
            "notes": [[ep.rounds[u[0]].sid, f] for u, f in rel]}
    if all_raw:   # "read with notes": extracted notes as a guide + ALL candidate rounds in raw form
        parts = []
        if rel:
            lines = [EVA_GUIDE_HEAD]
            for u, facts in rel:
                r0 = ep.rounds[u[0]]
                lines.append(f"\n[session {r0.sid} | {r0.date}]")
                lines += [f"- {f}" for f in facts]
            lines.append("\n=== END OF NOTES ===\n")
            parts.append("\n".join(lines))
        parts += _strip_q(_context_parts(ep, sorted(raw_idxs) if raw_idxs is not None else idxs, res, caps, False)) + \
            _question_parts(qa) + ["\nAnswer:"]
        g = generate(mm._merge(parts), model=reader, system=mm.system_prompt(ep.bench), thinking=thinking,
                     tag=tag + ":reduce", max_output_tokens=MAX_OUT, seed=SEED)
        return g["text"].strip(), usages + [g["usage"]], info
    if not rel:   # nothing verified: fall back to reading the top candidates directly
        pred, us = read(ep, qa, list(fallback_idxs) or list(idxs)[:10], res, caps, reader, thinking, tag=tag + ":fallback")
        info["fallback"] = True
        return pred, usages + us, info
    lines = [EVA_REDUCE_HEAD]
    for u, facts in rel:
        r0 = ep.rounds[u[0]]
        lines.append(f"\n[session {r0.sid} | {r0.date}]")
        lines += [f"- {f}" for f in facts]
    lines.append("\n=== END OF NOTES ===\n")
    parts = ["\n".join(lines)]
    if with_raw:
        parts.append("\n=== ORIGINAL ROUNDS OF THESE SESSIONS ===")
        for u, _ in rel:
            for j in u:
                parts += mm.render_round(ep.rounds[j], "raw", None, res, caps)
        parts.append("=== END ===\n")
    parts += _question_parts(qa) + ["\nAnswer:"]
    g = generate(mm._merge(parts), model=reader, system=mm.system_prompt(ep.bench), thinking=thinking,
                 tag=tag + ":reduce", max_output_tokens=MAX_OUT, seed=SEED)
    return g["text"].strip(), usages + [g["usage"]], info


def session_closure(ep: Episode, idxs: Sequence[int], window: Optional[int] = None) -> List[int]:
    """Complete a selected set within sessions: all rounds of every touched session (window=None), or the rounds within
    +-window of each selected round in the same session. Returns indices in chronological order."""
    if window is None:
        return [j for u in session_units(ep, idxs) for j in u]
    out = set(idxs)
    for i in idxs:
        for j in range(max(0, i - window), min(len(ep.rounds), i + window + 1)):
            if ep.rounds[j].sid == ep.rounds[i].sid:
                out.add(j)
    return sorted(out)


# ----------------------------------------------------------------------------- Phase 5: visibility-preserving gist index
GIST_SYS = (
    "You write index entries for a personal assistant's long-term memory. Given one round of a user's conversation (the "
    "user's message and short descriptions of any images they shared), write ONE line of at most 40 words, in telegraphic "
    "style, that lists every concrete thing the round mentions or shows that the user might later ask about: items and "
    "purchases with prices, people and pets, places, dates and times, plans, preferences, changes, numbers, and what the "
    "images show. Include brief asides, not only the main topic. Output only the line.")
GIST_MODEL = "gemini-2.5-flash-lite"
GISTS: Dict[str, str] = {}


def _gist_key(r: Round) -> str:
    return hashlib.sha256((r.user + "||" + "|".join(r.images)).encode()).hexdigest()[:24]


def build_gists(eps: Sequence[Episode], caps: Dict[str, str], workers: int = 16) -> Dict[str, str]:
    """One cheap call per unique round (user message + image captions), cached; fills mm2.GISTS."""
    todo = {}
    for e in eps:
        for r in e.rounds:
            k = _gist_key(r)
            if k not in GISTS and k not in todo:
                todo[k] = r

    def one(item):
        k, r = item
        txt = f"User message: {r.user}\n" + "".join(f"[image: {caps.get(p, '')}]\n" for p in r.images)
        g = generate([txt], model=GIST_MODEL, system=GIST_SYS, thinking=0, tag="p5:gist", max_output_tokens=120)
        return k, (g["text"] or "").strip().replace("\n", " ")[:400]

    for k, v in pmap(one, list(todo.items()), workers=workers, desc="gists"):
        GISTS[k] = v
    return GISTS


def build_anticipated_usage(eps: Sequence[Episode], path: str, max_items: int = 40) -> Dict[str, List[dict]]:
    """Phase 6 (AnSR): for each real question, the anticipated resolutions of its history (written at ingestion by the
    strong model from the index alone), most similar first."""
    from gem import embed
    A = json.load(open(path))["per_history"]
    out = {}
    for e in eps:
        res = A.get(e.name) or []
        if not res or not e.qas:
            continue
        Va = embed([mm.QA_QUERY_FMT.format(r["question"]) for r in res])
        Vq = embed([mm.QA_QUERY_FMT.format(qa.question) for qa in e.qas])
        S = Vq @ Va.T
        for i, qa in enumerate(e.qas):
            order = np.argsort(-S[i])[:max_items]
            out[qa.qid] = [{"question": res[j]["question"], "rids": res[j]["rids"]} for j in order]
    return out
