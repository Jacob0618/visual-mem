"""Multimodal memory pipeline: write-time captioning, round indexing, retrieval, reading, judging.

Memory unit = one dialogue round (user turn + assistant turn + any images shared in it), stored raw.
A *condition* = (selection, representation):
  selection:      fc (all rounds) | oracle (gold clue rounds) | <key>:<k> (cosine top-k over a round key)
  representation: raw:<res>  (original pixels at media resolution res)
                  cap:<style> (images replaced by write-time captions; no pixels)
                  both:<style>:<res> (caption + pixels)   none (image placeholders only)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_mm import QA, Episode, Round  # noqa: E402
from gem import DOC_FMT, QA_QUERY_FMT, Img, embed, generate, pmap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "prompts"
READER = "gemini-3.8-flash"
JUDGE = "gemini-3.8-flash"

# ----------------------------------------------------------------------------- write-time captioning (= rewriting)
CAPTION_PROMPTS = {
    # what typical memory pipelines store: a short generic caption
    "short": "Write a one- or two-sentence caption describing this image.",
    # strongest query-agnostic rewrite: exhaustive record (modelled on MemEye's 'task-aware' captions, App. D.2)
    "dense": (
        "You are writing the permanent memory record of an image that a user shared in a conversation. Later, someone "
        "may ask ANY question about this image, but nobody will be able to look at it again, so your text must "
        "replace the image. Write an exhaustive, strictly factual description:\n"
        "1. Overall: what kind of image it is (photo, screenshot, chart, drawing, ...) and the scene/setting.\n"
        "2. ALL visible text verbatim (signs, labels, UI text, numbers, chart axes/values, handwriting).\n"
        "3. Every salient object, person, animal or character: identity cues, appearance, clothing, colour (precise "
        "shade), material, texture, pattern, size, count, state/condition.\n"
        "4. Spatial layout: where things are (left/right/top/bottom/foreground/background), relative positions, "
        "ordering, grouping.\n"
        "5. Anything distinctive, unusual, damaged, missing, or easy to confuse with something similar.\n"
        "Do not speculate beyond what is visible. Use compact bullet points."),
}


def caption(path: str, style: str) -> str:
    r = generate([Img(path, "high"), CAPTION_PROMPTS[style]], model=READER, thinking="low", tag=f"caption:{style}")
    return r["text"].strip()


def build_captions(episodes: Sequence[Episode], style: str, workers: int = 24) -> Dict[str, str]:
    paths = sorted({p for e in episodes for r in e.rounds for p in r.images} |
                   {p for e in episodes for _, p in e.persona_images})
    caps = pmap(lambda p: caption(p, style), paths, workers=workers, desc=f"caption:{style}")
    return dict(zip(paths, caps))


# ----------------------------------------------------------------------------- rendering
def _image_label(r: Round, j: int) -> str:
    return r.image_ids[j] if j < len(r.image_ids) else f"{r.rid}#img{j + 1}"


def round_plain(r: Round, caps: Optional[Dict[str, str]] = None) -> str:
    s = f"[{r.sid} | {r.date}] User: {r.user}"
    for j, p in enumerate(r.images):
        s += f"\n[Image {_image_label(r, j)}]" + (f" {caps[p]}" if caps is not None else "")
    return s + f"\nAssistant: {r.assistant}"


def parse_rep(rep: str):
    """'raw:high' -> ('raw', None, 'high'); 'cap:dense' -> ('cap','dense',None); 'both:dense:low'; 'none';
    'mix:low:high' -> ('raw', None, 'low') with 'high' for the zoom set (see build_prompt hi_idxs)."""
    p = rep.split(":")
    if p[0] in ("raw", "mix"):
        return "raw", None, p[1]
    if p[0] == "cap":
        return "cap", p[1], None
    if p[0] == "both":
        return "both", p[1], p[2]
    return "none", None, None


def _render_images(label, paths, mode, style, res, caps):
    out = []
    for lab, p in zip(label, paths):
        if mode in ("raw", "both"):
            out += [f"\n[Image {lab}]", Img(p, res)]
        if mode in ("cap", "both"):
            out.append(f"\n[Image {lab} — description written when it was shared]: {caps[style][p]}")
        if mode == "none":
            out.append(f"\n[Image {lab} — not available]")
    return out


def render_round(r: Round, mode, style, res, caps) -> list:
    parts = [f"\n[{r.sid} | {r.date} | round {r.rid}]\nUser: {r.user}"]
    parts += _render_images([_image_label(r, j) for j in range(len(r.images))], r.images, mode, style, res, caps)
    parts.append(f"\nAssistant: {r.assistant}\n")
    return parts


def _merge(parts):
    out = []
    for p in parts:
        if isinstance(p, str) and out and isinstance(out[-1], str):
            out[-1] += p
        else:
            out.append(p)
    return out


SYSTEM = (
    "You are an AI assistant with long-term memory of a multi-session conversation with a user. You will be shown "
    "parts of that history (possibly all of it), then a question.\n"
    "Images in the history may appear either as the original images or as text descriptions that were written at "
    "the time the image was shared.\n"
    "Rules:\n"
    "1. Ground every answer in the provided history and images. Do not invent facts.\n"
    "2. When pieces of information conflict, prefer the most recent evidence unless the question asks about the "
    "conflict or an earlier state.\n"
    "3. Inspect images carefully; be precise about positions, colours, text, counts and other visual attributes.\n"
    "4. If the required information is genuinely absent, say so explicitly rather than guessing.\n"
    "5. {image_rule}\n"
    "6. Give ONLY the answer itself - no reasoning or restating of the question. Counting: reply with just the "
    "number. Yes/no: start with Yes or No, then only the essential corrective detail. Descriptive: one or two short "
    "sentences at most.")
IMAGE_RULE = {  # Mem-Gallery 'VS' gold answers are image IDs; MemEye gold answers describe images in words
    "memlens": "Information shared in images (receipts, screenshots, photos) counts as part of the conversation; read "
               "them carefully and follow the answer format requested in the question.",
    "memgal": "If the question asks which image, answer with the image ID shown in brackets (e.g. D1:IMG_002).",
    "memeye": "When the answer is an image or screen, name it the way the user/conversation described it, not by "
              "its bracketed ID.",
    "visualmem": "Images shared in the conversation are part of the user's history, and the answer may depend on "
                 "details that are visible only in them. Answer with the letter of the correct option only.",
    "camroll": "The history is the user's personal camera roll: every entry is one photo with its capture time, and "
               "the answer depends on details that are visible in the photos. Answer with the letter of the correct "
               "option only.",
    "wma": "Information shared in images (screenshots, photos, documents) counts as part of the conversation. If the "
           "question asks which image, answer with the image ID shown in brackets (e.g. personal_01_img_019); otherwise "
           "answer in words.",
}


def system_prompt(bench: str) -> str:
    return SYSTEM.format(image_rule=IMAGE_RULE[bench])


def build_prompt(ep: Episode, qa: QA, idxs: Sequence[int], rep: str, caps, hi_idxs: Sequence[int] = ()) -> list:
    """hi_idxs: rounds whose images are rendered at the second resolution of a 'mix:<lo>:<hi>' representation."""
    mode, style, res = parse_rep(rep)
    res_hi = rep.split(":")[2] if rep.startswith("mix:") else res
    hi = set(hi_idxs)
    parts = [f"User profile: {json.dumps(ep.profile, ensure_ascii=False)[:1500]}\n"]
    if ep.persona_images:
        parts.append("\nPeople who appear in the user's saved chats (reference photos):")
        parts += _render_images([lab for lab, _ in ep.persona_images], [p for _, p in ep.persona_images],
                                mode, style, res, caps)
        parts.append("\n")
    parts.append(f"\n=== CONVERSATION HISTORY ({len(idxs)} of {len(ep.rounds)} rounds, chronological) ===")
    for i in sorted(idxs):
        parts += render_round(ep.rounds[i], mode, style, res_hi if i in hi else res, caps)
    parts.append("\n=== END OF HISTORY ===\n\nQuestion: " + qa.question)
    if qa.question_image:
        parts += ["\n[Image attached to the question]", Img(qa.question_image, "high")]
    for lab, path in (getattr(qa, "option_images", None) or []):
        parts += [f"\n[Option {lab}]", Img(path, "low")]
    parts.append("\nAnswer:")
    return _merge(parts)


# ----------------------------------------------------------------------------- indexing / retrieval
class RoundIndex:
    """Per-episode round embeddings under several keys (all gemini-embedding-2, 3072-d, L2-normalized)."""

    def __init__(self, ep: Episode, caps: Dict[str, Dict[str, str]], keys: Sequence[str], dim: Optional[int] = None):
        self.ep, self.M = ep, {}
        R = ep.rounds
        for key in keys:
            if key == "text":
                items = [DOC_FMT.format(round_plain(r)) for r in R]
            elif key.startswith("cap:"):
                items = [DOC_FMT.format(round_plain(r, caps[key[4:]])) for r in R]
            elif key == "inter":   # round text + its images -> ONE aggregated embedding
                items = [[DOC_FMT.format(round_plain(r)), *[Img(p) for p in r.images]] for r in R]
            elif key == "img":     # image-only; rounds without images fall back to their text
                items = [[Img(p) for p in r.images] if r.images else DOC_FMT.format(round_plain(r)) for r in R]
            else:
                raise ValueError(key)
            self.M[key] = embed(items, dim=dim)

    def query_vec(self, qa: QA, dim=None):
        q = QA_QUERY_FMT.format(qa.question)
        item = [q, Img(qa.question_image)] if qa.question_image else q
        return embed([item], dim=dim)[0]

    def scores(self, qa: QA, key: str) -> np.ndarray:
        return self.M[key] @ self.query_vec(qa)

    def topk(self, qa: QA, key: str, k: int) -> List[int]:
        s = self.scores(qa, key)
        return [int(i) for i in np.argsort(-s)[:k]]


DECOMP_PROMPT = (
    "A user is asking a question about their own long multimodal conversation history (messages and shared images "
    "or screenshots). Before answering, we must search the history. Write 1 to 5 short search queries; each query "
    "should describe ONE specific past message or image that must be found to answer the question (include its "
    "distinguishing content or visual cues, and which item in a series it is if relevant). If the question compares "
    "or counts several items, write one query per item or group.\nQuestion: {q}")
_DECOMP_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                  "required": ["queries"]}


def decomposed_topk(ix: RoundIndex, qa: QA, key: str, k_total: int, k_orig: int = 10, k_sub: int = 5) -> List[int]:
    """Union of top-k_orig for the question and top-k_sub for each LLM-written sub-query, ranked by best score."""
    r = generate(DECOMP_PROMPT.format(q=qa.question), model=READER, thinking="low", json_schema=_DECOMP_SCHEMA,
                 tag="decomp")
    try:
        subs = [s for s in json.loads(r["text"])["queries"] if isinstance(s, str) and s.strip()][:5]
    except Exception:  # noqa: BLE001
        subs = []
    best = {}
    s0 = ix.scores(qa, key)
    for i in np.argsort(-s0)[:k_orig]:
        best[int(i)] = float(s0[i]) + 1.0  # original-question hits first
    if subs:
        Q = embed([QA_QUERY_FMT.format(s) for s in subs])
        S = Q @ ix.M[key].T
        for row in S:
            for i in np.argsort(-row)[:k_sub]:
                best[int(i)] = max(best.get(int(i), -9), float(row[i]))
    return [i for i, _ in sorted(best.items(), key=lambda x: -x[1])[:k_total]]


def clue_stats(ep: Episode, qa: QA, idxs: Sequence[int]) -> dict:
    r2i = ep.rid2idx
    gold = {r2i[c] for c in qa.clue if c in r2i}
    got = set(idxs)
    if not gold:
        return {"n_gold": 0}
    d = {"n_gold": len(gold), "cov": len(gold & got) / len(gold), "any": float(bool(gold & got)),
         "full": float(gold <= got)}
    if qa.latest_clue and qa.latest_clue in r2i:
        d["latest"] = float(r2i[qa.latest_clue] in got)
    return d


# ----------------------------------------------------------------------------- answer + judge
def answer(ep: Episode, qa: QA, idxs: Sequence[int], rep: str, caps, tag: str, hi_idxs: Sequence[int] = (),
           seed=None, reader: str = READER, thinking="low") -> dict:
    parts = build_prompt(ep, qa, idxs, rep, caps, hi_idxs)
    r = generate(parts, model=reader, system=system_prompt(ep.bench), thinking=thinking, tag=tag, seed=seed)
    return {"pred": r["text"].strip(), "usage": r["usage"], "latency_s": r.get("latency_s"), "cached": r["cached"]}


_JUDGE_T = (PROMPTS / "memeye_llm_judge.txt").read_text()
_JUDGE_SCHEMA = {"type": "object", "properties": {"score": {"type": "number", "enum": [0, 0.25, 0.5, 0.75, 1]},
                                                  "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]}


def judge(question: str, gt: str, pred: str, model: str = JUDGE) -> dict:
    pred = pred if len(pred) <= 2000 else pred[:2000] + " …[truncated]"  # guard against degenerate outputs
    p = _JUDGE_T.replace("{{question}}", question).replace("{{ground_truth}}", gt).replace("{{model_output}}", pred)
    for attempt in range(3):
        r = generate(p, model=model, thinking="low", json_schema=_JUDGE_SCHEMA, tag="judge", seed=attempt or None)
        try:
            o = json.loads(r["text"])
            return {"score": float(o["score"]), "reason": o.get("reasoning", "")}
        except Exception:  # noqa: BLE001
            m = re.search(r'"score"\s*:\s*([0-9.]+)', r["text"] or "")
            if m:
                return {"score": float(m.group(1)), "reason": "regex"}
    return {"score": 0.0, "reason": "unparseable"}
