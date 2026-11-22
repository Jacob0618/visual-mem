"""Text memory baseline: raw units (no rewriting) + gemini-embedding-2 cosine top-k, chronological, one reader call."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_text import TEpisode, TQA  # noqa: E402
from gem import DOC_FMT, QA_QUERY_FMT, embed, generate  # noqa: E402

READER = "gemini-3.8-flash"
EMB_CHARS = 24000  # ~6k tokens; embedding-2 input limit is 8192 tokens (reader always sees the full unit)

_idx_cache = {}


def unit_matrix(ep: TEpisode, dim: int = 768) -> np.ndarray:
    k = (ep.name, len(ep.units), dim)
    if k not in _idx_cache:
        _idx_cache[k] = embed([DOC_FMT.format(u.text[:EMB_CHARS]) for u in ep.units], dim=dim)
    return _idx_cache[k]


def topk(ep: TEpisode, qa: TQA, k: int, dim: int = 768) -> List[int]:
    M = unit_matrix(ep, dim)
    q = embed([QA_QUERY_FMT.format(qa.question)], dim=dim)[0]
    return [int(i) for i in np.argsort(-(M @ q))[:k]]


def evidence_recall(ep: TEpisode, qa: TQA, idxs: Sequence[int]) -> dict:
    gold = {i for i, u in enumerate(ep.units) if u.uid in set(qa.evidence)}
    if not gold:
        return {"n_gold": 0}
    got = set(idxs)
    return {"n_gold": len(gold), "cov": len(gold & got) / len(gold), "full": float(gold <= got)}


LOCOMO_SYS = (
    "You answer questions about the history of a long conversation between two people, using the excerpts provided "
    "(each line is one message with the date of its session). Excerpts are in chronological order. Resolve relative "
    "time expressions using the message dates (e.g. 'last year' said in 2023 means 2022; 'yesterday' on 8 May 2023 "
    "means 7 May 2023) and answer with absolute dates when asked when. Answer concisely (a short phrase). If the "
    "excerpts do not contain the answer, say that the information is not available.")
LME_SYS = (
    "You are a chat assistant with long-term memory of past conversations with the user. You are given excerpts of "
    "the past chat history (each excerpt is one exchange with its date, in chronological order) and the current "
    "date. Answer the user's question using this history. When facts changed over time, use the most recent "
    "information unless asked otherwise. Use the dates for any temporal reasoning. If the history does not contain "
    "the needed information, say so explicitly. Answer concisely.")


def build_prompt(ep: TEpisode, qa: TQA, idxs: Sequence[int], bench: str) -> str:
    lines = "\n\n".join(ep.units[i].text for i in sorted(idxs))
    head = f"=== HISTORY EXCERPTS ({len(idxs)} of {len(ep.units)} units) ===\n{lines}\n=== END ===\n\n"
    if bench == "lme":
        return head + f"Current date: {qa.question_date}\nQuestion: {qa.question}\nAnswer:"
    return head + f"Question: {qa.question}\nAnswer:"


def answer(ep: TEpisode, qa: TQA, idxs: Sequence[int], bench: str, tag: str, reader: str = READER) -> dict:
    r = generate(build_prompt(ep, qa, idxs, bench), model=reader, system=LME_SYS if bench == "lme" else LOCOMO_SYS,
                 thinking="low", tag=tag)
    return {"pred": r["text"].strip(), "usage": r["usage"], "latency_s": r.get("latency_s")}


# LongMemEval official answer-check templates (github.com/xiaowu0162/LongMemEval, src/evaluation/evaluate_qa.py, MIT)
_T_STD = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
          "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
          "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
          "only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect "
          "Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
_T_TEMP = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
           "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
           "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
           "only contains a subset of the information required by the answer, answer no. In addition, do not penalize "
           "off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., "
           "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response "
           "is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
           "correct? Answer yes or no only.")
_T_KU = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
         "contains the correct answer. Otherwise, answer no. If the response contains some previous information along "
         "with an updated answer, the response should be considered as correct as long as the updated answer is the "
         "required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
         "correct? Answer yes or no only.")
_T_PREF = ("I will give you a question, a rubric for desired personalized response, and a response from a model. Please "
           "answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to "
           "reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's "
           "personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model "
           "response correct? Answer yes or no only.")
_T_ABS = ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if "
          "the model correctly identifies the question as unanswerable. The model could say that the information is "
          "incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
          "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
          "Answer yes or no only.")


# Mem0's LoCoMo LLM-judge prompt (mem0ai/mem0 evaluation/metrics/llm_judge.py ACCURACY_PROMPT, as reproduced in
# public harnesses); lenient ("be generous") - used only as a secondary metric for comparability with the literature.
_T_MEM0 = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def judge_mem0(question: str, gold: str, pred: str, model: str = READER) -> float:
    r = generate(_T_MEM0.format(question=question, gold_answer=gold, generated_answer=pred), model=model,
                 thinking="low", tag="judge-mem0")
    t = (r["text"] or "").upper()
    return 1.0 if "CORRECT" in t and "WRONG" not in t else 0.0


def judge(qa: TQA, pred: str, model: str = READER) -> float:
    if qa.abstention:
        t = _T_ABS
    else:
        t = {"temporal-reasoning": _T_TEMP, "knowledge-update": _T_KU, "single-session-preference": _T_PREF}.get(
            qa.qtype, _T_STD)
    r = generate(t.format(qa.question, qa.answer, pred), model=model, thinking="low", tag="judge-text")
    return 1.0 if "yes" in (r["text"] or "").strip().lower()[:10] else 0.0
