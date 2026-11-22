"""Phase 22: ask two LLM reviewers for a critical review of the compiled paper.
  .venv/bin/python scripts/p22_review.py  -> results/p22/review_gpt5.md (text of the PDF), results/p22/review_gemini.md (page images)"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fitz  # noqa: E402
import gem  # noqa: E402

OUT = ROOT / "results/p22"
PDF = ROOT / "paper_acl/main_polished.pdf"
PROMPT = """You are an experienced, critical reviewer for the ACL main conference (long paper). Review the paper below as you
would for ACL Rolling Review. Be concrete and cite sections, tables, and figures. Do not be polite for its own sake.

Write the review with these parts:
1. Summary (3-5 sentences): what is claimed and what the evidence is.
2. Strengths (short).
3. Weaknesses, ordered from most to least serious. For each: what exactly is wrong or missing, where in the paper, why it
   matters for the claims, and what experiment or change would fix it. Separate (a) problems with the claims or evidence
   from (b) problems of presentation.
4. Claims that the evidence does not support, or that are stated more strongly than the numbers allow (quote the sentence).
5. Internal inconsistencies: numbers, definitions, or statements that disagree between the abstract, the text, tables,
   figures, and the appendix.
6. Missing experiments or controls that a reviewer would ask for, ordered by importance.
7. Questions for the authors.
8. Overall recommendation on the ARR scale (1-5) with a one-paragraph justification, and the two or three changes that would
   raise the score the most.
"""


def paper_text():
    d = fitz.open(PDF)
    pages = []
    for i in range(d.page_count):
        t = re.sub(r"\n\d{3,4}\n", "\n", d[i].get_text())
        pages.append(f"[page {i + 1}]\n{t}")
    return "\n".join(pages)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = paper_text()
    r = gem.generate_or(PROMPT + "\n\n===== PAPER =====\n" + text, model="openai/gpt-5", max_output_tokens=16000, tag="p22:review", use_cache=False)
    (OUT / "review_gpt5.md").write_text(r["text"])
    print("gpt-5 review:", len(r["text"]), "chars", flush=True)
    d = fitz.open(PDF)
    (OUT / "pages").mkdir(exist_ok=True)
    parts = [PROMPT, "\n===== PAPER (page images follow) =====\n"]
    for i in range(d.page_count):
        p = OUT / "pages" / f"p{i + 1:02d}.png"
        d[i].get_pixmap(dpi=110).save(p)
        parts += [f"\n[page {i + 1}]", gem.Img(str(p), "high")]
    try:
        g = gem.generate(parts, model="gemini-3.8-flash", thinking="high", tag="p22:review", max_output_tokens=16000, use_cache=False)
    except Exception as e:  # noqa: BLE001
        print("high thinking failed:", repr(e)[:200], flush=True)
        g = gem.generate(parts, model="gemini-3.8-flash", thinking="low", tag="p22:review", max_output_tokens=16000, use_cache=False)
    (OUT / "review_gemini.md").write_text(g["text"])
    print("gemini review:", len(g["text"]), "chars", flush=True)


if __name__ == "__main__":
    main()
