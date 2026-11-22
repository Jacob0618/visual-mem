RAG@30 and full-context figure — 2026-11-11

This figure uses new multimodal RAG@30 results generated with the existing scripts/p14_read_ids.py --reader strong. No evaluation source code was changed. Input JSON files take exactly the first 30 round IDs from each main-table v1 RAG@40 cosine ranking. Using these stored rankings avoids changing the retrieval index along with k. The existing mm2.read prompt, raw images, low media resolution, low thinking, 2048 output cap and benchmark judges were reused. The three runs contain 1,217 questions in total, with 0 errors, 0 empty predictions and exactly 30 retrieved rounds per question. All full-context scores come from existing paired records; no full-context inference was rerun. The logs estimate $9.23 in new API usage.

Benchmark          N      k       RAG(%)    Full(%)   Full-RAG(pp)
LoCoMo             1540   100     91.2338   91.7532    +0.5195
LongMemEval-S      500    30      90.4000   90.8000    +0.4000
MemEye             371    30      76.4151   81.9407    +5.5256
MemLens-32k        150    30      65.3333   74.0000    +8.6667
VisualMem          696    30      79.3103   96.6954   +17.3851

Figure conventions
Bars show RAG accuracy; white diamonds show full context. Signed labels above bars show Full minus RAG in percentage points. k is the true retrieval budget. LoCoMo 91.2@100 is retained as requested; the other four benchmarks use30. The MemLens 150 suffix is omitted from the figure, while the actual paired subset remains documented here and in Appendix B. Graph colors and typeface follow the paper.

Pairing and protocol
Every RAG/full-context pair uses identical question IDs and judges with Gemini 3.8 Flash as the answering model. LoCoMo uses the historical U2 prompt/e8 lenient binary judge for both conditions; its full run covers all 1,540 questions. LongMemEval-S uses the full 500 questions with the official judge and the same 2,000-character-per-round representation in both conditions. The MemLens comparison uses the unchanged shared 150-question main-table subset.

MemEye full context retains its original phase-1 generation setting without an explicit output cap; RAG@30 and the old RAG@40 both use 2,048. Stored full answers are short (at most 60 answer tokens), but 24 full runs report more than 2,048 thinking tokens, so the generation-budget difference remains documented. Finish reasons were not retained in those full records. All other visual full runs use the mm2 reader. This is not a uniform-budget causal test of modality across benchmarks; the per-benchmark scores and paired gaps are reported under their recorded conditions.

Reproduction and files
make_figure.py plus rag_accuracy.csv regenerate PDF/SVG/PNG without API calls.
paired_scores.csv contains all paired per-question scores.
data_provenance.json records exact input paths, filters, question sets and source hashes.
text_full_audit.json / visual_full_audit.json describe full-context matching and protocol checks.
completed_results_audit.json independently verifies the three new runs.
experiment_results/ contains the original k30 records, selected-ID inputs, run manifest, summary and exact existing-runner commands. Run those commands from the original visual mem project, using its existing environment and data; no new evaluation runner is needed.

Manuscript
Figure 1 is at the first page's upper right. Only its figure/caption and associated Appendix B protocol paragraph changed this round. Original manuscript files, abstract, existing tables and their results remain unchanged. Main text ends on page 7; the complete PDF has 18 pages.

Redesign — 2026-11-16
The figure now uses one bar per benchmark: the solid part is RAG accuracy, the hatched part on top is the gap to
full-context reading, and the label above gives that gap in points; light bands separate text and multimodal memory.
The data (rag_accuracy.csv) and all numbers are unchanged. make_figure.py draws the new design;
make_figure_v1_diamonds.py is the previous design (diamonds for full context). The caption in
sections_polished/fig_rag_motivation.tex was updated to match; the main text now ends on page 8.

Update — 2026-11-21 (phase 16)
The three multimodal bars were re-run: RAG@30 uses the first 30 rounds of the current cosine rankings (regenerated image
descriptions), the reader instruction without the example phrase on MemEye, and all 466 MemLens-32k test questions
(per-question files: results/p16/fig1/<bench>_rag30_strong.jsonl; full context: results/p16/final2/memeye, results/p14/runs/
memlens_fc + results/p16/final/memlens_fc, results/p14/runs/visualmem). New values: MemEye 75.0 / 81.0, MemLens 65.2 / 69.1,
VisualMem 80.6 / 96.7. The previous CSV is rag_accuracy_v1_150subset.csv; paired_scores.csv and the audit JSON files describe
that previous version.
