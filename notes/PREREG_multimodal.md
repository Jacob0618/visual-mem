# Pre-registration: held-out confirmation of the MemEye-dev findings

Written 2026-10-26 after the MemEye-dev pilots (results/mm/memeye_dev, 206 q) and BEFORE any answer was generated on
MemEye-test (165 q, tasks fixed in results/memeye_split.json) or on the Mem-Gallery subset (903 q, fixed in
results/memgal_subset.json, stratified by scenario x question type, seed 20261026). No method, prompt, k, resolution
or caption prompt will be changed for these runs. Reader = judge = gemini-3.8-flash (thinking=low, T=0);
MemEye 0/0.25/0.5/0.75/1 judge rubric (prompts/memeye_llm_judge.txt) for BOTH benchmarks; embedding
gemini-embedding-2 (3072-d); captions written by gemini-3.8-flash from the image alone (prompts in src/mm.py).

## Conditions (identical code paths to dev)
C1 fc|raw:low · C2 fc|raw:high · C3 fc|cap:dense · C4 cap:dense:10|raw:low · C5 cap:dense:40|raw:low ·
C6 oracle|raw:low · C7 oracle|cap:dense · C8 oracle|cap:short · C9 fc|none ·
C10 cap:dense:10|raw:high · C11 inter:10|raw:high · C12 fc+cap:dense:10|mix:low:high

## Hypotheses and decision rules (paired over the same questions; 95% bootstrap CI over questions, 10k reps)
H1 (read-everything beats retrieval when the history fits): C1 − C4 > 0 and C1 − C5 > 0 (CI lower bound > 0).
H2 (resolution): C1 is non-inferior to C2 with margin 0.03 (CI lower bound of C1 − C2 > −0.03).
H3 (write-time rewriting): C1 − C3 ≥ 0 (point estimate) AND mean prompt tokens of C3 > C1.
    Confirmed if CI lower bound of C1 − C3 > −0.03 and the token inequality holds; "strongly" if CI > 0.
H4 (rewrite granularity): C8 < C7 (CI upper bound of C8 − C7 < 0); and C7 ≤ C6 (point estimate).
H5 (index key; retrieval only, scripts/retrieval_mm.py): gold-clue coverage@10 cap:dense > inter > img
    (each pairwise CI excludes 0). End-to-end: C10 − C11 > 0.
H6 (zoom): C12 − C1 within ±0.03 (predicted null; reported two-sided).
Dev effect sizes for reference: H1 +0.170 / +0.104; H2 +0.021; H3 +0.044 (65k vs 32k tok); H4 −0.379, −0.049;
H5 0.758 / 0.647 / 0.210; H6 −0.022.

## Not hypotheses (exploratory, will be labelled as such)
Per-cell (X/Y, Mem-Gallery types) breakdowns; session/neighbour expansion; scale stress tests; test-retest noise.
Mem-Gallery differs from MemEye (persona-driven multi-topic chats, less visual necessity per MemEye's own analysis,
VS answers are image IDs), so H1-H6 failing there is informative about boundary conditions, not a pipeline bug.

## Deviation 1 (logged before any Mem-Gallery answer was generated)
Cost: on dev, C12 (zoom) cannot use prefix caching (~$26 per 1k questions). Primary conditions C1-C8 run on the full
903-question Mem-Gallery subset; secondary C9-C12 run on a fixed random half (results/memgal_subset_half.json, 451 q,
seed 7). MemEye-test runs all conditions on all 165 questions as registered.
