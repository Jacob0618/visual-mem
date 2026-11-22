# AGENTS.md — notes for future sessions in this project

- Python: `.venv/bin/python` (3.12). Install list in README.md. API key is read at runtime by `src/keys.py`
  (env `GEMINI_API_KEY`, else `./.env`, else `~/Desktop/jevmem/.env`); never print or copy it.
- `~/Desktop/jevmem/data` is the user's earlier project: read-only source of LoCoMo / LongMemEval files. Do not write there.
- Every Gemini call goes through `src/gem.py` (sqlite cache in `cache/`; identical requests are free). Check spend with
  `.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import gem; print(gem.spend())"`.
- Gotchas verified 2026-10-26:
  - `gemini-3.8-flash` rejects `thinking_level=minimal`; use `low`.
  - `gemini-embedding-2`: a list of strings returns ONE aggregated vector -> wrap each item in its own `Content`;
    `task_type` is ignored -> use text prefixes (`task: question answering | query: ...`, `title: none | text: ...`).
  - Send images via the Files API (`gem.ensure_uploaded`), not inline; inline multi-MB images collapse throughput.
  - Set an HTTP timeout on the genai client (done in gem.py) or long multimodal calls can hang forever.
  - `gemma-4-31b-it` is free but limited to 16k input tokens/min: use <=2 workers.
  - Implicit prefix caching works (84-100% hit after a warm-up request per shared history, up to 722k tokens):
    put the shared history first and the question last.
- Regenerate all report tables (no API calls): `.venv/bin/python scripts/make_tables.py` -> `results/TABLES.md`.
- Single-run differences of ~5 pp on <=100 questions can be pure decoding noise; replicate with `--seed 1`.
- gemini-2.5-flash-lite: thinking via budget (int; `thinking_level` -> 400); media resolution is per REQUEST and HIGH
  only applies to single-image requests (>=2 images -> 258 tok each); at T=0 without thinking it is deterministic
  (seeds change nothing). gem.generate lifts per-image Img.res into the request-level setting for 2.5 models.
- Phase-2 calls cap output at 2048 tokens (a JSON call once looped to 65k tokens). Do not `pkill -f` with patterns
  containing '|' (it is a regex alternation and kills unrelated runs).
- Phase-2 tables: `.venv/bin/python scripts/make_tables_p2.py` -> `results/TABLES_P2.md`.

## Phase 4 notes (held-out MemLens, cross-family models)
- MemLens (data/MemLens, CC-BY-4.0) is the HELD-OUT benchmark: do not tune prompts/index on it; its registration is
  RESEARCH_LOG P4-D2. Loader: src/data_memlens.py (one Episode per question; gold = answer-session rounds with a
  has_answer turn or a needle image). Judge: official prompts (data/MemLens/judge_prompts) via mm2.memlens_judge.
- Cross-family models go through gem.generate_or (OpenRouter, text only, cached like generate; key OPENROUTER_API_KEY,
  user's limit $50). Aliases in run_mm2/select_eval: qwen, qwent, gpt5m, gpt5, qwenvl, ds.
- mm2.llm() routes model names containing '/' to OpenRouter; Gemini cache keys are unchanged.
- MemEye scenario-level statistics: scripts/cluster_stats.py. Index visibility: results/index_visibility.txt.

## Phase 5 notes (mechanism study)
- MemLens dev/test split for new mechanisms: results/memlens_p5_split.json; the TEST part was used once (P5-D8/D9).
- Reader modes in run_mm2: eva/evar (session notes), evau/evaur (round notes), readn (round notes + all raw rounds);
  selectors: sess:k, gran:k, granf:k, toc38s:N (session closure), toc38n:N:w. --snippet 600, --snippet_mode head|query|gist.
- VGS is evaluated by combining per-question outputs with scripts/gate_eval.py and results/p5/visibility_hat.json.

## Phase 6 notes
- run_mm2 stores a question whose reading OR judging raised as {"error": ..., "score": 0, "pred": ""}. Always check the
  'errors' count before using a result file (gate_eval/route_eval/p6_confirm now refuse files with failed records).
  The OpenRouter key expired once (401) and silently zeroed many MemLens judgments (RESEARCH_LOG P6-D6).
- Explicit context caching: gem.generate(cache_prefix=...), mm2.XCACHE / run_mm2 --xcache; creations are logged with
  tag 'xcache:create' (prompt tokens + storage_usd). --fresh_sel draws a new QSS selection (bypasses the local cache).
- Set-ness router labels: results/p6/setness.json (scripts/setness.py). Visibility estimates at 64k:
  results/p6/visibility_hat_64k.json.

## Phase 7 notes
- Index compression: run_mm2 / select_eval --compact noassist,shortid,telegraph (mm2.NO_ASSIST / SHORT_IDS / TELEGRAPH).
  With shortid the selector answers r<k>; toc_select maps them back. For read+toc (index as reader context) use
  noassist,telegraph only, so that index lines and opened rounds share ids.
- gemini-3.8-flash: thinking_budget=0 is accepted but NOT honoured (thinking stays stochastic, ~1/3 of calls);
  'minimal' is rejected. Do not claim a no-thinking selector.
- Long histories: scripts/pool_scale.py (candidate-pool recall at 4k rounds); prefilter selector ptoc38:P:N.

## Phase 10 notes
- ACL paper: `paper_acl/` (official acl.sty from acl-org/acl-style-files); build with `cd paper_acl && ../tools/tectonic -X compile main.tex`.
  Main text must end by page 8 (Limitations/Ethics/References/Appendix do not count); a half-page main-figure placeholder sits at the top of page 3.
- Unified gated results are composed offline by `scripts/p10_unified.py` (no API calls).
- VisualMem: loader `src/data_visualmem.py` (bench `visualmem`, exact-letter scoring); images fetched by HTTP range requests from the
  HF zip with `scripts/p10_fetch_visualmem.py` (`src/remotezip.py`). Inclusion criterion is in RESEARCH_LOG P10-D0.
- If Gemini returns 400 FAILED_PRECONDITION "User location is not supported", the machine's network location is the cause; runs stall.

## Phase 11 notes (external memory systems)
- Baseline environments: `.venv_bl` (SimpleMem / Omni-SimpleMem, unified package in baselines/SimpleMem, lancedb pinned
  0.25.3, google-genai 2.33.0) and `.venv_mirix` (mirix 0.1.5). Runners: scripts/p11_baselines.py (use --threads for
  Omni; HF_HUB_OFFLINE=1), scripts/p11_mirix.py + scripts/p11_read_ctx.py; tables/figure: scripts/p11_table.py.
- MIRIX gotchas: it DELETES uploaded files (pass copies, delete_after_upload=False), ignores MIRIX_DIR (set HOME per
  history), needs gemini-embedding-001, cannot parse Gemini 3 responses, declares MIME from the file extension, posts
  without timeouts and hangs; on long MemLens sessions it costs ~$0.25 per session. Cap spend before running it again.
- Never pkill with a pattern that also appears in the launching command line (it kills the tool's own shell).

## Phase 12 notes
- The gem response/embedding caches (cache/gen.sqlite, cache/emb.sqlite) were deleted after phase 12 at the user's
  request: any new analysis that needs model calls will be charged again. Results and cost logs are in results/.
- httpcore2 (openai>=3) uses truststore on macOS, which can hang during concurrent TLS handshakes: set SSL_CERT_FILE to
  certifi's bundle. mem0/TeleMem route every LLM call to OpenRouter whenever OPENROUTER_API_KEY is set.
- Quote paths in shell scripts (the project path contains a space). OpenRouter key lives in the project .env
  (git is not used here; keep .env private).

## Phase 13 notes (latency, standard metrics)
- Latency runs: `P13_LATENCY=1 P13_QIDS=<json list>` for scripts/p12_*.py (outputs in results/p13/latency/<bench>/),
  `--out` for p11_baselines.py, `--out/--home_root` for p11_mirix.py and `--ctx/--out/--nojudge` for p11_read_ctx.py;
  run_mm2 records t_select_s / t_read_s and a warm flag. Compose with scripts/p13_latency.py (medians; a request
  already issued by another condition is served from the local cache in 0 s and takes the fresh latency).
- Retrieval-only re-runs of V-Mem / MMA: `P13_RETRIEVAL_ONLY=1 ... --readers none` (results/p13/retrieval/).
- Standard metrics: scripts/p13_metrics.py (no API calls); tables: scripts/p13_table.py [latency]; figures:
  scripts/make_figures_p13.py main and make_figures_p13_app.py (paper_acl/figures/*.pdf).
- The reinstalled .venv_bl needs `tantivy` for SimpleMem's keyword index (otherwise it silently skips keyword search).
- Install with `pip install --uploaded-prior-to <date>` (pip >= 26) to respect the 7-day rule.

## Phase 14 notes (story lock, review responses)
- Strong widened path = plain RAG@40: `STRONG_DEFER = False` (scripts/p11_analyze.py) and `DEFER = False`
  (scripts/p10_unified.py). Do not reintroduce the deferral without giving RAG@40 the same fallback.
- Re-runs with regenerated captions live in results/p14/runs/<bench>/ (v2); the v1 per-question files in results/mm2
  stay the source of Table 1 (main table). Visibility estimates for v2: results/p14/vhat_<bench>.json (scripts/p14_vhat.py).
- Analyses (no API calls): scripts/p14_gate.py [--v2] [--strong] -> results/p14/gate*.json; scripts/p14_tables.py
  [gate] -> tab_ablation.tex / tab_gate.tex; scripts/p14_ids_v2.py + scripts/p14_read_ids.py (reading given round
  lists, resumable) for the ranking/cosine controls.
- run_mm2 --toc_mode text|cap (index content ablation); select_eval output names now carry the benchmark prefix for
  non-MemEye benchmarks (results/mm2/select_eval_visualmem_*.json).
- Full-context reading with gemini-3.8-flash on VisualMem costs ~$107 per run (217k tokens per question, implicit
  caching ~7%): never re-run it without an explicit cache or a budget check.

## Phase 15 notes
- The user will confirm when the project is finished: until then do NOT delete virtual environments, caches, or
  baseline stores (.venv_bl was removed after phase 13 and is not reinstalled; .venv_mirix was reinstalled in phase 15).
- MIRIX with hangs: scripts/p15_mirix_driver.py runs scripts/p11_mirix.py --direct per history and restarts it after a
  hung add (resumes from <home>/ADDED; a session that hangs twice is skipped, see <home>/HUNG).
- Table 10 (tab_std): answer metrics for all methods, evidence metrics only for methods that return raw rounds.
- Table style (phase 15): colours hME/hML/hVM (benchmark headers), hHead (generic headers), hOurs via \oursrow,
  dA-dD (drop shades) defined in paper_acl/main.tex; captions go BELOW tables with a bold lead sentence.
  xcolor needs the table option before acl.sty loads it: \PassOptionsToPackage{table}{xcolor} at the top of main.tex.
- Figure 1 = paper_acl/sections/fig_overview.tex (TikZ, 16 cm wide); test it standalone in /tmp before compiling.

## Phase 16 notes (user-polished paper)
- The user's latest paper is `paper_acl/main_polished.tex` (+ `sections_polished/`, `figures_polished/`); edit and build
  that one. Its Figure 1 (RAG vs full context, RAG@30 for multimodal) comes from `paper_acl/figure_sources/rag_motivation/`.
- `mm2.memlens_judge` returns score 0 without an error when the judge text lacks `"answer_score"`, and
  `scripts/p14_read_ids.py` does not store judge rationales: check cached judge outputs (gem._gen_cache) before trusting a
  MemLens drop. The RAG@30 MemLens run was checked this way (all 150 judgments well-formed).
- Two strong-reader MemEye RAG@40 runs exist: 0.741 (phase 2; Tables 5 and 8, Fig. 3a, Fig. 7, judge check; its
  per-question file was overwritten) and 0.749 (phase 6; Table 1 and later tables, current file).
- Figure 1 (RAG vs full context, gemini-3.8-flash): edit paper_acl/figure_sources/rag_motivation/make_figure.py, run it
  with .venv, then copy rag_motivation.pdf into paper_acl/figures_polished/ (the old diamond design is make_figure_v1_diamonds.py).
- Intro structure fixed with the user (2026-11-16): memory -> RAG strong on text (Fig. 1) -> not on multimodal (Fig. 1,
  k=40 caveat for MemLens) -> main reason (evidence sets, low-similarity images) -> VGS -> three contributions.
- Phase 16: image-anchored index = `--index iac [--iac_kw 12 --iac_b 20]` in run_mm2 / select_eval (mm2.INDEX_MODE).
  select_eval output names do not encode the caption version: back up results/mm2/select_eval_*.json before re-running
  a condition (phase-16 outputs live in results/p16/sel/). The OpenRouter key expired on 2026-11-19 (401): MemLens
  judging needs a new key from the user.
- Phase 16 final: the paper's only VGS is the compact-index VGS (strong widened path = QSS U cosine top-40, `tocu:40:40`).
  Regenerate its numbers with `scripts/p16_paper.py stats|figs|tables|ablation|extras` (no API calls); inputs in
  results/p16/final, v_hat in results/p16/vhat_compact_*.json, ranks in results/p16/ranks_*.json.

## Phase 16 final revision notes (2026-11-21)
- Paper = `paper_acl/main_polished.tex` (+ `sections_polished/`, `figures_polished/`). VGS in the paper: compact index,
  HISTORY-level visibility gate (H(h) >= 0.45, `scripts/p16_hvis.py`, `scripts/p16_compose.py` GATE="history"), strong widened
  path = QSS U cosine top-40. Regenerate in this order (no API calls): `scripts/p16_paper.py all`, `scripts/p16_paper.py
  ablation`, `scripts/p16_final.py all` (main table, Table 2, routing/swap/retrieval/std/systems tables, cost + baseline
  figures). Never run `p16_paper.py tables` or `extras` (old layouts; they overwrite tab_main / tab_gate / tab_std).
- Memory-system baselines (no MIRIX): `scripts/p16_run_baselines.sh <vmem|mma|telemem|simplemem|omni> <bench> <n>` with
  `scripts/p16_bl_env.sh` (v2 descriptions, MemLens 466, V-Mem top-20 rounds +/-1, MMA 80 entries, ledgers in
  results/p16/budget/<bench>/, outputs results/p16/baselines/<bench>/<system>.jsonl). Do not edit the runner while it runs.
- Baseline pitfalls found in phase 16: Omni-SimpleMem appends to an existing run dir and silently skips text units when its
  embedder fails (grep its logs for "Text embedding empty"); SimpleMem loads a new 0.6B embedder per history (now shared;
  run it in threads, one process); TeleMem may spin at exit (os._exit added); kill ALL pool workers before a restart.
- MemEye reader prompt no longer contains the example phrase; a one-sentence prompt edit changed 40-58% of the MemEye
  answers, so MemEye differences below ~2 points are run-to-run noise. MemEye inputs of the paper: results/p16/final2/memeye.
- The machine has 24 GB RAM: at most one SimpleMem process plus light jobs at a time.

## Phase 17 notes (fourth benchmark, restructured paper; 2026-11-21)
- Fourth benchmark = CamRoll-yfcc20 (`src/data_camroll.py`, bench `camroll`, data in data/CamRoll + data/camroll_1024,
  exact-letter scoring). The Hugging Face hub client stalls on its large zips here: fetch with curl (resolve/main URLs).
  Chosen before any run; SMMBench / ATM-Bench / DMV-Bench were examined and not run (RESEARCH_LOG P17-D8, D9).
- Regenerate the paper (no API calls), in this order: `scripts/p17_analysis.py all`, `scripts/p17_final.py all` (the latter
  also writes the four-benchmark ablation tables; `scripts/p16_paper.py ablation` standalone writes three columns: do not use). Do not run `scripts/p16_final.py all/main` or `scripts/p16_paper.py all/figs/tables` any more:
  they write the three-benchmark layouts over tab_main / figures.
- Never edit a shell script while one of its instances is running (bash re-reads by byte offset); add a new script.
- `mm2.llm` gives OpenRouter reasoning models (gpt-5*, *thinking*) 8,192 output tokens; with 2,048 GPT-5-mini returned no
  JSON for 22% of selections.
- Baselines on CamRoll: SimpleMem / Omni-SimpleMem build one camera roll per thread; use as many threads as rolls.
- Phase 18: Table 1 has no re-rank row (appendix `tab_rerank.tex`) and carries relative-gain subscripts on the VGS row;
  Fig. 1 uses a 60-100 axis (`make_figure.py`; full-axis version `make_figure_v2_fullaxis.py`). An index-free strong reader
  was tested and rejected by a pre-stated rule (MemEye -3.5 points; RESEARCH_LOG P18-D1..D8); runs in results/p18/.
