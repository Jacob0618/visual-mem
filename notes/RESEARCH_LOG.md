# Research log (chronological; decisions, failed attempts, deviations)

## 2026-10-26 — D0: environment & model verification
- Workspace was empty. Reusing (read-only) benchmark files already downloaded in `~/Desktop/jevmem/data`
  (LoCoMo, LongMemEval-S/M). Key loaded at runtime from env or `~/Desktop/jevmem/.env` (never printed/copied).
- `models.list()` (results/models_list.json): `gemini-3.8-flash` exists (1M in / 65k out). There is **no**
  `gemini-embedding-002`; embedding models are `gemini-embedding-001` (text-only, 2k tok) and
  `gemini-embedding-2` (GA; "multimodal content", 8k tok). "Gemini002" is interpreted as `gemini-embedding-2`.
- Probes (results/probe_models*.json):
  - 3.8-flash: thinking_level `minimal` -> 400 error; `low` works. Per-part media resolution works:
    LOW 260-266, MEDIUM 532-540, HIGH 1064-1080 image tokens for a 512-900px image.
  - embedding-2: passing a *list of strings* returns ONE aggregated embedding (documented: multiple parts are
    aggregated). Must wrap each item in its own `Content`. `task_type` is silently ignored (cos=1.000 vs none);
    tasks must be given as text prefixes (`task: question answering | query: ...`, docs `title: none | text: ...`).
  - Cross-modal toy: text query -> matching image 0.39-0.45 vs text-text 0.78 (modality gap in absolute scores).
- Prices (ai.google.dev pricing page, 2026-10-26): 3.8-flash $0.75/M in, $0.075/M cached, $3.75/M out;
  embedding-2 $0.20/M text, $0.45/M image ($0.00012/img).
- Throughput probe: 48 calls / 6.7 s at 24 workers, no 429s.

## D1: literature (3 parallel agents + direct verification of key abstracts)
Verified directly from arXiv abstract/HTML pages:
- 2610.30755 (Raw-RAG, Oct 2026): dense retrieval + rerank over raw history ranks 1st in 6/7 settings
  (LoCoMo, LongMemEval, MEME, MemoryAgentBench); "complete access to transformed memory stores leaves an accuracy gap".
- 2610.24671 (Zhou et al., Oct 2026): raw-history RAG vs 6 explicit memory systems on LME-S, BEAM-100K: only
  localized gains for particular question types / tight budgets.
- 2609.34227 (Sharma & Lall): selection over raw turns non-inferior to extraction; cosine-only weaker at k=3.
- 2605.12493 LongMemEval-V2 (agent trajectories + screenshots, 25M/115M tokens): RAG 42.8-51.0% vs coding agent
  69.9-74.9% (but reader is Qwen3.5-9B and RAG controller is 9B while coding agent uses GPT-5.4-mini xhigh).
- 2605.15128 MemEye (visual-centric): SRAG(V) (MiniLM + SigLIP2, top-10 rounds, native images) is the best of 13
  methods (open LLM-judge 0.494); captions competitive at scene/region level, gap at instance/pixel level; task-aware
  captions shrink gap (+0.194 -> +0.041 high-X); Y3 (evolving state): Latest-clue recall@10 0.53, stale-dominance 0.77.
- 2601.03515 Mem-Gallery: simple multimodal RAG (MuRAG) best overall F1 0.69; full-context MM weak (small backbones).
User's own prior project (jevmem, text only): cos top-100 ≈ FULL on LoCoMo (91.2 vs 91.8) and LME-S (92.8 vs 90.8)
with 3.8-flash; on LME-M (100 q) E100c drops to 80 (-12 vs S) although gold recall@100 = 97.9%.

=> "Simple retrieval is strong" replicates broadly, including the two multimodal benchmarks. But every multimodal
benchmark result so far used WEAK components (MiniLM/SigLIP-base embedders, <=gpt-5.4-mini / 7B readers, 128k ctx).
Open: does it still hold with a strong unified multimodal embedder + strong 1M-context VLM reader, and what is left?

## D2: direction choice (pre-pilot)
Main thread = multimodal memory, because (i) the user's prior work already covers text extensively, (ii) images are
where the field still *defaults to write-time rewriting* (captioning), which is the exact opposite of the text lesson,
(iii) image tokens make full-context break ~1000x earlier (≈1k tokens/image), so the "retrieval vs read-everything"
question has a real budget dimension (resolution!) absent in text.
Secondary thread = compact independent verification of the text observation and its conditions (k, category, scale).

Candidate questions (to be pruned by pilots):
- Q-A: With strong components, is retrieval ≈ full context on MemEye / Mem-Gallery? Where do residual errors live
  (retrieval miss vs perception vs state selection vs label noise)?
- Q-B: Write-time captioning (lossy rewriting) vs raw pixels at read time; by granularity; at what token cost?
- Q-C: Breadth vs acuity under a fixed token budget: everything at LOW res vs top-k at HIGH res vs captions + few raw.
- Q-D: Retrieval key for image memories: caption text vs image embedding vs interleaved; modality gap.
- Q-E: Evolving state (stale vs latest).

Data discipline: MemEye tasks split 4 dev / 4 test by a deterministic balance criterion fixed before any results;
all method/hyper-parameter choices on dev only; MemEye-test and Mem-Gallery (different authors/pipeline) are held-out.

## D3: infrastructure incidents (kept for reproducibility)
- First pilot run hung: google-genai client had no HTTP timeout -> added 240 s timeout + retry on timeouts.
- Throughput collapsed (~0.1 q/s): every request re-uploaded multi-MB images inline. Fix: canonical API copies of all
  images (scripts/prep_images.py: downscale only if max side > 2048 px; 120/1985 images resized; 805 -> 408 MB) and
  all generation requests reference images via the Files API (uploaded once). Same image bytes are used by every
  condition (captioning, embedding, reading). Pre-fix results were discarded (results/mm/_smoke only).
- Answer prompt bug caught in smoke test: a rule "answer image questions with the bracketed image ID" (needed for
  Mem-Gallery VS questions) made the reader answer MemEye questions with IDs. Rule made benchmark-specific.

## D4: MemEye-dev retrieval diagnostics (n=206; results/mm/retrieval_memeye_dev.json)
gold-clue coverage@10: caption-dense text key 0.758 > caption-short 0.748 > interleaved text+image (one
aggregated emb-2 vector) 0.647 > text only 0.590 > image-only 0.210.
Modality gap: under the image-only key, a gold image round scores 0.549 on average vs 0.575 for an average
*irrelevant text* round -> in a mixed index, text rounds crowd out images. Within-modality z-normalisation (maxz)
recovers to 0.675, still below caption keys.

## D5: MemEye-dev answer pilot (n=206, reader/judge gemini-3.8-flash, MemEye 0-1 judge rubric)
fc|raw:low 0.823 (32k tok) | fc|raw:high 0.802 (86k) | oracle|raw:high 0.780 (4k) | fc|cap:dense 0.779 (65k)
oracle|cap:dense 0.737 | cap:dense:10|raw:high 0.649 | cap:dense:10|cap:dense 0.647 | both 0.634 |
inter:10|raw:high 0.580 | oracle|cap:short 0.358 | oracle|none 0.112.
Surprises (contradict my prior that retrieval would be competitive as in text):
- Full context with ALL images at LOW resolution is best; top-10 RAG loses ~17 pp. Opposite of the MemEye paper's
  ranking (SRAG(V) > FC(V)), which used gpt-5.4-mini/128k and MiniLM+SigLIP.
- Low resolution suffices in full context (0.823 vs 0.802 high, n.s. presumably).
- Dense captions are lossy (-2 to -4 pp, mostly X4 pixel-level) AND cost more read tokens than low-res pixels
  (≈800 tok/caption vs ≈266 tok/low-res image). Short captions destroy most visual information (0.358 oracle).
- FC > oracle: 16 vs 9 discordant questions; inspection shows annotated clue rounds often lack the contextual
  framing questions refer to ("Episode 2", "palace takeover arc", "first main-bench image"), so "gold clue
  recall" is necessary but not sufficient; isolated top-k rounds lose this context.
Next (still dev only): k-sweep, resolution sweep, all-low + top-k-high mix, fc with short captions / no visual.

## D6: text verification (independent pipeline; strict LongMemEval judge + lenient Mem0 judge)
LoCoMo 1,540 q: full 0.790 (lenient 0.894) | cos-100 0.781 (0.880), Δ −0.009 [−0.023, +0.006] | cos-50 0.767 |
cos-25 0.745 | oracle-evidence-turns 0.739 | cos-10 0.706. Same ordering under both judges; judge choice shifts
absolute accuracy by ~10 pp. 37% of full-context errors (lenient) fall on the 99 questions an independent audit
(dial481/locomo-audit) flags as score-corrupting; on unflagged questions full = 0.929, cos-100 = 0.920.
With implicit prefix caching full context costs $5.8 / 1k q vs $4.6 for cos-100 (LoCoMo asks many q per history).
LongMemEval-S stratified 119 q: full 0.950 | cos-50 0.941 (Δ −0.008) | oracle sessions 0.924 | cos-10 0.899 (Δ −0.050);
no caching possible (one haystack per q): full $85 / 1k q vs cos-50 $17. Only preference questions show a
retrieval deficit (0.71-0.77 vs 0.88, n=17).

## D7: MemEye-test (held-out, pre-registered) — results/mm/memeye_test
H1 confirmed: fc|raw:low − cap:dense:10 = +0.080 [+0.023, +0.139]; − cap:dense:40 = +0.047 [+0.014, +0.082]
   (smaller than dev; Card_Playlog has 30 rounds so k=40 = full history there).
H2 NOT confirmed (inconclusive): low − high = −0.008 [−0.050, +0.033]; lower bound below −0.03. Task-specific:
   Card_Playlog (UNO screenshots, counting small cards) high 0.854 vs low 0.771; Home low > high.
H3 confirmed (non-inferiority form): low − cap:dense = +0.036 [−0.018, +0.091]; tokens 21k vs 43k. BUT read $ was
   lower for cap:dense on this split because its prefix-cache hit rate was higher (87% vs 55%).
H4 half confirmed: cap:short ≪ cap:dense (0.459 vs 0.724). cap:dense ≤ raw:low FAILED in the oracle setting
   (0.724 vs 0.695; Card_Playlog X4 0.688 vs 0.500): a caption written from the high-res image beats re-reading a
   LOW-res copy when the detail is tiny.
H5 confirmed: cov@10 cap:dense − inter +0.094 [+0.048, +0.142], inter − img +0.289 [+0.220, +0.360];
   end-to-end cap:dense:10|raw:high 0.723 vs inter:10|raw:high 0.656.
H6 consistent with null: zoom − low = −0.003 [−0.050, +0.047] (CI wider than ±0.03; not an equivalence proof).
Test-retest (dev, seed 1 vs default): fc|raw:low 0.813 vs 0.823; cap:dense:40 0.709 vs 0.718 => decoding noise on
   condition means ≈ 0.01.

## D8: Mem-Gallery (held-out, different authors) — partial
fc|raw:low 0.917 | cap:dense:40|raw:low 0.912 (Δ −0.005) | cap:dense:10|raw:low 0.873 (Δ −0.044).
Retrieval cov@10: inter 0.877 ≈ text-only 0.876 ≈ cap:dense 0.866 ≫ img 0.637 -> caption keys do NOT help here:
Mem-Gallery's dialogue text already describes the images (MemEye's own "visual irreplaceability" analysis ranks it
low). So the RAG-vs-full gap depends on visual necessity / cross-image questions, not on modality per se.

## D9: scale stress test setup (exploratory)
MemEye-dev 100-question subset (results/memeye_dev_scale100.json, seed 11). Caveat found BEFORE scale results:
by chance this subset is much easier for cap:dense:40 RAG than its complement (0.820 vs 0.623; fc: 0.835 vs 0.811),
so at dx0 the FC−RAG40 gap on the subset is only +0.015 (vs +0.104 on all 206). Scale results are therefore
interpreted as paired changes across levels on the same 100 questions, not as absolute gaps.

## D10: failed attempt — LLM query decomposition for multi-evidence retrieval (dev, retrieval-only, not pursued)
Idea: the RAG deficit concentrates in questions with >=4 evidence rounds (pooled MemEye, k=10: FC−RAG +0.27 vs
+0.04 for 1 evidence round), so have the LLM write 1-5 sub-queries (one per needed item), union their top-5 with the
question's top-10, cap at K=20. Result at matched K=20 (MemEye-dev, 206 q): coverage 0.801 vs 0.850 for plain cosine
top-20; full-set recall 0.675 vs 0.767; worse in every evidence-count bucket (>=4: 0.441 vs 0.576). Dropped before
any answer generation. Plausible reason: MemEye questions already name the referenced items precisely; paraphrased
sub-queries are noisier than the original wording and displace useful ranks 11-20.
Other retrieval-side variants tried on dev, all ≈ "bigger k at the same token budget": session expansion
(sess+cap:dense:10 0.718 @9.6k tok), ±2 neighbours (0.715 @8.0k), top-40 (0.718 @9.8k), interleaved key top-40 (0.744).

## D11: remaining held-out + exploratory results (all in results/TABLES.md)
- Mem-Gallery held-out (903 q): FC-low 0.917; RAG@10 −0.044 [−0.059, −0.029]; RAG@40 −0.005 (n.s.); FC-high −0.007
  (H2 non-inferior); FC-dense-caption −0.012 [−0.024, −0.001] (H3 confirmed, CI>0); oracle short 0.931 ≥ dense
  0.919 (H4a FAILS: no visual necessity); text-only FC (no images, no captions) 0.892 vs 0.909 (half subset);
  caption key vs interleaved key end-to-end +0.001 (H5e fails), cov@10 cap:dense − inter −0.011 (H5 fails);
  zoom −0.002 (H6 null holds).
- Exploratory, all three sets: SHORT-caption keys beat interleaved multimodal keys on cov@10 (+0.101, +0.105,
  +0.029; all CIs > 0); image-only keys are worst everywhere (modality gap; on MemEye a relevant image round
  scores below an average irrelevant text round).
- Scale (MemEye-dev 100 q): FC-low 0.835 (32k tok) -> 0.825 (+250 img, 212k tok) -> 0.848 (+1000 img, 722k tok);
  Δ(+1000) = +0.013 [−0.037, +0.062]. RAG@40 0.820 -> 0.800 -> 0.762, Δ(+1000) = −0.058 [−0.102, −0.020].
  RAG@10 flat 0.69. Implicit prefix caching held at 722k tokens (≈100% after warm-up): $92 / 1k q.
- Second judge (gemma-4-31b-it, 60 dev q x 8 conditions): exact agreement 0.90-0.98, same ordering.
- Weak reader (gemini-3.1-flash-lite, MemEye-dev): FC-low 0.689, oracle-low 0.720, RAG@10 0.621, RAG@40 0.627:
  the weak reader gains less from full context (FC < oracle) and nothing from k 10 -> 40.

## D12: correction — the apparent RAG@40 degradation at scale is decoding noise
At +1000 distractor images, 62/100 questions had an IDENTICAL top-40 evidence set to +0 (no distractor retrieved),
yet their mean score moved −0.048 [−0.105, −0.008]. Re-running the +0 prompts with seed 1 on the same 62 questions
moves the mean −0.044 [−0.109, +0.004] with the same 6.5% score-flip rate. => The RAG@40 "drop" (−0.058) is not
attributable to retrieval; clue coverage was unchanged (0.956 -> 0.950). Only the FC-low result (no degradation up
to 722k tokens) is reported as a finding. Lesson: single-run differences of ~5 pp on ~60-100 questions can be pure
decoding noise; key held-out contrasts get a second seed (MemEye-test, below).

## D13: decoding-noise replication + reader strength (final runs)
- MemEye-test, second seed for 5 key conditions (results/mm/memeye_test_seedavg.md): seed-averaged
  H1a +0.098 [+0.047, +0.152], H1b +0.036 [+0.008, +0.064], H2 −0.002 [−0.034, +0.028], H3 +0.021 [−0.028, +0.069].
  FC-dense-caption moved 0.779 -> 0.818 between seeds, so H3's accuracy difference on MemEye is within decoding noise;
  the robust statement is "not more accurate, ~2x the read tokens".
- Weak reader gemini-3.1-flash-lite (MemEye): dev FC 0.689 / RAG@10 0.621 / RAG@40 0.627 / oracle 0.720;
  test FC 0.576 / RAG@10 0.576 / RAG@40 0.592 / oracle 0.662 (FC − oracle −0.086 [−0.165, −0.008]).
  => "read everything" wins only with a strong long-context reader; explains the opposite ranking in the MemEye paper.
- Total API spend: $195.64 (3.8-flash 185.80, embedding-2 4.65, 3.1-flash-lite 5.19, gemma 0).

# Phase 2 (user feedback 2026-10-26): capability-improving methods for MULTIMODAL memory
Directions from the user: set-level recall, reader-driven zoom, plus anything else promising; ignore abstention.
MemEye used as ONE pool (371 q; the dev/test split only guarded parameter selection). Main reader switched to the
cheaper gemini-2.5-flash-lite, with gemini-3.8-flash controls to separate method limits from reader limits.
Success criterion: beat RAG at matched cost (not necessarily beat full context); any reliable gain counts.

## P2-D0: gemini-2.5-flash-lite facts (probe, 2026-10-26)
$0.10/M in, $0.01/M cached, $0.40/M out. Thinking via thinking_budget (thinking_level -> 400). Media resolution is
PER REQUEST only: per-part settings are ignored (usage came back empty). Image tokens for a 1352x852 screenshot:
LOW 66, default/MEDIUM 258, HIGH 1806. => 2.5 "medium" (258) is the analogue of 3.8 "low" (266); reader-driven zoom
on 2.5 must be implemented with CROPS (a crop at the same per-image budget carries more pixels of the region).

## P2 plan (written before running any phase-2 condition)
Baselines (2.5-flash-lite, MemEye-371): FC at low/medium/high, RAG@10/20/40 (cap:dense key), oracle.
Candidate methods:
 M1 set expansion without LLM: top-k + caption-space nearest neighbours of the hits; adaptive k by score gap.
 M2 LLM index scan ("table of contents"): one cheap line per round (date, text snippet, short caption); the reader
    selects the rounds to load (incl. every member of a referenced set); then reads them as raw images, optionally
    keeping the whole TOC in context for global framing (episode labels, order). The TOC is a write-time rewrite used
    only for navigation, consistent with phase-1 finding "captions as keys, pixels as content".
 M3 reader-driven zoom: pass 1 sees the context at low/medium res and requests up to 3 zooms (image id + optional
    box); pass 2 receives the crops and answers.
 M4 read-time, question-aware extraction (map-reduce): per candidate image, a cheap call extracts what matters for
    THIS question at high res; the final reader works from the notes (+ TOC). The multimodal analogue of
    "don't rewrite at write time, read raw with the question in mind".
Fairness: every method is compared with RAG/FC at matched input tokens and $ (all calls counted), same reader, same
judge (gemini-3.8-flash, MemEye rubric). Robustness: split-half (old dev/test halves) consistency, 3.8-flash
control, second decoding seed for headline contrasts.

## P2-D1: gemini-2.5-flash-lite baselines (MemEye-371) + a platform constraint
FC medium (258 tok/img) 0.495 | FC low (66 tok/img) 0.425 | oracle medium 0.539 | RAG@10/20/40 (cap:dense key,
medium) 0.446 / 0.471 / 0.488. Weak reader: focused evidence > full context > RAG.
PLATFORM: on 2.5-flash-lite MEDIA_RESOLUTION_HIGH only applies when a request has exactly ONE image (probe: 1 img
-> 1806 tok; 2, 3, 6, 12 imgs -> 258 tok each). "fc|raw:high" / "oracle|raw:high" for this reader are therefore
NOT resolution manipulations (FC-high reproduced FC-medium exactly) and are excluded from analysis. With this reader,
high-res perception is only reachable through single-image calls (map step of map-reduce) or crops (zoom).

## P2-D2: run 1 (2.5-flash-lite, MemEye-371) and selection-only set-recall study
Answering (score / mean rounds loaded / $ per 1k q):  FC-medium 0.495 | oracle 0.539 | RAG@10 0.446 | RAG@20 0.471 |
RAG@40 0.488 | TOC(2.5 selects) 0.470 (7.3 rounds) | TOC + TOC-as-context 0.491 | TOC ∪ RAG@10 0.466 |
graph expansion gx:5:3:20 0.463 | map-reduce RAG@10 0.489 (+0.043 vs RAG@10, $1.58) | map-reduce RAG@20 0.478 |
map-reduce TOC 0.477. Per-task effects are heterogeneous and mostly within noise.
Failures: zoom run crashed (a pass-1 JSON call looped to 65,526 output tokens and a later persistent 500).
Fixed: max_output_tokens=2048 on all phase-2 calls, judge input truncated to 2,000 chars, per-question exception
capture (errors counted in the summary). The weak reader never asked for a zoom with the first prompt.
Strong-writer captions do not lift the weak reader in FC (fc|cap:dense 0.474 vs raw 0.495) but help a bit with
focused evidence (oracle|cap:dense 0.571 vs raw 0.539).

SELECTION-ONLY (full-set recall = all gold clue rounds loaded; RAG at the SAME per-question size in brackets):
  TOC global, 2.5-flash-lite selects:  9.8 rounds, full 0.712 [0.598]; >=4-evidence q 0.512 [0.244]; $0.37/1k
  TOC global, 3.8-flash selects:       7.3 rounds, full 0.903 [0.539]; >=4-evidence q 0.872 [0.151]; $6.43/1k
  chunked scan (30/chunk), 2.5:       18.4 rounds, full 0.765 [0.782]; >=4: 0.535 [0.512]; $1.28/1k
  chunked scan (30/chunk), 3.8:       10.3 rounds, full 0.881 [0.647]; >=4: 0.779 [0.326]; $9.54/1k
  cosine RAG needs k=40 for full 0.922.
=> Set-level recall is solved far better by a STRONG model reading a compact table of contents than by any
embedding variant; weak selectors are no better than cosine at matched size; chunking does not help.

## P2-D3: run 2/3 results (MemEye-371)
2.5-flash-lite reader: toc38:40 (3.8 selects, 2.5 reads 7.3 rounds) 0.541 (+0.070 [+0.027,+0.113] vs RAG@20; +0.046 vs FC);
  + forced zoom 0.559; oracle + forced zoom 0.592 (vs 0.539); FC + forced zoom 0.507 (vs 0.495); RAG@20 + zoom 0.474/0.481.
  toc38 + TOC-as-context 0.522 (TOC context hurts the weak reader); map-reduce on toc38 0.519, on oracle 0.528 (no gain).
3.8-flash reader: toc38:40|read:low 0.761, toc38:40|read+toc:low 0.796 (FC-low 0.820, RAG@40 0.740, RAG@10 0.689).
Mid-tier selectors under-select: 3.1-flash-lite 4.3 rounds / full 0.660; 3.5-flash-lite 3.9 / 0.685 (3.8: 7.3 / 0.903).
Zoom protocol flaw found: with 3.8, the two-pass JSON protocol LOWERS accuracy even when no zoom is requested
(86 non-zoomed q: 0.794 vs plain FC 0.860), i.e. the "answer-in-JSON + decide-to-zoom" format costs accuracy.
Fix: pass 1 only returns zoom requests; pass 2 is the plain answer prompt (identical to the baseline when no zoom is
requested, so zoom becomes strictly additive). Also add a "blind zoom" control (fixed tiles, no reader choice).
Cost note: the 3.8 selector's recorded cost ($6.4/1k) came from calls with only 39% prefix-cache hits; with a warm
cache the TOC prefix (static per user) would cost ≈ $2.5/1k.

## P2-D4: set-level recall at scale + zoom controls
Selection-only: cosine prefilter + 3.8 TOC selection — dx0 ptoc38:100:40: 7.7 rounds, full 0.919 (full-TOC 0.903);
dx1000 (≈4,000 rounds/history, MemEye + 1,000 distractor images) ptoc38:200:40: 10.8 rounds, full 0.914 vs cosine at
the same size 0.642; >=4-evidence questions 0.802 vs 0.314. Set recall of strong TOC selection survives 10x growth.
Zoom with 2.5 reader on oracle evidence (new strictly-additive protocol): read 0.539 | voluntary zoom 0.552 |
forced zoom 0.551 | BLIND zoom control (3 fixed tiles of the most query-similar image) 0.561 | old JSON protocol
forced zoom 0.592. => for this reader the benefit comes from extra pixels, not from the reader's choice of where
to look (new protocol); the old-protocol number needs checking (decode noise vs "answer-first then look").

## P2-D5: zoom with the strong reader is net negative; TOC ablation; Mem-Gallery / scale with the weak reader
- 3.8 FC-low + voluntary zoom (strictly-additive protocol): 0.790 vs FC-low 0.820; Δ −0.029 [−0.055, −0.003]. It zoomed
  on 246/371 questions; on those Δ = −0.047 [−0.083, −0.011]; on non-zoomed Δ = +0.006 (= decoding noise; seed-to-seed
  FC Δ −0.001 [−0.022, +0.020]). Only UNO screenshots gain (0.771 -> 0.812). Hypothesis: appended crops after the
  question are over-weighted / displace the global view. Next: in-place upgrade (zoomed image re-shown at HIGH in its
  original position + crop) = "zoom3". Killed the remaining old-style 3.8 FC zoom runs (zoom2f, blind) to save cost.
- TOC line ablation (3.8 selector): text-only TOC 12.3 rounds / full 0.906; captions-only 10.2 / 0.822; full 7.3 / 0.903.
  Dialogue text carries most of the set signal in MemEye; captions make selection more precise (fewer rounds).
- Mem-Gallery held-out, 2.5 reader (903 q): FC 0.817, RAG@10 0.814, RAG@40 0.839, toc38:40 0.852 (5.6 rounds).
- dx1000, 2.5 reader (371 q): RAG@20 0.452, RAG@40 0.471, ptoc38:200:40 0.538 (≈ its dx0 value 0.541).

## P2-D6: more negative results + split-half robustness
- Write-time group index (3.8 writes named groups once per history, $0.005-0.05 per history; 2.5 picks groups at query
  time, + 5 cosine hits as fallback): grp:40:5 0.466 (full 0.717), grp:40:0 0.426 (full 0.636, 21.9 rounds) — worse
  than RAG@20 (0.471 / 0.803). Groups came out session/topic-level ("Coca-Cola Reference Deck", "Episode 1 Frames"),
  i.e. query-agnostic organisation ≈ session expansion; question-specific sets cut across them.
- 3.8 FC + in-place zoom (zoom3): 0.803 vs 0.819, Δ −0.017 [−0.043, +0.009]; less harmful than appended crops
  (−0.029) but no gain, also not on UNO (0.750 vs 0.771). Reader-driven zoom = negative result for the strong reader.
- Split-half (old dev/test task halves) for headline phase-2 contrasts: 2.5 toc38 − RAG@20 +0.056 / +0.088,
  − RAG@40 +0.050 / +0.058, − FC +0.035 / +0.059; 3.8 toc38+TOC − RAG@40 +0.068 / +0.039 (both CIs > 0),
  − FC −0.036 / −0.008. Map-reduce at k=10 is NOT robust (+0.000 / +0.097). Zoom2f +0.019 / +0.012 (n.s.).

## P2-D7: decoding determinism + hint mode
- gemini-2.5-flash-lite at T=0, thinking off is deterministic: a seed-1 replicate (370 new API calls) reproduced all
  371 predictions byte-for-byte. => 2.5-reader contrasts carry no decoding noise (only question sampling); the
  3.8-flash reader (thinking on) does vary between seeds (±0.01-0.04 on condition means).
- "Hint" reading (full history + the selector's picks listed before the question): 2.5 0.485 vs FC 0.495 and
  toc38-filter 0.541 — for the weak reader, FILTERING to the selected set is what helps, not pointing at it.

## P2-D8: final phase-2 bookkeeping
- 3.8 reader + hint (full history + selector picks listed): 0.792 vs FC 0.819 — hints do not help either reader.
- Cheap selector WITH thinking (2.5-flash-lite, budget 2048): 4.6 rounds, full 0.636 — thinking does not turn a weak
  model into a set selector (no thinking: 9.8 rounds / 0.712).
- Warm-cache cost estimate (results/mm2/cost_warmcache.md): the 3.8 TOC-selection call costs $9.47/1k uncached vs
  $1.83/1k with the static per-user TOC cached; 3.8 pipeline "toc38 select + read + TOC context" 0.796 at ≈$4.8/1k
  (measured $15.7 with poor cache hits) vs RAG@40 0.741 at $10.0 (cannot be cached) vs FC-low 0.819 at $9.0.
- All phase-2 tables: results/TABLES_P2.md (scripts/make_tables_p2.py, no API calls).

# Phase 3 (user feedback): toward a new algorithm / mechanism / insight + small-model components; then a paper
## P3-D1: anatomy of set-level recall (results/mm2/set_analysis.txt)
- Full-set recall by gold set size (1 / 2-3 / 4-5 / >=6 rounds): cosine@10 0.916/0.614/0.316/0.000; cosine@40
  0.975/0.958/0.842/0.300; weak selector (2.5) 0.908/0.675/0.526/0.400; strong selector (3.8) 0.983/0.861/0.868/0.900.
- Adaptive budget: Spearman(selected size, gold size) strong 0.571, weak 0.382. Strong opens 3.8 rounds for
  single-evidence questions and 26.4 for >=6-round sets.
- Weak-selector failures (107 q): 71 partial sets, 36 total misses; only 14% are pure under-selection (weak opens
  11.1 rounds vs gold 4.3) -> it fails by MIS-selection. Qualitative: questions name evidence by narrative aliases that
  appear nowhere in the dialogue or captions ("the D9 summary", "the month-wrap comparison trend", "the provisional
  busy-day trend", "the night route at the red-awning corner"); the strong selector resolves them from the history's
  structure (day/session order, the session's opening line).
## P3-D2: attempts to give the cheap model this capability (selection-only, MemEye-371; full-set recall / rounds)
  toc (2.5) 0.712/9.8 | ptoc:40 0.704/9.7 | ptoc:60 0.720/10.0 | count-first prompt 0.663/8.1 |
  predicate-guided set completion (decompose into membership predicates + listwise membership checks) 0.617/8.7 |
  narrative coordinates + session headers 0.690/10.0 | write-time aliases by 3.8 in the index 0.736/9.3 |
  5-sample self-consistency: union 0.836/15.2, >=2 votes 0.749/10.7, >=3 votes 0.698/8.3; >=2 votes + aliases 0.760/10.2.
  Strong selector: 0.903/7.3; with coordinates 0.911; with aliases 0.898.
- Write-time aliases (3.8, $0.21 for all of MemEye) barely move cosine: full-set recall @10/20/40 0.633/0.803/0.922 ->
  0.642/0.811/0.933. => the capability is query-time reasoning, not missing index content.
## P3-D3: end-to-end lightweight components (2.5-flash-lite reader, MemEye-371)
  fully cheap: single-sample cheap selection 0.497; 5-sample >=2-vote selection 0.516 (RAG@20 0.471, RAG@40 0.488,
  FC 0.495); union (>=1 vote) 0.480 (precision matters for weak readers). Strong-writer captions next to the selected
  images: toc38 0.526 vs 0.541 (no gain); RAG@10 0.471 vs 0.446 (+0.025); RAG@20 0.474 vs 0.471.
  Reader thinking budget 1024 lifts everything: RAG@20 0.471->0.537, FC 0.495->0.571, toc38 0.541->0.597.
## P3-D4: more transfer attempts + resolution write-back (a learning mechanism)
- In-context distillation (3 leave-one-task-out demonstrations of teacher selections): cheap selector gets WORSE
  (toc 0.712 -> 0.639; vote 0.749 -> 0.706).
- Thinking for the cheap SELECTOR is harmful (it under-selects); a separate --sel_thinking flag now keeps --thinking
  for the reader only (an earlier confounded run of scu/toc with thinking was overwritten by the corrected run).
- Resolution write-back: earlier questions' evidence sets (resolved by the strong selector) are appended to the
  index as "resolved earlier questions"; paraphrase guard drops past questions with embedding cosine > 0.9.
  Selection-only (cheap selector): none 0.712 | self write-back (cheap resolutions) 0.725 | strong write-back for 30%
  of past questions 0.741 | strong write-back online (random order, ~22 past q) 0.752 | leave-one-out 0.776.
  LLM-free reuse q2q (evidence of the 3 most similar resolved past questions + cosine top-5): online 0.836 at 19.1
  rounds (cosine same size 0.787); with self resolutions 0.825 at 24.6 (cosine 0.846) — no gain.
- End-to-end (2.5 reader): thinking off — toc 0.497 -> +write-back 0.514; vote 0.516 -> 0.518; q2q 0.501 (RAG@20 0.471);
  thinking 1024 — toc 0.542 -> +write-back 0.578; vote 0.543 -> 0.575; q2q 0.547 (RAG@40 0.552, FC 0.571, QSS 0.597).
- Lightweight selection with a cosine-prefiltered pool is worse than over the full index (pscu100 0.478 vs scu 0.516);
  on Mem-Gallery pscu100 0.839 = RAG@40; at +1,000 images pscu200 0.493 vs RAG@40 0.471.
- Paper: paper/ (LaTeX, compiled with tools/tectonic 0.17.0, sha256-verified local binary).

## P3-D5: corrections from the internal review (3 reviewer subagents)
- COST CORRECTION: the v1 warm-cache estimate (cost_warmcache.md) under-counted reading tokens. Recomputed per question
  from logged token counts (results/mm2/cost_warmcache_v2.md): 3.8 QSS warm $5.80 / cold $13.32 (measured $10.27);
  3.8 QSS + index context warm $6.80 / cold $21.83 (measured $15.70; v1 said $4.77); 2.5 QSS warm $2.20 / cold $9.71.
  => "half the cost of top-40" was wrong; correct: ~two-thirds of RAG@40 ($10.00) with a warm index cache, ~2x without.
- Per-set-size FSR at +1,000 images (QSS prefilter 200): 0.983 / 0.922 / 0.868 / 0.300 for 1 / 2-3 / 4-5 / >=6 rounds
  (>=6 has n=10; prefilter likely cuts members). The claim "flat across set sizes at scale" holds only up to 5 rounds.
- Per-half CIs: 2.5 QSS - RAG@40 +0.050 [-0.013,+0.114] / +0.058 [-0.005,+0.120] (same sign, each half n.s.);
  3.8 QSS+index - RAG@40 +0.068 [+0.013,+0.124] / +0.039 [+0.006,+0.074]; 3.8 plain QSS - RAG@40 +0.044 / -0.009.
- Facts: MemEye histories 30-299 rounds, 30-89 images (paper said 87); Mem-Gallery half subset = 451 questions.
- RQ5 mislabel: 0.557 vs blind 0.523 on QSS sets is FORCED zoom; reader-chosen is 0.539.
- "tenfold" growth understated: mean rounds 126.5 -> 4,088.5 (~32x), FC tokens ~27k -> 713k (~26x).

## P3-D6: ranking control, second review round, final state
- Ranking control (scripts/rank_control.py; same strong model ranks the same cosine top-100 candidates, cut at the
  size QSS chose): FSR 0.860 vs QSS (P=100) 0.919 vs cosine 0.569. QSS - ranking +0.059 [+0.032,+0.089] overall,
  +0.008 / +0.048 / +0.151 [+0.070,+0.244] for 1 / 2-3 / >=4 rounds; ranking - cosine +0.291 [+0.240,+0.345];
  ranking with a fixed cut needs k=20 to reach 0.916. => joint judgment explains most of the gain over cosine; the set
  formulation adds recall on large sets and decides the budget.
- Internal review: round 1 (novelty, numbers audit, narrative) -> fixes listed in P3-D5 + reframing (principle:
  evidence sets are resolved at query time; QSS positioned in the index-navigation family with ReadAgent/MemWalker;
  write-back contrasted with MemPrompt/ExpeL/doc2query). Round 2 confirmed all 9 items fixed; residual nits fixed
  (bracket for consensus+write-back row: cosine 0.631 / >=4 0.256; variant labels; RQ3 wording; tab:keys ref).
- Remaining weaknesses named by the reviewers: single vendor incl. self-judging; design choices made on pooled MemEye;
  modest strong-reader effect without index context (+0.020, CI includes 0) and cache-dependent cost advantage.
- Paper: paper/main.pdf, 17 pages, compiles without overfull boxes or undefined references. Total spend $342.9.

# Phase 4 (two external reviews, 2026-10-27): what we accept and what we test
Accepted as correct and important: (1) evidence rests on one small benchmark whose 8 histories also shaped the method
-> held-out, visually necessary test set never used in development; question-level bootstrap understates scenario-level
uncertainty -> cluster (scenario) bootstrap; (2) end-to-end gains are smaller than retrieval gains and costs are
conditional -> same-cost strong baseline (model reranking with a DYNAMIC cut-off, graded relevance) and tighter claims
("finds complete evidence more efficiently", not "best accuracy-cost everywhere"); (3) single model family incl.
self-judging -> cross-family selectors and an independent judge; (4) "cannot be obtained at write time" over-reaches ->
restrict to the write-time methods tested. Not pursued now: confusable same-topic distractors (partly covered by the
128k MemLens haystacks, which add sessions from the same pool), human evaluation (not possible here).
## P4-D0: held-out benchmark = MemLens (arXiv:2605.14906, CC-BY-4.0, HF xiyuRenBill/MEMLENS)
789 questions (we drop the 90 abstention questions -> 699), one haystack per question, 32k/64k/128k/256k versions;
32k: 13.5 sessions, 12.7 images per question; authors' image ablation: removing evidence images drops frontier LVLMs
below 2% on the 80.4% of questions whose evidence includes images. Evidence structure: each answer session holds one
has_answer user turn (text) and a separate user turn with a needle image (e.g. the receipt with the amount).
Gold rounds := rounds (user+assistant pairs) of answer sessions that contain a has_answer turn or a needle image;
mean 5.9 per question; 353 questions span >=4 answer sessions. Never used for any design decision before this point.
Judge: official MemLens judge prompts (judge_prompts/, binary 0/1) with Qwen3-VL-235B-A22B (the official judge model
family) via OpenRouter; gemini-3.8-flash as a second judge on the same predictions.
Cross-family models via OpenRouter (user's key, $50 limit): Qwen3-235B-A22B-2507, GPT-5-mini (selectors/judges).
## P4-D1: scenario-level (cluster) uncertainty on MemEye (results/cluster_stats.md)
Two-stage bootstrap (scenarios, then questions) widens the intervals as the reviewers expected:
  2.5 QSS - RAG@40 +0.053: question CI [+0.009,+0.098] -> cluster CI [-0.014,+0.120]; 6/8 scenarios > 0; Wilcoxon p=0.078
  2.5(thinking) QSS - RAG@40 +0.045: cluster [-0.005,+0.095]; 6/8; p=0.031
  3.8 QSS+index - RAG@40 +0.055: cluster [+0.004,+0.126]; 6/8; p=0.031
  3.8 QSS+index - FC -0.024: cluster [-0.061,+0.022]; 1/8
  FSR QSS - cosine(same size) +0.364: cluster [+0.221,+0.551]; 8/8; p=0.008
  FSR QSS(P100) - ranking +0.059: cluster [+0.008,+0.111]; 5/8; p=0.062
=> the retrieval-level claims survive scenario-level resampling; the MemEye end-to-end gains of the lightweight reader
do not by themselves (CI includes 0), which makes the held-out MemLens evaluation (699 independent haystacks) decisive.
## P4-D2: PRE-REGISTRATION for the held-out MemLens evaluation (written before any MemLens condition was run)
Data: MemLens-32k, all 699 non-abstention questions; MemLens-128k: fixed random 200-question subset (seed 0).
Units/gold: rounds = user turn + following assistant turn; gold = rounds of answer sessions with a has_answer turn or
a needle image (5.9 per question). Secondary retrieval metric: session-level FSR (every answer session touched).
Frozen from MemEye (no tuning on MemLens): index line = round id + date + first 200 chars of the user message + image
captions (short, gemini-3.8-flash, image only) + first 100 chars of the assistant reply; selector prompt; N=40;
cosine key = round text + short caption (gemini-embedding-2); reader prompt (generic rules + format instruction).
Declared variant (motivated only by a data statistic: every has_answer user turn exceeds 200 chars, median 637):
"QSS-long" = same, but the index keeps the first 600 chars of the user message. Reported next to the frozen version.
Conditions, reader gemini-2.5-flash-lite (no thinking): FC; RAG@10/20/40; annotated evidence; QSS (3.8 selector);
QSS-long; dynamic-cutoff reranking = 3.8 grades every round 0-3 in one listwise call, keep >= 2 (fixed a priori);
lightweight consensus (2.5, 5 samples, >= 2 votes); cross-family QSS (selector Qwen3-235B-A22B-2507).
Selection only: + ranking control (3.8 ranks all rounds, cut at the QSS size), GPT-5-mini selector.
Strong reader gemini-3.8-flash on a fixed 200-question subset (seed 0): FC, RAG@40, QSS, QSS + index as context.
Judge: official MemLens prompts; primary Qwen3-VL-235B-A22B (OpenRouter); gemini-3.8-flash as second judge.
Hypotheses: H1 FSR(QSS) > FSR(cosine, same size); H2 acc(QSS) > acc(RAG@40), lightweight reader; H3 acc(QSS) >=
acc(dynamic-cutoff reranking) at comparable cost; H4 with the lightweight reader, FC degrades from 32k to 128k while
QSS does not; H5 the cross-family selector beats cosine at the same size. Statistics: question-level paired bootstrap
(each question has its own haystack), plus per-type breakdowns.
## P4-D3: cross-family selectors and an independent judge (MemEye)
Selection only, fixed 150-question MemEye subset (seed 0), same prompt and index (developed with Gemini):
  gemini-3.8-flash 7.7 rounds, FSR 0.887 [cosine same size 0.553] | gpt-5-mini 6.2, 0.700 [0.560] |
  gpt-5 2.8, 0.573 [0.433] | qwen3-235b-a22b-thinking 2.3, 0.407 [0.347] | gemini-2.5-flash-lite 9.6, 0.640 [0.613];
  all 371 q: gpt-5-mini 5.9, 0.728 [0.531]; qwen3-235b-a22b-instruct 1.5, 0.264 (mostly EMPTY selections).
  => with the same prompt the recall-oriented selection does not transfer across families: the reasoning models
  (gpt-5, qwen-thinking) return 2-3 rounds (precision-oriented). Set selection needs recall calibration; we did not
  adapt the prompt (that would be tuning on the reported data).
Independent judge (Qwen3-VL-235B, MemEye rubric) on stored predictions: lower absolute scores, exact agreement
0.69-0.87, all orderings preserved: 2.5 reader QSS 0.514 vs RAG@40 0.457 vs FC 0.469 (Gemini judge 0.541/0.488/0.495);
3.8 reader QSS+index 0.739 vs RAG@40 0.695 vs FC 0.758 (Gemini 0.796/0.741/0.819).
## P4-D4: first held-out MemLens results (frozen MemEye design) -- the end-to-end advantage does NOT replicate as is
2.5 reader, 699 q (Qwen3-VL judge / Gemini judge): FC 0.454/0.484 ($3.06) | RAG@10 0.426 | RAG@20 0.458 |
RAG@40 0.474/0.499 ($1.74) | annotated evidence 0.496/0.525 | QSS 0.484/0.512 (6.1 rounds, $6.84).
3.8 reader, 200 q: FC 0.725 | RAG@40 0.755 | QSS 0.675 (6.0 rounds).
Selection only: QSS opens 6.1 rounds for 5.9 gold rounds; FSR 0.375 [cosine same size 0.176]; image-round coverage 0.670
vs cosine 0.395; FSR over image gold rounds 0.381 vs cosine-40 0.648. Graded relevance + dynamic cut-off (>=2):
3.4 rounds, FSR 0.268 -> even more under-selection.
Interpretation (to be tested by the pre-declared snippet-600 variant): MemLens evidence is spread over long user
messages (has_answer turns median 637 chars) and generic photos ("a photo of a cup of coffee"); a 200-char snippet
plus a one-sentence caption hides much of it, so the index-based selector under-selects, whereas cosine over the full
round text still finds text evidence. The focused-reading ceiling is also low for the lightweight reader (annotated
evidence 0.496 vs RAG@40 0.474).
## P4-D5: held-out MemLens main results (results/TABLES_MEMLENS.md)
2.5 reader, 699 q: QSS 0.484 vs RAG@40 0.474 (+0.010 [-0.020,+0.040]) vs FC 0.454 (+0.030 [-0.001,+0.060]);
annotated evidence 0.496 -> little headroom for retrieval with this reader. By type QSS/RAG@40: IE 0.626/0.598,
TR 0.521/0.552, MSR 0.252/0.189, KU 0.405/0.431. Judges (Qwen3-VL vs Gemini, official prompt) agree 96-97%.
3.8 reader, 200 q: QSS 0.675 vs RAG@40 0.755 (-0.080 [-0.135,-0.030]); QSS+index 0.670; FC 0.725.
128k (2.5 reader, same 200 q): FC 0.440 (32k: 0.450), RAG@40 0.475, annotated 0.500, QSS 0.505 (+0.065 [+0.010,+0.120] vs FC).
Selection only (699 q): QSS 6.1 rounds FSR 0.375 [cos 0.176] cov 0.706 | lite 5.8/0.329 | lite consensus 6.4/0.346 |
graded relevance + dynamic cut-off (>=2) 3.4/0.268 | QSS with 600-char snippets 3.9/0.286 (MORE text -> FEWER rounds).
Pre-registered hypotheses: H1 supported (selection-level), H2 NOT supported, H4 weak (FC barely degrades at 128k),
H3 supported at selection level. => The end-to-end advantage of QSS is conditional. MemLens evidence = asides inserted
into unrelated sessions + incidental photos with generic captions; a compact index cannot expose it and the selector
under-selects (6 rounds for 5.9 gold rounds, 11.2 on MSR); full-text cosine with a wide budget plus a strong reader wins.
Post hoc (labelled as such): union of the QSS set and cosine top-k, to test complementarity.
## P4-D6: index visibility explains where index-based selection can work (results/index_visibility.txt)
Share of the question's content words that occur in a gold round AND are visible in its index line:
MemEye 0.95 (gold user messages median 45 chars) vs MemLens 0.37 (median 532 chars). Visibility on MemLens rises to
0.63 with 600-char snippets and 0.61 with question-matched snippets, yet the 600-char variant made the selector open
FEWER rounds (3.9) -> visibility is necessary but the selector's recall calibration on asides is the second factor.
No further prompt or index changes on MemLens (that would tune on the held-out set).
## P4-D7: more held-out MemLens conditions (2.5 reader, 699 q unless noted)
QSS-long (pre-declared 600-char snippets): 0.502, 3.9 rounds; vs RAG@40 +0.029 [-0.001,+0.059], vs FC +0.049 [+0.017,+0.080].
Dynamic-cutoff reranking (3.8 grades all rounds, keep >=2): 0.474 (= RAG@40), 3.4 rounds.
Lightweight consensus: 0.448 (-0.026 vs RAG@40). Post hoc union QSS + cosine top-20: 0.474 at 21.5 rounds.
3.8 reader (200 q): annotated evidence 0.730 < RAG@40 0.755 -> MemLens annotations omit context that the strong reader
uses, which caps any focused selection below wide retrieval for this reader.
MemEye selection only: dynamic-cutoff reranking (3.8, cosine top-100, keep >=2): 4.8 rounds, FSR 0.806 [cosine 0.445],
>=4: 0.686 [0.070] vs QSS (P=100) 7.7 / 0.919 and ranking-at-QSS-size 0.860.
Ranking control on MemLens: QSS - ranking (same size) +0.020 [+0.006,+0.034]; >=4 gold rounds +0.027 [+0.008,+0.048].
## P4-D8: same-cost strong baselines on MemEye (2.5 reader, 371 q)
Dynamic-cutoff reranking (3.8 grades cosine top-100, keep >=2): 0.509, 4.8 rounds, FSR 0.806, $10.63/1k;
vs QSS -0.032 [-0.065,-0.001]; vs RAG@40 +0.021 [-0.022,+0.064].
Post hoc union QSS + cosine top-20: 0.472 at 22.4 rounds (FSR 0.973); vs QSS -0.069 [-0.111,-0.027] -> precision
matters for the lightweight reader; no universal hybrid (on MemLens the union only matched RAG).
## P4-D9: cross-family selector on MemLens; union with the strong reader on MemEye
gpt-5-mini selector on MemLens (699 q, same prompt): 7.4 rounds, FSR 0.442 [cosine 0.205] -> higher than the Gemini
3.8 selector there (6.1, 0.375), while on MemEye it was lower (5.9, 0.728 vs 7.3, 0.903): recall calibration is model-
and data-dependent. MemEye 3.8 reader, union QSS + top-40: 0.771 (39.7 rounds, FSR 0.987); vs RAG@40 +0.030
[-0.003,+0.063]; vs QSS+index -0.025; vs FC -0.049 [-0.080,-0.018].
MemEye 3.8 reader, dynamic-cutoff reranking (grade38:100:2): 0.759 (4.8 rounds) ~ plain QSS 0.761; see analysis above.
gpt-5-mini selector end-to-end on MemLens (2.5 reader): 0.449 (7.4 rounds, FSR 0.442) < Gemini QSS 0.484 despite higher
FSR -> full-set recall over the annotated rounds is an imperfect proxy on MemLens.
## P4-D10: registered cross-family selector (H5) and final review round
Registered H5 model Qwen3-235B-A22B-2507 on MemLens: selection 1.4 rounds, FSR 0.149 [cosine same size 0.107] (mostly
tiny or empty selections); end to end with the 2.5 reader 0.313 (vs RAG@40 0.474). gpt-5-mini (added after the
registered runs) 7.4 rounds, FSR 0.442, end to end 0.449. => H5 holds formally at the selection level for both, but
neither cross-family selector helps answers.
Final review round (numbers audit + narrative): fixed title (question form), units (fractions), intro question
mapping, causal connector in Sec. 2, hedged RQ4 mechanism, post hoc labels in Table tab:memlens, H5 deviation disclosed,
all previously unrecorded statistics saved in results/memlens_stats.txt.

# Phase 5 (user request, 2026-10-27): algorithm-level improvement for selecting AND using evidence sets
Protocol: new mechanisms are developed only on MemEye (development benchmark) and a fixed MemLens dev split
(results/memlens_p5_split.json: 233 dev / 466 test, stratified by type and subtype, seed 1). MemLens-test is not
inspected until a pre-registered final run. Caveat: phase-4 aggregate MemLens results (all 699) motivated this phase.
## P5-D0: failure analysis (no API calls; results/p5/failure_*.txt, pool_recall.txt)
Reader side (dominant): with COMPLETE annotated evidence, the lightweight reader scores 0.06 on MemLens arithmetic
(n=50) and 0.09 on counting (n=46) vs 0.67/0.91 for the strong reader (200-q subset); duration comparison 0.30, order
ranking 0.21. MemEye with complete QSS sets: 0.64 / 0.52 / 0.45 for 1 / 2-3 / >=4 gold rounds. => aggregating over
several members in one pass is the main failure of the lightweight reader; better retrieval cannot fix it.
Selection side: MemEye QSS misses members in only 36 questions; 92% of missed members are in the top-10 unselected
rounds by question OR found-member similarity. MemLens QSS: 431 partial sets; 46% of missed members share a session
with a found member; median rank of missed members: 19 by question similarity vs 10 by similarity to found members.
Candidate pools (FSR / size): MemLens rounds top-30 0.542/30 vs sessions of top-10 0.638/30.6; sessions of top-20
0.880/48.8 vs rounds top-40 0.645/40. MemEye: rounds top-30 0.876/30 vs sessions of top-10 0.817/40.2.
## P5-D1: mechanism to test -- EVA (Expand, Verify, Aggregate)
1) recall-oriented candidate units: cosine top-k rounds, each expanded to its session context window, merged per session;
2) one cheap call per unit reads the RAW unit (text + images) and both decides membership and extracts question-specific
facts (values, names, dates) -> verification absorbs the precision cost of expansion and bypasses index visibility;
3) the reader answers from the compact list of extracted records (optionally plus the raw verified units), which turns
multi-member aggregation into reading a short list.
## P5-D2: dev results, round 1 (MemLens dev, 233 q, 2.5 reader; results/p5/dev_round1.md)
EVA with session units: oracle|eva 0.494, sess:10|eva 0.451 (RAG@40 0.489; oracle read 0.502). Only 1.5 of 3.1 gold
sessions judged relevant; 19% of questions fall back. Round units: oracle|evau 0.519, oracle|evaur (notes + raw of
relevant rounds) 0.532 (+0.030 over oracle read); arithmetic 0.00 -> 0.24, counting 0.07 -> 0.26, order 0.25 -> 0.50.
Map recall probe on gold units (40-q dev sample): round units 0.29 (2.5) / 0.33 (3.8); session units 0.41 / 0.51;
thinking does not help, a lenient prompt +0.02-0.06. Even the strong model marks half of the gold sessions irrelevant
-> our MemLens gold definition over-includes (needle images in answer sessions are not always evidence for THIS
question); FSR on MemLens is noisy, end-to-end accuracy is the reliable criterion.
Image-image member expansion: no better than question similarity on MemEye (median rank 8 vs 5) or MemLens (3 vs 3).
Sentence-granular index (max-sim over user sentences + captions): MemLens dev top-20 FSR 0.494 vs round 0.438
(coverage 0.832 vs 0.748); MemEye top-20 0.666 vs 0.803 (worse: short rounds). Not a general improvement.
## P5-D3: "read with notes" (readn) -- notes from per-round cheap verify+extract as a reading aid, PLUS all raw rounds
MemLens dev (233 q, 2.5 reader): rag:20|readn 0.537 (rag:20 read 0.438: +0.099; rag:40 read 0.489; rag:40 + thinking
0.515), $2.47/1k; oracle|readn 0.554 (oracle read 0.502); gran:20|readn 0.532. Thinking (1024) helps arithmetic/order
(0.29 / 0.62-0.88) but hurts date extraction (0.96 -> 0.69-0.85); readn helps entity/entity-resolution and keeps date
extraction at 0.96 -> complementary; next: readn + thinking, and readn on MemEye.
Instance expansion (cheap model lists concrete instances of the asked category; rounds scored by best match over
question + instances): MemLens dev top-20 FSR 0.442 vs 0.438; MemEye 0.782 vs 0.803 -> no gain; dropped.
## P5-D4: readn does not generalize; session closure completes sets but hurts the lightweight reader
MemEye (2.5 reader): rag:20|readn 0.471 (= rag:20 read 0.471); oracle|readn 0.545 (oracle 0.539); toc38|readn 0.525
(toc38 read 0.541). readn + thinking on MemLens dev: rag:20 0.519, oracle 0.541 (no gain over readn without thinking).
Query-time visibility estimate (top-10 cosine rounds, no gold): MemEye mean 0.94 (98.7% >= 0.7) vs MemLens dev 0.44
(12.4%) -> separates the two regimes almost perfectly, but does not predict per-question winners within MemLens.
Session closure (selected rounds + all rounds of their sessions): MemLens dev QSS-long 4.0 -> 16.0 rounds, FSR
0.292 -> 0.824, yet accuracy 0.549 -> 0.472; QSS 0.485 -> 0.476; MemEye QSS 0.541 -> 0.526 (FSR 0.903 -> 0.938);
+-1 neighbours 0.511. => for the lightweight reader, completeness bought with more rounds is a net loss: the reader,
not recall, binds. Next: closure + notes (lightweight), and closure with the strong reader (MemLens dev).
## P5-D5: strong reader on MemLens dev; notes on QSS-long sets
3.8 reader, MemLens dev (233 q): RAG@40 0.665 ($15.16) | annotated 0.678 | QSS-long (600-char snippets) 0.682 ($11.27)
| QSS-long + session closure 0.682 ($15.85). 2.5 reader: QSS-long + notes (readn) 0.562 (QSS-long read 0.549);
QSS-long + closure + notes 0.532 (closure read 0.472). Per-question oracle max(core, closure) = 0.584 (core 0.549).
=> completeness beyond a precise core does not help either reader; index FIDELITY (what the index line exposes) is the
lever that moved both readers. Next: model-written gist lines as a visibility-preserving index (write-time, cheap).
## P5-D6: model-written gist index lines (write-time, cheap) vs raw snippets
QSS with gist lines: MemLens dev 0.515 (frozen 0.485, QSS-long 0.549); MemEye 0.504 (QSS 0.541) -> gists lose exact
wording and aliases; raw text of adequate length beats rewritten text for the index as well. Lightweight selector with
gists: MemLens dev 0.476, MemEye 0.488 (frozen 0.497).
## P5-D7: dev summary and the candidate algorithm (visibility-gated selection and reading, VGS)
QSS-long on MemEye: 0.528 (QSS 0.541) -> longer snippets are not free on short-message data. Query-time visibility
estimate v_hat (top-10 cosine rounds; share of question words present in a round that its 200-char index line shows;
no gold needed) separates regimes: MemEye 98.7% >= 0.7, MemLens 12.4-12.7%.
VGS: v_hat >= 0.7 -> QSS with 200-char lines + standard reading; v_hat < 0.7 -> QSS with 600-char lines + per-round
notes (readn). VGS-cheap: low path = cosine top-20 + notes (no strong selector).
Dev (2.5 reader): MemLens dev VGS 0.558 (+0.069 [+0.013,+0.124] vs RAG@40), $8.60; VGS-cheap 0.541 (+0.052
[+0.004,+0.099]), $3.05; RAG@40 + thinking 0.515. MemEye VGS 0.541 (= QSS, +0.053), VGS-cheap 0.540.
## P5-D8: PRE-REGISTRATION of the held-out test (MemLens-test, 466 q; written before any run on the test split)
Frozen: gate tau = 0.7 on v_hat as defined above; QSS selector gemini-3.8-flash; 600-char lines on the low path;
notes = per-round verify+extract by gemini-2.5-flash-lite (EVA_MAP prompt as in src/mm2.py at this commit);
reader gemini-2.5-flash-lite without thinking unless stated; judge = official MemLens prompts with Qwen3-VL-235B.
Lightweight-reader conditions: RAG@40; RAG@40 + thinking 1024; QSS (200); QSS-long (600; its test-split outputs were
already computed in phase 4 as part of all 699 questions); QSS-long + notes; RAG@20 + notes; VGS; VGS-cheap.
Strong reader (gemini-3.8-flash, all 466 test questions): RAG@40; QSS (200); QSS-long (600); VGS-strong = gate(QSS |
QSS-long) (notes not used for the strong reader).
Hypotheses: H1 VGS > RAG@40 (lightweight); H2 VGS-cheap > RAG@40 at < 2x its cost; H3 QSS-long + notes >= QSS-long;
H4 VGS >= RAG@40 + thinking; H5 strong reader QSS-long >= RAG@40 at lower cost. Paired bootstrap over questions.
## P5-D9: held-out test results (MemLens-test, 466 q; pre-registered in P5-D8)
Lightweight reader: RAG@40 0.466 ($1.73) | RAG@40 + thinking 0.496 | QSS 0.483 | QSS-long 0.479 | QSS-long + notes
0.536 | RAG@20 + notes 0.485 | VGS 0.536 (+0.071 [+0.032,+0.109]) $8.63 | VGS-cheap 0.494 (+0.028 [-0.011,+0.069]).
H1 holds (VGS > RAG@40); H2 fails (VGS-cheap n.s.); H3 holds (notes over QSS-long +0.058 [+0.021,+0.097]); H4 borderline
(VGS - RAG@40 thinking +0.041 [-0.002,+0.084]); H5 fails for the strong reader: RAG@40 0.689 ($14.74) vs QSS 0.618,
QSS-long 0.659 (-0.030 [-0.062,+0.002]), VGS-strong 0.655. Dev-only gains that did NOT replicate: QSS-long alone (dev
+0.060 -> test +0.013), RAG@20 + notes (dev +0.047 -> test +0.019). By subtype VGS vs RAG@40 (lite): arithmetic 0.03
-> 0.21, counting 0.10 -> 0.23, order 0.12 -> 0.38, entity +0.08, entity resolution +0.10, update +0.05, previnfo +0.06,
date extraction 0.98 -> 0.94. Mem-Gallery (another benchmark, 903 q): 12.3% low-visibility questions; VGS 0.852 = QSS,
RAG@40 0.839; notes do not help its low-visibility subset (0.876 vs 0.878). MemEye: VGS = QSS on 98.7% (0.541).
Post hoc (dev only): cheaper selectors with notes do not reach VGS: lightweight selector (600-char lines) + notes 0.498
($1.88), cosine top-10 + notes 0.468 ($1.28) vs QSS-long + notes 0.562 -> the precise set from the strong selector is
what makes the notes useful. Deployment cost estimate for VGS with a cached per-user index: selection ~$1.7/1k +
notes ~$0.5 + reading ~$0.5 = ~$2.7/1k (MemLens measures $8.6 because every question has its own haystack).
Correction to the cost line above: the exact decomposition (results/p5/vgs_cost.txt) is selection $8.10 + notes $0.47 +
reading $0.31 = $8.88 measured; warm-cache estimate $1.67 + $0.47 + $0.31 = $2.45 (not ~$2.7).
## P5-D10: audit of the new RQ6 text -> wording fixes (closure numbers are for the 600-char set; H2/H4/H5 stated in full;
Mem-Gallery low-visibility subset gains nothing from notes; strong-reader aggregation rates are from the 200-q subset).

# Phase 6 (user request): bring QSS/VGS cost toward RAG (lightweight reader); beat 3.8 RAG at ~RAG cost (strong reader)
## P6-D0: cost anatomy and offline analyses (no new reading runs)
VGS on MemLens: $8.10 of $8.88 per 1k q is the 3.8 selection call; 3.8 RAG@40 $14.74 = 17.2k input tokens ($12.9) +
481 output tokens ($1.8). Any 3.8 call costs >= ~$0.9/1k in output (thinking 'low' is the minimum), i.e. as much as the
whole lightweight RAG@40 pipeline on MemEye ($0.86).
Set-ness router (results/p6/setness.json; 2.5-flash-lite reads only the question; ~$0.01/1k): multi-item share
MemEye 0.547, MemLens 0.288, Mem-Gallery 0.107.
Offline routing (single -> cheap path, multi -> set path), lightweight reader:
  MemEye: single RAG@40 / multi QSS 0.527 ($3.84); single RAG@40 / multi QSS+thinking 0.573 ($3.99; QSS 0.541 $6.73);
  single RAG@40 / multi RAG@40+thinking 0.545 ($1.03); RAG@40+thinking 0.552 ($1.12); FC+thinking 0.571 ($0.67, prefix
  cache across questions of one history).
  MemLens dev/test: single RAG@40 / multi VGS 0.536/0.502 ($3.92/$4.13; VGS 0.558/0.536 at ~$8.6); single RAG@40 / multi
  RAG@40+thinking 0.524/0.496 ($1.84/$1.81). VGS also gains on single-item questions (test 0.612 vs 0.563), which
  routing gives up.
Strong reader: MemEye single RAG@10 / multi QSS+index 0.784 ($10.16) vs RAG@40 0.741 ($10.00) -> +0.043 [+0.008,+0.078]
at equal cost. MemLens: 3.8 abstains on 8.2% of test questions (21 of 38 are date questions whose answer is the session
date); fallback to the lightweight RAG@40 answer on abstention: test 0.738 (+0.049), dev 0.682 (+0.017), MemEye +0.008.
## P6-D1: cheaper selectors and pools (selection only)
gemini-2.5-flash as selector: MemEye FSR 0.690 (no thinking) / 0.658 (thinking) vs 3.8 0.903; MemLens dev 0.197 (2.5
rounds) -> mid-tier models under-select like the lightweight one. Cosine prefilter + 3.8: MemEye P=20 0.733 ($2.19),
P=40 0.833 ($3.67); MemLens dev (600-char) P=20 0.236, P=30 0.262 (full 0.292). Cascade (lite selection U cosine
top-15 -> 3.8 within the pool): MemEye 7.0 rounds FSR 0.819 ($2.65; >=4 sets 0.640 vs 0.872), MemLens dev 0.232 ($3.34).
## P6-D2: anticipatory set resolution (AnSR) -- negative
3.8 reads each MemEye index once and writes 40 anticipated questions, each resolved by QSS (ingestion $2.42 for 8
histories); the lightweight selector sees the most similar anticipated resolutions: FSR 0.717 (none: 0.712; real strong
resolutions of other questions: 0.752); model-free reuse 0.677. Anticipated questions do not hit the sets users ask about.
3.8 reading notes over top-40 + only top-10 raw: MemLens dev 0.528 (RAG@40 0.665) -> lightweight notes cannot replace
raw rounds for the strong reader.
## P6-D3: measured cost of a cache-native index (explicit context caching), MemEye
Implicit caching does not help in practice: with a per-history warm-up schedule only 3.3% (selection) and 5.0% (reading)
of prompt tokens were cached. Explicit context caching (gem.explicit_cache: system instruction + static index text in a
cache; requests send only the question / the opened rounds; creation + storage logged as 'xcache:create'):
  QSS selection $1.76/1k (uncached $9.4); creation + storage $0.26/1k amortized over 371 questions.
  Lightweight reader: QSS 0.507 ($2.34) | QSS + thinking 0.590 ($2.56; uncached 0.597 at $6.97) | routed (single RAG@40,
  multi QSS + thinking) 0.568 ($1.81) | RAG@40 0.488 ($0.86) | RAG@40 + thinking 0.552 ($1.12).
  Strong reader: QSS + index (cached) 0.800 ($6.78) vs RAG@40 0.749 ($9.78): +0.051 [+0.015,+0.088] at 69% of the cost.
Selection-draw variance: three independent 3.8 selection draws give the lightweight reader 0.541 / 0.537 / 0.507 (only
~52% of the sets are identical between draws); with a thinking reader the draws agree (0.597 / 0.590).
Other negatives: 3.8 with 'medium' thinking: MemLens dev 0.605, MemEye 0.732 (low: 0.665 / 0.749); 3.8 RAG@20 on MemLens
dev 0.519 (RAG@40 0.665); 3.8 RAG@40 + lightweight notes 0.597 (notes mislead the strong reader); routing single-item
questions to RAG@10 hurts Mem-Gallery (-0.036). Lightweight multi-item questions on MemLens dev: QSS-long + thinking 0.435
vs QSS-long + notes 0.387 vs RAG@40 + thinking 0.339 (n=62).
## P6-D4: PRE-REGISTRATION of the phase-6 confirmation (written before any run below)
(A) MemLens-64k, test-split questions (466; new, twice longer haystacks: 124.8 rounds, 25.5 images per question; the
questions were seen at 32k). Index, prompts, gate tau=0.7 (v_hat recomputed on the 64k haystacks), set-ness labels
(question-only, unchanged) frozen. Lightweight reader: RAG@40; RAG@40 + thinking 1024; VGS (QSS 200-char | QSS 600-char +
notes); routed thinking = single RAG@40 / multi RAG@40 + thinking; routed VGS = single RAG@40 / multi VGS. Strong reader
(3.8): RAG@40; RAG@40 + abstention fallback to the lightweight RAG@40 answer (regex abstention detector as in
results/p6/abstention.txt, frozen).
Hypotheses: A1 routed thinking > RAG@40 at <= 1.2x its cost; A2 routed VGS > RAG@40 at <= 0.6x the cost of VGS;
A3 VGS > RAG@40; A4 strong RAG@40 + fallback > strong RAG@40 at <= 1.05x its cost.
(B) Mem-Gallery (903 q; persistent histories, 20 scenarios), strong reader: QSS + index with the explicit index cache vs
RAG@40. Hypothesis B1: cost below RAG@40 with accuracy non-inferior (difference CI lower bound > -0.03).
## P6-D5: results so far (registered P6-D4)
Mem-Gallery (B1), strong reader: QSS + index with the cached index 0.909 vs RAG@40 0.912 (-0.002 [-0.013,+0.008]);
cost $5.05/1k + cache creation/storage $0.91/1k = $5.96 vs RAG@40 $5.70 -> non-inferior at ~equal cost; "cost below
RAG@40" fails narrowly (+5%).
Extra selection draws on MemEye (lightweight reader, no thinking): 0.541 / 0.537 (uncached) and 0.507 / 0.531 (cached
index) -> mean 0.529; the cached-prompt format does not shift accuracy systematically.
MemLens-64k (test-split questions): lightweight RAG@40 0.455 ($1.13); VGS high-visibility path 0.487 (78 q, $8.67),
low-visibility path 0.490 (388 q, $12.26). Remaining runs: RAG@40 + thinking (lightweight), RAG@40 (strong).
## P6-D6: INVALIDATED RESULTS -- the OpenRouter key expired (401 "API key expired"); MemLens judging failed
The last successful Qwen3-VL judge call was on 2026-10-27 11:52. Every MemLens run made on 2026-11-01 contains records
whose judge call failed; run_mm2 stored them as errors with score 0 and an empty prediction (the reading calls are in
the local cache, so re-running after the judge is available recovers them without new reading cost). Affected (errors /
records): MemLens-32k dev, strong reader: RAG@10 63/233, RAG@20 57/233, thinking medium 44/233, notes + 40 raw 25/233,
notes + top-10 raw 58/233, notes + top-20 raw 56/233; dev multi-item lightweight runs 15-16/62; MemLens-64k: all four
runs (19/78, 80/388, 122/466, 101/466). => All phase-6 MemLens numbers in P6-D1, P6-D3 (MemLens part) and P6-D5
(64k part) are biased downward and must not be used; the conclusions drawn from them ("notes / more thinking / smaller k
hurt the strong reader on MemLens", the 64k confirmation) are withdrawn until re-judged. Unaffected: everything on MemEye
and Mem-Gallery (Gemini judge), all phase-4/5 MemLens runs (0 errors), selection-only metrics, and the offline routing
analyses built on phase-4/5 records.
## P6-D7: re-judged results (new OpenRouter key supplied by the user, passed via environment only; 0 failed records)
MemLens-32k dev, strong reader: RAG@40 0.665 | thinking 'medium' 0.661 | notes + 40 raw 0.631 | notes + top-20 raw 0.601
| notes + top-10 raw 0.579 | RAG@20 0.605 | RAG@10 0.575 -> wide retrieval remains best; lightweight notes do not help the
strong reader; more thinking neither helps nor hurts (the earlier 0.605 / 0.528 / 0.541 / 0.519 were judge failures).
Registered confirmation (P6-D4), MemLens-64k test-split questions (466):
  lightweight: RAG@40 0.468 ($1.45) | RAG@40 + thinking 0.476 | VGS 0.524 (+0.056 [+0.015,+0.097], $14.81) | routed
  thinking 0.481 (+0.013 [-0.009,+0.036], $1.52) | routed VGS 0.494 (+0.026 [+0.004,+0.049], $5.86 = 0.40x VGS).
  strong: RAG@40 0.655 ($12.75); abstains on 7.3%; + fallback to the lightweight answer 0.697 (+0.043 [+0.026,+0.062],
  $12.85).
  -> A1 fails (n.s.), A2 holds, A3 holds (VGS replicates on 2x longer haystacks), A4 holds.
Mem-Gallery (B1): cached QSS + index 0.910 vs RAG@40 0.912 (-0.001 [-0.012,+0.009]); $5.06 + $0.91 cache overhead =
$5.97 vs $5.70 -> non-inferior, cost +5% (B1 fails narrowly on cost).

# Phase 7 (user request): further cost control; what goes into the index and how compactly
## P7-D0: index anatomy and compression (dev; results/p7_index_anatomy.txt, p7_index_table.md, p7_e2e_dev.md)
Line composition (MemEye, ~73 tokens/line): header 9%, user snippet 30%, caption 37%, assistant snippet 23% (often
"Correct." / "Got it"). MemLens 600-char lines (~128 tokens): user 66%, assistant 18%.
Compression (deterministic, no model calls; mm2.NO_ASSIST / SHORT_IDS / TELEGRAPH, run_mm2 --compact):
  noassist = drop the assistant snippet; shortid = rounds listed as r<k> under one "== S3 | date ==" header per
  session (session label kept when the benchmark's own label is short, e.g. S3 / D9); telegraph = drop function words
  and filler punctuation from the user text before truncation (names, numbers, labels kept).
Selection (3.8): MemEye tokens 11,286 -> 8,976 / 6,658 / 5,797 (noassist / +shortid / all three); FSR 0.903 -> 0.900 /
0.881 / 0.876. MemLens dev 600-char: 9,328 -> 4,026 tokens, FSR 0.292 -> 0.266.
End to end (dev): MemEye lite+thinking QSS 0.597 (frozen) vs 0.608 / 0.596 / 0.600 (noassist / +telegraph / all three;
all |Δ| <= 0.011, CIs +-0.03); MemLens dev VGS low path (600 + notes) 0.562 frozen vs 0.562 compact, $8.83 -> $4.49;
compact 200 + notes 0.515 (-0.047 [-0.086,-0.009]) -> the telegraphed 600 chars, not a short window, carry the gain.
Long histories (MemEye + 1k images, 4,068 rounds): cosine pool recall P=100/200/320/500: 0.951/0.970/0.981/0.989
(sets >= 6 rounds: 0.3/0.5/0.6/0.9); BM25 fusion no better. Selection: frozen P=200 FSR 0.914 (16.8k tokens);
compact P=200 0.922 (10.7k); compact P=320 (same tokens as frozen P=200) 0.919 -> the selector, not the pool, limits;
compress the lines rather than enlarge the pool.
## P7-D1: PRE-REGISTRATION (written before the runs below)
MemLens-64k test-split questions (466), lightweight reader, gate as registered in P6-D4 (v_hat on frozen 200-char lines):
compressed VGS = high-visibility path QSS with compact 200-char lines; low-visibility path QSS with compact 600-char lines
+ notes (compact = noassist + shortid + telegraph). Hypotheses: C1 compressed VGS non-inferior to frozen VGS (difference
CI lower bound > -0.03) at <= 0.6x its cost; C2 compressed VGS > RAG@40; C3 routed compressed VGS (multi-item questions
only) > RAG@40 at <= 0.3x the cost of frozen VGS.
## P7-D2: results of P7-D1 (MemLens-64k test-split questions, 466) and the selector's thinking
C1 holds: compressed VGS 0.528 vs frozen VGS 0.524 (+0.004 [-0.019,+0.028]) at $7.03 vs $14.81 (0.47x).
C2 holds: compressed VGS vs RAG@40 (0.468) +0.060 [+0.021,+0.099]. C3 holds: routed compressed VGS 0.502 (+0.034
[+0.013,+0.056]) at $3.36 = 0.23x frozen VGS (RAG@40 $1.45).
MemEye, persistent memory, compact index + explicit cache: lightweight reader QSS + thinking 0.589 at $1.83/1k (frozen
cached 0.590 at $2.58); routed 0.570 at $1.52; strong reader QSS + index (no assistant + telegraph) 0.803 (+0.053
[+0.020,+0.088] vs RAG@40) at $6.27 vs $9.78. Cached compact selection call per 1k q: uncached part $0.03 + cached
index $0.43 + output $0.21 + thinking $0.50.
Selector without thinking (gemini-3.8-flash accepts thinking_budget=0; 0 thinking tokens): MemEye frozen index FSR 0.911
(thinking 'low' 0.903), compact 0.881 (0.876); MemLens dev compact 600: 0.275 (0.266) -> thinking is not needed for
set selection; it was the largest item of the cached selection cost.
PRE-REGISTRATION (written before the runs): C4 on the same 466 MemLens-64k questions: compressed VGS with a no-thinking
selector is non-inferior to compressed VGS with thinking (difference CI lower bound > -0.03). The 64k test was already
used for C1-C3; the no-thinking variant was chosen on MemEye and the MemLens dev part only.
## P7-D3: CORRECTION -- gemini-3.8-flash does not honour thinking_budget=0
The claim in P7-D2 that a budget of 0 turns thinking off was based on one call. On 32 fresh MemEye selection calls:
mean thinking tokens 'low' 129 (thinks on 44% of calls), budget 0: 124 (31%), budget 128: 127 (38%). Thinking is
stochastic per call and cannot be disabled for this model ('minimal' is rejected). The "no-thinking" runs are therefore
just another selection draw: MemEye lite + thinking 0.583 (compact, cached; the thinking-selector run gave 0.589);
MemLens-64k compressed VGS 0.513 vs 0.528 (C4 is moot). The selector's output + thinking (~$0.7/1k) stays a fixed floor.
## P7-D4: paper revision with the writing-as-qy skill (no new experiments)
Structure: six RQs (zooming/captions moved to Appendix C); RQ5 = VGS (failure analysis, development exploration,
registered test), RQ6 = cost (cheaper calls, cache-native index, compact index, single-use histories), each in the
goal -> observations -> answer form. Abstract and introduction rebuilt around four challenges (measure sets, decide
membership, reach answers on held-out data, stay affordable) with five contributions. Method gains an overview and labels;
Setup states selector randomness and the RQ6 registrations; Limitations split into data / models and measurement /
protocol and cost. New appendix units: compact index (Table 14), set-aware router check, measured cached costs vs the
warm-cache estimates (within 8%). Related work adds Chain-of-Note, LLMLingua, Prompt Cache (verified via the arXiv API).
Number check: every number in the revised text occurs in the previously audited paper or in stored result files.

# Phase 8 (user request): address the shared, substantive points of two AI reviews
Shared points chosen (claim tightening and multi-draw requests skipped as instructed): (i) index visibility is only a
correlate and the gate looks like a benchmark detector; (ii) VGS is validated within one benchmark; (iii) the
set-formulation control differs from QSS also in its recall-oriented instruction; (iv) full-set recall on MemLens rests
on derived gold; (v) privacy of persistent cached memories; (vi) missing multimodal-memory related work.
## P8-D0: offline analyses (results/p8_offline.md; no API calls)
v_hat vs oracle visibility (MemLens-32k, 679 q with defined oracle visibility): Spearman 0.48; AUC 0.75 for oracle
visibility >= 0.5; means 0.427 / 0.430. Gate per path on the registered test part (466 q): high v_hat (59 q) RAG@40
0.559, QSS-200 0.542, notes path 0.542, VGS - RAG@40 -0.017 [-0.136,+0.085]; low v_hat (407 q) RAG@40 0.452, QSS-200
0.474, QSS-600 0.469, notes path 0.536, VGS - RAG@40 +0.084 [+0.042,+0.125]. Notes-path gain over QSS-200 by v_hat
quartile: +0.026 / +0.128 / +0.052 / +0.009 (largest at low-intermediate visibility, none at high).
FSR under alternative gold definitions (699 q): QSS-200 vs equal-size cosine: derived 0.375 vs 0.176; evidence-turn
rounds 0.614 vs 0.385; needle-image rounds 0.251 vs 0.106; answer sessions covered 0.687 vs 0.554. QSS-600: evidence-turn
0.828 (cosine 0.320) but needle-image 0.133 (0.052) -> 600-char lines expose the text evidence, the selector stops after
it and misses the image rounds whose captions stay generic: this resolves the "more visibility, lower FSR" tension.
Audit (strong reader, 200-q subset): 13 questions with RAG@40 right and annotated evidence wrong (8 the other way); in
all 13, RAG@40 also retrieved other rounds of an answer session, in 6 the answer words occur in retrieved non-gold
rounds, in 5 in the gold rounds -> the derived gold omits same-session context.
## P8-D1: PRE-REGISTRATION of the external test on WorldMemArena (written before any run on it)
Data: WorldMemArena lifelong (personal + project; arXiv 2605.29341), all 190 checkpoint prefixes, 2,090 questions
(median user message 388 characters). Nothing in our design is changed: our short captions, the 200-character index,
the QSS prompt, v_hat as registered, tau = 0.7, low path = 600-character lines + per-member notes; lightweight reader
gemini-2.5-flash-lite without thinking; explicit index cache (persistent histories); MemEye rubric judge
(gemini-3.8-flash). Primary: W1 VGS > RAG@40 (lightweight reader). Secondary: W2 QSS > RAG@40 (lightweight); W3 the
notes path gains more over QSS-200 on low-v_hat than on high-v_hat questions; W4 strong reader (gemini-3.8-flash):
QSS + index with the cached index >= RAG@40 at lower cost.
## P8-D2: recall-oriented ranking control (review point iii)
RANK_RECALL_SYS = the QSS instruction (every possible member, label-defining rounds, coarse images) asking for a full
ranking; cut at the QSS size of each question. MemEye (P=100): rank@QSS-size 0.898 (original ranking prompt 0.860),
QSS 0.919 -> QSS - rank +0.022 [+0.003,+0.043] (was +0.059); >=4 gold +0.047 [-0.012,+0.105] (was +0.151); recall
ranking at fixed k: 0.868 (k=10), 0.946 (k=20). MemLens-32k (all rounds): rank@QSS-size 0.362 (original 0.355), QSS
0.375 -> +0.013 [+0.001,+0.024]; >=4 gold +0.024 [+0.008,+0.043]. => Most of the earlier MemEye gap came from the
recall-oriented instruction; the unordered set output adds a small further gain even when the ranking receives the QSS
size for free, and it chooses the size itself (fixed-size ranking needs 10-20 rounds to match QSS at 7.7).
## P8-D3: causal manipulation of index visibility on MemEye (review point i)
Every content word of the question is replaced by "[...]" with probability p in ALL index lines (gold and non-gold
alike; deterministic per question/round/word); reading unchanged; lightweight reader with thinking.
p=0.5: FSR 0.879 (-0.024 [-0.057,+0.008]), 8.15 rounds, accuracy 0.604 (+0.007 [-0.024,+0.038]).
p=1.0: FSR 0.814 (-0.089 [-0.124,-0.054]), 8.51 rounds, accuracy 0.511 (-0.086 [-0.127,-0.045]) -- below RAG@40 with
thinking (0.552), i.e. hiding the question words from the index reproduces the MemLens pattern on MemEye. The selector
compensates for partial masking (more rounds, semantic cues), so lexical visibility is a causal factor but not the
whole story.
## P8-D1a: AMENDMENT to P8-D1 (written before any run on WorldMemArena)
The unauthenticated Hugging Face download is rate-limited (ETA > 4 h for the project images); all 20 lifelong-personal
samples are complete. The registered test is therefore restricted to WorldMemArena lifelong-personal: 100 checkpoint
prefixes, 1,100 questions. Everything else in P8-D1 is unchanged; the notes path is run on all questions so that W3 can be
tested on both sides of the threshold.
## P8-D4: RESULTS of the registered external test (P8-D1/D1a), WorldMemArena lifelong-personal (1,100 q)
v_hat mean 0.438 (MemLens 0.427); 3.0% of questions >= 0.7, so VGS takes the notes path for 97%.
Lightweight reader: RAG@40 0.692 ($1.18) | QSS (cached) 0.700 (+0.009 [-0.008,+0.026], $4.28 incl. caches) | QSS-600 +
notes (cached) 0.692 (+0.000 [-0.018,+0.019], $6.37) | VGS 0.694 (+0.003 [-0.015,+0.021], $6.31).
W1 (primary, VGS > RAG@40) FAILS. W2 (QSS > RAG@40) fails. W3: notes path - QSS-200 = -0.083 (33 high-v_hat q) vs -0.006
(1,067 low-v_hat q); difference +0.077 [-0.034,+0.198], n.s. W4 (strong QSS + index cached >= RAG@40 at lower cost)
FAILS: 0.795 vs 0.802 (-0.007 [-0.018,+0.004]); $10.22 vs $9.01 = calls $6.47 + selection caches $1.87 + reading caches
$1.88 (analytic attribution, equal to the logged creations for the QSS run).
Diagnostics: notes per question 14.6 (MemLens test 3.7), notes-path selection opens 10.4 rounds (MemLens 3.9); by type the
notes path gains on memory conflict (+0.047) and visual fact recall (+0.050) and loses on temporal reasoning (-0.083) and
memory boundary (-0.047); the lightweight reader is already at 0.69 with RAG@40 (MemLens 0.47).
Retrieval still transfers at the session level: all evidence sessions touched by QSS 0.743 vs equal-size cosine 0.689
(+0.054 [+0.034,+0.074]).
Cost condition: each cached prefix serves 11 questions (MemEye 46.4); break-even of the cached strong pipeline against
RAG@40 ~ 16 questions per cached index (overhead $3.75/1k / saving $2.54/1k x 11).
=> VGS is not a general improvement: low index visibility alone does not identify where notes help; the MemLens gain
coincides with a reader that fails to aggregate few members. The cost advantage of the cache-native index requires
enough questions per cached index.
- correction: the median user message of the personal part (full histories) is 397 characters (388 was computed over all lifelong prefixes).

# Phase 9 (user request): was the WorldMemArena null result caused by images? Can QSS/VGS be improved?
## P9-D0: image check and an evaluation bug
All 1,803 images of lifelong-personal are present; readers received images (RAG@40 2,718 image tokens per question).
Bug: data_wma did not set Round.image_ids, so images were labelled by round ("S05:R5#img1") while Visual-Search gold
answers are benchmark image ids ("personal_01_img_019") -> VS was unanswerable for every method (0.04-0.08). Fixed
(image_ids from attachments; IMAGE_RULE["wma"] asks for the bracketed id when the question asks which image).
Re-run, tag _imgid (1,100 q): RAG@40 0.779 ($1.17) | QSS 0.791 (+0.012 [-0.005,+0.028], $2.42) | notes path 0.785 |
VGS 0.786 (+0.007 [-0.011,+0.025], $3.88). By half: dev QSS -0.003, test QSS +0.027 [+0.005,+0.048].
## P9-D1: visual dependence of WorldMemArena
RAG@40 with captions instead of images 0.767 (-0.012); with no visual information 0.699 (-0.080 [-0.100,-0.061]):
~90% of the accuracy survives without any visual information (MemEye: 0.191 vs 0.495 = 39%).
Question-text classifier "visual vs text" (gemini-2.5-flash-lite, results/p9/visualq.json): set path minus RAG@40 on
visual / text questions: MemEye +0.039 / +0.071; MemLens test (VGS) +0.062 / +0.075; Mem-Gallery +0.018 / +0.010;
WMA (VGS) +0.020 / +0.003. => gains are a property of the benchmark, not of visual questions within it; they are large
where RAG@40 is weak (MemEye 0.49, MemLens 0.47) and small where it is strong (Mem-Gallery 0.84, WMA 0.78).
Pooled Mem-Gallery + WMA, QSS - RAG@40: +0.012 [+0.000,+0.025] (borderline).
## P9-D2: improvement attempts (development data only)
Answer arbitration between the set view and the wide view (lite arbiter on disagreements, evidence = QSS set + cosine
top-10): oracle max(set, wide) is 0.623/0.556/0.897/0.764 on MemEye/MemLens/Mem-Gallery/WMA, but the arbiter reaches
MemEye 0.528 (no thinking) / 0.542 (thinking) vs QSS 0.541, MemLens dev 0.562 vs VGS 0.558. Not useful.
WMA dev half (550 q) vs RAG@40 0.773: QSS -0.003; QSS U cosine top-10 -0.004; notes path +0.010; notes + raw union with
top-40 -0.016; RAG@40 + index as context -0.014; QSS + thinking 0.812 vs RAG@40 + thinking 0.834 (-0.022).
MemLens dev: notes + raw union with top-40 0.537 (notes path 0.562, RAG@40 0.489); RAG@40 + index 0.506 (+0.017);
QSS + index 0.481. Set-ness routing on WMA: QSS gains +0.012 on both single- and multi-item questions (no lever).
Visual routing (visual questions -> set path): WMA +0.005 at $1.71 vs RAG $1.17; MemLens +0.021 at $3.79.
## P9-D3: strong reader on WorldMemArena with image ids fixed
RAG@40 0.889 ($8.89) vs QSS + index (cached) 0.882 (-0.007 [-0.018,+0.005]; calls $6.44 + caches $3.76 = $10.21);
break-even ~17 questions per cached index. Paper Table 8 now reports the corrected run (registered run disclosed).
## P9 conclusion
No variant tried here lifts WorldMemArena or Mem-Gallery to a significant gain; the configuration that already beats
RAG@40 by >= 2 points on two benchmarks is VGS (MemEye +0.053 through its QSS path, MemLens test +0.071, 64k +0.056).
The gains track the weakness of RAG@40 for the lightweight reader (< 0.5 on MemEye/MemLens, > 0.75 on Mem-Gallery/WMA),
not the visual modality of individual questions. Spend this phase ~ $46 (Gemini).

# Phase 10 (user request): ACL paper, image-dependent benchmarks only, unified gated method
## P10-D0: benchmark scope (written before any run on a new benchmark)
Scope = memory whose answers depend on images. Inclusion criterion, fixed now: a benchmark is in scope if top-40
retrieval with the lightweight reader keeps < 70% of its accuracy when all visual information is removed (images and
descriptions). Measured so far: MemEye 39% (in), MemLens: authors report < 2% without evidence images (in),
Mem-Gallery ~100% (out), WorldMemArena 90% (out). Out-of-scope benchmarks are reported briefly in the appendix.
Candidates checked: Memory-QA (Meta, EMNLP 2025) -- each question searches only ~31 memories (RAG@40 already sees the
whole bank) and 93% of questions have one positive memory: no retrieval problem, not used. VisualMem (NeurIPS 2026,
viettmab/visualmem): 10 personas x ~171 events, one image per event, 696 multiple-choice questions whose evidence is
visual by design -> check the criterion first; if in scope, run the frozen unified method once and report whatever
the result: lightweight reader RAG@40 vs VGS (tau = 0.7, frozen), strong reader RAG@40 vs gated strong path, explicit
cache (persistent personas). Scoring: exact match of the option letter.
## P10-D1: ACL version (paper_acl/), unified gated method, WorldMemArena removed from the paper
- Correction to P9-D1: "MemEye keeps 39% without visual information" mixed readers (strong-reader no-visual 0.191 over
  lightweight full context 0.495). Correct: strong-reader full context 0.819 -> 0.191 without visual information (23%).
  Fixed in paper/ as well; the inclusion criterion of P10-D0 will be measured on VisualMem with the same reader pair.
- Unified gated method composed from stored outputs (scripts/p10_unified.py -> results/p10_unified.md), same tau = 0.7:
  lite VGS: MemEye 0.541 vs 0.488 (+0.053 [+0.009,+0.098], $6.76 uncached); MemLens-32k test 0.536 vs 0.466 (+0.071,
  registered); 64k 0.524 (+0.056, registered), compact 0.528 (+0.060, registered, $7.03).
  strong VGS (standard path = QSS + index when cached, QSS otherwise; widened path = RAG@40 with abstention deferral):
  MemEye 0.797 vs 0.749 (+0.048 [+0.015,+0.082], $6.21 vs $9.78, compact cached); MemLens-32k test 0.727 vs 0.689
  (+0.039 [+0.015,+0.062]; post hoc composition because the deferral was designed after the 32k test; the widened path
  alone gives 0.738). 64k strong VGS needs strong QSS on the 78 high-visibility questions (blocked, see below).
- Gemini API returns 400 FAILED_PRECONDITION "User location is not supported" since ~00:40 (network location of the
  machine); VisualMem runs and the 64k strong standard path are blocked until the network changes.
- Memory-QA rejected (P10-D0). VisualMem images are being fetched by HTTP range requests (scripts/p10_fetch_visualmem.py).
## P10-D2: writing study of ACL 2025 award papers and restructuring (no new experiments)
- API still blocked: Google rejects the egress IP (HostHatch VPS, Singapore, ip-api "hosting": true) with
  "User location is not supported"; VisualMem images are complete (1,950) and wait for the API.
- Studied the abstracts of the four ACL 2025 best papers (NSA, Language Models Resist Alignment, Fairness through
  Difference Awareness, A Theory of Response Sampling) and outstanding papers (BLT, Token Recycling, Llama See Llama Do,
  I0T), and the section structure of NSA, BLT, Token Recycling: no RQ labels, finding-style section titles, qualitative
  result claims in best-paper abstracts, a "rethinking" analysis section before the method (NSA).
- paper_acl: abstract rewritten (204 words, no result numbers), simpler wording in intro and Section 2 (claim-style
  paragraph titles), Results restructured into five finding-titled subsections with 3 tables (selection, main, ablation)
  and 3 figures (32x longer histories, visibility masking + gate quartiles, accuracy-cost); new figures in
  scripts/make_figures_acl.py. Main text ends on page 8.
## P10-D3: VisualMem protocol, step 1 (scope check; written before any run on VisualMem)
Lightweight reader RAG@40 with images vs with no visual information (read0), all 696 questions, exact-letter scoring.
In scope iff accuracy without visual information < 70% of the accuracy with images (criterion of P10-D0).
If in scope, step 2 runs the frozen unified method once (tau = 0.7, v_hat as defined, prompts and index unchanged):
lite VGS vs lite RAG@40 (primary V1: VGS > RAG@40), strong VGS (standard = QSS + cached index, widened = RAG@40 +
abstention deferral) vs strong RAG@40 (secondary V2: >=), explicit cache per persona. Results reported regardless.
## P10-D4: MemLens-64k strong standard path
Strong reader, QSS without index (single-use histories), on the 78 high-visibility 64k test questions, to complete
strong VGS at 64k with the same composition rule as at 32k.
## P10-D5: VisualMem scope check result and AMENDMENT of the criterion (before any run of our method on VisualMem)
Registered criterion (RAG@40 lite with vs without visual information): 0.454 vs 0.411 -> 91% retained -> out of scope.
Diagnostic: with the annotated evidence, the lite reader drops from 0.609 to 0.297 without visual information (49%;
chance 0.25), and RAG@40 contains the evidence round for only 45.4% of the questions. The registered criterion is
therefore flawed: it confounds visual dependence with retrieval failure (a retriever that misses the evidence image
cannot lose accuracy when images are removed). Amendment: visual dependence is measured on the annotated evidence
(in scope iff < 70% retained). VisualMem is in scope under the amended criterion. No condition of our method has been
run on VisualMem; the frozen unified method is now run once (lite: QSS standard path on all questions, widened path on
the low-visibility questions; strong: RAG@40 on all, QSS + cached index on the high-visibility questions, widened
path = RAG@40 + deferral), and the result is reported regardless (primary V1: lite VGS > lite RAG@40).
## P10-D6: VisualMem results (frozen unified method, run once; results/p10/vm_results.md)
- lite: RAG@40 0.454 -> VGS 0.569 (+0.115 [+0.075, +0.157]); V1 supported. QSS standard path for all: 0.568.
  low-visibility (415): both paths +0.17 (standard 0.573, widened 0.576); high-visibility (281): +0.025 (n.s.).
- strong: RAG@40 0.812 -> VGS 0.843 (+0.032 [+0.014, +0.050]); V2 supported (high-visibility: QSS + cached index
  0.929 vs 0.851; no abstentions to defer on multiple choice).
- selection: QSS finds the evidence round for 0.776 with 6.1 rounds; cosine of the same size 0.247; top-40 0.454.
  Oracle index visibility 0.769 (MemEye 0.95, MemLens 0.37).
- cost per 1k questions incl. cache creation and storage: lite VGS $13.70 vs RAG@40 $0.96; strong VGS $16.80 vs
  RAG@40 $8.79. The 200-character index has 99.1k tokens per persona and serves ~70 questions, so caching amortizes
  poorly: the strong-reader cost advantage of MemEye does not hold on 1,000-round histories.
- MemLens-64k strong VGS (P10-D4): 0.685 vs 0.655 (+0.030 [+0.004, +0.056]); high-visibility 78 q: QSS 0.603 vs
  RAG@40 + deferral 0.679.

# Phase 11 (user request): external 2026 memory systems as baselines
## P11-D1: protocol (written before the full runs)
Systems: SimpleMem (text; arXiv 2601.02553) and Omni-SimpleMem (multimodal; arXiv 2604.01007), code from
aiming-lab/SimpleMem (unified package 0.3.0, lancedb pinned to 0.25.3 as in its requirements); MIRIX to follow.
Same protocol for every system: memory construction with the system's own pipeline and gemini-3.8-flash (low thinking)
through the OpenAI-compatible Gemini endpoint; images described by the same one-sentence captions as our index
(Omni-SimpleMem also keeps the raw image for on-demand loading; its own VLM captioning is switched off because the
caption is supplied); retrieval = the system's own retrieval on the full question text; reading = the same readers,
system prompt and answer format as our methods (lite and strong); official judges. Omni-SimpleMem: SimpleMem-style
restatement extraction (window 40, as its LoCoMo runner), MiniLM embeddings, hybrid BM25, graph traversal, top-20,
no visual encoder (as its LoCoMo runner), no benchmark-specific category rules. SimpleMem: default retrieval
(planning + reflection), Qwen3-Embedding-0.6B. Benchmarks: MemEye (371), VisualMem (696), MemLens-32k test (466).
Results are reported whatever they are. Script: scripts/p11_baselines.py (.venv_bl).
## P11-D2: implementation notes (before any baseline result was used)
- SimpleMem: the released keyword search passes raw keywords to the tantivy parser, which rejects quotes, apostrophes
  and hyphens ("Syntax Error"); we strip non-alphanumeric characters before the full-text query (bug fix only) and
  restarted SimpleMem on MemEye from scratch.
- MIRIX (pip mirix==0.1.5): its default embedding model text-embedding-004 is no longer served -> gemini-embedding-001;
  MIRIX ignores MIRIX_DIR, so every history runs with its own HOME; MIRIX 0.1.5 fails on Gemini 3 responses
  (KeyError 'role' in its response converter), so its memory agents use gemini-2.5-flash, the strongest model in its
  supported list. Sessions are added one at a time with mirix.add(text, image_uris=raw images); retrieval uses
  extract_memory_for_system_prompt(question); the live "Current Time" line is removed from the context.
- First results (MemEye, all 371): SimpleMem lite 0.355 / strong 0.482; Omni-SimpleMem lite 0.285 / strong 0.315.
## P11-D3: MemLens subset for the external baselines (cost): 150 test questions, stratified by type, seed 0
(results/p11_memlens_sub150.json); RAG@40 and VGS are compared on the same subset. Chosen before any MemLens baseline run.
## P11-D4: Omni-SimpleMem model roles
With gemini-3.8-flash in every role, Omni-SimpleMem issues one thinking call per memory unit and needs >10 h for
VisualMem. Its default configuration uses a small model for per-unit summaries (summary_model gpt-4o-mini) and a large
one for queries and captions; we follow these roles (summary_model = gemini-2.5-flash-lite; extraction, query and
caption models = gemini-3.8-flash) and re-run Omni-SimpleMem on every benchmark, including MemEye (the earlier all-3.8
MemEye run is archived: lite 0.285, strong 0.315).
## P11-D5: incident -- MIRIX deleted dataset images (fixed, data restored)
MIRIX 0.1.5's send_message deletes uploaded image files by default (delete_after_upload=True). Our first MIRIX runs
passed dataset paths, so MIRIX deleted 426 MemEye canonical API copies (data/_api_images) and 204 VisualMem 1,024-pixel
copies. Originals were untouched. Both sets were regenerated with the original deterministic procedures
(scripts/prep_images.py; thumbnail 1,024 + JPEG q90); every regenerated file is byte-identical (all caption lookups hit
the content-hash cache, 0 new calls). The MIRIX runner now passes temporary copies and delete_after_upload=False.
Partial MIRIX builds (3 MemEye histories, all VisualMem personas) were discarded and are re-run; the 5 finished MemEye
histories are kept (their memories were complete before the deletion).
## P11-D6: Omni-SimpleMem adapter fixes found before using its results (all Omni results are re-run)
1. Disabling MPS by patching torch before transformers' lazy import broke sentence-transformers; Omni-SimpleMem then
   silently fell back to no embeddings (91% empty retrievals on MemLens). Those records are archived
   (results/baselines/_archive/omni_memlens_broken_embeddings.jsonl); the worker now resolves the imports first and
   fails loudly if the embedding model cannot load, and the build checks that the memory is searchable.
2. Omni-SimpleMem's query heuristics restrict questions that mention images to VISUAL units, while caption-only image
   units are stored as text, so such questions retrieved nothing; its own Mem-Gallery adapter bypasses this with a
   separate caption search. We drop the modality filter (rest of the strategy unchanged). Dates in questions are not
   turned into time filters on ingestion timestamps (time_range covers all units).
## P11-D7: run notes (cost accounting and stability)
- Thread mode (one process, shared models) replaced process mode for Omni-SimpleMem after memory pressure (each
  process ~3 GB with MPS); construction/query cost of thread-mode runs = the run's logged calls (fresh + replayed)
  divided by its questions (scripts/p11_fix_costs.py). MIRIX costs count only uninterrupted runs.
- Requests are capped at 8,192 output tokens when not cached (a few Omni extraction requests looped until timeout)
  and time out after 420 s; neither changes cached requests.
- Paper: Table 2 is now generated by scripts/p11_table.py (identical question sets incl. the 150-question MemLens
  subset); the registered full-test table moved to the appendix (tab:fulltest).
## P11-D8: MIRIX on MemLens -- image format fix
26 of the 150 MemLens histories contain .jpg files that hold PNG or WebP data. MIRIX declares the MIME type from the
file extension, Gemini rejects the mismatch (400 INVALID_ARGUMENT), and MIRIX's meta agent then fails or hangs. Our
other methods detect the real format, so this only penalised MIRIX. MIRIX now receives real JPEG copies (re-encoded
when needed); the 26 affected histories and one hung history are rebuilt from scratch (their earlier MIRIX records are
archived in results/baselines/_archive/mirix_memlens_mime_affected.jsonl). MemEye and VisualMem copies were already
correct (normalised during preprocessing).
## P11-D9: final status and spend
- MIRIX hung inside its agents on 2 MemEye histories (Home_Renovation, Multi-Scene; even after rebuilding with image
  copies) and 10 MemLens histories (results/p11_mirix_memlens_hung.json) despite request timeouts and a 15-minute
  watchdog per session; it is reported on the other 269 MemEye and 140 MemLens questions (dagger in Table 2).
  The 5 MemEye histories first built without images (MIRIX had deleted them) were rebuilt; 3 completed, the 2 above hung.
  About 2% of MIRIX image uploads failed with network errors (MIRIX does not retry).
- Final numbers: results/p11_baselines_ci.txt (generated by scripts/p11_table.py), paper Table 2 and appendix.
- Spend: phase 11 about $651 (total about $1,344); baseline construction/query $631, of which MIRIX on MemLens $488
  (about 41 calls and $0.25 per long session) -- far above the ~$70 estimate; cost was not capped during the run.

# Phase 12 (user request): three more 2026 baselines, $100 cap each
## P12-D1: selection and protocol (2026-11-08/09)
- Searched 2026 multimodal memory systems with code. Chosen: V-Mem (arXiv 2608.01543; raw rounds, modality-routed
  retrieval, zero-LLM build), MMA (arXiv 2602.16493; confidence-aware multimodal memory, implementation of the
  official MemEye framework, benchmark/mma), TeleMem (arXiv 2601.06037; Mem0-compatible extraction with semantic
  deduplication). Not run: M2A (2602.07624; per-turn ReAct memory manager, V-Mem reports 4.3B tokens on Mem-Gallery,
  far above the cap), EverMemOS (ACL 2026; the vendored copy in the MemEye framework lacks its tools package and the
  repository was rewritten as EverOS), MemOS 2.0 (needs a graph database server), WorldMM/M3-Agent (video).
- Hard cap: every fresh model call (construction, query, reading, judging) is charged to results/p12_budget/<system>
  and runs stop at $100 (scripts/p12_budget.py). Same protocol as phase 11: system's own memory and retrieval, our
  readers and judges, identical question sets (MemLens: the 150-question subset).
- Substitutions: SigLIP2 / CLIP (1.5 GB / 0.6 GB downloads at ~80 KB/s) -> gemini-embedding-2 as the joint
  text-image encoder (V-Mem's image lane serves only the 14 questions with a question image; MMA uses it for image
  entries and the text-to-image projection); V-Mem anchor LLM gemini-2.5-flash-lite (its paper uses gpt-4o-mini);
  TeleMem backbone gemini-2.5-flash-lite + gemini-embedding-001 (its default is local Qwen3-8B), English translations
  of its Chinese extraction prompts, a keyword bug fix in its buffer flush (similar_mem_text vs similary_mem_text),
  30 memories per scope; mem0 routes calls to OpenRouter whenever OPENROUTER_API_KEY is set, so it is unset for TeleMem.
- The OpenRouter key expired (401), so MemLens judgments (Qwen3-VL) are deferred: predictions are stored with
  score None and are judged by scripts/p12_judge_pending.py once a valid key is available.
## P12-D2: disk cleanup (user-approved list)
Removed: jevmem LongMemEval generation-* memory snapshots (45,208 dirs; answers, decisions, prompts, retrieval,
usage kept), jevmem/data/models and memory/pilot/hf (model weights), old venvs (jevmem/.venv*, memory/pilot/.venv),
~/.cache/uv and pip cache; in this project data/WorldMemArena, data/visualmem_data (originals; 1,024-px copies kept),
cache/crops, MIRIX/Omni/SimpleMem run stores, .venv_mirix, cache/mirix_requests.sqlite. Free space 354 -> 627 GB.
Kept by the user's choice: jevmem LLM caches and memory stores, ~/.cache/huggingface.
## P12-D3: final status
- All three systems complete on all question sets (MemEye 371, MemLens subset 150, VisualMem 696; both readers).
  Spend: V-Mem $10.97, MMA $7.15, TeleMem $10.75 (caps not reached). MemLens judged with the renewed OpenRouter key.
- TeleMem per-question construction cost recomputed from its ledger per benchmark (scripts/p12_fix_costs.py), because
  sharded processes shared one ledger.
- No memory system beats RAG@40 in any column; VGS is best in every column (results/p11_baselines_ci.txt).
- Cleanup of this project after the runs: .venv_bl, baseline run stores, gem/OpenAI response and embedding caches,
  cache/crops removed; cost logs copied to results/cost_logs/. Project 23.9 GB -> 3.8 GB; disk free 354 -> 638 GB.

# Phase 13 (user request): latency, standard metrics, compact multi-panel figures
## P13-D1: which standard metrics apply (decided before computing them)
- Answer level: token F1 and BLEU-1 (LoCoMo normalisation) apply to the open-ended benchmarks MemEye and MemLens; VisualMem
  is multiple choice, so its only answer metric is exact match (= its accuracy). Judge accuracy stays primary.
- Evidence level: Recall@10 and nDCG@10 (binary relevance, gold evidence rounds of each loader) apply only to methods that
  hand raw rounds to the reader, in the method's own order: RAG@40 (cosine rank), VGS (selector order; rounds of its
  lightweight-reader path), V-Mem and MMA. Text-memory systems (SimpleMem, Omni-SimpleMem, MIRIX, TeleMem) return derived
  records, so these metrics do not apply. Recomputed recall from the stored retrieved lists equals the stored coverage.
- V-Mem's retrieved rounds were not stored in phase 12, so its retrieval was re-run (P13_RETRIEVAL_ONLY=1). Its index
  contains image captions, and the gem response cache had been deleted at the user's request after phase 12, so the
  captions were regenerated and the re-run matches the stored number of rounds on only 94/371 MemEye questions. MMA
  indexes raw text and images and its re-run reproduces the stored retrieval (n_images identical).
## P13-D2: latency protocol
- Online latency per question = wall-clock from receiving the question to the reader's answer, memory already built,
  one question at a time; fixed samples: 30 questions of MemEye Home_Renovation_Interior_Design (120 rounds), 10
  MemLens-32k questions (random from the 150 subset, seed 0), 30 questions of VisualMem persona 0 (1,013 rounds).
  First question of a persistent history excluded (cache creation). Medians reported (about 3% of calls of every method
  stalled until the 240-s client timeout and were retried; means are dominated by these).
- Instrumentation: run_mm2 records t_select_s / t_read_s (+ warm flag); p11_baselines, p12_* and p11_mirix/p11_read_ctx
  record t_retrieve_s / t_read_s / build time (P13_LATENCY=1, P13_QIDS=<file>, outputs in results/p13/latency/).
  A selection or embedding request already issued by another condition is served from the local cache in 0 s; such
  records take the same question's fresh latency (scripts/p13_latency.py).
- MIRIX hung inside its agents again while building the MemLens and VisualMem samples (15-min watchdog per session;
  20 of 171 VisualMem sessions in 90 minutes), so its latency is measured on MemEye only (build 1,694 s for 120 rounds).
## P13-D3: paper layout
- Main text: four full-width rows of small panels (Fig. 2 problem, Fig. 3 longer histories + readings, which replaces
  the ablation table, Fig. 4 visibility, Fig. 5 cost + latency); Table 3 = standard metrics + latency.
- Appendix: 11 tables converted into 9 figures (keys/readers, scenario CIs, robustness, selection capability,
  development/registered test, MemLens types/gold definitions, compact index/zoom, VisualMem, baseline differences);
  kept as tables: text benchmarks, end-to-end development, gate split, cost, deployment variants, full held-out results.
- MemLens subset for external systems: decided for cost before any MemLens baseline ran (P11-D3); every external system
  (phases 11 and 12) ran only on the 150-question subset.
## P13-D4: results
- Standard metrics (results/p13/standard_metrics.json): F1 / BLEU-1 order the methods as the judge does; the only tie is
  the strong reader on MemEye (VGS F1 .511 vs RAG@40 .508, BLEU-1 .447 vs .448). nDCG@10 (light path): VGS best on MemEye
  (.829) and VisualMem (.656); on MemLens MMA's turn-level ranking is best (.808 vs VGS .723, RAG@40 .537), while MMA's
  answers are less accurate (0.473 vs 0.587 light). V-Mem re-run: same number of rounds as stored for 25/57/50% of the
  MemEye/MemLens/VisualMem questions (regenerated captions); MMA re-run: same number of images for 98/91/98%.
- Latency, light reader medians (s), MemEye / MemLens / VisualMem: RAG@40 5.4 / 16.5 / 15.7; VGS 6.3 / 14.1 / 12.7
  (selection 4.1 / 6.6 / 5.5); SimpleMem 22.0 / 22.6 / 23.8; Omni-SimpleMem 3.0 / 4.2 / 5.5; MIRIX 5.0 / - / -;
  V-Mem 3.6 / 2.8 / 2.4; MMA 2.3 / 5.7 / 4.1; TeleMem 2.0 / 3.9 / 1.8. Construction per history (s; MemLens median): ours 20 / 11 / 45;
  Omni 685 / 172 / 2,701; TeleMem 86 / 38 / 507; SimpleMem 158 / 39 / 808 (first runs); MMA 632 / 21 / 276 (fresh
  re-measurement, scripts/p13_build_time_mma.py); V-Mem 0.4 / 0.6 / 6 (local encoders); MIRIX 1,694 (MemEye).
- SimpleMem's first latency runs lacked its keyword index (tantivy and pylance missing in the reinstalled venv); its
  queries were re-run with the index and fresh query-time calls (P13_LATENCY bypasses the local caches at query time).

# Phase 14 (user request after an AI review): story, gate evidence, ablation, cost compression
## P14-D1: decisions
- Story: raw retrieval suffices for text memory (within 1 point of full context) but not for multimodal memory; VGS
  is the fix. Memory systems stay in Table 1 as references.
- Paper text: development/registration/amendment narratives removed (kept in README "Development protocol"); one
  neutral sentence in the setup says where tau and the prompts were set. Defensive baseline details removed.
- Strong reader: the abstention deferral (designed after seeing the MemLens-32k test split) is dropped; the strong
  widened path is plain RAG@40 (p11_analyze.STRONG_DEFER = False, p10_unified.DEFER = False). MemLens strong:
  VGS 0.687 vs RAG@40 0.689 (32k test), 0.648 vs 0.655 (64k), 0.767 vs 0.760 (150 subset): VGS matches retrieval.
  RAG@40 + deferral would be 0.738 (32k test) -- the earlier strong gain was the deferral.
## P14-D2: results (lightweight reader unless noted; "v2" = re-run with regenerated captions, caches had been deleted)
- Gate (results/p14/gate*.json): fixed paths vs gate vs message-length gate (threshold 165 chars from MemEye + MemLens
  dev) vs random routing with the same share. Lite v2: MemEye std .518 wid .506 gate .519 len .501; MemLens (v1) std
  .483 wid .536 gate .536 len .530; VisualMem v2 std .575 wid .592 gate .601 len .593 rand .585. Strong (v1): MemEye
  QSS .803 RAG .749 gate .797; MemLens QSS .618 RAG .689 gate .687. AUC of v_hat for "widened beats standard" within a
  benchmark: .46 / .55 / .61 -> the gate separates histories, it does not pick questions. tau in [0.6, 0.8]: within 1 pt.
- Ablation (results/p14/ablation.json, tab_ablation.tex): cosine top-k at the QSS size vs model ranking (QSS
  instruction, 100-round pool) at the same size vs QSS: FSR .527/.878/.916 (MemEye v2), .163/.356/.371 (MemLens),
  .260/.555/.555 (VisualMem v2); accuracy QSS - rank@k +.001/+.011/-.006 (n.s.), QSS - rank@8 +.030/+.028/-.019 (n.s.),
  QSS - cos@k +.050/+.067/+.141 (all significant). Reading the index matters; the set output mainly supplies the size.
- 100-round cosine pool: VisualMem pool recall .601, QSS .519 vs .575 whole index (VGS .527 vs .601); MemEye pool recall
  .992, no loss. Compact whole index on VisualMem: VGS .579 at $6.4/1k vs .601 at $11.7/1k (caches included).
- Masking question words (p = 1): MemEye FSR -.089, accuracy lead over RAG +4.5 -> -4.1 pts; MemLens v2 FSR +.032
  [+.006, +.058], acc -.006 (n.s.); VisualMem (150 q) FSR .000, acc -.007. Removing image descriptions from the
  VisualMem index: FSR .773 -> .433 (= RAG@40), acc -.093 [-.153, -.040]. Text-only MemEye index: FSR .906 (12.3 rounds).
- Full context: lite .495 / .520 (MemLens 150) / .454 (VisualMem) = no better than RAG; strong .819 / .740 / .967
  (VisualMem FC strong costs $154/1k: 217k input tokens per question, 7% implicitly cached).
- Replication with regenerated captions: MemEye lite VGS +.017 [-.023, +.057] (was +.053), VisualMem lite +.151 (was
  +.115), MemEye strong +.053 [+.018, +.090] (was +.048).
- Phase-14 API spend: $222 (of which full-context reading with the strong reader on VisualMem ~ $107).

# Phase 15 (user request): blanks in Table 10/11
- Table 10 (tab_std) had 24 not-applicable cells (Recall@10 / nDCG@10 for the four systems that return rewritten
  records) and 2 MIRIX latency cells. Re-laid out: answer metrics for all methods, evidence metrics only for the
  methods that return raw rounds; latency only in Table 11.
- MIRIX latency on MemLens (10 sampled histories) and VisualMem (persona 0): rebuilt with scripts/p15_mirix_driver.py
  (per-history process, restart after a hung add, resume from ADDED; sessions that hang twice are skipped).
- The user asked to keep environments and caches until they confirm the end of the project.
- Phase 15 layout (user request, following the table style of EMNLP 2026 main 1544): Table 1 with coloured benchmark
  headers, vertical rules between benchmarks, method groups separated by rules, VGS row highlighted, captions below
  tables with a bold lead; ablation (Table 3) in "w/o" form with cells shaded by the drop from the highlighted row
  (scripts/p14_tables.py wo; the FSR version moved to the appendix as tab:ablation_full); Figure 1 is a TikZ flowchart
  (paper_acl/sections/fig_overview.tex). Appendix tables follow the same style.
- MIRIX latency filled (Table 12, latency): MemLens 10 histories (130 sessions, a few hung once and resumed):
  median 5.4 s light / 6.3 s strong, retrieval 3.8 s, construction median 868 s per history; VisualMem persona 0
  (171 sessions, 10 hung once and resumed, none skipped): 4.9 s / 8.3 s, retrieval 3.1 s, construction 11,738 s
  (summed completed additions). Phase-15 MIRIX spend about $12.

# Phase 16 (user request): always prefilter? a cheaper index that is not "one summary line per round"
## P16-D1: decision on prefiltering (existing data, no new runs)
- Cosine prefilter for every history is not worth it: VisualMem 100-round pool holds the complete evidence for 60% of
  questions, VGS .601 -> .527 at $10.1 vs $11.7 (cached whole index). Cached tokens cost 1/10, so a cached compact index
  (~44k) costs about as much as an uncached 100-line pool. MemEye P=20/40 FSR .733/.833 vs .903 (P6-D1).
## P16-D2: image-anchored index (IAC), selection-only pilot (gemini-3.8-flash low, v2 captions)
- Layout (mm2.INDEX_MODE="iac", IAC_KW=12, IAC_B=20; run_mm2/select_eval --index iac): one header per session (handle
  s<k>, the session's own label, date, size, 12 rarest key words of its rounds without images), one line per image round
  (handle r<k>, 20 telegraphed user words, each image description). s<k> opens the whole session. Motivation: gold rounds
  are image rounds in 100% / 70% / 100% of cases (MemEye / MemLens / VisualMem) while text-only rounds are 48% / 80% /
  83% of rounds and most index tokens (VisualMem: user + assistant snippets 83%).
- FSR / opened rounds / selector input tokens:
  MemEye 371: full v2 .889 / 7.6 / ~11.3k; compact v2 .863 / 6.9 / 5.8k; IAC .871 / 9.0 / 4.9k.
  MemLens dev 233: full v2 .378 / 6.2 / 7.2k; compact v2 .395 / 5.8 / 3.2k; IAC .584 / 13.3 / 1.9k.
  VisualMem 150 (vm_mask_qids): full v2 .773 / 6.2 / ~99k; compact v2 .793 / 5.8 / ~44k; IAC .793 / 14.6 / 19.9k.
- Same IAC units, same per-question budget, similarity access: BM25 .412 / .202 / .347; cosine .542 / .270 / .327.
- End to end, lightweight reader, standard path: MemEye RAG@40 .502, full .518, IAC .530 (vs full +.012 [-.018,+.043]);
  VisualMem 150: RAG@40 .467, full .613, compact .573, IAC .580. MemLens dev e2e not run: the OpenRouter key expired (401).
- Projected cached VisualMem cost of the standard path: ~$4/1k with IAC vs $6.4 (compact) and $11.7-13.7 (full).
- Outputs: results/p16/sel/*.json (moved there because select_eval names collide with the v1 files, which were restored
  in results/mm2), results/p16/e2e/, offline analyses results/p16/{analyze,analyze_iac,compare}.py. Spend: $10.71.
## P16-D3: IAC end to end (dev data) and choice of the single paper variant
- IAC, lightweight reader: MemEye .530 (RAG@40 .502, lines .518); MemLens dev .481 / +notes .489 / IAC-wide(40,60)+notes
  .485 vs RAG@40 .489 and compact-600 lines + notes .532; VisualMem 150 .580 (kw24-b40 .567) vs lines .613. Strong:
  MemEye IAC + IAC context .802 (lines compact .799); MemLens dev IAC .631 vs RAG@40 .674. IAC opens whole sessions
  (VisualMem: always), 9-15 rounds. -> IAC is not kept: no gain on MemLens, -3 to -5 on VisualMem; it still needs
  text-level lines for widening. Its BM25/cosine-on-same-units control stays useful as an argument.
- Single variant chosen: the compact line index (no assistant snippet, short ids under session headers, telegraphed
  user text). v_hat on the compact lines (MemEye .870 / 90% >= .7; MemLens test .420 / 13%; VisualMem .580 / 27%), tau .7.
  Strong widened path = QSS rounds U cosine top-40 (+ compact index context on persistent histories): `tocu:40:40`.
- Test results (v2 captions; scripts/p16_compose.py; results/p16/final): lightweight VGS vs RAG@40: MemEye .532 vs .502
  (+.030 [-.015,+.077]), MemLens-32k 466 .509 vs .470 (+.039 [-.002,+.079]), 150 subset .547 vs .533, VisualMem .580 vs
  .450 (+.131 [+.091,+.171]). Strong: MemEye .799 vs .746 (+.053 [+.018,+.090]); MemLens 466 .691 vs .682 (+.009); 150
  .740 vs .727; VisualMem .920 vs .826 (+.093 [+.068,+.121]); with RAG@40 as the strong widened path VisualMem was .851
  while the standard path alone gave .925 -> the union makes widening safe (gate within .5 of the better path).
- Story analyses (results/p16/story_final.py): gold rounds beyond cosine rank 40: MemEye 13%, MemLens 14%, VisualMem 55%;
  VGS opens 85% / 15% / 71% of them. RAG@40 accuracy complete vs incomplete set: .521/.297, .576/.274, .563/.355; VGS gain
  over RAG@40 on questions where RAG@40 misses evidence: +.062 / +.049 / +.216 (elsewhere +.027 / +.033 / +.028).
- Spend: phase 16 so far $69.91 Gemini (this round $59.20) + ~$0.7 OpenRouter. New OpenRouter key installed in .env.
## P16-D4: the paper now uses the compact-index VGS only (sections_polished, main_polished.pdf)
- New runs (v2 descriptions): MemLens-64k (466; lite RAG/std/wid, strong RAG/std/union), ablations on the compact index
  (100-round pool std/wid for MemEye and VisualMem; ranking of the pool with the QSS instruction -> rank8 / rankqss;
  cosine at the QSS size), latency of the new VGS (P13 samples, sequential, fresh calls via gem.FRESH_TAGS=['p']).
- 64k: lite VGS .524 vs RAG .468 (+.056 [+.011,+.101]); strong .667 vs .655 (+.013 [-.013,+.039]).
- Ablation (lite): MemEye VGS .532 | w/o widened .526 | w/o whole index .502 | w/o standard .489 | pool QSS .494, cosine@k
  .467, rank8 .478, rank@k .508. MemLens .509 | .453 | .509 | .513 | .453, .382, .431, .453. VisualMem .580 | .569 | .532 |
  .579 | .522, .392, .532, .519. VisualMem 100-round pool holds the complete evidence for 60% of questions.
- Gate: within 0.5 of the better fixed path everywhere; tau in [0.6, 0.8] changes accuracy by <= 0.9 points; AUC .43-.63.
- Latency (median s, light / strong): MemEye 5.0 / 8.9, MemLens 8.8 / 8.6, VisualMem 9.7 / 14.0.
- MemEye scenario level: light +.030 [-.054,+.106] (4/8 positive), strong +.053 [-.011,+.137] (6/8).
- Generators: scripts/p16_compose.py, p16_paper.py [stats|figs|tables|ablation|extras], p16_ranks.py, p16_ids.py;
  make_figures_p13.py fig_problem now reads results/p16 (set selection = new QSS). Appendix sections tied to the old
  variant were removed (selector models, independent run, longer histories, Mem-Gallery, masking, gold definitions,
  deployment variants, full question sets). Spend this round $45.37 Gemini + ~$0.4 OpenRouter.
## P16-D5: image-description versions; lightweight-reader exploration (dev data)
- The first-generation image descriptions (used by the six memory systems in phases 11-12) cannot be recovered: they lived
  only in cache/gen.sqlite, deleted after phase 12; result files and logs (p11_*.log, results/baselines) hold questions and
  memory-system outputs, not the descriptions. Both generations use the same captioner and prompt; RAG@40 changes by at most
  1.4 points between them (MemEye .488/.502, MemLens-32k 466 .466/.470, VisualMem .454/.450). Setup text now says all systems
  use descriptions from the same captioner and prompt; memory systems are not rebuilt.
- Exploration (lite reader; MemLens dev 233 and MemEye 371; results/p16/explore): widened path = QSS U cosine top-20
  + notes (run_mm2 `tocu38:40:20|readn:medium`), and the lightweight reader with the compact index as context.
- Results: MemLens dev, widened = QSS U top-20 + notes .549 vs RAG .489 (+.060 [+.013,+.107]) vs current widened .532;
  on the MemLens-32k test it does not transfer: VGS .500 (+.030) vs current VGS .509 (+.039) at $6.09 vs $4.49 -> not
  adopted (64k and VisualMem runs stopped). MemEye: union-20 widened .486, lightweight reader with the index as context
  .520 (standard path .526) -> neither adopted. The paper keeps the P16-D4 configuration.
## P16-D6 (in progress): revision after the AI review (not a real submission)
- Done: main text reports point estimates (Table 2 marks gains whose interval excludes zero with *); appendix keeps
  scenario-level intervals without scenario counts; abstract/contributions now say "lightweight on all three, strong on
  two, significant on VisualMem and MemLens-64k".
- Reader prompt leak removed (src/mm.py IMAGE_RULE["memeye"] no longer quotes 'the provisional busy-day trend'); MemEye
  re-reads of every paper condition run into results/p16/final2/memeye (lite: rag, std, wid, fc, pool std/wid, controls;
  strong: rag, std, union, fc). Point p16_compose SRC (memeye) and p16_ids/ablation at final2 once STATUS_lite/strong exist.
- Running: MemLens dev compact standard path (results/p16/explore/memlens, for the learned router).
- Plan (user approved "按你说的先改"): (1) drop MIRIX; rebuild SimpleMem, Omni-SimpleMem, TeleMem, V-Mem, MMA with the v2
  descriptions on all three benchmarks, MemLens on all 466 test questions, V-Mem/MMA at budget 40, cap <= $100 per system;
  add hybrid BM25+dense, multi-query rewriting, LLM rerank over top-100, two-step retrieval baselines. (2) gate: history-
  level visibility (question-independent, write time), calibration, learned router (logistic, trained on MemEye + MemLens
  dev), embedding-based visibility proxy; VisualMem and MemLens test are held out for tau. (3) cross-swap of the reading
  paths (strong reader on the notes path, lightweight reader on the union path). (4) open-weight selector/reader check.
  (5) RAG@k curve k = 10/20/30/40 in the appendix (keep RAG@40 in the main table). (6) reframe novelty: diagnostic finding,
  global evidence navigation, write-time visibility adaptation.
- MemEye re-read without the leaking example (results/p16/final2/memeye, wired into p16_compose/p16_paper): lite RAG .493,
  standard .543, widened .476, VGS .549 (+.056 [+.005,+.106], now significant); strong RAG .764, standard .793, union .819,
  VGS .799 (+.034 [.000,+.070]); full context lite .500 ($0.37), strong .810 ($21.9, uncached). Tables 1/2/4, Fig. 4, routing
  table regenerated. TODO: put FC .500/.810 into the main table, then update text numbers that cite MemEye (sec. 5.1, 5.2
  story numbers, 5.3 ablation, 5.5 cost, intro gap 7.3 -> 4.6, contributions/abstract significance list).
- User (2026-11-20): do not narrate the V-Mem/MMA budget change in the paper (just list the budget used); extra retrieval
  baselines only if cheap or already available. Note to self: report every baseline that is run, not only weaker ones.
- 2026-11-20: user approved all remaining items. Launched cross-swap runs (strong reader on the notes path `toc38:40|readn:low`
  --snippet 600; lightweight reader on the union path `tocu38:40:40|read:medium`) and lightweight RAG@10/20/30 curves on all
  three benchmarks -> results/p16/swap/<bench>/ (STATUS_me/ml/vm). Next: baseline rebuild (no MIRIX; five systems, v2
  descriptions, MemLens 466, V-Mem/MMA budget 40 listed only in the system table), cheap extra retrieval baselines (hybrid
  BM25+dense offline; LLM rerank of top-100 from the ranking control), open-weight check, history-level gate write-up.
- 2026-11-20: .venv_bl reinstalled (uv cpython 3.12 base; baselines/SimpleMem requirements + editable package, lancedb
  0.25.3, google-genai 2.33.0, pylance, tantivy, telemem 1.10.1; pip --uploaded-prior-to 2026-11-10); imports OK.
  Cross-swap and lightweight RAG@10/20/30 runs finished (results/p16/swap/<bench>/, not yet analysed). Next: analyse swap
  and RAG@k; rebuild the five memory systems (scripts/p11_baselines.py simplemem/omni, scripts/p12_telemem/vmem/mma) with
  the v2 descriptions, MemLens on all 466 test questions, V-Mem/MMA budget 40; drop MIRIX from the paper.
- Cross-swap (results/p16/swap): strong reader on the notes path vs its union path: MemEye .754 vs .819, MemLens-32k
  .667 vs .693, VisualMem .862 vs .921; lightweight reader on the union path vs its notes path: MemEye .493 vs .476,
  MemLens .470 vs .513, VisualMem .519 vs .579 -> swapping hurts in 5 of 6 cases (reader-specific widening justified).
- Lightweight RAG@k (k = 10/20/30/40): MemLens-32k .442/.485/.459/.470; VisualMem .405/.438/.438/.450 (RAG@40 stays the
  strongest or near-strongest budget; MemEye curve pending).
- MemEye lightweight RAG@10/20/30: .449/.475/.479 (RAG@40 .493 with the leak-free prompt).
## P16-D7: baseline rebuild set-up; honest reading of the MemEye prompt change (2026-11-20)
- MemEye prompt change (leak-free reader prompt, results/p16/final2): the one-sentence edit changed 145-216 of 371 answers
  and 48-60 scores per condition, in both directions. The two questions whose reference contains the removed phrase
  (Personal_Health_Dashboard_Assistant:Q1, Q3) explain only -0.34 pt (lite RAG@40: 0.25->0, 1->0) and -0.27 pt (lite
  standard path: 0->0, 1->0). So RAG .502->.493 and VGS .532->.549 are mostly prompt sensitivity, NOT the leak; the
  lightweight MemEye gain moved +3.0 -> +5.6 because of a one-sentence prompt edit, i.e. run-to-run variation is as large
  as the effect. Correct the earlier explanation given to the user; do not present the new significance as robust.
  (Also: p14 lite RAG@40 had 7 retrieved sets that differ from the strong file; final2 lite/strong retrieved lists agree.)
- Baseline environment: .venv_bl works for all five systems (smoke test on 2 MemLens questions, results/p16/smoke).
- Code changes: P12_LEDGER_DIR (p12_budget), P16_BASELINE_DIR / P16_MEMLENS_QIDS / P16_SHARD (p12_common), P16_VMEM_TOPK
  (p12_vmem; top-k rounds before the +/-1 windows), P16_MMA_TOPK (p12_mma; entries), ledger + cap for p11_baselines.py.
  scripts/p16_bl_env.sh holds the settings: V-Mem top-20 rounds (+/-1 neighbours; 33-43 rounds read in the smoke test),
  MMA 80 entries (= turns of 40 rounds), MemLens = all 466 test questions, v2 descriptions from the cache, cap $60 per
  (system, benchmark), outputs results/p16/baselines/<bench>/<system>.jsonl (shards in results/p16/bl_shards/<k>),
  ledgers results/p16/budget/<bench>/<system>.jsonl. Costs for the table: mean usd_read + ledger(build+query)/n questions.
- Paper: MIRIX is dropped; the systems table lists the budgets actually used (no narrative about changing them; the text
  must not say "default settings" for these budgets).
- Omni-SimpleMem thread-mode bug (found 2026-11-20 before using any result): with 8 episodes in threads, every
  EmbeddingService loaded its own SentenceTransformer concurrently; torch raised "Cannot copy out of meta tensor" and
  Omni-SimpleMem silently skipped those text units (2,561 skipped units in 58 histories). That partial run is discarded
  (results/p16/bl_discarded/). Fix in scripts/p11_baselines.py: all services share the model loaded in _init_worker,
  embedding calls are serialised, and an empty embedding raises. 6-thread smoke test: 0 warnings, 0 errors.
  Always grep the baseline logs for "embedding empty" / "ERROR" before using a baseline result.
## P16-D8: baseline rebuild running; retrieval baselines; history-level visibility (2026-11-20/21)
- Baseline rebuild launched with scripts/p16_run_baselines.sh <system> <bench> <n> (chains memlens -> memeye -> visualmem per
  system; STATUS files in results/p16/bl_logs/STATUS_<system>_<bench>; logs there too). SimpleMem needed a fix: it loaded a new
  Qwen3-Embedding-0.6B per history and never released it (10+ GB per worker); the adapter now loads it once per worker
  process (2 processes, ~5.6 GB each). Machine: 24 GB RAM -> do not start more memory-heavy jobs while SimpleMem runs.
- Retrieval baselines at the RAG@40 budget (scripts/p16_retrieval_ids.py -> results/p16/ids/<bench>_{hybrid40,rerank40}.json;
  readings results/p16/final/<bench>/ret_<name>_lite.jsonl): FSR dense@40 / BM25@40 / hybrid(RRF)@40 / dense-100->LLM
  rerank@40: MemEye .914/.906/.922/.976; MemLens .648/.590/.642/.856; VisualMem .454/.328/.481/.586.
  Lightweight reader: MemEye hybrid .463, rerank .474 (RAG@40 .493, VGS .549); MemLens hybrid .459, rerank .446 (RAG@40
  .470, VGS .509); VisualMem pending. More complete 40-round sets do not help the lightweight reader.
- Full context, strong reader, MemLens-32k: remaining 316 test questions .668 ($24.9/1k) (results/p16/final/memlens_fc);
  with the 150-question subset (.740) this gives the full 466.
- History-level visibility (scripts/p16_hvis.py -> results/p16/hvis_<bench>.json; H = mean over rounds of the share of a
  round's content words kept by its compact index line): MemEye .692 (histories .552-.849), MemLens dev .278 (.227-.353),
  MemLens test .276 (.223-.346), MemLens-64k .272, VisualMem .398 (.390-.409). Any threshold in (.353, .552) separates the
  development data (MemEye -> standard, MemLens dev -> widened); midpoint 0.45. VisualMem then goes to the widened path.
- Observation: for the strong reader the union path is >= the standard path on all three benchmarks (.819/.693/.921 vs
  .793/.584/.925) but costs about twice as much on MemEye; keep the gate for both readers (cost argument), report this.
## P16-D9: decisions for the revision (2026-11-21, while the baseline rebuild runs) -- READ THIS FIRST AFTER A CONTEXT RESET
State of long jobs: baseline chains (scripts/p16_run_baselines.sh; STATUS files in results/p16/bl_logs/): vmem memlens DONE,
memeye/visualmem following; mma, telemem, omni (8 threads), simplemem (2 procs, ~3 h for MemLens, then memeye, visualmem;
total ~7-8 h). Check `ls results/p16/bl_logs | grep STATUS`, error counts in results/p16/baselines/*/*.jsonl and
results/p16/bl_shards/*/*/*.jsonl (vmem/telemem shards are concatenated by the runner when a benchmark finishes), and
`grep -c "Text embedding empty"` in the omni logs (must be 0).
Decisions:
1. Gate -> history-level, write-time visibility: H(h) = mean over rounds of |W(index line) & W(round)| / |W(round)| (round =
   user + assistant + image descriptions; compact 200-char line). tau_h = 0.45 = midpoint of the development gap (MemLens dev
   max .353, MemEye min .552). Routing: MemEye -> standard; MemLens (32k/64k), VisualMem (.39-.41) -> widened. Implement in
   scripts/p16_compose.py (P16_GATE=history|question; default history), regenerate everything offline. The per-question
   estimate v_hat (old gate, tau .7) stays as an analysis row (AUC .47-.63; learned router; length gate; oracle).
   Expected VGS (history gate): lite .543/.513/.579 (64k .526); strong .793/.693/.921 (64k .674).
   Figure 4: (c) add pooled bars; (d) replace tau sensitivity by "gain from widening vs history visibility" (calibration).
2. Baselines: MIRIX dropped; five systems rebuilt (v2 descriptions, leak-free prompt, MemLens 466, V-Mem top-20 rounds
   +/-1 neighbours, MMA 80 entries). Paper lists budgets in the systems table only; no "default settings" wording for them.
   Costs: (ledger build+query)/n + mean read cost. Main table moves to MemLens-32k (466): RAG lite .470 / strong .682; FC lite
   .446 (results/mm2/memlens_32k/fc lite, 466 test) / strong .691 (150 from p14 memlens_fc + 316 from results/p16/final/memlens_fc).
3. Retrieval baselines: hybrid40 (lite only, text mention) and dense-100 -> LLM rerank@40 (lite done; strong to run, ~$16).
   Report all results, also if a baseline beats VGS.
4. Cross-swap table and RAG@k curve go to the appendix; setup must not claim k=40 is the best budget for both readers
   (lite MemLens: k=20 .485 > k=40 .470).
5. Honesty items: (a) MemEye lite significance after the prompt edit is fragile (see P16-D7); add a "run-to-run variation"
   paragraph (two reader-instruction variants on MemEye; 2 extra selection draws: MemEye standard path, MemLens-32k widened
   path, lite). (b) setup claims a second judge on MemEye but the appendix paragraph was deleted -> re-run the Qwen3-VL judge
   on the new MemEye outputs and restore it. (c) pooled estimates (MemEye + MemLens-32k; all three) with intervals.
6. Figure 1 (RAG@30 vs full context, strong reader): MemEye bar must be re-run with the leak-free prompt (FC = .810);
   MemLens bar moves to 466 questions (FC .691; RAG@30 strong on 466 to run); VisualMem bar unchanged.
7. Still to do after the above: V-Mem/MMA latency re-measurement with the new budgets (P13_LATENCY=1, P13_QIDS, sequential);
   open-weight check (Qwen3-235B selector via OpenRouter on the compact index; optionally Qwen3-VL reader);
   paper rewrite (method 3.2, setup, results 5.1-5.5, appendix tables, limitations), compile, visual check; final audit.
Budget: Gemini total was 1782.10 at the start of this stretch; quoted ceiling ~1980. At 00:20 on 11-21: 1828.
- 00:45 11-21 status: history gate implemented in scripts/p16_compose.py (GATE="history", TAU_H=0.45, hvis files in
  results/p16/hvis_*.json; qgate() = per-question v_hat router for analysis). With it: lite VGS .543/.513/.579 (64k .526),
  strong .793/.693/.921 (64k .674); lite gains +5.0 [0.0,10.0], +4.3 [0.0,8.6], +12.9; borderline ones need exact p-values
  and a pooled estimate. p16_paper.py still has to be adapted (stats/figs/tables/ablation/extras call P.gate(V, ...) or use V).
- Per-history calibration (lite, widened - standard): MemEye histories (H .55-.85): negative in 7 of 8; VisualMem personas
  (H .39-.41): mean +.010; MemLens quintiles (H .25-.30): +.03 to +.10. Strong reader (union - standard): MemLens +.03 to
  +.15, MemEye mixed (+.026 mean), VisualMem ~0.
- Retrieval baselines, lightweight reader, all three benchmarks done: hybrid@40 .463/.459/.457, rerank@40 .474/.446/.499
  (RAG@40 .493/.470/.450; VGS .543/.513/.579). Strong-reader rerank@40 running (chain X, results/p16/final/<bench>/
  ret_rerank40_strong.jsonl), then Figure-1 RAG@30 strong re-runs (results/p16/fig1/<bench>_rag30_strong.jsonl; ids from the
  v2 RAG@40 rankings; MemLens on 466). Chain Y: selection draws 2 and 3 (results/p16/var/{memeye,memlens}/..__draw{2,3}.jsonl).
- Baselines so far (new settings; n complete unless noted): V-Mem lite/strong MemEye .469/.730, MemLens-466 .461/.665,
  VisualMem .427/.809; MMA MemEye .414/.761, MemLens .442/.629, VisualMem (261 of 696) .372/.862 -> MMA strong may beat
  RAG@40 (.826) on VisualMem: the sentence "RAG outperforms all memory systems in every column" must be checked and
  rewritten if false. Ledgers so far $76.5. Gemini total 1871.6 at 00:40.
- 00:50 11-21 results: (a) rerank@40 (dense top-100 -> LLM rerank, 40 rounds), STRONG reader, MemEye: .790 vs RAG@40 .764
  (+.026 [-.002,+.056]) vs VGS .793 (standard) / union path .819 -> on MemEye the reranker ties VGS for the strong reader;
  must be reported as is. MemLens / VisualMem strong rerank pending (chain X).
  (b) Selection draws, lightweight reader: MemEye standard path .543 / .527 / .518 (mean .529; RAG@40 .493 -> +3.6 mean,
  range +2.5 to +5.0); MemLens-32k widened path .513 / .519 / .517 (mean .516; RAG@40 .470 -> +4.6). The paper's MemEye
  lite number is the first draw (the highest): report the three-draw mean and range in the robustness paragraph and do
  not call the MemEye lite gain significant without this caveat. Files: results/p16/var/{memeye,memlens}/*__draw{2,3}.jsonl.
- TeleMem shard 0 on MemLens had written all records but kept spinning at exit (mem0 worker thread); killed after checking 932/932 records; p12_telemem.py now calls os._exit(0) after main().
- 01:05 11-21 launched: (A1) Qwen3-VL second judge on the six final2 MemEye files (field score_qwen_qwen3_vl_235b_a22b_instruct;
  STATUS_judge2); (A2) MemEye strong-reader diagnostics with the leak-free prompt for Section 2 / Table 5: rag:10, oracle,
  fc|read0 (no visual information) -> results/p16/sec2 (STATUS_sec2); (B) open-weight selector toc@qwen (Qwen3-235B, OpenRouter)
  on the compact index + lightweight reader: MemEye standard path, MemLens-32k and VisualMem widened path -> results/p16/open
  (STATUS_open). Plan for paper consistency: Fig. 3a/3b and Table 5 MemEye rows use only leak-free runs (FC .810, RAG@40 .764,
  RAG@10 / oracle / no-visual from sec2); drop Table 5 rows without a re-run (high resolution, dense captions) and Fig. 6b.
- 01:10 launched V-Mem/MMA latency re-measurement at the new budgets (results/p16/latency_bl/<bench>/{vmem,mma}.jsonl; P13 samples; sequential; fresh reader calls; STATUS file there).
## P16-D10: two contamination problems in the Omni-SimpleMem / SimpleMem MemLens rebuild (found 01:20 11-21, before use)
1. Omni-SimpleMem APPENDS to an existing data_dir (its MAU store loads old files). 77 MemLens run dirs existed before the
   fixed run started (2 smoke questions, the discarded meta-tensor run, the 6-thread smoke): list in
   results/p16/omni_memlens_reused_dirs.json. Their memories hold duplicated units -> records of these 77 questions must be
   removed, their run dirs deleted, and the episodes re-run (extraction calls are cached; cheap).
2. SimpleMem: at the third restart (01:11) two old pool workers survived the kill (RSS filter missed them), kept building
   q_9bad18be and q_cc4da426 while the new run built the same episodes in the same directories -> lance "Not found" errors,
   q_9bad18be has n_items = 0. Both must be re-run from clean dirs; also re-run any episode whose run dir is older than the
   run that produced its records (in progress at an earlier kill).
Fixes: scripts/p11_baselines.py wipes the run dir before every build (new processes only); fix-up after each affected stage
with scripts/p16_fix_reused.py (removes records + dirs) and a re-run of the stage (resumable runner). Do NOT edit
scripts/p16_run_baselines.sh while chains run (bash reads scripts incrementally).
MemEye / VisualMem run dirs did not exist before (baselines/runs had only mirix dirs), so those stages start clean.
## P16-D11: results in (01:25 11-21) -- retrieval baselines (strong), open-weight selector, Section-2 diagnostics
- dense top-100 -> LLM rerank @40, STRONG reader (results/p16/final/<bench>/ret_rerank40_strong.jsonl, 0 errors):
  MemEye .790 (RAG@40 .764; VGS .793; VGS - rerank +.003 [-.027,+.033]); MemLens-32k 466 .674 (RAG .682; VGS .693; +.019
  [-.013,+.052]); VisualMem .861 (RAG .826; VGS .921; +.060 [+.037,+.085]). Cost $/1k read only: 10.5 / 17.7 / 9.9 (+ the
  ranking call, about $5.2/1k on MemEye). Lightweight reader (P16-D8): rerank .474/.446/.499, hybrid .463/.459/.457.
  => An LLM reranker over 100 candidates matches VGS for the strong reader on MemEye and MemLens; VGS is clearly ahead only
  on VisualMem (long histories, evidence outside the pool) and for the lightweight reader. Report exactly this.
- Open-weight selector Qwen3-235B (OpenRouter, same prompt, compact index) + lightweight reader, path of the history gate
  (results/p16/open/<bench>/): MemEye .477 (RAG .493, Gemini-selector VGS .543), FSR .598 vs .860, 4.8 vs 7.2 rounds;
  MemLens-32k .464 (RAG .470, VGS .513), FSR .176 vs .270, 2.4 vs 4.2 rounds; VisualMem pending. => with this open-weight
  selector VGS does not beat RAG@40: the selector under-selects. Limitation to state plainly.
- MemEye strong reader, leak-free prompt (results/p16/sec2): annotated evidence .780 (2.8 rounds), RAG@10 .670;
  full context without visual information pending. With FC .810 and RAG@40 .764: RAG@40 - FC = -.046, RAG@10 - FC = -.140.
- 01:35 11-21: scripts/p16_paper.py stats/figs now use the history gate (perm_p = paired sign-flip test; calibration();
  r["qgate"] = per-question router; Fig. 4d = history visibility vs widened - standard). History-gate VGS vs RAG@40:
  MemEye lite .543 vs .493 (+5.0 [0.0,+10.0], p=.055), strong .793 vs .764 (+2.9, p=.12); MemLens-32k lite .513 vs .470
  (+4.3 [0.0,+8.6], p=.058), strong .693 vs .682 (+1.1); MemLens-64k lite .526 vs .468 (+5.8, p=.014), strong .674 vs
  .655 (+1.9, p=.16); VisualMem lite .579 vs .450 (+12.9), strong .921 vs .826 (+9.5). Costs/1k (incl. caches): MemEye 1.6 /
  5.5, MemLens 4.6 / 13.8, VisualMem 6.6 / 23.1. TODO in p16_paper.py: tables() (main table on MemLens-466 with the rebuilt
  baselines from results/p16/baselines, no MIRIX; FC lite .446 / strong .691 on 466), ablation() (P.gate(V,..) -> routes),
  extras() (cost figure, std metrics, latency incl. new V-Mem/MMA), new appendix tables (cross-swap, RAG@k, retrieval
  baselines, pooled estimate, selection draws, open-weight selector, second judge).
## P16-D12: state at 01:45 11-21 and the paper plan (READ FIRST AFTER A CONTEXT RESET)
New generator: scripts/p16_final.py (check | main ...): cells(bench, reader) on FULL question sets (MemEye 371, MemLens-32k 466,
VisualMem 696) for rag, rerank (dense-100 -> LLM rerank@40, cost includes the ranking call), fc, five rebuilt systems
(results/p16/baselines; cost = mean usd_read + ledger(build+query)/n), vgs (history gate, + cache overhead). `check` prints
what is still missing. main_table() writes sections_polished/tab_main.tex (no MIRIX; row "+ LLM rerank"; * = permutation p<.05).
Numbers so far (lite / strong): RAG@40 ME .493/.764, ML .470/.682, VM .450/.826; rerank ME .474/.790, ML .446/.674, VM
.499/.861; FC ME .500/.810, ML .446/.691, VM .454/.967; TeleMem ME .339/.398, ML .200/.159; V-Mem ME .469/.730, ML .461/.665,
VM .427/.809; MMA ME .414/.761, ML .442/.629, VM .407/.868 (MMA strong > RAG@40 on VisualMem); VGS ME .543/.793, ML .513/.693,
VM .579/.921. SimpleMem / Omni pending on all three, TeleMem pending on VisualMem.
Pending jobs: omni chain (memlens -> memeye -> visualmem), simplemem chain (same; 6 threads), telemem visualmem; chain X
(Fig.-1 RAG@30 strong: results/p16/fig1/<bench>_rag30_strong.jsonl); second judge (results/p16/logs/STATUS_judge2);
sec2 fc|read0 (STATUS_sec2); open-weight selector on VisualMem (STATUS_open); V-Mem/MMA latency (results/p16/latency_bl/STATUS).
AFTER omni memlens finishes: `.venv/bin/python scripts/p16_fix_reused.py omni memlens results/p16/omni_memlens_reused_dirs.json`
then `scripts/p16_run_baselines.sh omni memlens 4`. AFTER simplemem memlens finishes: same with
results/p16/simplemem_memlens_rerun.json and `scripts/p16_run_baselines.sh simplemem memlens 3`. Then check n_items == 0 records,
"Error during" (simplemem log) and "Text embedding empty" (omni logs) again.
Honest narrative for the paper (decided): (1) VGS helps the lightweight reader on all three benchmarks against RAG@40, reranked
RAG, full context and every memory system (+5.0, +4.3, +12.9; individually p = .055, .058, <.001; pooled ME+ML to compute);
(2) for the strong reader the clear gain is on VisualMem (+9.5; long histories, evidence outside the similarity pool); on
MemEye/MemLens VGS matches an LLM reranker over 100 dense candidates and RAG@40 within noise, at lower cost on MemEye;
(3) gate = history-level visibility (write time), tau_h .45; (4) limitations to state: open-weight selector (Qwen3-235B)
under-selects -> no gain over RAG; selection-draw variation on MemEye lite (.518-.543, mean .529); prompt sensitivity;
MMA strong beats RAG@40 on VisualMem. Remove "RAG outperforms all memory systems in every column".
Paper items still to write: abstract/intro (gate wording, numbers), method 3.2 (history visibility), setup (MemLens 466, five
systems, permutation test, no "strongest budget" claim), results 5.1-5.5, limitations, appendix (robustness: draws, prompt,
second judge, scenario-level, pooled; routing table with per-question estimate; cross-swap; retrieval baselines; RAG@k;
open-weight selector; systems table without MIRIX; std metrics; latency), Fig. 1 (new RAG@30 strong; MemLens 466), Fig. 3a/b and
Table 5 MemEye rows from leak-free runs (FC .810, RAG@40 .764, RAG@10 .670, annotated evidence .780, no-visual pending),
drop Table 5 rows without re-runs and Fig. 6.
Spend: Gemini 1931.7 at 01:25 (stretch start 1782.1 -> +$150; projected ~+$220 at the end = about 10% above the $150-200 quoted).
- 01:50 11-21 results: second judge Qwen3-VL on MemEye (leak-free outputs; field score_qwen_qwen3_vl_235b_a22b_instruct in
  results/p16/final2/memeye/*.jsonl): lite RAG@40 .456, VGS standard .510, FC .466; strong RAG@40 .710, VGS standard .738, FC
  .760; exact agreement with the Gemini judge .77-.84; same ordering. MemEye strong diagnostics (results/p16/sec2): full
  context without visual information .216, annotated evidence .780 (2.8 rounds), RAG@10 .670.
## P16-D13: pooled estimates, three-draw means, Section-2 numbers (02:00 11-21; history gate; results/p16/draws.json)
- Pooled paired differences VGS - RAG@40 (permutation p): MemEye + MemLens-32k lite +.046 [+.013,+.079] p=.007; strong
  +.019 [-.003,+.040] p=.09. All three: lite +.084 [+.058,+.109]; strong +.053 [+.037,+.070] (both p<.001).
- Three selection draws averaged per question (lightweight reader): MemEye +.036 [-.008,+.080] p=.11 (draws .543/.527/.518);
  MemLens-32k +.046 [+.006,+.087] p=.025 (draws .513/.519/.517); pooled +.042 [+.012,+.072] p=.006.
  => Claim for the paper: on MemEye and MemLens the lightweight gain is 4-5 points, significant when pooled (p<.01) and for
  MemLens-64k alone; MemEye alone is not significant (and its table number is the best of three draws).
- Section 2 (MemEye, strong reader, leak-free prompt): RAG@10 - FC = -.140 [-.184,-.098]; RAG@40 - FC = -.046 [-.075,-.018];
  when top-10 holds the complete evidence (242 q) the gap to FC is .000; when it does not (129 q) it is .403. FC without
  visual information .216. (Replaces the old 3.4 / 29.6 points and -.079 / -.130 in section 2, Fig. 3a and Table 5.)
- scripts/p16_final.py now also has: appendix() -> tab_gate.tex (history gate + per-question routers), tab_swap.tex,
  tab_retrieval.tex (RAG@k, hybrid, rerank), tab_std.tex (all systems, full sets), tab_systems.tex (no MIRIX; V-Mem 20
  rounds + neighbours, MMA 80 turns); figures() -> cost.pdf (uses latency() -> results/p16/latency_final.json and the old
  p13 latency for the other systems) and app_baselines.pdf; latency() also rewrites the V-Mem / MMA / VGS rows of
  tab_latency.tex and drops the MIRIX row. Run `p16_final.py check` first; `all` once every cell is present.
  Paper text edits (method 3.2, setup, results, appendix, limitations, Fig. 1) are NOT done yet as of this entry.
- 02:10 11-21 paper text progress (sections_polished; backup of all .tex in /tmp/polished_backup_20261116/before_final):
  DONE: 03_method.tex section 3.2 (history visibility H(h), Eq. eq:vis, tau = 0.45, pointer to app:vis for the two widened
  designs); 04_setup.tex (tau = midpoint .35/.55 on dev data; "Baselines and controls" with LLM rerank and app:retrieval;
  five memory systems; permutation test). New labels needed in the appendix: app:retrieval (RAG@k / hybrid / rerank table),
  tab:swap inside app:vis. TODO: 00_abstract, 01_intro (gate wording + numbers), 02_problem (MemEye numbers: 3.4/29.6 ->
  0.0/40.3; fig caption), 05_results (all subsections), 08_limitations, 09_appendix, fig_overview.tex (gate box text),
  Fig. 1 data, Fig. 3 (problem.pdf panel a/b MemEye values), Table 5 (tab:prelim) MemEye rows.
- 02:20 11-21 numbers for the results text (history gate; results/p16/paper_stats.json): story: gold rounds beyond rank 40:
  ME 13.0%, ML 13.5%, VM 54.7%; VGS opens 85% / 14% / 72% of them; RAG@40 lite accuracy complete vs incomplete set .510/.320,
  .576/.274, .563/.355; VGS gain on RAG-incomplete questions +6.2 / +6.7 / +22.1 vs +4.9 / +3.0 / +1.9 elsewhere.
  Costs $/1k incl. caches: VGS lite 1.6 / 4.6 / 6.6, strong 5.5 / 13.8 / 23.1; RAG 0.9/9.7, 1.7/14.0, 1.0/8.8.
  Latency medians (s; lite / strong): VGS ME 5.0/8.9, ML 8.8/8.6, VM 11.0/14.0 (history-gate path); RAG 5.4/8.8, 16.5/9.9,
  15.7/8.4; V-Mem new budget ME 3.2/6.1, VM 1.9/5.7; MMA ME 5.5/10.4 (ML/VM pending; results/p16/latency_bl).
  Fig. 1 RAG@30 strong (leak-free / v2 rankings): MemEye .750 (FC .810, gap 6.0), MemLens-466 .652 (FC .691, gap 3.9),
  VisualMem pending (old .793 vs FC .967). Ablation (lite): standard-for-all .543/.453/.569, widened-for-all .476/.513/.579,
  w/o whole index .507/.513/.519; pool QSS .507/.453/.522; cosine@k .465/.382/.392; rank8 .500/.431/.532; rank@k .504/.453/.519.
- 02:35 11-21: 05_results.tex rewritten for the history gate / full question sets / rerank baseline (new labels used:
  tab:swap, tab:gate, app:retrieval). The paragraph "Memory systems." in 5.1 holds the placeholder MEMSYS_SENTENCE: replace
  it once SimpleMem / Omni / TeleMem-VisualMem finish, after checking which claims are true (MMA strong > RAG@40 on
  VisualMem .868 vs .826; text-record systems vs RAG). Numbers in 5.1-5.5 to re-verify against paper_stats.json /
  main_cells.json after the final `p16_final.py all`: lite gains 5.0/4.3/12.9, pooled 4.6 p=.007, strong ranges 4.6 / 1.9,
  cost ratio 56%, swap 5 of 6, oracle +4 to +9, AUC .47-.63, "seven of eight MemEye histories", costs 5.5/9.7, 13.8/14.0,
  6.6/1.0, 23.1/8.8, latency 5.0/5.4, 8.8/11.0 vs 16.5/15.7.
- 02:45 11-21 REMAINING STEPS (in order): (1) abstract + intro: gate wording (history-level), claims (lite: all three, +4.3 to
  +12.9, pooled significant; strong: clear gain on VisualMem only, on par elsewhere), Fig.-1 numbers, replace "RAG outperforms
  the six memory systems"; (2) fig_overview.tex gate box (H(h) >= tau); (3) 02_problem.tex + problem.pdf panels a/b with
  leak-free MemEye numbers (RAG@10 - FC -.140 [-.184,-.098], RAG@40 - FC -.046 [-.075,-.018]; complete/incomplete gap 0.0 /
  40.3; FC .810, no-visual .216); (4) 09_appendix.tex: Table 5 MemEye rows (FC .810; RAG@40 .764; RAG@10 .670; annotated
  evidence .780; no visual .216; drop high-res / dense-caption rows and Fig. 6 unless re-run), robustness (scenario-level with
  history gate, selection draws, prompt sensitivity, second judge, pooled), ablation protocol, visibility analyses
  (per-question estimate v_hat definition moved here; tab:gate; tab:swap), app:retrieval (tab:retrieval), open-weight
  selector paragraph, systems table via \input{sections_polished/tab_systems}, MIRIX removed everywhere; (5) Figure 1:
  paper_acl/figure_sources/rag_motivation/rag_accuracy.csv -> MemEye .750/.810 (371), MemLens .652/.691 (466), VisualMem from
  results/p16/fig1/visualmem_rag30_strong.jsonl vs .967; update README.txt there + Appendix B text (MemLens 466); (6)
  08_limitations.tex (open-weight selector, selection draws, prompt sensitivity, lexical visibility, reranker parity);
  (7) after baselines: p16_fix_reused.py + reruns (see P16-D12), `p16_final.py check` -> `main`, `appendix`, `figs`,
  `p16_paper.py tables` must NOT overwrite tab_main (its tables() rewrites tab_main rows for the 150 subset: call only the
  Table-2 part or re-run p16_final.py main afterwards), fill MEMSYS_SENTENCE in 05_results.tex, compile
  (cd paper_acl && ../tools/tectonic -X compile main_polished.tex), conclusion must end by page 8, check ?? and labels;
  (8) final audit list + report to the user (spend, caveats: MemEye prompt-edit explanation was wrong; gate changed to
  history level; reranker parity for the strong reader on MemEye/MemLens; open-weight selector fails; SimpleMem/Omni bugs).
- 02:55 11-21 DONE: abstract (gate = histories poorly represented; claims lite 4-13 points on all three, strong on long
  histories), intro (gate wording, contribution 3, overview caption), fig_overview.tex gate box (H(h) >= tau, write time),
  02_problem.tex (complete/incomplete: matches FC / trails by 40 points), problem.pdf panels a/b with leak-free MemEye values.
  STILL TODO in intro paragraph 3: Fig.-1 numbers ("65 to 79%", "5.5 to 17.4 points", "4.6 points on MemEye and 14.1 on
  VisualMem", "outperforms the six memory systems") -> after the VisualMem RAG@30 run and the baselines finish.
- NOTE: the clock times written in P16-D9..D13 entries are about 70 minutes ahead of the real clock (real time at this entry: 01:42 on 11-21). latency_bl DONE.
- 01:44 real time: 09_appendix.tex rewritten (prompts without the leak sentence; Table 5 MemEye rows leak-free; robustness: pooled, scenario-level, selection draws, reader instruction, second judge, open-weight selector with placeholder OPENVM for the VisualMem result; visibility analyses with tab_gate + tab_swap; new section Retrieval Baselines (app:retrieval, tab_retrieval); latency/evidence text; five systems via tab_systems.tex). Figure 6 (app_keys_reader) removed. Generated tables come from scripts/p16_final.py appendix.
- 01:45 real time, results: Fig.-1 RAG@30 strong (leak-free prompt, v2 rankings; results/p16/fig1/*_rag30_strong.jsonl,
  0 errors): MemEye .750 (FC .810 -> gap 6.0), MemLens-32k 466 .652 (FC .691 -> gap 3.9), VisualMem .806 (FC .967 -> gap 16.1).
  Open-weight selector on VisualMem (widened path, lite): .388 vs RAG@40 .450 (-6.2 [-10.2,-2.2]); FSR .280 vs .805; 11.1 vs
  5.8 rounds -> replace OPENVM in 09_appendix.tex robustness paragraph ("and 0.388 against 0.450 on VisualMem").
  Chains X, Y, judge2, sec2, open, latency_bl are all DONE. Still running: omni memlens (-> memeye -> visualmem), simplemem
  memlens (-> memeye -> visualmem).
- 01:46: intro paragraph 3 (65-81%, gaps 3.9-16.1; 'five memory systems do not close this gap either'), appendix open-weight sentence (VisualMem .388 vs .450; FSR .280 vs .805), 08_limitations.tex rewritten; Figure 1 regenerated from rag_accuracy.csv (MemEye 75.0/81.0, MemLens-466 65.2/69.1, VisualMem 80.6/96.7). Remaining: baselines (omni, simplemem) -> fix-ups -> p16_final.py main/appendix/figs -> MEMSYS_SENTENCE -> Table 2 -> compile -> audit.
- 01:46: p16_final.py memlens -> tab_memlens_full.tex (history gate; star = permutation p<.05). Do NOT run p16_paper.py tables (it would overwrite tab_main/tab_gate with the old layout).
- 01:46: p16_final.py all ran with SimpleMem/Omni cells still missing ('--' in tab_main); intermediate compile done to check layout. Re-run 'p16_final.py all' when the baselines finish.
- 01:47: MEMSYS_SENTENCE replaced by a provisional paragraph (marked % VERIFY-AFTER-BASELINES in 05_results.tex; claims to re-check once SimpleMem/Omni finish: RAG@40 > every system with the lite reader; V-Mem trails by 1.7-3.4 strong; MMA matches on MemEye and exceeds on VisualMem .868 vs .826; VGS > every system in every column). Compile fixed.
- 01:48 (real time) intermediate compile OK: 16 pages, conclusion ends on page 8, no unresolved references; Tables 1-4 on
  page 6, Fig. 4 on page 7. tab_main still shows '--' for SimpleMem / Omni-SimpleMem (runs pending). Small fixes pending:
  5.4 "oracle would add 4 to 9" -> "2 to 9"; conclusion + related-work gate wording; Fig. 4d legend overlaps points.
- 01:48: conclusion + related-work gate wording updated. The conclusion now ends with 'adapted to the history and the reader' (use this phrase to locate the end of the main text in the PDF).
## P16-D14: CHECKPOINT 01:49 real time, 11-21 -- what is done, what is running, what to do next
DONE (all written to disk): history gate (p16_compose.py), stats/figure (p16_paper.py stats|figs|ablation), p16_final.py
(check | main | appendix | figs | memlens | all), paper text for abstract, intro, section 2, 3.2, setup, results 5.1-5.5,
related, conclusion, limitations, appendix; Figure 1 regenerated; intermediate compile: 16 pages, conclusion ends on page 8.
RUNNING: (1) omni chain: memlens (868/932 records at this time) -> memeye -> visualmem (automatic);
         (2) simplemem chain: memlens (614/932) -> memeye -> visualmem (automatic, 6 threads, ~3.5 GB).
NEXT, in this order:
 a. When results/p16/bl_logs/STATUS_omni_memlens exists:
      .venv/bin/python scripts/p16_fix_reused.py omni memlens results/p16/omni_memlens_reused_dirs.json
      scripts/p16_run_baselines.sh omni memlens 4      (may run while the omni memeye stage runs)
    When STATUS_simplemem_memlens exists:
      .venv/bin/python scripts/p16_fix_reused.py simplemem memlens results/p16/simplemem_memlens_rerun.json
      scripts/p16_run_baselines.sh simplemem memlens 3  (wait until the simplemem memeye stage is not loading its model, or
      run it after the whole simplemem chain; memory: one simplemem process at a time is safest)
    NOTE: the runner writes STATUS_<sys>_<bench> again at the end of a re-run; check record counts (932 / 742 / 1392), 0
    errors, 0 unjudged, no n_items == 0, omni logs without "Text embedding empty", simplemem log without "Error during".
 b. .venv/bin/python scripts/p16_final.py check   -> no MISSING; then: scripts/p16_paper.py all; scripts/p16_paper.py ablation;
    scripts/p16_final.py all   (order matters: p16_final.py last; never run p16_paper.py tables / extras).
 c. Re-verify the numbers quoted in 05_results.tex (5.1 memory-systems paragraph is marked % VERIFY-AFTER-BASELINES), intro
    ("five memory systems ... do not close this gap either"), abstract; fix Fig. 4d legend overlap (p16_paper.py figs).
 d. Compile: cd paper_acl && ../tools/tectonic -X compile main_polished.tex ; conclusion must end on page <= 8 (search the
    PDF text for "adapted to the history and the reader"); no "??"; view pages 5-8 and the appendix.
 e. Audit of direction/narrative + list of missing validations for the user; update REPORT.md (phase 16), README.md (paper
    generators table: p16_final.py), AGENTS.md (phase-16 notes: history gate, baselines env vars, pitfalls).
 f. Final message to the user (Chinese): what changed, honest caveats (see P16-D7, D11, D13), spend.
Spend: Gemini 1951 at 01:45 (stretch start 1782.1), OpenRouter 18.2.
- 02:01: omni memlens finished (932 records); fix-up started: removed the records + run dirs of the 77 reused-dir questions (backup in results/p16/bl_discarded/) and re-running them (STATUS_omni_memlens_fix when done). omni memeye stage is running in parallel.
- 02:11: omni memlens fix-up finished: 932 records (see check above in the session; verify again with p16_final.py check). Waiting for: omni memeye -> visualmem; simplemem memlens (844/932) -> memeye -> visualmem; then simplemem memlens fix-up (3 questions).
- 02:23: simplemem memlens finished (932 records); fix-up started for q_9bad18be, q_cc4da426, q_37d32eab (records + run dirs removed, backup in results/p16/bl_discarded/; STATUS_simplemem_memlens_fix when done). simplemem memeye stage runs in parallel; omni memeye at 314/742.
- 02:31: simplemem memlens fix-up done (check printed in session; 932 records expected, no n_items==0). Remaining: omni memeye (566/742) -> visualmem; simplemem memeye -> visualmem (about 2 h). Then step b of P16-D14.
- 02:41: omni memeye DONE (742); omni visualmem started; simplemem memeye 114/742 -> then visualmem. All other jobs finished. Next: wait, then P16-D14 step b (p16_final.py check -> p16_paper.py all -> ablation -> p16_final.py all), verify 5.1 memory-systems paragraph, compile, audit, final report.
- 02:49 CHECKPOINT: only two chains still run: omni visualmem (0/1392 records; ~1 h) and simplemem memeye (414/742) ->
  simplemem visualmem (~1-1.5 h). Everything else of P16-D14 is done up to step a. When both STATUS_omni_visualmem and
  STATUS_simplemem_visualmem exist: step b (p16_final.py check; p16_paper.py all; p16_paper.py ablation; p16_final.py all),
  step c (verify 5.1 paragraph marked % VERIFY-AFTER-BASELINES and remove the marker; intro/abstract claims), step d (compile,
  page-8 check with "adapted to the history and the reader"), step e (REPORT.md phase 16 section; audit), step f (final
  message in Chinese with caveats: wrong earlier explanation of the MemEye prompt effect; gate now history-level; strong
  reader ties an LLM reranker on MemEye/MemLens; open-weight selector fails; MMA strong beats RAG on VisualMem; baseline bugs
  fixed; spend).
- 03:06: simplemem memeye DONE (742); simplemem visualmem started; omni visualmem still building (0 records).
- 03:07: Omni-SimpleMem logs contain a few 'Entity extraction failed' lines (its entity extractor could not parse the model's JSON for a unit; the system's own fallback, the unit is kept). Counts per log printed in session; not a harness error.
- 03:17: still waiting for omni visualmem and simplemem visualmem (both 0 records; builds of 10 long personas). No other jobs running.
- 03:27: both VisualMem chains are alive (omni building, ledger $3.5; simplemem answering queries, ledger $4.4, about $0.18/min -> expect ~$15 total). Nothing else is running. After both STATUS files exist follow P16-D14 steps b-f.
- 03:37: omni visualmem 114/1392 records, simplemem visualmem 0 (answering). Context nearly full: after a reset, read P16-D12, D13, D14 in this log and continue at P16-D14 step b once both VisualMem STATUS files exist.
- 03:56: omni visualmem 676/1392, simplemem visualmem 510/1392; 0 warnings / 0 'Error during'. Continue at P16-D14 step b when both STATUS files exist.
- 04:16: omni visualmem: 5 of 10 personas done (676 records, 0 errors), second batch of 5 running (alive, last ledger charge 04:16); simplemem visualmem 790/1392. Continue at P16-D14 step b when STATUS_omni_visualmem and STATUS_simplemem_visualmem exist.
- 04:36: omni visualmem 790/1392, simplemem visualmem 926/1392; both chains alive. Continue at P16-D14 step b when both STATUS files exist.
- 04:41: user asked to raise Gemini parallelism; answered: remaining jobs (omni / simplemem on VisualMem) are limited by the systems' sequential per-history question loops and RAM, all remaining histories already run in parallel; no restart. omni vm 790, simplemem vm 1054 of 1392.
- 04:50: omni visualmem DONE (1392). simplemem visualmem 1218/1392. When STATUS_simplemem_visualmem exists -> P16-D14 step b.
- 04:51: ALL baseline chains DONE (omni + simplemem visualmem 1392 each). Starting P16-D14 step b.
- 04:52: step b done (p16_paper.py all + ablation, p16_final.py all with all five systems; tab_main complete); 5.1 memory-systems paragraph verified against results/p16/main_cells.json and finalized; compiled.
## P16-D15: FINAL STATE 04:53 11-21 (steps a-d of P16-D14 done)
- All five rebuilt systems complete (0 errors, 0 empty retrievals; MemEye 371, MemLens-32k 466, VisualMem 696 questions).
  Main table (lite / strong): RAG@40 .493/.764, .470/.682, .450/.826; + LLM rerank .474/.790, .446/.674, .499/.861; FC
  .500/.810, .446/.691, .454/.967; SimpleMem .355/.445, .395/.403, .414/.747; Omni .311/.373, .384/.421, .425/.746; TeleMem
  .339/.398, .200/.159, .368/.703; V-Mem .469/.730, .461/.665, .427/.809; MMA .414/.761, .442/.629, .407/.868; VGS
  .543/.793, .513/.693, .579/.921 (costs in tab_main.tex; results/p16/main_cells.json).
- Paper compiled: paper_acl/main_polished.pdf, 16 pages, conclusion ends on page 8 (Limitations heading follows on page 8),
  no unresolved references. Remaining: step e (REPORT.md phase-16 section, audit list) and step f (final message).
- Audit findings to tell the user (not nitpicks): (1) strong-reader story is VisualMem-only; an LLM reranker over 100
  candidates ties VGS on MemEye/MemLens; (2) open-weight selector (Qwen3-235B) gives no gain over RAG -> method depends on a
  strong long-context selector; (3) lite gains on MemEye/MemLens are 4-5 points, significant only pooled (p=.007) and on
  MemLens-64k; MemEye number is the best of three draws (.543/.527/.518); (4) the gate is a history-level switch; with one
  path per benchmark it cannot be validated beyond 3 benchmarks -> a benchmark with mixed-visibility histories would be the
  real test; (5) VisualMem (synthetic, multiple choice) carries the headline; a second long-history real-image benchmark
  would be the most valuable addition; (6) earlier claim "prompt leak explained the MemEye change" was wrong (P16-D7).
## P17-D0: NEW USER REQUEST (08:46 11-21) -- plan (READ FIRST AFTER A CONTEXT RESET)
User (Chinese): (1) do the missing validations: ultra-long history with the new VGS; gate validation; selector dependence
(stronger open-weight selector / prompt); (2) add exactly ONE more dataset (4 in total): suitable, VGS works well, do not add
two; (3) Table 1 footnote about "* ... permutation test ... Figure 6": explain; delete if unimportant -> DELETE the star and
the sentence from Table 1 (scripts/p16_final.py main_table; keep the appendix interval figure); (4) which appendix content
deleted during the "focus the story" pass (e.g. old Fig. 8 = selector models) matters for the main line -> review the old
appendix (paper_acl/sections/09_appendix.tex = pre-polish version; /tmp/polished_backup_20261116/) and restore what matters,
RE-RUN WITH THE NEW VGS (compact index, history gate): selector-model comparison, longer histories (MemEye + 1k distractor
images), possibly text-dominant Mem-Gallery contrast; (5) user asks why "strong reader relies on VisualMem only" when VGS is
ahead of RAG@40 on all three: answer = numerically ahead on all three (+2.9, +1.1, +9.5) but only VisualMem is significant
(p=.12, .51 on MemEye, MemLens) and an LLM reranker ties there; paper may say "ahead on all three, significantly on VisualMem";
(6) restructure: main text = main-line experiments only, about 7.5 pages (user adds an open-source backbone later to reach 8;
do not pad); Fig. 3a may duplicate Fig. 1 -> replace with something more informative; I may add experiments the narrative lacks.
Steps: a. Table-1 star removal. b. grep this log for WorldMemArena (bench "wma", src/data_wma.py, results/p8*, p9) and
MemoryQA (p10) to pick the 4th dataset by the VisualMem inclusion criterion (P10-D0: image-dependent, long history); run
RAG@40 / FC / VGS (both readers) + history visibility on it (it is untouched by tuning -> also the held-out gate check).
c. selector models on the compact index (selection-only FSR on MemEye, + e2e lite for 1-2): gemini-3.8-flash, lighter Gemini,
Qwen3-235B (done: results/p16/open), one stronger open-weight model; try a selector instruction variant for open models on
DEV data only. d. long histories: run_mm2 --grow 1000 on MemEye with the compact index (prefilter ptoc38:200:40 because the
whole index would exceed 100k tokens), RAG@40 and VGS, lite (+ strong if cheap). e. restructure paper + regenerate + compile.
Budget guess: 4th dataset $30-60, selectors $10, long histories $20. Report spend; Gemini total now 1997.7.
## P17-D1: state 08:47 11-21 (continue here)
- DONE: Table 1 star + footnote sentence removed (scripts/p16_final.py main_table; tab_main.tex regenerated; the pdf is NOT yet
  recompiled after this change).
- 4th dataset, facts so far: data/ holds Mem-Gallery, MemEye, MemLens, MemoryQA, VisualMem. Already measured and OUT of scope by
  the P10-D0 inclusion criterion (RAG@40 lite keeps < 70% of its accuracy without visual information): Mem-Gallery (~100%
  kept), WorldMemArena (90% kept; VGS .694 vs RAG .692, P8-D4), Memory-QA (no retrieval problem). So none of these may be
  added as "the 4th benchmark" except as an out-of-scope contrast. TODO: look for a NEW image-dependent long-history benchmark
  among the cited ones (SMMBench chai2026smmbench arXiv 2605.15710, DMV-Bench tang2026dmvbench, others via web search), check
  the inclusion criterion FIRST, then run RAG@40 / FC / VGS once (both readers; history gate; no tuning) and report whatever
  comes out. Tell the user plainly about every dataset tried; do not pick by outcome.
- Then: selector models on the compact index (c), long histories --grow 1000 (d), appendix restoration review (compare
  paper_acl/sections/09_appendix.tex), restructure to ~7.5 pages (Fig. 3a overlaps Fig. 1 -> replace), recompile, report.
- Answer owed to the user: strong reader is numerically ahead on all three (+2.9 / +1.1 / +9.5) but significant only on
  VisualMem; LLM rerank ties on MemEye / MemLens.
## P17-D2: 4th-dataset search notes (08:47 11-21) -- continue here
- Candidates checked by web search: (1) SMMBench (arXiv 2605.15710; HF dataset HuacanChai/SMMBench; 1,877 samples over 264
  sources: conversations, profiles, screenshots, tables, images, documents; evidence must come from >= 2 sources; CC-BY-4.0)
  -> fits "evidence sets across items"; must check: memory size per sample (is there a retrieval problem?), image dependence
  (P10-D0 criterion), loader effort. (2) DMV-Bench (EMNLP 2026; interactive shopping agent with incidental image cues; not QA
  over a stored history) -> not suitable for our reader protocol. Not yet checked: liu2026worldmemarena is out of scope (90%
  kept without visuals); others cited in refs.bib (jiang2025memoryqa out; guo2026seeingsurvey lists benchmarks at
  https://seeing-maintaining-learning.github.io/benchmarks/ -> browse that list for long-history, image-dependent QA sets).
- NEXT: inspect HuacanChai/SMMBench files (huggingface_hub is installed in .venv), decide by the inclusion criterion, write
  src/data_smm.py + bench name in run_mm.episodes, run RAG@40 / FC / VGS once (history gate, both readers), report as is.
  Then P17-D0 steps c (selector models), d (long histories --grow 1000), appendix restoration, restructure (~7.5 pages).
- Nothing is running in the background. Table-1 star removal is in tab_main.tex but the PDF is not recompiled yet.
## P17-D3: SMMBench inspection in progress (08:48 11-21) -- continue here
- HF dataset HuacanChai/SMMBench: 3,165 files = Images/ (2,838 png/jpg) + Samples/ (325 json, per cluster: Samples/cluster_<k>/
  QA_sample.json ...). README + first sample file downloaded to data/SMMBench/. A sample has: id, domain (e.g. MMCV), category
  (e.g. Multi_Hop_QA), question (may cite a figure id: "Based on Fig. a17b3905, ..."), answer, evidence {text_evidence,
  image_evidence, table_evidence, mis_* (misleading), json_evidence}, multi_choice_QA {answer index, 4 options},
  evidence_assignment {...}, meta_data. Evaluation code: github.com/FatCatCHC/SMMBench.
- STILL UNKNOWN (check before any run): what the "memory" of a cluster is (the 264 sources: where are they stored? other json
  files in each cluster dir?), how many items per cluster (is there a retrieval problem at 40 rounds?), share of questions
  whose evidence includes images, and the P10-D0 inclusion test (RAG@40 lite with vs. without visual information).
  If SMMBench has no usable long history or fails the criterion, tell the user and propose the fallback: keep three
  in-scope benchmarks + report Mem-Gallery / WorldMemArena as out-of-scope contrasts (already measured).
- Everything else of P17-D0 (selector models c, long histories d, appendix restoration, restructure to ~7.5 pages, recompile)
  is still TODO. No background jobs. Gemini total 1997.7, OpenRouter 18.3.
## P17-D4: SMMBench structure (08:48 11-21) -- continue here
- 61 clusters (Samples/cluster_<k>/): each has QA_sample.json (cluster_1: 25 samples; categories Multi_Hop_QA, Single_Hop_QA,
  Preference_QA, ...; domains MMCV, MMDocRAG, ChartQA_Pro, MMPB, LifeSim) plus the SOURCES = several conversation files
  (group_chat_<topics>_<id>.json, user_assistant_conversation_<id>.json; 264 in total). Evidence is INSERTED into conversations:
  evidence_assignment gives {conversation_name, insert_conversation_turn (e.g. 236, 578), reference_name "Fig. a17b3905"} for
  text / image / table / json evidence and misleading evidence. Images/ is flat (2,838 files, named <domain>_<hash>.png).
  23 of 25 cluster-1 samples have image evidence; questions also carry a 4-option multiple-choice form.
  => long multi-source histories (hundreds of turns per conversation, several conversations per cluster) with image evidence
  spread over sources: fits the paper (evidence sets across items; long history). Still to check: format of a conversation
  file (turn fields, how images/tables appear), total turns per cluster, and the inclusion criterion (RAG@40 lite with vs
  without visual information) BEFORE running VGS. Loader plan: src/data_smm.py -> one Episode per cluster (sessions =
  conversations; rounds = turns or user+assistant pairs; images from Images/), QA.clue = rounds at the insert turns, scoring by
  the multiple-choice letter (exact match, like VisualMem) to avoid a new judge.
- Download only cluster files first (hf_hub_download per file, local_dir=data/SMMBench); images on demand.
## P17-D5: SMMBench conversation format + NEXT STEPS (08:49 11-21) -- READ FIRST AFTER A CONTEXT RESET
- A conversation file = {"conversation": [turns], "topics": ...}. Turn fields: sender_name, sender_agent_id, timestamp,
  content, content_type ("text" | "image"), image_path (file name in Images/, e.g. MMCV_96c7...png; null for text),
  subtopic, metadata, caption (for images). A group chat has ~1,009 turns; a user-assistant conversation ~90. Image evidence
  appears as two turns: a text turn "Look at this figure: Fig. a17b3905" and an image turn with image_path. Cluster 1 has 3
  group chats + 2 user-assistant conversations (~3,200 turns) and 25 samples, 23 with image evidence -> a long, multi-source,
  image-bearing history: suitable in structure. Files downloaded so far: data/SMMBench/README.md, Samples/cluster_1/
  QA_sample.json, two conversation files.
- NEXT (4th dataset): (1) download all Samples/ + the images referenced in conversations (snapshot_download with
  allow_patterns); (2) src/data_smm.py: Episode per cluster; session = conversation; round = one turn (group chats have no
  user/assistant pairs: put the turn text in Round.user with "sender: text", image turns carry Round.images; assistant "");
  rid = "<conv short id>:R<turn>"; QA.question = sample question + its 4 options (letters), QA.answer = letter; QA.clue =
  rids at insert_conversation_turn of text/image/table/json evidence (check 0- vs 1-based against reference_name turns);
  register bench "smm" in scripts/run_mm.py episodes(), mm.IMAGE_RULE / system prompt (reuse the visualmem rule), key
  cap:short, judging = mc_score like visualmem (see run_mm2.py / p14_read_ids.py / p12_common.judge for the visualmem branch);
  (3) cost check: ~3,200 turns x 61 clusters = 195k rounds to embed (cheap) and caption ~2,800 images (~$3); index per
  cluster ~100k+ tokens -> expensive selection (bash.08+/question uncached; use --xcache, compact index); so first run a PILOT on
  ~8 clusters (~200 questions): inclusion criterion (RAG@40 lite with / without visual info: rag:40|read:medium vs
  rag:40|read0:medium), then RAG@40, FC (may exceed context: skip FC if > 1M tokens), VGS (history gate -> compute H with
  scripts/p16_hvis.py logic), both readers. Report the result whatever it is; do NOT tune on it.
  (4) If SMMBench is unsuitable or VGS does not help, tell the user plainly and keep three benchmarks.
- STILL TODO from P17-D0: c. selector models on the compact index; d. long histories (--grow 1000); appendix restoration
  review (old appendix: paper_acl/sections/09_appendix.tex); restructure main text to ~7.5 pages (replace Fig. 3a, which
  overlaps Fig. 1); recompile (Table-1 star removal not yet compiled); answer the user's strong-reader question (ahead on all
  three: +2.9 / +1.1 / +9.5; significant only on VisualMem; LLM rerank ties on MemEye / MemLens).
- No background jobs. Spend: Gemini 1997.7, OpenRouter 18.3 (before any P17 run).
## P17-D6: CHECKPOINT (08:50 11-21) -- READ P17-D0 .. P17-D6 FIRST AFTER A CONTEXT RESET
- Background job: SMMBench download (data/SMMBench/Samples then Images; log results/p16/logs/smm_download.log; STATUS file
  results/p16/logs/STATUS_smm_download when done). Nothing else runs.
- Data classes (src/data_mm.py): Round(rid, sid, date, order, user, assistant, images[abs paths], image_ids, given_captions);
  QA(qid, task, question, answer, qtype, clue[rids], x, y, question_image, mcq, latest_clue, stale_clues, option_images);
  Episode(name, bench, rounds, qas, profile, persona_images) with rid2idx. VisualMem loader = src/data_visualmem.py (copy its
  multiple-choice conventions: question text + "\nA. ..." options, answer letter, mc_score).
- Immediate next actions: (1) write src/data_smm.py + register bench "smm" (scripts/run_mm.py episodes(); mm.IMAGE_RULE;
  run_mm2 / p14_read_ids / select_eval branches that test bench in ("memlens","wma","visualmem")); (2) pilot on ~8 clusters:
  inclusion criterion, RAG@40, VGS (compact index, --xcache, history gate), both readers; report as is; (3) P17-D0 steps c, d;
  (4) appendix restoration review; (5) restructure main text to ~7.5 pages and recompile (Table-1 star already removed in
  tab_main.tex); (6) answer the user's questions in the final message (Table-1 star meaning + removal; which deleted appendix
  items matter and were restored; why strong-reader claim was "VisualMem only": ahead on all three, significant only there).
- 08:51: SMMBench Samples downloaded (61 clusters); Images downloading slowly (~2 files/s, 2,838 files, ~25 min;
  results/p16/logs/smm_download.log, STATUS_smm_download when done). While waiting: write src/data_smm.py following
  src/data_visualmem.py (load_visualmem: Round per user+assistant pair, image paths, QA with letter options, mc_score), then
  the pilot of P17-D5 step (3). User's open questions to answer in the final message are listed in P17-D6 item (6).
## P17-D7: SMMBench facts (08:51 11-21) -- continue here
- 61 clusters, 1,877 questions (1,671 with image evidence); turns per cluster: mean 3,599 (1,835-4,534). Categories:
  Single_Hop_QA 572, Preference_QA 581, Conflict_Resolution_QA 371, Multi_Hop_QA 144, Function_Call 108 (no options), Other
  101. multi_choice_QA = {multi_choice_QA_answer: int index, multi_choice_QA_options: [4 options (1,650 q), 2 (112), 3 (7)]}.
  Evidence assignments: text 1,966, image 1,752, json 405, table 125, misleading text 2,431 (conflict questions).
  Each assignment = {conversation_name, insert_conversation_turn, reference_name}.
- Plan for the loader (src/data_smm.py): keep the 1,769 questions with options (drop Function_Call: no options, action
  prediction); question = original text + lettered options; answer = letter of multi_choice_QA_answer (CHECK 0- vs 1-based:
  first sample answer "Stockholm" is options[3] with multi_choice_QA_answer = 3 -> 0-based); gold = the turns at
  insert_conversation_turn of the (non-misleading) text/image/table/json evidence (check index base against the image turn
  with the matching image_path / reference_name); pilot on a fixed random subset of clusters first.
- Images still downloading (88 / 2,838 at this time; log results/p16/logs/smm_download.log).
## P17-D8: SMMBench assessment + other candidates (08:54 11-21) -- continue here
- SMMBench facts that make it a WEAK fit: every question names its figure/doc/table id ("Based on Fig. a17b3905, ..."), so the
  image is found by an explicit handle, not by similarity (our story is low-similarity image evidence); only the second,
  textual evidence item needs retrieval. Histories are ~3,600 messages per cluster (group chats, 11-15 senders; no
  user/assistant rounds), index ~150k tokens per cluster, json "documents" with embedded images. Not run. Images are still
  downloading in the background (harmless; data/SMMBench).
- Benchmark list of the survey (guo2026seeingsurvey site, 19 image benchmarks). Candidates closer to our scope (long personal
  history, answers depend on REAL images, QA + retrieval): ATM-Bench ("According to Me: Long-Term Personalized Referential
  Memory QA", real data), CamRoll (camera-roll VQA, real data), M3Exam (hybrid). Check availability / size / structure of
  ATM-Bench first, then CamRoll; decide by the inclusion criterion BEFORE looking at VGS results; run once; report as is.
  A real-image long-history benchmark would also answer the "synthetic VisualMem" weakness.
## P17-D9: 4th dataset DECISION = CamRoll-yfcc20 (08:55 11-21) -- continue here
- Chosen BEFORE any run, on suitability only: thaoshibe/camroll-yfcc20 (arXiv 2606.05275 "Personal AI Agent for Camera Roll
  VQA"; HF dataset, 2.29 GB, CC-BY-NC-SA-4.0): 20 real personal camera rolls from YFCC100M (~500-1,000 photos each over
  years; 15.6k photos), per user 10 semantic + 40 episodic multiple-choice questions (3 options, option "a" is correct ->
  shuffle deterministically), episodic questions carry image_ref (gold photos), metadata.csv = capture time per image,
  profile.jpg = reference photo of the album owner. Fit: real images (answers the "synthetic VisualMem" weakness), long
  visual history, evidence only in images, evidence annotations, exact-letter scoring.
- Rejected before any run: SMMBench (explicit figure handles; 150k-token indexes), ATM-Bench (one 11k-item corpus with video,
  open-ended), DMV-Bench (interactive). The SMMBench image download was stopped (partial files in data/SMMBench, unused).
- Commitment: run once with the frozen method (compact index, history gate tau .45, same prompts), report the outcome
  whatever it is; tell the user about every dataset examined.
- Loader plan: src/data_camroll.py, bench "camroll": Episode per user; round = one photo (session = calendar day; date =
  capture time); persona_images = [owner reference photo]; images downscaled to 1,024 px in data/camroll_1024/; QA with
  lettered shuffled options; clue = image_ref rounds (episodic). Register in scripts/run_mm.py episodes(), mm.IMAGE_RULE,
  and every ("memlens","wma","visualmem") tuple / visualmem scoring branch (run_mm2.py 115-117 + 322, select_eval.py 88-100,
  p14_read_ids.py 31 + 44, p12_common.judge, p11_baselines._judge, results/p16/load.py).
- Cost guess for the full column (20 users, 1,000 questions): captions ~$15, RAG/VGS/rerank ~$25, FC ~$50 (check
  first), five memory systems ~$70. Run the core (captions, RAG@40, VGS, both readers) first, then FC + systems.
- 08:58: CamRoll download restarted with curl (hub client stalled on large files; curl gets ~3.4 MB/s): /tmp/camroll_fetch.sh via xargs -P 5; STATUS results/p16/logs/STATUS_camroll_download; file list data/CamRoll/files.json (83 files, 2.29 GB). Caption cost measured from the usage log: $0.0011 per image -> ~$17 for 15.6k photos.
## P17-D10: CamRoll loader written (09:00 11-21) -- continue here
- src/data_camroll.py: users(), prepare(user) (unzips images.zip -> data/camroll_1024/<user>/<stem>.jpg at 1,024 px +
  _profile.jpg), load_camroll(qids, only_users): Episode "camroll_<user>", bench "camroll"; round = photo (rid D<day>:P<j>, sid
  D<day>, date = day, user text "Photo taken at HH:MM.", image id D<day>_P<j>); QA ids cr<UU>_<s|e><NN>; options shuffled by
  md5(qid); answer letter; clue = image_ref rounds; qtype semantic|episodic; persona_images = owner reference photo.
  `python src/data_camroll.py --stats` prepares all downloaded users and prints counts.
- Registered: scripts/run_mm.py episodes("camroll"), mm.IMAGE_RULE["camroll"], run_mm2.py / select_eval.py bench tuples,
  letter scoring in run_mm2.py and p14_read_ids.py. NOT yet: p12_common.judge, p11_baselines._judge + argparse choices of
  the baseline runners (needed only when memory systems are run on CamRoll), results/p16/load.py, p16_hvis.py NAMES,
  p16_compose.py SRC/HVIS, p16_final.py BENCH/NQ.
- Download: nohup /tmp/camroll_fetch_all.sh (4 curl lanes, resumable; log results/p16/logs/camroll_download.log; STATUS file
  results/p16/logs/STATUS_camroll_download). ~475 MB of 2.29 GB at this time.
- NEXT: after download: prepare + stats; pilot timing on 1 user (captions, RAG@40, VGS standard path lite); then all 20 users:
  rag:40|read:medium (lite), rag:40|read:low (strong, --reader gemini-3.8-flash), toc38:40|read:medium and
  toc38:40|read+toc:low with --compact noassist,shortid,telegraph (lite) / noassist,telegraph (strong) --xcache, widened
  paths only if the history gate asks for them (compute H with scripts/p16_hvis.py after adding "camroll" to
  results/p16/load.py); FC lite/strong after a cost check.
## P17-D11: appendix-restoration review + structure plan (09:02 11-21) -- READ FIRST AFTER A CONTEXT RESET
Old appendix (paper_acl/sections/09_appendix.tex) vs current polished appendix; items deleted in the "focus" pass:
 (1) Longer histories (MemEye + 1,000 distractor images, 4,088 rounds) -- HIGH importance: controlled test of the claim that
     the strong-reader gain needs long histories. RE-RUN with the new VGS and put in the MAIN TEXT (accuracy vs history length).
 (2) Selector models (old Fig. "app_ablate": FSR of selector models under the shared prompt on MemEye) -- HIGH: answers
     "single model family" and my finding that Qwen3-235B fails. RE-RUN on the compact index (selection only; several models
     via OpenRouter + lighter Gemini), then one end-to-end run with the best non-Gemini selector.
 (3) Masking experiment (causal evidence that index visibility matters) -- MEDIUM-HIGH. Replace by a cheap dose-response on
     the compact index: selection-only FSR vs index line length (snippet 50/100/200/400/600) on MemLens (H rises with length).
 (4) Mem-Gallery / WorldMemArena with VGS (out-of-scope benchmarks) -- LOW/optional; Fig. 3b already shows Mem-Gallery is not
     image-dependent. (5) retrieval keys / reader-by-input figure, gold definitions, deployment variants, old MemEye
     end-to-end table, independent-run table -- NOT needed (superseded or setup details).
Main-text plan (~7.5 pages; user adds an open-source backbone later): Fig. 3 = diagnosis only: replace 3a (duplicates Fig. 1)
by "share of gold rounds by similarity rank" + "RAG@40 accuracy with complete vs incomplete evidence" (now Fig. 4a/b, RAG
part); keep FSR by set size and visual dependence. Table 1 = accuracy only for FOUR benchmarks (costs -> Fig. 5 + appendix
table). Table 2 (MemLens 64k) -> appendix. Results: 5.1 main; 5.2 evidence recovery (Table 3 + VGS panels); 5.3 ablation
(Table 4); 5.4 gate (paths + visibility panels); 5.5 history length (NEW figure panel) + cost. Appendix: everything else.
4th benchmark = CamRoll-yfcc20 (P17-D9/D10). Download running (nohup /tmp/camroll_fetch_all.sh).
## P17-D12: launched (09:04 11-21) -- scripts/p17_runs.sh long | vis | sel (nohup, resumable; logs results/p17/logs)
- long: MemEye grown by 250 / 1000 / 500 distractor images (Mem-Gallery sessions; scale.grow): rag:40 (lite, strong) and the
  VGS standard path (lite: toc38:40|read:medium compact+shortid, strong: toc38:40|read+toc:low compact) with --xcache and the
  WHOLE index -> results/p17/long/dx<g>/ ; STATUS_long_<g>. Base point = paper numbers (RAG .493/.764, VGS .543/.793).
- vis: visibility manipulated by index line length: MemEye standard path with --snippet 30, 60 (results/p17/vis/memeye);
  MemLens-32k test standard path with --snippet 600, 2000 and notes path at snippet 200 (results/p17/vis/memlens);
  STATUS_vis. Compare with widened paths (.476 MemEye, .513 MemLens) and standard (.543, .453); compute H(s) offline
  (p16_hvis.history_stats with mm2.SNIPPET = s).
- sel: selector models on the compact index, selection only, MemEye: toc@{qwent,gpt5m,ds,2.5f,2.5fl,gpt41m}:40 ->
  results/mm2/select_eval_*.json (new names); STATUS_sel. Existing: gemini-3.8-flash FSR .860 (7.2 rounds), Qwen3-235B
  FSR .598 (4.8 rounds; from the e2e run in results/p16/open/memeye).
- CamRoll download still running (1.6 of 2.29 GB).
- 09:06: selector-model runs relaunched in parallel (one process per model: gpt5m, ds, 2.5f, 2.5fl, gpt41m full MemEye; qwent on --limit_q 150 because it takes ~23 s per question); logs results/p17/logs/sel_<m>.log; outputs results/mm2/select_eval_toc@<m>-40_cnoassist+shortid+telegraph*.json. CamRoll download finished.
## P17-D13: status (09:15 11-21) -- continue here
- CamRoll: 19 of 20 users prepared (data/camroll_1024/<user>/; user 12754005_N00's 561 MB zip still downloading via
  /tmp/camroll_refetch2.sh -> STATUS results/p16/logs/STATUS_camroll_refetch; then run `python src/data_camroll.py` again).
  19 users = 14,776 rounds (505-979 photos per user, ~230 day-sessions), 950 questions (190 semantic, 760 episodic; 665
  episodic with gold photos, 1-2 each); 3 options -> chance .333.
- Pilot (user 0, 50 questions; results/p17/camroll_pilot; logs results/p17/logs/camroll_pilot_*.log; STATUS_camroll_pilot):
  rag:40|read:medium, then toc38:40|read:medium compact+xcache. Use it only to validate the pipeline and the cost.
- results/p16/load.py and scripts/p16_hvis.py now accept "camroll" (history visibility -> gate).
- First vis result: MemEye standard path with 30-char index lines = .529 (200 chars: .543; widened .476): little effect,
  MemEye messages are short and image descriptions carry the content -> compute H(30) before interpreting.
- Selector runs mostly finished (results/mm2/select_eval_toc@*-40_cnoassist+shortid+telegraph.json); collect when all done.
## P17-D14: selector models on the compact index, MemEye, selection only (09:18 11-21)
FSR / rounds / cosine FSR at the same size / $ per 1k: gemini-3.8-flash .860 (v2 run) / 7.2 / .52 / 5.1; gemini-2.5-flash .712 /
6.2 / .453 / 0.5; gemini-2.5-flash-lite .690 / 10.2 / .588 / 0.3; DeepSeek-V3.1 .838 / 17.6 / .739 / 1.2; GPT-4.1-mini .536 /
4.2 / .391 / 0.9; GPT-5-mini .666 / 6.4 / .526 / 2.9 BUT 82 empty selections (2,048 output tokens were used up by reasoning) ->
mm2.llm now gives gpt-5* / *thinking* models 8,192 tokens; GPT-5-mini is being re-run (truncated file kept in
results/p17/sel_truncated/); Qwen3-235B .598 / 4.8 (from the e2e run); Qwen3-235B-thinking running on 150 questions (check
empties; re-run if truncated). Every selector beats cosine at its own size by 10-35 points; absolute quality varies.
Files: results/mm2/select_eval_toc@<alias>-40_cnoassist+shortid+telegraph.json.
TODO: end-to-end lightweight-reader runs with DeepSeek-V3.1 (open weights) as selector on MemEye (standard), MemLens (widened),
VisualMem (widened): run_mm2 --cond "toc@ds:40|read:medium" / "toc@ds:40|readn:medium" --snippet 600 (as in
results/p16/open for Qwen) -> results/p17/open_ds/<bench>.
## P17-D15: CamRoll pilot + full chain launched (09:23 11-21) -- continue here
- Pilot, user 0 (50 q), lightweight reader: RAG@40 .620 (episodic .60, semantic .70; chance .33), FSR .875, $1.3/1k.
  Top-40 retrieval is already fairly complete on this user; the VGS pilot (standard path) is running
  (results/p17/camroll_pilot/toc38-40__read-medium__...). Pilot is for pipeline validation only.
- Full chain: `nohup scripts/p17_runs.sh camroll` waits for the last zip (STATUS_camroll_refetch) and the caption job
  (scripts/p17_camroll_captions.py, ~55 min at 4 images/s), then runs on all users -> results/p17/camroll/: rag:40 lite,
  standard path lite, rag:40 strong, standard path strong (STATUS_camroll_core), then widened lite (notes, 600-char lines),
  union strong (tocu:40:40), full context lite (STATUS_camroll_paths). Full context with the strong reader is NOT run (about
  800 photos per question -> ~$100). Still to add after the core: rerank@40 (select_eval rankr38:100 + p14_read_ids), the five
  memory systems (runners accept --bench camroll; needs P16_BASELINE_DIR / ledgers as in scripts/p16_run_baselines.sh), history
  visibility (scripts/p16_hvis.py camroll), integration into p16_compose.py / p16_paper.py / p16_final.py.
- Other running jobs: long (dx250 -> dx1000 -> dx500), vis (MemLens 600 / 2000 / notes-200), sel (qwent 150 q, gpt5m rerun),
  DeepSeek e2e (results/p17/open_ds). Results so far: long dx250 RAG lite .488 (base .493); vis MemEye snippet 30 -> .529,
  60 -> .526 (200 -> .543).
- Spend check needed (Gemini was 1997.7 before phase 17).
- 09:24: scripts/p17_runs.sh was edited while its long/vis chains were running (bash re-reads scripts by offset) -> restored to the original bytes at once; the camroll chain now lives in scripts/p17_camroll.sh (the running camroll instance was started from the temporarily edited file and keeps its parsed code). NEVER edit a running shell script; add new scripts instead.
## P17-D16: selector results + cross-family e2e launched (09:28 11-21)
- scripts/p17_analysis.py [long|vis|sel|open|all] written (results/p17/{long,vis,selectors,open_weight}.json).
- Selector models, MemEye, compact index, selection only (FSR / rounds / cosine FSR at the same size): gemini-3.8-flash .860 /
  7.2; GPT-5-mini .854 / 10.1 / .615 (after the output-limit fix; the truncated run had 82 empty sets); DeepSeek-V3.1 .838 /
  17.6 / .739; gemini-2.5-flash .712 / 6.2 / .453; gemini-2.5-flash-lite .690 / 10.2 / .588; Qwen3-235B .598 / 4.8;
  GPT-4.1-mini .536 / 4.2 / .391; Qwen3-235B-thinking: pending (150 q).
- End-to-end, lightweight reader, open-weight selectors: DeepSeek-V3.1 on MemEye .474 (RAG .493; VGS .543) although its
  FSR is .838 -> it opens 17.6 rounds; precision matters for the lightweight reader. MemLens / VisualMem for DeepSeek
  running (results/p17/open_ds). Qwen3-235B: .477 / .464 / .388 (RAG .493 / .470 / .450).
- Launched: GPT-5-mini as selector end to end (results/p17/open_gpt5m/<bench>; STATUS_open_gpt5m): MemEye standard path,
  MemLens + VisualMem widened path, lightweight reader -> tests whether a second model family reproduces the gain.
- MemLens decomposition of the widened path (lite): standard 200-char lines .453; 600-char lines without notes .483; 600-char
  lines + notes .513 (RAG@40 .470) -> half of the gain from longer lines, half from notes. Pending: notes with 200-char
  lines; 2,000-char lines.
## P17-D17: CamRoll pilot result + baseline launcher (09:34 11-21) -- continue here
- Pilot (user 0, 50 q, lightweight reader): RAG@40 .620 -> VGS standard path .800 (episodic .60 -> .80, semantic .70 -> .80);
  7.4 photos opened, FSR .90 (RAG@40 .875), $5.7 vs $1.3 per 1k. Pipeline works. Full runs follow the caption job.
- scripts/p17_camroll_bl.sh (nohup, waiting for the captions): five memory systems on CamRoll via
  scripts/p16_run_baselines.sh <system> camroll <n> (outputs results/p16/baselines/camroll/<system>.jsonl; ledgers
  results/p16/budget/camroll/; logs results/p16/bl_logs/*_camroll*.log; STATUS_<system>_camroll there) + LLM rerank
  (select_eval rankr38:100 -> scripts/p17_camroll_ids.py -> results/p17/camroll/ret_rerank40_{lite,strong}.jsonl);
  results/p17/logs/STATUS_camroll_bl at the end. CHECK the select_eval output name (expected
  results/mm2/select_eval_camroll_rankr38-100_cnoassist+shortid+telegraph.json).
- More results: long dx250 lite: RAG .488, VGS .513 (9.6 rounds; base .493 / .543 with 7.2) -> the VGS margin shrinks from
  +5.0 to +2.6 with 1.1k rounds of off-topic distractors; GPT-5-mini selector e2e MemEye lite .505 (RAG .493, VGS .543);
  MemLens standard path with 2,000-char lines .485 (600: .483; 200: .453; widened .513).
- 09:35: scripts/p17_final.py written (check|stats|main|figs|all): registers camroll in p16_compose (SRC/VHAT/HVIS), 4-benchmark accuracy-only Table 1 + appendix cost table (tab_cost.tex, label tab:cost), new Fig. 3 (diagnosis: visual input, similarity rank of evidence, RAG@40 complete vs incomplete, FSR by set size), new Fig. 4 (VGS opens by rank, gain where RAG incomplete, paths+gate, visibility vs widening), new Fig. 5 (accuracy vs cost on every benchmark). HAVE_CR switches camroll on when its six result files + hvis_camroll.json + vhat_compact_camroll.json exist (also needs ranks_camroll.json for stats). Tested on three benchmarks (figures regenerated).
- 09:37: paper edits (backup of all polished .tex in /tmp/polished_backup_20261116/before_p17): 02_problem.tex paragraphs + Fig. 3 caption for the new diagnosis panels (text memory now cites Fig. 1); 05_results.tex: Table 2 input removed from the main text (must be \input in the appendix robustness section: sections_polished/tab_memlens_full), Fig. 4 + Fig. 5 captions, 5.2 second paragraph. STILL TODO in 05_results: 5.1 with CamRoll, 5.4 (held-out gate check on CamRoll, lines-vs-notes decomposition), 5.5 (latency sentence -> appendix pointer; cost numbers for CamRoll); setup (CamRoll paragraph); appendix additions; abstract/intro numbers.
- 09:42: p17_final.py appendix(): tab_gate (all benchmarks), tab_swap / tab_retrieval / tab_std (three benchmarks), tab_systems, tab_memlens_full, tab_selectors.tex (label tab:selectors), tab_long.tex (label tab:long), app_baselines figure for all benchmarks. Qwen3-235B-thinking selector re-running with the 8,192-token limit (first run was truncated: 0.72 rounds on average). vis results: MemEye H(30/60/200 chars) = .579/.640/.692 with accuracy .529/.526/.543; MemLens H(200/600/2000) = .276/.333/.334 with standard-path accuracy .453/.483/.485; notes with 200-char lines .489; widened (600 + notes) .513 -> on MemLens longer lines +3.0, notes +3.6, both +6.0 (results/p17/vis.json).
- 09:43: after the CamRoll runs finish, run (no API): `.venv/bin/python scripts/p16_hvis.py camroll`,
  `.venv/bin/python results/p16/vhat_compact.py camroll`, `.venv/bin/python scripts/p16_ranks.py camroll` (NAMES patched),
  then `scripts/p17_analysis.py all`, `scripts/p16_paper.py ablation`, `scripts/p17_final.py all` (stats -> main -> figs ->
  appendix). p17_final.HAVE_CR needs the six camroll files + hvis + vhat.
- 09:46: bib entry nguyen2026camroll (arXiv 2606.05275; Nguyen, Singh, Kim, Lee, Li) added; colour hCR defined in main_polished.tex. Setup paragraph for CamRoll still to write (after the numbers).
## P17-D18: long-history result at +250 images (10:01 11-21)
MemEye grown to 1,151 rounds per history (base 106 by episode mean): lite RAG@40 .488 / VGS .513 (+2.6 [-2.1,+7.4], p=.31;
base +5.0); STRONG RAG@40 .705 / VGS .771 (+6.6 [+2.8,+10.6], p=.001; base .764 / .793, +2.9 n.s.). RAG@40 full-set recall
stays .906, so the strong reader's RAG loss comes from what else is in the 40 rounds: at base, 40 rounds are a third of the
history; at 1,151 rounds they are 3.5%. => supports "strong-reader gain grows with history length"; lite gain shrinks.
VGS cost at 1,151 rounds: lite $4.9/1k (index 44k tokens), strong $15.1/1k vs RAG $9.6. dx1000 and dx500 running.
GPT-5-mini selector e2e on MemLens finished (see results/p17/open_gpt5m/memlens); DeepSeek VisualMem at 92%.
- 10:10: open-weight selectors, lightweight reader, gate path (results/p17/open_weight.json): DeepSeek-V3.1 MemEye .474,
  MemLens .466, VisualMem .348 (RAG@40 .493 / .470 / .450; VGS .543 / .513 / .579); Qwen3-235B .477 / .464 / .388. On
  VisualMem both find the answer-bearing image for only .26-.28 of the questions (gemini-3.8-flash .805) -> with these
  open-weight selectors VGS is at or below RAG@40. GPT-5-mini e2e: MemEye .505; MemLens done (read the file); VisualMem
  running (results/p17/open_gpt5m). Caption job at 93%.
## P17-D19: CamRoll runs started (10:25 11-21) -- continue here
- All 20 camera rolls prepared and described (15,627 photos, mean description 150 chars). Core chain running (results/p17/
  camroll/; logs results/p17/logs/camroll_*.log; STATUS_camroll_core, then STATUS_camroll_paths). Baseline launcher running:
  five systems (results/p16/baselines/camroll/, logs results/p16/bl_logs/*camroll*), rank + rerank readings
  (STATUS_camroll_bl in results/p17/logs when everything is done; check STATUS_<system>_camroll in results/p16/bl_logs).
- After the runs: see the command list at the end of P17-D16/D17 entries (hvis, vhat, ranks for camroll -> p17_analysis.py all
  -> p16_paper.py ablation -> p17_final.py all), then text: setup (CamRoll paragraph: 20 camera rolls from YFCC100M, 15.6k
  photos, 1,000 multiple-choice questions = 200 semantic + 800 episodic, 3 options, round = photo with capture time, owner
  reference photo given to every reader; cite nguyen2026camroll), abstract/intro ("four benchmarks"; numbers), results 5.1
  (+CamRoll), 5.4 (held-out gate check on CamRoll; decomposition on MemLens: lines +3.0, notes +3.6, both +6.0), 5.5 (cost;
  latency sentence -> appendix), appendix (tab_cost, tab_memlens_full in robustness, tab_selectors + cross-family paragraph,
  tab_long, CamRoll details, visibility manipulation), limitations (selector dependence wording), compile to ~7.5 pages.
- 10:33: CamRoll history visibility H = .992 (20 histories, .984-.996) -> the frozen gate (tau .45) sends CamRoll to the STANDARD path. vhat_compact / ranks for camroll must be re-run after the RAG lite run has cached the question embeddings (cache-only loader failed).
- 10:43: the 'long' chain shell was stopped after launching dx1000's last run (the dx500 point is skipped to save about $10 and 40 min); the running dx1000 strong VGS process continues; check results/p17/long/dx1000/*.jsonl for 371 records each (no STATUS_long_1000 file will appear). CamRoll runs are slow at first because every photo is uploaded once through the Files API.
- 10:53: waiting: CamRoll rag lite at 95% (then std lite, rag strong, std strong -> STATUS_camroll_core; widened paths + FC lite -> STATUS_camroll_paths), rerank readings, five systems on CamRoll, dx1000 strong VGS at 72%, GPT-5-mini VisualMem e2e, Qwen-thinking selector v2.
## P17-D20: results (11:02 11-21): long histories, cross-family selector, CamRoll RAG
- Long histories (MemEye + distractor chats; results/p17/long.json; mean rounds per history 106 / 1,151 / 4,068):
  lite RAG@40 .493 / .488 / .482, VGS .543 / .513 / .512 (gain +5.0, +2.6 n.s., +3.0 n.s.);
  strong RAG@40 .764 / .705 / .747, VGS .793 / .771 / .795 (gain +2.9 n.s., +6.6 p=.001, +4.8 p=.02).
  RAG@40 full-set recall stays .90; VGS FSR .86 / .90 / .88-.90. VGS cost grows with the index: lite $1.6 / 4.9 / 12.5,
  strong $5.5 / 15.1 / 39.2 per 1k (RAG $0.8 / 9.6) -> whole-index reading is accurate at 4k rounds but 4x the cost of RAG.
  dx500 was skipped.
- GPT-5-mini as selector, lightweight reader, gate path (results/p17/open_gpt5m): MemEye .505 (RAG .493; +1.2 n.s.; Gemini
  selector .543), MemLens-32k .513 (RAG .470; +4.3 [+0.4,+8.2]; Gemini .513), VisualMem .404 (RAG .450; -4.6; FSR .451 vs
  .805). => a second model family reproduces the MemLens gain, part of the MemEye gain, and fails on the long VisualMem
  index. Open-weight selectors (DeepSeek-V3.1, Qwen3-235B) give no gain anywhere. Selector quality on long indexes is the
  key dependency -> limitation + appendix table (tab_selectors) + one sentence in the main text.
- CamRoll, lightweight reader, RAG@40: .724 (episodic .714, semantic .765), FSR .865, $1.3/1k (1,000 questions, 0 errors).
- 11:12: 09_appendix.tex: robustness now has Longer MemLens histories (+tab_memlens_full), Longer histories (+tab_long), Selector models (+tab_selectors; replaces the Open-weight selector paragraph); visibility analyses got 'Parts of the widened path' and 'Shorter index lines'; tab_cost is input before Reproducibility. Numbers in the selector paragraph must be re-checked against results/p17/selectors.json + open_weight.json at the end (Qwen-thinking v2 pending; p17_analysis SELECTORS key fixed to n150_toc@qwent-40).
## P17-D21: CamRoll lightweight-reader results (11:19 11-21)
1,000 questions (20 camera rolls): RAG@40 .724 (episodic .714 / semantic .765; FSR .865); RAG@40 + LLM rerank .734 (FSR .896);
VGS standard path .826 (episodic .826 / semantic .825; 7.4 photos; FSR .868; $4.7 vs $1.3 per 1k) -> +10.2 points.
One failed record (SSL error) in the standard-path file: re-run that condition at the end (same command; cached calls are
free) and check 0 errors. TeleMem and V-Mem finished on CamRoll (results/p16/baselines/camroll/). Strong-reader runs next.
- 11:20: camroll vhat (mean .996; 99.8% >= .7) and ranks written (results/p16/vhat_compact_camroll.json,
  ranks_camroll.json): gold photos by similarity rank: 74% top 10, 15% 11-40, 11% beyond 40. TeleMem on CamRoll lite .639 /
  strong .744; V-Mem .691 / .828 (2,000 records each, 0 errors). Waiting for: rag strong, std strong (STATUS_camroll_core),
  widened lite, union strong, FC lite (STATUS_camroll_paths), rerank strong, MMA, Omni, SimpleMem (STATUS_camroll_bl).
- 11:29: 04_setup.tex: 'four benchmarks' + CamRoll description (20 camera rolls, 781 photos on average, 1,000 MC questions, photo = round with capture time, owner reference photo, run after the method was fixed).
- 11:54: CamRoll strong reader: RAG@40 .894; RAG@40 + LLM rerank .901 (lite .734). VGS standard path (strong)
  running (~30 min), then STATUS_camroll_core. Selector table final: Qwen3-235B-thinking (150 q, 8,192-token limit) FSR .587,
  5.2 rounds, 11 empty sets.
- 11:56: p17_final.gate_table (own version, all benchmarks). Spend at 11:55: Gemini 2155.1 (P17 +$157), OpenRouter 36.0 (P17 +$17.7); camroll ledgers $48.6 so far.
- 12:13: 05_results.tex 5.5 renamed 'Cost'; latency reduced to one sentence with a pointer to app:latency. CamRoll cost sentence still to add once std strong finishes.
## P17-D22: CamRoll core results (12:26 11-21; 1,000 questions; STATUS_camroll_core)
Lightweight reader: RAG@40 .724 | + LLM rerank .734 | VGS (standard path, chosen by the gate, H = .99) .826 (+10.2).
Strong reader: RAG@40 .894 | + LLM rerank .901 | VGS (standard path) .899 (+0.5; on par) at $12.9 vs $11.4 per 1k.
Systems so far: TeleMem .639 / .744; V-Mem .691 / .828; MMA 1,700 of 2,000 records; Omni 500; SimpleMem 600 (slow).
Explicit-cache overhead for CamRoll lite about $1.0 per 1k (cache creations in the run window).
One failed record (cr01_s00, SSL error) in the lite standard-path file -> re-run that command once the chain is idle.
Running: widened lite (notes), union strong, FC lite (STATUS_camroll_paths); MMA / Omni / SimpleMem on CamRoll.
- 12:26: SimpleMem and Omni-SimpleMem on CamRoll were slow (6 and 5 of 20 camera rolls after 2 h with 6 / 5 threads):
  stopped (runner shells first, then the python processes) and relaunched with 14 / 15 threads so that all remaining camera
  rolls build in parallel (finished records are kept; LLM calls of interrupted builds are replayed from the request cache;
  run dirs are wiped before each build). results/p17/logs/STATUS_camroll_bl may now appear prematurely: IGNORE it and use
  results/p16/bl_logs/STATUS_{simplemem,omni,mma}_camroll + record counts (2,000 each, 0 errors) instead.
- 12:28: CamRoll lite standard path re-run for the failed record: 1,000 records, 0 errors, accuracy .827 (VGS lite on
  CamRoll = .827 vs RAG@40 .724 -> +10.3). Remaining: widened lite, union strong, FC lite (chain), MMA / Omni / SimpleMem.
- 12:36: CamRoll facts for the paper: 20 rolls, 15,607 photos (505-979, mean 780), 225 days per roll on average, 1,000
  questions (200 semantic, 800 episodic; 705 episodic with gold photos, 1,025 gold photos). Appendix section renamed
  "VisualMem and CamRoll Details" with a CamRoll paragraph. CamRoll widened lite (notes) = .815 (standard .827): the gate's
  choice (standard) is the better path on the held-out benchmark. MMA on CamRoll done (2,000 records). Pending: union strong,
  FC lite, Omni (500/2000), SimpleMem (600/2000) relaunched with 15 / 14 threads.
- 12:45: text updated for four benchmarks: abstract, intro contribution 3, 5.1 (lite: +5.0/+4.3/+12.9/+10.3; strong: at
  least as accurate on all four, VisualMem +9.5, within 2.9 points elsewhere, long histories +6.6/+4.8), 5.3 (controls on
  three benchmarks), 5.4 (CamRoll held-out gate check .827 vs .815; MemLens lines vs notes), conclusion. PLACEHOLDER
  "FCLITE_SENTENCE" in 05_results.tex 5.1 (needs CamRoll FC lite + rerank lite: previous text was "full context stays within
  2.4 points of RAG@40, and re-ranking helps only on VisualMem"). 5.1 "Memory systems" paragraph and 5.5 cost numbers for
  CamRoll still to update after Omni / SimpleMem finish. Check "within 2.9 points" (ME .764/.790/.793, ML .682/.674/.693,
  CR .894/.901/.899) and the strong union path on CamRoll for 5.4.
- 12:58: waiting on CamRoll union strong (80%), FC lite (after it), Omni (800/2000), SimpleMem (1100/2000). When all
  are done: p16_paper.py ablation -> p17_analysis.py all -> p17_final.py all; fill FCLITE_SENTENCE; update the 5.1 memory
  systems paragraph + 5.5 cost for CamRoll; compile; check ~7.5 pages; update README/AGENTS/REPORT; final report.
- 13:08: CamRoll strong union path .907 (standard .899, RAG@40 .894, rerank .901). FC lite running; Omni 900/2000,
  SimpleMem 1600/2000.
- 13:09: p17_final.cache_usd fixed (uses the run windows of results/p17/logs/camroll_std_{lite,strong}.log: $1.02 and $2.81 per 1k). First full generation with CamRoll done (FC lite, SimpleMem, Omni cells still missing).
- 13:10: compile with CamRoll: 17 pages, no unresolved refs; main-text floats: p6 Table 1 + Table 2, p7 Fig. 4 + Table 3,
  p8 Fig. 5. Check where the conclusion ends (target about 7.5 pages).
- 13:10: Fig. 4 panel labels fixed in p17_final.figures; cost figure (fig:cost) moved from 5.5 to the appendix (after tab_cost); 5.5 is one paragraph (costs incl. CamRoll: lite $5.7 vs $1.3, strong $15.7 vs $11.4; verify after final regeneration). Target: main text about 7.5 pages (was 7.9 before this move).
- 13:12: text: related work tightened (MIRIX clause removed, CamRoll cited), 5.4 rewritten compactly with CamRoll (lite
  standard .827 vs widened .815; strong union +0.8) and the lines-vs-notes pointer, limitations rewritten (CamRoll strong
  parity; selector dependence incl. GPT-5-mini; whole-index cost at 4k rounds). Before these edits the main text ended at
  about 7.75 pages; target 7.5. STILL TODO: 5.1 memory-systems paragraph + "% VERIFY-FCLITE" sentence after FC lite / Omni /
  SimpleMem on CamRoll finish; final regeneration; compile; page check; README / AGENTS / REPORT; final report.
## P17-D23: CHECKPOINT (13:13 11-21) -- READ FIRST AFTER A CONTEXT RESET
DONE in phase 17: Table-1 star removed; 4th benchmark CamRoll-yfcc20 chosen before any run (P17-D9), loader + registration,
core + paths + rerank + TeleMem / V-Mem / MMA finished; selector models; cross-family + open-weight selectors end to end; long
histories (dx250, dx1000); visibility manipulation + widened-path parts; generators scripts/p17_analysis.py and
scripts/p17_final.py; paper text restructured (Fig. 3 = diagnosis, Fig. 4 = what VGS changes, Table 1 = accuracy on four
benchmarks, Table 2 (MemLens 64k) + cost figure + cost table + latency moved to the appendix; appendix: longer MemLens
histories, longer histories, selector models, widened-path parts, shorter index lines, CamRoll details).
STILL RUNNING: CamRoll full context lite (results/p17/logs/camroll_fc_lite.log; ~245k tokens per question, slow, ~$25;
STATUS_camroll_paths when done), Omni-SimpleMem (900/2000) and SimpleMem (1700/2000) on CamRoll
(results/p16/bl_logs/STATUS_{omni,simplemem}_camroll when done; ignore results/p17/logs/STATUS_camroll_bl).
WHEN THEY FINISH: (1) check 2,000 records / 0 errors / no empty retrievals for omni + simplemem, omni log without "Text
embedding empty", simplemem log without "Error during"; (2) `scripts/p16_paper.py ablation`, `scripts/p17_analysis.py all`,
`scripts/p17_final.py all`; (3) 05_results.tex: rewrite the "Memory systems." paragraph for four benchmarks and replace the
sentence marked "% VERIFY-FCLITE" (full context vs RAG@40 for the lightweight reader incl. CamRoll; re-ranking: +0.0-1.0 on
CamRoll); verify every number in 5.1-5.5, abstract, intro, limitations against results/p16/main_cells.json,
paper_stats.json, results/p17/*.json; (4) compile, check the conclusion ends near the middle of page 8 (about 7.5 pages),
no "??"; look at pages 6-8 and the appendix tables; (5) README (paper generators: p17_final.py), AGENTS.md (phase 17
notes), REPORT.md (phase 17 summary); (6) final message to the user (Chinese): answers to their questions (Table-1 star;
which deleted appendix items mattered and what was restored; strong-reader wording: ahead on all four numerically, significant
only on VisualMem and on enlarged MemEye histories), the 4th dataset and the datasets examined, new experiments, honest
caveats (selector dependence; MemEye lite not significant alone; CamRoll strong parity), structure (about 7.5 pages), spend.
Spend so far: check gem.spend() (Gemini was 1997.68 and OpenRouter 18.31 before phase 17).
- 13:13: CamRoll full-context reading with the lightweight reader was STOPPED (245k tokens per question, about 2
  questions per minute -> 8 h; low value): the table shows "--" for full context on CamRoll for both readers, with a note in
  the caption and the appendix. 5.1 sentence now: "where it is run, full context stays within 2.4 points of RAG@40, and
  re-ranking adds at most 1.0 point except on VisualMem" (check: rerank lite ME -1.9, ML -2.4, VM +4.9, CR +1.0).
- 13:15: long-history sizes are now question-weighted like the setup (126.5 -> 1,171.5 -> 4,088.5 rounds; text says 127 / 1,172 / 4,089); section 2 + 5.2 mention CamRoll (11% beyond rank 40; VGS opens 61% of them; gain +16.8 on RAG-incomplete vs +9.2).
- 13:16: trims for the 7.5-page target (main text ended at ~7.73 pages: Limitations heading at y=396 in the right column of page 8): setup memory-systems paragraph, section-2 FSR note, related work (cormack2015autonomy no longer cited). Next: 5.1 memory-systems paragraph after Omni/SimpleMem, recompile, re-measure.
- 13:17: REPORT.md phase-17 section written; appendix selector paragraph numbers corrected (10 to 34 points; within one point; 80%); 5.2 got the CamRoll FSR sentence (.868 vs .865 with 7.4 photos).
- 13:22: number audit of 5.1-5.5 / abstract / intro / limitations / appendix long-history paragraph against
  paper_stats.json, main_cells.json, results/p17/long.json done by hand: consistent (lite gains 5.0/4.3/12.9/10.3; strong
  within 2.9 points on ME/ML/CR; widening strong +2.6/+10.9/-0.4/+0.8; oracle +2.3 to +8.5; costs). Wording fixes: 5.5 "tens of
  thousands of tokens"; limitations "almost as completely". PENDING ONLY: Omni (1100/2000) and SimpleMem (1800/2000) on
  CamRoll -> then the 5.1 "Memory systems." paragraph (claims to verify: lite RAG@40 > every system on all four; strong
  V-Mem trails RAG@40 by 1.7 to 6.6; MMA matches on MemEye, trails by 5.3 on MemLens and 4.0 on CamRoll, exceeds on
  VisualMem .868 vs .826; VGS > every system in every column), final regeneration, compile, page check, final report.
- 13:39: waiting for Omni (1600/2000) and SimpleMem (1900/2000) on CamRoll; both processes alive.
- 13:48: SimpleMem on CamRoll DONE (2,000 records); Omni at 1700/2000 (last 3 camera rolls).
- 11-22 00:34: RESUMED after a pause. All five memory systems on CamRoll are complete (2,000 records each; STATUS files present); nothing is running. Doing now: validation of the CamRoll baseline files -> regeneration (p16_paper.py ablation, p17_analysis.py all, p17_final.py all) -> 5.1 memory-systems paragraph -> compile -> page check -> final report.
## P17-D24: all runs finished; final regeneration (11-22 00:37)
- CamRoll memory systems validated (2,000 records each, 0 errors / unjudged / duplicates / empty retrievals; Omni log: 0 "Text
  embedding empty", 160 entity-extraction parse failures = its own fallback, 2 "LLM call failed" lines; SimpleMem log clean).
  CamRoll (lite / strong): RAG@40 .724 / .894; + LLM rerank .734 / .901; SimpleMem .673 / .819; Omni .614 / .718; TeleMem
  .639 / .744; V-Mem .691 / .828; MMA .600 / .854; VGS .827 / .899. Full context not run on CamRoll (both readers "--").
  CamRoll ledgers $82.7 (five systems).
- Regenerated: p16_paper.py ablation, p17_analysis.py all, p17_final.py all. 5.1 "Memory systems." paragraph rewritten for
  four benchmarks (verified: lite RAG@40 > every system on all four; strong V-Mem -1.7 to -6.6; MMA -0.3 / -5.3 / +4.2 / -4.0).
- Latency table: V-Mem / MMA rows re-measured at the paper's budgets (p16_final.latency keeps MemLens records, where each
  question has its own history): V-Mem 3.2/6.1, 2.1/8.4, 1.9/5.7 s; MMA 5.5/10.4, 2.3/4.5, 5.4/10.2 s.
- Spend: Gemini 2225.8 (phase 17: +$228), OpenRouter 36.0 (phase 17: +$17.7).
- NEXT: p17_final.py appendix (std caption), compile, page check (~7.5 pages), view pages, final report.
- 11-22 00:38: trimmed (section 2 text-memory paragraph to one sentence, 5.1 cost clause, 5.4 per-question paragraph,
  related-work last sentence) and recompiled: 17 pages, no unresolved refs; main text ends on page 8 at about 58% of the
  page (Conclusion starts at the bottom of the left column, Limitations heading at y=189 in the right column) -> ~7.6 pages.
  Floats: p1 Fig. 1, p3 Fig. 2, p4 Fig. 3, p6 Tables 1-3, p7 Fig. 4; appendix p12-17 (Tables 4-15, Figs. 5-6).
  NEXT: view appendix pages, final consistency check, final report.
- 11-22 00:43: Table 2 (selection without answering) now has CamRoll columns (cosine same size 7.4 / .599, top-40 .865, QSS 7.4 / .868; computed from ranks_camroll.json + the standard-path file; the same code reproduces the VisualMem cells .254 / .454 / .787); selector table shows cosine-at-same-size .523 for gemini-3.8-flash; history sizes written as 126 / 1,171 / 4,088; Fig. 3 uses ME/VM/MG labels + caption legend; limitations got the text-dominant scope sentence. Recompiled.
- 11-22 00:43: Table 2 (tab:select) transposed: benchmarks as rows (MemEye, MemEye >=4 gold rounds [11.4 rounds, measured], MemLens, VisualMem, CamRoll) x (Rds., cosine same size, cosine top-40, QSS). Recompiled.
- 11-22 00:45: history-level (two-stage) bootstrap of VGS - RAG@40 (results/p17/cluster_bootstrap.json), lite / strong:
  MemEye (8 scenarios) +.050 [-.018,+.120] / +.029 [-.030,+.097]; VisualMem (10 personas) +.129 [+.066,+.186] / +.095
  [+.062,+.134]; CamRoll (20 rolls) +.103 [+.052,+.158] / +.005 [-.019,+.030]; lite gain positive on 7/8, 9/10, 17/20
  histories. Appendix paragraph "History-level uncertainty" replaces "Scenario-level uncertainty".
- 11-22 00:45: compiled: 17 pages, no unresolved refs; main text ends on page 8 (Limitations heading at y=270 in the
  right column) = about 7.6 pages. Remaining small items: Ethics statement sentence for CamRoll (real YFCC photos, CC
  licences), final number audit, final report.
## P17-D25: FINAL STATE of phase 17 (11-22 00:46)
- Number audit of abstract / intro / 5.1-5.5 / limitations / appendix against results/p16/main_cells.json, paper_stats.json,
  results/p17/{long,vis,selectors,open_weight,cluster_bootstrap}.json: consistent (one fix: MMA on MemLens -5.4).
- Paper: paper_acl/main_polished.pdf, 17 pages, main text ends on page 8 at about 60% of the page (~7.6 pages), no
  unresolved references. Main-text floats: Fig. 1 (RAG vs full context), Fig. 2 (overview), Fig. 3 (diagnosis), Table 1
  (accuracy, four benchmarks), Table 2 (selection without answering, four benchmarks), Table 3 (ablations), Fig. 4 (what
  VGS changes + gate). Appendix: prompts; text benchmarks; robustness (pooled, history-level bootstrap, selection draws,
  reader instruction, second judge, MemLens-64k table, longer histories table, selector-model table); ablation protocol;
  visibility analyses (routing table, swap table, parts of the widened path, shorter index lines); retrieval baselines;
  cost (cost table + figure); reproducibility; VisualMem and CamRoll details; standard metrics and latency; external systems.
- Nothing is running. Spend in phase 17: Gemini about $228, OpenRouter about $17.7.
- Open items for the user: open-source backbone experiment (they will add it; our data: DeepSeek-V3.1 / Qwen3-235B selectors
  give no gain, GPT-5-mini partly); Mem-Gallery / WorldMemArena were not re-run with the final method (scope sentence in
  the limitations); CamRoll full context not run.
## P18-D0: NEW USER REQUEST (11-22 01:04) -- plan (READ FIRST AFTER A CONTEXT RESET)
User: (1) CamRoll result "not very good" + cost high: analyse why; can an algorithm/architecture change to VGS improve accuracy
and cut cost? (2) can the MemLens gap over RAG be increased? (3) Table 1: put the RELATIVE improvement over RAG@40 at the
lower right of our scores, e.g. (79.3-76.4)/76.4; (4) cost in the appendix is fine; (5) consider moving "+ LLM rerank" to the
appendix (without cost next to it readers may think we do not improve on it); (6) Fig. 1: y-axis 60-100.
Plan: a. offline analysis (free): CamRoll accuracy by question type / evidence completeness / path, cost split (selection vs
reading vs cache); MemLens headroom (full context = ceiling for the strong reader). b. option "strong reader always reads
selected U top-40" (no gate for the strong reader): compose offline, compare accuracy and cost on all four benchmarks.
c. cheap run: CamRoll strong standard path WITHOUT the index in the reader context (selection cached; ~$3). d. presentation:
Table 1 subscripts = relative change over RAG@40 in % (labelled as relative in the caption); rerank row -> appendix table with
accuracy AND cost on four benchmarks, main text keeps the honest sentences (strong reader: reranker within 2 points on
short histories, at higher cost); Fig. 1 ylim 60-100 with value labels kept. e. regenerate, compile, report.
Rules: any method change must be one uniform rule on all benchmarks; if CamRoll results are used to choose it, the paper may
no longer call CamRoll held-out for that choice. Budget for this round: keep new API spend under ~$40 unless the user agrees.
## P18-D1: CamRoll analysis + pre-stated test of a cheaper strong path (11-22 01:06)
CamRoll analysis (offline; 1,000 q = 200 semantic + 800 episodic; 705 with gold photos, RAG@40 set complete for 610):
- lite: RAG .724 -> VGS .827: gain everywhere (semantic +6.5, RAG-complete +9.2, RAG-incomplete +16.9): the lightweight
  reader is distracted by 40 photos and reads 7.4 well. Cost $4.7/1k of which selection $4.4 (43.5k cached index tokens).
- strong: RAG .894, VGS std .899, union .907. RAG-complete questions: .902 vs .905 (no room); RAG-incomplete (95 q): .768 vs
  .863 (+9.5). Where VGS's own set is incomplete (107 q): VGS .804 vs RAG .869. Both right 85.9%, RAG only 3.5%, VGS only
  4.0%, both wrong 6.6% (oracle of the two .934). => no headroom: retrieval is complete for 87% and the strong reader
  handles 40 photos. Cost: std strong $12.9/1k = selection $5.6 (56k-token index variant without short ids) + reader call
  that also carries the cached index (56k tokens, ~$4.2) + photos; + cache creation $2.8.
PRE-STATED TEST (before seeing results): drop the index from the strong reader's context (reader gets only the opened
rounds; the strong path can then reuse the lightweight path's selection call and its short-id index cache).
Decision rule: adopt it as the method iff accuracy on the gate-chosen strong path is within 1.0 point of the current one on
each persistent-history benchmark (MemEye standard .793, CamRoll standard .899, VisualMem union .921). Otherwise keep the
current method and report the variant as a cheaper option in the appendix. Runs (results/p18/<bench>/; ~$10):
MemEye + CamRoll: --reader gemini-3.8-flash --cond "toc38:40|read:low" --compact noassist,shortid,telegraph --xcache;
VisualMem: --cond "tocu:40:40|read:low" same flags. Plus CamRoll "oracle|read:low" (annotated photos only) for headroom.
MemLens headroom (to tell the user): strong reader full context .691 = ceiling (65 rounds; RAG@40 reads 62% of them); VGS
.693, RAG .682 -> nothing left for retrieval. Lite: RAG@20 .485 is the best RAG budget; VGS .513.
Presentation changes requested: Table 1 relative-gain subscripts on the VGS row; rerank row -> appendix (keep the sentence
in 5.1); Fig. 1 y-axis 60-100.
- 11-22 01:07: Fig. 1 y-axis now 60-100 (make_figure.py; previous full-axis version kept as make_figure_v2_fullaxis.py); values stay printed on the bars.
- 11-22 01:09: p17_final.main_table now writes Table 1 WITHOUT the rerank row (MAIN_ROWS) and with relative-gain
  subscripts on the VGS row (relative change over RAG@40 in %: ME +10.1 / +3.8, ML +9.1 / +1.6, VM +28.8 / +11.5, CR +14.2 /
  +0.6; results/p17/relative_gain.json), plus sections_polished/tab_rerank.tex (label tab:rerank: RAG@40, RAG@40 + LLM
  rerank, VGS with accuracy AND cost on four benchmarks). The cost table keeps every method. TODO in the text: \input
  tab_rerank in the appendix section "Retrieval Baselines"; change the reference in 5.3 ("bounds re-ranked retrieval end to
  end (Table tab:main)") to tab:rerank; 5.1 sentences about re-ranking point to the appendix table.
- p18 test runs (scripts/p18_runs.sh test; nohup) started: strong reader without the index in its context.
## P18-D2: test result 1 (11-22 01:15): strong reader WITHOUT the index in its context
MemEye standard path: .758 (with index .793; RAG@40 .764) at $4.9 vs $5.2 per 1k -> -3.5 points for a 6% saving. By the
pre-stated rule (within 1.0 point on every persistent benchmark) the variant is NOT adopted; the index stays in the strong
reader's context. CamRoll / VisualMem runs of the test still finish (analysis only; file names in results/p18/<bench>/).
Other cost lever to measure (analysis / ablation column, not a method change): candidate pool of 100 rounds on CamRoll
(ptoc38:100:40) -> also completes the CamRoll column of the ablation table (needs pool QSS, cosine@k, rank@8, rank@k readings).
Presentation done: Fig. 1 axis 60-100; Table 1 relative subscripts; rerank row moved to appendix tab_rerank with costs
(VGS - rerank: lite +6.9/+6.7/+8.0/+9.3, strong +0.3/+1.9/+6.0/-0.2; rerank costs more than VGS in 7 of 8 settings).
## P18-D3: MemLens analysis + second pre-stated variant (11-22 01:18) -- continue here
MemLens-32k test, lightweight reader, by type (RAG@40 -> VGS): info extraction (previnfo 84 q) .643 -> .702; (entity 80)
.475 -> .550; knowledge update (77) .416 -> .429; temporal duration (61) .344 -> .311; temporal info (53) .981 -> .925;
multi-session arithmetic (33) .030 -> .212; entity (31) .484 -> .516; counting (31) .097 -> .194; order ranking (16) .188 ->
.375. Oracle of the two .601. Strong reader: full context .691 is the ceiling (65 rounds; RAG@40 reads 62% of them); VGS
.693; RAG .682 -> no headroom. Only 99 of 323 reference answers have half of their words in the gold rounds' user text
(123 in assistant text). Conclusion for the user: no honest lever found for MemLens; a question-type router would be
test-set tuning.
SECOND PRE-STATED VARIANT "U0" (written before seeing any union-without-index result): the strong reader ALWAYS reads the
selected rounds U the 40 most similar rounds, WITHOUT the index in its context, using the same selection call as the
lightweight path (short-id compact index). No gate for the strong reader; MemLens is unchanged (its union path never had
the index). Motivation: with 40 retrieved rounds as context the index may be unnecessary; cost = RAG@40 reading + one
selection call. ADOPT U0 iff on EACH of MemEye, VisualMem, CamRoll its accuracy is within 1.0 point of the union path WITH
the index (.819, .921, .907). If adopted: strong VGS = U0 everywhere (expected gains over RAG@40 about +5.5, +1.1, +9.5,
+1.3), the gate applies to the lightweight reader only, long-history strong points and the strong latency must be re-run.
If not adopted: keep the current method; offer the user "strong reader always union WITH index" (MemEye .819, CamRoll .907
at about twice the cost on MemEye) as an option. Runs: VisualMem U0 is in 'p18_runs.sh test' (running); MemEye and CamRoll
U0 are the first and third commands of 'p18_runs.sh rest'.
- 11-22 01:18: launched scripts/p18_u0.sh (MemEye + CamRoll U0; STATUS results/p18/logs/STATUS_u0). VisualMem U0 comes from 'p18_runs.sh test' (STATUS_test).
## P18-D4: U0 result on MemEye (11-22 01:26) -> NOT adopted
MemEye, strong reader, selected U top-40 WITHOUT the index: .785 (union WITH index .819; standard with index .793; RAG@40
.764). -3.4 points > the pre-stated 1.0-point tolerance -> variant U0 is rejected; both index-free strong variants lose
about 3.5 points on MemEye, so the index in the strong reader's context is doing real work. CamRoll / VisualMem U0 runs
finish for the record only. The method in the paper stays unchanged.
Remaining honest options to give the user (no silent change): (A) strong reader always takes the union path WITH the index
(MemEye .819, MemLens .693, VisualMem .921, CamRoll .907 -> gains over RAG@40 +5.5 / +1.1 / +9.5 / +1.3) at higher cost on
MemEye and CamRoll; it is a post-hoc choice made with test results in view and must be described as such if taken.
(B) cost lever for nearly complete similarity pools: 100-round candidate pool (the "w/o whole index" ablation) -> run on
CamRoll now, together with the three selection controls, so that the ablation table gets its CamRoll column.
- 11-22 01:28: launched scripts/p18_abl.sh (CamRoll ablation column: pool QSS ptoc38:100:40 -> results/p16/final/camroll/,
  ids via scripts/p16_ids.py camroll, three control readings abl_{cosqss,rank8,rankqss}.jsonl; STATUS results/p18/logs/STATUS_abl).
  scripts/p16_paper.ablation() is now generic over BENCH (headers; camroll special case) and must be called THROUGH
  p17_final (which sets PP.BENCH to four benchmarks): add PP.ablation() to p17_final 'all' and stop running
  'p16_paper.py ablation' standalone once the CamRoll column exists (standalone = three columns).
## P18-D5: CHECKPOINT (11-22 01:36) -- READ P18-D0 .. D5 FIRST AFTER A CONTEXT RESET
Running (nohup): scripts/p18_runs.sh test (CamRoll std-without-index done?, VisualMem U0 at 73%, then CamRoll annotated
evidence 'oracle|read:low'; STATUS results/p18/logs/STATUS_test), scripts/p18_u0.sh (MemEye U0 done = .785; CamRoll U0 at
90%; STATUS_u0), scripts/p18_abl.sh (CamRoll pool QSS at 49%, then ids + three control readings; STATUS_abl).
Decisions already taken: both index-free strong variants are REJECTED by the pre-stated rules (MemEye -3.5 / -3.4 points);
the method is unchanged. Presentation changes done: Fig. 1 axis 60-100; Table 1 relative subscripts + rerank row moved to
appendix table tab_rerank (text updated in setup, 5.1, 5.3, appendix Retrieval Baselines).
WHEN RUNS FINISH: (1) collect results/p18/*/*.jsonl (errors = 0?) and write the numbers here; (2) `.venv/bin/python
scripts/p17_final.py all` (now also runs PP.ablation() for four benchmarks -> Table 3 + full ablation table with a CamRoll
column; check Table 3 fits one column: headers switch to ME/ML/VM/CR); (3) update text: 5.3 ("each of the three benchmarks"
-> four; the sentence "these controls use the three benchmarks on which VGS was developed" must go; add the CamRoll pool
result: w/o whole index on CamRoll), appendix ablation-protocol paragraph, cost paragraph if the pool is a useful cost lever;
(4) optional appendix sentence on the index-free strong reader (MemEye .758 standard / .785 union vs .793 / .819 with the
index) as evidence that the index in the reader context matters; (5) compile, page check (~7.6 pages), view Table 1 with
subscripts and Table 3; (6) report to the user in Chinese: why CamRoll strong is flat (retrieval complete for 87%, strong
reader handles 40 photos; oracle of RAG and VGS only .934), cost split (selection $4.4-5.6 of the total; index 44-56k
tokens), what was tried (two index-free variants: rejected; pool: result), MemLens (no honest lever: strong ceiling .691 =
full context; lite analysis by type), options A/B for the user, presentation changes, spend of this round.
## P18-D6: index-free strong reader, all results (11-22 01:41) -- method UNCHANGED
Strong reader with / without the index in its context (accuracy; $ per 1k before cache overhead):
  MemEye standard .793 ($5.2) / .758 ($4.9); MemEye union .819 ($11.4) / .785 ($11.2);
  CamRoll standard .899 ($12.9) / .908 ($7.7); CamRoll union .907 ($21.7) / .910 ($16.4);
  VisualMem union .921 ($19.9) / .914 ($14.2).  (RAG@40: .764 $9.7, .894 $11.4, .826 $8.8.)
=> the index in the reader's context matters on MemEye (conversations: -3.5 points without it) and not on the photo
collections (CamRoll +0.9 at 40% lower cost, VisualMem -0.7 at 29% lower cost). Both pre-stated rules required <= 1.0
point loss on every persistent benchmark -> REJECTED because of MemEye. Not adopted; reported as a cost option in the
appendix (Cost section) and offered to the user as a disclosed option ("no index in the reader context for photo
collections" would be a rule chosen after seeing test results).
Cost facts for the report: CamRoll lite $5.7/1k = selection $4.4 (43.5k cached index tokens per call) + reading $0.3 +
cache $1.0; strong $15.7 = selection $5.6 + reader call carrying the 56k-token index (~$4.2) + photos + cache $2.8.
A 100-round pool does not cut cost much (4.5k uncached tokens cost about as much as 43.5k cached ones).
Files: results/p18/<bench>/*.jsonl (0 errors).
## P18-D7: CamRoll ablation column + annotated-evidence reading (11-22 01:52)
- CamRoll, lightweight reader (results/p16/final/camroll/): QSS over a 100-round cosine pool .816 (whole index .827 ->
  -1.1); cosine at the QSS size .753 (FSR .641; 8.5 rounds); model ranking top 8 .799 (FSR .828); ranking at the QSS size:
  running. STATUS_test + STATUS_u0 done; STATUS_abl pending (last reading).
- CamRoll, strong reader, annotated evidence only (results/p18/camroll/oracle__read-low__...): .833 overall (episodic .876,
  semantic .660: semantic questions have no annotated photos, so that part is not a ceiling). RAG@40 strong episodic .891,
  VGS .902 -> on episodic questions both retrieval methods already beat reading the annotated photos alone: no headroom.
- 11-22 01:54: CamRoll ranking at the QSS size .829 (FSR .841; 8.5 rounds) -> all p18 runs done (STATUS_abl, STATUS_test, STATUS_u0). Regenerating with p17_final.py all (now includes the four-benchmark ablation).
- 11-22 01:56: text updated for the four-benchmark ablation (5.3: cosine at matched k -4.2 to -12.9 on every benchmark; ranking within 1.3 / 2.1; w/o whole index -3.6 ME, -1.1 CR, -6.0 VM), appendix (protocol sentence for CamRoll, index size, new paragraph 'Index in the reader context' with the index-free strong reader results). Recompiled.
## P18-D8: FINAL STATE of phase 18 (11-22 01:59)
- Index-free strong reader vs RAG@40 and vs the current VGS path (cost incl. cache): CamRoll .908 (RAG .894: +1.4, p=.13;
  current .899) at $8.7 vs RAG $11.4; VisualMem .914 (current .921: -0.7) at $14.9 vs $23.1; MemEye .758 (current .793:
  -3.5 [-6.8,-0.3]). Pre-stated rule failed on MemEye -> method UNCHANGED; reported in the appendix paragraph "Index in the
  reader context". Offered to the user as a disclosed option (drop the index from the reader for photo collections).
- CamRoll ablation column done (pool .816, cosine@k .753, rank@8 .799, rank@k .829); Table 3 + full ablation table now
  cover four benchmarks (generated by PP.ablation() inside `scripts/p17_final.py all`; do NOT run
  `scripts/p16_paper.py ablation` standalone any more: it writes three columns).
- Presentation: Fig. 1 axis 60-100; Table 1 = RAG@40, full context, five systems, VGS with relative-gain subscripts
  (+10.1/+3.8, +9.1/+1.6, +28.8/+11.5, +14.2/+0.6 %); rerank row -> appendix Table (tab:rerank) with costs; cost table
  keeps every method.
- Paper compiled: 17 pages, main text ends on page 8 (Limitations heading at y=270, right column; about 7.6 pages), no
  unresolved references, no overfull tables in the main text.
- Spend this round: Gemini about $25. Nothing is running.
- Regeneration order from now on: `scripts/p17_analysis.py all` -> `scripts/p17_final.py all`.
## P19-D0: NEW USER MESSAGE (11-22 02:52) -- READ FIRST AFTER A CONTEXT RESET
User: (1) the "no index for photo collections" option looks like per-dataset tuning and breaks "one uniform method on several
backbones" -> NOT adopted (agrees with my recommendation; method stays as in P18-D8). (2) Question: does the paper say that
the other three datasets were "designed per dataset" / that the method was developed on them? -> grep and quote the exact
sentences (setup: prompts + tau set on MemEye and 233 MemLens dev questions; "CamRoll was run after the method, prompts, and
threshold were fixed"; 5.4 "CamRoll played no part in setting tau"); offer neutral wording. (3) Wants to know: is there a
more suitable dataset as an option, OR an algorithm/architecture update that improves the index's retrieval (uniformly).
Plan for this turn (cheap, mostly offline; no big runs without the user's decision):
 a. grep paper statements; b. failure analysis of the CamRoll selection (questions where the selected set misses gold photos:
 does the photo description lack the asked detail?); c. compose offline the uniform option "strong reader always reads
 selected U top-40 with the index" on all four benchmarks (accuracy, CI, cost); d. answer with options + recommendation and
 ask the user to choose (new dataset costs about $150-200 with all baselines; replacing CamRoll after seeing its result
 would be selection by outcome -> if another dataset is added, CamRoll must still be reported, e.g. in the appendix).
## P19-D1: offline analysis for the user's question (11-22 02:54); NO method or paper change made in this turn
- Paper statements about development data (exact): 04_setup "The prompts and the gate threshold were set on MemEye and the
  remaining 233 MemLens questions"; "CamRoll was run after the method, prompts, and threshold were fixed."; 05_results 5.4
  "any threshold between the two development sets (0.35 to 0.55)"; "CamRoll played no part in setting tau". Nothing says the
  method was designed per dataset. Note: the strong union path was designed in phase 16 after seeing VisualMem, so do not
  claim VisualMem was held out for the architecture (only for prompts / tau).
- Strong reader, share of gold-annotated questions whose RAG@40 set is incomplete, and union - RAG@40 on incomplete / complete:
  MemEye 9% (+22.7 / +3.8); CamRoll 13% (+10.5 / +0.5); MemLens-32k 35% (+1.8 / +0.7); MemLens-64k 49% (+4.4 / -0.4);
  VisualMem 55% (+17.4 / 0.0). => the strong-reader gain sits on the questions where top-40 retrieval misses evidence; CamRoll
  has few of them. This is the property a "more suitable" dataset would need (low top-40 completeness).
- UNIFORM OPTION "strong reader always reads selected U top-40 with the index" (no gate for the strong reader; gate only for
  the lightweight reader), composed offline: MemEye .819 vs RAG .764 (+5.5 [+2.2,+9.0], p=.002; now .793), MemLens .693
  (unchanged), VisualMem .921 (unchanged), CamRoll .907 vs .894 (+1.3 [-0.3,+2.9], p=.14; now .899). Cost per 1k: MemEye
  $11.7 (now 5.5; RAG 9.7), CamRoll $21.7 (now 15.7; RAG 11.4). Justifiable on the declared development benchmark alone
  (MemEye: union .819 > standard .793); the only test number that moves is CamRoll (+0.8). If adopted: re-run the strong
  long-history points with the union path (about $30), update Table 1 / routing table / Fig. 4c / method 3.2 / 5.1 / 5.4 /
  5.5 / cost table / latency (MemEye strong) / abstract wording; the "cheaper than RAG on MemEye" claim goes away.
- Dataset option: no examined candidate is clearly better (ATM-Bench: one 11k-item corpus with video, open-ended, index too
  large; SMMBench: explicit figure handles; DMV-Bench: interactive). Replacing CamRoll after seeing its result would be
  selection by outcome; if a dataset is added, CamRoll must still be reported. A new dataset with all baselines costs about
  $150-250.
- Waiting for the user's choice. Nothing is running.
## P20-D0: NEW USER MESSAGE (11-22 03:17) -- READ FIRST AFTER A CONTEXT RESET
User: (1) asks whether VisualMem took part in tuning and whether "strong reader relies on VisualMem only" meant the other two
were tuned on. ANSWER: no -- that remark was about significance only. Prompts + tau were fixed with MemEye and 233 MemLens
dev questions; MemLens test, VisualMem, CamRoll were not used for them. Honest caveat to tell: the strong reader's widened
path (union) was redesigned in phase 16 after the first VisualMem run (.851 -> .921). (2) Does NOT want higher cost
(union-for-all option rejected). Wants other ideas at similar cost; ablation rows in "w/o" format. (3) Thinks CamRoll is
saturated (RAG strong ~.90); asks for another dataset like the first three (not too long, image-heavy, RAG not near
ceiling); if none, keep CamRoll. (4) Wording: the setup sentences about tuning data + "CamRoll was run after the method was
fixed" read defensively -> polish across the paper.
PLAN: a. wording: setup keeps only "we report 466 test questions; the other 233 serve as development questions"; tau
rationale moves to method 3.2 as a property of the threshold (between long-message and short-message histories, any value
in [0.35, 0.55] routes identically); drop the CamRoll "run after" sentence and "played no part"; a neutral development note
goes to the appendix (Reproducibility). b. ablation table in w/o format (labels in scripts/p16_paper.ablation; regenerate
with scripts/p17_final.py ablation). c. dataset candidates from the survey list not yet examined (M3Exam, MMRC, Persona-MME,
RefMem-Bench, LCMP, MemLeak): check by web search only; propose a RAG-only pilot before any full run; if CamRoll is replaced
in the main table it should still be reported (appendix). d. no method change this turn.
- 11-22 03:17: P20 edits applied + compiled (17 pages, main text still ends on page 8, Limitations heading y=270):
  setup now says only "We report its 466 test questions; the other 233 serve as development questions." (tau rationale moved to
  method 3.2: separates long-message histories (visibility up to 0.35) from short-message ones (0.55 and above); CamRoll
  "run after ..." sentence and "played no part in setting tau" removed; 5.4 CamRoll sentence neutral); appendix
  Reproducibility got one neutral protocol sentence (prompts + threshold fixed with MemEye and the 233 MemLens development
  questions; strong widened path revised once after the first VisualMem run). Ablation table rows are now in w/o format
  (w/o widened path, w/o standard path, w/o whole index | on the 100-round pool: w/o index reading = cosine at the same
  size, w/o set output = ranking cut at the same size, w/o adaptive size = ranking cut at 8); labels live in
  scripts/p16_paper.ablation; regenerate via scripts/p17_final.py ablation.
  TODO this turn: view Table 3; dataset candidates by web search (M3Exam, MMRC, Persona-MME, RefMem-Bench, LCMP, MemLeak);
  answer the user (VisualMem tuning question; options).
## P20-D1: dataset candidates checked by web search (11-22 03:18) -- continue here
- MMRC (ACL 2025; HF WUUE/MMRC_Real_World_Conversation): conversations of 4-22 turns -> no retrieval problem; NOT suitable.
- M3Exam (arXiv 2606.07402 "M^3Exam: Benchmarking Multimodal Memory for Realistic User-Agent Interactions"; GitHub
  EverM0re/M-3-Exam; anonymous.4open.science/r/M-3-Exam-128D): multi-session user-agent histories with images and PDF
  documents per persona (sessions.json, question.json, images/, pdfs/), questions annotated with supporting-fact rounds,
  tests cross-modal grounding and cross-session reasoning. PROMISING (conversation + images like MemEye/MemLens). To check:
  is the full data downloadable (repo shows example_set), number of personas / sessions / questions, answer format
  (open-ended -> needs a judge), share of image-dependent questions, PDF pages (render to images?).
- Table 3 in w/o format verified in the PDF. Wording edits of P20-D0 applied and compiled.
- Not yet answered to the user: VisualMem tuning question; cost-neutral ideas; dataset proposal (RAG-only pilot first).
- 11-22 03:18: M3Exam GitHub repo (EverM0re/M-3-Exam, Apache-2.0) ships only example_set/ with TWO personas
  (Alex_Veterinarian, Noah_BaristaApprentice); README names no download for the full data (paper footnote: anonymous.4open.science/r/M-3-Exam-128D).
  Next: look at one example persona (sessions.json / question.json sizes, answer format, images) and the paper's statistics
  to judge whether it is usable; if the full data is not public, it cannot be the 4th benchmark.
- 11-22 03:19: M3Exam example persona Noah_BaristaApprentice (files in /tmp/m3exam): sessions.json = list of 17 sessions
  {session_id, date, dialogues}; question.json = 360 questions {question, answer (list of accepted forms), supporting_facts
  (e.g. "D12:1"), type, label}; types ss 50, tr 50, mr 100, fm 40, fj 30, th 30, ms 40, ii 20; 117 image refs + 6 pdf refs in
  the sessions. Only TWO personas are in the public repo. Still to check: rounds per persona, full-data availability
  (paper statistics), whether 2 personas (720 questions) would be enough.
## P20-D2: dataset search result + next exploration (11-22 03:19) -- READ FIRST AFTER A CONTEXT RESET
- M3Exam: example persona has 17 sessions / 210 rounds, only 33 rounds with images and 3 with PDFs, 360 questions that are
  mostly about the dialogue text; the public repo holds TWO personas only. Not suitable (text-dominant, data not released).
  MMRC: 4-22 turns per conversation, no retrieval problem. Earlier: SMMBench (figure handles), ATM-Bench (11k-item corpus,
  video), DMV-Bench (interactive), Mem-Gallery / WorldMemArena / Memory-QA (out of scope by the inclusion criterion).
  => no better fourth dataset found; recommendation: keep CamRoll. Not examined: Persona-MME, RefMem-Bench, LCMP, MemLeak.
- User does not want higher cost and wants a uniform method. Cost-near idea to explore ON MEMEYE ONLY (the declared
  development benchmark): strong reader = selected rounds U the k most similar rounds with the index, small k (10, 20)
  instead of 40 ("tocu:40:10|read+toc:low", "tocu:40:20|read+toc:low"; selection calls are cached). Current MemEye strong:
  standard .793 ($5.2), union-40 .819 ($11.4), RAG@40 .764 ($9.7). If a small k recovers most of the +2.6 at a cost near
  the standard path, propose to the user: one uniform strong path "selected U top-k" for every history (no gate for the
  strong reader), k fixed on MemEye, then re-run MemLens / VisualMem / CamRoll / long histories (about $40) -- only after
  the user agrees. Results go to results/p20/memeye/.
- Done this turn: defensive wording removed (setup / 5.4), tau rationale in method 3.2, protocol note in the appendix,
  ablation table in w/o format; compiled (17 pages, main text ends on page 8).
- 11-22 03:31: P20 MemEye strong, selected U top-10 with the index: .801 (FSR .935, 13.3 rounds, $6.5/1k before cache)
  vs standard .793 ($5.2), union-40 .819 ($11.4), RAG@40 .764 ($9.7). U top-20 running (results/p20/memeye/).
## P20-D3: small-union test on MemEye -> no stable gain; method UNCHANGED (11-22 03:34)
MemEye, strong reader, selected rounds U top-k with the index (results/p20/memeye/; 0 errors): k=0 (standard) .793 ($5.2),
k=10 .801 ($6.5; +3.7 vs RAG@40, p=.03), k=20 .777 ($8.3; +1.3, p=.47), k=40 .819 ($11.4; +5.5, p=.002); RAG@40 .764.
Non-monotonic in k: differences of 1-2 points between these paths are run-to-run noise of the strong reader, so no
cost-neutral variant is supported. Nothing adopted. No better fourth dataset found (P20-D2). Paper changes this turn:
neutral wording (setup, 3.2, 5.4, appendix protocol note), w/o-format ablation table, one explanatory clause in 5.1.
- 11-22 03:34: P20 FINAL: 5.1 got the clause "where the 40 most similar rounds already contain all the evidence for 91%,
  65%, and 87% of the questions" (RAG@40 FSR .914 / .648 / .865). Compiled: 17 pages, main text ends on page 8 (Limitations
  heading y=318, right column; ~7.65 pages), no unresolved refs. Spend this turn: Gemini $4.7. Nothing running.
  Open for the user: keep CamRoll in Table 1 (recommended) or move it to the appendix; whether to examine the remaining
  candidates (Persona-MME, RefMem-Bench, LCMP, MemLeak).
## P21: wording requested by the user (11-22 04:06)
Setup (end of the Benchmarks paragraph): "We chose the prompts and the gate threshold on development data, MemEye and the 233
MemLens development questions. The MemLens test questions, VisualMem, and CamRoll were not used for these choices, so they
test how the same configuration holds on unseen data." 5.4 (lightweight reader): the gate selects the better path on all
three, "including the MemLens test questions and VisualMem, which were not used to set the threshold", and on CamRoll,
"likewise unseen". Accurate for prompts + threshold (lite gate picks the better path on ME / ML-test / VM / CR: .543>.476,
.513>.453, .579>.569, .827>.815). The appendix Reproducibility note still states that the strong reader's widened path was
revised once after the first VisualMem run. Compiled; no method or number changed.
- 11-22 04:07: P21 final wording. Setup: "Prompts and the gate threshold were chosen on development data (MemEye and
  the 233 MemLens development questions); the MemLens test questions, VisualMem, and CamRoll test the same configuration on
  unseen data." 5.4: "... on all three, and also on the unseen CamRoll: its index lines show each photo's description in
  full (visibility 0.99), and the standard path is 1.2 points more accurate than widening." Appendix Reproducibility note
  now lists both design choices revised after first runs on MemEye / MemLens / VisualMem (history-level gate; strong
  widened path = union) and says CamRoll was run with the final configuration. Caveat told to the user: only CamRoll is
  unseen for the whole design; MemLens test / VisualMem are unseen for prompts and the threshold value.
## P22-D0: NEW USER MESSAGE (11-22 04:13) -- READ FIRST AFTER A CONTEXT RESET
User: (1) keep CamRoll in Table 1 (decided). (2) Call an LLM to review the paper and list remaining flaws. (3) Tables 12 and 7
have many empty cells -> fill them. Table 7 = selector models (tab_selectors: accuracy missing for gemini-2.5-flash,
gemini-2.5-flash-lite, GPT-4.1-mini, Qwen3-thinking; cosine missing for Qwen3-235B); Table 12 = retrieval baselines
(tab_retrieval: strong reader missing for top-10/20/hybrid; no CamRoll columns).
PLAN: scripts/p22_runs.sh (nohup; logs results/p22/logs; STATUS file there): e2e lite MemEye runs with selectors 2.5f, 2.5fl,
gpt41m, qwent (results/p22/sel); RAG@10/20(/30) for the missing reader x benchmark cells (results/p22/ret/<bench>/);
hybrid@40 strong on three benchmarks (results/p16/final/<b>/ret_hybrid40_strong.jsonl) and lite+strong on CamRoll
(results/p17/camroll/ret_hybrid40_{lite,strong}.jsonl; ids via scripts/p16_retrieval_ids.py camroll). Estimated ~$60.
Generators: p17_final.retrieval_table4 (four benchmarks x two readers, table*) and selectors_table (accuracy for every
selector, recomputed FSR/cosine from the e2e file when needed). Review: scripts/p22_review.py -> results/p22/review_gpt5.md
(openai/gpt-5 via OpenRouter, text of the PDF) and results/p22/review_gemini.md (gemini-3.8-flash, page images).
Then: update the appendix paragraph "Retrieval Baselines", regenerate (p17_analysis.py all; p17_final.py all), compile,
report the review's main points with my assessment (fix clear errors; list the rest for the user).
- 11-22 04:14: launched scripts/p22_runs.sh (fill runs; STATUS results/p22/logs/STATUS) and scripts/p22_review.py (two LLM reviews -> results/p22/review_gpt5.md, review_gemini.md; log results/p22/logs/review.log).
## P22-D1: GPT-5 review digest (11-22 04:17); full text in results/p22/review_gpt5.md -- continue here
Score: 3 (weak accept). Weaknesses, most serious first: (1) dependence on the Gemini stack; open-weight selectors give no
gain; wants an open-weight selector + reader end to end. (2) Over-claim: contribution 3 says VGS "exceeds re-ranked
retrieval, full-context reading, and five memory systems" without saying this holds for the LIGHTWEIGHT reader only
(strong: FC .810 > VGS .793 on MemEye; rerank .901 > .899 on CamRoll); "at least as accurate as RAG on all four" should
say the three small deltas are not significant. (3) Baselines: LLM rerank limited to a 100-round pool (handicapped on
VisualMem); no multi-hop / iterative retrieval baseline. (4) Gate is crude and history-level; per-question signal is weak.
(5) Assistant replies not indexed: wants an ablation that indexes them. (6) Cost conclusions depend on cached-token
pricing: wants a no-discount sensitivity. (7) Multiple-choice benchmarks, approximate MemLens gold, LLM judges.
Presentation: mark significance in Table 1 or put it in the main text; say "tuned once on dev and held fixed" for the
threshold; one judge summary. Missing experiments ranked: open-weight stack; multi-hop retrieval baseline / larger rerank
pool on VisualMem; assistant-inclusive index ablation; learned per-question router; caption quality / truncation sweep.
MY TRIAGE: fix now (text): contribution wording (2); threshold wording. Cheap experiments worth doing: rerank with a larger
pool on VisualMem (rankr38:300 -> top-40; ~$10), assistant-inclusive index ablation (selection only on MemEye + MemLens:
run select_eval without 'noassist'; ~$5), cost without the cached-token discount (offline from token counts).
Leave to the user: open-weight stack (they plan it), multi-hop baseline, learned router.
Gemini review pending (results/p22/review_gemini.md). Fill runs for Tables 7 / 12 running (results/p22/logs/STATUS).
- 11-22 04:18: P22 fill runs in progress (camroll lite rag:10 done; sel 2.5f/2.5fl/gpt41m near done; qwent e2e slow); Gemini review still pending. Next: collect results, write retrieval_table4 + selectors accuracy in scripts/p17_final.py, apply the text fixes from the GPT-5 review (contribution wording: 'lightweight reader' qualifier; threshold 'chosen once on development data and held fixed').
## P22-D2: Gemini review digest (11-22 04:22); full text results/p22/review_gemini.md -- READ FIRST AFTER A CONTEXT RESET
Weaknesses (claims/evidence): (1) the visibility gate is a benchmark-level switch: every MemEye/CamRoll history is above tau,
every MemLens/VisualMem history below; per-question AUC .47-.63; for the strong reader the gate's choice is WORSE than
widening on MemEye (.793 vs .819) and CamRoll (.899 vs .907) -> wants a dataset with mixed visibility inside one benchmark or
a reframing as a history-level heuristic. (2) fails with open-weight selectors. (3) set output adds nothing over ranking cut
at the same size (Table 3 "w/o set output"), so "Set Reading" is over-emphasised; the gain is from reading the global index.
(4) strong-reader gains are non-significant on three of four benchmarks; rerank beats VGS on CamRoll; must say so in the
abstract / 5.1. (5) cost relies on amortised caches (46-70 questions per history); wants un-amortised cost / streaming case.
Presentation: the two "widened paths" are two different mechanisms under one label; (rest of the review not read yet:
Figure-1 point, unsupported claims, inconsistencies, missing experiments, recommendation -> read results/p22/review_gemini.md
from "Problems of Presentation" on).
STATE OF WORK: fill runs for Tables 7 / 12 still running (results/p22/logs/STATUS when done; qwent e2e slow). First results:
e2e lite MemEye with a weaker selector .463 (one of 2.5f / 2.5fl / gpt41m; read results/p22/sel/*.jsonl), RAG@20 strong
MemEye .723, hybrid@40 strong MemEye .770.
TODO NEXT (in order): (a) when p22 runs finish: extend scripts/p17_final.py: retrieval table on four benchmarks x two readers
(RAG@10/20/30/40, hybrid, rerank, VGS; files: results/p22/ret/<bench>/rag-{10,20,30}__read-{medium,low}__*.jsonl,
results/p16/swap/<bench>/rag-{10,20,30}__read-medium__*.jsonl, results/p16/fig1/<bench>_rag30_strong.jsonl,
results/p16/final/<bench>/ret_hybrid40_{lite,strong}.jsonl, results/p17/camroll/ret_hybrid40_*.jsonl), selector table with
accuracy for every selector (results/p22/sel/toc@{2.5f,2.5fl,gpt41m,qwent}-40__read-medium__*.jsonl; fill Qwen3-235B's
cosine-at-same-size from ranks_memeye.json); update the appendix text of both; (b) text fixes from the reviews: contribution
3 + abstract: say the "exceeds re-ranked retrieval, full context, memory systems" claim is for the lightweight reader and
that strong-reader gains on three benchmarks are not significant; threshold "chosen once on development data and held
fixed"; say plainly that the two widened paths are reader-specific designs; soften "set" emphasis is already in 5.3;
(c) cheap extra controls if budget allows: rerank with a larger pool on VisualMem; assistant-inclusive index (selection
only); cost without the cached-token discount (offline); (d) regenerate, compile, report to the user (Chinese): what the
reviewers said, what was fixed, what remains (open-weight stack; mixed-visibility benchmark; multi-hop baseline).
## P22-D3: Gemini review, second half (11-22 04:22) -- score 2 (GPT-5 gave 3)
- Fig. 1 uses RAG@30 while the paper's baseline is RAG@40 (MemLens gap 3.9 at k=30 vs 0.9 at k=40): label the bars "RAG@30"
  in the figure or use k=40.
- Unsupported claims: "exceeds ... full-context reading" on all four (full context was not run on CamRoll) -> say "where it
  is run"; abstract's gate sentence (gate is a static history-level threshold); "at least as accurate as RAG on all four"
  hides that only VisualMem is significant and that rerank is .901 vs .899 on CamRoll.
- Inconsistencies: Fig. 1 budget vs Table 4; strong-reader gate picks the less accurate path on MemEye / CamRoll while the
  text says the gate selects the suited path; Table 3 is a subset of Table 8 (say so). (Its claim that the Qwen judge flips
  the order of VGS and full context is wrong: both judges give FC > VGS > RAG for the strong reader.)
- Missing: streaming / un-amortised cost; making open-weight selectors work (few-shot, constrained output); non-Gemini
  judge on all benchmarks; full reader x widening factorial in the main text.
- Top three changes it asks for: reframe (the gain comes from reading the global index, not from the set format or the
  gate); a routing mechanism that works within a benchmark, or drop the gate; open-weight viability.
TEXT FIXES TO APPLY (no new runs): contribution 3 + abstract qualifiers; 5.4 strong-reader sentence: say the gate is a
cost choice for the strong reader, not an accuracy choice; Fig. 1 legend "RAG@30"?; Table 3 caption "(subset of Table 8)";
threshold "chosen once on development data and held fixed".
- 11-22 04:23: review text fixes applied (not yet compiled): intro contribution 3 (lightweight-reader qualifier; "full
  context where that is run"; strong reader "significant on long histories ... within noise of RAG elsewhere"); setup
  ("chosen once ... and held fixed"); 5.4 (strong reader: the gate is a cost choice, not an accuracy choice); Table 3
  caption points to Table tab:ablation_full (generator p16_paper.ablation; regenerate with p17_final.py ablation).
  STILL TODO: wait for results/p22/logs/STATUS; extend p17_final.py (retrieval table on four benchmarks x two readers;
  selector table accuracy cells) + appendix text; regenerate; compile; report both reviews to the user.
- 11-22 04:30: generators extended (not yet run): p17_analysis.sel() takes Qwen3-235B / Qwen3-235B-thinking rows from
  their end-to-end files (371 q; cosine at the same size from ranks_memeye.json; thinking row only once
  results/p22/sel/toc@qwent-40__read-medium__*.jsonl has 371 records); p17_final.selectors_table reads accuracies from
  results/p22/sel; p17_final.retrieval_table4(T) writes tab_retrieval.tex as a table* for four benchmarks x two readers
  (results/p17/retrieval_baselines4.json; prints missing cells). After results/p22/logs/STATUS exists: run
  `scripts/p17_analysis.py all` then `scripts/p17_final.py all`, update the appendix paragraph "Retrieval Baselines"
  (numbers; FSR sentence stays for three benchmarks + add CamRoll hybrid/rerank FSR if wanted) and the selector paragraph,
  compile, report.
- 11-22 04:31: p22 fill runs still going (MemLens strong hybrid 36%, MemLens strong RAG@10/20 running, CamRoll strong
  RAG@10/20/30 just started, VisualMem strong RAG@10/20 + hybrid queued, Qwen3-thinking e2e 20%). Finished so far: CamRoll
  lite RAG@10/20/30, MemEye strong RAG@20 (.723) and hybrid (.770), selectors 2.5f / 2.5fl / gpt41m e2e on MemEye.
  When results/p22/logs/STATUS appears: `scripts/p17_analysis.py all` -> `scripts/p17_final.py all` -> fix appendix text
  for Tables 7 / 12 -> compile -> report (reviews: GPT-5 score 3, Gemini score 2; digests in P22-D1..D3).
- 11-22 04:41: p22 runs: MemLens strong hybrid 73%, CamRoll strong RAG@10 68%, Qwen-thinking e2e 36%; memlens_s.log shows no reading progress yet (check it: results/p22/ret/memlens/). Waiting for results/p22/logs/STATUS.
- 11-22 04:50: p22 runs: MemLens strong RAG@10/20 done; CamRoll strong RAG@20 30%, VisualMem strong RAG@10 57%, VisualMem strong hybrid 42%, Qwen-thinking e2e 64%. Still waiting for results/p22/logs/STATUS; then p17_analysis.py all -> p17_final.py all -> appendix text for Tables 7/12 -> compile -> report.
- 11-22 05:00: p22 runs: VisualMem strong hybrid done; CamRoll strong RAG@20 88% (then RAG@30, then CamRoll hybrid ids + lite/strong readings); VisualMem strong RAG@20 46%; Qwen-thinking e2e 80%. Waiting for results/p22/logs/STATUS.
- 11-22 05:10: p22 runs: VisualMem strong RAG@10/20 done; Qwen-thinking e2e 96%; CamRoll strong RAG@30 38% (then CamRoll hybrid ids + lite/strong readings ~15 min). Waiting for results/p22/logs/STATUS; then: scripts/p17_analysis.py all -> scripts/p17_final.py all -> appendix text for the selector table (Table 7) and retrieval table (Table 12) -> compile -> report both reviews (digests P22-D1..D3; text fixes already applied in intro / setup / 5.4 / Table 3 caption).
- 11-22 05:20: p22 runs: Qwen-thinking e2e done (FSR .658, 5.1 rounds); CamRoll strong RAG@30 93%; then CamRoll hybrid (ids + lite + strong readings). Waiting for results/p22/logs/STATUS.
- 11-22 05:29: p22: CamRoll strong RAG@10/20/30 done; CamRoll retrieval FSR@40: dense .865, BM25 .484, hybrid .817,
  rerank .896 (dense@100 .909); CamRoll hybrid lite reading 76%, strong next (~15 min) -> then results/p22/logs/STATUS.
  NEXT (unchanged): `.venv/bin/python scripts/p17_analysis.py all` -> `.venv/bin/python scripts/p17_final.py all` (prints
  missing cells of the retrieval table) -> rewrite the appendix paragraphs for the selector table and the retrieval table
  with the new numbers -> compile (cd paper_acl && ../tools/tectonic -X compile main_polished.tex) -> check pages -> report
  the two LLM reviews (GPT-5: 3, Gemini: 2; digests P22-D1..D3) and what was fixed / what remains.
- 11-22 05:38: p22: CamRoll hybrid lite .702 (FSR .817) done; CamRoll hybrid strong 27% (last p22 job; ~14 min). Then STATUS -> p17_analysis.py all -> p17_final.py all -> appendix text (Tables 7 / 12) -> compile -> report.
- 11-22 05:48: p22: last job (CamRoll hybrid strong) at 79%. NEXT when results/p22/logs/STATUS exists: (1) .venv/bin/python scripts/p17_analysis.py all; (2) .venv/bin/python scripts/p17_final.py all (check 'retrieval table (four benchmarks): missing cells: []' and tab_selectors.tex accuracy cells); (3) rewrite appendix paragraphs 'Selector models' (section Robustness) and 'Retrieval Baselines' in paper_acl/sections_polished/09_appendix.tex with the new numbers; (4) compile + page check; (5) report to the user: both reviews (P22-D1..D3), fixes applied, tables filled, spend (Gemini was 2255.16 before P22).
## P23-D0: NEW USER MESSAGE (11-22 06:37) -- READ FIRST AFTER A CONTEXT RESET
User: (1) check progress; (2) whether finished or not, push the important code + project progress to GitHub:
  git remote add origin https://github.com/Jacob0618/visual-mem.git ; git branch -M main ; git push -u origin main
  (the project has NO git repo yet: git init; .gitignore must exclude .env, cache/, data/, .venv*, baselines/, tools/,
  large results; grep for keys "sk-or-" / "AIza" before committing; never change git config; if no author identity ask or
  use -c user.name/-c user.email for that commit only; if the push fails for authentication, tell the user);
  (3) write a readable .md explaining the current VGS design idea and how it works (no code; for presenting to others)
  -> docs/VGS_DESIGN.md (Chinese).
State: all p22 runs are DONE (results/p22/logs/STATUS). Pending from P22: run scripts/p17_analysis.py all + p17_final.py all
(new four-benchmark retrieval table, selector accuracies), rewrite the two appendix paragraphs, compile, report the two
LLM reviews (digests in P22-D1..D3; text fixes already applied).
Order now: regenerate + compile -> docs/VGS_DESIGN.md -> git init / commit / push -> report.
## P23-D1: Tables 7 / 12 filled (11-22 06:38) -- READ FIRST AFTER A CONTEXT RESET
Regenerated (p17_analysis.py all; p17_final.py all): no missing cells.
- Selector table (MemEye; FSR / rounds / cosine same size / lite accuracy): gemini-3.8-flash .860/7.2/.523/.543; GPT-5-mini
  .854/10.1/.615/.505; DeepSeek-V3.1 .838/17.6/.739/.474; gemini-2.5-flash .712/6.2/.453/.478; gemini-2.5-flash-lite
  .690/10.2/.588/.480; Qwen3-235B-thinking (371 q now) .658/5.1/.445/.509; Qwen3-235B .598/4.8/.404/.477; GPT-4.1-mini
  .536/4.2/.391/.463; RAG@40 .914/40/--/.493.
- Retrieval table (lite/strong; ME | ML | VM | CR): top-10 .449/.670 | .442/.543 | .405/.750 | .754/.853; top-20 .475/.723 |
  .485/.624 | .438/.795 | .749/.884; top-30 .479/.750 | .459/.652 | .438/.806 | .730/.890; top-40 .493/.764 | .470/.682 |
  .450/.826 | .724/.894; hybrid .463/.770 | .459/.700 | .457/.836 | .702/.891; rerank .474/.790 | .446/.674 | .499/.861 |
  .734/.901; VGS .543/.793 | .513/.693 | .579/.921 | .827/.899.
  NOTE for honesty: (i) lite on CamRoll: top-10 (.754) beats top-40 (.724); VGS .827 is still +7.3 over the best budget;
  (ii) strong on MemLens: hybrid .700 > VGS .693. Both must be stated in the appendix paragraph.
TODO NOW (in order, context is short): (1) fix the two appendix paragraphs; (2) compile; (3) docs/VGS_DESIGN.md (Chinese,
no code); (4) git init + .gitignore + commit + push to https://github.com/Jacob0618/visual-mem.git (main); (5) report
(progress, two LLM reviews P22-D1..D3, filled tables, push result, spend: Gemini before P22 = 2255.16).
- 11-22 06:39: appendix paragraphs for the selector table and retrieval table rewritten; compiled: 18 pages, no
  unresolved refs, main text ends on page 8 (Limitations heading y=518, right column; about 7.85 pages after the review
  wording fixes). NEXT: docs/VGS_DESIGN.md -> git init/commit/push -> report.
- 11-22 06:39: docs/VGS_DESIGN.md written (Chinese, 107 lines, no code). NEXT: git init (no repo yet), .gitignore (exclude .env, cache/, data/, .venv*, baselines/, tools/, large results), secret scan, commit, push to https://github.com/Jacob0618/visual-mem.git main; then final report (reviews P22-D1..D3, filled tables P23-D1, spend).
- 11-22 06:41: git repo initialised, commit dca7233 on branch main (457 files, 26 MB; .env / cache / data / baselines / tools / results raw outputs not tracked; secret scan clean), remote origin = https://github.com/Jacob0618/visual-mem.git. PUSH FAILED: no GitHub credentials for HTTPS ('could not read Username'). Ask the user to push (or to log in with gh / a credential helper); do not change git config.
## P24 (11-22 06:47): user: remove the hedged wording of contribution 3 (restore the plain version); find a way to push for them (install gh; auth needs the user's device-code approval; push with 'git -c credential.helper="!gh auth git-credential" push -u origin main' so that git config is not changed). Local commits exist (main; origin = https://github.com/Jacob0618/visual-mem.git); push still pending.
