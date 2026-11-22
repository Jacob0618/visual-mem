1. Summary
The paper introduces Visibility-Gated Set Reading (VGS), a training-free memory architecture for multimodal agent memory. VGS separates evidence selection from reading: a selector reads a compact per-round index of the full history to select an evidence set (query-time set selection, QSS), and a “visibility gate” routes histories to either a standard reading path (open only the selected rounds) or a widened path (augment access, different for lightweight vs. strong readers) depending on how much of the history’s content the index retains. The central claim is that simple similarity-based RAG fails to retrieve complete multimodal evidence sets (especially when evidence spans multiple images and cross-round references), and VGS recovers more complete evidence and improves accuracy. Evidence includes: (i) selection analyses showing higher full-set recall than cosine retrieval (Table 2, Table 8), (ii) main accuracy gains over RAG@40 and five memory baselines on four multimodal benchmarks (Table 1), (iii) ablations that attribute gains to whole-index reading and the visibility-gated reading path (Table 3, Figure 4), and (iv) cost/latency analyses (Table 11–13, Figure 5).

2. Strengths
- Clear, important problem: similarity retrieval misses multimodal evidence spread across several images and conversational references (Section 2; Figure 3).
- Method is conceptually simple, training-free, and well-motivated: global navigation over a compact index plus routing by index visibility (Section 3; Figure 2).
- Thorough experiments on four multimodal benchmarks with detailed diagnostics: full-set recall, selection vs. answering, long-history stress tests, re-ranked retrieval baselines, multiple memory systems, and cost/latency (Sections 4–5; Tables 1–3, 8–16; Figures 3–6).
- Ablations pinpoint what matters (whole-index reading improves selection; widening helps low-visibility histories; widened path must be matched to reader capacity) (Section 5.3–5.4; Table 3; Figure 4; Tables 8–10).

3. Weaknesses (a) Substantive issues with claims/evidence (b) Presentation issues
1) Heavy dependence on proprietary Gemini stack; limited evidence of generality (a)
- What/where: Results and gains depend heavily on gemini-3.8-flash for selection and readers (Sections 4–5; Table 7). Open-weight selectors (Qwen3-235B, DeepSeek-V3.1) did not improve over RAG@40 on any benchmark, and even GPT-5-mini underperforms on VisualMem (Appendix C, Table 7).
- Why it matters: Core claims about the effectiveness of VGS’s global navigation and visibility gate may hinge on model-specific long-context reading and instruction-following. Without a strong open-weight selector/reader, the method may not be practical or reproducible.
- Fix: Provide end-to-end results with at least one strong open-weight vision-language reader and an open-weight selector (e.g., Qwen-VL or LLaVA-Next for the reader; a long-context open-weight LLM for selection). Alternatively, distill the selector with supervised data or prompt-program the selector to be less model-sensitive; report results. At minimum, demonstrate VGS’s gains with a non-Gemini reader.

2) Overclaim relative to significance and breadth of improvements (a)
- What/where: Many reported gains with the lightweight reader are small/marginally significant (MemEye +5.0 points, p=0.055; MemLens +4.3, p=0.058; Section 5.1), and with the strong reader they are mostly modest except on VisualMem (+9.5 points). Yet the contributions (page 2) state that VGS “exceeds re-ranked retrieval, full-context reading, and five recent memory systems” without the qualifier that this holds for the lightweight reader only (Table 1, Table 11).
- Why it matters: The paper risks overselling consistent superiority. In practice, with strong readers on shorter histories, VGS ≈ RAG@40 (Section 5.1), which weakens a universal recommendation to adopt VGS.
- Fix: Calibrate claims throughout to match significance and scope. Explicitly state that exceeding full-context reading holds for the lightweight reader (Table 1: MemEye .543 vs .500; MemLens .513 vs .446; VisualMem .579 vs .454), and that with strong readers VGS mainly helps on long histories (VisualMem) and is otherwise comparable to RAG/LLM-rerank.

3) Fairness and completeness of baselines for retrieval and memory systems (a)
- What/where: The “LLM rerank” baseline re-ranks only the top-100 cosine candidates (Table 11), which under-covers VisualMem evidence (Section 5.3). Multi-hop retrieval methods are not included, and dense+BM25 fusion did not use iterative/hop-aware retrieval (Table 12). Competing memory systems are run with shared components but possibly non-optimal settings (Table 16).
- Why it matters: Since VGS’s advantage partly comes from scanning the whole index (Table 3; “w/o whole index”), retrieval baselines constrained to top-100 candidates are handicapped on long histories. Stronger multi-hop or expansion-based retrievers (e.g., Hotpot-style multi-hop, query expansion/self-ask) could narrow the gap.
- Fix: Add multi-hop retrieval baselines (e.g., iterative retrieve-expand-retrieve or multi-hop dense retrievers) with budgets matching VGS’s average opened rounds (Table 2, Table 8). For LLM reranking, allow access to a larger prefilter pool on long histories (VisualMem). Document per-system hyperparameters and run sensitivity sweeps to ensure solid baselines.

4) The visibility signal is crude and history-level; the per-question routing is weak by the authors’ own analysis (a)
- What/where: The gate uses H(h), a bag-of-words overlap between index line and round content (Eq. 2), applied per history; per-question visibility ˆv(q) is almost uninformative (AUC 0.47–0.63; Table 9). The widened path benefits differ across readers (Table 10), so routing matters.
- Why it matters: A method contribution is the gate. As implemented, it does not reliably identify which questions need widening (Table 9). The useful signal appears to be “long user messages were truncated,” not a principled measure of missing visual content or cross-round references.
- Fix: Train a lightweight classifier (or even a learned threshold) to predict widening benefit from richer signals (e.g., presence of references, counts/comparisons required, entropy of selector confidence, cross-round coreference cues) and evaluate per-question routing against the oracle (Table 9). Show gate robustness across datasets without retuning.

5) Limited analysis of content indexed, especially omission of assistant replies (a)
- What/where: Assistant replies are not indexed because they “mostly restate the user’s content” (Section 3.1). The paper itself notes that “evidence stated only by the assistant must be found through the user’s turns” (Limitations).
- Why it matters: In realistic agent logs, assistant messages often paraphrase, normalize, or compute over user-provided artifacts and can be the only textual surface form that bridges image content to query wording. Excluding them could systematically miss cross-turn references and distort the visibility metric.
- Fix: Add an ablation that includes assistant replies in index lines (and/or a separate assistant index) and measure impacts on selection recall (Table 2/Table 8) and route decisions (Figure 4d). Report changes on MemLens (where evidence often lies beyond 200 chars) and on MemEye cross-reference cases.

6) Cost conclusions hinge on provider-specific caching terms and may not generalize (a)
- What/where: A key cost advantage vs. RAG@40 with strong reader on MemEye and parity on MemLens relies on cheap cached-token input for the index (Table 11, 13; Section 5.5). On longer histories (VisualMem, CamRoll), VGS is more expensive than retrieval due to large index reads (Table 13).
- Why it matters: The cost advantages depend on a specific provider and cache pricing; others may not offer identical caching or discount. This weakens the practical deployment claim.
- Fix: Add a sensitivity analysis with no cached-token discount and with varying index compression rates (Appendix G briefly mentions compression without changing recall). Also show costs when the index is included/excluded from the strong reader’s context across tasks (Appendix G mentions small accuracy changes on photo datasets).

7) Benchmark coverage and evaluation biases (a)
- What/where: VisualMem uses generated images and multiple-choice QA; CamRoll uses real photos but multiple-choice as well (Appendix I). MemLens gold evidence is approximated from session annotations (Appendix E), possibly omitting context. Judging relies on LLMs (Gemini; Qwen for MemLens), with known variance (Appendix C).
- Why it matters: Multiple-choice can inflate differences through option priors; approximate gold can mis-estimate full-set recall. LLM judging variability (Appendix C) further blurs small gains.
- Fix: Add free-form QA evaluation for at least one image-centric benchmark with human spot-checks; or report calibration via a third independent judge family. Where possible, refine MemLens gold evidence through manual spot-audits of disputed cases.

(b) Presentation issues
- Some claims are stated globally (e.g., “exceeds re-ranked retrieval, full-context reading,” page 2) without clarifying that this is with the lightweight reader; rephrase throughout to avoid overgeneralization.
- The gate’s intuition would benefit from a compact diagnostic example showing where H(h) diverges from true “visibility” due to cross-round referencing or assistant-only mentions.
- Place the key statistical significance results in the main text (currently scattered between Section 5.1 and Appendix C), and mark significant/non-significant deltas in Table 1 for immediate clarity.

4. Claims that are not fully supported
- “VGS … exceeds re-ranked retrieval, full-context reading, and five recent memory systems.” (page 2, contributions) This holds clearly for the lightweight reader (Table 1, Table 11) but not for the strong reader (e.g., MemEye: VGS .793 < Full context .810; CamRoll: rerank .901 > VGS .899).
- “With a strong reader, it is at least as accurate as RAG on all four…” (page 2) True numerically in Table 1, but the deltas on MemEye (+2.9), MemLens (+1.1), CamRoll (+0.5) are small and reported as not statistically significant (Section 5.1); the statement should be softened to reflect uncertainty.

5. Internal inconsistencies
- Judges: The main text says “using a Gemini judge for MemEye and Qwen3-VL-235B for MemLens” (Section 4), but Section B also refers to strict vs. lenient judges and switching to the LongMemEval strict judge for LoCoMo/LongMemEval-S. While not contradictory, the judging setup is complex and easy to misinterpret; consolidate clearly in the main paper which judge is used where.
- Figure 1 vs. main results: Figure 1 uses gemini-3.8-flash as reader for RAG comparisons on text/multimodal with different retrieval budgets (100 turns on LoCoMo, 30 rounds elsewhere), whereas main results fix RAG@40 (Table 1). This is methodologically fine but could be read as inconsistency without a brief note.
- “The gate uses one threshold for all benchmarks” (Section 3.2) vs. “Prompts and the gate threshold were chosen on development data” (Section 4). This is not a contradiction, but the phrasing should explicitly say “tuned once on dev and held fixed for all test benchmarks.”

6. Missing experiments or controls (ordered)
- Open-weight stack: End-to-end results with an open-weight long-context selector and at least one open-weight strong multimodal reader to demonstrate generality.
- Multi-hop retrieval baselines: Iterative retrieve-expand-retrieve and multi-hop dense retrievers with larger candidate pools on long histories; compare at matched round budgets and costs.
- Assistant-inclusive index ablation: Include assistant replies in index lines and reassess selection recall, gate routing, and accuracy.
- Learned per-question router: Train a small classifier for widening vs. standard path using features available at query time; compare against the history-level gate and oracle (Table 9).
- Index content/quality: Vary image captioning quality (longer/denser captions, CLIP tags) and index truncation lengths; report the trade-off between index size/cost and selection recall (beyond Appendix E’s short analyses).
- Larger candidate pool for LLM rerank on long histories (VisualMem) to make the comparison fair against whole-index QSS.

7. Questions for the authors
- How sensitive is QSS to prompt phrasing and to “low thinking” vs. “no thinking” settings for the selector? Can you provide selection stability metrics beyond the three draws in Appendix C?
- Could you report results when assistant replies are indexed? Given cross-round references often originate in assistant turns, this seems important.
- For long histories (e.g., VisualMem at ~44k-token index), what are the selector’s actual context limits and truncation behavior? Do any histories exceed the selector’s context window?
- For LLM reranking on VisualMem, what happens if you expand the prefilter pool beyond 100 rounds (e.g., 500) or add a second hop?
- How often does widening actually remove helpful context for the lightweight reader (e.g., when notes mis-extract values)? Can you quantify note extraction precision/recall on a subset?
- Can you share a breakdown of error types where VGS fails despite selecting the full gold set (e.g., visual perception errors vs. reasoning vs. note extraction errors)?

8. Overall recommendation: 3 (Weak Accept)
Justification: The paper addresses an important and timely gap: similarity retrieval is insufficient for multimodal, cross-image evidence, and a training-free selection-over-index approach with adaptive reading is compelling. The experimental study is broad and careful, with solid analyses tying gains to better evidence-set recovery and appropriate reading paths (Tables 1–3; Figures 3–4). However, the dependence on the proprietary Gemini stack (particularly for the selector) and the lack of strong open-weight baselines reduce confidence in generality. Some claims are overstated relative to significance and should be calibrated. Retrieval/memory baselines on long histories could be stronger (multi-hop/expanded-candidate rerank).

Most impactful changes to raise the score:
- Demonstrate VGS with at least one open-weight selector and reader, or provide a distilled selector that recovers most gains.
- Add a competitive multi-hop retrieval baseline with expanded candidate pools on long histories, and/or increase the reranking pool size on VisualMem.
- Include an ablation indexing assistant messages (and/or improved image captions) to validate that the index design is near-optimal.