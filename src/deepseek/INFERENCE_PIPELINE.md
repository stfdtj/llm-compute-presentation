# Inference Pipeline ? Six-step teaching flow

The default lesson now has **6 major steps instead of 23**: Backbone Anchor, input preparation, three Draft Blocks, five candidates, Backbone verification, and acceptance / next cycle. The same symbolic example `我 喜欢 ? 先 ? A B C D E` is used throughout; these are teaching values, not model output.

Each step has three fixed Input / Processing / Output sections. The persistent SVG highlights modules and connecting arrows together. Probability tables, all-position logits, causal mask, cache ledger, per-position Markov details and algorithm selectors are inside a closed Advanced Details disclosure. The computed greedy / probabilistic algorithms remain unchanged. Main-diagram cards similarly use three short Chinese sentences, retaining source detail under Technical Details.

The diagram has separate Anchor and Backbone-feature inputs, separate LM Head and Markov modules, and a Confidence branch fed by hidden states and previous-token features. Candidate IDs and confidence scores reach conceptual Backbone Verification separately; accepted/corrected output supplies the next Anchor.

The default six steps compress the existing-history boundary into the first Backbone step. The advanced example uses cached 我 and current input 喜欢; it does not claim public generate.py splits every prompt this way. At start_pos=0, forward_spec only seeds its private window. Complete speculative serving remains conceptual.

## Deterministic numerical examples

The entire toy vocabulary is `A B C D E X Y Z 先 EOS` (10 symbols, not a truncated real-model vocabulary). All vectors are complete over this toy vocabulary and normalized.

| Default C-divergence dataset | Greedy | Probabilistic |
| --- | --- | --- |
| A, B | Target Top-1 matches → Accept | q=.80, p=.60, alpha=.75, fixed u=.50 → Accept |
| C | Target Top-1 X differs → Reject | q=.80, p=.20, alpha=.25, fixed u=.50 → Reject |
| D, E | Discard; Target predictions already computed | Discard; no later acceptance decision |
| Target correction | X, from Target argmax | Y, from normalized positive residual and fixed u=.82 |
| Valid Target KV after commit | Prompt + 先 + A + B | Prompt + 先 + A + B |
| Next Target input | X | Y |

The residual has r(X)=.70 and r(Y)=.25, plus remaining positive mass on other toy symbols. Its sampling result is intentionally different from Target Top-1. The scalar q(C) and p(C) alone are not used to fabricate a replacement distribution. Full p/q vectors are retained. Probability mode uses T=1 with no top-k/top-p; q includes the Markov bias. Greedy mode compares Target argmax IDs, not raw-logit ratios. Other datasets exercise all accepted, first rejected and last rejected cases. All-accepted uses the distribution at input E to select Target bonus Z, which has no Target KV of its own until the next Forward.

## Source accuracy and boundaries

| Category | Evidence / representation |
| --- | --- |
| Verified architecture | `models/DeepSeek-V4.1-Flash/config.json`: 40 main layers, 3 Draft Blocks, 5 slots, 128-token Draft window, selected layers 37/38/39. |
| Verified feature extraction | `Transformer.forward`: features collected before those layers execute; mean over mHC streams, then concatenate. `forward_embed`: main_proj + main_norm; shared embedding; five slots expanded to four mHC streams. |
| Verified Draft head | `forward_head`: collapse mHC + Norm + shared Head for five base-logit rows; previous token drives rank-256 Markov embedding/bias; sequential sample loop; confidence projection uses hidden and Markov embedding. Draft current slots can see each other; Target causal restrictions are different. |
| Report architecture | `technical_report.txt` §2.2: CED encoder 20 / decoder 20, global KV projected from encoder features, decoder SWA bounded replay. §2.4.3: semi-autoregressive Draft and confidence/load-aware scheduling. |
| Standard algorithms | Greedy token comparison and speculative sampling with residual correction: [Leviathan et al.](https://arxiv.org/abs/2211.17192). Semi-autoregressive Markov drafting and confidence scheduling: [DSpark](https://arxiv.org/abs/2607.05147). |
| Hypothetical teaching data | All logits, Markov biases, probabilities, draws and resulting symbols. The toy base-logit/bias decomposition illustrates the verified operations; it is not a model measurement. Chinese prompt word blocks and A–E are symbolic tokenizer stand-ins. |
| Unimplemented runtime details | `forward_spec(start_pos=0)` only seeds private Draft window KV and returns None. The reference demo generates drafts only on subsequent positive-start calls. The illustration now explicitly uses a positive-start current Backbone Decode before drafting, matching the implemented module dependency. Complete Verification, positional/state handoff, confidence scheduling and rollback/commit still need serving-runtime support; the figure is not runnable model inference. |

The reference Target Forward still executes all 40 layers; optimized CED Prefill is described from the report rather than claimed as implemented there. Its Head defaults to last-position logits, so a Verification runtime would need to request all required positions. The conceptual Target Verification uses the same Target architecture as Decode, but is not asserted to have the same workload as optimized Prefill. KV cells show logical validity, not physical memory layout, copy, clearing or rollback operations. Other model state must also remain aligned. The visualization stops at the next-cycle entry and does not execute a model, produce a real EOS continuation, or report measured speedup.

## Build and validation

- Build: `.venv/Scripts/python.exe src/deepseek/build.py`.
- Numerical tests: `npm run test:pipeline`. Checks 8 algorithm/dataset combinations, normalization, base+Markov composition, fixed sampling, argmax, position alignment and committed/pending state. Also checks the full categorical identity `accepted draft mass + residual correction mass = Target distribution` at every toy position.
- Browser: `npm run test:browser:pipeline` checks all six steps at four sizes (24 scenes), fixed Input/Processing/Output, collapsed advanced content, changing arrow highlights, persistent SVG nodes, eight algorithm/dataset outcomes, causal prediction alignment, playback and tab pausing.
- Main diagram: `npm run test:browser:dspark` checks independent input/output connections, nine card topics, collapsed technical detail, keyboard dismissal, bounds, and links.
- Screenshots are generated under `qa/pipeline/` and `qa/dspark_overview/` (ignored PNGs). Both desktop sizes were visually inspected. Validation JSONs record the latest results.

Editable sources: `inference_pipeline_data.js` (pure teaching math), `inference_pipeline.js` (interaction and persistent graph), and `inference_pipeline.html` (layout). `build.py` embeds them into the existing standalone HTML. Model files and calculation data are unchanged. The connected DSpark branch is added to the architecture SVG while all rectangle/path geometry outside the DSpark group is preserved. The SVG preservation hash is refreshed only after this structural containment check; all other pinned calculation-source hashes remain unchanged.


## Connected main architecture overview (2026-10-21)

The top overview now has a compact DSpark branch in its free upper-right area, with the existing Backbone boxes in their original positions. A selected Anchor from the Backbone prediction and a grouped feature tap before layers 37/38/39 both feed the builder. The branch shows `[Anchor + 4 Noise]`, three mtp blocks over five positions, shared-Head base logits, sequential Markov refinement, five candidates and Confidence Head output. A dashed edge leads to a clearly labeled conceptual Scheduler / Target verification consumer. Existing encoder/decoder and calculation geometry remains protected by the repository validator.

Click cards cover Anchor, features, input construction, Draft blocks, Markov Head, Confidence Head, Verification and implementation status. Cards link to the existing enlarged DSpark structure and Inference Pipeline. Longer explanations stay out of the main drawing, and separate cards do not change parameter/FLOPs ledgers.

The public Hugging Face files were inspected on 2026-10-21:

- [inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py): implements Backbone Forward and DSpark Forward; features are collected before the specified layers, and the Anchor comes from Backbone `self.head(self.norm(h))` followed by sampling. `forward_spec(0)` seeds private caches, returning None.
- [inference/generate.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/generate.py): `generate` processes the initial prompt chunk, then calls `model.forward(...)[0]` each round and writes one next token. It does not call `forward_spec` or provide the full speculative verification loop. The page labels the complete serving loop as conceptual.

Additional validation: `npm run test:browser:dspark` checks the connected branch, popup content, keyboard dismissal, panel bounds, text containment, detail/pipeline links and Backbone-before-Anchor order at 1366x768, 1920x1080, 768x900 and 390x844. Screenshots are in `qa/dspark_overview/` (ignored PNGs); `qa/dspark_overview/validation.json` records results. The main architecture and Anchor cards were inspected at both desktop sizes.

New editable sources: `dspark_overview.py` and `dspark_ui.html`. Rebuild with the same `build.py` entry point. No model parameters or existing calculation formulas are modified.

## Simplification validation (2026-10-21)

Local reference model.py and inference/config.json were checked. generate.py is absent from the local reference directory, so its official Hugging Face raw source was checked: the loop calls model.forward and never forward_spec. Six lessons and both verification algorithms are exercised by the browser tests, with screenshots at 1920, 1366, 768 and 390 pixels. The SVG preservation baseline was refreshed only after comparing all elements outside dspark-overview (the only other intentional change is the prediction caption).
