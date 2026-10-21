# Inference Pipeline — Detailed Data Flow

The existing **Inference Pipeline** tab in `deepseek_v4_1_interactive.html` now follows one fixed example: Prompt `我喜欢`, Target anchor `先`, and five symbolic Draft candidates `A B C D E`. Open the standalone HTML directly with `file://`; no server or network assets are required.

The SVG overview is created once and stays visible while the detail panel scrolls. Its persistent connections carry Prefill outputs into anchor selection, DSpark features and Target KV reuse, then Draft IDs/q into Target input, Target distributions, aligned comparison, confirmation and the next iteration. Every action names **Input → Operation → Output → Next consumer**. Controls provide previous/next, deterministic play/pause, reset, chapter navigation, candidate inspection, Normal/DSpark and Greedy/Probabilistic modes.

## What changed

- Prefill retains its initial Logits/distribution, selected hidden features and valid Prompt KV throughout the demonstration.
- Three Draft Blocks are shown as a **3-row × 5-position** computation. Input slots are `[先, noise, noise, noise, noise]`. After three blocks, the shared vocabulary Head produces five base-logit rows. Five separate Markov actions show the previous token, its bias, the resulting full q distribution, fixed sampling value, and selected candidate.
- The handoff explicitly constructs `[先,A,B,C,D,E]` plus existing Prompt KV. Those six IDs enter Target Embedding and the 40-layer Backbone. Resulting new KV positions are provisional.
- Selecting C highlights input B, Target p3 and Draft q3 together. The position table and q/p inspector appear side by side on desktop. Contexts, logits and full Softmax vectors can be expanded.
- The former independent outcome selector is replaced by **Target distribution datasets**. All decisions are computed from those distributions; selecting a dataset changes the input data, not an arbitrary result flag.
- State views separate confirmed text, committed Target KV, provisional/invalid KV and the new pending Target correction/bonus token. Later candidate predictions remain visible after a rejection but are not evaluated for acceptance or committed.
- Normal Decode follows the same Target greedy distributions one position at a time, stopping at the first correction or the all-matched bonus boundary; it does not continue using distributions conditioned on a rejected Draft suffix.

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
| Unimplemented runtime details | `forward_spec(start_pos=0)` only seeds private Draft window KV and returns None. The reference demo generates drafts only on subsequent positive-start calls. The depicted immediate first-anchor handoff, complete Verification, positional handoff, confidence scheduler and rollback/commit lifecycle need runtime support; the figure is not runnable reference-model inference. |

The reference Target Forward still executes all 40 layers; optimized CED Prefill is described from the report rather than claimed as implemented there. Its Head defaults to last-position logits, so a Verification runtime would need to request all required positions. The conceptual Target Verification uses the same Target architecture as Decode, but is not asserted to have the same workload as optimized Prefill. KV cells show logical validity, not physical memory layout, copy, clearing or rollback operations. Other model state must also remain aligned. The visualization stops at the next-cycle entry and does not execute a model, produce a real EOS continuation, or report measured speedup.

## Build and validation

- Build: `.venv/Scripts/python.exe src/deepseek/build.py`.
- Numerical tests: `npm run test:pipeline`. Checks 8 algorithm/dataset combinations, normalization, base+Markov composition, fixed sampling, argmax, position alignment and committed/pending state. Also checks the full categorical identity `accepted draft mass + residual correction mass = Target distribution` at every toy position.
- Browser: `npm run test:browser:pipeline`. Drives real controls; verifies persistent SVG identity and connected edges, all three blocks/five slots, all five proposal stages, prediction shift and causal masking, all comparison outcomes, residual correction, KV boundaries, normal Decode, offline use, playback, tab isolation and Shape Trace restoration. Checks 36 scenes at 1366×768, 1920×1080, 768×900 and 390×844, including expanded numerical tables.
- Existing regressions: `npm run test:browser:deepseek`, `npm run test:browser:walkthrough`, `npm run test:browser:layout`.
- Actual screenshots are generated under `qa/pipeline/` and ignored by Git. Machine-readable coverage is in `qa/pipeline/validation.json`. Desktop screenshots at 1366×768 and 1920×1080 were visually inspected, including the 3×5 block matrix, input-position/q/p alignment, acceptance table and residual correction. A narrow-screen overflow with expanded vectors was found and fixed.

Editable sources: `inference_pipeline_data.js` (pure teaching math), `inference_pipeline.js` (interaction and persistent graph), and `inference_pipeline.html` (layout). `build.py` embeds them into the existing standalone HTML. Model files, calculation tabs and architecture SVG are unchanged.
