# Inference Pipeline

`deepseek_v4_1_interactive.html` now includes an **Inference Pipeline** tab beside Model Architecture and Step-by-step · CSA2. Open the HTML directly; no server or external runtime assets are needed.

The tab shares one Prompt / Prefill / Target-anchor sequence between Normal and DSpark modes. Normal Decode processes one pending token per iteration. DSpark separates Draft, Confidence Scheduling, Target Verification, Accept / Reject, state commit, and the next-cycle entry. Controls support step navigation, deterministic playback, reset, four outcomes, and verification lengths 1–3. Switching views pauses the departing animation and preserves its step; Shape Trace restores its prior view.

Verification shows the input-to-next-token shift, Target Backbone → LM Head / Logits → Softmax, a causal-visibility matrix, and the serial-target-round comparison. The fixed example uses prompt `我喜欢`, a Target-generated anchor `先`, and Draft `[吃] [苹果] [呀]`. A Target bonus or correction remains a pending input until the next Forward computes its KV. Rejects stay visible, including the discarded suffix. The probabilistic explanation includes residual-distribution replacement sampling.

## Verified model behavior

| Evidence | Behavior represented |
| --- | --- |
| `models/DeepSeek-V4.1-Flash/config.json` | 40 main layers, 3 DSpark blocks, block size 5, target feature layers 37/38/39, Draft window 128. |
| `model.py`: `Transformer.forward`, `forward_spec` | Main Forward iterates only the 40 backbone layers. DSpark runs separately through `mtp`; structure indices 40–42 do not add mandatory ordinary Decode layers. |
| `model.py`: `forward_embed`, `forward_head`, `DSparkConfidenceHead` | Features at the attention inputs of target layers are averaged over mHC streams and concatenated/projected. Shared embedding/head, anchor-plus-noise slots, parallel base logits, lightweight sequential Markov sampling, and returned candidates/logits/confidence. The confidence projection returns raw scores; the page invents no calibrated numbers. |
| `technical_report.txt` §2.2 | CED uses a 20-layer causal encoder and 20-layer decoder. Decoder global KV comes from encoder features; Decoder local SWA still requires bounded replay. |
| `technical_report.txt` §2.4.3 | Five draft positions, three blocks, conditional acceptance prediction, prefix-survival estimates and profiled engine throughput curves for scheduling. |

## Teaching boundaries and limitations

- The reference `Transformer.forward` still executes all 40 main layers. Optimized CED Prefill is described from the report, not claimed as implemented in that function.
- `forward_spec(start_pos=0)` only seeds private Draft window caches and returns `None`. The repository does not implement the complete Scheduler / Target Verification / Accept-Reject / state-rollback loop.
- The multi-position Target Forward, KV commit/discard and next-cycle organization are conceptual runtime illustrations. Reference `ParallelHead` defaults to the last position; a verification runtime must expose all required positions. KV labels describe logical validity, not specific memory operations.
- The mask illustrates the causal restriction; real CSA2/SWA attention has additional sparsity/window restrictions. Draft slot visibility is distinguished from Target causal visibility.
- The UI uses three readable Chinese word blocks in place of the configured five draft candidates and actual tokenizer IDs. Length selection is manual teaching input, not a production Scheduler. The four outcomes represent different hypothetical Target decisions. No model is executed, no speedup is measured, and playback uses no random sampling.
- One complete cycle and its next-cycle entry are illustrated. Generation itself is not executed through EOS. At short viewport heights, explanations remain available through vertical scrolling.
- Greedy and probabilistic verification are standard speculative-decoding theory, not asserted as an implemented model-specific runtime. Sources: [Leviathan et al.](https://arxiv.org/abs/2211.17192), [DSpark](https://arxiv.org/abs/2607.05147).

## Validation

Build: `.venv/Scripts/python.exe src/deepseek/build.py`.

Browser checks:

- `npm run test:browser:pipeline`: four outcomes; all scheduled lengths; correct probability alignment and causal matrix; accepted-prefix/cache/pending-token invariants; baseline Decode; reset/play/pause; keyboard navigation; tab isolation; Shape Trace restoration; offline file access with no network requests. All nine scenes checked at 1366×768, 1920×1080, 768×900 and 390×844.
- `npm run test:browser:deepseek`: architecture interactions, parameter panels, all 40 Shape Trace layers, indexer, and attention blocks.
- `npm run test:browser:walkthrough`, `npm run test:browser:layout`, `node src/deepseek/verify_walkthrough_motion.cjs`: existing CSA2 interaction, layout and animation regressions.

Actual screenshots are generated under `src/deepseek/qa/pipeline/` (ignored image artifacts). Screenshots at 1366×768 and 1920×1080 were inspected, including the detailed Verification and middle-Reject scenes; phone layout was also inspected. Machine-readable pipeline results are in `qa/pipeline/validation.json`. Existing model parameters, calculation content and architecture SVG are preserved.

Editable sources: `inference_pipeline.html` and `inference_pipeline.js`. `build.py` embeds them into the standalone HTML. The existing walkthrough tab controller handles all three top-level tabs.
