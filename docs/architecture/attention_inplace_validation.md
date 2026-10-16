# Original architecture: in-place Attention refinement

Base: `eb9356cdfb341955b2018f25b4ca3f806cb1b590`. Visual reference inspected: `D:/working/sharing/architecture_DSv4.1.png`.

The original page, header, calculation inspector, zoom/pan controls and 1672×941 overview remain. The original positions of the overview, per-layer computation, two Attention blocks, HC/MoE and Engram blocks are retained. The tabbed replacement from the earlier attempt is removed.

## Final behavior

The original Encoder and Decoder Attention blocks now show independent Query and Local KV paths, configured source/reuse behavior, inline Indexer selection and their common Attention output. **Enlarge detail** and the original C · Encoder / C · Decoder focus menu reveal larger drawings within the **same architecture SVG**. No new page or modal is used for the diagrams. Indexer boxes highlight inline; the explicit **Full Indexer calculations** link preserves the Full / Reindex / Reuse teaching tool from `eb9356c`. Parameter, Forward and Backward modes still use the existing calculation ledger.

The larger blocks include low-rank Query projection/RMSNorm, own Local KV/RoPE/window cache, source Compressor, Index Query/Key projections, learned per-head weights, weighted ReLU scoring, causal/candidate masks, Top-k positions, KV gathering, sink softmax, inverse RoPE and grouped output projection. Decoder-only candidate-block masking is excluded from the Encoder path.

The top-level boundary keeps the original position but labels the actual four-stream state and delayed `pre_mix`. Its main-state arrow bypasses the Compressor. The Compressor at layer 20 consumes that layer's HC-merged/normalized input; it is not the only KV source. The overview annotates KV owners 2/8/14/20 and index updates 20/24/28/32/36. A small DSpark entry opens an original-menu focus region containing three distinct `mtp` stages, inputs, private draft KV, heads, outputs and invocation limits.

## Source verification

Inspected `models/DeepSeek-V4.1-Flash/{model.py,config.json,engram.py}` against pinned `references/deepseek/inference/` and the existing audit. SHA256 hashes, function locations and all 40 configured layer roles are recorded in `src/deepseek/source_view_evidence.json`.

| Diagram relationship | Source support |
| --- | --- |
| One 40-Block backbone, Engram before Blocks 1/14, four HC streams, final collapse/Norm/Head | `Transformer.forward()` |
| HC Pre/Norm → Attention → HC Post, delayed Pre into FFN and next Block | `Block.forward()`, `hc_mixes()`, `hc_pre()`, `hc_post()` |
| Every layer independently builds Local KV | `Attention._window_kv()` |
| KV generators 2/8/14/20; 2:1 Encoder pooling and 1:1 Decoder projection | config, `Compressor.forward()`, `Attention._compress_kv()` |
| Index source layers 2/8/14/20/24/28/32/36; other compressed layers reuse Top-k IDs | `_compress_topk_idxs()`, `Indexer.forward()` |
| Reindex changes positions, not KV vectors; candidate masks originate at 20 | `Indexer.forward()`, `select_candidate_blocks()` |
| Local and selected Global KV share one main attention, sink and grouped output | `Attention.forward()`, `kernel.py:sparse_attn()` |
| DSpark features captured before Blocks 37/38/39, three separate draft calls | `Transformer.forward()`, `forward_spec()`, `DSparkBlock`, `DSparkAttention` |

KV source caches remain allocated in their owners. A new generator replaces the active shared reference; downstream layers follow that reference until the next generator. Encoder owners therefore serve 2–7, 8–13 and 14–19; KV20 serves 20–39. Reindex-only layers 24/28/32/36 republish selected IDs while retaining KV20. Local KV is never shared between backbone layers.

The local source differs from the pinned reference in two relevant places. Local `Indexer.forward()` publishes `shared_attn.index_k` only when a new latent exists; pinned source rebinds the owning cache even when a ratio-2 group is incomplete. The local incomplete-group branch may read a reference from the prior call and is not runtime-verified. Pinned main/draft Attention also quantize grouped output before `einsum`; local code omits that extra `act_quant`. The diagrams follow local output code and disclose the Index Key discrepancy in their calculation/source notes. No model source was changed.

DSpark uses its own projections and private window caches; it does not directly borrow backbone KV tensors. `forward_spec(start_pos=0)` seeds caches and returns None. Decode runs three draft stages and returns the seed plus five candidates, logits and confidence. The external verification/acceptance scheduler is not implemented in this model source. Existing backbone-ledger links are explicitly marked as comparisons when opened from Draft; no draft-specific FLOPs are invented.

## Validation and preservation

* Build: `.venv/Scripts/python.exe src/deepseek/build.py`.
* Browser regression: `npm run test:browser:deepseek` includes all five original suites plus `verify_attention_blocks.cjs`.
* Parameter/source audit: `verify_audit.py`.
* Equations and original geometry guard: `verify_single_token.py`, `attention_preservation.py`.
* Small numerical-gradient checks: `verify_calculations.py`.

The numeric/parameter ledger files remain byte-for-byte unchanged. The only changed entry in `step4_preservation.json` is the intentionally revised architecture SVG hash. The new geometry guard runs at build time and in numeric regression, checking 157 protected original geometry elements outside the authorized Attention blocks and KV boundary arrows. The root viewBox remains exactly `0 0 1672 941`.

All suites passed: 189 module/mode combinations, 83 numeric equations, 160 matrix products, original Indexer Full/Reindex/Reuse behavior across 38 layers, original information flow and zoom/pan, and the new original-layout/detail checks. New checks cover 1920×1080 and 1366×768, inline Indexer behavior, twelve Attention calculation-mode checks, panel close/return and the three Draft stages, with zero JavaScript errors and zero network requests. Actual screenshots were inspected for clipping and overlap. No weights, GPU model execution or official training/backward implementation was run.

Screenshots:

| View | 1920×1080 | 1366×768 |
| --- | --- | --- |
| Original layout | [original_layout_1920.png](../../src/deepseek/qa/attention_inplace/original_layout_1920.png) | [original_layout_1366.png](../../src/deepseek/qa/attention_inplace/original_layout_1366.png) |
| Encoder Attention | [encoder_1920.png](../../src/deepseek/qa/attention_inplace/encoder_1920.png) | [encoder_1366.png](../../src/deepseek/qa/attention_inplace/encoder_1366.png) |
| Decoder Attention | [decoder_1920.png](../../src/deepseek/qa/attention_inplace/decoder_1920.png) | [decoder_1366.png](../../src/deepseek/qa/attention_inplace/decoder_1366.png) |
| DSpark | [draft_1920.png](../../src/deepseek/qa/attention_inplace/draft_1920.png) | [draft_1366.png](../../src/deepseek/qa/attention_inplace/draft_1366.png) |

Calculation-panel screenshots and machine-readable `validation.json` are in that same directory. The `.claude/` directory present before this work is not included in the commit.

Remote `main` already has six later commits with a different layout, so this requested `eb9356c`-based revision is published on a separate branch without force-pushing or replacing remote main.
