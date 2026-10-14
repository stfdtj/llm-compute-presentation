# FLOPs calculation audit

## Current revision: kernel analysis and Training Forward

The historical audit below is retained for comparison. Its missing-kernel statements are superseded by this revision. The English report now keeps the original Decode result in sections 1–17 and gives the current kernel derivations and Training Forward analysis in sections 18–22. The old optional backward note was removed; Backward, Optimizer, Loss, and replay costs remain outside scope.

- **HC is now source-verified:** [kernel.py:407](DeepSeek-V4.1-Flash/kernel.py) computes pre/post sigmoids, a 4×4 row softmax, one column normalization, then 19 row/column pairs. Literal source expressions contribute 1,192 Basic + 664 Special = 1,856 modeled operations per token/instance, or 148,480 across 80 Decode instances. `U_HC` is resolved at the mathematical-expression level. Hoisting repeated denominator epsilon additions would reduce Basic by 468 per instance; the report does not claim exact compiled instruction counts.
- **Attention uses online softmax:** [kernel.py:311](DeepSeek-V4.1-Flash/kernel.py) processes fixed tiles of 64 gathered positions. It rescales the output accumulator after each tile, adds the sink exponential to the denominator after all tiles, and divides 512 output coordinates per head. Masked entries still enter the full tile GEMMs. Revised Decode attention is 3,238,581,248 Basic + 2,910,720 Special = 3,241,491,968, an increase of 12,397,056 over the original reference estimate.
- **Unified accounting:** Add/subtract/multiply and GEMM 2MKN are Basic. Division and nonlinear calls are simplified Special estimates; Total Modeled Operations is their sum. Evidence status is separate from cost category. Every current module table includes these three columns and excluded-cost tags. Quantization can contain floating-point arithmetic; it is separately excluded, rather than called non-FLOP work. Engram's explicit table dequantization remains counted once.
- **Training evidence limit:** [Transformer.forward](DeepSeek-V4.1-Flash/model.py) is inference-only and normally emits last-position logits. No separate `inference/kernel.py` or trainer was found. The report defines **Scenario P**, using the full-sequence source graph, supplied sparse kernel, fresh sequence activations and all-token logits. This is an explicit training-forward assumption, not a claim about official DeepSeek training.
- **Sharing / CED:** The acronym CED is not defined in local code. The verified source-layer dependency table accounts for 4 compressors, 4 Index Key producers and 8 indexers across 40 main layers. Scenario P preserves that sharing within each forward; it never reuses an earlier training step's KV cache.
- **Sequence-specific corrections:** Index scoring is rectangular before causal/candidate masking: `B*T*(3*floor(T/2)+5*T)` pairs. Ratio-2 pooling runs on every completed group. Attention uses padded index-list tiles, not just useful causal pairs. Engram's `[4,5120]` parameter product runs once per layer call; multiplying all Decode Engram work by `B*T` overcounts it.

| Current scenario | Basic FLOPs | Special estimate | Total modeled operations |
|---|---:|---:|---:|
| Decode, batch 1, S=1024 historical | 35,580,865,451 | 3,655,814 | **35,584,521,265** |
| Training Scenario P, B=1, T=1024 | 36,434,932,621,312 | 3,746,705,408 | **36,438,679,326,720** |

Scenario P averages **35,584,647,780 operations/token** after division by 1024. Its modeled batch total is 36.438679326720 TFLOPs-equivalent. Actual official Training Forward and complete device-kernel totals remain **Unknown**. Dense-kernel padding, quantization/scale work, routing-dependent expert padding, memory, comparisons, and communication are not silently assigned zero.

Verification uses the updated Python script to enumerate the HC and attention operations independently of their closed-form formulas, check all three ledgers (historical Decode, revised Decode, Training Scenario P), test odd/short sequence lengths, and verify the Engram batch-broadcast correction. The cross-scenario difference also reconciles exactly: Training Scenario P minus 1024 revised Decode forwards is 129,551,360 modeled operations. No GPU execution or official trainer validation is claimed.

## Historical audit (original source snapshot)

The calculation uses the local implementation, batch 1, one warm-decode token, and **S=1024 historical tokens excluding the current token**. The current position is 1024 and the visible length is 1025. All layer IDs are zero-based. `model.py`, `engram.py`, and `config.json` are present; `kernel.py` is absent. No model source was changed.

## Result and limits

**Source snapshot note:** `DeepSeek-V4.1-Flash/kernel.py` was added after this calculation. The results and kernel-availability statements in this audit describe the earlier source snapshot; the new kernel has not yet been incorporated into the FLOPs audit.

| Quantity | FLOPs/token |
|---|---:|
| Source-verifiable multiply/add subtotal, with GEMMs approximated by 2MKN | 32,342,188,843 |
| Simplified division/nonlinear operations outside main attention | 691,974 |
| Reference sparse attention, including softmax/sink | 3,229,094,912 |
| Quantified modeled backbone subtotal | **35,571,975,729 (~35.572 GFLOPs)** |

The last row still excludes **80 unknown `hc_split_sinkhorn` calls and unquantified quantization/kernel overhead**. It is not a complete exact whole-model total. Main-attention arithmetic is an explicit mathematical estimate because the kernel is missing. Means use one division; rsqrt/sqrt/exp/softplus/sigmoid/division each receive one simplified operation, not a hardware instruction-cost claim. Tensor-parallel communication is excluded by using `world_size=1` logical work.

## Review of original sections 2–9

| Section | Finding and correction |
|---|---|
| 2: Q low-rank projection | Shapes and 40 executions correct: 3,879,731,200 FLOPs. Made intermediate norm placement explicit. |
| 3: Q RMSNorm | The original 4d+2 total is correct under unit-cost special operations. Code uses a mean, rsqrt, and two elementwise multiplications. Split into 204,800 multiply/add plus 80 special operations. |
| 4: shared KV | The 209,715,200 projection count is correct. Added 40 KV norms. This window KV is shared across heads/K/V, not across layers; compressed KV has separate owners. |
| 5: RoPE | Per-call counts correct. Applied 40 executions and included inverse RoPE. Added compressed-KV RoPE under Compressor; Index RoPE remains under Indexer. |
| 6: Indexer | Eight indexers, four potential key owners, but only **one new key** at this step. Scores scan 3*512+5*1025=6661 positions. Revised combined total to 141,675,261. Candidate masks do not reduce the preceding einsum. |
| 7: Compressor | Projection subtotal 36,700,160 correct. Three ratio-2 calls project but return `None` on this step; only the ratio-1 call normalizes/emits. Added conditional pooling, norms, and compressed RoPE. Combined step total 36,702,402. |
| 8: grouped output | Shapes, grouping, arithmetic, and 40 executions correct: 6,039,797,760. |
| 9: HC | Original coefficient projection correct. Added flattened-stream RMS statistic, 24 coefficient multiplies, pre-mixing, and post/comb. 99,534,720 multiply/add plus 160 special operations; external coefficient transforms remain unknown. |

Correct existing dimensions/formulas remain visible in sections 2–9. Added attention/FFN norms, main attention, router, experts, Engram, final collapse/norm/head, and the small forward-wrapper operations. Norms and shared projections appear in only one subtotal each.

The original Indexer total, 142,029,320, is arithmetically correct for **1024 visible tokens including the current token** (S=1023 here), when all four owners emit. Its correction is driven by the requested historical-context convention and emission timing, not by an addition error.

## Execution details that affect the result

- Local-only layers: 0–1; ratio-2 layers: 2–19; ratio-1 layers: 20–39. Main attention therefore reads 128 entries twice and 640 entries 38 times. Local and compressed entries are distinct slots even where their source token spans overlap.
- KV source layers 2/8/14/20 run four compressors. Ratio-2 emission is gated by `(start_pos+1)%2==0`. Historical projections are reused; historical keys still participate in scoring.
- Indexers run at 2/8/14/20/24/28/32/36. Layer 20 scores 129 candidate blocks of size 8; its 2048-block limit keeps them all here. Later indexers apply candidate masks **after** computing all scores. Top-k, sorting, ReLU, and candidate maxima are comparison/indexing costs.
- Every MoE uses six routed experts and one shared expert. Routing weights multiply **2304 intermediate activations before the down projection**, not 5120 output coordinates. Source accumulation executes seven vector additions, including the first addition into zeros.
- Engram runs twice, with 24 rows of width 256 per call and a `[6144,25600]` projection. Hashing is integer work computed once for both layers. Its explicit FP8 dequantization contributes 12,288 floating-point multiplies. Table capacity is not FLOPs.
- The last HC pre-mix is reused for final collapse: no extra coefficient projection. The full 129280-token head is counted once.

## Cache publication issue in the supplied code

[`Indexer.forward`](DeepSeek-V4.1-Flash/model.py) assigns `shared_attn.index_k = self.k_cache` only inside `if self.owns_k and latent is not None`. After a normal prior forward, that shared pointer refers to layer 20's ratio-1 cache. At S=1024 the three ratio-2 owners receive `latent=None`, so each can score the first 512 entries of that layer-20 cache rather than its own cached keys. `_compress_kv` republishes compressed KV unconditionally, but does not repair the Index Key pointer.

The arithmetic shapes and counts in this audit are unchanged. At this particular length, ratio-2 top-k selects all 512 positions, so the wrong scores do not change the selected index sets; longer incomplete-group steps can be affected. The natural repair is to republish each owner's existing Index Key cache even without an emission. This task only edits calculation documents, so the model remains unchanged.

## Source references

| Area | Implementation |
|---|---|
| Dimensions, layer ownership, ratios | [config.json](DeepSeek-V4.1-Flash/config.json) |
| Runtime defaults, including router options | [ModelArgs](DeepSeek-V4.1-Flash/model.py) |
| Quantized linears / RMSNorm | [linear](DeepSeek-V4.1-Flash/model.py), [RMSNorm.forward](DeepSeek-V4.1-Flash/model.py) |
| RoPE / Compressor | [apply_rotary_emb](DeepSeek-V4.1-Flash/model.py), [Compressor.forward](DeepSeek-V4.1-Flash/model.py) |
| Index scores / candidates / cache path | [Indexer.forward](DeepSeek-V4.1-Flash/model.py), [select_candidate_blocks](DeepSeek-V4.1-Flash/model.py), [Attention](DeepSeek-V4.1-Flash/model.py) |
| Router / experts / accumulation | [Gate.forward](DeepSeek-V4.1-Flash/model.py), [Expert.forward](DeepSeek-V4.1-Flash/model.py), [MoE.forward](DeepSeek-V4.1-Flash/model.py) |
| HC and sublayer order | [Block.hc_mixes](DeepSeek-V4.1-Flash/model.py), [Block.forward](DeepSeek-V4.1-Flash/model.py) |
| Engram float operations / integer hash | [ParallelEngramEmbedding.forward](DeepSeek-V4.1-Flash/model.py), [Engram.forward](DeepSeek-V4.1-Flash/model.py), [NgramHashState.forward](DeepSeek-V4.1-Flash/engram.py) |
| Final output and wrapper | [ParallelHead.forward](DeepSeek-V4.1-Flash/model.py), [Transformer.forward](DeepSeek-V4.1-Flash/model.py), [sample](DeepSeek-V4.1-Flash/model.py) |

## Unknowns, sequence dependence, and verification

- Missing `kernel.py`: sparse-attention sink behavior, fused/online softmax, tiling/padding, extra rescaling, `hc_split_sinkhorn`'s actual 20-iteration operations, activation quantization, and quantized GEMM overhead cannot be verified. The imported `vision.py` and `image_processor.py` are also absent but their forward paths are outside this text-only scope. No runtime model execution is claimed.
- Projections, experts and Engram have fixed per-token work. Index scores grow with `3*floor((S+1)/2)+5*(S+1)`. Main attention caps compressed selections at 512 per layer. New ratio-2 norms/keys/rotations/pooling alternate with group completion. A 1024-visible-token convention instead corresponds to S=1023 and yields a different modeled subtotal: 35,572,351,874.
- Prefill projects all input tokens and emits completed groups. The index einsum computes the full rectangular score tensor before causal masking; warm-decode formulas must not be reused as a prefill/training total.
- The unmodified forward also makes three DSpark support means and samples a token. These add 837,119 modeled operations at default temperature 1, plus unknown random-number-generation work; both are reported separately from the requested backbone. `forward_spec` and vision are excluded.
- Run [FLOPs_calculation_verify.py](FLOPs_calculation_verify.py) with `python FLOPs_calculation_verify.py`. It reads the configuration, independently computes the ledger, checks numerical equations and summary rows, and checks execution counts. `--history 1023` evaluates another decode step. It does not import Torch or allocate model weights.
- The earlier optional backward note is removed from the current report; the new analysis is limited to Forward.
