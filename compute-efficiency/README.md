# Compute Efficiency — Roofline Analysis & MFU (DeepSeek V4.1 Flash)

A standalone, offline, two-tab **teaching slide**, not a performance-analysis
dashboard: one main idea, one chart, one worked example, one takeaway per
tab, explaining the **Roofline model** and **Model FLOPs Utilization (MFU)**
using DeepSeek V4.1 Flash as the worked example. Deeper derivations and
secondary controls are intentionally tucked behind "Show Calculation" and
"Advanced Settings" — the page is designed to be read and understood on a
single screen during a live presentation, without opening either.

## How to open

Open `index.html` directly in a browser (double-click it, or `file://` works).
No server, build step, or dependency installation is required. `app.js` and
`calculations.js` are plain `<script>` tags; there is no bundler.

## File structure

```
compute-efficiency/
├── index.html        # page shell + static explanatory content for both tabs
├── style.css          # layout, typography, chart container styling, KaTeX sizing
├── calculations.js    # pure calculation functions (Node + browser compatible, no DOM)
├── app.js              # DOM wiring: tabs, controls, SVG charts, formatting, KaTeX rendering
└── vendor/katex/       # self-hosted KaTeX 0.16.9 (JS, CSS, woff2 fonts) — see "Math typesetting" below
```

`calculations.js` has no dependency on `app.js` or the DOM, so it can also be
loaded under Node for independent testing (see "Independent verification"
below).

## Math typesetting

All formulas, worked derivations, and notation tables are rendered with
[KaTeX](https://katex.org/) 0.16.9 instead of plain text, so fractions,
subscripts, and multi-step derivations (`\begin{aligned}...\end{aligned}`,
`\boxed{}`) render as proper typeset math.

KaTeX is **vendored locally** under `vendor/katex/` (`katex.min.js`,
`katex.min.css`, and its `fonts/*.woff2`) rather than loaded from a CDN, so
the page keeps working when opened directly via `file://` with no network
access. Nothing here needs rebuilding — `vendor/katex/` is a static,
version-pinned copy; to upgrade it, re-download the same three asset types
for a newer KaTeX release.

`app.js` renders math through a small set of helpers (`renderMath`,
`katexSpan`, `renderNotationTable`) that call into `window.katex`. Every
numerical substitution shown in a derivation (the Roofline AI examples, the
MFU worked example) is computed live from `calculations.js` and re-rendered
whenever a slider or Advanced Settings input changes — nothing is a
hardcoded example value.

## Where the model parameters come from

Read directly from `models/DeepSeek-V4.1-Flash/config.json` in this
repository, and cross-checked against the module definitions in
`models/DeepSeek-V4.1-Flash/model.py`:

| Parameter | Value | Source |
|---|---|---|
| Hidden Size (`dim`) | 5120 | `config.json` |
| Q LoRA Rank (`q_lora_rank`) | 1280 | `config.json` |
| Query Heads (`n_heads`) | 64 | `config.json` |
| Head Dimension (`head_dim`) | 512 | `config.json` |
| Backbone layers (`n_layers`) | 40 | `config.json` |
| MoE intermediate dim (`moe_inter_dim`) | 2304 | `config.json` |
| Routed experts (`n_routed_experts`) | 384 | `config.json` |
| Activated routed experts (`n_activated_experts`) | 6 | `config.json` |
| Shared experts (`n_shared_experts`) | 1 | `config.json` |

The `Expert` module (`model.py`, `class Expert`) has three `Linear(dim,
inter_dim)`-shaped matrices (`w1`, `w2`, `w3` — a SwiGLU FFN), so one expert's
parameter count is `3 × dim × moe_inter_dim`. The `Gate` module (`class
Gate`) has a `weight` parameter of shape `[n_routed_experts, dim]`. Both are
used directly in the MFU tab's MoE active-vs-total comparison — see
`calculations.js: deepseekMoeActiveVsTotal()`.

## Reused from the existing implementation

`src/compute_lab/` (bundled into the repository's root `index.html`) already
has a working, source-verified Roofline and MFU implementation for a
different purpose (comparing MHA/GQA/MLA/SWA/MSA/GDA attention variants, not
DeepSeek V4.1 Flash specifically). This page is an **independent,
new** implementation under `compute-efficiency/` — no files there were
modified — but it deliberately reuses the same *formulas and conventions*,
checked against `src/compute_lab/model.js`:

- **Roofline formulas** (`AI = FLOPs/Bytes`, `P_roof = min(P_peak, AI×BW)`,
  `ridge = P_peak/BW`) — identical to `src/compute_lab/model.js: roofline()`.
- **NVIDIA H100 hardware preset** (989 TFLOP/s dense BF16, 3.35 TB/s HBM) —
  the exact figures and sources from `src/compute_lab/model.js:
  hardwarePresets.h100`, re-verified here (see "Hardware assumptions" below)
  rather than taken on faith.
- **MFU formula** (`MFU = ModelFLOPs/s ÷ (PeakTF × 1e12 × devices)`) —
  identical to `src/compute_lab/model.js: calculateMfu()`. The page's default
  MFU example is **not** reused from `compute_lab` — it uses DeepSeek V4.1
  Flash's own reported training batch size plus illustrative hardware/step
  assumptions; see "The MFU tab's worked example" below.

Everything DeepSeek-specific (the Query Projection worked example, the MFU
worked example, the MoE active-vs-total parameter comparison, the "DeepSeek V4.1 Flash — MoE-only"
MFU mode) is new, computed directly from `config.json`/`model.py`, not
reused from `compute_lab`.

## Calculation formulas

### Roofline — Query Low-Rank Projection

```
X [M, 5120] → W_QA [5120, 1280] → H [M, 1280] → (RMSNorm, excluded) → W_QB [1280, 32768] → Q [M, 32768]

FLOPs (2MKN convention, both linear layers only):
  F(M) = 2 × M × (5120×1280 + 1280×32768) = 96,993,280 × M

Memory traffic (BF16, 2 bytes/element; weights read once per batched call):
  weightBytes       = (5120×1280 + 1280×32768) × 2            = 96,993,280 B
  inputBytes(M)      = M × 5120 × 2
  intermediateBytes(M) = M × 1280 × 2 × 2   (H is written, then read back)
  outputBytes(M)     = M × 32768 × 2
  totalBytes(M)      = weightBytes + inputBytes + intermediateBytes + outputBytes

  AI(M) = F(M) / totalBytes(M)
```

Verified (see "Independent verification"):

| M | FLOPs | Memory Traffic | AI |
|---|---|---|---|
| 1 | 96,993,280 (≈96.99 MFLOPs) | 97,074,176 B (≈97.07 MB) | ≈0.999 ≈ 1 |
| 1024 | 99,321,118,720 (≈99.32 GFLOPs) | 179,830,784 B (≈179.83 MB) | ≈552.3 |

Both match the task's expected values exactly. At the default H100 preset
(ridge ≈ 295.2 FLOPs/Byte), M=1 (AI≈1) is **memory-bound** and M=1024
(AI≈552) is **compute-bound** — the same weight matrices, two very different
regimes, purely because of how much activation traffic is amortized over.

### MFU

```
MFU = Model FLOPs/s ÷ Theoretical Peak FLOPs/s
Model FLOPs/s = Training FLOPs/token × Tokens/s
Theoretical Peak FLOPs/s = Peak TFLOP/s per device × 1e12 × Device Count
```

### The MFU tab's worked example — what's real and what's illustrative

The default MFU example mixes one figure taken directly from the DeepSeek
V4.1 Flash technical report with several explicitly hypothetical teaching
assumptions. It does **not** claim to be DeepSeek V4.1 Flash's real training
MFU — the repository does not contain a verified complete (attention + MoE +
HC + Indexer + Compressor + Engram, forward + backward) training-FLOPs-per-
token figure for DeepSeek V4.1 Flash, so none is presented as if it were one.

| Parameter | Default value | Status |
|---|---|---|
| Tokens per training step | 100,600,000 (100.6M) | **Reported** — "We keep the batch size fixed at 100.6 million tokens throughout training," DeepSeek V4.1 Flash technical report (arXiv:2609.19969) |
| Peak compute per GPU | 989.5 TFLOP/s | **Derived from a published spec** — NVIDIA's H100 SXM product page lists BF16 Tensor Core at 1,979 TFLOP/s "with sparsity"; the dense (non-sparse) figure is exactly half, the standard convention, since NVIDIA does not list a dense number directly |
| Training FLOPs per token | 60 GFLOPs | Hypothetical teaching assumption |
| Number of GPUs | 2,048 | Hypothetical teaching assumption |
| Training step time | 6 s | Hypothetical teaching assumption |

Worked out: `F_step = 100.6×10⁶ × 60×10⁹ = 6.036×10¹⁸ FLOPs`;
`P_total = 2048 × 989.5 = 2,026,496 TFLOP/s`;
`MFU = 6.036×10¹⁸ / (6 × 2,026,496×10¹²) × 100% ≈ 49.6%` — matches what the
page computes live via `calculations.js: mfuCalculate()`. Both sources are
linked in the page's "References" section at the bottom, and in the MFU
tab's "Show Calculation" and "Advanced Settings" notes.

`calculations.js` still exports `deepseekMoeActiveVsTotal()` /
`deepseekMoePartialTrainingFlopsPerToken()` (unchanged, still correct) for
anyone who wants to see *why* a MoE model's "FLOPs per token" isn't simply
`6 × total parameters` — referenced in the MFU tab's "Show Calculation"
note, but deliberately not used to drive the headline percentage:

```
expertParams          = 3 × dim × moe_inter_dim                      = 35,389,440
routerParams           = dim × n_routed_experts                       = 1,966,080
totalParamsPerLayer     = (n_routed_experts + n_shared_experts) × expertParams + routerParams
activeParamsPerLayer     = (n_activated_experts + n_shared_experts) × expertParams + routerParams
× 40 layers → totalParamsAllLayers ≈ 545.08B, activeParamsAllLayers ≈ 9.99B (≈1.83% active)
```

## Hardware assumptions

Default preset: **NVIDIA H100 SXM 80GB**, dense BF16 Tensor Core throughput
**989 TFLOP/s** and HBM bandwidth **3.35 TB/s**.

- Bandwidth and memory capacity: [NVIDIA H100 product
  page](https://www.nvidia.com/en-us/data-center/h100/).
- Dense BF16 compute: [NVIDIA's own recsys-examples benchmark
  documentation](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/training/benchmark/E2E_BENCHMARK.md).
  The H100 product page also lists "1,979 TFLOP/s" for BF16, but that figure
  carries a footnote of **"with sparsity"** — it is not the dense ceiling and
  is deliberately *not* used here.

An "Illustrative accelerator (hypothetical)" preset and a free-form "Custom"
option are also available, both clearly labeled as not being real,
specific-vendor hardware specifications.

## Numerical precision assumptions

- Memory traffic is computed in **decimal** bytes/KB/MB/GB (powers of 1000),
  not binary (KiB/MiB/GiB), matching the task's instruction to use decimal
  prefixes by default.
- BF16/FP16 = 2 bytes/element, FP32 = 4 bytes/element, FP8 = 1 byte/element.
- The Roofline traffic model assumes weights are read **exactly once** per
  batched computation (ideal reuse across all M tokens processed together in
  one call) and that the intermediate activation `H` is written once and
  read back once. RMSNorm, caching effects, and kernel fusion are excluded —
  this is a simplified logical HBM model, not a hardware trace.

## Known limitations

- The Roofline worked example covers **only the two Query-projection linear
  layers**. It does not model RMSNorm, the rest of Attention, MoE, HC
  mixing, the Indexer, the Compressor, or Engram.
- The MFU example's batch size (100.6M tokens/step) is a reported figure, and
  its peak compute (989.5 TFLOP/s) is derived from a published spec, but
  FLOPs/token, GPU count, and step time remain explicitly hypothetical — see
  "The MFU tab's worked example" above. No complete, verified DeepSeek V4.1
  Flash training-FLOPs-per-token figure is claimed anywhere on this page.
- The Roofline chart's third, slider-driven point ("Tokens Processed
  Together") uses the same Query-projection formula as the Decode/Batch
  reference points; it does not represent a different operation, and is
  hidden automatically whenever it would land on/near Decode or Batch.
- Multi-device scaling in the MFU calculator is a simple linear
  `peakTF × devices` aggregate — no communication overhead, topology, or
  parallelism strategy is modeled.
- No part of this page loads model weights, runs a GPU kernel, or reports a
  measured (as opposed to theoretical/estimated) performance number.

## Independent verification

`calculations.js` is dependency-free and runs under Node:

```bash
node -e "
const C = require('./calculations.js');
console.log(C.queryProjectionScenario(1));
console.log(C.queryProjectionScenario(1024));
console.log(C.mfuCalculate({trainingFlopsPerToken:60e9, deviceCount:2048, peakTF:989.5, rateMode:'step', tokensPerStep:100.6e6, stepSeconds:6}));
"
```

Checked by hand against the task's expected values:
Decode ≈96.99 MFLOPs / ≈97.07 MB / AI≈1 (memory-bound); Batch ≈99.32 GFLOPs /
≈179.83 MB / AI≈552 (compute-bound); ridge ≈295.22 FLOPs/Byte; the default
MFU example ≈ 49.6%.

Visual testing was done by loading the page in a real browser (desktop
1920×1080, 1366×768, and a 390px mobile width) and exercising both tabs: the
Tokens-Processed-Together slider across its full range, every hardware
preset and the precision switch, the Training-Step-Time slider, invalid-input
handling (zero/negative peak compute, bandwidth, device count — each
produces a clear on-page message or a safe clamped value instead of a crash
or silent `NaN`), both "Show Calculation" sections, both "Advanced Settings"
sections, and Reset.

The KaTeX-rendered math was separately checked at all three widths above:
every formula card, notation table, and expandable derivation was confirmed
free of clipping/overflow (`scrollWidth` vs. `clientWidth` on every math
container, plus whole-page overflow), hardware-input changes were confirmed
to live-update the Roofline derivations (not just the chart), and the
browser console was checked clean of errors.
