/* Pure calculation functions for Roofline and MFU analysis. No DOM access.
 * Runs in the browser (window.ComputeEfficiency) and in Node (module.exports) for tests.
 *
 * DeepSeek V4.1 Flash parameters below are read directly from
 * models/DeepSeek-V4.1-Flash/config.json and verified against model.py
 * (Attention / Expert / Gate classes). See README.md for the exact citations.
 */
(function (root) {
  'use strict';

  // ---- DeepSeek V4.1 Flash model parameters (source: config.json) ----------
  const DEEPSEEK = Object.freeze({
    dim: 5120,            // config.json: dim
    qLoraRank: 1280,      // config.json: q_lora_rank
    nHeads: 64,           // config.json: n_heads
    headDim: 512,         // config.json: head_dim
    nLayers: 40,          // config.json: n_layers (backbone only, excludes the 3 Draft/MTP layers)
    moeInterDim: 2304,    // config.json: moe_inter_dim
    nRoutedExperts: 384,  // config.json: n_routed_experts
    nActivatedExperts: 6, // config.json: n_activated_experts (top-k routed experts per token)
    nSharedExperts: 1,    // config.json: n_shared_experts (always active)
  });

  const bytesPerElement = Object.freeze({ BF16: 2, FP16: 2, FP32: 4, FP8: 1 });

  // ---- Part A: Roofline — Query Low-Rank Projection worked example ---------
  //
  //   X [M, dim] -> W_QA [dim, qLoraRank] -> H [M, qLoraRank]
  //                 -> (RMSNorm, excluded from this baseline) ->
  //                 -> W_QB [qLoraRank, nHeads*headDim] -> Q [M, nHeads*headDim]
  //
  // FLOPs use the common 2MKN matmul convention (one multiply + one add per
  // output element per reduction step), summed over the two linear layers.
  function queryProjectionFlops(M, p = DEEPSEEK) {
    const totalQDim = p.nHeads * p.headDim;
    return 2 * M * (p.dim * p.qLoraRank + p.qLoraRank * totalQDim);
  }

  // HBM traffic model (decimal bytes, not binary). Weights are read once per
  // batched computation (a single kernel launch processing all M tokens
  // together reads each weight matrix exactly once); activations scale with M.
  // Intermediate H is written after W_QA and read back before W_QB, so it is
  // counted twice. RMSNorm and any kernel-fusion/caching effects are excluded
  // — this is a simplified logical HBM model, not a measured hardware trace.
  function queryProjectionTraffic(M, precision = 'BF16', p = DEEPSEEK) {
    const b = bytesPerElement[precision];
    if (!b) throw new Error(`Unsupported precision: ${precision}`);
    const totalQDim = p.nHeads * p.headDim;
    const weightElements = p.dim * p.qLoraRank + p.qLoraRank * totalQDim;
    const weightBytes = weightElements * b;
    const inputBytes = M * p.dim * b;
    const intermediateBytes = M * p.qLoraRank * b * 2; // write + read
    const outputBytes = M * totalQDim * b;
    const totalBytes = weightBytes + inputBytes + intermediateBytes + outputBytes;
    return { weightBytes, inputBytes, intermediateBytes, outputBytes, totalBytes, weightElements, bytesPerElement: b };
  }

  function queryProjectionScenario(M, precision = 'BF16', p = DEEPSEEK) {
    const flops = queryProjectionFlops(M, p);
    const traffic = queryProjectionTraffic(M, precision, p);
    return { M, precision, flops, traffic, arithmeticIntensity: flops / traffic.totalBytes };
  }

  // ---- Roofline model --------------------------------------------------
  // AI = FLOPs / Bytes; P_roof = min(P_peak, AI x BW_peak); ridge = P_peak / BW_peak.
  function roofline(flops, bytesMoved, peakFlopsPerSecond, bandwidthBytesPerSecond) {
    if (!(peakFlopsPerSecond > 0) || !(bandwidthBytesPerSecond > 0)) {
      throw new Error('Peak compute and bandwidth must be positive numbers.');
    }
    const ridge = peakFlopsPerSecond / bandwidthBytesPerSecond;
    if (!(flops > 0) || !(bytesMoved > 0)) {
      return { ai: null, ridge, performance: null, bottleneck: 'N/A', computeSeconds: null, memorySeconds: null, lowerSeconds: null };
    }
    const ai = flops / bytesMoved;
    const computeSeconds = flops / peakFlopsPerSecond;
    const memorySeconds = bytesMoved / bandwidthBytesPerSecond;
    const performance = Math.min(peakFlopsPerSecond, ai * bandwidthBytesPerSecond);
    const lowerSeconds = Math.max(computeSeconds, memorySeconds);
    return {
      ai, ridge, performance, computeSeconds, memorySeconds, lowerSeconds,
      bottleneck: ai < ridge ? 'Memory-Bound' : 'Compute-Bound',
    };
  }

  // Hardware presets. H100 figures are real, sourced NVIDIA specifications
  // (dense BF16 Tensor Core throughput, not the "with sparsity" number on the
  // product page) — reused here as an illustrative reference point, not a
  // claim about any other accelerator.
  const HARDWARE_PRESETS = Object.freeze({
    h100: Object.freeze({
      id: 'h100', name: 'NVIDIA H100 SXM 80GB (reference)', peakTF: 989, bandwidth: 3.35, bandwidthUnit: 'TB/s',
      precision: 'BF16', computeType: 'Dense Tensor Core (not "with sparsity")',
      bandwidthSource: 'https://www.nvidia.com/en-us/data-center/h100/',
      computeSource: 'https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/training/benchmark/E2E_BENCHMARK.md',
    }),
    illustrative: Object.freeze({
      id: 'illustrative', name: 'Illustrative accelerator (hypothetical — edit me)', peakTF: 500, bandwidth: 2, bandwidthUnit: 'TB/s',
      precision: 'BF16', computeType: 'Hypothetical, for teaching only',
      bandwidthSource: null, computeSource: null,
    }),
  });

  function bandwidthBytesPerSecond(bandwidth, unit) {
    const mult = unit === 'TB/s' ? 1e12 : unit === 'GB/s' ? 1e9 : null;
    if (!mult) throw new Error('Bandwidth unit must be GB/s or TB/s.');
    return bandwidth * mult;
  }

  // ---- Part B: MFU ------------------------------------------------------
  function mfuCalculate(input) {
    const { trainingFlopsPerToken, deviceCount, peakTF, rateMode } = input;
    if (!(trainingFlopsPerToken > 0)) throw new Error('Training FLOPs/token must be a positive number.');
    if (!Number.isInteger(deviceCount) || deviceCount < 1) throw new Error('Device count must be a positive integer.');
    if (!(peakTF > 0)) throw new Error('Peak compute per accelerator must be a positive number.');
    let tokensPerSecond;
    if (rateMode === 'step') {
      const { tokensPerStep, stepSeconds } = input;
      if (!(tokensPerStep > 0) || !(stepSeconds > 0)) throw new Error('Tokens/step and step time must be positive numbers.');
      tokensPerSecond = tokensPerStep / stepSeconds;
    } else {
      tokensPerSecond = input.tokensPerSecond;
      if (!(tokensPerSecond > 0)) throw new Error('Tokens/second must be a positive number.');
    }
    const modelFlopsPerSecond = trainingFlopsPerToken * tokensPerSecond;
    const hardwarePeakFlopsPerSecond = peakTF * 1e12 * deviceCount;
    const mfu = modelFlopsPerSecond / hardwarePeakFlopsPerSecond;
    return { tokensPerSecond, modelFlopsPerSecond, hardwarePeakFlopsPerSecond, mfu, percent: 100 * mfu, exceedsPeak: mfu > 1 };
  }

  // DeepSeek V4.1 Flash MoE active-vs-total parameters, derived directly from
  // config.json and the Expert/Gate module shapes in model.py:
  //   Expert (SwiGLU): w1[dim,inter], w2[inter,dim], w3[dim,inter] -> 3 x dim x inter_dim params.
  //   Gate (router): weight[n_routed_experts, dim] params.
  // Every token activates nActivatedExperts routed experts + nSharedExperts
  // shared experts (always active) + the router itself; it never touches the
  // other (nRoutedExperts - nActivatedExperts) routed experts' weights.
  function deepseekMoeActiveVsTotal(p = DEEPSEEK) {
    const expertParams = 3 * p.dim * p.moeInterDim;
    const routerParams = p.dim * p.nRoutedExperts;
    const totalExpertsPerLayer = p.nRoutedExperts + p.nSharedExperts;
    const activeExpertsPerLayer = p.nActivatedExperts + p.nSharedExperts;
    const totalParamsPerLayer = totalExpertsPerLayer * expertParams + routerParams;
    const activeParamsPerLayer = activeExpertsPerLayer * expertParams + routerParams;
    const totalParamsAllLayers = totalParamsPerLayer * p.nLayers;
    const activeParamsAllLayers = activeParamsPerLayer * p.nLayers;
    return {
      expertParams, routerParams, totalExpertsPerLayer, activeExpertsPerLayer,
      totalParamsPerLayer, activeParamsPerLayer, totalParamsAllLayers, activeParamsAllLayers,
      activeShare: activeParamsAllLayers / totalParamsAllLayers,
    };
  }

  // A clearly-PARTIAL approximate training-FLOPs-per-token figure using the
  // standard "6 x active parameters" convention, applied ONLY to the MoE
  // expert+router weights (the part where total vs. active parameters differ).
  // This intentionally EXCLUDES attention, HC mixing, Indexer, Compressor and
  // Engram FLOPs, so it understates the true per-token training cost — it
  // exists to demonstrate the MoE active/total correction, not to stand in
  // for a complete model estimate. See README.md "Known limitations".
  function deepseekMoePartialTrainingFlopsPerToken(p = DEEPSEEK) {
    const moe = deepseekMoeActiveVsTotal(p);
    return 6 * moe.activeParamsAllLayers;
  }

  const api = {
    DEEPSEEK, bytesPerElement,
    queryProjectionFlops, queryProjectionTraffic, queryProjectionScenario,
    roofline, HARDWARE_PRESETS, bandwidthBytesPerSecond,
    mfuCalculate, deepseekMoeActiveVsTotal, deepseekMoePartialTrainingFlopsPerToken,
  };
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.ComputeEfficiency = api;
})(typeof globalThis === 'object' ? globalThis : this);
