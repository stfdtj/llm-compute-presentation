/* DOM wiring, charts and interaction. Uses window.ComputeEfficiency from calculations.js. */
(function () {
  'use strict';
  const C = window.ComputeEfficiency;
  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- format
  const fmtFlops = (n) => n == null ? 'N/A' :
    n >= 1e12 ? (n / 1e12).toFixed(3) + ' TFLOPs' :
    n >= 1e9 ? (n / 1e9).toFixed(2) + ' GFLOPs' :
    n >= 1e6 ? (n / 1e6).toFixed(2) + ' MFLOPs' :
    n.toLocaleString('en-US') + ' FLOPs';
  const fmtBytes = (n) => n == null ? 'N/A' :
    n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' :
    n >= 1e6 ? (n / 1e6).toFixed(2) + ' MB' :
    n >= 1e3 ? (n / 1e3).toFixed(2) + ' KB' :
    n.toFixed(0) + ' B';
  const fmtNum = (n, d = 2) => n == null ? 'N/A' : n.toLocaleString('en-US', { maximumFractionDigits: d });
  const fmtTime = (s) => s == null ? 'N/A' : s >= 1 ? s.toFixed(4) + ' s' : s >= 1e-3 ? (s * 1e3).toFixed(3) + ' ms' : (s * 1e6).toFixed(3) + ' μs';
  const fmtParams = (n) => n == null ? 'N/A' : n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : (n / 1e6).toFixed(1) + 'M';

  // ---------------------------------------------------------------- tabs
  document.querySelectorAll('nav.tabs button').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('nav.tabs button').forEach((b) => { b.setAttribute('aria-selected', 'false'); b.tabIndex = -1; });
      btn.setAttribute('aria-selected', 'true'); btn.tabIndex = 0;
      document.querySelectorAll('main > section[role=tabpanel]').forEach((p) => { p.hidden = true; });
      $('tab-' + btn.dataset.tab).hidden = false;
    });
  });

  // ======================================================= ROOFLINE TAB
  function rooflineInputs() {
    return {
      M: Math.max(1, Math.round(Number($('in-M').value) || 1)),
      precision: $('in-precision').value,
      peakTF: Number($('in-peak-tf').value),
      bandwidth: Number($('in-bandwidth').value),
      bandwidthUnit: $('in-bw-unit').value,
    };
  }

  function renderTrafficFormula() {
    const p = C.DEEPSEEK;
    const totalQDim = p.nHeads * p.headDim;
    $('traffic-formula').textContent =
      `Weight Traffic  = (5120×1280 + 1280×${totalQDim}) × bytes             — read once\n` +
      `Input Traffic   = M × 5120 × bytes\n` +
      `Intermediate    = M × 1280 × bytes × 2    (write H, then read it back)\n` +
      `Output Traffic  = M × ${totalQDim} × bytes\n` +
      `Total Bytes     = Weight + Input + Intermediate + Output`;
  }

  function scenarioCard(M, precision) {
    const s = C.queryProjectionScenario(M, precision);
    const ridge = currentRidge();
    const bottleneck = s.arithmeticIntensity < ridge ? 'Memory-Bound' : 'Compute-Bound';
    const cls = bottleneck === 'Memory-Bound' ? 'mem' : 'comp';
    return `<div class="scenario-card">
      <h3>${M === 1 ? 'Scenario A — Single-token Decode' : 'Scenario B — Batched Computation'} (M = ${M.toLocaleString()})</h3>
      <div class="big">${fmtFlops(s.flops)}</div>
      <dl>
        <dt>Memory traffic</dt><dd>${fmtBytes(s.traffic.totalBytes)}</dd>
        <dt>&nbsp;&nbsp;· weights</dt><dd>${fmtBytes(s.traffic.weightBytes)}</dd>
        <dt>&nbsp;&nbsp;· input</dt><dd>${fmtBytes(s.traffic.inputBytes)}</dd>
        <dt>&nbsp;&nbsp;· intermediate</dt><dd>${fmtBytes(s.traffic.intermediateBytes)}</dd>
        <dt>&nbsp;&nbsp;· output</dt><dd>${fmtBytes(s.traffic.outputBytes)}</dd>
        <dt>Arithmetic Intensity</dt><dd>${fmtNum(s.arithmeticIntensity)} FLOPs/Byte</dd>
      </dl>
      <span class="bottleneck ${cls}">${bottleneck}</span>
    </div>`;
  }

  function renderScenarios() {
    const precision = $('in-precision').value;
    $('scenario-cards').innerHTML = scenarioCard(1, precision) + scenarioCard(1024, precision);
  }

  function currentHardware() {
    const i = rooflineInputs();
    const peak = i.peakTF * 1e12;
    const bw = C.bandwidthBytesPerSecond(i.bandwidth, i.bandwidthUnit);
    return { peak, bw };
  }
  function currentRidge() { const { peak, bw } = currentHardware(); return peak / bw; }

  function applyHardwarePreset(id) {
    if (id === 'custom') { $('hw-source').textContent = 'Custom hardware — values entered manually; verify precision and units yourself.'; return; }
    const h = C.HARDWARE_PRESETS[id];
    $('in-peak-tf').value = h.peakTF;
    $('in-bandwidth').value = h.bandwidth;
    $('in-bw-unit').value = h.bandwidthUnit;
    $('in-precision').value = h.precision;
    $('hw-source').innerHTML = h.bandwidthSource
      ? `${h.name} · ${h.computeType}. Bandwidth: <a href="${h.bandwidthSource}" target="_blank" rel="noopener">source</a> · Compute (dense, not "with sparsity"): <a href="${h.computeSource}" target="_blank" rel="noopener">source</a>.`
      : `${h.name} — hypothetical numbers for teaching only, not a real accelerator's specification. Edit freely.`;
  }

  function drawRooflineChart() {
    const i = rooflineInputs();
    const { peak, bw } = currentHardware();
    const ridge = peak / bw;
    // The "Tokens processed together (M)" control drives its own point so the
    // chart genuinely responds to it; Scenario A (M=1) and B (M=1024) stay
    // fixed as the canonical worked-example comparison from the task.
    const Ms = (i.M === 1 || i.M === 1024) ? [1, 1024] : [1, 1024, i.M];
    const points = Ms.map((M) => {
      const s = C.queryProjectionScenario(M, i.precision);
      const r = C.roofline(s.flops, s.traffic.totalBytes, peak, bw);
      return { M, s, r };
    });
    const allAi = [ridge, ...points.map((p) => p.r.ai)];
    const xmin = Math.pow(10, Math.floor(Math.log10(Math.min(...allAi) / 3)));
    const xmax = Math.pow(10, Math.ceil(Math.log10(Math.max(...allAi) * 3)));
    const values = [peak, xmin * bw, ...points.map((p) => p.r.performance)];
    const ymin = Math.pow(10, Math.floor(Math.log10(Math.min(...values) * 0.5)));
    const ymax = Math.pow(10, Math.ceil(Math.log10(Math.max(...values) * 1.5)));
    const left = 86, right = 880, top = 24, bottom = 360;
    const x = (v) => left + (Math.log10(v) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin)) * (right - left);
    const y = (v) => bottom - (Math.log10(v) - Math.log10(ymin)) / (Math.log10(ymax) - Math.log10(ymin)) * (bottom - top);

    let svg = `<rect x="0" y="0" width="900" height="420" fill="none"/>`;
    svg += `<text x="${left}" y="14" font-size="11" fill="#5c6b85">Performance (TFLOP/s) — log scale</text>`;
    for (let k = Math.round(Math.log10(xmin)); k <= Math.round(Math.log10(xmax)); k++) {
      const v = Math.pow(10, k);
      svg += `<path d="M${x(v)},${top}V${bottom}" stroke="#eef1f7"/><text x="${x(v)}" y="${bottom + 18}" text-anchor="middle" font-size="10" fill="#5c6b85">10^${k}</text>`;
    }
    for (let k = Math.round(Math.log10(ymin)); k <= Math.round(Math.log10(ymax)); k++) {
      const v = Math.pow(10, k);
      svg += `<path d="M${left},${y(v)}H${right}" stroke="#eef1f7"/><text x="${left - 10}" y="${y(v) + 4}" text-anchor="end" font-size="10" fill="#5c6b85">${fmtNum(v / 1e12, 4)}</text>`;
    }
    // Memory-bound shaded region (left of ridge) vs compute-bound (right).
    svg += `<rect x="${left}" y="${top}" width="${x(ridge) - left}" height="${bottom - top}" fill="#c25b8f" opacity="0.045"/>`;
    svg += `<rect x="${x(ridge)}" y="${top}" width="${right - x(ridge)}" height="${bottom - top}" fill="#3a6fd8" opacity="0.045"/>`;
    // Roofline ceiling: sloped memory limit then horizontal compute limit.
    svg += `<path d="M${x(xmin)},${y(xmin * bw)} L${x(ridge)},${y(peak)} L${x(xmax)},${y(peak)}" fill="none" stroke="#16233f" stroke-width="3"/>`;
    svg += `<path d="M${x(ridge)},${top}V${bottom}" stroke="#9daec3" stroke-dasharray="5 4"/>`;
    svg += `<text x="${Math.min(x(ridge) + 6, right - 150)}" y="${top + 14}" font-size="11" fill="#5c6b85">Ridge AI = ${fmtNum(ridge)}</text>`;
    svg += `<text x="${left + 14}" y="${top + 16}" font-size="11" fill="#c25b8f">Memory-bound</text>`;
    svg += `<text x="${right - 12}" y="${top + 16}" text-anchor="end" font-size="11" fill="#3a6fd8">Compute-bound</text>`;

    const pointLabel = (M) => M === 1 ? 'A (M=1)' : M === 1024 ? 'B (M=1024)' : `Custom (M=${M.toLocaleString()})`;
    const pointColor = (M) => M === 1 ? '#c25b8f' : M === 1024 ? '#3a6fd8' : '#1f8f5f';
    points.forEach((p) => {
      const cx = x(p.r.ai), cy = y(p.r.performance);
      const label = pointLabel(p.M), color = pointColor(p.M);
      svg += `<g data-point="${p.M}" tabindex="0" role="button" aria-label="Scenario ${label}, AI ${fmtNum(p.r.ai)}, ${p.r.bottleneck}">`;
      svg += `<circle cx="${cx}" cy="${cy}" r="7" fill="${color}" stroke="#fff" stroke-width="2"/>`;
      svg += `<text x="${cx}" y="${cy - 13}" text-anchor="middle" font-size="11" font-weight="700" fill="${color}">${label}</text>`;
      svg += `</g>`;
    });
    svg += `<text x="480" y="398" text-anchor="middle" font-size="11" fill="#5c6b85">Arithmetic Intensity (FLOPs/Byte) — log scale</text>`;
    $('roofline-chart').innerHTML = svg;

    // Tooltips via hover/focus.
    const tooltip = $('roofline-tooltip');
    $('roofline-chart').querySelectorAll('[data-point]').forEach((g) => {
      const M = Number(g.dataset.point);
      const p = points.find((q) => q.M === M);
      const show = (evt) => {
        tooltip.hidden = false;
        tooltip.innerHTML = `<b>${pointLabel(M)}</b><br>AI: ${fmtNum(p.r.ai)} FLOPs/Byte<br>FLOPs: ${fmtFlops(p.s.flops)}<br>Memory traffic: ${fmtBytes(p.s.traffic.totalBytes)}<br>Roofline ceiling: ${fmtNum(p.r.performance / 1e12, 3)} TFLOP/s<br>Limiting resource: ${p.r.bottleneck}`;
        const rect = $('roofline-chart').getBoundingClientRect();
        const wrapRect = $('roofline-chart-wrap').getBoundingClientRect();
        const px = rect.left - wrapRect.left + (evt?.offsetX ?? 450) * rect.width / 900;
        const py = rect.top - wrapRect.top + (evt?.offsetY ?? 200) * rect.height / 420;
        tooltip.style.left = Math.min(px + 14, wrapRect.width - 250) + 'px';
        tooltip.style.top = Math.max(py - 10, 0) + 'px';
      };
      g.addEventListener('mouseenter', show);
      g.addEventListener('focus', show);
      g.addEventListener('mouseleave', () => { tooltip.hidden = true; });
      g.addEventListener('blur', () => { tooltip.hidden = true; });
    });

    const metrics = [
      ['Ridge Point', `${fmtNum(ridge)} FLOPs/Byte`],
      ['Scenario A · Ceiling', `${fmtNum(points[0].r.performance / 1e12, 3)} TFLOP/s (${points[0].r.bottleneck})`],
      ['Scenario B · Ceiling', `${fmtNum(points[1].r.performance / 1e12, 3)} TFLOP/s (${points[1].r.bottleneck})`],
    ];
    if (points.length > 2) metrics.push([`Custom (M=${points[2].M.toLocaleString()}) · Ceiling`, `${fmtNum(points[2].r.performance / 1e12, 3)} TFLOP/s (${points[2].r.bottleneck})`]);
    $('roofline-metrics').innerHTML = metrics.map(([label, val]) => `<div class="metric"><small>${label}</small><strong>${val}</strong></div>`).join('');
  }

  function updateRoofline() {
    const i = rooflineInputs();
    const errors = [];
    if (!(i.M >= 1)) errors.push('M must be a positive integer.');
    if (!(i.peakTF > 0)) errors.push('Peak compute must be a positive number.');
    if (!(i.bandwidth > 0)) errors.push('Peak memory bandwidth must be a positive number.');
    if (errors.length) {
      $('roofline-error').hidden = false;
      $('roofline-error').textContent = errors.join(' ') + ' The chart and scenarios below show the last valid configuration.';
      return;
    }
    $('roofline-error').hidden = true;
    renderScenarios();
    drawRooflineChart();
  }

  $('in-M').addEventListener('input', updateRoofline);
  $('in-precision').addEventListener('change', updateRoofline);
  $('in-peak-tf').addEventListener('input', () => { $('in-hw-preset').value = 'custom'; updateRoofline(); });
  $('in-bandwidth').addEventListener('input', () => { $('in-hw-preset').value = 'custom'; updateRoofline(); });
  $('in-bw-unit').addEventListener('change', () => { $('in-hw-preset').value = 'custom'; updateRoofline(); });
  $('in-hw-preset').addEventListener('change', (e) => { applyHardwarePreset(e.target.value); updateRoofline(); });

  // ======================================================= MFU TAB
  let mfuMode = 'generic';
  const MFU_PRESETS = {
    generic: { flopsPerToken: 1e12, devices: 4, peakTF: 1000, rateMode: 'rate', tokensPerSec: 2000, note: 'A generic, hypothetical worked example (not DeepSeek-specific) — the same numbers used throughout this field as an illustrative 50% MFU case.' },
    deepseek: { devices: 4, peakTF: 1000, rateMode: 'rate', tokensPerSec: 2000, note: '' }, // flopsPerToken filled from calculations.js
    user: { flopsPerToken: '', devices: 4, peakTF: 1000, rateMode: 'rate', tokensPerSec: 2000, note: 'Enter your own measured or estimated training FLOPs/token — this mode makes no DeepSeek-specific claim.' },
  };

  function renderMoeBars() {
    const moe = C.deepseekMoeActiveVsTotal();
    const max = moe.totalParamsAllLayers;
    $('moe-bars').innerHTML = [
      ['Total stored (384 routed + 1 shared, ×40 layers)', moe.totalParamsAllLayers, '#9daec3'],
      ['Active per token (6 routed + 1 shared + router, ×40 layers)', moe.activeParamsAllLayers, '#3a6fd8'],
    ].map(([label, v, color]) => `<div class="moe-bar-row"><span>${label}</span><div class="moe-bar-track"><div class="moe-bar-fill" style="width:${100 * v / max}%;background:${color}"></div></div><b>${fmtParams(v)}</b></div>`).join('');
    $('moe-explain').innerHTML = `Only <b>${(moe.activeShare * 100).toFixed(2)}%</b> of the MoE weights are touched by any given token (expert size = 3 × dim × moe_inter_dim = 3 × 5120 × 2304, from the <code>Expert</code> SwiGLU module in <code>model.py</code>; router = dim × n_routed_experts, from the <code>Gate</code> module). This is <b>only the MoE portion</b> — attention, HC mixing, Indexer, Compressor and Engram weights are dense and always fully active regardless of routing, so they are not part of this ratio.`;
  }

  function setMfuMode(mode) {
    mfuMode = mode;
    document.querySelectorAll('.mode-switch button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    const preset = MFU_PRESETS[mode];
    if (mode === 'deepseek') {
      const flops = C.deepseekMoePartialTrainingFlopsPerToken();
      $('in-flops-per-token').value = flops;
      $('mode-note').innerHTML = `<b>Partial estimate</b> — 6 × (active MoE+router parameters across all 40 layers) = 6 × ${fmtParams(C.deepseekMoeActiveVsTotal().activeParamsAllLayers)} = ${fmtFlops(flops)}/token. This <b>excludes</b> attention, HC mixing, Indexer, Compressor and Engram FLOPs, and excludes sequence-length effects on attention cost — it is a documented <em>lower bound</em> on the MoE-layer contribution only, not a complete per-token training FLOPs figure for the whole model.`;
      $('in-flops-per-token').readOnly = false;
    } else {
      $('in-flops-per-token').value = preset.flopsPerToken === '' ? $('in-flops-per-token').value || 1e11 : preset.flopsPerToken;
      $('mode-note').textContent = preset.note;
      $('in-flops-per-token').readOnly = false;
    }
    updateMfu();
  }

  function mfuReadInputs() {
    const rateMode = $('in-rate-mode').value;
    return {
      trainingFlopsPerToken: Number($('in-flops-per-token').value),
      deviceCount: Math.max(1, Math.round(Number($('in-devices').value) || 1)),
      peakTF: Number($('in-mfu-peak-tf').value),
      rateMode,
      tokensPerSecond: Number($('in-tokens-per-sec').value),
      tokensPerStep: Number($('in-tokens-per-step').value),
      stepSeconds: Number($('in-step-seconds').value),
    };
  }

  function updateMfu() {
    $('in-rate-mode').value === 'step'
      ? ($('label-tokens-per-sec').hidden = true, $('label-tokens-per-step').hidden = false, $('label-step-seconds').hidden = false)
      : ($('label-tokens-per-sec').hidden = false, $('label-tokens-per-step').hidden = true, $('label-step-seconds').hidden = true);
    const input = mfuReadInputs();
    let result, error = null;
    try { result = C.mfuCalculate(input); } catch (e) { error = e.message; }
    if (error) {
      $('mfu-steps').textContent = error;
      $('mfu-result').innerHTML = '';
      $('mfu-gauge').innerHTML = '';
      $('mfu-meaning').textContent = '';
      return;
    }
    const peakStr = `${input.peakTF} × 10¹² × ${input.deviceCount} devices`;
    $('mfu-steps').textContent =
      `Model FLOPs/s   = ${fmtFlops(input.trainingFlopsPerToken)}/token × ${fmtNum(result.tokensPerSecond)} tokens/s = ${fmtFlops(result.modelFlopsPerSecond)}/s\n` +
      `Hardware Peak   = ${peakStr} = ${fmtFlops(result.hardwarePeakFlopsPerSecond)}/s\n` +
      `MFU             = ${fmtFlops(result.modelFlopsPerSecond)}/s ÷ ${fmtFlops(result.hardwarePeakFlopsPerSecond)}/s = ${(result.mfu * 100).toFixed(2)}%`;
    $('mfu-result').innerHTML = `<span class="pct">${result.percent.toFixed(1)}%</span><span class="label">Model FLOPs Utilization${result.exceedsPeak ? ' — exceeds 100%, check your inputs' : ''}</span>`;
    drawMfuGauge(result);
    $('mfu-meaning').textContent = `MFU = ${result.percent.toFixed(0)}% means the model's useful training computation accounts for throughput equivalent to ${result.percent.toFixed(0)}% of the hardware's theoretical peak. The remaining ${(100 - result.percent).toFixed(0)}% is not necessarily idle time — it may reflect communication, memory bottlenecks, recomputation, scheduling overhead, or other non-matmul operations.`;
    renderSensitivity(input, result);
  }

  function drawMfuGauge(result) {
    const w = 900, h = 140, left = 40, right = 860, barY = 55, barH = 36;
    const pct = Math.min(100, result.percent);
    let svg = `<rect x="${left}" y="${barY}" width="${right - left}" height="${barH}" rx="8" fill="#eef2f9"/>`;
    svg += `<rect x="${left}" y="${barY}" width="${(right - left) * pct / 100}" height="${barH}" rx="8" fill="#3a6fd8"/>`;
    svg += `<text x="${left}" y="${barY - 10}" font-size="12" fill="#5c6b85">Useful Model Training Throughput vs. Theoretical Peak</text>`;
    svg += `<text x="${left + (right - left) * pct / 100 + 8}" y="${barY + barH / 2 + 5}" font-size="14" font-weight="700" fill="#16233f">${result.percent.toFixed(1)}%</text>`;
    svg += `<text x="${right}" y="${barY + barH + 20}" text-anchor="end" font-size="11" fill="#5c6b85">100% = ${fmtFlops(result.hardwarePeakFlopsPerSecond)}/s</text>`;
    $('mfu-gauge').innerHTML = svg;
  }

  function renderSensitivity(input, result) {
    const half = { ...input };
    if (input.rateMode === 'step') half.stepSeconds = input.stepSeconds / 2; else half.tokensPerSecond = input.tokensPerSecond * 2;
    const resultHalf = C.mfuCalculate(half);
    $('sensitivity-table').innerHTML = `<table>
      <tr><th></th><th>Current</th><th>Step time halved (2× throughput)</th></tr>
      <tr><td>Tokens/s</td><td>${fmtNum(result.tokensPerSecond)}</td><td>${fmtNum(resultHalf.tokensPerSecond)}</td></tr>
      <tr><td>Model FLOPs/s</td><td>${fmtFlops(result.modelFlopsPerSecond)}</td><td>${fmtFlops(resultHalf.modelFlopsPerSecond)}</td></tr>
      <tr><td>MFU</td><td>${result.percent.toFixed(1)}%</td><td>${resultHalf.percent.toFixed(1)}%</td></tr>
    </table>`;
  }

  document.querySelectorAll('.mode-switch button').forEach((b) => b.addEventListener('click', () => setMfuMode(b.dataset.mode)));
  ['in-flops-per-token', 'in-devices', 'in-mfu-peak-tf', 'in-tokens-per-sec', 'in-tokens-per-step', 'in-step-seconds'].forEach((id) => $(id).addEventListener('input', updateMfu));
  $('in-rate-mode').addEventListener('change', updateMfu);

  // ======================================================= RESET
  const DEFAULTS = {
    'in-M': '1024', 'in-precision': 'BF16', 'in-hw-preset': 'h100', 'in-peak-tf': '989', 'in-bandwidth': '3.35', 'in-bw-unit': 'TB/s',
    'in-devices': '4', 'in-mfu-peak-tf': '1000', 'in-rate-mode': 'rate', 'in-tokens-per-sec': '2000', 'in-tokens-per-step': '8000', 'in-step-seconds': '4',
  };
  function resetDefaults() {
    Object.entries(DEFAULTS).forEach(([id, v]) => { $(id).value = v; });
    applyHardwarePreset('h100');
    setMfuMode('generic');
    updateRoofline();
  }
  $('reset-defaults').addEventListener('click', resetDefaults);

  // ======================================================= INIT
  renderTrafficFormula();
  applyHardwarePreset('h100');
  updateRoofline();
  renderMoeBars();
  setMfuMode('generic');
})();
