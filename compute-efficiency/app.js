/* Simplified teaching UI. Uses window.ComputeEfficiency from calculations.js. */
(function () {
  'use strict';
  const C = window.ComputeEfficiency;
  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- format
  const fmtFlops = (n) => n == null ? 'N/A' :
    n >= 1e18 ? (n / 1e18).toFixed(2) + ' EFLOPs' :
    n >= 1e15 ? (n / 1e15).toFixed(2) + ' PFLOPs' :
    n >= 1e12 ? (n / 1e12).toFixed(2) + ' TFLOPs' :
    n >= 1e9 ? (n / 1e9).toFixed(2) + ' GFLOPs' :
    n >= 1e6 ? (n / 1e6).toFixed(2) + ' MFLOPs' :
    n.toLocaleString('en-US') + ' FLOPs';
  const fmtBytes = (n) => n == null ? 'N/A' :
    n >= 1e9 ? (n / 1e9).toFixed(2) + ' GB' :
    n >= 1e6 ? (n / 1e6).toFixed(2) + ' MB' :
    n >= 1e3 ? (n / 1e3).toFixed(2) + ' KB' :
    n.toFixed(0) + ' B';
  const fmtNum = (n, d = 2) => n == null ? 'N/A' : n.toLocaleString('en-US', { maximumFractionDigits: d });
  // Wrap a formatted "value + unit" string for use inside KaTeX math mode as upright text
  // (avoids math-mode comma/spacing quirks for thousands separators and unit abbreviations).
  const tx = (s) => `\\text{${s}}`;

  // ---------------------------------------------------------------- KaTeX
  function renderMath(id, tex, displayMode = true) {
    const el = $(id);
    if (!el || !window.katex) return;
    try { katex.render(tex, el, { throwOnError: false, displayMode, strict: false }); }
    catch (e) { el.textContent = tex; }
  }
  function katexSpan(tex) {
    if (!window.katex) return tex;
    try { return katex.renderToString(tex, { throwOnError: false, displayMode: false, strict: false }); }
    catch (e) { return tex; }
  }
  function renderNotationTable(id, rows) {
    $(id).innerHTML = `<thead><tr><th>Symbol</th><th>Meaning</th><th>Unit</th></tr></thead><tbody>` +
      rows.map(([sym, meaning, unit]) => `<tr><td>${katexSpan(sym)}</td><td>${meaning}</td><td>${unit}</td></tr>`).join('') +
      `</tbody>`;
  }

  function renderStaticMath() {
    renderMath('math-ai', 'AI = \\dfrac{F_{\\mathrm{ops}}}{M_{\\mathrm{bytes}}}');
    renderMath('math-proof', 'P_{\\mathrm{roof}}(AI) = \\min\\left(P_{\\mathrm{peak}},\\ AI \\times B_{\\mathrm{peak}}\\right)');
    renderMath('math-ridge', 'AI_{\\mathrm{ridge}} = \\dfrac{P_{\\mathrm{peak}}}{B_{\\mathrm{peak}}}');
    renderMath('math-mfu', '\\mathrm{MFU} = \\dfrac{F_{\\mathrm{step}}}{t_{\\mathrm{step}} \\times N_{\\mathrm{GPU}} \\times P_{\\mathrm{peak}}} \\times 100\\%');
    renderMath('math-flops-formula', 'F_{\\mathrm{ops}} = 2 \\times \\mathrm{Tokens} \\times (5120\\times1280 + 1280\\times32768)');
    renderNotationTable('roofline-notation', [
      ['F_{\\mathrm{ops}}', 'Floating-point operations', 'FLOPs'],
      ['M_{\\mathrm{bytes}}', 'Data transferred', 'Bytes'],
      ['AI', 'Arithmetic intensity', 'FLOPs/Byte'],
      ['P_{\\mathrm{peak}}', 'Peak compute throughput', 'TFLOP/s'],
      ['B_{\\mathrm{peak}}', 'Peak memory bandwidth', 'TB/s or GB/s'],
      ['AI_{\\mathrm{ridge}}', 'Arithmetic intensity at the ridge point', 'FLOPs/Byte'],
    ]);
    renderNotationTable('mfu-notation', [
      ['F_{\\mathrm{step}}', 'Useful model FLOPs for one training step', 'FLOPs'],
      ['t_{\\mathrm{step}}', 'Training step time', 's'],
      ['N_{\\mathrm{GPU}}', 'Number of GPUs', 'count'],
      ['P_{\\mathrm{peak}}', 'Peak compute per GPU', 'FLOP/s'],
      ['\\mathrm{MFU}', 'Model FLOPs Utilization', '%'],
    ]);
  }

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
  function hardwareInputs() {
    return { peakTF: Number($('in-peak-tf').value), bandwidth: Number($('in-bandwidth').value), bandwidthUnit: $('in-bw-unit').value, precision: $('in-precision').value };
  }
  function hardwareBytesPerSecond() {
    const h = hardwareInputs();
    return { peak: h.peakTF * 1e12, bw: C.bandwidthBytesPerSecond(h.bandwidth, h.bandwidthUnit) };
  }

  function applyHardwarePreset(id) {
    if (id === 'custom') { $('hw-source').textContent = 'Custom hardware — values entered manually.'; return; }
    const h = C.HARDWARE_PRESETS[id];
    $('in-peak-tf').value = h.peakTF; $('in-bandwidth').value = h.bandwidth; $('in-bw-unit').value = h.bandwidthUnit; $('in-precision').value = h.precision;
    $('hw-source').innerHTML = h.bandwidthSource
      ? `${h.name}. Sources: <a href="${h.bandwidthSource}" target="_blank" rel="noopener">bandwidth</a>, <a href="${h.computeSource}" target="_blank" rel="noopener">dense compute</a>.`
      : `${h.name} — hypothetical numbers for teaching only.`;
  }

  function sliderToM() { return Math.max(1, Math.round(Math.pow(10, Number($('in-M').value)))); }

  function conclusionLabel(bottleneck) { return bottleneck === 'Memory-Bound' ? 'Memory-bound' : 'Compute-bound'; }

  function renderComparisonCards() {
    const precision = $('in-precision').value;
    const { peak, bw } = hardwareBytesPerSecond();
    [[1, 'card-decode'], [1024, 'card-batch']].forEach(([M, id]) => {
      const s = C.queryProjectionScenario(M, precision);
      const r = C.roofline(s.flops, s.traffic.totalBytes, peak, bw);
      const card = $(id);
      card.querySelector('dl').innerHTML =
        `<dt>FLOPs</dt><dd>${fmtFlops(s.flops)}</dd>` +
        `<dt>Est. Memory Traffic</dt><dd>${fmtBytes(s.traffic.totalBytes)}</dd>` +
        `<dt>AI</dt><dd>${fmtNum(s.arithmeticIntensity)} FLOPs/Byte</dd>`;
      const cls = r.bottleneck === 'Memory-Bound' ? 'mem' : 'comp';
      const label = conclusionLabel(r.bottleneck);
      card.querySelector('.conclusion').className = 'conclusion ' + cls;
      card.querySelector('.conclusion').textContent = `${label} under the illustrative hardware configuration`;
    });
  }

  function renderCalcDetails() {
    const p = C.DEEPSEEK;
    const totalQDim = p.nHeads * p.headDim;
    renderMath('traffic-formula',
      '\\begin{aligned}' +
      `\\text{Weight bytes} &= (5120{\\times}1280 + 1280{\\times}${totalQDim}) \\times \\text{bytes} \\quad \\text{(read once)}\\\\[4pt]` +
      '\\text{Input bytes} &= \\text{Tokens} \\times 5120 \\times \\text{bytes}\\\\[4pt]' +
      '\\text{Intermediate} &= \\text{Tokens} \\times 1280 \\times \\text{bytes} \\times 2 \\quad \\text{(write, then read back)}\\\\[4pt]' +
      `\\text{Output bytes} &= \\text{Tokens} \\times ${totalQDim} \\times \\text{bytes}\\\\[4pt]` +
      '\\text{Total} &= \\text{Weight} + \\text{Input} + \\text{Intermediate} + \\text{Output}' +
      '\\end{aligned}');
    const precision = $('in-precision').value;
    const rows = [1, 1024].map((M) => {
      const s = C.queryProjectionScenario(M, precision);
      return `<tr><td>${M.toLocaleString()}</td><td>${fmtBytes(s.traffic.weightBytes)}</td><td>${fmtBytes(s.traffic.inputBytes)}</td><td>${fmtBytes(s.traffic.intermediateBytes)}</td><td>${fmtBytes(s.traffic.outputBytes)}</td><td>${fmtBytes(s.traffic.totalBytes)}</td></tr>`;
    }).join('');
    $('full-breakdown').innerHTML = `<table class="mini-table"><thead><tr><th>Tokens</th><th>Weights</th><th>Input</th><th>Intermediate</th><th>Output</th><th>Total</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function aiDerivationTex(scenario) {
    return '\\begin{aligned}' +
      'AI &= \\dfrac{F_{\\mathrm{ops}}}{M_{\\mathrm{bytes}}}\\\\[6pt]' +
      `&= \\dfrac{${tx(fmtNum(scenario.flops, 0) + ' FLOPs')}}{${tx(fmtNum(scenario.traffic.totalBytes, 0) + ' Bytes')}}\\\\[6pt]` +
      `&= \\boxed{${tx(fmtNum(scenario.arithmeticIntensity) + ' FLOPs/Byte')}}` +
      '\\end{aligned}';
  }

  function renderDerivations() {
    const precision = $('in-precision').value;
    const { peak, bw } = hardwareBytesPerSecond();
    const decode = C.queryProjectionScenario(1, precision);
    const batch = C.queryProjectionScenario(1024, precision);
    renderMath('derive-decode', aiDerivationTex(decode));
    $('derive-decode').insertAdjacentHTML('afterbegin', '<h4>Decode — 1 token</h4>');
    renderMath('derive-batch', aiDerivationTex(batch));
    $('derive-batch').insertAdjacentHTML('afterbegin', '<h4>Batch — 1,024 tokens</h4>');
    if (peak > 0 && bw > 0) {
      const h = hardwareInputs();
      renderMath('derive-ridge',
        '\\begin{aligned}' +
        'AI_{\\mathrm{ridge}} &= \\dfrac{P_{\\mathrm{peak}}}{B_{\\mathrm{peak}}}\\\\[6pt]' +
        `&= \\dfrac{${tx(fmtNum(h.peakTF) + ' TFLOP/s')}}{${tx(fmtNum(h.bandwidth) + ' ' + h.bandwidthUnit)}}\\\\[6pt]` +
        `&= \\boxed{${tx(fmtNum(peak / bw) + ' FLOPs/Byte')}}` +
        '\\end{aligned}');
      $('derive-ridge').insertAdjacentHTML('afterbegin', '<h4>Ridge Point — at the current hardware setting</h4>');
    }
  }

  function drawRooflineChart() {
    const { peak, bw } = hardwareBytesPerSecond();
    const precision = $('in-precision').value;
    let errors = [];
    if (!(peak > 0)) errors.push('Peak compute must be positive.');
    if (!(bw > 0)) errors.push('Bandwidth must be positive.');
    if (errors.length) { $('roofline-error').hidden = false; $('roofline-error').textContent = errors.join(' '); return; }
    $('roofline-error').hidden = true;
    const ridge = peak / bw;

    const decode = C.queryProjectionScenario(1, precision);
    const batch = C.queryProjectionScenario(1024, precision);
    const decodeR = C.roofline(decode.flops, decode.traffic.totalBytes, peak, bw);
    const batchR = C.roofline(batch.flops, batch.traffic.totalBytes, peak, bw);

    const rawM = sliderToM();
    const near0 = (a, b) => Math.abs(a - b) <= Math.max(1, b * 0.04);
    const exploreM = near0(rawM, 1) ? 1 : near0(rawM, 1024) ? 1024 : rawM;
    $('M-readout').textContent = exploreM.toLocaleString();
    const explore = C.queryProjectionScenario(exploreM, precision);
    const exploreR = C.roofline(explore.flops, explore.traffic.totalBytes, peak, bw);
    const readout = $('M-ai-readout');
    readout.textContent = `AI ≈ ${fmtNum(exploreR.ai)} · ${conclusionLabel(exploreR.bottleneck)}`;
    readout.className = 'ai-readout ' + (exploreR.bottleneck === 'Memory-Bound' ? 'mem' : 'comp');

    const points = [
      { label: 'Decode', M: 1, ai: decodeR.ai, perf: decodeR.performance, color: '#c25b8f', bold: true },
      { label: 'Batch', M: 1024, ai: batchR.ai, perf: batchR.performance, color: '#3a6fd8', bold: true },
    ];
    const showExplore = exploreM !== 1 && exploreM !== 1024;
    if (showExplore) points.push({ label: `M=${exploreM.toLocaleString()}`, M: exploreM, ai: exploreR.ai, perf: exploreR.performance, color: '#1f8f5f', bold: false });

    const allAi = [ridge, ...points.map((p) => p.ai)];
    const xmin = Math.pow(10, Math.floor(Math.log10(Math.min(...allAi) / 3)));
    const xmax = Math.pow(10, Math.ceil(Math.log10(Math.max(...allAi) * 3)));
    const values = [peak, xmin * bw, ...points.map((p) => p.perf)];
    const ymin = Math.pow(10, Math.floor(Math.log10(Math.min(...values) * 0.5)));
    const ymax = Math.pow(10, Math.ceil(Math.log10(Math.max(...values) * 1.5)));
    const left = 70, right = 940, top = 50, bottom = 370;
    const x = (v) => left + (Math.log10(v) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin)) * (right - left);
    const y = (v) => bottom - (Math.log10(v) - Math.log10(ymin)) / (Math.log10(ymax) - Math.log10(ymin)) * (bottom - top);

    let svg = '';
    // Shaded regions + ceiling line.
    svg += `<rect x="${left}" y="${top}" width="${x(ridge) - left}" height="${bottom - top}" fill="#c25b8f" opacity="0.07"/>`;
    svg += `<rect x="${x(ridge)}" y="${top}" width="${right - x(ridge)}" height="${bottom - top}" fill="#3a6fd8" opacity="0.07"/>`;
    svg += `<path d="M${x(xmin)},${y(xmin * bw)} L${x(ridge)},${y(peak)} L${x(xmax)},${y(peak)}" fill="none" stroke="#16233f" stroke-width="4"/>`;
    svg += `<path d="M${x(ridge)},${top}V${bottom}" stroke="#9daec3" stroke-dasharray="6 5" stroke-width="2"/>`;
    // Big always-visible region labels.
    svg += `<text x="${left + 18}" y="${top + 30}" font-size="22" font-weight="700" fill="#c25b8f">Memory-bound</text>`;
    svg += `<text x="${right - 18}" y="${top + 30}" text-anchor="end" font-size="22" font-weight="700" fill="#3a6fd8">Compute-bound</text>`;
    svg += `<text x="${x(ridge)}" y="${top - 18}" text-anchor="middle" font-size="15" fill="#5c6b85">Ridge Point</text>`;
    // Minimal axis labels (just endpoints + ridge, not a dense grid).
    svg += `<text x="${left}" y="${bottom + 32}" font-size="14" fill="#5c6b85">low</text>`;
    svg += `<text x="${right}" y="${bottom + 32}" text-anchor="end" font-size="14" fill="#5c6b85">high</text>`;
    svg += `<text x="${(left + right) / 2}" y="${bottom + 32}" text-anchor="middle" font-size="15" fill="#30405f">Arithmetic Intensity (FLOPs / Byte)</text>`;
    svg += `<text x="${left - 46}" y="${(top + bottom) / 2}" text-anchor="middle" font-size="15" fill="#30405f" transform="rotate(-90 ${left - 46} ${(top + bottom) / 2})">Performance</text>`;

    points.forEach((p) => {
      const cx = x(p.ai), cy = y(p.perf);
      const r = p.bold ? 10 : 8;
      svg += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${p.color}" stroke="#fff" stroke-width="3"/>`;
      const ly = cy - 18 < top + 50 ? cy + 30 : cy - 18;
      svg += `<text x="${cx}" y="${ly}" text-anchor="middle" font-size="${p.bold ? 19 : 15}" font-weight="700" fill="${p.color}">${p.label}</text>`;
    });

    $('roofline-chart').innerHTML = svg;
  }

  function updateRoofline() {
    renderComparisonCards();
    renderCalcDetails();
    renderDerivations();
    drawRooflineChart();
  }

  $('in-M').addEventListener('input', drawRooflineChart);
  $('in-precision').addEventListener('change', updateRoofline);
  $('in-peak-tf').addEventListener('input', () => { $('in-hw-preset').value = 'custom'; drawRooflineChart(); renderDerivations(); });
  $('in-bandwidth').addEventListener('input', () => { $('in-hw-preset').value = 'custom'; drawRooflineChart(); renderDerivations(); });
  $('in-bw-unit').addEventListener('change', () => { $('in-hw-preset').value = 'custom'; drawRooflineChart(); renderDerivations(); });
  $('in-hw-preset').addEventListener('change', (e) => { applyHardwarePreset(e.target.value); updateRoofline(); });

  // ======================================================= MFU TAB
  function mfuInputs() {
    return {
      deviceCount: Math.max(1, Math.round(Number($('in-devices').value) || 1)),
      peakTF: Number($('in-mfu-peak-tf').value),
      tokensPerStep: Number($('in-tokens-per-step').value),
      trainingFlopsPerToken: Number($('in-flops-token-gflops').value) * 1e9,
      stepSeconds: Number($('in-step-seconds').value),
      rateMode: 'step',
    };
  }

  function updateMfu() {
    $('step-readout').textContent = Number($('in-step-seconds').value).toFixed(1) + ' s';
    $('flops-token-readout').textContent = fmtNum(Number($('in-flops-token-gflops').value), 0) + ' GFLOPs';
    const input = mfuInputs();
    let result, error = null;
    try { result = C.mfuCalculate(input); } catch (e) { error = e.message; }
    if (error) {
      ['mfu-step1', 'mfu-step2', 'mfu-step3'].forEach((id) => { $(id).textContent = ''; });
      $('mfu-step1').textContent = error;
      $('mfu-percent').textContent = 'N/A';
      $('mfu-bar').innerHTML = '';
      $('mfu-takeaway').textContent = '';
      return;
    }
    const fStep = input.trainingFlopsPerToken * input.tokensPerStep;
    const peakTotal = input.deviceCount * input.peakTF;

    renderMath('mfu-step1',
      '\\begin{aligned}' +
      'F_{\\mathrm{step}} &= N_{\\mathrm{tokens}} \\times F_{\\mathrm{token}}\\\\[6pt]' +
      `&= ${tx(fmtNum(input.tokensPerStep, 0) + ' tokens')} \\times ${tx(fmtFlops(input.trainingFlopsPerToken))}\\\\[6pt]` +
      `&= \\boxed{${tx(fmtFlops(fStep))}}` +
      '\\end{aligned}');
    $('mfu-step1').insertAdjacentHTML('afterbegin', '<h4>Step 1 — Useful model FLOPs per training step</h4>');

    renderMath('mfu-step2',
      '\\begin{aligned}' +
      'P_{\\mathrm{total}} &= N_{\\mathrm{GPU}} \\times P_{\\mathrm{peak}}\\\\[6pt]' +
      `&= ${input.deviceCount.toLocaleString('en-US')} \\times ${tx(fmtNum(input.peakTF) + ' TFLOP/s')}\\\\[6pt]` +
      `&= \\boxed{${tx(fmtNum(peakTotal, 0) + ' TFLOP/s')}}` +
      '\\end{aligned}');
    $('mfu-step2').insertAdjacentHTML('afterbegin', '<h4>Step 2 — Total hardware peak throughput</h4>');

    renderMath('mfu-step3',
      '\\begin{aligned}' +
      '\\mathrm{MFU} &= \\dfrac{F_{\\mathrm{step}}}{t_{\\mathrm{step}} \\times N_{\\mathrm{GPU}} \\times P_{\\mathrm{peak}}} \\times 100\\%\\\\[6pt]' +
      `&= \\dfrac{${tx(fmtFlops(fStep))}}{${tx(input.stepSeconds.toFixed(1) + ' s')} \\times ${input.deviceCount.toLocaleString('en-US')} \\times ${tx(fmtNum(input.peakTF) + ' TFLOP/s')}} \\times 100\\%\\\\[6pt]` +
      `&\\approx \\boxed{${tx(result.percent.toFixed(1) + '\\%')}}` +
      '\\end{aligned}');
    $('mfu-step3').insertAdjacentHTML('afterbegin', '<h4>Step 3 — MFU</h4>');

    $('mfu-percent').textContent = result.percent.toFixed(0) + '%';
    drawMfuBar(Math.min(100, result.percent));
    $('mfu-takeaway').textContent = `Under these illustrative assumptions, useful model training computations achieve throughput equivalent to approximately ${result.percent.toFixed(1)}% of the hardware's theoretical peak.`;
    if (result.exceedsPeak) $('mfu-takeaway').textContent = 'This configuration exceeds 100% of peak — check the Advanced Settings values.';
  }

  function drawMfuBar(pct) {
    const left = 20, right = 940, top = 20, h = 50;
    let svg = `<rect x="${left}" y="${top}" width="${right - left}" height="${h}" rx="12" fill="#eef2f9"/>`;
    svg += `<rect x="${left}" y="${top}" width="${(right - left) * pct / 100}" height="${h}" rx="12" fill="#3a6fd8"/>`;
    svg += `<text x="${right}" y="${top + h + 24}" text-anchor="end" font-size="13" fill="#5c6b85">100% = hardware's theoretical peak</text>`;
    $('mfu-bar').innerHTML = svg;
  }

  ['in-devices', 'in-mfu-peak-tf', 'in-tokens-per-step', 'in-flops-token-gflops'].forEach((id) => $(id).addEventListener('input', updateMfu));
  $('in-step-seconds').addEventListener('input', updateMfu);

  // ======================================================= RESET
  const DEFAULTS = {
    'in-M': '3.0103', 'in-precision': 'BF16', 'in-hw-preset': 'h100', 'in-peak-tf': '989', 'in-bandwidth': '3.35', 'in-bw-unit': 'TB/s',
    'in-devices': '2048', 'in-mfu-peak-tf': '989.5', 'in-tokens-per-step': '100600000', 'in-flops-token-gflops': '60', 'in-step-seconds': '6',
  };
  function resetDefaults() {
    Object.entries(DEFAULTS).forEach(([id, v]) => { $(id).value = v; });
    applyHardwarePreset('h100');
    updateRoofline();
    updateMfu();
  }
  $('reset-defaults').addEventListener('click', resetDefaults);

  // ======================================================= INIT
  renderStaticMath();
  applyHardwarePreset('h100');
  updateRoofline();
  updateMfu();
})();
