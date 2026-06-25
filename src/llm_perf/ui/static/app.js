/**
 * llm-perf SPA — Application Logic
 *
 * Handles: accordion, tabs, form state, API calls, Plotly charts.
 */

/* ── State ────────────────────────────────────────────── */
const state = {
  templates: {},
  hardware: {},
  presets: {},
  lastResult: null,
  lastSearch: null,
  hasRun: false,
  activeParPhase: 'train',
  scenario: 'post_training', // 'inference' | 'pretraining' | 'post_training'
  modified: { model: false, hardware: false, rl: false, search: false },
};

/* Per-scenario UI config: brand tagline, RL section title, KPI cards. */
const SCENARIOS = {
  inference: {
    tagline: 'Inference Performance Modeling',
    rlTitle: 'Workload',
    parLegend: 'Parallelism (generation)',
  },
  pretraining: {
    tagline: 'Pretraining Performance Modeling',
    rlTitle: 'Training',
    parLegend: 'Parallelism (training)',
  },
  post_training: {
    tagline: 'Post-Training (RL) Performance Modeling',
    rlTitle: 'RL Training',
    parLegend: 'Parallelism',
  },
};

/* ── DOM refs ─────────────────────────────────────────── */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

/* ── Init ─────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  initAccordion();
  initTabs();
  initDrawer();
  initScenarioTabs();
  initSegmentedControls();
  initConditionalFields();
  initAutoComputed();
  initButtons();
  initJsonEditor();
  applyScenario('post_training', { rerun: false });
  await loadConfigs();
  initTemplateListener();
});

/* ── Scenario Tabs ────────────────────────────────────── */
function initScenarioTabs() {
  $$('.scenario-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const scn = tab.dataset.scenario;
      if (scn === state.scenario) return;
      applyScenario(scn, { rerun: true });
    });
  });
}

function applyScenario(scn, { rerun = false } = {}) {
  state.scenario = scn;
  const cfg = SCENARIOS[scn] || SCENARIOS.post_training;

  // Active tab styling
  $$('.scenario-tab').forEach((t) =>
    t.classList.toggle('active', t.dataset.scenario === scn)
  );

  // Header tagline + RL section title + parallelism legend
  const tagline = $('#brand-tagline');
  if (tagline) tagline.textContent = cfg.tagline;
  const rlTitle = $('#rl-section-title');
  if (rlTitle) rlTitle.textContent = cfg.rlTitle;
  const parLegend = $('#parallelism-legend');
  if (parLegend) parLegend.textContent = cfg.parLegend;

  applyScenarioVisibility(scn);
  configureKPIs(scn);

  // Reset chart back to Timeline (search-results may be hidden now)
  const activeTab = $('.chart-tab.active');
  if (activeTab && activeTab.classList.contains('hidden')) {
    $$('.chart-tab').forEach((t) => t.classList.remove('active'));
    $$('.chart-panel').forEach((p) => p.classList.remove('active'));
    $('.chart-tab[data-tab="timeline"]').classList.add('active');
    $('#panel-timeline').classList.add('active');
  }

  if (rerun && state.hasRun) {
    runAnalysis();
  }
}

/* Toggle elements carrying a data-scn attribute for the active scenario. */
function applyScenarioVisibility(scn) {
  $$('[data-scn]').forEach((el) => {
    const scopes = el.dataset.scn.split(/\s+/).filter(Boolean);
    el.classList.toggle('hidden', !scopes.includes(scn));
  });

  // Parallelism panels: post-training uses the Train/Gen/Ref switcher;
  // inference shows only the gen panel, pretraining only the train panel.
  const panels = {
    train: $('#par-panel-train'),
    gen: $('#par-panel-gen'),
    ref: $('#par-panel-ref'),
  };
  if (scn === 'inference') {
    panels.train.classList.add('hidden');
    panels.gen.classList.remove('hidden');
    panels.ref.classList.add('hidden');
  } else if (scn === 'pretraining') {
    panels.train.classList.remove('hidden');
    panels.gen.classList.add('hidden');
    panels.ref.classList.add('hidden');
  } else {
    // post_training: honor the active phase from the switcher
    const phase = state.activeParPhase;
    panels.train.classList.toggle('hidden', phase !== 'train');
    panels.gen.classList.toggle('hidden', phase !== 'gen');
    panels.ref.classList.toggle('hidden', phase !== 'ref');
  }
}

/* Configure the KPI card labels (and which are visible) per scenario. */
const KPI_CONFIG = {
  inference: [
    { id: 'kpi-epoch', label: 'Gen Time' },
    { id: 'kpi-gen', label: 'Gen TPS' },
    { id: 'kpi-train', label: 'Samples/s' },
    { id: 'kpi-ref', label: 'KV Cache' },
  ],
  pretraining: [
    { id: 'kpi-epoch', label: 'Step Time' },
    { id: 'kpi-gen', label: 'Train TPS' },
    { id: 'kpi-train', label: 'Samples/s' },
    { id: 'kpi-ref', label: 'Train Mem' },
  ],
  post_training: [
    { id: 'kpi-epoch', label: 'Step Time' },
    { id: 'kpi-gen', label: 'Gen TPS' },
    { id: 'kpi-train', label: 'Train TPS' },
    { id: 'kpi-ref', label: 'Ref TPS' },
  ],
};

function configureKPIs(scn) {
  const cfg = KPI_CONFIG[scn] || KPI_CONFIG.post_training;
  cfg.forEach((k) => {
    const card = $(`#${k.id}`);
    if (!card) return;
    card.classList.remove('hidden');
    const labelEl = card.querySelector('.kpi-label');
    if (labelEl) labelEl.textContent = k.label;
    if (!state.hasRun) {
      card.querySelector('.kpi-value').innerHTML = '&mdash;';
      card.querySelector('.kpi-detail').textContent = 'awaiting analysis';
      card.dataset.status = 'neutral';
    }
  });
}

/* ── Accordion ────────────────────────────────────────── */
function initAccordion() {
  // Open the first section by default
  const first = $('.accordion-section[data-section="model"]');
  if (first) first.setAttribute('data-open', '');

  $$('.accordion-trigger').forEach((trigger) => {
    trigger.addEventListener('click', () => {
      const section = trigger.closest('.accordion-section');
      const isOpen = section.hasAttribute('data-open');

      // Close all
      $$('.accordion-section').forEach((s) => {
        s.removeAttribute('data-open');
        s.querySelector('.accordion-trigger').setAttribute('aria-expanded', 'false');
      });

      // Open clicked (if it was closed)
      if (!isOpen) {
        section.setAttribute('data-open', '');
        trigger.setAttribute('aria-expanded', 'true');
      }
    });
  });
}

/* ── Tabs ─────────────────────────────────────────────── */
function initTabs() {
  $$('.chart-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.chart-tab').forEach((t) => t.classList.remove('active'));
      $$('.chart-panel').forEach((p) => p.classList.remove('active'));
      tab.classList.add('active');
      $(`#panel-${tab.dataset.tab}`).classList.add('active');
    });
  });
}

/* ── Drawer (mobile) ──────────────────────────────────── */
function initDrawer() {
  const toggle = $('#drawer-toggle');
  const panel = $('#config-panel');
  const overlay = $('#drawer-overlay');

  if (!toggle) return;

  toggle.addEventListener('click', () => {
    panel.classList.toggle('open');
    overlay.classList.toggle('visible');
  });

  overlay.addEventListener('click', () => {
    panel.classList.remove('open');
    overlay.classList.remove('visible');
  });
}

/* ── Segmented Controls ───────────────────────────────── */
function initSegmentedControls() {
  // Model source
  const modelSource = $('#model-source');
  if (modelSource) {
    modelSource.querySelectorAll('.seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        modelSource.querySelectorAll('.seg-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const val = btn.dataset.value;

        $('#field-template').classList.toggle('hidden', val !== 'template');
        $('#field-hf').classList.toggle('hidden', val !== 'huggingface');

        markModified('model');
      });
    });
  }

  // Search mode
  const searchMode = $('#search-mode');
  if (searchMode) {
    searchMode.querySelectorAll('.seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        searchMode.querySelectorAll('.seg-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const val = btn.dataset.value;

        $('#pareto-fields').classList.toggle('hidden', val !== 'pareto');
        $('#sensitivity-fields').classList.toggle('hidden', val !== 'sensitivity');

        updateSearchSummary();
        markModified('search');
      });
    });
  }

  // Parallelism phase
  const parPhase = $('#par-phase');
  if (parPhase) {
    parPhase.querySelectorAll('.seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        parPhase.querySelectorAll('.seg-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const val = btn.dataset.value;
        state.activeParPhase = val;

        $('#par-panel-train').classList.toggle('hidden', val !== 'train');
        $('#par-panel-gen').classList.toggle('hidden', val !== 'gen');
        $('#par-panel-ref').classList.toggle('hidden', val !== 'ref');

        markModified('hardware');
      });
    });
  }
}

/* ── Conditional Fields ───────────────────────────────── */
function initConditionalFields() {
  const attnSelect = $('#attention-type');
  const ffnSelect = $('#ffn-type');
  const residualSelect = $('#residual-type');
  const specDecode = $('#spec-decode');
  const hasMtp = $('#has-mtp-aux');

  if (attnSelect) {
    attnSelect.addEventListener('change', () => {
      updateConditionalFields();
      markModified('model');
    });
  }
  if (ffnSelect) {
    ffnSelect.addEventListener('change', () => {
      updateConditionalFields();
      markModified('model');
    });
  }
  if (residualSelect) {
    residualSelect.addEventListener('change', () => {
      updateConditionalFields();
      markModified('model');
    });
  }
  if (specDecode) {
    specDecode.addEventListener('change', () => {
      $('#mtp-fields').classList.toggle('hidden', !specDecode.checked);
      markModified('rl');
    });
  }
  if (hasMtp) {
    hasMtp.addEventListener('change', () => {
      updateConditionalFields();
      markModified('model');
    });
  }

  // Track all input changes for modification dots
  $$('.accordion-section[data-section="model"] input, .accordion-section[data-section="model"] select').forEach((el) => {
    el.addEventListener('change', () => markModified('model'));
  });
  $$('.accordion-section[data-section="hardware"] input, .accordion-section[data-section="hardware"] select').forEach((el) => {
    el.addEventListener('change', () => markModified('hardware'));
  });
  $$('.accordion-section[data-section="rl"] input, .accordion-section[data-section="rl"] select').forEach((el) => {
    el.addEventListener('change', () => markModified('rl'));
  });
  $$('.accordion-section[data-section="search"] input, .accordion-section[data-section="search"] select').forEach((el) => {
    el.addEventListener('change', () => markModified('search'));
  });
}

function updateConditionalFields() {
  const attn = $('#attention-type').value;
  const ffn = $('#ffn-type').value;
  const residual = $('#residual-type').value;

  // DSA reuses the compression dims (MLA fields) and the window (SWA field),
  // plus its own dsa-fields block.
  $('#mla-fields').classList.toggle('hidden', attn !== 'MLA' && attn !== 'DSA');
  $('#swa-fields').classList.toggle('hidden', attn !== 'SWA' && attn !== 'DSA');
  $('#dsa-fields').classList.toggle('hidden', attn !== 'DSA');
  $('#moe-fields').classList.toggle('hidden', ffn !== 'MoE');
  $('#mhc-fields').classList.toggle('hidden', residual !== 'mHC');

  // When switching attention type, update num_kv_heads and type-specific params:
  // - MHA: kv_heads must equal num_heads
  // - GQA: kv_heads < num_heads (keep current or default to common ratio)
  // - MLA: kv_heads not used; auto-fill kv_compression_dim etc. if zero
  // - SWA: same kv_heads as GQA; auto-fill window_size if zero
  const numHeads = intVal('num-heads');
  const kvHeadsEl = $('#num-kv-heads');
  if (attn === 'MHA') {
    kvHeadsEl.value = numHeads;
  } else if (attn === 'GQA' || attn === 'SWA') {
    // If current kv_heads equals num_heads (was MHA), reset to a typical GQA ratio
    if (intVal('num-kv-heads') === numHeads || intVal('num-kv-heads') === 0) {
      kvHeadsEl.value = Math.max(1, Math.round(numHeads / 8));
    }
  }
  // MLA: auto-fill compression dims if they are zero
  if (attn === 'MLA') {
    const hiddenSize = intVal('hidden-size');
    if (intVal('kv-compression-dim') === 0) {
      $('#kv-compression-dim').value = Math.round(hiddenSize / 4);
    }
    if (intVal('query-compression-dim') === 0) {
      $('#query-compression-dim').value = Math.round(hiddenSize / 3);
    }
    if (intVal('rope-dim') === 0) {
      $('#rope-dim').value = 64;
    }
  }
  // SWA: auto-fill window_size if zero
  if (attn === 'SWA') {
    if (intVal('window-size') === 0) {
      $('#window-size').value = 4096;
    }
  }

  // Update layers-summary badge to reflect the current attention type
  updateLayersSummaryBadge(attn);

  // MTP auxiliary sub-fields
  const hasMtp = $('#has-mtp-aux');
  if (hasMtp) {
    const mtpFields = $('#mtp-aux-fields');
    const mtpSubFields = mtpFields.querySelectorAll('.field:not(:first-child)');
    mtpSubFields.forEach((el) => el.classList.toggle('hidden', !hasMtp.checked));
  }
}

/* ── Auto-computed Fields ─────────────────────────────── */
function initAutoComputed() {
  // DP = total_devices / (TP * PP * CP). EP shares the DP devices (experts are
  // split within the data-parallel group), so it does NOT divide device count.
  const recomputeDP = (phase) => {
    const total = intVal('total-devices');
    const tp = intVal(`${phase}-tp`);
    const pp = intVal(`${phase}-pp`);
    const cp = intVal(`${phase}-cp`);
    const divisor = tp * pp * cp;
    const dp = divisor > 0 ? Math.max(1, Math.floor(total / divisor)) : total;
    $(`#${phase}-dp`).value = dp;
  };

  const recomputeAllDP = () => {
    ['train', 'gen', 'ref'].forEach(recomputeDP);

    // Nodes
    const total = intVal('total-devices');
    const perNode = intVal('devices-per-node');
    const nodes = perNode > 0 ? Math.ceil(total / perNode) : 1;
    $('#num-nodes').value = nodes;

    updateHardwareSummary();
  };

  ['total-devices'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.addEventListener('input', recomputeAllDP);
  });

  // Train phase inputs
  ['train-tp', 'train-pp', 'train-ep', 'train-cp'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.addEventListener('input', () => recomputeDP('train'));
  });
  // Gen phase inputs
  ['gen-tp', 'gen-pp', 'gen-ep', 'gen-cp'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.addEventListener('input', () => recomputeDP('gen'));
  });
  // Ref phase inputs
  ['ref-tp', 'ref-pp', 'ref-ep', 'ref-cp'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.addEventListener('input', () => recomputeDP('ref'));
  });

  // Group size change updates RL summary
  const recomputeRLSummary = () => {
    updateRLSummary();
  };

  ['group-size'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.addEventListener('input', recomputeRLSummary);
  });

  // Hardware profile change
  $('#hw-profile').addEventListener('change', () => {
    const prof = state.hardware[$('#hw-profile').value];
    if (prof) {
      $('#devices-per-node').value = prof.devices_per_node;
      recomputeAllDP();
    }
    markModified('hardware');
  });

  $('#ref-model').addEventListener('change', () => updateRLSummary());
}

/* ── Summary Updaters ─────────────────────────────────── */
function updateModelSummary() {
  const name = $('#model-name').value || 'Custom';
  const layers = $('#num-layers').value;
  const attn = $('#attention-type').value;
  const dtype = $('#dtype').value;
  $('#model-summary').textContent = `${name} \u00b7 ${layers}L \u00b7 ${attn} \u00b7 ${dtype}`;
}

function updateLayersSummaryBadge(attn) {
  const badge = $('#layers-summary-badge');
  const field = $('#field-layers-summary');
  if (!badge || !field) return;

  const current = badge.textContent.trim();
  if (!current) return;

  // Replace the attention type in the summary string.
  // Format: "4x GQA+SwiGLU, 47x GQA+MoE E80"
  // → "4x MHA+SwiGLU, 47x MHA+MoE E80" etc.
  const knownAttnTypes = ['MHA', 'GQA', 'MLA', 'SWA'];
  const updated = current.replace(
    new RegExp('\\b(' + knownAttnTypes.join('|') + ')\\b', 'g'),
    attn
  );
  badge.textContent = updated;
}

function updateHardwareSummary() {
  const hw = $('#hw-profile').value;
  const devices = $('#total-devices').value;
  const nodes = $('#num-nodes').value;
  $('#hardware-summary').textContent = `${hw} \u00b7 ${devices} devices \u00b7 ${nodes} node${nodes > 1 ? 's' : ''}`;
}

function updateRLSummary() {
  const trainBatch = intVal('train-batch-size');
  const group = $('#group-size').value;
  const ref = $('#ref-model').checked ? 'ref model' : 'no ref';
  $('#rl-summary').textContent = `batch=${trainBatch} \u00b7 grp=${group} \u00b7 ${ref}`;
}

function updateSearchSummary() {
  const mode = getSearchMode();
  if (mode === 'pareto') {
    const counts = $('#device-counts').value.split(',').filter(Boolean).length;
    $('#search-summary').textContent = `Pareto \u00b7 ${counts} device counts`;
  } else {
    const param = $('#sweep-param').value;
    $('#search-summary').textContent = `Sensitivity \u00b7 ${param}`;
  }
}

function markModified(section) {
  state.modified[section] = true;
  const dot = $(`.accordion-section[data-section="${section}"] .accordion-dot`);
  if (dot) {
    dot.classList.remove('dot-default');
    dot.classList.add('dot-modified');
  }

  // Update summaries
  if (section === 'model') updateModelSummary();
  if (section === 'hardware') updateHardwareSummary();
  if (section === 'rl') updateRLSummary();
  if (section === 'search') updateSearchSummary();
}

function resetModified() {
  Object.keys(state.modified).forEach((k) => {
    state.modified[k] = false;
    const dot = $(`.accordion-section[data-section="${k}"] .accordion-dot`);
    if (dot) {
      dot.classList.remove('dot-modified');
      dot.classList.add('dot-default');
    }
  });
}

/* ── Config Loading ───────────────────────────────────── */
/* Find the runtime preset (model + hardware + parallelism + RL, matching the
   demo configs) for a model-template name. Preset keys carry a device suffix
   like "DeepSeekV3-671B (128x 910C)", so match by exact name or "<name> ("
   prefix — this makes selecting a model load the same config the demo uses. */
function findPresetForModel(name) {
  if (state.presets[name]) return state.presets[name];
  const key = Object.keys(state.presets).find((k) => k.startsWith(name + ' ('));
  return key ? state.presets[key] : null;
}

async function loadConfigs() {
  try {
    const [modelsResp, hwResp, presetsResp] = await Promise.all([
      fetch('/api/models'),
      fetch('/api/hardware'),
      fetch('/api/presets'),
    ]);
    const modelsData = await modelsResp.json();
    const hwData = await hwResp.json();
    const presetsData = await presetsResp.json();

    state.templates = modelsData.templates || {};
    state.hardware = hwData.profiles || {};
    state.presets = presetsData.presets || {};

    // Apply first hardware
    const firstHW = Object.keys(state.hardware)[0];
    if (firstHW && state.hardware[firstHW]) {
      $('#devices-per-node').value = state.hardware[firstHW].devices_per_node;
    }

    // If presets available, apply the first preset and auto-run
    const firstPreset = Object.keys(state.presets)[0];
    if (firstPreset) {
      const templateVal = $('#model-template').value;
      applyPreset(findPresetForModel(templateVal) || state.presets[firstPreset]);
      // Auto-run prediction on load
      await runAnalysis();
    } else {
      // Fallback: apply first model template
      const firstTemplate = Object.keys(state.templates)[0];
      if (firstTemplate) {
        applyModelTemplate(firstTemplate);
      }
    }
  } catch (e) {
    console.error('Failed to load configs:', e);
  }
}

/* Fill the DSA-specific form fields from a layer object (shared by template /
   preset / HF loaders). Safe to call for non-DSA layers (values default to 0). */
function applyDsaFields(layer) {
  if (!layer) return;
  $('#compress-ratio').value = layer.compress_ratio || 0;
  $('#compress-c-kv').value = layer.compress_c_kv || 0;
  $('#compress-coeff').value = layer.compress_coeff || 0;
  $('#index-n-heads').value = layer.index_n_heads || 0;
  $('#index-head-dim').value = layer.index_head_dim || 0;
  $('#index-topk').value = layer.index_topk || 0;
  $('#q-lora-rank').value = layer.q_lora_rank || 0;
  $('#o-lora-rank').value = layer.o_lora_rank || 0;
  $('#o-groups').value = layer.o_groups || 0;
}

function applyModelTemplate(name) {
  const t = state.templates[name];
  if (!t) return;

  $('#model-name').value = t.name;
  $('#hidden-size').value = t.hidden_size;
  $('#vocab-size').value = t.vocab_size;
  $('#num-layers').value = t.num_layers;
  $('#dtype').value = t.dtype;

  if (t.layer) {
    $('#attention-type').value = t.layer.attention || 'GQA';
    $('#ffn-type').value = t.layer.ffn || 'SwiGLU';
    $('#residual-type').value = t.layer.residual || 'standard';
    $('#num-heads').value = t.layer.num_heads || 32;
    $('#num-kv-heads').value = t.layer.num_kv_heads || 8;
    $('#head-dim').value = t.layer.head_dim || 128;
    $('#intermediate-size').value = t.layer.intermediate_size || 14336;
    $('#num-experts').value = t.layer.num_experts || 1;
    $('#top-k').value = t.layer.top_k || 1;
    $('#shared-experts').value = t.layer.num_shared_experts || 0;
    $('#expert-intermediate-size').value = t.layer.expert_intermediate_size || 0;
    $('#shared-intermediate-size').value = t.layer.shared_intermediate_size || 0;
    $('#kv-compression-dim').value = t.layer.kv_compression_dim || 0;
    $('#query-compression-dim').value = t.layer.query_compression_dim || 0;
    $('#rope-dim').value = t.layer.rope_dim || 0;
    $('#window-size').value = t.layer.window_size || 0;
    $('#mhc-expansion').value = t.layer.mhc_expansion || 4;
    applyDsaFields(t.layer);
  }

  // Auxiliary
  if (t.auxiliary && t.auxiliary.mtp_depth) {
    $('#mtp-aux-fields').classList.remove('hidden');
    $('#has-mtp-aux').checked = true;
    $('#mtp-depth').value = t.auxiliary.mtp_depth;
  } else {
    $('#has-mtp-aux').checked = false;
  }

  // Layers summary badge
  if (t.layers_summary) {
    $('#field-layers-summary').classList.remove('hidden');
    $('#layers-summary-badge').textContent = t.layers_summary;
  } else {
    $('#field-layers-summary').classList.add('hidden');
  }

  updateConditionalFields();
  updateModelSummary();
}

function initTemplateListener() {
  $('#model-template').addEventListener('change', async (e) => {
    const name = e.target.value;
    // If a full preset exists for this template, apply it and auto-run
    const preset = findPresetForModel(name);
    if (preset) {
      applyPreset(preset);
      await runAnalysis();
    } else {
      applyModelTemplate(name);
      markModified('model');
    }
  });
}

/* ── Buttons ──────────────────────────────────────────── */
function initButtons() {
  // Run Analysis
  $('#run-btn').addEventListener('click', runAnalysis);

  // Run Search
  $('#run-search-btn').addEventListener('click', runSearch);

  // HF Load
  $('#hf-load-btn').addEventListener('click', loadHFModel);

  // Error close
  $('#error-close').addEventListener('click', () => {
    $('#error-banner').classList.add('hidden');
  });
}

/* ── Build Parallelism Input ──────────────────────────── */
function buildParallelismInput(phase) {
  const prefix = phase;
  return {
    tp: intVal(`${prefix}-tp`),
    pp: intVal(`${prefix}-pp`),
    dp: intVal(`${prefix}-dp`),
    ep: intVal(`${prefix}-ep`),
    cp: intVal(`${prefix}-cp`),
    cp_type: phase === 'train' ? $('#train-cp-type').value : 'ring',
    sp: phase === 'train' ? $('#train-sp').checked : false,
    zero_stage: phase === 'train' ? intVal('train-zero-stage') : 0,
    pp_schedule: phase === 'train' ? $('#train-pp-schedule').value : '1f1b',
    recompute_attention: phase === 'train' ? $('#train-recompute-attn').checked : false,
    full_recomputation: phase === 'train' ? $('#train-full-recompute').checked : false,
    optimizer_offload: phase === 'train' ? $('#train-opt-offload').checked : false,
    activation_offload: phase === 'train' ? $('#train-act-offload').checked : false,
    param_offload: phase === 'train' ? $('#train-param-offload').checked : false,
    grad_offload: phase === 'train' ? $('#train-grad-offload').checked : false,
  };
}

/* ── Build Request Payloads ───────────────────────────── */
function buildPredictRequest() {
  const req = {
    scenario: state.scenario,
    model: {
      name: $('#model-name').value,
      hidden_size: intVal('hidden-size'),
      vocab_size: intVal('vocab-size'),
      num_layers: intVal('num-layers'),
      dtype: $('#dtype').value,
      layer: {
        attention: $('#attention-type').value,
        num_heads: intVal('num-heads'),
        num_kv_heads: intVal('num-kv-heads'),
        head_dim: intVal('head-dim'),
        ffn: $('#ffn-type').value,
        intermediate_size: intVal('intermediate-size'),
        residual: $('#residual-type').value,
        num_experts: intVal('num-experts'),
        top_k: intVal('top-k'),
        num_shared_experts: intVal('shared-experts'),
        expert_intermediate_size: intVal('expert-intermediate-size'),
        shared_intermediate_size: intVal('shared-intermediate-size'),
        kv_compression_dim: intVal('kv-compression-dim'),
        query_compression_dim: intVal('query-compression-dim'),
        rope_dim: intVal('rope-dim'),
        window_size: intVal('window-size'),
        mhc_expansion: intVal('mhc-expansion'),
        compress_ratio: intVal('compress-ratio'),
        compress_c_kv: intVal('compress-c-kv'),
        compress_coeff: parseFloat($('#compress-coeff').value) || 0,
        index_n_heads: intVal('index-n-heads'),
        index_head_dim: intVal('index-head-dim'),
        index_topk: intVal('index-topk'),
        q_lora_rank: intVal('q-lora-rank'),
        o_lora_rank: intVal('o-lora-rank'),
        o_groups: intVal('o-groups'),
      },
    },
    hardware: $('#hw-profile').value,
    total_devices: intVal('total-devices'),
    parallelism: buildParallelismInput('train'),
    gen_parallelism: buildParallelismInput('gen'),
    ref_parallelism: buildParallelismInput('ref'),
    rl: {
      group_size: intVal('group-size'),
      avg_prompt_len: intVal('avg-prompt-len'),
      avg_response_len: intVal('avg-response-len'),
      max_response_len: intVal('max-response-len'),
      std_response_len: intValOrNull('std-response-len'),
      train_micro_batch_size: intVal('train-mbs'),
      gradient_accumulation_steps: intVal('grad-accum'),
      train_batch_size: intVal('train-batch-size'),
      gen_batch_size: intVal('gen-batch-size'),
      reference_model: $('#ref-model').checked,
      ref_offload_cpu: $('#ref-offload').checked,
      use_speculative_decoding: $('#spec-decode').checked,
      mtp_acceptance_len: $('#spec-decode').checked ? intValOrNull('mtp-acceptance-len') : null,
    },
  };

  // Auxiliary
  if ($('#has-mtp-aux') && $('#has-mtp-aux').checked) {
    req.model.auxiliary = { mtp_depth: intVal('mtp-depth') };
  }

  // Layers summary — always send if present. The backend _expand_layers_summary
  // now respects the user's layer.attention override (only FFN structure comes
  // from the summary string).
  const badge = $('#layers-summary-badge');
  if (badge && badge.textContent.trim()) {
    req.model.layers_summary = badge.textContent.trim();
  }

  return req;
}

function getSearchMode() {
  const active = $('#search-mode .seg-btn.active');
  return active ? active.dataset.value : 'pareto';
}

function buildSearchRequest() {
  const base = buildPredictRequest();
  const mode = getSearchMode();

  if (mode === 'pareto') {
    base.search = {
      mode: 'pareto',
      device_counts: $('#device-counts').value.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n)),
      optimization_target: $('#opt-target').value,
    };
  } else {
    base.search = {
      mode: 'sensitivity',
      sweep_param: $('#sweep-param').value,
      sweep_values: $('#sweep-values').value.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n)),
    };
  }
  return base;
}

/* ── API Calls ────────────────────────────────────────── */
async function runAnalysis() {
  const btn = $('#run-btn');
  const label = btn.querySelector('.btn-label');
  const spinner = btn.querySelector('.btn-spinner');

  setLoading(true);
  btn.disabled = true;
  label.textContent = 'Running...';
  spinner.classList.remove('hidden');
  hideError();

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPredictRequest()),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `Server error ${resp.status}`);
    }

    const data = await resp.json();
    state.lastResult = data;
    state.hasRun = true;
    resetModified();

    renderKPIs(data.kpis, data.memory);
    renderTimeline(data.timeline);
    renderMemory(data.memory);
    renderTopology(data.topology);
    hideEmptyState();
  } catch (e) {
    showError(e.message);
    renderErrorKPIs();
  } finally {
    setLoading(false);
    btn.disabled = false;
    label.textContent = 'Run Analysis';
    spinner.classList.add('hidden');
  }
}

async function runSearch() {
  const btn = $('#run-search-btn');
  btn.disabled = true;
  btn.textContent = 'Searching...';
  hideError();

  try {
    const resp = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildSearchRequest()),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `Server error ${resp.status}`);
    }

    const data = await resp.json();
    state.lastSearch = data;
    state.hasRun = true;
    hideEmptyState();

    renderSearchResults(data, getSearchMode());

    // Switch to search results tab
    $$('.chart-tab').forEach((t) => t.classList.remove('active'));
    $$('.chart-panel').forEach((p) => p.classList.remove('active'));
    $('.chart-tab[data-tab="search-results"]').classList.add('active');
    $('#panel-search-results').classList.add('active');
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Search';
  }
}

async function loadHFModel() {
  const btn = $('#hf-load-btn');
  const modelId = $('#hf-model-id').value.trim();
  const status = $('#hf-status');

  if (!modelId) return;

  btn.disabled = true;
  btn.textContent = 'Loading...';
  status.classList.remove('hidden', 'error', 'success');
  status.textContent = 'Fetching config...';

  try {
    const resp = await fetch('/api/hf-import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `Error ${resp.status}`);
    }

    const data = await resp.json();
    applyHFData(data);

    status.classList.add('success');
    status.textContent = `Loaded ${data.name}`;
    markModified('model');
  } catch (e) {
    status.classList.add('error');
    status.textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Load';
  }
}

function applyHFData(data) {
  $('#model-name').value = data.name || '';
  $('#hidden-size').value = data.hidden_size || 0;
  $('#vocab-size').value = data.vocab_size || 0;
  $('#num-layers').value = data.num_layers || 0;
  $('#dtype').value = data.dtype || 'bf16';

  if (data.layer) {
    $('#attention-type').value = data.layer.attention || 'GQA';
    $('#ffn-type').value = data.layer.ffn || 'SwiGLU';
    $('#residual-type').value = data.layer.residual || 'standard';
    $('#num-heads').value = data.layer.num_heads || 32;
    $('#num-kv-heads').value = data.layer.num_kv_heads || 8;
    $('#head-dim').value = data.layer.head_dim || 128;
    $('#intermediate-size').value = data.layer.intermediate_size || 14336;
    $('#num-experts').value = data.layer.num_experts || 1;
    $('#top-k').value = data.layer.top_k || 1;
    $('#shared-experts').value = data.layer.num_shared_experts || 0;
    $('#expert-intermediate-size').value = data.layer.expert_intermediate_size || 0;
    $('#shared-intermediate-size').value = data.layer.shared_intermediate_size || 0;
    $('#kv-compression-dim').value = data.layer.kv_compression_dim || 0;
    $('#query-compression-dim').value = data.layer.query_compression_dim || 0;
    $('#rope-dim').value = data.layer.rope_dim || 0;
    $('#window-size').value = data.layer.window_size || 0;
    $('#mhc-expansion').value = data.layer.mhc_expansion || 4;
    applyDsaFields(data.layer);
  }

  // Auxiliary
  if (data.auxiliary && data.auxiliary.mtp_depth) {
    $('#mtp-aux-fields').classList.remove('hidden');
    $('#has-mtp-aux').checked = true;
    $('#mtp-depth').value = data.auxiliary.mtp_depth;
  }

  // Layers summary
  if (data.layers_summary) {
    $('#field-layers-summary').classList.remove('hidden');
    $('#layers-summary-badge').textContent = data.layers_summary;
  }

  updateConditionalFields();
  updateModelSummary();
}

/* ── Preset Application ──────────────────────────────── */
function applyParallelismToPanel(phase, config) {
  const prefix = phase;
  $(`#${prefix}-tp`).value = config.tp || 1;
  $(`#${prefix}-pp`).value = config.pp || 1;
  $(`#${prefix}-ep`).value = config.ep || 1;
  $(`#${prefix}-cp`).value = config.cp || 1;

  // If preset provides an explicit DP value, set it and mark as explicit
  // so recomputeDPFromPreset won't overwrite it.
  if (config.dp !== undefined && config.dp !== null) {
    $(`#${prefix}-dp`).value = config.dp;
    $(`#${prefix}-dp`).dataset.explicit = 'true';
  }

  if (phase === 'train') {
    $('#train-cp-type').value = config.cp_type || 'ring';
    $('#train-sp').checked = !!config.sp;
    $('#train-zero-stage').value = config.zero_stage || 0;
    $('#train-pp-schedule').value = config.pp_schedule || '1f1b';
    $('#train-recompute-attn').checked = !!config.recompute_attention;
    $('#train-full-recompute').checked = !!config.full_recomputation;
    $('#train-opt-offload').checked = !!config.optimizer_offload;
    $('#train-act-offload').checked = !!config.activation_offload;
    $('#train-param-offload').checked = !!config.param_offload;
    $('#train-grad-offload').checked = !!config.grad_offload;
  }
}

function applyPreset(preset) {
  // Apply model params
  if (preset.model) {
    const m = preset.model;
    $('#model-name').value = m.name || '';
    $('#hidden-size').value = m.hidden_size || 4096;
    $('#vocab-size').value = m.vocab_size || 32000;
    $('#num-layers').value = m.num_layers || 32;
    $('#dtype').value = m.dtype || 'bf16';

    if (m.layer) {
      $('#attention-type').value = m.layer.attention || 'GQA';
      $('#ffn-type').value = m.layer.ffn || 'SwiGLU';
      $('#residual-type').value = m.layer.residual || 'standard';
      $('#num-heads').value = m.layer.num_heads || 32;
      $('#num-kv-heads').value = m.layer.num_kv_heads || 8;
      $('#head-dim').value = m.layer.head_dim || 128;
      $('#intermediate-size').value = m.layer.intermediate_size || 14336;
      $('#num-experts').value = m.layer.num_experts || 1;
      $('#top-k').value = m.layer.top_k || 1;
      $('#shared-experts').value = m.layer.num_shared_experts || 0;
      $('#expert-intermediate-size').value = m.layer.expert_intermediate_size || 0;
      $('#shared-intermediate-size').value = m.layer.shared_intermediate_size || 0;
      $('#kv-compression-dim').value = m.layer.kv_compression_dim || 0;
      $('#query-compression-dim').value = m.layer.query_compression_dim || 0;
      $('#rope-dim').value = m.layer.rope_dim || 0;
      $('#window-size').value = m.layer.window_size || 0;
      $('#mhc-expansion').value = m.layer.mhc_expansion || 4;
      applyDsaFields(m.layer);
    }

    // Auxiliary
    if (m.auxiliary && m.auxiliary.mtp_depth) {
      $('#mtp-aux-fields').classList.remove('hidden');
      $('#has-mtp-aux').checked = true;
      $('#mtp-depth').value = m.auxiliary.mtp_depth;
    } else {
      $('#has-mtp-aux').checked = false;
    }

    // Layers summary
    if (m.layers_summary) {
      $('#field-layers-summary').classList.remove('hidden');
      $('#layers-summary-badge').textContent = m.layers_summary;
    } else {
      $('#field-layers-summary').classList.add('hidden');
    }
  }

  // Apply hardware
  if (preset.hardware) {
    $('#hw-profile').value = preset.hardware;
    const prof = state.hardware[preset.hardware];
    if (prof) {
      $('#devices-per-node').value = prof.devices_per_node;
    }
  }
  if (preset.total_devices) {
    $('#total-devices').value = preset.total_devices;
  }

  // Apply train parallelism
  if (preset.parallelism) {
    applyParallelismToPanel('train', preset.parallelism);
  }

  // Apply gen parallelism (explicit or derived from train)
  if (preset.gen_parallelism) {
    applyParallelismToPanel('gen', preset.gen_parallelism);
  } else if (preset.parallelism) {
    // Default gen parallelism = full train parallelism (tp/pp/ep/cp), matching
    // the demo; dp is recomputed from total devices.
    const pc = preset.parallelism;
    applyParallelismToPanel('gen', { tp: pc.tp || 1, pp: pc.pp || 1, ep: pc.ep || 1, cp: pc.cp || 1 });
  }

  // Apply ref parallelism (explicit or derived from train TP)
  if (preset.ref_parallelism) {
    applyParallelismToPanel('ref', preset.ref_parallelism);
  } else if (preset.parallelism) {
    // Default ref parallelism = full train parallelism (matches the demo).
    const pc = preset.parallelism;
    applyParallelismToPanel('ref', { tp: pc.tp || 1, pp: pc.pp || 1, ep: pc.ep || 1, cp: pc.cp || 1 });
  }

  // Apply RL config
  if (preset.rl) {
    const r = preset.rl;
    $('#group-size').value = r.group_size || 8;
    $('#avg-prompt-len').value = r.avg_prompt_len || 512;
    $('#avg-response-len').value = r.avg_response_len || 2048;
    $('#max-response-len').value = r.max_response_len || 4096;
    $('#std-response-len').value = r.std_response_len || '';
    $('#train-mbs').value = r.train_micro_batch_size || 4;
    $('#grad-accum').value = r.gradient_accumulation_steps || 1;
    $('#train-batch-size').value = r.train_batch_size || 36;
    $('#gen-batch-size').value = r.gen_batch_size || 64;
    $('#ref-model').checked = r.reference_model !== false;
    $('#ref-offload').checked = !!r.ref_offload_cpu;
    $('#spec-decode').checked = !!r.use_speculative_decoding;
    if (r.mtp_acceptance_len) {
      $('#mtp-acceptance-len').value = r.mtp_acceptance_len;
    }
    $('#mtp-fields').classList.toggle('hidden', !r.use_speculative_decoding);
  }

  // Recompute derived fields
  recomputeDPFromPreset();
  updateConditionalFields();
  updateModelSummary();
  updateHardwareSummary();
  updateRLSummary();
  resetModified();
  refreshJsonEditor();
}

function recomputeDPFromPreset() {
  const total = intVal('total-devices');
  ['train', 'gen', 'ref'].forEach((phase) => {
    const dpEl = $(`#${phase}-dp`);
    // If the preset already set an explicit DP value (non-default), keep it
    if (dpEl && dpEl.dataset.explicit === 'true') {
      delete dpEl.dataset.explicit;
      return;
    }
    const tp = intVal(`${phase}-tp`);
    const pp = intVal(`${phase}-pp`);
    const cp = intVal(`${phase}-cp`);
    const divisor = tp * pp * cp;  // EP shares DP devices, not counted here
    const dp = divisor > 0 ? Math.max(1, Math.floor(total / divisor)) : total;
    dpEl.value = dp;
  });

  const perNode = intVal('devices-per-node');
  const nodes = perNode > 0 ? Math.ceil(total / perNode) : 1;
  $('#num-nodes').value = nodes;
}

function applyConfigToForm(config) {
  // Wrapper: apply a full predict-request-shaped config object to form fields
  if (config.model) {
    // Reshape to preset format
    const preset = {
      model: config.model,
      hardware: config.hardware,
      total_devices: config.total_devices,
      parallelism: config.parallelism,
      gen_parallelism: config.gen_parallelism,
      ref_parallelism: config.ref_parallelism,
      rl: config.rl,
    };
    applyPreset(preset);
  }
}

/* ── JSON Config Editor ──────────────────────────────── */
function initJsonEditor() {
  const refreshBtn = $('#json-refresh-btn');
  const applyBtn = $('#json-apply-btn');

  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      refreshJsonEditor();
    });
  }

  if (applyBtn) {
    applyBtn.addEventListener('click', async () => {
      const textarea = $('#json-config-editor');
      const status = $('#json-editor-status');
      const raw = textarea.value.trim();

      if (!raw) return;

      try {
        const config = JSON.parse(raw);
        textarea.classList.remove('json-error');
        textarea.classList.add('json-success');

        applyConfigToForm(config);

        status.classList.remove('hidden', 'error');
        status.classList.add('success');
        status.textContent = 'JSON applied to form fields.';

        // Auto-run after applying JSON
        await runAnalysis();

        // Clear success state after a moment
        setTimeout(() => {
          textarea.classList.remove('json-success');
          status.classList.add('hidden');
        }, 2000);
      } catch (e) {
        textarea.classList.remove('json-success');
        textarea.classList.add('json-error');
        status.classList.remove('hidden', 'success');
        status.classList.add('error');
        status.textContent = `Invalid JSON: ${e.message}`;
      }
    });
  }
}

function refreshJsonEditor() {
  const textarea = $('#json-config-editor');
  if (textarea) {
    const config = buildPredictRequest();
    textarea.value = JSON.stringify(config, null, 2);
    textarea.classList.remove('json-error', 'json-success');
    const status = $('#json-editor-status');
    if (status) status.classList.add('hidden');
  }
}

/* ── Loading / Error States ───────────────────────────── */
function setLoading(loading) {
  const kpis = $$('.kpi-card');
  const chartLoading = $('#chart-loading');

  kpis.forEach((kpi) => {
    if (loading) {
      kpi.classList.add('loading');
      kpi.classList.remove('fade-in');
    } else {
      kpi.classList.remove('loading');
    }
  });

  chartLoading.classList.toggle('hidden', !loading);
}

function showError(msg) {
  const banner = $('#error-banner');
  const text = $('#error-text');
  text.textContent = msg;
  banner.classList.remove('hidden');
}

function hideError() {
  $('#error-banner').classList.add('hidden');
}

function hideEmptyState() {
  const el = $('#empty-state');
  if (el) el.classList.add('hidden');
}

/* ── KPI Rendering ────────────────────────────────────── */
function renderKPIs(kpis, memory) {
  if (state.scenario === 'inference') return renderKPIsInference(kpis, memory);
  if (state.scenario === 'pretraining') return renderKPIsPretraining(kpis, memory);

  const feasible = kpis.feasible && memory.train_feasible && memory.gen_feasible && memory.ref_feasible;

  // Step Time
  const reshardS = (kpis.reshard_gen_ref_seconds || 0) + (kpis.reshard_ref_train_seconds || 0);
  const stepDetail = feasible ? 'Feasible' : 'Infeasible';
  setKPI('kpi-epoch', {
    value: `${kpis.step_time_seconds.toFixed(1)}s`,
    detail: stepDetail + (reshardS > 0 ? ` | reshard: ${reshardS.toFixed(1)}s` : ''),
    status: feasible ? 'success' : 'error',
  });

  // Gen TPS
  setKPI('kpi-gen', {
    value: formatNumber(kpis.gen_tps_target),
    detail: `gen: ${kpis.gen_time_seconds.toFixed(1)}s`,
    status: memory.gen_feasible ? 'success' : 'error',
  });

  // Train TPS
  setKPI('kpi-train', {
    value: formatNumber(kpis.train_tps_target),
    detail: `train: ${kpis.train_time_seconds.toFixed(1)}s`,
    status: memory.train_feasible ? 'success' : 'error',
  });

  // Ref TPS
  setKPI('kpi-ref', {
    value: formatNumber(kpis.ref_tps_target),
    detail: `ref: ${kpis.ref_time_seconds.toFixed(1)}s`,
    status: memory.ref_feasible ? 'success' : 'error',
  });
}

function renderKPIsInference(kpis, memory) {
  const ok = memory.gen_feasible;
  setKPI('kpi-epoch', {
    value: `${kpis.gen_time_seconds.toFixed(1)}s`,
    detail: `prefill ${kpis.prefill_seconds.toFixed(2)}s + decode ${kpis.decode_seconds.toFixed(1)}s`,
    status: ok ? 'success' : 'error',
  });
  setKPI('kpi-gen', {
    value: formatNumber(kpis.gen_tps_target),
    detail: 'tokens / sec',
    status: ok ? 'success' : 'error',
  });
  setKPI('kpi-train', {
    value: kpis.gen_samples_per_sec.toFixed(2),
    detail: `${kpis.decode_ms_per_token.toFixed(1)} ms/token`,
    status: ok ? 'success' : 'error',
  });
  setKPI('kpi-ref', {
    value: `${kpis.kv_cache_gb.toFixed(1)}GB`,
    detail: ok ? 'Fits HBM' : 'OOM',
    status: ok ? 'success' : 'error',
  });
}

function renderKPIsPretraining(kpis, memory) {
  const ok = memory.train_feasible;
  setKPI('kpi-epoch', {
    value: `${kpis.step_time_seconds.toFixed(1)}s`,
    detail: ok ? 'Feasible' : 'Infeasible',
    status: ok ? 'success' : 'error',
  });
  setKPI('kpi-gen', {
    value: formatNumber(kpis.train_tps_target),
    detail: 'tokens / sec',
    status: ok ? 'success' : 'error',
  });
  setKPI('kpi-train', {
    value: kpis.train_samples_per_sec.toFixed(2),
    detail: 'samples / sec',
    status: ok ? 'success' : 'error',
  });
  setKPI('kpi-ref', {
    value: `${kpis.total_train_gb.toFixed(1)}GB`,
    detail: `of ${memory.usable_hbm_gb}GB`,
    status: ok ? 'success' : 'error',
  });
}

function renderErrorKPIs() {
  ['kpi-epoch', 'kpi-gen', 'kpi-train', 'kpi-ref'].forEach((id) => {
    setKPI(id, { value: '\u2014', detail: 'error', status: 'error' });
  });
}

function setKPI(id, { value, detail, status }) {
  const card = $(`#${id}`);
  if (!card) return;
  card.querySelector('.kpi-value').textContent = value;
  card.querySelector('.kpi-detail').textContent = detail;
  card.dataset.status = status;
  card.classList.add('fade-in');
}

/* ── Chart: Timeline ──────────────────────────────────── */
function renderTimeline(timeline) {
  if (state.scenario === 'inference') return renderTimelineInference(timeline);
  if (state.scenario === 'pretraining') return renderTimelinePretraining(timeline);

  const genS = timeline.gen_seconds;
  const trainS = timeline.train_seconds;
  const refS = timeline.ref_seconds || 0;
  const reshardGR = timeline.reshard_gen_ref_seconds || 0;
  const reshardRT = timeline.reshard_ref_train_seconds || 0;
  const stepS = timeline.step_time_seconds || 0;

  // Phases run serially on the shared device pool:
  // gen → reshard → ref → reshard → train. Stacked bar shows cumulative time.
  const traces = [];
  const seg = (x, name, color, light) => {
    if (!(x > 0)) return;
    traces.push({
      y: ['Step'], x: [x], type: 'bar', orientation: 'h', name,
      marker: { color },
      text: [`${x.toFixed(1)}s`], textposition: 'inside',
      textfont: { color: light ? '#1a1a1a' : '#fff', size: light ? 10 : 12 },
    });
  };

  seg(genS, 'Generation', '#7c3aed');
  seg(reshardGR, 'Reshard (gen→ref)', '#fbbf24', true);
  seg(refS, 'Reference', '#06b6d4');
  seg(reshardRT, 'Reshard (ref→train)', '#fbbf24', true);
  seg(trainS, 'Training', '#ea580c');

  const layout = {
    barmode: 'stack',
    ...chartLayout('Timeline (seconds)'),
    xaxis: { title: 'Seconds', gridcolor: '#f0eeeb', zeroline: false },
    yaxis: { automargin: true },
    height: 160,
    margin: { l: 80, r: 30, t: 30, b: 40 },
    annotations: stepS > 0 ? [{
      x: stepS,
      y: 0,
      text: `${stepS.toFixed(1)}s`,
      showarrow: false,
      xanchor: 'left',
      font: { size: 12, color: '#1a1a1a' },
    }] : [],
  };

  Plotly.newPlot('chart-timeline', traces, layout, plotlyConfig());
}

function renderTimelineInference(timeline) {
  const prefill = timeline.prefill_seconds || 0;
  const decode = timeline.decode_seconds || 0;
  const total = timeline.gen_seconds || prefill + decode;

  const traces = [
    {
      y: ['Generation'], x: [prefill], type: 'bar', orientation: 'h',
      name: 'Prefill', marker: { color: '#7c3aed' },
      text: [`${prefill.toFixed(2)}s`], textposition: 'inside',
      textfont: { color: '#fff', size: 12 },
    },
    {
      y: ['Generation'], x: [decode], type: 'bar', orientation: 'h',
      name: 'Decode', marker: { color: '#06b6d4' },
      text: [`${decode.toFixed(1)}s`], textposition: 'inside',
      textfont: { color: '#fff', size: 12 },
    },
  ];

  const layout = {
    barmode: 'stack',
    ...chartLayout('Generation Timeline (seconds)'),
    xaxis: { title: 'Seconds', gridcolor: '#f0eeeb', zeroline: false },
    yaxis: { automargin: true },
    height: 160,
    margin: { l: 90, r: 30, t: 30, b: 40 },
    annotations: total > 0 ? [{
      x: total, y: 0, text: `${total.toFixed(1)}s`, showarrow: false,
      xanchor: 'left', font: { size: 12, color: '#1a1a1a' },
    }] : [],
  };

  Plotly.newPlot('chart-timeline', traces, layout, plotlyConfig());
}

function renderTimelinePretraining(timeline) {
  const bd = timeline.breakdown || {};
  const parts = [
    { name: 'Policy Update', value: bd.policy_update || 0, color: '#ea580c' },
    { name: 'Recompute', value: bd.recompute || 0, color: '#f59e0b' },
    { name: 'PP Bubble', value: bd.pp_bubble || 0, color: '#fbbf24' },
    { name: 'Optim Offload', value: bd.optim_offload || 0, color: '#c4b5fd' },
  ].filter((p) => p.value > 0);

  const total = bd.total || parts.reduce((a, p) => a + p.value, 0);

  const traces = parts.map((p) => ({
    y: ['Step'], x: [p.value], type: 'bar', orientation: 'h',
    name: p.name, marker: { color: p.color },
    text: [`${p.value.toFixed(2)}s`], textposition: 'inside',
    textfont: { color: '#fff', size: 11 },
  }));

  const layout = {
    barmode: 'stack',
    ...chartLayout('Training Step Breakdown (seconds)'),
    xaxis: { title: 'Seconds', gridcolor: '#f0eeeb', zeroline: false },
    yaxis: { automargin: true },
    height: 160,
    margin: { l: 80, r: 30, t: 30, b: 40 },
    annotations: total > 0 ? [{
      x: total, y: 0, text: `${total.toFixed(1)}s`, showarrow: false,
      xanchor: 'left', font: { size: 12, color: '#1a1a1a' },
    }] : [],
  };

  Plotly.newPlot('chart-timeline', traces, layout, plotlyConfig());
}

/* ── Chart: Memory ────────────────────────────────────── */
function renderMemory(memory) {
  if (state.scenario === 'inference') return renderMemoryInference(memory);
  if (state.scenario === 'pretraining') return renderMemoryPretraining(memory);

  const categories = ['Training', 'Generation', 'Reference'];

  // Each phase has its own per-device weight (different parallelism configs).
  // gen_weight_gb: weight per device under gen parallelism (e.g. TP=4, EP=8)
  // ref_weight_gb: weight per device under ref parallelism (e.g. TP=1)
  const genWeight = memory.gen_weight_gb || 0;
  const refWeight = memory.ref_weight_gb || 0;

  // When ref_offload_cpu is True, ref weights are not resident on GPU during
  // the ref phase (streamed layer-by-layer from CPU). In that case,
  // total_ref = ref_activation_peak + ve_weight_ref, and we should NOT
  // show the full ref_weight in the stacked bar (it would far exceed total_ref).
  // Detect this by checking if ref_weight exceeds total_ref (indicates offload).
  const refOffloaded = refWeight > 0 && refWeight > (memory.total_ref_gb + 0.1);
  const refWeightBar = refOffloaded ? 0 : refWeight;

  const trainTraces = [
    { y: categories, x: [memory.weight_gb, genWeight, refWeightBar], name: 'Weights', type: 'bar', orientation: 'h', marker: { color: '#7c3aed' } },
    { y: categories, x: [memory.grad_gb || 0, 0, 0], name: 'Gradients', type: 'bar', orientation: 'h', marker: { color: '#a78bfa' } },
    { y: categories, x: [memory.optimizer_gb, 0, 0], name: 'Optimizer', type: 'bar', orientation: 'h', marker: { color: '#c4b5fd' } },
    { y: categories, x: [memory.activation_peak_gb, 0, 0], name: 'Activations', type: 'bar', orientation: 'h', marker: { color: '#f59e0b' } },
    { y: categories, x: [memory.ref_model_gb, 0, 0], name: 'Ref (resident)', type: 'bar', orientation: 'h', marker: { color: '#06b6d4' } },
    { y: categories, x: [0, memory.kv_cache_gb, 0], name: 'KV Cache', type: 'bar', orientation: 'h', marker: { color: '#16a34a' } },
    { y: categories, x: [0, 0, memory.ref_activation_peak_gb], name: 'Ref Activations', type: 'bar', orientation: 'h', marker: { color: '#67e8f9' } },
  ];
  if (memory.reward_model_gb > 0) {
    trainTraces.push({ y: categories, x: [memory.reward_model_gb, 0, 0], name: 'Reward', type: 'bar', orientation: 'h', marker: { color: '#f97316' } });
  }

  const shapes = [
    {
      type: 'line',
      x0: memory.usable_hbm_gb,
      x1: memory.usable_hbm_gb,
      y0: -0.5,
      y1: 2.5,
      line: { color: '#dc2626', width: 2, dash: 'dash' },
    },
  ];

  const annotations = [
    {
      x: memory.usable_hbm_gb,
      y: 1.0,
      text: `HBM Limit: ${memory.usable_hbm_gb}GB`,
      showarrow: false,
      font: { size: 11, color: '#dc2626' },
      yref: 'paper',
      yanchor: 'bottom',
    },
  ];

  // Feasibility annotations
  const feasAnnotations = [];
  const phaseInfo = [
    { name: 'Training', total: memory.total_train_gb, feasible: memory.train_feasible, idx: 0 },
    { name: 'Generation', total: memory.total_gen_gb, feasible: memory.gen_feasible, idx: 1 },
    { name: 'Reference', total: memory.total_ref_gb, feasible: memory.ref_feasible, idx: 2 },
  ];
  phaseInfo.forEach((p) => {
    feasAnnotations.push({
      x: p.total,
      y: p.name,
      text: `${p.total.toFixed(1)}GB ${p.feasible ? 'OK' : 'OOM'}`,
      showarrow: false,
      xanchor: 'left',
      font: { size: 10, color: p.feasible ? '#16a34a' : '#dc2626' },
    });
  });

  const layout = {
    barmode: 'stack',
    ...chartLayout('Memory Breakdown (GB)'),
    xaxis: {
      title: 'GB',
      gridcolor: '#f0eeeb',
      zeroline: false,
    },
    yaxis: {
      automargin: true,
    },
    shapes,
    annotations: [...annotations, ...feasAnnotations],
    height: 280,
    margin: { l: 120, r: 60, t: 30, b: 40 },
  };

  Plotly.newPlot('chart-memory', trainTraces, layout, plotlyConfig());
}

function renderMemoryBars(category, components, totalGb, usableGb, feasible) {
  // components: [{name, value, color}]
  const traces = components
    .filter((c) => c.value > 0)
    .map((c) => ({
      y: [category], x: [c.value], name: c.name, type: 'bar', orientation: 'h',
      marker: { color: c.color },
    }));

  const shapes = [{
    type: 'line', x0: usableGb, x1: usableGb, y0: -0.5, y1: 0.5,
    line: { color: '#dc2626', width: 2, dash: 'dash' },
  }];

  const annotations = [
    {
      x: usableGb, y: 0.5, text: `HBM Limit: ${usableGb}GB`, showarrow: false,
      font: { size: 11, color: '#dc2626' }, yref: 'paper', yanchor: 'bottom',
    },
    {
      x: totalGb, y: category, text: `${totalGb.toFixed(1)}GB ${feasible ? 'OK' : 'OOM'}`,
      showarrow: false, xanchor: 'left',
      font: { size: 11, color: feasible ? '#16a34a' : '#dc2626' },
    },
  ];

  const layout = {
    barmode: 'stack',
    ...chartLayout('Memory Breakdown (GB)'),
    xaxis: { title: 'GB', gridcolor: '#f0eeeb', zeroline: false },
    yaxis: { automargin: true },
    shapes,
    annotations,
    height: 200,
    margin: { l: 100, r: 60, t: 30, b: 40 },
  };

  Plotly.newPlot('chart-memory', traces, layout, plotlyConfig());
}

function renderMemoryInference(memory) {
  renderMemoryBars(
    'Generation',
    [
      { name: 'Weights', value: memory.gen_weight_gb, color: '#7c3aed' },
      { name: 'KV Cache', value: memory.kv_cache_gb, color: '#16a34a' },
    ],
    memory.total_gen_gb, memory.usable_hbm_gb, memory.gen_feasible
  );
}

function renderMemoryPretraining(memory) {
  renderMemoryBars(
    'Training',
    [
      { name: 'Weights', value: memory.weight_gb, color: '#7c3aed' },
      { name: 'Gradients', value: memory.grad_gb || 0, color: '#a78bfa' },
      { name: 'Optimizer', value: memory.optimizer_gb, color: '#c4b5fd' },
      { name: 'Activations', value: memory.activation_peak_gb, color: '#f59e0b' },
    ],
    memory.total_train_gb, memory.usable_hbm_gb, memory.train_feasible
  );
}

/* ── Chart: Topology ──────────────────────────────────── */
function renderTopology(topology) {
  const ranks = topology.ranks;
  if (!ranks || ranks.length === 0) return;

  const x = ranks.map((r) => r.tp_rank);
  const y = ranks.map((r) => r.pp_stage);
  const text = ranks.map(
    (r) =>
      `Rank ${r.global_rank}<br>TP=${r.tp_rank} PP=${r.pp_stage}<br>DP=${r.dp_rank} EP=${r.ep_rank}<br>Node ${r.node}, GPU ${r.local_gpu}<br>Layers ${r.layer_start}-${r.layer_end}`
  );

  // Color by PP stage
  const colors = ranks.map((r) => r.pp_stage);
  // Opacity: full for dp_rank 0, lower for replicas
  const opacities = ranks.map((r) => (r.dp_rank === 0 ? 1.0 : 0.35));

  // Border by EP group
  const borderColors = ranks.map((r) => {
    const epColors = ['#1a1a1a', '#2563eb', '#16a34a', '#ea580c', '#7c3aed', '#06b6d4'];
    return epColors[r.ep_rank % epColors.length];
  });

  const trace = {
    x,
    y,
    text,
    mode: 'markers',
    type: 'scatter',
    hoverinfo: 'text',
    marker: {
      size: 28,
      color: colors,
      colorscale: 'Viridis',
      opacity: opacities,
      line: {
        width: 2,
        color: borderColors,
      },
    },
  };

  const layout = {
    ...chartLayout('Device Topology'),
    xaxis: {
      title: 'TP Rank',
      dtick: 1,
      gridcolor: '#f0eeeb',
      zeroline: false,
    },
    yaxis: {
      title: 'PP Stage',
      dtick: 1,
      autorange: 'reversed',
      gridcolor: '#f0eeeb',
      zeroline: false,
    },
    height: 400,
    margin: { l: 60, r: 30, t: 30, b: 50 },
  };

  Plotly.newPlot('chart-topology', [trace], layout, plotlyConfig());
}

/* ── Chart: Search Results ────────────────────────────── */
function renderSearchResults(data, mode) {
  const results = data.results || [];
  if (results.length === 0) return;

  if (mode === 'pareto') {
    renderParetoChart(results);
  } else {
    renderSensitivityChart(results);
  }

  renderSearchTable(results, mode);
}

function renderParetoChart(results) {
  const feasible = results.filter((r) => !r.is_oom);
  const pareto = results.filter((r) => r.is_pareto);
  const oom = results.filter((r) => r.is_oom);

  function hoverText(r) {
    const p = r.parallelism || {};
    const cp = p.cp || 1;
    const sp = p.sp ? ', SP' : '';
    const cpStr = cp > 1 ? `, CP=${cp}${sp}` : '';
    return `${r.devices} devices<br>TP=${p.tp} PP=${p.pp} EP=${p.ep||1} DP=${p.dp}${cpStr}<br>${(r.step_time_seconds||0).toFixed(1)}s`;
  }

  const traces = [];

  if (feasible.length > 0) {
    traces.push({
      x: feasible.map((r) => r.devices),
      y: feasible.map((r) => r.step_time_seconds),
      mode: 'markers',
      type: 'scatter',
      name: 'Feasible',
      marker: { color: '#6b7280', size: 8, opacity: 0.5 },
      text: feasible.map(hoverText),
      hoverinfo: 'text',
    });
  }

  if (pareto.length > 0) {
    // Sort Pareto points by devices for a clean connecting line
    const sorted = [...pareto].sort((a, b) => a.devices - b.devices || a.step_time_seconds - b.step_time_seconds);
    traces.push({
      x: sorted.map((r) => r.devices),
      y: sorted.map((r) => r.step_time_seconds),
      mode: 'markers+lines',
      type: 'scatter',
      name: 'Pareto Optimal',
      marker: { color: '#CF0A2C', size: 12, symbol: 'diamond' },
      line: { color: '#CF0A2C', width: 1, dash: 'dot' },
      text: sorted.map(hoverText),
      hoverinfo: 'text',
    });
  }

  if (oom.length > 0) {
    traces.push({
      x: oom.map((r) => r.devices),
      y: oom.map((r) => r.step_time_seconds),
      mode: 'markers',
      type: 'scatter',
      name: 'OOM',
      marker: { color: '#dc2626', size: 8, symbol: 'x', opacity: 0.5 },
      text: oom.map(hoverText),
      hoverinfo: 'text',
    });
  }

  const layout = {
    ...chartLayout('Pareto Search — Devices vs Step Time'),
    xaxis: { title: 'Devices', gridcolor: '#f0eeeb', zeroline: false },
    yaxis: { title: 'Step Time (seconds)', gridcolor: '#f0eeeb', zeroline: false },
    height: 350,
    margin: { l: 60, r: 30, t: 30, b: 50 },
  };

  Plotly.newPlot('chart-search', traces, layout, plotlyConfig());
}

function renderSensitivityChart(results) {
  const xVals = results.map((r) => String(r.sweep_value ?? ''));
  const yVals = results.map((r) => r.step_time_seconds ?? 0);
  const colors = results.map((r) => (r.is_oom ? '#dc2626' : r.feasible ? '#16a34a' : '#f59e0b'));

  const trace = {
    x: xVals,
    y: yVals,
    type: 'bar',
    marker: { color: colors },
    text: yVals.map((v) => `${v.toFixed(1)}s`),
    textposition: 'outside',
    textfont: { size: 11 },
  };

  const layout = {
    ...chartLayout('Sensitivity Analysis'),
    xaxis: {
      title: $('#sweep-param').value,
      gridcolor: '#f0eeeb',
      zeroline: false,
      type: 'category',
    },
    yaxis: {
      title: 'Step Time (seconds)',
      gridcolor: '#f0eeeb',
      zeroline: false,
    },
    height: 350,
    margin: { l: 60, r: 30, t: 30, b: 50 },
  };

  Plotly.newPlot('chart-search', [trace], layout, plotlyConfig());
}

function renderSearchTable(results, mode) {
  const wrap = $('#search-table-wrap');
  const table = $('#search-table');
  const thead = table.querySelector('thead');
  const tbody = table.querySelector('tbody');

  wrap.classList.remove('hidden');

  if (mode === 'pareto') {
    thead.innerHTML = `<tr>
      <th>Devices</th><th>TP</th><th>PP</th><th>DP</th><th>EP</th><th>CP</th>
      <th>Step (s)</th><th>Gen TPS</th><th>Train TPS</th><th>Ref TPS</th><th>Status</th>
    </tr>`;
    tbody.innerHTML = results
      .map((r) => {
        const cls = r.is_oom ? 'oom' : r.is_pareto ? 'pareto' : '';
        const statusText = r.is_oom ? 'OOM' : r.feasible ? (r.is_pareto ? 'Pareto' : 'OK') : 'Infeasible';
        const p = r.parallelism || {};
        const cp = p.cp || 1;
        return `<tr class="${cls}">
          <td>${r.devices ?? ''}</td><td>${p.tp ?? ''}</td><td>${p.pp ?? ''}</td>
          <td>${p.dp ?? ''}</td><td>${p.ep ?? ''}</td><td>${cp}</td>
          <td>${(r.step_time_seconds ?? 0).toFixed(1)}</td><td>${formatNumber(r.gen_tps)}</td>
          <td>${formatNumber(r.train_tps)}</td><td>${formatNumber(r.ref_tps)}</td><td>${statusText}</td>
        </tr>`;
      })
      .join('');
  } else {
    thead.innerHTML = `<tr>
      <th>Value</th><th>Step (s)</th><th>Gen TPS</th><th>Train TPS</th><th>Ref TPS</th><th>Status</th>
    </tr>`;
    tbody.innerHTML = results
      .map((r) => {
        const cls = r.is_oom ? 'oom' : '';
        const statusText = r.is_oom ? 'OOM' : r.feasible ? 'OK' : 'Infeasible';
        return `<tr class="${cls}">
          <td>${r.sweep_value ?? ''}</td>
          <td>${(r.step_time_seconds ?? 0).toFixed(1)}</td><td>${formatNumber(r.gen_tps)}</td>
          <td>${formatNumber(r.train_tps)}</td><td>${formatNumber(r.ref_tps)}</td><td>${statusText}</td>
        </tr>`;
      })
      .join('');
  }
}

/* ── Plotly Helpers ────────────────────────────────────── */
function chartLayout(title) {
  return {
    title: { text: '', font: { size: 14 } },
    paper_bgcolor: '#FAFAF8',
    plot_bgcolor: '#FAFAF8',
    font: {
      family: 'DM Sans, system-ui, sans-serif',
      size: 12,
      color: '#1a1a1a',
    },
    showlegend: true,
    legend: {
      orientation: 'h',
      y: -0.25,
      x: 0,
      font: { size: 11 },
    },
  };
}

function plotlyConfig() {
  return {
    responsive: true,
    displayModeBar: false,
  };
}

/* ── Utility ──────────────────────────────────────────── */
function intVal(id) {
  const el = $(`#${id}`);
  if (!el) return 0;
  return parseInt(el.value, 10) || 0;
}

function intValOrNull(id) {
  const el = $(`#${id}`);
  if (!el) return null;
  const v = el.value;
  if (v === '' || v === null || v === undefined) return null;
  const n = parseInt(v, 10);
  return isNaN(n) ? null : n;
}

function formatNumber(n) {
  if (n === null || n === undefined || isNaN(n)) return '\u2014';
  return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
}

function capitalize(s) {
  if (!s) return '';
  return s.charAt(0).toUpperCase() + s.slice(1);
}