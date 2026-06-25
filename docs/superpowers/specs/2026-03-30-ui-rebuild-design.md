# llm-perf UI Rebuild — Design Spec

**Date:** 2026-03-30
**Status:** Approved
**Replaces:** Gradio-based GUI (src/llm_perf/ui/*.py)

## Overview

Rebuild the llm-perf GUI from scratch using FastAPI + vanilla HTML/CSS/JS + Plotly.js. The current Gradio UI scored 16/20 on the Impeccable audit but still looks generic due to Gradio's rendering constraints. The new UI will have full design control, following Impeccable `/frontend-design` guidelines.

## Architecture

```
Browser (SPA)                    FastAPI Server
┌─────────────────────┐          ┌──────────────────┐
│  index.html          │  JSON   │  /api/predict     │
│  styles.css          │◄──────►│  /api/search      │
│  app.js              │  fetch  │  /api/models      │
│  Plotly.js (CDN)     │         │  /api/hardware    │
└─────────────────────┘          └──────────────────┘
   Static files served                Uses existing
   by FastAPI                         llm_perf.model,
                                      llm_perf.search,
                                      llm_perf.config
```

- **FastAPI** serves static files + 4 REST endpoints
- **Single HTML page** with vanilla JS (no React/Vue/build step)
- **Plotly.js via CDN** for charts
- **Zero npm/node dependencies** — stays a pure Python project

## Layout

Side-by-side: config panel (left, 380px fixed) | results panel (right, flexible).

```
┌──────────────────────────────────────────────────┐
│  llm-perf    LLM Performance Modeling      │
├──────────┬───────────────────────────────────────┤
│ CONFIG   │  RESULTS                               │
│ (380px)  │                                        │
│          │  ┌────┐ ┌────┐ ┌────┐ ┌────┐          │
│ ▸ Model  │  │Epch│ │Gen │ │Trn │ │Btl │  KPIs   │
│          │  │1.38│ │38.7│ │41.6│ │Trn │          │
│ ▾ Hardw  │  └────┘ └────┘ └────┘ └────┘          │
│   TP: 1  │                                        │
│   PP: 1  │  [Timeline] [Memory] [Topology]        │
│   DP: 8  │  ┌──────────────────────────────┐      │
│          │  │       Plotly Chart            │      │
│ ▸ RL     │  │                              │      │
│          │  └──────────────────────────────┘      │
│ ▸ Search │                                        │
│          │                                        │
│ [Run ▶]  │                                        │
└──────────┴───────────────────────────────────────┘
```

### Responsive

- **≥1024px**: Side-by-side layout
- **<1024px**: Config panel collapses to a slide-out drawer (hamburger toggle). Results take full width.
- **<480px**: KPI cards stack 2×2 instead of 4×1

## Config Panel (Left)

### Interaction Model

Collapsible accordion. 4 sections: **Model**, **Hardware**, **RL Training**, **Search**.

- Only one section open at a time (clicking another closes the current)
- Collapsed sections show a **one-line summary** of key values
- Status indicator dot: green = configured with defaults, amber = user-modified since last run
- **"Run Analysis" button** sticky at bottom, always visible

### Section: Model

**Collapsed summary:** `Llama-3.1-8B · 32L · GQA · bf16`

**Expanded fields:**
- Source selector: Template | HuggingFace | Custom (radio or segmented control)
- Template dropdown (when Template selected): Llama-3.1-8B, Qwen2.5-72B, Mistral-7B, Qwen3-235B-MoE, DeepSeekV3-671B
- HuggingFace Model ID text input + "Load" button (when HuggingFace selected)
- Core fields (always shown): Model Name, Hidden Size, Vocab Size, Num Layers, Dtype
- Layer config: Attention type (MHA/GQA/MLA/SWA/Mamba), FFN type (SwiGLU/MoE), Residual (standard/mHC)
- Attention params: Num Heads, Num KV Heads, Head Dim
- Intermediate Size
- **Conditional fields** (shown/hidden based on selections):
  - MLA: KV Compression Dim, Query Compression Dim, RoPE Dim
  - SWA: Window Size
  - MoE: Num Experts, Top-K, Shared Experts, Expert Intermediate Size, Shared Intermediate Size
  - mHC: Expansion factor

### Section: Hardware

**Collapsed summary:** `Ascend 910C · 8 devices · 1 node`

**Expanded fields:**
- Hardware Profile dropdown: Ascend 910C, CloudMatrix 384
- Total Devices (number input)
- Devices per Node (read-only, from profile)
- Nodes (auto-computed, read-only)
- **6D Parallelism** — compact grid layout:
  - TP, PP, DP (auto-computed), EP, CP, CP Type
- SP checkbox, ZeRO Stage dropdown, PP Schedule dropdown
- **Memory Optimizations** — checkboxes:
  - Recompute Attention, Full Recomputation, Optimizer Offload, Activation Offload

### Section: RL Training

**Collapsed summary:** `10k prompts · grp=8 · separate · ref model`

**Expanded fields:**
- Data: Total Prompts, Group Size, Total Responses (auto-computed)
- Time Budget (hours)
- Sequence Lengths: Avg Prompt Length, Avg Response Length, Max Response Length, Std Response Length
- Batch Settings: Train Micro Batch Size, Gradient Accumulation Steps, Gen Batch Size
- Deployment: Mode (Colocated/Separate), Reference Model checkbox, Ref Offload CPU, Speculative Decoding
- Conditional: MTP Acceptance Length (when speculative decoding enabled)

### Section: Search

**Collapsed summary:** `Pareto · 5 device counts` or `Sensitivity · group_size`

**Expanded fields:**
- Mode: Pareto Search | Sensitivity Analysis (radio)
- Pareto: Device Counts (comma-separated text), Optimization Target dropdown
- Sensitivity: Sweep Parameter dropdown, Sweep Values (comma-separated text)

## Results Panel (Right)

### KPI Cards

4 cards in a single row, equal width. Each card:
- Left border color-coded by status:
  - `#dc2626` (red) = infeasible / OOM
  - `#16a34a` (green) = feasible, within budget
  - `#f59e0b` (amber) = warning / bottleneck indicator
- Content: uppercase label (small), large bold value, detail text (small, muted)

**Cards:**
1. **Epoch Time** — value in hours, detail: feasibility status
2. **Gen TPS** — value in tok/s, detail: generation time
3. **Train TPS** — value in tok/s, detail: training time
4. **Bottleneck** — value: "Generation" or "Training", detail: slack percentage

### Chart Area

Tabbed interface below KPIs. Tabs:

1. **Timeline** — Horizontal stacked bar (Gantt-style) showing generation and training phases
2. **Memory** — Stacked bar chart: per-device memory breakdown (weights, optimizer, activations, ref model, KV cache) with dashed HBM limit line
3. **Topology** — 2D scatter mesh showing device rank assignments across TP/PP/DP/EP groups
4. **Search Results** — Shown only when Search section has been used. Contains:
   - Pareto: scatter plot (devices vs epoch time) + comparison table
   - Sensitivity: bar chart (swept param vs epoch time) + comparison table

### Empty State

Before first run, show a centered instructional message:
> "Configure your model and hardware, then click **Run Analysis** to see performance predictions."

No empty chart frames. Just the message.

### Loading State

While prediction is running:
- "Run Analysis" button shows a spinner and disables
- KPI cards show skeleton loading animation (pulsing gray bars)
- Chart area shows a centered spinner

### Error State

On prediction error:
- KPI cards show "—" with error indicator
- A styled error banner appears above the chart area with the error message
- Red left border, light red background, escaped error text

## REST API

### `GET /api/models`

Returns list of available model templates.

```json
{
  "templates": {
    "Llama-3.1-8B": {"name": "Llama-3.1-8B", "hidden_size": 4096, ...},
    "Qwen2.5-72B": {...},
    ...
  }
}
```

### `GET /api/hardware`

Returns list of available hardware profiles.

```json
{
  "profiles": {
    "Ascend 910C": {"devices_per_node": 8, "hbm_gb": 128, ...},
    "CloudMatrix 384": {...}
  }
}
```

### `POST /api/predict`

Input: Full configuration as JSON.

```json
{
  "model": {
    "name": "Llama-3.1-8B",
    "hidden_size": 4096,
    "vocab_size": 128256,
    "num_layers": 32,
    "dtype": "bf16",
    "layer": {
      "attention": "GQA",
      "num_heads": 32,
      "num_kv_heads": 8,
      "head_dim": 128,
      "ffn": "SwiGLU",
      "intermediate_size": 14336,
      "residual": "standard"
    }
  },
  "hardware": "Ascend 910C",
  "total_devices": 8,
  "parallelism": {
    "tp": 1, "pp": 1, "dp": 8, "ep": 1, "cp": 1,
    "cp_type": "ring", "sp": false,
    "zero_stage": 0, "pp_schedule": "1f1b",
    "recompute_attention": false, "full_recomputation": false,
    "optimizer_offload": false, "activation_offload": false
  },
  "rl": {
    "total_prompts": 10000,
    "group_size": 8,
    "avg_prompt_len": 512,
    "avg_response_len": 2048,
    "max_response_len": 4096,
    "std_response_len": null,
    "train_micro_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "gen_batch_size": 64,
    "colocated": false,
    "reference_model": true,
    "ref_offload_cpu": false,
    "use_speculative_decoding": false,
    "mtp_acceptance_len": null
  }
}
```

Output:

```json
{
  "kpis": {
    "epoch_time_hours": 1.38,
    "gen_tps_target": 38721,
    "train_tps_target": 41586,
    "gen_time_hours": 1.18,
    "train_time_hours": 1.37,
    "bottleneck": "training",
    "bottleneck_slack": 0.164,
    "feasible": false,
    "within_budget": true
  },
  "memory": {
    "weight_gb": 14.0,
    "optimizer_gb": 83.8,
    "activation_peak_gb": 14.0,
    "ref_model_gb": 0,
    "kv_cache_gb": 38.7,
    "usable_hbm_gb": 109,
    "train_feasible": false,
    "gen_feasible": true
  },
  "timeline": {
    "gen_hours": 1.18,
    "train_hours": 1.37,
    "colocated": false
  },
  "topology": {
    "ranks": [...],
    "tp": 1, "pp": 1, "dp": 8, "ep": 1
  }
}
```

Charts are rendered client-side by app.js using Plotly.js with the returned data. No server-side figure generation.

### `POST /api/search`

Input: Same config as predict, plus search parameters:

```json
{
  "model": {...},
  "hardware": "Ascend 910C",
  "total_devices": 8,
  "parallelism": {...},
  "rl": {...},
  "search": {
    "mode": "pareto",
    "device_counts": [8, 16, 32, 64, 128],
    "optimization_target": "epoch_time_hours"
  }
}
```

Or for sensitivity:

```json
{
  "search": {
    "mode": "sensitivity",
    "sweep_param": "group_size",
    "sweep_values": [4, 8, 16, 32]
  }
}
```

Output:

```json
{
  "results": [
    {
      "devices": 8,
      "parallelism": {"tp": 1, "pp": 1, "dp": 8, "ep": 1},
      "epoch_time_hours": 1.38,
      "gen_tps": 38721,
      "train_tps": 41586,
      "feasible": false,
      "is_pareto": true,
      "is_oom": false
    }
  ],
  "status": "Pareto search complete. 70 configs evaluated."
}
```

### `POST /api/hf-import`

Input: `{"model_id": "meta-llama/Meta-Llama-3-8B"}`

Output: ModelConfig JSON (same shape as model templates).

## Design System

Per `.impeccable.md` and Impeccable `/frontend-design`:

### Typography
- **Font**: Loaded via `<link>` tag in HTML head (not CSS @import). Choose a distinctive sans-serif — not Inter/Roboto/Open Sans.
- **Type scale**: Modular scale with CSS custom properties (--text-xs through --text-3xl)
- **KPI values**: Large (--text-3xl), bold (700), tabular-nums
- **Labels**: Small (--text-xs), uppercase, letter-spacing 0.06em
- **Body**: 1rem, line-height 1.5

### Colors (CSS custom properties)
```css
--accent: #CF0A2C;         /* Huawei red */
--accent-hover: #A80823;
--accent-light: #FDE8EB;
--bg-page: #FAF8F6;        /* warm off-white */
--bg-card: #FFFFFF;
--bg-panel: #FFFFFF;        /* config panel background */
--text-primary: #1a1a1a;
--text-secondary: #6b7280;
--text-tertiary: #9ca3af;
--border-subtle: #E8E5E0;
--status-success: #16a34a;
--status-error: #dc2626;
--status-warning: #f59e0b;
--chart-bg: #FAFAF8;
```

### Anti-Patterns to Avoid
Per Impeccable DON'T list:
- No glassmorphism, no gradient text, no purple-blue gradients
- No hero metric template (all 4 KPIs equal weight)
- No bounce/elastic easing
- No cards-in-cards nesting
- No pure black (#000) or pure white (#fff) for large areas
- No modals (use inline editing in the accordion)
- No centered-everything layout (left-aligned config, asymmetric composition)

### Motion
- Accordion open/close: `grid-template-rows` transition, 250ms ease-out
- Chart tab switch: opacity crossfade, 200ms
- KPI card appear: staggered fade-in on first results load
- Button hover: 150ms background-color transition
- Respect `prefers-reduced-motion`

## File Structure

```
src/llm_perf/ui/
├── __init__.py
├── api.py              # FastAPI app, REST endpoints, static file serving
└── static/
    ├── index.html      # Single page — structure, <link> fonts, <script> Plotly CDN
    ├── styles.css      # All styles: tokens, layout, config panel, results, charts
    └── app.js          # Vanilla JS: accordion, tabs, fetch API, Plotly rendering
```

## Files to Delete

All existing Gradio UI files:
- `app.py`, `tab_model.py`, `tab_hardware.py`, `tab_rl.py`, `tab_search.py`
- `results.py`, `plots.py`, `topology.py`, `_theme.py`
- `hf_import.py` (logic absorbed into `api.py`)

## Files to Keep (Unchanged)

- `model.py`, `search.py`, `config.py`, `pipeline.py`, `builder.py`, `simulator.py`, `ops.py`, `report.py`
- `cli.py` (update the `ui` subcommand to launch FastAPI instead of Gradio)
- All config YAMLs

## Dependencies

Add to `pyproject.toml`:
- `fastapi>=0.100`
- `uvicorn[standard]>=0.20`

Remove:
- `gradio>=4.0` (or keep as optional if CLI still wants it)

## Success Criteria

1. All existing functionality preserved (predict, search, HF import, topology)
2. Passes Impeccable `/audit` with score ≥ 14/20
3. No "AI slop" tells — passes the "did AI make this?" test
4. Charts render in <1s (client-side Plotly, no server-side figure serialization)
5. Config changes + re-run takes <2 clicks
6. Works on desktop (≥1024px) and tablet (≥768px)
