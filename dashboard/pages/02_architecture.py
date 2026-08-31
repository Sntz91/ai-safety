import yaml
import json
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

COLORS = {
    "text": "#24273a",
    "cluster_text": "#cad3f5",
    "cluster_fill": "#1e2030",
    "cluster_stroke": "#494d64",
    "edge_label_bg": "#1e2030",
    "diag_fill": "#a6da95",
    "diag_stroke": "#a6da95",
    "mon_fill": "#8aadf4",
    "mon_stroke": "#8aadf4",
    "eval_fill": "#ed8796",
    "eval_stroke": "#ed8796",
    "model_fill": "#c6a0f6",
    "model_stroke": "#c6a0f6",
}

st.set_page_config(page_title="Data Pipeline Architecture", layout="wide")
st.title("Architecture Graph (Mermaid)")

st.markdown("""
This graph is generated **dynamically** by parsing the actual `config.yaml` of your runs.
It displays how data flows through the Diagnostic and Monitoring phases, including where the `LOCKED` thresholds are mathematically calculated during validation.

- **LOCKED (85% Sens)**: The model calculates its threshold on the validation set to strictly hit 85% Sensitivity. This is harmonized across both the Diagnostic and Monitor pipelines.
- **OPTIMAL**: Computed directly on the test sets dynamically (cheating) to find the theoretical performance ceiling.
""")

# Find Monitor Runs
monitor_runs_dir = Path("runs/monitor")
monitor_runs = []
if monitor_runs_dir.exists():
    monitor_runs = [d.name for d in monitor_runs_dir.iterdir() if d.is_dir() and (d / "config.yaml").exists()]

if not monitor_runs:
    st.error("No monitor runs found in runs/monitor/.")
    st.stop()
    
selected_monitor_run = st.sidebar.selectbox("Select Monitor Run", sorted(monitor_runs, reverse=True))

# Load configs
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

try:
    mon_cfg = load_config(monitor_runs_dir / selected_monitor_run / "config.yaml")
    diag_dir = mon_cfg.get('diagnostic', {}).get('run_dir', '')
    diag_cfg = load_config(Path(diag_dir) / "config.yaml")
except FileNotFoundError:
    st.error(f"Config file not found for monitor run {selected_monitor_run} or its associated diagnostic run.")
    st.stop()

# Helpers
def get_names(data_list):
    names = []
    for item in data_list:
        if 'name' in item:
            names.append(item['name'])
        elif 'split' in item:
            names.append(Path(item['split']).stem)
        else:
            names.append("unknown")
    return names

def sanitize(name):
    return name.replace("-", "_").replace(".", "_")

# Extract names
diag_model = diag_cfg['model']['model'].upper()
mon_model = mon_cfg['model']['model'].upper()

diag_train = get_names(diag_cfg['data'].get('train', []))
diag_val = get_names(diag_cfg['data'].get('val', []))
diag_predict = get_names(diag_cfg['data'].get('predict', []))

mon_train = get_names(mon_cfg['data'].get('train', []))
mon_val = get_names(mon_cfg['data'].get('val', []))
mon_eval = get_names(mon_cfg['data'].get('predict', []))

# Build Node Labels dictionary
node_labels = {}
for ds in set(diag_train + diag_val + mon_train + mon_val + mon_eval):
    node_labels[ds] = ds

# Try to load metrics to inject into labels
diag_run_name = Path(diag_dir).name
diag_metrics_file = Path("experiments/outputs/diagnostic") / diag_run_name / "metrics.json"

if diag_metrics_file.exists():
    with open(diag_metrics_file, 'r') as f:
        diag_metrics = json.load(f)
        
    for ds in diag_predict:
        if ds in diag_metrics and ds in node_labels:
            subtypes = list(diag_metrics[ds].keys())
            if subtypes:
                first_sub = subtypes[0]
                levels = diag_metrics[ds][first_sub]
                lvl = "slice_level" if "slice_level" in levels else list(levels.keys())[0]
                
                metrics = levels[lvl]
                auroc = metrics["continuous"]["auroc"]["value"]
                node_labels[ds] += f"<br/>Diag AUROC: {auroc:.3f}"

mon_metrics_file = Path("experiments/outputs/monitor") / selected_monitor_run / "metrics.json"
if mon_metrics_file.exists():
    with open(mon_metrics_file, 'r') as f:
        mon_metrics = json.load(f)
        
    for ds in mon_eval:
        if ds in mon_metrics and ds in node_labels:
            subtypes = list(mon_metrics[ds].keys())
            if subtypes:
                first_sub = subtypes[0]
                levels = mon_metrics[ds][first_sub]
                lvl = "slice_level" if "slice_level" in levels else list(levels.keys())[0]
                model_block = levels[lvl]
                model = "monitor" if "monitor" in model_block else list(model_block.keys())[0]
                metrics = model_block[model]
                auroc = metrics["continuous"]["auroc"]["value"]
                node_labels[ds] += f"<br/>Mon AUROC: {auroc:.3f}"


def fmt_thresholds(vals):
    if not vals:
        return ""
    formatted = [f"{v:.3f}" if isinstance(v, (int, float)) else str(v) for v in vals]
    return f"[{', '.join(formatted)}]" if len(formatted) > 1 else formatted[0]

diag_thresh_file = Path(diag_dir) / "thresholds.yaml"
diag_t_slice = fmt_thresholds(load_config(diag_thresh_file).get("slice", [])) if diag_thresh_file.exists() else ""

mon_thresh_file = monitor_runs_dir / selected_monitor_run / "thresholds.yaml"
mon_t_slice = fmt_thresholds(load_config(mon_thresh_file).get("slice", [])) if mon_thresh_file.exists() else ""

diag_val_edge = f"Validates and Locks Threshold - 85% Sens (tau={diag_t_slice})" if diag_t_slice else "Validates and Locks Threshold - 85% Sens"
mon_val_edge = f"Validates and Locks Threshold - 85% Sens (tau={mon_t_slice})" if mon_t_slice else "Validates and Locks Threshold - 85% Sens"

# Build Mermaid Graph dynamically
lines = ['graph LR']

lines.append('    subgraph cluster_diagnostic [Diagnostic Phase]')
for ds in diag_train:
    lines.append(f'        {sanitize(ds)}["{node_labels[ds]}"]:::diagData')
for ds in diag_val:
    lines.append(f'        {sanitize(ds)}["{node_labels[ds]}"]:::diagData')
lines.append(f'        DiagModel(("Diagnostic Model<br/>{diag_model}{"<br/>tau=" + diag_t_slice if diag_t_slice else ""}")):::model')

for ds in diag_train:
    lines.append(f'        {sanitize(ds)} -->|Trains| DiagModel')
for ds in diag_val:
    lines.append(f'        {sanitize(ds)} -->|"{diag_val_edge}"| DiagModel')
lines.append('    end')


intermediate_sets = set(mon_train + mon_val)
pure_eval_sets = set(mon_eval)

lines.append('    subgraph cluster_monitoring [Monitoring Phase]')
for ds in intermediate_sets:
    lines.append(f'        {sanitize(ds)}["{node_labels[ds]}"]:::monData')
lines.append(f'        MonModel(("Monitor Model<br/>{mon_model}{"<br/>tau=" + mon_t_slice if mon_t_slice else ""}")):::model')

for ds in mon_train:
    lines.append(f'        {sanitize(ds)} -->|Trains on failures| MonModel')
for ds in mon_val:
    lines.append(f'        {sanitize(ds)} -->|"{mon_val_edge}"| MonModel')
lines.append('    end')


lines.append('    subgraph cluster_evaluation [Final Evaluation Unseen]')
for ds in pure_eval_sets:
    lines.append(f'        {sanitize(ds)}["{node_labels[ds]}"]:::evalData')
lines.append('    end')

# Inter-cluster edges
for ds in diag_predict:
    if ds in intermediate_sets or ds in pure_eval_sets:
        lines.append(f'    DiagModel -->|Predicts| {sanitize(ds)}')
        
for ds in mon_eval:
    lines.append(f'    MonModel -->|Evaluates Risk| {sanitize(ds)}')

# Apply styles
lines.append(f'    classDef diagData fill:{COLORS["diag_fill"]},stroke:{COLORS["diag_stroke"]},stroke-width:2px,color:{COLORS["text"]};')
lines.append(f'    classDef monData fill:{COLORS["mon_fill"]},stroke:{COLORS["mon_stroke"]},stroke-width:2px,color:{COLORS["text"]};')
lines.append(f'    classDef evalData fill:{COLORS["eval_fill"]},stroke:{COLORS["eval_stroke"]},stroke-width:2px,color:{COLORS["text"]};')
lines.append(f'    classDef model fill:{COLORS["model_fill"]},stroke:{COLORS["model_stroke"]},stroke-width:3px,color:{COLORS["text"]};')

cluster_style = f'fill:{COLORS["cluster_fill"]},stroke:{COLORS["cluster_stroke"]},stroke-width:2px,color:{COLORS["cluster_text"]},stroke-dasharray: 5 5'
lines.append(f'    style cluster_diagnostic {cluster_style}')
lines.append(f'    style cluster_monitoring {cluster_style}')
lines.append(f'    style cluster_evaluation {cluster_style}')

mermaid_code = "\n".join(lines)

# Render Mermaid via HTML
html_code = f"""
    <div style="text-align: right; margin-bottom: 10px;">
        <a id="download-link" href="#" download="architecture.svg" style="color: #cad3f5; text-decoration: none; font-family: sans-serif; background-color: #363a4f; padding: 8px 12px; border-radius: 6px; font-size: 14px;">
            Download SVG
        </a>
    </div>
    <div id="graph-container" style="display: flex; justify-content: center; width: 100%;">
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: false, 
            theme: 'dark',
            securityLevel: 'loose',
            htmlLabels: true,
            themeVariables: {{
                edgeLabelBackground: '{COLORS["edge_label_bg"]}'
            }}
        }});
        
        const graphDefinition = `{mermaid_code}`;
        
        mermaid.render('mermaid-svg', graphDefinition).then((result) => {{
            let svg = result.svg;
            // Mermaid's internal HTML engine often spits out unclosed <br> tags 
            // which breaks strict XML parsers when the SVG is downloaded.
            svg = svg.replace(/<br>/g, '<br/>');
            
            document.getElementById('graph-container').innerHTML = svg;
            
            // Make the SVG fill the container appropriately
            const svgElement = document.getElementById('mermaid-svg');
            if(svgElement) {{
                svgElement.style.maxWidth = '100%';
                svgElement.style.height = 'auto';
            }}
            
            // Create a blob URL for the download link
            const blob = new Blob([svg], {{ type: 'image/svg+xml' }});
            const url = URL.createObjectURL(blob);
            document.getElementById('download-link').href = url;
        }});
    </script>
"""

components.html(html_code, height=800, scrolling=True)
