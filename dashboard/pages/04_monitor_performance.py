import json
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import yaml
import streamlit.components.v1 as components
import sys
import numpy as np

st.set_page_config(page_title="Monitor Model Performance", layout="wide")
st.title("Monitor Model Performance")

# Ensure src modules can be imported for aggregation
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from ai_safety.utils.aggregation import aggregate_to_scan_level, aggregate_dual_pooling_to_scan_level

# Scan output directory for monitor runs
out_dir = Path("experiments/outputs/monitor")
available_runs = []
if out_dir.exists():
    available_runs = sorted([d.name for d in out_dir.iterdir() if d.is_dir()], reverse=True)

if not available_runs:
    st.error("No monitor evaluation outputs found.")
    st.stop()

selected_runs = st.sidebar.multiselect("Select Monitor Runs to Compare", available_runs, default=[available_runs[0]])

if not selected_runs:
    st.info("Please select at least one run.")
    st.stop()

runs_data = {}
for run in selected_runs:
    run_path = out_dir / run
    metrics_path = run_path / "metrics.json"
    curves_path = run_path / "curves.json"
    if metrics_path.exists() and curves_path.exists():
        with open(metrics_path, "r") as f:
            m = json.load(f)
        with open(curves_path, "r") as f:
            c = json.load(f)
        runs_data[run] = {"metrics": m, "curves": c}
    else:
        st.warning(f"Metrics or curves missing for {run}.")

if not runs_data:
    st.stop()

base_run = selected_runs[0]
base_metrics = runs_data[base_run]["metrics"]

datasets = list(base_metrics.keys())
selected_ds = st.sidebar.selectbox("Select Dataset Split", datasets)
subtypes = list(base_metrics[selected_ds].keys()) if selected_ds in base_metrics else []
selected_subtype = st.sidebar.selectbox("Select Target Class", subtypes)

# Select Level (Slice vs Scan)
levels = list(base_metrics[selected_ds][selected_subtype].keys()) if selected_subtype in subtypes else []
default_idx = levels.index("slice_level") if "slice_level" in levels else 0
selected_level = st.sidebar.radio("Aggregation Level", levels, index=default_idx)

# Select Model (Monitor vs Threshold-distance baseline)
models = list(base_metrics[selected_ds][selected_subtype][selected_level].keys()) if selected_level in levels else []
selected_model = st.sidebar.radio("Model", models)

st.subheader(f"Metrics for {selected_ds} ({selected_subtype}) - {selected_level.replace('_', ' ').title()}")

tab1, tab2, tab3, tab4 = st.tabs(["Metrics Overview", "Performance Curves", "Safety Net Flow", "Configuration"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Continuous Metrics Comparison ({selected_model})")
        metric_keys = [("AUROC", "auroc"), ("AUPRC", "auprc"), ("Brier Score", "brier"), ("ECE", "ece"), ("AdaECE", "ada_ece")]
        comp_data = []
        for name, key in metric_keys:
            row = {"Metric": name}
            for run in selected_runs:
                val_str = "N/A"
                if run in runs_data:
                    m_blk = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {}).get(selected_model, {})
                    if "continuous" in m_blk and key in m_blk["continuous"]:
                        val = m_blk["continuous"][key]["value"]
                        ci_l = m_blk["continuous"][key]["ci_lower"]
                        ci_u = m_blk["continuous"][key]["ci_upper"]
                        val_str = f"{val:.4f} [{ci_l:.4f}, {ci_u:.4f}]"
                row[run] = val_str
            comp_data.append(row)
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
        
        if len(selected_runs) == 1:
            base_m = runs_data[base_run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {}).get(selected_model, {})
            if "recall_at_0.05" in base_m.get("continuous", {}):
                st.markdown("**Clinical Utility (Budget vs Recall)**")
                data_budget = []
                for frac in [0.05, 0.1, 0.2]:
                    recall = base_m["continuous"].get(f"recall_at_{frac}", 0.0)
                    frr = base_m["continuous"].get(f"frr_at_{frac}", 0.0)
                    data_budget.append({"Budget": f"{int(frac*100)}%", "Recall": f"{recall*100:.2f}%", "FRR": f"{frr*100:.2f}%"})
                st.dataframe(pd.DataFrame(data_budget), use_container_width=True, hide_index=True)
        
    with col2:
        st.markdown("### Operating Point (Discrete)")
        if len(selected_runs) == 1:
            run = selected_runs[0]
            ds_metrics = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {}).get(selected_model, {})
            if ds_metrics and "discrete" in ds_metrics:
                threshold_names = list(ds_metrics["discrete"].keys())
                selected_thresh = st.selectbox("Threshold Method", threshold_names, label_visibility="collapsed")
                thresh_data = ds_metrics["discrete"][selected_thresh]
                st.caption(f"Applied Threshold Value: {thresh_data['threshold']:.4f}")
                data_disc = []
                for name, key in [("F1 Score", "f1"), ("Sensitivity", "sensitivity"), ("Specificity", "specificity"), ("Precision", "ppv")]:
                    val = thresh_data[key]["value"]
                    ci_l = thresh_data[key]["ci_lower"]
                    ci_u = thresh_data[key]["ci_upper"]
                    data_disc.append({"Metric": name, "Value": f"{val:.4f}", "95% CI": f"[{ci_l:.4f}, {ci_u:.4f}]"})
                st.dataframe(pd.DataFrame(data_disc), use_container_width=True, hide_index=True)
                c = thresh_data["confusion"]
                st.markdown(f"**Confusion Matrix:** TP: `{c['tp']}` | FP: `{c['fp']}` | TN: `{c['tn']}` | FN: `{c['fn']}`")
        else:
            st.info("Operating Point thresholds are shown for single-run view. Select 1 run to view discrete thresholds.")

with tab2:
    fig_cols = st.columns(2)
    fig_roc = go.Figure()
    fig_pr = go.Figure()
    colors = ['#8aadf4', '#a6da95', '#ed8796', '#eed49f', '#c6a0f6']
    
    for i, run in enumerate(selected_runs):
        color = colors[i % len(colors)]
        if run in runs_data:
            c = runs_data[run]["curves"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {}).get(selected_model, {})
            m = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {}).get(selected_model, {})
            if "roc" in c:
                fpr = c["roc"]["fpr"]
                tpr = c["roc"]["tpr"]
                auroc = m["continuous"]["auroc"]["value"] if "continuous" in m and "auroc" in m["continuous"] else 0
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{run} ({selected_model}, AUC: {auroc:.3f})', line=dict(color=color, width=2)))
            if "pr" in c:
                prec = c["pr"]["precision"]
                rec = c["pr"]["recall"]
                auprc = m["continuous"]["auprc"]["value"] if "continuous" in m and "auprc" in m["continuous"] else 0
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name=f'{run} ({selected_model}, AUC: {auprc:.3f})', line=dict(color=color, width=2)))

    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random', showlegend=False))
    fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark")
    fig_cols[0].plotly_chart(fig_roc, use_container_width=True)

    fig_pr.update_layout(title="Precision-Recall Curve Comparison", xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark")
    fig_cols[1].plotly_chart(fig_pr, use_container_width=True)


def load_config(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

@st.cache_data
def load_and_merge_data(mon_run, diag_run, dataset):
    from pathlib import Path
    import pandas as pd
    mon_path = Path("runs/monitor") / mon_run / f"predictions-{dataset}.csv"
    diag_path = Path("runs/diagnostic") / diag_run / f"predictions-{dataset}.csv"
    
    if not mon_path.exists() or not diag_path.exists():
        return None
        
    df_mon = pd.read_csv(mon_path)
    df_diag = pd.read_csv(diag_path)
    
    df = df_mon.merge(df_diag, on=['sop_uid', 'series_id', 'dataset'], suffixes=('_mon', '_diag'))
    return df

@st.cache_data
def load_thresholds(run_dir):
    from pathlib import Path
    import yaml
    th_path = Path(run_dir) / "thresholds.yaml"
    if th_path.exists():
        with open(th_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

@st.cache_data
def compute_flow_data(df, subtype, level, d_thresh, m_thresh):
    if level == "slice_level":
        diag_gt = df[f'label_{subtype}_diag'].values
        diag_prob = df[f'prob_{subtype}_diag'].values
        mon_prob = df[f'prob_{subtype}_mon'].values
    else:
        series_ids = df['series_id'].values
        orig_diag_gts = df[f'label_{subtype}_diag'].values
        orig_diag_probs = df[f'prob_{subtype}_diag'].values
        
        _, diag_prob, diag_gt = aggregate_to_scan_level(series_ids, orig_diag_probs, orig_diag_gts, k=3)
        _, mon_prob = aggregate_dual_pooling_to_scan_level(
            series_ids, diag_probs=orig_diag_probs, mon_probs=df[f'prob_{subtype}_mon'].values, diag_threshold=d_thresh, k=3
        )
        
    diag_pred = (diag_prob >= d_thresh).astype(int)
    mon_pred = (mon_prob >= m_thresh).astype(int)
    
    res = {
        'total': len(diag_gt),
        'dis_pos': {'total': 0, 'diag_tp': {'total': 0, 'mon_fp': 0, 'mon_tn': 0}, 'diag_fn': {'total': 0, 'mon_tp': 0, 'mon_fn': 0}},
        'dis_neg': {'total': 0, 'diag_tn': {'total': 0, 'mon_fp': 0, 'mon_tn': 0}, 'diag_fp': {'total': 0, 'mon_tp': 0, 'mon_fn': 0}},
    }
    
    for i in range(len(diag_gt)):
        d_gt = diag_gt[i]
        d_pd = diag_pred[i]
        m_pd = mon_pred[i]
        
        if d_gt == 1:
            res['dis_pos']['total'] += 1
            if d_pd == 1:
                res['dis_pos']['diag_tp']['total'] += 1
                if m_pd == 1: res['dis_pos']['diag_tp']['mon_fp'] += 1
                else: res['dis_pos']['diag_tp']['mon_tn'] += 1
            else:
                res['dis_pos']['diag_fn']['total'] += 1
                if m_pd == 1: res['dis_pos']['diag_fn']['mon_tp'] += 1
                else: res['dis_pos']['diag_fn']['mon_fn'] += 1
        else:
            res['dis_neg']['total'] += 1
            if d_pd == 0:
                res['dis_neg']['diag_tn']['total'] += 1
                if m_pd == 1: res['dis_neg']['diag_tn']['mon_fp'] += 1
                else: res['dis_neg']['diag_tn']['mon_tn'] += 1
            else:
                res['dis_neg']['diag_fp']['total'] += 1
                if m_pd == 1: res['dis_neg']['diag_fp']['mon_tp'] += 1
                else: res['dis_neg']['diag_fp']['mon_fn'] += 1
                
    return res

with tab3:
    st.markdown("Visualizes how diagnostic errors flow into the monitoring system, highlighting safety net successes and double failures.")
    active_run = st.selectbox("Select Monitor Run for Flow", selected_runs, key="flow_run_select") if len(selected_runs) > 1 else selected_runs[0]
    try:
        mon_cfg = load_config(Path("runs/monitor") / active_run / "config.yaml")
        diag_dir = Path(mon_cfg.get('diagnostic', {}).get('run_dir', ''))
        diag_run_name = diag_dir.name
        
        df = load_and_merge_data(active_run, diag_run_name, selected_ds)
        if df is None:
            st.warning(f"Prediction files missing for {selected_ds}. Cannot generate Safety Net Flow.")
        else:
            diag_thresh = load_thresholds(diag_dir)
            mon_thresh = load_thresholds(Path("runs/monitor") / active_run)
            
            subtypes_list = []
            for col in df.columns:
                if col.startswith("label_") and col.endswith("_diag"):
                    subtypes_list.append(col.replace("label_", "").replace("_diag", ""))
            
            subtype_idx = subtypes_list.index(selected_subtype) if selected_subtype in subtypes_list else 0
            level_key = 'scan' if selected_level == "scan_level" else 'slice'
            
            def_diag_t = diag_thresh.get(level_key, [0.5])[subtype_idx] if level_key in diag_thresh and subtype_idx < len(diag_thresh[level_key]) else 0.5
            def_mon_t = mon_thresh.get(level_key, [0.5])[subtype_idx] if level_key in mon_thresh and subtype_idx < len(mon_thresh[level_key]) else 0.5
            
            flow = compute_flow_data(df, selected_subtype, selected_level, float(def_diag_t), float(def_mon_t))
            
            lines = [
                "flowchart TD",
                # Root
                f'    A["Total Scans - N={flow["total"]}"]',
                
                # Level 1: Ground Truth
                f'    B["Disease Positive - N={flow["dis_pos"]["total"]}"]',
                f'    C["Healthy - N={flow["dis_neg"]["total"]}"]',
                "    A --> B",
                "    A --> C",
                
                # Level 2: Diagnostic Model
                f'    B_TP["Diagnostic Correct TP - N={flow["dis_pos"]["diag_tp"]["total"]}"]',
                f'    B_FN["Diagnostic Miss FN - N={flow["dis_pos"]["diag_fn"]["total"]}"]:::critical',
                "    B --> B_TP",
                "    B -->|Missed Disease| B_FN",
                
                f'    C_TN["Diagnostic Correct TN - N={flow["dis_neg"]["diag_tn"]["total"]}"]',
                f'    C_FP["Diagnostic False Alarm FP - N={flow["dis_neg"]["diag_fp"]["total"]}"]:::critical',
                "    C --> C_TN",
                "    C -->|False Alarm| C_FP",
                
                # Level 3: Monitor Action on Diagnostic Correct
                f'    B_TP_FP["Monitor Flags FP - N={flow["dis_pos"]["diag_tp"]["mon_fp"]}"]:::waste',
                f'    B_TP_TN["Monitor Ignores TN - N={flow["dis_pos"]["diag_tp"]["mon_tn"]}"]:::harmony',
                "    B_TP -->|Flags| B_TP_FP",
                "    B_TP -->|Ignores| B_TP_TN",
                
                f'    C_TN_FP["Monitor Flags FP - N={flow["dis_neg"]["diag_tn"]["mon_fp"]}"]:::waste',
                f'    C_TN_TN["Monitor Ignores TN - N={flow["dis_neg"]["diag_tn"]["mon_tn"]}"]:::harmony',
                "    C_TN -->|Flags| C_TN_FP",
                "    C_TN -->|Ignores| C_TN_TN",
                
                # Level 3: Monitor Action on Diagnostic Errors (The Safety Net)
                f'    B_FN_TP["Safety Net Catches It TP - N={flow["dis_pos"]["diag_fn"]["mon_tp"]}"]:::success',
                f'    B_FN_FN["Double Failure FN - N={flow["dis_pos"]["diag_fn"]["mon_fn"]}"]:::fail',
                "    B_FN -->|Flags| B_FN_TP",
                "    B_FN -->|Ignores| B_FN_FN",
                
                f'    C_FP_TP["Safety Net Catches It TP - N={flow["dis_neg"]["diag_fp"]["mon_tp"]}"]:::success',
                f'    C_FP_FN["Double Failure FN - N={flow["dis_neg"]["diag_fp"]["mon_fn"]}"]:::fail',
                "    C_FP -->|Flags| C_FP_TP",
                "    C_FP -->|Ignores| C_FP_FN",
                
                # Styling
                "    classDef success fill:#a6da95,stroke:#24273a,stroke-width:2px,color:#24273a;",
                "    classDef harmony fill:#8aadf4,stroke:#24273a,stroke-width:2px,color:#24273a;",
                "    classDef waste fill:#eed49f,stroke:#24273a,stroke-width:2px,color:#24273a;",
                "    classDef fail fill:#ed8796,stroke:#24273a,stroke-width:2px,color:#24273a;",
                "    classDef critical stroke:#ed8796,stroke-width:3px,stroke-dasharray:5;"
            ]
            
            mermaid_code = "\n".join(lines)
            html_code = f"""
                <div style="text-align: right; margin-bottom: 10px;">
                    <a id="download-link" href="#" download="safety_net_flow.svg" style="color: #cad3f5; text-decoration: none; font-family: sans-serif; background-color: #363a4f; padding: 8px 12px; border-radius: 6px; font-size: 14px;">
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
                        securityLevel: 'loose'
                    }});
                    
                    const graphDefinition = `{mermaid_code}`;
                    
                    const renderMermaid = () => {{
                        mermaid.render('mermaid-svg', graphDefinition).then((result) => {{
                            let svg = result.svg;
                            svg = svg.replace(/<br>/g, '<br/>');
                            document.getElementById('graph-container').innerHTML = svg;
                            const svgElement = document.getElementById('mermaid-svg');
                            if(svgElement) {{
                                svgElement.style.maxWidth = '100%';
                                svgElement.style.height = 'auto';
                            }}
                            const blob = new Blob([svg], {{ type: 'image/svg+xml' }});
                            document.getElementById('download-link').href = URL.createObjectURL(blob);
                        }}).catch((err) => {{
                            console.error(err);
                        }});
                    }};

                    // Only render when the tab is actually visible to prevent 0x0 bounding box crashes
                    const observer = new IntersectionObserver((entries) => {{
                        if (entries[0].isIntersecting) {{
                            renderMermaid();
                            observer.disconnect();
                        }}
                    }});
                    observer.observe(document.body);
                </script>
            """
            
            components.html(html_code, height=800, scrolling=True)
            
            st.markdown("""
            ### Legend
            - **[Green / Success] Safety Net Success:** The Diagnostic model made an error, and the Monitor successfully flagged it.
            - **[Blue / Harmony] Perfect Harmony:** The Diagnostic model was correct, and the Monitor agreed.
            - **[Yellow / Warning] Wasted Review:** The Diagnostic model was correct, but the Monitor incorrectly flagged it for review.
            - **[Red / Critical] Double Failure:** The Diagnostic model made an error, and the Monitor completely missed it.
            """)
    except FileNotFoundError:
        st.warning("Config file for this run could not be found.")


with tab4:
    st.markdown("### Experiment Configuration")
    run_configs = {}
    for run in selected_runs:
        cfg_path = Path(f"runs/monitor/{run}/config.yaml")
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                run_configs[run] = yaml.safe_load(f)

    if not run_configs:
        st.warning("No configuration files found for the selected runs in runs/monitor/.")
    elif len(selected_runs) > 1:
        st.markdown("#### Configuration Comparison")
        only_diffs = st.checkbox("Only show differences", value=True, key="mon_only_diffs")

        def flatten_dict(d, parent_key=""):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                elif isinstance(v, list):
                    if all(isinstance(item, dict) for item in v):
                        summary = ", ".join(item.get("name", item.get("dataset", str(item))) for item in v)
                        items.append((new_key, summary))
                    else:
                        items.append((new_key, ", ".join(str(i) for i in v)))
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_configs = {run: flatten_dict(cfg) for run, cfg in run_configs.items()}
        all_keys = []
        for flat in flat_configs.values():
            for k in flat:
                if k not in all_keys:
                    all_keys.append(k)

        rows = []
        for key in all_keys:
            row = {"Parameter": key}
            values = []
            for run in selected_runs:
                val = flat_configs.get(run, {}).get(key, "-")
                row[run] = str(val) if val is not None else "None"
                values.append(row[run])

            all_equal = len(set(values)) == 1
            if not (only_diffs and all_equal):
                rows.append(row)

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("All selected runs have identical configurations.")

        st.markdown("#### Raw Configuration Files")
        cols = st.columns(len(selected_runs))
        for col, run in zip(cols, selected_runs):
            with col:
                st.caption(f"**Run: {run}**")
                if run in run_configs:
                    st.code(yaml.dump(run_configs[run], sort_keys=False), language="yaml")
                else:
                    st.warning("Config not found")
    else:
        run = selected_runs[0]
        cfg = run_configs.get(run, {})
        st.markdown(f"#### Configuration for Run `{run}`")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Diagnostic Reference & Model**")
            st.json({
                "diagnostic": cfg.get("diagnostic", {}),
                "monitor": cfg.get("monitor", {}),
                "model": cfg.get("model", {}),
                "loss": cfg.get("loss", {}),
            })
        with col2:
            st.markdown("**Training & Optimizer**")
            st.json({
                "training": cfg.get("training", {}),
                "optimizer": cfg.get("optimizer", {}),
                "scheduler": cfg.get("scheduler", {}),
            })
        with st.expander("View Full Raw YAML Configuration", expanded=False):
            st.code(yaml.dump(cfg, sort_keys=False), language="yaml")

