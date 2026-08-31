import json
import yaml
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from ai_safety.utils.aggregation import aggregate_to_scan_level

st.set_page_config(page_title="Diagnostic Performance", layout="wide")
st.title("Diagnostic Model Performance")

results_dir = Path("experiments/outputs/diagnostic")
if not results_dir.exists() or not list(results_dir.iterdir()):
    st.error(f"No diagnostic results found in {results_dir}. Please run 01_diagnostic_performance.py first.")
    st.stop()

run_ids = sorted([d.name for d in results_dir.iterdir() if d.is_dir()], reverse=True)
selected_runs = st.sidebar.multiselect("Select Diagnostic Runs to Compare", run_ids, default=[run_ids[0]])

if not selected_runs:
    st.info("Please select at least one run.")
    st.stop()

# Load all selected runs
runs_data = {}
for run in selected_runs:
    run_path = results_dir / run
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

# Use the first selected run to determine available options
base_run = selected_runs[0]
base_metrics = runs_data[base_run]["metrics"]

datasets = list(base_metrics.keys())
selected_ds = st.sidebar.selectbox("Select Dataset Split", datasets)
subtypes = list(base_metrics[selected_ds].keys()) if selected_ds in base_metrics else []
selected_subtype = st.sidebar.selectbox("Select Target Class", subtypes)

levels = list(base_metrics[selected_ds][selected_subtype].keys()) if selected_subtype in subtypes else []
default_idx = levels.index("slice_level") if "slice_level" in levels else 0
selected_level = st.sidebar.radio("Aggregation Level", levels, index=default_idx)

st.subheader(f"Metrics for {selected_ds} ({selected_subtype}) - {selected_level.replace('_', ' ').title()}")

tab1, tab2, tab3, tab4 = st.tabs(["Metrics Overview", "Performance Curves", "Probability Distribution", "Configuration"])

with tab1:
    st.markdown("### Continuous Metrics Comparison")
    
    # Build comparison dataframe
    metric_keys = [("AUROC", "auroc"), ("AUPRC", "auprc"), ("Brier Score", "brier"), ("ECE", "ece"), ("AdaECE", "ada_ece")]
    comp_data = []
    
    for name, key in metric_keys:
        row = {"Metric": name}
        for run in selected_runs:
            val_str = "N/A"
            if run in runs_data:
                m = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
                if "continuous" in m and key in m["continuous"]:
                    val = m["continuous"][key]["value"]
                    ci_l = m["continuous"][key]["ci_lower"]
                    ci_u = m["continuous"][key]["ci_upper"]
                    val_str = f"{val:.4f} [{ci_l:.4f}, {ci_u:.4f}]"
            row[run] = val_str
        comp_data.append(row)
        
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    
    st.markdown("### Operating Point (Discrete)")
    if len(selected_runs) == 1:
        run = selected_runs[0]
        ds_metrics = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
        ds_curves = runs_data[run]["curves"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
        
        if ds_metrics and "discrete" in ds_metrics:
            threshold_names = list(ds_metrics["discrete"].keys()) + ["Manual (Slider)"]
            selected_thresh = st.selectbox("Threshold Method", threshold_names, label_visibility="collapsed")
            
            if selected_thresh == "Manual (Slider)":
                slider_val = st.slider("Set Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                import numpy as np
                th_arr = np.clip(ds_curves["roc"]["thresholds"], 0.0, 1.0)
                idx = np.abs(th_arr - slider_val).argmin()
                
                tpr_val = ds_curves["roc"]["tpr"][idx]
                fpr_val = ds_curves["roc"]["fpr"][idx]
                
                base_c = ds_metrics["discrete"][list(ds_metrics["discrete"].keys())[0]]["confusion"]
                P = base_c["tp"] + base_c["fn"]
                N = base_c["fp"] + base_c["tn"]
                
                tp = int(round(tpr_val * P))
                fp = int(round(fpr_val * N))
                fn = P - tp
                tn = N - fp
                
                sens = tp / P if P > 0 else np.nan
                spec = tn / N if N > 0 else np.nan
                ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else np.nan
                
                st.caption(f"Applied Threshold Value: {slider_val:.4f}")
                data_disc = [
                    {"Metric": "F1 Score", "Value": f"{f1:.4f}"},
                    {"Metric": "Sensitivity", "Value": f"{sens:.4f}"},
                    {"Metric": "Specificity", "Value": f"{spec:.4f}"},
                    {"Metric": "Precision", "Value": f"{ppv:.4f}"}
                ]
                st.dataframe(pd.DataFrame(data_disc), use_container_width=True, hide_index=True)
                st.markdown(f"**Confusion Matrix:** TP: `{tp}` | FP: `{fp}` | TN: `{tn}` | FN: `{fn}`")
                
            else:
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
        st.info("Operating Point thresholds are disabled when comparing multiple runs simultaneously to avoid clutter. Please select a single run to view discrete thresholds and confusion matrices.")


with tab2:
    fig_cols = st.columns(2)
    fig_roc = go.Figure()
    fig_pr = go.Figure()

    colors = ['#8aadf4', '#a6da95', '#ed8796', '#eed49f', '#c6a0f6']
    
    for i, run in enumerate(selected_runs):
        color = colors[i % len(colors)]
        if run in runs_data:
            c = runs_data[run]["curves"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
            m = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
            
            if "roc" in c:
                fpr = c["roc"]["fpr"]
                tpr = c["roc"]["tpr"]
                auroc = m["continuous"]["auroc"]["value"] if "continuous" in m else 0
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{run} (AUC: {auroc:.3f})', line=dict(color=color, width=2)))
                
            if "pr" in c:
                prec = c["pr"]["precision"]
                rec = c["pr"]["recall"]
                auprc = m["continuous"]["auprc"]["value"] if "continuous" in m else 0
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name=f'{run} (AUC: {auprc:.3f})', line=dict(color=color, width=2)))

    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random', showlegend=False))
    fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark")
    fig_cols[0].plotly_chart(fig_roc, use_container_width=True)

    fig_pr.update_layout(title="Precision-Recall Curve Comparison", xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark")
    fig_cols[1].plotly_chart(fig_pr, use_container_width=True)


with tab3:
    st.markdown("### Predicted Probability Distributions")
    st.caption("Visualizes the raw probability outputs of the models for Positive (Disease) vs Negative (Healthy) cases.")
    
    for run in selected_runs:
        pred_csv = Path(f"runs/diagnostic/{run}/predictions-{selected_ds}.csv")
        if pred_csv.exists():
            df = pd.read_csv(pred_csv)
            
            prob_col = f"prob_{selected_subtype}"
            label_col = f"label_{selected_subtype}"
            
            if prob_col in df.columns and label_col in df.columns:
                orig_probs = df[prob_col].values
                orig_labels = df[label_col].values
                
                if selected_level == "scan_level" and "series_id" in df.columns:
                    series_ids = df["series_id"].values
                    _, probs, labels = aggregate_to_scan_level(series_ids, orig_probs, orig_labels, k=3)
                else:
                    probs = orig_probs
                    labels = orig_labels
                
                pos_probs = probs[labels == 1]
                neg_probs = probs[labels == 0]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=neg_probs, name="Healthy (Negatives)", opacity=0.75, marker_color="#a6da95", nbinsx=20))
                fig.add_trace(go.Histogram(x=pos_probs, name="Disease (Positives)", opacity=0.75, marker_color="#ed8796", nbinsx=20))
                
                fig.update_layout(
                    title=f"Distribution for {run} ({selected_level.replace('_', ' ').title()})",
                    xaxis_title="Predicted Probability",
                    yaxis_title="Count",
                    barmode="overlay",
                    template="plotly_dark",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Could not find probability/label columns for {selected_subtype} in {run}.")
        else:
            st.warning(f"Predictions CSV not found for {run} on dataset {selected_ds}.")


with tab4:
    st.markdown("### Experiment Configuration")
    run_configs = {}
    for run in selected_runs:
        cfg_path = Path(f"runs/diagnostic/{run}/config.yaml")
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                run_configs[run] = yaml.safe_load(f)

    if not run_configs:
        st.warning("No configuration files found for the selected runs in runs/diagnostic/.")
    elif len(selected_runs) > 1:
        st.markdown("#### Configuration Comparison")
        only_diffs = st.checkbox("Only show differences", value=True, key="diag_only_diffs")

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
            st.markdown("**Model & Loss**")
            st.json({
                "model": cfg.get("model", {}),
                "loss": cfg.get("loss", {}),
            })
        with col2:
            st.markdown("**Training & Optimizer**")
            st.json({
                "training": cfg.get("training", {}),
                "optimizer": cfg.get("optimizer", {}),
            })
        with st.expander("View Full Raw YAML Configuration", expanded=False):
            st.code(yaml.dump(cfg, sort_keys=False), language="yaml")

