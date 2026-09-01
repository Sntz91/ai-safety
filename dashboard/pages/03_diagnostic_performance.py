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

def load_thresholds(run_id):
    th_p = Path("runs/diagnostic") / run_id / "thresholds.yaml"
    if th_p.exists():
        with open(th_p, "r") as f:
            return yaml.safe_load(f)
    return {}


def compute_failure_stats(df, subtype, level, d_thresh):
    p_col = f"prob_{subtype}"
    y_col = f"label_{subtype}"
    if p_col not in df.columns or y_col not in df.columns:
        return None
    p = df[p_col].values
    y = df[y_col].values
    if level == "scan_level" and "series_id" in df.columns:
        _, p, y = aggregate_to_scan_level(df["series_id"].values, p, y, k=3)

    n_total = len(p)
    y_pred = (p >= d_thresh).astype(int)
    is_err = (y != y_pred)
    n_err = int(is_err.sum())
    n_silent_fn = int(((p <= 0.10) & (y == 1)).sum())
    n_silent_fp = int(((p >= 0.80) & (y == 0)).sum())
    n_silent = n_silent_fn + n_silent_fp
    n_bound = max(0, n_err - n_silent)
    return {
        "total": n_total,
        "errors": n_err,
        "boundary": n_bound,
        "silent": n_silent,
        "silent_fn": n_silent_fn,
        "silent_fp": n_silent_fp,
    }


base_thresh = load_thresholds(base_run)
subtypes_list = [c for c in subtypes]
subtype_idx = subtypes_list.index(selected_subtype) if selected_subtype in subtypes_list else 0
lvl_key = "scan" if selected_level == "scan_level" else "slice"
t_diag = base_thresh.get(lvl_key, [0.5])[subtype_idx] if lvl_key in base_thresh and subtype_idx < len(base_thresh[lvl_key]) else 0.5

# Top KPI Summary Cards for selected dataset
pred_csv_cur = Path(f"runs/diagnostic/{base_run}/predictions-{selected_ds}.csv")
if pred_csv_cur.exists():
    df_cur = pd.read_csv(pred_csv_cur)
    cur_stats = compute_failure_stats(df_cur, selected_subtype, selected_level, t_diag)
    if cur_stats:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        unit = "Slices" if selected_level == "slice_level" else "Scans"
        kpi1.metric("Total Workload", f"{cur_stats['total']:,} {unit}")
        err_pct = cur_stats["errors"] / cur_stats["total"] * 100 if cur_stats["total"] > 0 else 0
        kpi2.metric("Diagnostic Errors", f"{cur_stats['errors']:,} ({err_pct:.1f}%)")
        bound_pct = cur_stats["boundary"] / max(1, cur_stats["errors"]) * 100
        kpi3.metric("Boundary Errors", f"{cur_stats['boundary']:,} ({bound_pct:.1f}%)")
        silent_pct = cur_stats["silent"] / max(1, cur_stats["errors"]) * 100
        kpi4.metric("Silent Mistakes", f"{cur_stats['silent']:,} ({silent_pct:.1f}%)")

tab1, tab2, tab3, tab4 = st.tabs(["Metrics Overview", "Performance Curves", "Failure Analysis & Domain Shift", "Configuration"])

with tab1:
    col_disc, col_cal = st.columns(2)
    
    with col_disc:
        st.markdown("#### Discrimination")
        disc_keys = [("AUROC ↑", "auroc"), ("AUPRC ↑", "auprc")]
        comp_disc = []
        for name, key in disc_keys:
            row = {"Metric": name}
            for run in selected_runs:
                val_str = "N/A"
                if run in runs_data:
                    m = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
                    if "discrimination" in m and key in m["discrimination"]:
                        val = m["discrimination"][key]["value"]
                        ci_l = m["discrimination"][key]["ci_lower"]
                        ci_u = m["discrimination"][key]["ci_upper"]
                        val_str = f"{val:.4f} [{ci_l:.4f}, {ci_u:.4f}]"
                row[run] = val_str
            comp_disc.append(row)
        st.dataframe(pd.DataFrame(comp_disc), use_container_width=True, hide_index=True)
        
    with col_cal:
        st.markdown("#### Calibration")
        cal_keys = [("Brier Score ↓", "brier"), ("ECE ↓", "ece"), ("AdaECE ↓", "ada_ece")]
        comp_cal = []
        for name, key in cal_keys:
            row = {"Metric": name}
            for run in selected_runs:
                val_str = "N/A"
                if run in runs_data:
                    m = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
                    if "calibration" in m and key in m["calibration"]:
                        val = m["calibration"][key]["value"]
                        ci_l = m["calibration"][key]["ci_lower"]
                        ci_u = m["calibration"][key]["ci_upper"]
                        val_str = f"{val:.4f} [{ci_l:.4f}, {ci_u:.4f}]"
                row[run] = val_str
            comp_cal.append(row)
        st.dataframe(pd.DataFrame(comp_cal), use_container_width=True, hide_index=True)
        
    st.markdown("---")
    st.markdown("#### Threshold Dependent")
    
    # Collect available threshold methods across selected runs
    thresh_options = []
    for run in selected_runs:
        if run in runs_data:
            m = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
            if "operating_point" in m:
                for k in m["operating_point"].keys():
                    if k not in thresh_options:
                        thresh_options.append(k)
                        
    if not thresh_options:
        thresh_options = ["LOCKED (85% Sens)"]
    if "Manual (Slider)" not in thresh_options:
        thresh_options.append("Manual (Slider)")
        
    selected_thresh = st.selectbox("Threshold Method", thresh_options, label_visibility="collapsed")
    
    if selected_thresh == "Manual (Slider)":
        slider_val = st.slider("Set Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        import numpy as np
        
        comp_disc = [
            {"Metric": "Threshold"},
            {"Metric": "F1 Score ↑"},
            {"Metric": "Sensitivity ↑"},
            {"Metric": "Specificity ↑"},
            {"Metric": "Precision ↑"},
        ]
        conf_strings = {}
        
        for run in selected_runs:
            val_th = f"{slider_val:.4f}"
            val_f1 = "N/A"
            val_sens = "N/A"
            val_spec = "N/A"
            val_ppv = "N/A"
            
            if run in runs_data:
                m = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
                c = runs_data[run]["curves"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
                if "operating_point" in m and "roc" in c:
                    th_arr = np.clip(c["roc"]["thresholds"], 0.0, 1.0)
                    idx = np.abs(th_arr - slider_val).argmin()
                    tpr_val = c["roc"]["tpr"][idx]
                    fpr_val = c["roc"]["fpr"][idx]
                    
                    base_c = m["operating_point"][list(m["operating_point"].keys())[0]]["confusion"]
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
                    
                    val_f1 = f"{f1:.4f}"
                    val_sens = f"{sens:.4f}"
                    val_spec = f"{spec:.4f}"
                    val_ppv = f"{ppv:.4f}"
                    conf_strings[run] = f"TP: `{tp}` | FP: `{fp}` | TN: `{tn}` | FN: `{fn}`"
            
            comp_disc[0][run] = val_th
            comp_disc[1][run] = val_f1
            comp_disc[2][run] = val_sens
            comp_disc[3][run] = val_spec
            comp_disc[4][run] = val_ppv
            
        st.dataframe(pd.DataFrame(comp_disc), use_container_width=True, hide_index=True)
        if len(selected_runs) == 1 and selected_runs[0] in conf_strings:
            st.markdown(f"**Confusion Matrix:** {conf_strings[selected_runs[0]]}")
        elif len(selected_runs) > 1:
            for run, c_str in conf_strings.items():
                st.markdown(f"**{run} Confusion:** {c_str}")
    else:
        metric_rows = [
            ("Threshold", "threshold"),
            ("F1 Score ↑", "f1"),
            ("Sensitivity ↑", "sensitivity"),
            ("Specificity ↑", "specificity"),
            ("Precision ↑", "precision")
        ]
        comp_disc = []
        conf_strings = {}
        
        for name, key in metric_rows:
            row = {"Metric": name}
            for run in selected_runs:
                val_str = "N/A"
                if run in runs_data:
                    m = runs_data[run]["metrics"].get(selected_ds, {}).get(selected_subtype, {}).get(selected_level, {})
                    if "operating_point" in m and selected_thresh in m["operating_point"]:
                        thresh_data = m["operating_point"][selected_thresh]
                        if key == "threshold":
                            th_v = thresh_data.get("threshold")
                            val_str = f"{th_v:.4f}" if th_v is not None else "N/A"
                        elif key in thresh_data:
                            val = thresh_data[key]["value"]
                            ci_l = thresh_data[key]["ci_lower"]
                            ci_u = thresh_data[key]["ci_upper"]
                            if ci_l is not None and ci_u is not None:
                                val_str = f"{val:.4f} [{ci_l:.4f}, {ci_u:.4f}]"
                            else:
                                val_str = f"{val:.4f}"
                        if "confusion" in thresh_data:
                            c = thresh_data["confusion"]
                            conf_strings[run] = f"TP: `{c['tp']}` | FP: `{c['fp']}` | TN: `{c['tn']}` | FN: `{c['fn']}`"
                row[run] = val_str
            comp_disc.append(row)
            
        st.dataframe(pd.DataFrame(comp_disc), use_container_width=True, hide_index=True)
        if len(selected_runs) == 1 and selected_runs[0] in conf_strings:
            st.markdown(f"**Confusion Matrix:** {conf_strings[selected_runs[0]]}")
        elif len(selected_runs) > 1:
            for run, c_str in conf_strings.items():
                st.markdown(f"**{run} Confusion:** {c_str}")


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
                auroc = m.get("discrimination", {}).get("auroc", {}).get("value", 0)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{run} (AUC: {auroc:.3f})', line=dict(color=color, width=2)))
                
            if "pr" in c:
                prec = c["pr"]["precision"]
                rec = c["pr"]["recall"]
                auprc = m.get("discrimination", {}).get("auprc", {}).get("value", 0)
                fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name=f'{run} (AUC: {auprc:.3f})', line=dict(color=color, width=2)))

    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random', showlegend=False))
    fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark")
    fig_cols[0].plotly_chart(fig_roc, use_container_width=True)

    fig_pr.update_layout(title="Precision-Recall Curve Comparison", xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark")
    fig_cols[1].plotly_chart(fig_pr, use_container_width=True)


with tab3:
    # Dynamically compute failure stats across all available prediction datasets for the selected run
    stats_per_ds = {}
    for ds_name in datasets:
        p_csv = Path(f"runs/diagnostic/{base_run}/predictions-{ds_name}.csv")
        if p_csv.exists():
            df_ds = pd.read_csv(p_csv)
            ds_st = compute_failure_stats(df_ds, selected_subtype, selected_level, t_diag)
            if ds_st:
                stats_per_ds[ds_name] = ds_st

    if stats_per_ds:
        # 1. Summary Table across all evaluated datasets
        table_rows = [
            ("Total Workload", lambda s: f"{s['total']:,}"),
            ("Diagnostic Errors", lambda s: f"{s['errors']:,} ({s['errors']/s['total']*100:.1f}%)"),
            ("Boundary Errors", lambda s: f"{s['boundary']:,} ({s['boundary']/max(1, s['errors'])*100:.1f}% of errors)"),
            ("Silent Mistakes (High-Confidence)", lambda s: f"{s['silent']:,} ({s['silent']/max(1, s['errors'])*100:.1f}% of errors)"),
            ("  - Silent Missed Bleeds (p <= 0.10)", lambda s: f"{s['silent_fn']:,} ({s['silent_fn']/max(1, s['errors'])*100:.1f}%)"),
            ("  - Silent False Alarms (p >= 0.80)", lambda s: f"{s['silent_fp']:,} ({s['silent_fp']/max(1, s['errors'])*100:.1f}%)"),
        ]
        table_data = []
        for label, fmt in table_rows:
            r = {"Metric": label}
            for ds_k, st_v in stats_per_ds.items():
                r[ds_k] = fmt(st_v)
            table_data.append(r)
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # 2. 100% Stacked Bar Chart
        st.markdown("---")
        ds_labels = list(stats_per_ds.keys())
        bound_pcts = [stats_per_ds[d]["boundary"] / max(1, stats_per_ds[d]["errors"]) * 100 for d in ds_labels]
        silent_fn_pcts = [stats_per_ds[d]["silent_fn"] / max(1, stats_per_ds[d]["errors"]) * 100 for d in ds_labels]
        silent_fp_pcts = [stats_per_ds[d]["silent_fp"] / max(1, stats_per_ds[d]["errors"]) * 100 for d in ds_labels]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Boundary Errors", y=ds_labels, x=bound_pcts, orientation="h",
            marker_color="#eed49f", text=[f"{v:.1f}%" for v in bound_pcts], textposition="inside"
        ))
        fig_comp.add_trace(go.Bar(
            name="Silent Missed Bleeds (p <= 0.10)", y=ds_labels, x=silent_fn_pcts, orientation="h",
            marker_color="#ed8796", text=[f"{v:.1f}%" for v in silent_fn_pcts], textposition="inside"
        ))
        fig_comp.add_trace(go.Bar(
            name="Silent False Alarms (p >= 0.80)", y=ds_labels, x=silent_fp_pcts, orientation="h",
            marker_color="#8aadf4", text=[f"{v:.1f}%" for v in silent_fp_pcts], textposition="inside"
        ))

        fig_comp.update_layout(
            title=f"Diagnostic Error Composition Across Datasets ({selected_level.replace('_', ' ').title()})",
            xaxis_title="Percentage of Total Diagnostic Errors",
            barmode="stack",
            xaxis=dict(ticksuffix="%", range=[0, 100]),
            template="plotly_dark",
            height=240 + 50 * len(ds_labels),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # 3. Probability Distribution Histogram for Selected Dataset
    st.markdown("---")

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
    run_configs = {}
    for run in selected_runs:
        cfg_path = Path(f"runs/diagnostic/{run}/config.yaml")
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                run_configs[run] = yaml.safe_load(f)

    if not run_configs:
        st.warning("No configuration files found for the selected runs in runs/diagnostic/.")
    elif len(selected_runs) > 1:
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
        with st.expander("View Full Raw YAML Configuration", expanded=False):
            st.code(yaml.dump(cfg, sort_keys=False), language="yaml")

