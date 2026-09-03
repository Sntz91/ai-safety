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
from ai_safety.models.diagnostic.aggregation import aggregate_to_scan_level
from ai_safety.models.monitor.aggregation import aggregate_dual_pooling_to_scan_level, aggregate_topk_saliency_to_scan_level
from ai_safety.models.monitor.threshold_distance import compute_decision_distance

# Scan output directory for monitor runs
out_dir = Path("experiments/outputs/monitor")
available_runs = []
if out_dir.exists():
    available_runs = sorted([d.name for d in out_dir.iterdir() if d.is_dir()], reverse=True)

if not available_runs:
    st.error("No monitor evaluation outputs found.")
    st.stop()

selected_run = st.sidebar.selectbox("Select Monitor Run", available_runs)

run_path = out_dir / selected_run
metrics_path = run_path / "metrics.json"
curves_path = run_path / "curves.json"

if not metrics_path.exists() or not curves_path.exists():
    st.error(f"Metrics or curves missing for run {selected_run}.")
    st.stop()

with open(metrics_path, "r") as f:
    run_metrics = json.load(f)
with open(curves_path, "r") as f:
    run_curves = json.load(f)

datasets = list(run_metrics.keys())
selected_ds = st.sidebar.selectbox("Select Dataset Split", datasets)
subtypes = list(run_metrics[selected_ds].keys()) if selected_ds in run_metrics else []
selected_subtype = st.sidebar.selectbox("Select Target Class", subtypes)

# Aggregation Level
selected_level = st.sidebar.radio(
    "Aggregation Level",
    ["slice_level", "scan_level"],
    index=0,
    format_func=lambda x: x.replace("_", " ").title()
)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_and_merge_data(mon_run, diag_run, dataset):
    mon_path = Path("runs/monitor") / mon_run / f"predictions-{dataset}.csv"
    diag_path = Path("runs/diagnostic") / diag_run / f"predictions-{dataset}.csv"
    if not mon_path.exists() or not diag_path.exists():
        return None
    df_mon = pd.read_csv(mon_path)
    df_diag = pd.read_csv(diag_path)
    return df_mon.merge(df_diag, on=["sop_uid", "series_id", "dataset"], suffixes=("_mon", "_diag"))

@st.cache_data
def load_thresholds(run_dir):
    th_path = Path(run_dir) / "thresholds.yaml"
    if th_path.exists():
        with open(th_path, "r") as f:
            return yaml.safe_load(f)
    return {}

# KPI Summary Counts
mon_cfg_path = Path("runs/monitor") / selected_run / "config.yaml"
diag_run_name = ""
if mon_cfg_path.exists():
    mon_cfg = load_config(mon_cfg_path)
    diag_run_name = Path(mon_cfg.get("diagnostic", {}).get("run_dir", "")).name

df_merged = load_and_merge_data(selected_run, diag_run_name, selected_ds) if diag_run_name else None

total_samples = 0
total_errors = 0
boundary_errors = 0
silent_mistakes = 0
hc_fp_count = 0
hc_fn_count = 0

if df_merged is not None and f"prob_{selected_subtype}_diag" in df_merged.columns:
    diag_thresh = load_thresholds(Path("runs/diagnostic") / diag_run_name)
    subtypes_list = [c.replace("label_", "").replace("_diag", "") for c in df_merged.columns if c.startswith("label_") and c.endswith("_diag")]
    subtype_idx = subtypes_list.index(selected_subtype) if selected_subtype in subtypes_list else 0
    t_diag_slice = diag_thresh.get("slice", [0.5])[subtype_idx] if "slice" in diag_thresh and subtype_idx < len(diag_thresh["slice"]) else 0.5
    t_diag_scan = diag_thresh.get("scan", [0.5])[subtype_idx] if "scan" in diag_thresh and subtype_idx < len(diag_thresh["scan"]) else 0.5

    p_d = df_merged[f"prob_{selected_subtype}_diag"].values
    y_d = df_merged[f"label_{selected_subtype}_diag"].values

    if selected_level == "slice_level":
        total_samples = len(df_merged)
        total_errors = int((y_d != (p_d >= t_diag_slice)).sum())
        is_fn = (y_d == 1) & (p_d < t_diag_slice)
        is_fp = (y_d == 0) & (p_d >= t_diag_slice)
        hc_fp_count = int((is_fp & (p_d >= 0.80)).sum())
        hc_fn_count = int((is_fn & (p_d <= 0.10)).sum())
    else:
        series_ids = df_merged["series_id"].values
        _, sc_probs, sc_gts = aggregate_to_scan_level(series_ids, p_d, y_d, k=3)
        total_samples = len(sc_probs)
        total_errors = int((sc_gts != (sc_probs >= t_diag_scan)).sum())
        is_fn = (sc_gts == 1) & (sc_probs < t_diag_scan)
        is_fp = (sc_gts == 0) & (sc_probs >= t_diag_scan)
        hc_fp_count = int((is_fp & (sc_probs >= 0.80)).sum())
        hc_fn_count = int((is_fn & (sc_probs <= 0.10)).sum())

    silent_mistakes = hc_fp_count + hc_fn_count
    boundary_errors = max(0, total_errors - silent_mistakes)

unit = "Slices" if selected_level == "slice_level" else "Scans"

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Clinical Safety & Subgroups", "Discrimination & Calibration", "Monitor Flow", "Aggregation Comparison", "Configuration"])

# Helper to retrieve metric block across new/legacy schema
def get_block(ds, subtype, cohort, level, model, kind="metrics"):
    root = run_metrics if kind == "metrics" else run_curves
    sub = root.get(ds, {}).get(subtype, {})
    if "slice_level" in sub or "scan_level" in sub:
        return sub.get(level, {}).get(model, {})
    return sub.get(cohort, {}).get(level, {}).get(model, {})

with tab1:
    subgroups = [
        ("High-Confidence False Positives (p >= 0.80, False Alarms)", "high_conf_fp", hc_fp_count),
        ("High-Confidence False Negatives (p <= 0.10, Missed Bleeds)", "high_conf_fn", hc_fn_count),
        ("High-Confidence Errors", "high_conf", silent_mistakes),
        ("All Diagnostic Errors", "all", total_errors)
    ]

    html_table = [
        '<div style="overflow-x: auto; border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 8px; margin-bottom: 20px;">',
        '<table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; line-height: 1.5;">',
        '<thead><tr style="border-bottom: 2px solid rgba(255, 255, 255, 0.15); background-color: rgba(255, 255, 255, 0.04);">',
        '<th style="padding: 12px 16px; font-weight: 600;">Error Subgroup</th>',
        '<th style="padding: 12px 16px; font-weight: 600;">Method</th>',
        '<th style="padding: 12px 16px; text-align: right; font-weight: 600;">Recall @ 5%</th>',
        '<th style="padding: 12px 16px; text-align: right; font-weight: 600;">Recall @ 10%</th>',
        '<th style="padding: 12px 16px; text-align: right; font-weight: 600;">Recall @ 20%</th>',
        '</tr></thead><tbody>',
    ]

    for i, (sg_title, sg_key, count) in enumerate(subgroups):
        mon_cu = get_block(selected_ds, selected_subtype, sg_key, selected_level, "monitor").get("clinical_utility", {})
        sdist_cu = get_block(selected_ds, selected_subtype, sg_key, selected_level, "threshold_distance").get("clinical_utility", {})

        bg = "background-color: rgba(255, 255, 255, 0.02);" if i % 2 == 0 else "background-color: transparent;"
        border_top = "border-top: 1px solid rgba(255, 255, 255, 0.08);" if i > 0 else ""
        count_str = f"{count:,} {unit.lower()}" if count > 0 else ""

        html_table.append(f'<tr style="{bg} {border_top}">')
        html_table.append(f'<td rowspan="2" style="padding: 14px 16px; vertical-align: middle; font-weight: 600; border-right: 1px solid rgba(255, 255, 255, 0.08); min-width: 280px;">{sg_title}<br><span style="font-size: 12px; font-weight: 400; opacity: 0.7;">{count_str}</span></td>')
        html_table.append('<td style="padding: 10px 16px; opacity: 0.85;">Baseline s_dist</td>')
        html_table.append(f'<td style="padding: 10px 16px; text-align: right; opacity: 0.85;">{sdist_cu.get("recall_at_0.05", 0)*100:.1f}%</td>')
        html_table.append(f'<td style="padding: 10px 16px; text-align: right; opacity: 0.85;">{sdist_cu.get("recall_at_0.1", 0)*100:.1f}%</td>')
        html_table.append(f'<td style="padding: 10px 16px; text-align: right; opacity: 0.85;">{sdist_cu.get("recall_at_0.2", 0)*100:.1f}%</td>')
        html_table.append('</tr>')

        html_table.append(f'<tr style="{bg}">')
        html_table.append('<td style="padding: 10px 16px; font-weight: 600; color: #8aadf4;">Visual Monitor</td>')
        html_table.append(f'<td style="padding: 10px 16px; text-align: right; font-weight: 600; color: #8aadf4;">{mon_cu.get("recall_at_0.05", 0)*100:.1f}%</td>')
        html_table.append(f'<td style="padding: 10px 16px; text-align: right; font-weight: 600; color: #8aadf4;">{mon_cu.get("recall_at_0.1", 0)*100:.1f}%</td>')
        html_table.append(f'<td style="padding: 10px 16px; text-align: right; font-weight: 600; color: #8aadf4;">{mon_cu.get("recall_at_0.2", 0)*100:.1f}%</td>')
        html_table.append('</tr>')

    html_table.append('</tbody></table></div>')
    st.markdown("".join(html_table), unsafe_allow_html=True)

    st.markdown("---")
    
    fig_budget = go.Figure()
    
    # Subgroup plot lines from curves.json
    for sg_title, sg_key, color in [
        ("High-Conf False Positives", "high_conf_fp", "#ed8796"),
        ("High-Conf False Negatives", "high_conf_fn", "#8aadf4"),
        ("All Diagnostic Errors", "all", "#a6da95")
    ]:
        mon_bc = get_block(selected_ds, selected_subtype, sg_key, selected_level, "monitor", kind="curves").get("budget_curve", {})
        sdist_bc = get_block(selected_ds, selected_subtype, sg_key, selected_level, "threshold_distance", kind="curves").get("budget_curve", {})
        
        if "budgets" in mon_bc and "recalls" in mon_bc:
            budgets_pct = [b * 100 for b in mon_bc["budgets"]]
            mon_recalls_pct = [r * 100 for r in mon_bc["recalls"]]
            fig_budget.add_trace(go.Scatter(
                x=budgets_pct, y=mon_recalls_pct, mode="lines+markers",
                name=f"Monitor: {sg_title}",
                line=dict(color=color, width=3)
            ))
            
        if "budgets" in sdist_bc and "recalls" in sdist_bc:
            budgets_pct = [b * 100 for b in sdist_bc["budgets"]]
            sdist_recalls_pct = [r * 100 for r in sdist_bc["recalls"]]
            fig_budget.add_trace(go.Scatter(
                x=budgets_pct, y=sdist_recalls_pct, mode="lines+markers",
                name=f"s_dist: {sg_title}",
                line=dict(color=color, width=2, dash="dash")
            ))
        
    fig_budget.update_layout(
        title="Audit Budget vs Recall Across Error Subgroups (0% to 50%)",
        xaxis_title="Clinical Review Budget (% of Total Workload)",
        yaxis_title="Error Recall (% Intercepted)",
        template="plotly_dark",
        yaxis=dict(range=[0, 105], dtick=10),
        xaxis=dict(range=[0, 50], dtick=5, ticksuffix="%"),
    )
    st.plotly_chart(fig_budget, use_container_width=True)

with tab2:
    mon_all = get_block(selected_ds, selected_subtype, "all", selected_level, "monitor")
    sdist_all = get_block(selected_ds, selected_subtype, "all", selected_level, "threshold_distance")
    
    metrics_list = [
        ("AUROC ↑", "discrimination", "auroc"),
        ("AUPRC ↑", "discrimination", "auprc"),
        ("Brier Score ↓", "calibration", "brier"),
        ("ECE ↓", "calibration", "ece"),
        ("AdaECE ↓", "calibration", "ada_ece"),
    ]
    
    comp_metrics = []
    for name, cat, key in metrics_list:
        mon_dict = mon_all.get(cat, {}).get(key, {})
        sdist_dict = sdist_all.get(cat, {}).get(key, {})
        
        def fmt(d):
            if not d or "value" not in d:
                return "N/A"
            val = d["value"]
            ci_l = d.get("ci_lower")
            ci_u = d.get("ci_upper")
            if ci_l is not None and ci_u is not None:
                return f"{val:.4f} [{ci_l:.4f}, {ci_u:.4f}]"
            return f"{val:.4f}"
            
        comp_metrics.append({
            "Metric": name,
            "Visual Monitor": fmt(mon_dict),
            "Threshold-Distance Baseline": fmt(sdist_dict),
        })
        
    st.dataframe(pd.DataFrame(comp_metrics), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    fig_cols = st.columns(2)
    fig_roc = go.Figure()
    fig_pr = go.Figure()
    
    mon_c = get_block(selected_ds, selected_subtype, "all", selected_level, "monitor", kind="curves")
    sdist_c = get_block(selected_ds, selected_subtype, "all", selected_level, "threshold_distance", kind="curves")
    
    mon_auroc = mon_all.get("discrimination", {}).get("auroc", {}).get("value", 0)
    sdist_auroc = sdist_all.get("discrimination", {}).get("auroc", {}).get("value", 0)
    mon_auprc = mon_all.get("discrimination", {}).get("auprc", {}).get("value", 0)
    sdist_auprc = sdist_all.get("discrimination", {}).get("auprc", {}).get("value", 0)
    
    if "roc" in mon_c:
        fig_roc.add_trace(go.Scatter(x=mon_c["roc"]["fpr"], y=mon_c["roc"]["tpr"], mode="lines", name=f"Monitor (AUC: {mon_auroc:.3f})", line=dict(color="#8aadf4", width=3)))
    if "roc" in sdist_c:
        fig_roc.add_trace(go.Scatter(x=sdist_c["roc"]["fpr"], y=sdist_c["roc"]["tpr"], mode="lines", name=f"s_dist (AUC: {sdist_auroc:.3f})", line=dict(color="#6e738d", width=2, dash="dash")))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="gray"), name="Random", showlegend=False))
    fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", template="plotly_dark")
    fig_cols[0].plotly_chart(fig_roc, use_container_width=True)
    
    if "pr" in mon_c:
        fig_pr.add_trace(go.Scatter(x=mon_c["pr"]["recall"], y=mon_c["pr"]["precision"], mode="lines", name=f"Monitor (AUC: {mon_auprc:.3f})", line=dict(color="#a6da95", width=3)))
    if "pr" in sdist_c:
        fig_pr.add_trace(go.Scatter(x=sdist_c["pr"]["recall"], y=sdist_c["pr"]["precision"], mode="lines", name=f"s_dist (AUC: {sdist_auprc:.3f})", line=dict(color="#6e738d", width=2, dash="dash")))
    fig_pr.update_layout(title="Precision-Recall Curve Comparison", xaxis_title="Recall", yaxis_title="Precision", template="plotly_dark")
    fig_cols[1].plotly_chart(fig_pr, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Threshold-dependent")
    
    thresh_options = list(mon_all.get("operating_point", {}).keys())
    if not thresh_options:
        thresh_options = ["LOCKED (85% Sens)"]
        
    selected_thresh = st.selectbox("Threshold Method", thresh_options, label_visibility="collapsed")
    
    mon_thresh_data = mon_all.get("operating_point", {}).get(selected_thresh, {})
    sdist_thresh_data = sdist_all.get("operating_point", {}).get(selected_thresh, {})
    
    op_rows = []
    for name, key in [
        ("Threshold", "threshold"),
        ("F1 Score ↑", "f1"),
        ("Sensitivity ↑", "sensitivity"),
        ("Specificity ↑", "specificity"),
        ("Precision ↑", "precision"),
    ]:
        def fmt_op(td):
            if not td: return "N/A"
            if key == "threshold":
                th = td.get("threshold")
                return f"{th:.4f}" if th is not None else "N/A"
            if key in td:
                val = td[key]["value"]
                ci_l = td[key].get("ci_lower")
                ci_u = td[key].get("ci_upper")
                if ci_l is not None and ci_u is not None:
                    return f"{val:.4f} [{ci_l:.4f}, {ci_u:.4f}]"
                return f"{val:.4f}"
            return "N/A"
            
        op_rows.append({
            "Metric": name,
            "Visual Monitor": fmt_op(mon_thresh_data),
            "Threshold-Distance Baseline": fmt_op(sdist_thresh_data),
        })
        
    st.dataframe(pd.DataFrame(op_rows), use_container_width=True, hide_index=True)
    
    if "confusion" in mon_thresh_data and "confusion" in sdist_thresh_data:
        c_m = mon_thresh_data["confusion"]
        c_s = sdist_thresh_data["confusion"]
        st.markdown(f"**Visual Monitor Confusion:** TP: `{c_m['tp']}` | FP: `{c_m['fp']}` | TN: `{c_m['tn']}` | FN: `{c_m['fn']}`")
        st.markdown(f"**Baseline s_dist Confusion:** TP: `{c_s['tp']}` | FP: `{c_s['fp']}` | TN: `{c_s['tn']}` | FN: `{c_s['fn']}`")
def compute_flow_data(df, subtype, level, d_thresh, m_thresh):
    if level == "slice_level":
        diag_gt = df[f"label_{subtype}_diag"].values
        diag_prob = df[f"prob_{subtype}_diag"].values
        mon_prob = df[f"prob_{subtype}_mon"].values
    else:
        series_ids = df["series_id"].values
        orig_diag_gts = df[f"label_{subtype}_diag"].values
        orig_diag_probs = df[f"prob_{subtype}_diag"].values

        _, diag_prob, diag_gt = aggregate_to_scan_level(series_ids, orig_diag_probs, orig_diag_gts, k=3)
        _, mon_prob = aggregate_dual_pooling_to_scan_level(
            series_ids, diag_probs=orig_diag_probs, mon_probs=df[f"prob_{subtype}_mon"].values, diag_threshold=d_thresh, k=3
        )

    diag_pred = (diag_prob >= d_thresh).astype(int)
    mon_pred = (mon_prob >= m_thresh).astype(int)

    res = {
        "total": len(diag_gt),
        "dis_pos": {"total": 0, "diag_tp": {"total": 0, "mon_fp": 0, "mon_tn": 0}, "diag_fn": {"total": 0, "mon_tp": 0, "mon_fn": 0}},
        "dis_neg": {"total": 0, "diag_tn": {"total": 0, "mon_fp": 0, "mon_tn": 0}, "diag_fp": {"total": 0, "mon_tp": 0, "mon_fn": 0}},
    }

    for i in range(len(diag_gt)):
        d_gt = diag_gt[i]
        d_pd = diag_pred[i]
        m_pd = mon_pred[i]

        if d_gt == 1:
            res["dis_pos"]["total"] += 1
            if d_pd == 1:
                res["dis_pos"]["diag_tp"]["total"] += 1
                if m_pd == 1:
                    res["dis_pos"]["diag_tp"]["mon_fp"] += 1
                else:
                    res["dis_pos"]["diag_tp"]["mon_tn"] += 1
            else:
                res["dis_pos"]["diag_fn"]["total"] += 1
                if m_pd == 1:
                    res["dis_pos"]["diag_fn"]["mon_tp"] += 1
                else:
                    res["dis_pos"]["diag_fn"]["mon_fn"] += 1
        else:
            res["dis_neg"]["total"] += 1
            if d_pd == 0:
                res["dis_neg"]["diag_tn"]["total"] += 1
                if m_pd == 1:
                    res["dis_neg"]["diag_tn"]["mon_fp"] += 1
                else:
                    res["dis_neg"]["diag_tn"]["mon_tn"] += 1
            else:
                res["dis_neg"]["diag_fp"]["total"] += 1
                if m_pd == 1:
                    res["dis_neg"]["diag_fp"]["mon_tp"] += 1
                else:
                    res["dis_neg"]["diag_fp"]["mon_fn"] += 1

    return res


with tab3:
    if df_merged is None:
        st.warning(f"Prediction files missing for {selected_ds}. Cannot generate Monitor Flow.")
    else:
        diag_thresh = load_thresholds(Path("runs/diagnostic") / diag_run_name)
        mon_thresh = load_thresholds(Path("runs/monitor") / selected_run)

        subtypes_list = []
        for col in df_merged.columns:
            if col.startswith("label_") and col.endswith("_diag"):
                subtypes_list.append(col.replace("label_", "").replace("_diag", ""))

        subtype_idx = subtypes_list.index(selected_subtype) if selected_subtype in subtypes_list else 0
        level_key = "scan" if selected_level == "scan_level" else "slice"

        def_diag_t = diag_thresh.get(level_key, [0.5])[subtype_idx] if level_key in diag_thresh and subtype_idx < len(diag_thresh[level_key]) else 0.5
        def_mon_t = mon_thresh.get(level_key, [0.5])[subtype_idx] if level_key in mon_thresh and subtype_idx < len(mon_thresh[level_key]) else 0.5

        flow = compute_flow_data(df_merged, selected_subtype, selected_level, float(def_diag_t), float(def_mon_t))

        total_scans = flow["total"]
        pos_total = flow["dis_pos"]["total"]
        neg_total = flow["dis_neg"]["total"]


        def render_mermaid_html(mermaid_code: str, container_id: str, download_filename: str) -> str:
            svg_id = f"mermaid-svg-{container_id}"
            return f"""
                <div style="text-align: right; margin-bottom: 8px;">
                    <a id="download-{container_id}" href="#" download="{download_filename}" style="color: #cad3f5; text-decoration: none; font-family: sans-serif; background-color: #363a4f; padding: 6px 12px; border-radius: 6px; font-size: 13px;">
                        Download SVG
                    </a>
                </div>
                <div id="{container_id}" style="display: flex; justify-content: center; width: 100%;">
                </div>
                
                <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
                    mermaid.initialize({{ 
                        startOnLoad: false, 
                        theme: 'dark',
                        securityLevel: 'loose',
                        htmlLabels: true
                    }});
                    
                    const graphDefinition = `{mermaid_code}`;
                    
                    const render = () => {{
                        mermaid.render('{svg_id}', graphDefinition).then((result) => {{
                            let svg = result.svg.replace(/<br>/g, '<br/>');
                            const container = document.getElementById('{container_id}');
                            if (container) {{
                                container.innerHTML = svg;
                            }}
                            const svgElement = document.getElementById('{svg_id}');
                            if (svgElement) {{
                                svgElement.style.maxWidth = '100%';
                                svgElement.style.height = 'auto';
                            }}
                            const blob = new Blob([svg], {{ type: 'image/svg+xml' }});
                            const link = document.getElementById('download-{container_id}');
                            if (link) {{
                                link.href = URL.createObjectURL(blob);
                            }}
                        }}).catch((err) => {{
                            console.error(err);
                        }});
                    }};

                    const observer = new IntersectionObserver((entries) => {{
                        if (entries[0].isIntersecting) {{
                            render();
                            observer.disconnect();
                        }}
                    }});
                    observer.observe(document.body);
                </script>
            """

        # 1. Disease Positive Cohort Flow
        tp_tot = flow["dis_pos"]["diag_tp"]["total"]
        fn_tot = flow["dis_pos"]["diag_fn"]["total"]
        tp_pct = (tp_tot / pos_total * 100) if pos_total > 0 else 0
        fn_pct = (fn_tot / pos_total * 100) if pos_total > 0 else 0

        tp_fp = flow["dis_pos"]["diag_tp"]["mon_fp"]
        tp_tn = flow["dis_pos"]["diag_tp"]["mon_tn"]
        fn_tp = flow["dis_pos"]["diag_fn"]["mon_tp"]
        fn_fn = flow["dis_pos"]["diag_fn"]["mon_fn"]
        fn_catch_pct = (fn_tp / fn_tot * 100) if fn_tot > 0 else 0

        pos_lines = [
            "flowchart TD",
            f'    B["Disease Positive Cohort<br/><b>N={pos_total}</b>"]',
            f'    B_TP["Diagnostic Correct (TP)<br/><b>N={tp_tot}</b> ({tp_pct:.1f}%)"]',
            f'    B_FN["Diagnostic Miss (FN)<br/><b>N={fn_tot}</b> ({fn_pct:.1f}%)"]:::critical',
            f'    B -->|"Predicts Pos (p &ge; {def_diag_t:.3f})"| B_TP',
            f'    B -->|"Missed Disease (p &lt; {def_diag_t:.3f})"| B_FN',
            f'    B_TP_FP["Monitor Flags (FP)<br/><b>N={tp_fp}</b>"]:::waste',
            f'    B_TP_TN["Monitor Ignores (TN)<br/><b>N={tp_tn}</b>"]:::harmony',
            f'    B_TP -->|"Flags (p &ge; {def_mon_t:.3f})"| B_TP_FP',
            f'    B_TP -->|"Ignores (p &lt; {def_mon_t:.3f})"| B_TP_TN',
            f'    B_FN_TP["Safety Net Catches It (TP)<br/><b>N={fn_tp}</b> ({fn_catch_pct:.1f}%)"]:::success',
            f'    B_FN_FN["Double Failure (FN)<br/><b>N={fn_fn}</b>"]:::fail',
            f'    B_FN -->|"Flags (p &ge; {def_mon_t:.3f})"| B_FN_TP',
            f'    B_FN -->|"Ignores (p &lt; {def_mon_t:.3f})"| B_FN_FN',
            "    classDef success fill:#a6da95,stroke:#24273a,stroke-width:2px,color:#24273a;",
            "    classDef harmony fill:#8aadf4,stroke:#24273a,stroke-width:2px,color:#24273a;",
            "    classDef waste fill:#eed49f,stroke:#24273a,stroke-width:2px,color:#24273a;",
            "    classDef fail fill:#ed8796,stroke:#24273a,stroke-width:2px,color:#24273a;",
            "    classDef critical stroke:#ed8796,stroke-width:3px,stroke-dasharray:5;",
        ]

        # 2. Healthy Cohort Flow
        tn_tot = flow["dis_neg"]["diag_tn"]["total"]
        fp_tot = flow["dis_neg"]["diag_fp"]["total"]
        tn_pct = (tn_tot / neg_total * 100) if neg_total > 0 else 0
        fp_pct = (fp_tot / neg_total * 100) if neg_total > 0 else 0

        tn_fp = flow["dis_neg"]["diag_tn"]["mon_fp"]
        tn_tn = flow["dis_neg"]["diag_tn"]["mon_tn"]
        fp_tp = flow["dis_neg"]["diag_fp"]["mon_tp"]
        fp_fn = flow["dis_neg"]["diag_fp"]["mon_fn"]
        fp_catch_pct = (fp_tp / fp_tot * 100) if fp_tot > 0 else 0

        neg_lines = [
            "flowchart TD",
            f'    C["Healthy / Negative Cohort<br/><b>N={neg_total}</b>"]',
            f'    C_TN["Diagnostic Correct (TN)<br/><b>N={tn_tot}</b> ({tn_pct:.1f}%)"]',
            f'    C_FP["Diagnostic False Alarm (FP)<br/><b>N={fp_tot}</b> ({fp_pct:.1f}%)"]:::critical',
            f'    C -->|"Predicts Neg (p &lt; {def_diag_t:.3f})"| C_TN',
            f'    C -->|"False Alarm (p &ge; {def_diag_t:.3f})"| C_FP',
            f'    C_TN_FP["Monitor Flags (FP)<br/><b>N={tn_fp}</b>"]:::waste',
            f'    C_TN_TN["Monitor Ignores (TN)<br/><b>N={tn_tn}</b>"]:::harmony',
            f'    C_TN -->|"Flags (p &ge; {def_mon_t:.3f})"| C_TN_FP',
            f'    C_TN -->|"Ignores (p &lt; {def_mon_t:.3f})"| C_TN_TN',
            f'    C_FP_TP["Safety Net Catches It (TP)<br/><b>N={fp_tp}</b> ({fp_catch_pct:.1f}%)"]:::success',
            f'    C_FP_FN["Double Failure (FN)<br/><b>N={fp_fn}</b>"]:::fail',
            f'    C_FP -->|"Flags (p &ge; {def_mon_t:.3f})"| C_FP_TP',
            f'    C_FP -->|"Ignores (p &lt; {def_mon_t:.3f})"| C_FP_FN',
            "    classDef success fill:#a6da95,stroke:#24273a,stroke-width:2px,color:#24273a;",
            "    classDef harmony fill:#8aadf4,stroke:#24273a,stroke-width:2px,color:#24273a;",
            "    classDef waste fill:#eed49f,stroke:#24273a,stroke-width:2px,color:#24273a;",
            "    classDef fail fill:#ed8796,stroke:#24273a,stroke-width:2px,color:#24273a;",
            "    classDef critical stroke:#ed8796,stroke-width:3px,stroke-dasharray:5;",
        ]

        components.html(render_mermaid_html("\n".join(pos_lines), "pos_flow", "disease_positive_flow.svg"), height=620, scrolling=False)

        components.html(render_mermaid_html("\n".join(neg_lines), "neg_flow", "healthy_negative_flow.svg"), height=620, scrolling=False)


with tab4:
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    def compute_aggregation_strategies(df, subtype, diag_t_scan, k=3):
        series_ids = df["series_id"].values
        orig_diag_probs = df[f"prob_{subtype}_diag"].values
        orig_diag_gts = df[f"label_{subtype}_diag"].values
        risk_slice = df[f"prob_{subtype}_mon"].values

        _, scan_diag_probs, scan_diag_gts = aggregate_to_scan_level(series_ids, orig_diag_probs, orig_diag_gts, k=k)
        scan_diag_preds = (scan_diag_probs >= diag_t_scan).astype(int)
        scan_true = (scan_diag_gts != scan_diag_preds).astype(int)

        _, r_pure_top3 = aggregate_to_scan_level(series_ids, risk_slice, k=k)
        r_sdist = compute_decision_distance(scan_diag_probs, diag_threshold=diag_t_scan)
        _, r_top3_diag = aggregate_topk_saliency_to_scan_level(series_ids, orig_diag_probs, risk_slice, k=k)
        r_hybrid = (r_top3_diag + r_sdist) / 2.0

        strategies = {
            "Pure Black-Box (Top-3 Mean)": {
                "scores": r_pure_top3,
                "input_req": "Image Pixels Only (0 Model Access)",
            },
            "Decision Boundary Baseline": {
                "scores": r_sdist,
                "input_req": "Model Probability & Threshold (|p - τ|)",
            },
            "Top-3 Diagnostic Slices Monitor": {
                "scores": r_top3_diag,
                "input_req": "Image Pixels + Diagnostic Saliency",
            },
            "Hybrid Safety Monitor (Top-3 + Decision)": {
                "scores": r_hybrid,
                "input_req": "Image Pixels + Diagnostic Output + τ",
            },
        }
        return scan_diag_probs, scan_diag_gts, scan_diag_preds, scan_true, strategies

    if df_merged is None:
        st.warning(f"Prediction files missing for {selected_ds}. Cannot generate Aggregation Comparison.")
    else:
        subtypes_list = [c.replace("label_", "").replace("_diag", "") for c in df_merged.columns if c.startswith("label_") and c.endswith("_diag")]
        subtype_idx = subtypes_list.index(selected_subtype) if selected_subtype in subtypes_list else 0
        diag_thresh = load_thresholds(Path("runs/diagnostic") / diag_run_name)
        t_scan = diag_thresh.get("scan", [0.5])[subtype_idx] if "scan" in diag_thresh and subtype_idx < len(diag_thresh["scan"]) else 0.5
        scan_diag_probs, scan_diag_gts, scan_diag_preds, scan_true, strategies = compute_aggregation_strategies(
            df_merged, selected_subtype, float(t_scan)
        )

        n_scans = len(scan_true)
        n_errors = int(np.sum(scan_true))
        n_fp = int(np.sum((scan_diag_gts == 0) & (scan_diag_preds == 1)))
        n_fn = int(np.sum((scan_diag_gts == 1) & (scan_diag_preds == 0)))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scans", f"{n_scans:,}")
        c2.metric("Diagnostic Errors", f"{n_errors:,} ({n_errors/n_scans:.1%})")
        c3.metric("False Positives", f"{n_fp:,}")
        c4.metric("False Negatives", f"{n_fn:,}")

        st.markdown("---")

        table_rows = []
        for name, data in strategies.items():
            scores = data["scores"]
            auc = float(roc_auc_score(scan_true, scores))
            ap = float(average_precision_score(scan_true, scores))

            fpr, tpr, roc_thresh = roc_curve(scan_true, scores)

            sens_80 = 0.0
            for f, t in zip(fpr, tpr):
                if f <= 0.20:
                    sens_80 = max(sens_80, float(t))

            cutoff_idx = np.argmin(np.abs(fpr - 0.20))
            th_cut = float(roc_thresh[cutoff_idx]) if cutoff_idx < len(roc_thresh) else 0.5
            flagged = scores >= th_cut

            fp_mask = (scan_diag_gts == 0) & (scan_diag_preds == 1)
            fn_mask = (scan_diag_gts == 1) & (scan_diag_preds == 0)
            hc_mask = ((scan_diag_probs <= 0.10) & (scan_diag_gts == 1)) | ((scan_diag_probs >= 0.80) & (scan_diag_gts == 0))

            fp_rec = float(np.sum(flagged[fp_mask]) / np.sum(fp_mask)) if np.sum(fp_mask) > 0 else 0.0
            fn_rec = float(np.sum(flagged[fn_mask]) / np.sum(fn_mask)) if np.sum(fn_mask) > 0 else 0.0
            hc_rec = float(np.sum(flagged[hc_mask]) / np.sum(hc_mask)) if np.sum(hc_mask) > 0 else 0.0

            table_rows.append({
                "Aggregation Strategy": name,
                "Information Required": data["input_req"],
                "AUROC": f"{auc:.3f}",
                "AUPRC": f"{ap:.3f}",
                "Overall Error Recall (@ 80% Spec)": f"{sens_80:.1%}",
                "FP Detection Rate": f"{fp_rec:.1%}",
                "FN Detection Rate": f"{fn_rec:.1%}",
                "High-Confidence Error Recall": f"{hc_rec:.1%}",
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


with tab5:
    cfg_path = Path(f"runs/monitor/{selected_run}/config.yaml")
    if cfg_path.exists():
        cfg = load_config(cfg_path)
        st.code(yaml.dump(cfg, sort_keys=False), language="yaml")
    else:
        st.warning(f"No configuration file found for run {selected_run}.")

