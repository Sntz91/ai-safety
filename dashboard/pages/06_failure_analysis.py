import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import streamlit as st
import torchvision.transforms.v2 as T_v2

from ai_safety.constants import HIGH_CONF_THRESHOLD_LOW, HIGH_CONF_THRESHOLD_HIGH
from ai_safety.data import DATASET_REGISTRY
from ai_safety.data.dataset import CTSliceDataset
from ai_safety.data.transforms import Transform, apply_window
from ai_safety.utils.helpers import load_thresholds_from_run, resolve_threshold, extract_subtypes
from dashboard.utils import (
    force_garbage_collect,
    load_models,
    compute_gradcam_overlay,
    require_runs_dir,
    load_run_configs,
)

st.set_page_config(page_title="Failure Analysis", layout="wide")
st.title("Failure Analysis")


MON_DIR = Path("runs/monitor")
require_runs_dir(MON_DIR, "No monitor runs found in runs/monitor/.")

monitor_runs = sorted([d.name for d in MON_DIR.iterdir() if d.is_dir()], reverse=True)

st.sidebar.header("Evaluation Selection")
selected_mon_run = st.sidebar.selectbox("Select Monitor Run", monitor_runs)

# Load monitor config to resolve diagnostic run
mon_run_path = MON_DIR / selected_mon_run
mon_cfg, mon_run_path, diag_cfg, diag_run_path = load_run_configs(mon_run_path)
diag_run_id = diag_run_path.name

# Find available prediction files for this monitor run
pred_files = list(mon_run_path.glob("predictions-*.csv"))
if not pred_files:
    st.error(f"No prediction files found in {mon_run_path}.")
    st.stop()

available_datasets = sorted([f.name.replace("predictions-", "").replace(".csv", "") for f in pred_files])
selected_ds = st.sidebar.selectbox("Select Evaluation Cohort", available_datasets)

# Load thresholds
diag_thresh_dict = load_thresholds_from_run(diag_run_path)
mon_thresh_dict = load_thresholds_from_run(mon_run_path)


@st.cache_data
def load_and_merge_predictions(mon_run, diag_run, dataset_name):
    mon_csv = Path("runs/monitor") / mon_run / f"predictions-{dataset_name}.csv"
    diag_csv = Path("runs/diagnostic") / diag_run / f"predictions-{dataset_name}.csv"
    if not mon_csv.exists() or not diag_csv.exists():
        return None
    df_m = pd.read_csv(mon_csv)
    df_d = pd.read_csv(diag_csv)
    return df_m.merge(df_d, on=["sop_uid", "series_id", "dataset"], suffixes=("_mon", "_diag"))


df_merged = load_and_merge_predictions(selected_mon_run, diag_run_id, selected_ds)
if df_merged is None or df_merged.empty:
    st.error(f"Could not load matching prediction files for dataset `{selected_ds}`.")
    st.stop()

# Find available subtypes
subtypes = extract_subtypes(df_merged.columns, suffix="_diag")
if not subtypes:
    st.error("No valid diagnostic label columns found in prediction file.")
    st.stop()

selected_subtype = st.sidebar.selectbox("Select Target Subtype", subtypes)
subtype_idx = subtypes.index(selected_subtype)

diag_slice_tau = resolve_threshold(diag_thresh_dict, "slice", subtype_idx)
mon_slice_tau = resolve_threshold(mon_thresh_dict, "slice", subtype_idx)

st.sidebar.markdown(f"**Diagnostic Model**: `{diag_run_id}` ($\\tau={diag_slice_tau:.4f}$)")
st.sidebar.markdown(f"**Secondary Monitor**: `{selected_mon_run}` ($\\tau={mon_slice_tau:.4f}$)")

# Compute failure metrics and categories
p_diag_all = df_merged[f"prob_{selected_subtype}_diag"].values
p_mon_all = df_merged[f"prob_{selected_subtype}_mon"].values
y_true_all = df_merged[f"label_{selected_subtype}_diag"].values

diag_preds = (p_diag_all >= diag_slice_tau).astype(int)
mon_preds = (p_mon_all >= mon_slice_tau).astype(int)
is_error = diag_preds != y_true_all

df_eval = df_merged.copy()
df_eval["p_diag"] = p_diag_all
df_eval["p_mon"] = p_mon_all
df_eval["y_true"] = y_true_all
df_eval["diag_pred"] = diag_preds
df_eval["mon_pred"] = mon_preds
df_eval["is_error"] = is_error
df_eval["diag_error_magnitude"] = np.abs(p_diag_all - y_true_all)

total_cohort = len(df_eval)
total_errors = int(is_error.sum())
all_fn_mask = (y_true_all == 1) & (diag_preds == 0)
all_fp_mask = (y_true_all == 0) & (diag_preds == 1)
hc_fn_mask = all_fn_mask & (p_diag_all <= HIGH_CONF_THRESHOLD_LOW)
hc_fp_mask = all_fp_mask & (p_diag_all >= HIGH_CONF_THRESHOLD_HIGH)
mon_false_alarm_mask = (~is_error) & (mon_preds == 1)

hc_fn_count = int(hc_fn_mask.sum())
hc_fp_count = int(hc_fp_mask.sum())
all_fn_count = int(all_fn_mask.sum())
all_fp_count = int(all_fp_mask.sum())
mon_fa_count = int(mon_false_alarm_mask.sum())

# Cohort KPI summary cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Evaluated Slices", f"{total_cohort:,}")
k2.metric("Total Diagnostic Errors", f"{total_errors:,}")
k3.metric("Silent Misses (High-Conf FN)", f"{hc_fn_count:,}")
k4.metric("Severe False Alarms (High-Conf FP)", f"{hc_fp_count:,}")

# Subgroup filter selection
category_options = {
    f"High-Confidence False Negatives (p_diag <= 0.10, y=1) [{hc_fn_count:,}]": hc_fn_mask,
    f"High-Confidence False Positives (p_diag >= 0.80, y=0) [{hc_fp_count:,}]": hc_fp_mask,
    f"All False Negatives (Missed Bleeds) [{all_fn_count:,}]": all_fn_mask,
    f"All False Positives (False Alarms) [{all_fp_count:,}]": all_fp_mask,
    f"Monitor False Alarms (Diag Correct, Monitor Flagged) [{mon_fa_count:,}]": mon_false_alarm_mask,
    f"All Diagnostic Errors [{total_errors:,}]": is_error,
}

selected_category = st.radio("Select Failure Subgroup to Inspect:", list(category_options.keys()), horizontal=False)
active_mask = category_options[selected_category]
df_subgroup = df_eval[active_mask].copy()

if df_subgroup.empty:
    st.info("No cases match the selected filter.")
    st.stop()

# Sorting options
sort_col1, sort_col2 = st.columns([2, 1])
with sort_col1:
    sort_criterion = st.selectbox(
        "Sort Subgroup By:",
        [
            "Secondary Monitor Risk (p_mon) [Descending - Top Audited First]",
            "Diagnostic Error Magnitude (|p_diag - y_true|) [Descending]",
        ]
    )

if "p_mon" in sort_criterion:
    df_subgroup = df_subgroup.sort_values(by="p_mon", ascending=False)
else:
    df_subgroup = df_subgroup.sort_values(by="diag_error_magnitude", ascending=False)

# Case selector dropdown
case_labels = []
for rank, (_, row) in enumerate(df_subgroup.iterrows()):
    gt_str = "ICH" if row["y_true"] == 1 else "Normal"
    case_labels.append(
        f"#{rank+1:03d} | {row['sop_uid']} | p_diag: {row['p_diag']:.4f} | p_mon: {row['p_mon']:.4f} | GT: {gt_str}"
    )

selected_case_label = st.selectbox("Select Case to Inspect:", case_labels, index=0)
case_idx = case_labels.index(selected_case_label)
selected_row = df_subgroup.iloc[case_idx]

# Case Telemetry Bar
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    gt_val = int(selected_row["y_true"])
    col_m1.metric("Ground Truth", "ICH" if gt_val == 1 else "Normal")

with col_m2:
    p_d_val = float(selected_row["p_diag"])
    pred_str = "Positive (Bleed)" if p_d_val >= diag_slice_tau else "Negative (Normal)"
    col_m2.metric("Diagnostic Model (p_diag)", f"{p_d_val:.4f}", f"Pred: {pred_str}")

with col_m3:
    p_m_val = float(selected_row["p_mon"])

with col_m4:
    is_err_val = bool(selected_row["is_error"])
    mon_flagged = p_m_val >= mon_slice_tau
    if is_err_val and mon_flagged:
        col_m4.metric("Safety Net Outcome", "Intercepted")
    elif is_err_val and not mon_flagged:
        col_m4.metric("Safety Net Outcome", "Double Failure")
    elif not is_err_val and mon_flagged:
        col_m4.metric("Safety Net Outcome", "Wasted Review")
    else:
        col_m4.metric("Safety Net Outcome", "Harmony")

st.markdown("### Visual Feature Attribution")


# Retrieve raw image
sample_dataset_name = str(selected_row["dataset"])
target_sop_uid = str(selected_row["sop_uid"])

raw_img = None
if sample_dataset_name in DATASET_REGISTRY and Path(DATASET_REGISTRY[sample_dataset_name]).exists():
    try:
        ds = CTSliceDataset(DATASET_REGISTRY[sample_dataset_name], return_sopuid=True)
        ds.filter_by_sop_uids([target_sop_uid])
        if len(ds.records) > 0:
            raw_img = ds.get_image(ds.records[0])
    except Exception as e:
        st.warning(f"Could not read slice from dataset archive: {e}")

if raw_img is not None:
    # Brain window
    disp_img = apply_window(raw_img, 40, 80)
    disp_rgb = np.stack([disp_img] * 3, axis=-1)
    disp_pil = Image.fromarray((disp_rgb * 255).astype(np.uint8)).resize((224, 224))
    disp_uint8 = np.array(disp_pil, dtype=np.uint8)
    vis_base = disp_uint8.astype(np.float32) / 255.0

    transform = Transform(train=False)
    t_input = transform(raw_img).unsqueeze(0)

    try:
        d_model, m_model, d_arch, m_arch, device = load_models(str(diag_run_path), str(mon_run_path))
        t_batch = t_input.to(device)

        d_overlay = compute_gradcam_overlay(d_model, d_arch, t_batch, vis_base)
        m_overlay = compute_gradcam_overlay(m_model, m_arch, t_batch, vis_base)
        del t_batch

        c_img1, c_img2, c_img3 = st.columns(3)
        with c_img1:
            st.image(disp_uint8, caption="Input CT Slice (Brain Window [40, 80 HU])", use_container_width=True)
        with c_img2:
            diag_label = "CORRECT" if not selected_row["is_error"] else "ERROR"
            st.image(d_overlay, caption=f"Diagnostic Attention ({d_arch.upper()}) [{diag_label}]\nProbability: {p_d_val:.4f}", use_container_width=True)
        with c_img3:
            mon_label = "AUDIT FLAGGED" if p_m_val >= mon_slice_tau else "PASSED"
            st.image(m_overlay, caption=f"Secondary Monitor Attention ({m_arch.upper()}) [{mon_label}]\nFailure Risk: {p_m_val:.4f}", use_container_width=True)

        del d_overlay, m_overlay
    except Exception as e:
        st.error(f"Grad-CAM generation error: {e}")
        st.image(disp_uint8, caption="Input CT Slice (Brain Window [40, 80 HU])", use_container_width=True)
else:
    st.info(f"Telemetry above is loaded directly from validated predictions. Raw image archive for '{sample_dataset_name}' is not mounted at '{DATASET_REGISTRY.get(sample_dataset_name, '')}'.")

force_garbage_collect()
