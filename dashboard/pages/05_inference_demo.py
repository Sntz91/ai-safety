import os
import torch
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import streamlit as st
import torchvision.transforms.v2 as T_v2

from ai_safety.models.monitor.threshold_distance import compute_s_dist
from ai_safety.data.transforms import Transform, apply_window
from ai_safety.models.diagnostic.aggregation import aggregate_to_scan_level
from ai_safety.models.monitor.aggregation import aggregate_dual_pooling_to_scan_level
from ai_safety.utils.helpers import load_thresholds_from_run, resolve_threshold
from dashboard.utils import (
    force_garbage_collect,
    load_models,
    compute_gradcam_overlay,
    require_runs_dir,
    load_run_configs,
)

st.set_page_config(page_title="Demo", layout="wide")
st.title("Monitoring / Diagnostic Demo")

MON_DIR = Path("runs/monitor")
require_runs_dir(MON_DIR, "No monitor runs found in runs/monitor/.")

monitor_runs = sorted([d.name for d in MON_DIR.iterdir() if d.is_dir()], reverse=True)

st.sidebar.header("Model Selection")
selected_mon_run = st.sidebar.selectbox("Select Monitor Run", monitor_runs)

# Load monitor config to resolve diagnostic run
mon_run_path = MON_DIR / selected_mon_run
mon_cfg, mon_run_path, diag_cfg, diag_run_path = load_run_configs(mon_run_path)
diag_run_id = diag_run_path.name

# Load thresholds
diag_thresh = load_thresholds_from_run(diag_run_path)
diag_slice_tau = resolve_threshold(diag_thresh, "slice", 0)
diag_scan_tau = resolve_threshold(diag_thresh, "scan", 0)

mon_thresh = load_thresholds_from_run(mon_run_path)
mon_slice_tau = resolve_threshold(mon_thresh, "slice", 0)
mon_scan_tau = resolve_threshold(mon_thresh, "scan", 0)

st.sidebar.markdown(f"**Diagnostic Model**: `{diag_run_id}`")
st.sidebar.markdown(f"**Diag Slice tau**: `{diag_slice_tau:.4f}` | **Scan tau**: `{diag_scan_tau:.4f}`")
st.sidebar.markdown(f"**Monitor Slice tau**: `{mon_slice_tau:.4f}` | **Scan tau**: `{mon_scan_tau:.4f}`")

st.sidebar.markdown("---")
st.sidebar.header("Inference Settings")
aggregation = st.sidebar.radio("Aggregation Level", ["Slice-level", "Scan-level (Top-3 Mean)"])
enable_gradcam = st.sidebar.checkbox("Compute Grad-CAM Attentions", value=True)


def clear_session_cache():
    """Clears all session-cached data and tensors."""
    keys_to_clear = [
        "current_file_signature", "num_slices", "display_rgbs",
        "all_tensors", "p_diag_arr", "p_mon_arr", "s_dist_arr"
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    force_garbage_collect()


# File Uploader (Accepts all file types including extensionless DICOM files)
uploaded_files = st.file_uploader(
    "Upload CT DICOM files or image files (multi-file supported for full scan volumes)",
    accept_multiple_files=True
)


def process_file_to_slice(file, transform, rgb_normalize):
    """Processes a single file directly into a model tensor and compact display array."""
    fname = file.name.lower()
    is_standard_img = any(fname.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])

    # 1. Try DICOM first
    if not is_standard_img:
        try:
            file.seek(0)
            dcm = pydicom.dcmread(file, force=True)
            if hasattr(dcm, "pixel_array"):
                arr = dcm.pixel_array.astype(np.float32)
                slope = float(getattr(dcm, "RescaleSlope", 1.0))
                intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
                hu = arr * slope + intercept
                inst_num = float(getattr(dcm, "InstanceNumber", 0))
                loc = float(getattr(dcm, "SliceLocation", inst_num))
                sop = str(getattr(dcm, "SOPInstanceUID", file.name))

                t_img = transform(hu).cpu()

                disp_img = apply_window(hu, 40, 80)
                disp_rgb = np.stack([disp_img] * 3, axis=-1)
                disp_pil = Image.fromarray((disp_rgb * 255).astype(np.uint8)).resize((224, 224))
                disp_uint8 = np.array(disp_pil, dtype=np.uint8)

                del arr, hu, disp_img, disp_rgb, disp_pil, dcm

                return {
                    "tensor": t_img,
                    "display_rgb": disp_uint8,
                    "slice_loc": loc,
                    "sop_uid": sop,
                    "name": file.name,
                    "is_dicom": True
                }
        except Exception:
            file.seek(0)

    # 2. Try Standard Image (PIL)
    try:
        file.seek(0)
        pil_img = Image.open(file).convert("RGB")
        disp_pil = pil_img.resize((224, 224))
        disp_uint8 = np.array(disp_pil, dtype=np.uint8)

        t_img = torch.from_numpy(disp_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
        t_img = rgb_normalize(t_img).cpu()

        del pil_img, disp_pil

        return {
            "tensor": t_img,
            "display_rgb": disp_uint8,
            "slice_loc": 0.0,
            "sop_uid": file.name,
            "name": file.name,
            "is_dicom": False
        }
    except Exception:
        file.seek(0)

    # 3. Try NumPy array (.npy)
    try:
        file.seek(0)
        raw_arr = np.load(file).astype(np.float32)
        if raw_arr.ndim == 2:
            t_img = transform(raw_arr).cpu()
            disp_img = apply_window(raw_arr, 40, 80)
            disp_rgb = np.stack([disp_img] * 3, axis=-1)
            disp_pil = Image.fromarray((disp_rgb * 255).astype(np.uint8)).resize((224, 224))
            disp_uint8 = np.array(disp_pil, dtype=np.uint8)
            del raw_arr, disp_img, disp_rgb, disp_pil
            return {
                "tensor": t_img,
                "display_rgb": disp_uint8,
                "slice_loc": 0.0,
                "sop_uid": file.name,
                "name": file.name,
                "is_dicom": True
            }
        elif raw_arr.ndim == 3 and raw_arr.shape[0] == 3:
            disp_rgb = np.transpose(raw_arr, (1, 2, 0))
            disp_pil = Image.fromarray((np.clip(disp_rgb, 0, 1) * 255).astype(np.uint8)).resize((224, 224))
            disp_uint8 = np.array(disp_pil, dtype=np.uint8)
            t_img = torch.from_numpy(disp_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
            t_img = rgb_normalize(t_img).cpu()
            del raw_arr, disp_rgb, disp_pil
            return {
                "tensor": t_img,
                "display_rgb": disp_uint8,
                "slice_loc": 0.0,
                "sop_uid": file.name,
                "name": file.name,
                "is_dicom": False
            }
    except Exception as e:
        raise ValueError(f"Could not parse file {file.name}: {e}")


if uploaded_files:
    # Load models on-demand only when files are uploaded
    try:
        d_model, m_model, d_arch, m_arch, device = load_models(str(diag_run_path), str(mon_run_path))
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()

    # Build unique cache key from selected models and uploaded files
    file_signature = f"{selected_mon_run}_{len(uploaded_files)}_{'_'.join(f.name for f in uploaded_files[:5])}"

    if st.session_state.get("current_file_signature") != file_signature:
        clear_session_cache()

        with st.spinner("Processing input files and computing model predictions..."):
            transform = Transform(train=False)
            rgb_normalize = T_v2.Compose([
                T_v2.Resize((224, 224), interpolation=T_v2.InterpolationMode.BILINEAR, antialias=True),
                T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            slices_data = []
            for f in uploaded_files:
                try:
                    item = process_file_to_slice(f, transform, rgb_normalize)
                    if item is not None:
                        slices_data.append(item)
                except Exception as e:
                    st.warning(f"Could not process {f.name}: {e}")

            if not slices_data:
                st.error("No valid CT slices found.")
                st.stop()

            # Sort spatially if all are DICOM
            if all(s.get("is_dicom", False) for s in slices_data):
                slices_data.sort(key=lambda x: x["slice_loc"])

            tensor_list = [s["tensor"] for s in slices_data]
            display_rgbs = [s["display_rgb"] for s in slices_data]

            # Clear temporary slice list
            del slices_data
            force_garbage_collect()

            # Keep all_tensors on CPU to avoid GPU VRAM bloat
            all_tensors = torch.stack(tensor_list)  # CPU tensor [N, 3, 224, 224]
            del tensor_list

            p_diag_list = []
            p_mon_list = []
            batch_size = 32

            with torch.inference_mode():
                for b_start in range(0, len(all_tensors), batch_size):
                    b_chunk = all_tensors[b_start:b_start + batch_size].to(device, non_blocking=True)
                    d_out = d_model(b_chunk)
                    m_out = m_model(b_chunk)
                    p_diag_list.extend(torch.sigmoid(d_out).cpu().numpy().reshape(-1).tolist())
                    p_mon_list.extend(torch.sigmoid(m_out).cpu().numpy().reshape(-1).tolist())
                    del b_chunk

            force_garbage_collect()

            p_diag_arr = np.array(p_diag_list)
            p_mon_arr = np.array(p_mon_list)
            s_dist_arr = compute_s_dist(p_diag_arr, [diag_slice_tau])

            st.session_state["current_file_signature"] = file_signature
            st.session_state["num_slices"] = len(display_rgbs)
            st.session_state["display_rgbs"] = display_rgbs
            st.session_state["all_tensors"] = all_tensors
            st.session_state["p_diag_arr"] = p_diag_arr
            st.session_state["p_mon_arr"] = p_mon_arr
            st.session_state["s_dist_arr"] = s_dist_arr

    # Load from session cache
    num_slices = st.session_state["num_slices"]
    display_rgbs = st.session_state["display_rgbs"]
    all_tensors = st.session_state["all_tensors"]
    p_diag_arr = st.session_state["p_diag_arr"]
    p_mon_arr = st.session_state["p_mon_arr"]
    s_dist_arr = st.session_state["s_dist_arr"]

    st.markdown("---")

    def render_gradcam_slice(idx):
        disp_uint8 = display_rgbs[idx]
        vis_base = disp_uint8.astype(np.float32) / 255.0

        if enable_gradcam:
            t_batch = all_tensors[idx:idx+1].to(device)
            d_overlay = compute_gradcam_overlay(d_model, d_arch, t_batch, vis_base)
            m_overlay = compute_gradcam_overlay(m_model, m_arch, t_batch, vis_base)
            del t_batch

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(disp_uint8, caption="Input CT Slice (Brain Window)", use_container_width=True)
            with col2:
                st.image(d_overlay, caption=f"Diagnostic Attention (ViT)\nProbability: {p_diag_arr[idx]:.4f}", use_container_width=True)
            with col3:
                st.image(m_overlay, caption=f"Visual Monitor Attention (DenseNet)\nFailure Risk: {p_mon_arr[idx]:.4f}", use_container_width=True)

            del d_overlay, m_overlay
        else:
            st.image(disp_uint8, caption="Input CT Slice (Brain Window)", use_container_width=True)

    if aggregation == "Slice-level":
        if num_slices > 1:
            slice_num = st.slider("Select Slice to Inspect", min_value=1, max_value=num_slices, value=1)
            active_idx = slice_num - 1
        else:
            active_idx = 0

        p_d = p_diag_arr[active_idx]
        p_m = p_mon_arr[active_idx]
        s_d = s_dist_arr[active_idx]
        is_pos = p_d >= diag_slice_tau

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("p_diag", f"{p_d:.4f}", f"Threshold: {diag_slice_tau:.4f}")
        k2.metric("Diagnosis", "Positive" if is_pos else "Negative")
        k3.metric("s_dist", f"{s_d:.4f}", f"Threshold: {0.5:.4f}")
        k4.metric("Monitor", f"{p_m:.4f}", f"Threshold: {mon_slice_tau:.4f}")

        st.markdown("### Visual Feature Attribution")
        render_gradcam_slice(active_idx)

    else: # Scan-level (Top-3 Mean)
        top_k = min(3, num_slices)
        dummy_scan_ids = np.zeros(num_slices)
        _, scan_diag_probs, _ = aggregate_to_scan_level(dummy_scan_ids, p_diag_arr, np.ones(num_slices), k=top_k)
        _, scan_mon_probs = aggregate_dual_pooling_to_scan_level(
            dummy_scan_ids, p_diag_arr, p_mon_arr, diag_threshold=diag_slice_tau, k=top_k
        )

        scan_p_d = float(scan_diag_probs[0])
        scan_p_m = float(scan_mon_probs[0])
        scan_s_d = float(compute_s_dist(np.array([scan_p_d]), [diag_scan_tau])[0])
        scan_is_pos = scan_p_d >= diag_scan_tau

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("p_diag", f"{scan_p_d:.4f}", f"Threshold: {diag_scan_tau:.4f}")
        k2.metric("Decision", "Positive" if scan_is_pos else "Negative")
        k3.metric("s_dist", f"{scan_s_d:.4f}", f"Threshold: {mon_scan_tau:.4f}")
        k4.metric("Monitor", f"{scan_p_m:.4f}", f"Threshold: {mon_scan_tau:.4f}")

        st.markdown("---")
        st.markdown(f"### Top {top_k} Predictive Slices")
        top_indices = np.argsort(p_diag_arr)[-top_k:][::-1]

        cols = st.columns(top_k)
        for rank, s_idx in enumerate(top_indices):
            with cols[rank]:
                st.markdown(f"**Rank #{rank+1} (Slice {s_idx+1})**")
                disp_uint8 = display_rgbs[s_idx]
                vis_base = disp_uint8.astype(np.float32) / 255.0

                if enable_gradcam:
                    t_batch = all_tensors[s_idx:s_idx+1].to(device)
                    d_overlay = compute_gradcam_overlay(d_model, d_arch, t_batch, vis_base)
                    del t_batch
                    st.image(d_overlay, caption=f"p_diag: {p_diag_arr[s_idx]:.4f} | p_mon: {p_mon_arr[s_idx]:.4f}", use_container_width=True)
                    del d_overlay
                else:
                    st.image(disp_uint8, caption=f"p_diag: {p_diag_arr[s_idx]:.4f} | p_mon: {p_mon_arr[s_idx]:.4f}", use_container_width=True)

        if num_slices > 1:
            st.markdown("---")
            st.markdown("### Scan Volume Explorer")
            sc_num = st.slider("Scrub through full CT volume", min_value=1, max_value=num_slices, value=1, key="scan_scrub")
            render_gradcam_slice(sc_num - 1)

    # Force memory cleanup after rendering the active slice
    force_garbage_collect()

else:
    # Initial state / files cleared:
    clear_session_cache()
    st.cache_resource.clear()
    force_garbage_collect()
    st.info("Upload one or multiple CT slice files (.dcm, .png, .jpg) to run inference.")
