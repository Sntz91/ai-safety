import gc
import ctypes
import yaml
import torch
from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import streamlit as st

from ai_safety.models.diagnostic import get_model as get_diag_model
from ai_safety.models.monitor import get_model as get_mon_model


def force_garbage_collect():
    """Forces Python GC, releases CUDA memory cache and IPC, and trims C heap fragmentation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


@st.cache_resource(max_entries=1)
def load_models(diag_dir_str, mon_dir_str):
    """Load the diagnostic and monitor models from run directories."""
    force_garbage_collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(Path(diag_dir_str) / "config.yaml", "r") as f:
        d_cfg = yaml.safe_load(f)
    with open(Path(mon_dir_str) / "config.yaml", "r") as f:
        m_cfg = yaml.safe_load(f)

    # Diagnostic model
    d_arch = d_cfg["model"]["model"]
    DiagClass = get_diag_model(d_arch)
    d_model = DiagClass(num_classes=1, **d_cfg["model"].get("params", {})).to(device)
    d_ckpt = torch.load(Path(diag_dir_str) / "best_model.pt", map_location=device, weights_only=False)
    d_model.load_state_dict(d_ckpt["model_state_dict"])
    d_model.eval()

    # Monitor model
    m_arch = m_cfg["model"]["model"]
    MonClass = get_mon_model(m_arch)
    m_model = MonClass(num_classes=1, **m_cfg["model"].get("params", {})).to(device)
    m_ckpt = torch.load(Path(mon_dir_str) / "best_model.pt", map_location=device, weights_only=False)
    m_model.load_state_dict(m_ckpt["model_state_dict"])
    m_model.eval()

    return d_model, m_model, d_arch, m_arch, device


def compute_gradcam_overlay(model, arch_name, input_tensor, vis_rgb):
    try:
        if arch_name == "vit":
            target_layers = [model.backbone.blocks[-1].norm1]

            def reshape_transform(tensor, height=14, width=14):
                result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                result = result.transpose(2, 3).transpose(1, 2)
                return result

            with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
                cam_map = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0, :]
                return show_cam_on_image(vis_rgb, cam_map, use_rgb=True)
        elif arch_name == "densenet":
            target_layers = [model.backbone.features[-1]]
            with GradCAM(model=model, target_layers=target_layers) as cam:
                cam_map = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0, :]
                return show_cam_on_image(vis_rgb, cam_map, use_rgb=True)
        else:
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                    with GradCAM(model=model, target_layers=[module]) as cam:
                        cam_map = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0, :]
                        return show_cam_on_image(vis_rgb, cam_map, use_rgb=True)
            raise ValueError(f"No suitable convolutional target layer found for {arch_name}")
    finally:
        # Zero out and release all parameter gradients created during the GradCAM backward pass
        model.zero_grad(set_to_none=True)


def require_runs_dir(run_dir, message=None):
    """Stop the app if a run directory does not exist or is empty."""
    if not run_dir.exists() or not list(run_dir.iterdir()):
        st.error(message or f"No runs found in {run_dir}.")
        st.stop()


def load_run_configs(mon_run_path):
    """Load the monitor config and resolve/load the linked diagnostic config.

    Args:
        mon_run_path: Path to the monitor run directory.

    Returns:
        tuple: (mon_cfg, mon_run_path, diag_cfg, diag_run_path)
    """
    mon_cfg_path = mon_run_path / "config.yaml"
    if not mon_cfg_path.exists():
        st.error(f"Config not found at {mon_cfg_path}.")
        st.stop()

    with open(mon_cfg_path, "r") as f:
        mon_cfg = yaml.safe_load(f)

    diag_run_path = Path(mon_cfg.get("diagnostic", {}).get("run_dir", "runs/diagnostic"))
    if not diag_run_path.exists():
        st.error(f"Diagnostic run path not found at {diag_run_path}.")
        st.stop()
    diag_cfg_path = diag_run_path / "config.yaml"
    if not diag_cfg_path.exists():
        st.error(f"Diagnostic run config not found at {diag_cfg_path}.")
        st.stop()

    with open(diag_cfg_path, "r") as f:
        diag_cfg = yaml.safe_load(f)

    return mon_cfg, mon_run_path, diag_cfg, diag_run_path
