import yaml
from pathlib import Path


def resolve_threshold(thresh_dict, level, idx):
    """Resolve a single threshold value by key and index. Fails loudly if missing."""
    if level not in thresh_dict:
        raise KeyError(
            f"Threshold key '{level}' not found in thresholds dict. "
            f"Available keys: {list(thresh_dict.keys())}"
        )
    values = thresh_dict[level]
    if idx >= len(values):
        raise IndexError(
            f"Threshold index {idx} out of range for key '{level}' "
            f"({len(values)} entries available)."
        )
    return float(values[idx])


def load_thresholds_from_run(run_dir):
    """Load the thresholds.yaml file from a run directory. Fails loudly if missing."""
    th_path = Path(run_dir) / "thresholds.yaml"
    if not th_path.exists():
        raise FileNotFoundError(f"thresholds.yaml not found at {th_path}")
    with open(th_path, "r") as f:
        return yaml.safe_load(f)


def is_valid_scan_id(scan_id):
    """Return whether a scan ID is a usable value (not None / 'None' / 'nan')."""
    return scan_id is not None and str(scan_id) != "None" and str(scan_id) != "nan"


def extract_subtypes(columns, suffix="_diag"):
    """Extract subtype names from label columns, e.g. 'label_epidural_diag' -> 'epidural'."""
    return [
        c.replace("label_", "").replace(suffix, "")
        for c in columns
        if c.startswith("label_") and c.endswith(suffix)
    ]


def format_metric(d, show_ci=True):
    """Format a metric dict as a string, optionally including its confidence interval."""
    if d is None or d.get("value") is None:
        return "N/A"
    s = f"{d['value']:.4f}"
    if show_ci and d.get("ci_lower") is not None:
        s += f" [{d['ci_lower']:.4f}, {d['ci_upper']:.4f}]"
    return s
