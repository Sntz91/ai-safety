import argparse
import json
import yaml
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_safety.evaluation import metrics, discover_metrics, evaluate_monitor_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate Monitor Model Performance")
    parser.add_argument("--run_id", type=str, required=True, help="Monitor run ID")
    parser.add_argument("--bootstraps", type=int, default=100, help="Number of bootstrap iterations")
    args = parser.parse_args()

    run_dir = Path("runs/monitor") / args.run_id
    out_dir = Path("experiments/outputs/monitor") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "r") as f:
        mon_cfg = yaml.safe_load(f)
    diag_dir = Path(mon_cfg["diagnostic"]["run_dir"])

    # Load diagnostic thresholds (needed for dual pooling)
    with open(diag_dir / "thresholds.yaml", "r") as f:
        diag_thresh = yaml.safe_load(f)
    diag_slice_thresholds = diag_thresh["slice"]
    diag_scan_thresholds = diag_thresh.get("scan", [0.5] * len(diag_slice_thresholds))

    # Load monitor thresholds
    with open(run_dir / "thresholds.yaml", "r") as f:
        mon_thresh = yaml.safe_load(f)
    slice_thresholds = mon_thresh["slice"]
    scan_thresholds = mon_thresh["scan"]

    metric_funcs = discover_metrics(metrics.shared, metrics.monitor)
    print("Discovered metrics:", list(metric_funcs.keys()))

    metrics_out = {}
    curves_out = {}

    pred_files = list(run_dir.glob("predictions-*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {run_dir}")

    for p_file in pred_files:
        df_mon = pd.read_csv(p_file)

        diag_file = diag_dir / p_file.name
        if not diag_file.exists():
            print(f"Warning: {diag_file} not found! Skipping {p_file.name}")
            continue
        df_diag = pd.read_csv(diag_file)

        # Merge to align diagnostic and monitor predictions
        df = df_mon.merge(df_diag, on=["sop_uid", "series_id", "dataset"], suffixes=("_mon", "_diag"))

        dataset_name = p_file.stem.replace("predictions-", "")
        print(f"Processing {dataset_name} ({len(df)} samples)...")

        ds_metrics, ds_curves = evaluate_monitor_dataset(
            df=df,
            diag_slice_thresholds=diag_slice_thresholds,
            diag_scan_thresholds=diag_scan_thresholds,
            mon_slice_thresholds=slice_thresholds,
            mon_scan_thresholds=scan_thresholds,
            bootstraps=args.bootstraps,
            metric_funcs=metric_funcs,
        )

        metrics_out[dataset_name] = ds_metrics
        curves_out[dataset_name] = ds_curves

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=4)

    with open(out_dir / "curves.json", "w") as f:
        json.dump(curves_out, f)


if __name__ == "__main__":
    main()
