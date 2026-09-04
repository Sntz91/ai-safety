import argparse
import yaml
import pandas as pd
from pathlib import Path

from ai_safety.evaluation import metrics, evaluate_monitor_dataset, evaluate_run
from ai_safety.utils.helpers import load_thresholds_from_run


def main():
    parser = argparse.ArgumentParser(description="Evaluate Monitor Model Performance")
    parser.add_argument("--run_id", type=str, required=True, help="Monitor run ID")
    parser.add_argument("--bootstraps", type=int, default=100, help="Number of bootstrap iterations")
    args = parser.parse_args()

    run_dir = Path("runs/monitor") / args.run_id
    out_dir = Path("experiments/outputs/monitor") / args.run_id

    with open(run_dir / "config.yaml", "r") as f:
        mon_cfg = yaml.safe_load(f)
    diag_dir = Path(mon_cfg["diagnostic"]["run_dir"])

    # Diagnostic thresholds (needed for dual pooling)
    diag_thresh = load_thresholds_from_run(diag_dir)
    diag_slice_thresholds = diag_thresh["slice"]
    diag_scan_thresholds = diag_thresh["scan"]

    # Monitor thresholds
    mon_thresh = load_thresholds_from_run(run_dir)
    mon_slice_thresholds = mon_thresh["slice"]
    mon_scan_thresholds = mon_thresh["scan"]

    def merge_with_diagnostic(df_mon, p_file):
        diag_file = diag_dir / p_file.name
        if not diag_file.exists():
            raise FileNotFoundError(f"Diagnostic prediction file not found: {diag_file}")
        df_diag = pd.read_csv(diag_file)
        return df_mon.merge(df_diag, on=["sop_uid", "series_id", "dataset"], suffixes=("_mon", "_diag"))

    evaluate_run(
        run_dir=run_dir,
        out_dir=out_dir,
        metric_packages=[metrics.shared, metrics.monitor],
        evaluate_fn=evaluate_monitor_dataset,
        diag_slice_thresholds=diag_slice_thresholds,
        diag_scan_thresholds=diag_scan_thresholds,
        mon_slice_thresholds=mon_slice_thresholds,
        mon_scan_thresholds=mon_scan_thresholds,
        bootstraps=args.bootstraps,
        preprocess_fn=merge_with_diagnostic,
    )


if __name__ == "__main__":
    main()
