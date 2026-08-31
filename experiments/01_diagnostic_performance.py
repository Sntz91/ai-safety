import argparse
import json
import yaml
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_safety.evaluation import metrics, discover_metrics, evaluate_diagnostic_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diagnostic Model Performance")
    parser.add_argument("--run_id", type=str, required=True, help="Diagnostic run ID")
    parser.add_argument("--bootstraps", type=int, default=100, help="Number of bootstrap iterations")
    args = parser.parse_args()

    run_dir = Path("runs/diagnostic") / args.run_id
    out_dir = Path("experiments/outputs/diagnostic") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "thresholds.yaml", "r") as f:
        thresholds = yaml.safe_load(f)

    slice_thresholds = thresholds["slice"]
    scan_thresholds = thresholds["scan"]

    metric_funcs = discover_metrics(metrics.shared, metrics.diagnostic)
    print(f"Discovered metrics: {list(metric_funcs.keys())}")

    metrics_out = {}
    curves_out = {}

    for p_file in sorted(run_dir.glob("predictions-*.csv")):
        df = pd.read_csv(p_file)
        dataset_name = p_file.stem.replace("predictions-", "")
        print(f"Processing {dataset_name} ({len(df)} samples)...")

        ds_metrics, ds_curves = evaluate_diagnostic_dataset(
            df=df,
            slice_thresholds=slice_thresholds,
            scan_thresholds=scan_thresholds,
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
