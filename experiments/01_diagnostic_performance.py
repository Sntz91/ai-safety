import argparse
from pathlib import Path

from ai_safety.evaluation import metrics, evaluate_diagnostic_dataset, evaluate_run
from ai_safety.utils.helpers import load_thresholds_from_run


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diagnostic Model Performance")
    parser.add_argument("--run_id", type=str, required=True, help="Diagnostic run ID")
    parser.add_argument("--bootstraps", type=int, default=100, help="Number of bootstrap iterations")
    args = parser.parse_args()

    run_dir = Path("runs/diagnostic") / args.run_id
    out_dir = Path("experiments/outputs/diagnostic") / args.run_id

    thresholds = load_thresholds_from_run(run_dir)

    evaluate_run(
        run_dir=run_dir,
        out_dir=out_dir,
        metric_packages=[metrics.shared, metrics.diagnostic],
        evaluate_fn=evaluate_diagnostic_dataset,
        slice_thresholds=thresholds["slice"],
        scan_thresholds=thresholds["scan"],
        bootstraps=args.bootstraps,
    )


if __name__ == "__main__":
    main()
