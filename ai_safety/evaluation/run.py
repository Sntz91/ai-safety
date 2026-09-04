import json
import pandas as pd
from pathlib import Path

from .registry import discover_metrics


def evaluate_run(run_dir, out_dir, metric_packages, evaluate_fn, bootstraps=100, preprocess_fn=None, **kwargs):
    """Evaluate all `predictions-*.csv` files in a run directory and write results as JSON.

    Discovers metric modules, iterates over the `predictions-*.csv` files in `run_dir`,
    calls `evaluate_fn` for each dataset split, and writes the aggregated metrics and curves
    to `out_dir` as `metrics.json` and `curves.json`.

    Args:
        run_dir: Path to the run directory containing prediction CSVs.
        out_dir: Path where metrics.json and curves.json will be written.
        metric_packages: Iterable of packages to pass to discover_metrics.
        evaluate_fn: Callable(df, metric_funcs, bootstraps, **kwargs)
            returning (ds_metrics, ds_curves). Thresholds and any other per-run
            settings are forwarded via **kwargs.
        bootstraps: Number of bootstrap iterations.
        preprocess_fn: Optional Callable(df, p_file) -> df applied to each prediction
            frame before evaluation (e.g., to merge with another run's predictions).
        **kwargs: Extra keyword arguments forwarded to evaluate_fn.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_funcs = discover_metrics(*metric_packages)
    print(f"Discovered metrics: {list(metric_funcs.keys())}")

    metrics_out = {}
    curves_out = {}

    pred_files = sorted(run_dir.glob("predictions-*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {run_dir}")

    for p_file in pred_files:
        df = pd.read_csv(p_file)
        if preprocess_fn is not None:
            df = preprocess_fn(df, p_file)
        dataset_name = p_file.stem.replace("predictions-", "")
        print(f"Processing {dataset_name} ({len(df)} samples)...")

        ds_metrics, ds_curves = evaluate_fn(
            df=df,
            metric_funcs=metric_funcs,
            bootstraps=bootstraps,
            **kwargs,
        )
        metrics_out[dataset_name] = ds_metrics
        curves_out[dataset_name] = ds_curves

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=4)

    with open(out_dir / "curves.json", "w") as f:
        json.dump(curves_out, f)

    print(f"Results written to {out_dir}")
