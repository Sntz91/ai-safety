import argparse
import json
import yaml
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_safety.evaluation.stats import discover_metrics, run_evaluation
from ai_safety.evaluation.risk import evaluate_risk
from ai_safety.models.monitor.threshold_distance import compute_s_dist

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
    diag_dir = Path(mon_cfg['diagnostic']['run_dir'])

    # Load diagnostic thresholds (needed for dual pooling)
    with open(diag_dir / "thresholds.yaml", "r") as f:
        diag_thresh = yaml.safe_load(f)
    diag_slice_thresholds = diag_thresh['slice']

    # Load monitor thresholds
    with open(run_dir / "thresholds.yaml", "r") as f:
        mon_thresh = yaml.safe_load(f)
    slice_thresholds = mon_thresh['slice']
    scan_thresholds = mon_thresh['scan']
    
    metrics_out = {}
    curves_out = {}

    import ai_safety.evaluation.shared
    import ai_safety.evaluation.monitor
    # Auto-discover metrics from shared and monitor folders
    metric_funcs = discover_metrics(ai_safety.evaluation.shared, ai_safety.evaluation.monitor)
    print(f"Discovered metrics: {list(metric_funcs.keys())}")

    for p_file in run_dir.glob("predictions-*.csv"):
        df_mon = pd.read_csv(p_file)
        
        diag_file = diag_dir / p_file.name
        if not diag_file.exists():
            print(f"Warning: {diag_file} not found! Skipping {p_file.name}")
            continue
        df_diag = pd.read_csv(diag_file)
        
        # Merge to align diagnostic and monitor predictions
        df = df_mon.merge(df_diag, on=['sop_uid', 'series_id', 'dataset'], suffixes=('_mon', '_diag'))
        
        dataset_name = p_file.stem.replace('predictions-', '')
        print(f"Processing {dataset_name} ({len(df)} samples)...")
        
        label_cols = sorted([c for c in df_mon.columns if c.startswith('label_')])
        
        ds_metrics = {}
        ds_curves = {}
        
        for idx, l_col in enumerate(label_cols):
            subtype = l_col.replace('label_', '')
            mon_risk = df[f'prob_{subtype}_mon'].values   # Monitor predicted error prob

            diag_t_slice = diag_slice_thresholds[idx] if idx < len(diag_slice_thresholds) else 0.5
            diag_t_scan = diag_thresh.get('scan', [0.5])[idx] if 'scan' in diag_thresh and idx < len(diag_thresh['scan']) else 0.5
            t_slice = slice_thresholds[idx] if idx < len(slice_thresholds) else 0.5
            t_scan = scan_thresholds[idx] if idx < len(scan_thresholds) else 0.5

            # Trained monitor risk
            m_sl, m_slc, m_sc, m_scc = evaluate_risk(
                df, subtype, mon_risk, diag_t_slice, diag_t_scan, t_slice, t_scan, args.bootstraps, metric_funcs
            )

            # Threshold-distance baseline: s_dist(diag_prob, tau_diag)
            sdist_risk = compute_s_dist(df[f'prob_{subtype}_diag'].values, diag_t_slice)
            t_sl, t_slc, t_sc, t_scc = evaluate_risk(
                df, subtype, sdist_risk, diag_t_slice, diag_t_scan, t_slice, t_scan, args.bootstraps, metric_funcs
            )

            ds_metrics[subtype] = {
                "slice_level": {"monitor": m_sl, "threshold_distance": t_sl},
                "scan_level": {"monitor": m_sc, "threshold_distance": t_sc},
            }
            ds_curves[subtype] = {
                "slice_level": {"monitor": m_slc, "threshold_distance": t_slc},
                "scan_level": {"monitor": m_scc, "threshold_distance": t_scc},
            }
            
        metrics_out[dataset_name] = ds_metrics
        curves_out[dataset_name] = ds_curves

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=4)
        
    with open(out_dir / "curves.json", "w") as f:
        json.dump(curves_out, f)

if __name__ == "__main__":
    main()
