import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai_safety.utils.curves import get_roc_curve, get_pr_curve
from ai_safety.utils.aggregation import aggregate_to_scan_level
from ai_safety.evaluation.stats import discover_metrics, run_evaluation

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
        
    slice_thresholds = thresholds['slice']
    scan_thresholds = thresholds['scan']
    
    metrics_out = {}
    curves_out = {}

    import ai_safety.evaluation.shared
    import ai_safety.evaluation.diagnostic
    # Auto-discover metrics from shared and diagnostic folders
    metric_funcs = discover_metrics(ai_safety.evaluation.shared, ai_safety.evaluation.diagnostic)
    print(f"Discovered metrics: {list(metric_funcs.keys())}")

    for p_file in run_dir.glob("predictions-*.csv"):
        df = pd.read_csv(p_file)
        dataset_name = p_file.stem.replace('predictions-', '')
        print(f"Processing {dataset_name} ({len(df)} samples)...")
        
        label_cols = sorted([c for c in df.columns if c.startswith('label_')])
        prob_cols = sorted([c for c in df.columns if c.startswith('prob_')])
        
        ds_metrics = {}
        ds_curves = {}
        
        for idx, (l_col, p_col) in enumerate(zip(label_cols, prob_cols)):
            subtype = l_col.replace('label_', '')
            y_true = df[l_col].values
            y_prob = df[p_col].values
            t_slice = slice_thresholds[idx] if idx < len(slice_thresholds) else 0.5
            
            # Slice-Level (LOCKED 85% Sens)
            sl_metrics, sl_curves = run_evaluation(y_true, y_prob, t_slice, "LOCKED (85% Sens)", args.bootstraps, metric_funcs)
            
            # Slice-Level (OPTIMAL 85% Sens)
            fpr_arr, tpr_arr, th_arr = get_roc_curve(y_prob, y_true)
            valid_idx = np.where(tpr_arr >= 0.85)[0]
            dyn_t_slice_sens = float(np.clip(th_arr[valid_idx[0]], 0.0, 1.0)) if len(valid_idx) > 0 else 0.5
            sl_opt_metrics_sens, _ = run_evaluation(y_true, y_prob, dyn_t_slice_sens, "OPTIMAL (85% Sens)", args.bootstraps, metric_funcs)
            sl_metrics['discrete'].update(sl_opt_metrics_sens['discrete'])
            
            # Slice-Level (OPTIMAL Max F1)
            prec_arr, rec_arr, th_arr_pr = get_pr_curve(y_prob, y_true)
            f1_arr = np.divide(2 * prec_arr * rec_arr, prec_arr + rec_arr, out=np.zeros_like(prec_arr), where=(prec_arr + rec_arr) != 0)
            idx_best = np.argmax(f1_arr)
            dyn_t_slice_f1 = float(np.clip(th_arr_pr[idx_best], 0.0, 1.0)) if idx_best < len(th_arr_pr) else 0.5
            sl_opt_metrics_f1, _ = run_evaluation(y_true, y_prob, dyn_t_slice_f1, "OPTIMAL (Max F1)", args.bootstraps, metric_funcs)
            sl_metrics['discrete'].update(sl_opt_metrics_f1['discrete'])
            
            # Scan-Level
            if 'series_id' in df.columns:
                series_ids = df['series_id'].values
                _, scan_prob, scan_true = aggregate_to_scan_level(series_ids, y_prob, y_true, k=3)
                
                # Scan-Level (LOCKED 85% Sens)
                t_scan = scan_thresholds[idx] if idx < len(scan_thresholds) else 0.5
                sc_metrics, sc_curves = run_evaluation(scan_true, scan_prob, t_scan, "LOCKED (85% Sens)", args.bootstraps, metric_funcs)
                
                # Scan-Level (OPTIMAL 85% Sens)
                fpr_arr, tpr_arr, th_arr = get_roc_curve(scan_prob, scan_true)
                valid_idx = np.where(tpr_arr >= 0.85)[0]
                dyn_t_scan_sens = float(np.clip(th_arr[valid_idx[0]], 0.0, 1.0)) if len(valid_idx) > 0 else 0.5
                sc_opt_metrics_sens, _ = run_evaluation(scan_true, scan_prob, dyn_t_scan_sens, "OPTIMAL (85% Sens)", args.bootstraps, metric_funcs)
                sc_metrics['discrete'].update(sc_opt_metrics_sens['discrete'])
                
                # Scan-Level (OPTIMAL Max F1)
                prec_arr, rec_arr, th_arr_pr = get_pr_curve(scan_prob, scan_true)
                f1_arr = np.divide(2 * prec_arr * rec_arr, prec_arr + rec_arr, out=np.zeros_like(prec_arr), where=(prec_arr + rec_arr) != 0)
                idx_best = np.argmax(f1_arr)
                dyn_t_scan_f1 = float(np.clip(th_arr_pr[idx_best], 0.0, 1.0)) if idx_best < len(th_arr_pr) else 0.5
                sc_opt_metrics_f1, _ = run_evaluation(scan_true, scan_prob, dyn_t_scan_f1, "OPTIMAL (Max F1)", args.bootstraps, metric_funcs)
                sc_metrics['discrete'].update(sc_opt_metrics_f1['discrete'])
                
            else:
                sc_metrics, sc_curves = {}, {}

            ds_metrics[subtype] = {"slice_level": sl_metrics, "scan_level": sc_metrics}
            ds_curves[subtype] = {"slice_level": sl_curves, "scan_level": sc_curves}
            
        metrics_out[dataset_name] = ds_metrics
        curves_out[dataset_name] = ds_curves

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=4)
        
    with open(out_dir / "curves.json", "w") as f:
        json.dump(curves_out, f)

if __name__ == "__main__":
    main()
