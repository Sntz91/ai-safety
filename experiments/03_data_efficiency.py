""" Monitor Training Data-Efficiency Experiment. 

Trains and evaluates the monitor at multiple training-data 
fractions across several random seeds.

"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

from ai_safety.data.transforms import Transform
from ai_safety.data.dataset import build_monitor_dataset
from ai_safety.utils.losses import get_loss
from ai_safety.utils.trainer import train_model, generate_predictions
from ai_safety.evaluation.metrics.monitor.budgets import evaluate as evaluate_budgets

DEFAULT_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_EVAL_SPLIT = "rsna-no-match-sinoct-test"
CSV_COLUMNS = ["fraction", "seed", "train_samples", "fail_auroc", "fail_auprc",
               "recall_at_0.1", "recall_at_0.2"]


def load_thresholds(diag_dir):
    thresh_path = Path(diag_dir) / "thresholds.yaml"
    if not thresh_path.exists():
        raise FileNotFoundError(f"Diagnostic thresholds not found at {thresh_path}! Run diagnostic train first.")
    with open(thresh_path, "r") as f:
        thresholds = yaml.safe_load(f)
    return thresholds["slice"], thresholds.get("scan", [0.5] * len(thresholds["slice"]))


def evaluate_fraction(df_diag, df_mon, diag_tau):
    """Compute failure AUROC / AUPRC / recall@budget for the monitor on a merged frame.

    Mirrors the legacy data-efficiency methodology: the monitor's failure scores are
    ranked against the diagnostic error ground truth derived from the locked thresholds.
    """
    key_col = "sop_uid" if "sop_uid" in df_diag.columns and "sop_uid" in df_mon.columns else df_diag.columns[0]
    merged = df_diag.merge(df_mon, on=key_col, suffixes=("_diag", "_mon"))

    d_prob_col = "prob_ich_diag" if "prob_ich_diag" in merged.columns else merged.columns[-4]
    d_gt_col = "label_ich_diag" if "label_ich_diag" in merged.columns else merged.columns[-3]
    m_prob_col = "prob_ich_mon" if "prob_ich_mon" in merged.columns else merged.columns[-1]

    d_probs = merged[d_prob_col].values.astype(float)
    d_gt = merged[d_gt_col].values.astype(int)
    m_probs = merged[m_prob_col].values.astype(float)

    d_pred = (d_probs >= diag_tau).astype(int)
    y_error = (d_pred != d_gt).astype(int)

    if len(np.unique(y_error)) < 2:
        fail_auroc = float(y_error.sum() / len(y_error))
        fail_auprc = 0.0
    else:
        fail_auroc = float(roc_auc_score(y_error, m_probs))
        fail_auprc = float(average_precision_score(y_error, m_probs))

    budgets = evaluate_budgets(m_probs, y_error)
    return {
        "fail_auroc": fail_auroc,
        "fail_auprc": fail_auprc,
        "recall_at_0.1": budgets["recall_at_0.1"],
        "recall_at_0.2": budgets["recall_at_0.2"],
    }


def run_fraction(config, config_path, diag_dir, diag_tau, frac, seed, output_dir, device, eval_split):
    """Train one monitor model at a given data fraction and seed, then evaluate it."""
    run_dir = Path(output_dir) / f"frac_{int(round(frac * 100)):03d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.yaml")

    binary = config["monitor"]["binary"]
    subtype = config["monitor"]["subtype"]
    num_workers = config["training"]["num_workers"]

    t_train = Transform(train=True)
    t_val = Transform(train=False)

    train_ds = build_monitor_dataset(config["data"]["train"], diag_dir, diag_tau, t_train, binary, subtype)
    val_ds = build_monitor_dataset(config["data"]["val"], diag_dir, diag_tau, t_val, binary, subtype)

    from ai_safety.models.monitor import get_model
    Model = get_model(config["model"]["model"])
    sample_target = train_ds[0][1]
    num_classes = sample_target.shape[0] if sample_target.ndim > 0 else 1

    model_kwargs = config["model"].get("params", {})
    model = Model(num_classes=num_classes, **model_kwargs).to(device)

    use_pos_weights = config["loss"]["use_pos_weights"]
    pos_weight = train_ds.compute_pos_weights().to(device) if use_pos_weights else None
    Loss = get_loss(config["loss"]["name"])
    loss_kwargs = config["loss"].get("params", {})
    criterion = Loss(pos_weight=pos_weight, **loss_kwargs) if use_pos_weights else Loss(**loss_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"]["lr"],
                                  weight_decay=config["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=config["optimizer"].get("lr_min", 1.0e-06))

    train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=str(run_dir),
        epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        batch_size=config["training"]["batch_size"],
        num_workers=num_workers,
        eval_metric=config["training"]["eval_metric"],
        sample_fraction=frac,
        subsample_seed=seed,
    )

    # Predict on eval split
    pred_ds = build_monitor_dataset(
        [{"name": eval_split}], diag_dir, diag_tau, t_val, binary, subtype, return_sopuid=True
    )
    preds = generate_predictions(
        model=model, dataset=pred_ds, dataset_name=eval_split, device=device,
        batch_size=config["training"]["batch_size"], num_workers=num_workers,
    )
    preds.to_csv(run_dir / f"predictions-{eval_split}.csv")

    # Load diagnostic predictions to combine
    diag_pred = pd.read_csv(Path(diag_dir) / f"predictions-{eval_split}.csv")
    mon_pred = preds.to_dataframe()
    res = evaluate_fraction(diag_pred, mon_pred, diag_tau)

    # Track actual train sample count (from the prediction index used by trainer)
    try:
        n_train = len(train_ds.records)
    except Exception:
        n_train = len(train_ds)
    return res, n_train


def main():
    parser = argparse.ArgumentParser(description="Monitor Training Data-Efficiency Experiment")
    parser.add_argument("--config", type=str, default="configs/monitor/base.yaml", help="Monitor config YAML")
    parser.add_argument("--output-dir", type=str, required=True, help="Output dir for all fraction/seed runs + results.csv")
    parser.add_argument("--diag-run-dir", type=str, default=None, help="Diagnostic run dir (defaults to config['diagnostic']['run_dir'])")
    parser.add_argument("--eval-split", type=str, default=DEFAULT_EVAL_SPLIT)
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    diag_dir = Path(args.diag_run_dir or config["diagnostic"]["run_dir"])
    diag_slice_tau, diag_scan_tau = load_thresholds(diag_dir)
    diag_tau = diag_slice_tau[0] if diag_slice_tau else 0.5

    csv_path = output_dir / "results.csv"
    if not csv_path.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)

    for frac in args.fractions:
        for seed in args.seeds:
            res, n_train = run_fraction(
                config, args.config, diag_dir, diag_tau, frac, seed, output_dir, args.device, args.eval_split
            )
            row = {"fraction": frac, "seed": seed, "train_samples": n_train, **res}
            df = pd.read_csv(csv_path)
            df = df[~((df["fraction"] == frac) & (df["seed"] == seed))]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"Logged [{row['fraction']} | {row['seed']}] AUROC={row['fail_auroc']:.4f} R@20={row['recall_at_0.2']:.4f}")

    print(f"\nData-efficiency experiments complete. Results at {csv_path}")


if __name__ == "__main__":
    main()
