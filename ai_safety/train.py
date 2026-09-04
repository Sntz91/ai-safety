import os
import yaml
import argparse
import shutil
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset

from ai_safety.data.transforms import Transform
from ai_safety.data.dataset import build_diagnostic_dataset, build_monitor_dataset
from ai_safety.utils.losses import get_loss
from ai_safety.utils.trainer import train_model, generate_predictions
from ai_safety.evaluation.thresholds import (
    get_thresholds_for_sensitivity,
    get_scan_thresholds_for_sensitivity,
)
from ai_safety.constants import DEFAULT_TARGET_SENSITIVITY

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--task", type=str, choices=['diagnostic', 'monitor'], required=True, help="Task to train")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(args.output_dir, "config.yaml"))

    device = cfg["training"]["device"]
    
    t_train = Transform(train=True)
    t_val = Transform(train=False)
    
    with open(os.path.join(args.output_dir, "transforms.txt"), "w") as f:
        f.write("--- Train Transform ---\n")
        f.write(str(t_train) + "\n\n")
        f.write("--- Val Transform ---\n")
        f.write(str(t_val) + "\n")

    # --- 1. Dataset Setup ---
    if args.task == 'diagnostic':
        binary = cfg["model"]["binary"]
        train_ds = build_diagnostic_dataset(cfg["data"]["train"], transform=t_train, binary=binary)
        val_ds = build_diagnostic_dataset(cfg["data"]["val"], transform=t_val, binary=binary)
        
        # Load decoupled diagnostic model
        from ai_safety.models.diagnostic import get_model
        Model = get_model(cfg["model"]["model"])
        label_cols = train_ds.datasets[0].label_cols if isinstance(train_ds, ConcatDataset) else train_ds.label_cols
        num_classes = 1 if binary else len(label_cols)
        
    else:
        binary = cfg["monitor"]["binary"]
        subtype = cfg["monitor"]["subtype"]
        diag_dir = cfg["diagnostic"]["run_dir"]
        
        thresh_path = Path(diag_dir) / "thresholds.yaml"
        if not thresh_path.exists():
            raise FileNotFoundError(f"Diagnostic thresholds not found at {thresh_path}! Run diagnostic train first.")
        with open(thresh_path, 'r') as f:
            thresholds = yaml.safe_load(f)
            
        diag_slice_thresholds = thresholds['slice']

        train_ds = build_monitor_dataset(cfg["data"]["train"], diag_dir, diag_slice_thresholds, transform=t_train, binary=binary, subtype=subtype)
        val_ds = build_monitor_dataset(cfg["data"]["val"], diag_dir, diag_slice_thresholds, transform=t_val, binary=binary, subtype=subtype)
        
        # Load decoupled monitor model
        from ai_safety.models.monitor import get_model
        Model = get_model(cfg["model"]["model"])
        sample_target = train_ds[0][1]
        num_classes = sample_target.shape[0] if sample_target.ndim > 0 else 1

    # --- 2. Model, Loss, Optimizer ---
    model_kwargs = cfg["model"].get("params", {})
    model = Model(
        num_classes=num_classes, 
        **model_kwargs
    ).to(device)

    use_pos_weights = cfg["loss"]["use_pos_weights"]
    pos_weight = train_ds.compute_pos_weights().to(device) if use_pos_weights else None

    Loss = get_loss(cfg["loss"]["name"])
    loss_kwargs = cfg["loss"].get("params", {})
    criterion = Loss(pos_weight=pos_weight, **loss_kwargs) if use_pos_weights else Loss(**loss_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=cfg["optimizer"].get("lr_min", 1.0e-06),
    )

    # --- 3. Training Loop ---
    train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=args.output_dir,
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        eval_metric=cfg["training"]["eval_metric"],
    )
    print('Finished training.')

    # --- 4. Post-Training (Thresholds & Inference) ---
    best_ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    print(f'Computing validation thresholds for {args.task} model...')
    if isinstance(val_ds, ConcatDataset):
        for ds in val_ds.datasets: ds.return_sopuid = True
    else:
        val_ds.return_sopuid = True

    val_preds = generate_predictions(
        model=model, dataset=val_ds, dataset_name='val', device=device,
        batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"]
    )
    
    slice_thresholds = get_thresholds_for_sensitivity(val_preds, target_sens=DEFAULT_TARGET_SENSITIVITY)
    scan_thresholds = get_scan_thresholds_for_sensitivity(val_preds, target_sens=DEFAULT_TARGET_SENSITIVITY, k=3)
    thresholds = {'slice': slice_thresholds, 'scan': scan_thresholds}
        
    thresh_path = os.path.join(args.output_dir, "thresholds.yaml")
    with open(thresh_path, "w") as f:
        yaml.dump(thresholds, f)
    print(f"Saved thresholds: {thresholds} to {thresh_path}")

    # Generate prediction CSVs
    predict_splits = cfg["data"]["predict"]
    if predict_splits:
        print('Start generating prediction files...')
        for item in predict_splits:
            if args.task == 'diagnostic':
                pred_ds = build_diagnostic_dataset(item, transform=t_val, binary=binary, return_sopuid=True)
            else:
                pred_ds = build_monitor_dataset([item], diag_dir, diag_slice_thresholds, transform=t_val, binary=binary, subtype=subtype, return_sopuid=True)
                
            out_csv = os.path.join(args.output_dir, f"predictions-{item['name']}.csv")
            preds = generate_predictions(
                model=model, dataset=pred_ds, dataset_name=item["name"], device=device,
                batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"]
            )
            preds.to_csv(out_csv)
        print('Finished.')

if __name__ == '__main__':
    main()
