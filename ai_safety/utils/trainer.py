import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchmetrics import AUROC, AveragePrecision
from sklearn.model_selection import StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def subsample_dataset(dataset, fraction=1.0, seed=42, stratify=True):
    """ Subsamples a dataset to a fraction while preserving class prevalence. """
    if fraction >= 1.0:
        return dataset

    n_total = len(dataset)
    X = np.zeros(n_total)
    y = dataset.get_labels()

    # 2. Multi-Label Stratification (Preserves prevalence of all individual subtypes)
    if stratify and y.shape[1] > 1: # subtypes
        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        _, indices = next(splitter.split(X, y))
    elif stratify: # binary
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        _, indices = next(splitter.split(X, y.flatten()))
    else: # no stratification
        n_sample = max(1, int(round(fraction * n_total)))
        indices = np.random.default_rng(seed).choice(n_total, size=n_sample, replace=False)

    return Subset(dataset, indices.tolist())


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_cuda=True):
    """ Performs one training epoch. """
    model.train()
    total_loss, steps = 0.0, 0
    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_cuda):
            logits = model(images)
            loss = criterion(logits, labels.float())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, steps)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_cuda=True):
    """Evaluates model, returning mean loss, probabilities, and targets."""
    model.eval()
    total_loss, steps = 0.0, 0
    probs_list, targets_list = [], []

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)

        with torch.amp.autocast("cuda", enabled=use_cuda):
            logits = model(images)
            loss = criterion(logits, labels.float())
            probs = torch.sigmoid(logits)

        probs_list.append(probs.cpu())
        targets_list.append(labels.cpu())
        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps), torch.cat(probs_list), torch.cat(targets_list)


def compute_metrics(val_probs: torch.Tensor, val_targets: torch.Tensor):
    """Computes binary or multi-label AUROC and PR-AUC."""
    n_classes = val_probs.shape[1] if val_probs.ndim > 1 else 1
    if n_classes == 1:
        p, t = val_probs.flatten(), val_targets.flatten().int()
        auroc = AUROC(task="binary")(p, t).item()
        pr_auc = AveragePrecision(task="binary")(p, t).item()
    else:
        t = val_targets.int()
        auroc = AUROC(task="multilabel", num_labels=n_classes, average="macro")(val_probs, t).item()
        pr_auc = AveragePrecision(task="multilabel", num_labels=n_classes, average="macro")(val_probs, t).item()
    return auroc, pr_auc


def train_model(
    model,
    train_dataset,
    val_dataset,
    criterion,
    optimizer,
    scheduler=None,
    epochs: int = 10,
    patience: int = 5,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    device: str = "cuda",
    output_dir: str = "runs/exp1",
    eval_metric: str = "auroc",
    sample_fraction: float = 1.0,
    subsample_seed: int = 42,
    sampler=None,
):
    """ Training pipeline with subsampling, early stopping, and checkpointing. """
    os.makedirs(output_dir, exist_ok=True)
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    # 1. Subsample dataset if fraction < 1.0 (RQ2 Data Efficiency)
    if sample_fraction < 1.0:
        train_dataset = subsample_dataset(train_dataset, fraction=sample_fraction, seed=subsample_seed)
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    best_metric = -1.0 if eval_metric in ("auroc", "pr_auc") else float("inf")
    patience_counter = 0
    curve_path = os.path.join(output_dir, "training-curve.csv")
    with open(curve_path, "w") as f:
        f.write("epoch,lr,train_loss,val_loss,val_auroc,val_pr_auc\n")

    print(f"Start training: {epochs} max epochs, patience={patience}, metric={eval_metric}, device={device}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_cuda)
        val_loss, val_probs, val_targets = evaluate(model, val_loader, criterion, device, use_cuda)
        val_auroc, val_pr = compute_metrics(val_probs, val_targets)

        if scheduler:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | LR: {lr:.2e} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Val AUROC: {val_auroc:.4f} | Val PR-AUC: {val_pr:.4f}"
        )
        with open(curve_path, "a") as f:
            f.write(f"{epoch+1},{lr:.6e},{train_loss:.6f},{val_loss:.6f},{val_auroc:.6f},{val_pr:.6f}\n")

        # Early stopping logic
        current_metric = {"auroc": val_auroc, "pr_auc": val_pr, "loss": val_loss}[eval_metric]
        improved = (current_metric > best_metric) if eval_metric in('auroc', 'pr_auc') else (current_metric < best_metric)

        if improved:
            best_metric = current_metric
            patience_counter = 0
            ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "val_pr_auc": val_pr,
            }, ckpt_path)
            print(f'--> Saved best model ({eval_metric}={best_metric:.4f}) to {ckpt_path}')
        else:
            patience_counter += 1
            print(f'(Metric did not improve: {patience_counter}/{patience})')
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}.')
                break

    print(f'Training complete. Best {eval_metric}: {best_metric:.4f}')

@torch.no_grad()
def generate_predictions(model, dataset, dataset_name, device="cuda", batch_size=256, num_workers=4):
    """Runs batched inference across a dataset and returns a Predictions object."""
    from ai_safety.utils.io import Predictions
    model.eval()
    use_cuda = (device == "cuda" and torch.cuda.is_available())

    orig_flag = getattr(dataset, "return_sopuid", None)
    if hasattr(dataset, "return_sopuid"):
        dataset.return_sopuid = True

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    sop_uids, probs_list, targets_list = [], [], []
    for batch in tqdm(loader, desc=f"Predicting -> {dataset_name}", leave=False):
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1]
        uids = batch[2] if len(batch) > 2 else [f"ID_{i}" for i in range(len(labels))]

        with torch.amp.autocast("cuda", enabled=use_cuda):
            logits = model(images)
            probs = torch.sigmoid(logits)

        probs_list.append(probs.cpu().numpy())
        targets_list.append(labels.numpy())
        sop_uids.extend(uids)

    if orig_flag is not None:
        dataset.return_sopuid = orig_flag

    all_probs = np.concatenate(probs_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)

    # Reconstruct metadata for Predictions CSV
    sop_to_series = {}
    sop_to_dataset = {}
    
    datasets_list = dataset.datasets if hasattr(dataset, "datasets") else [dataset]
    for ds in datasets_list:
        ds_name = getattr(ds, "dataset_name", dataset_name)
        for r in getattr(ds, "records", []):
            uid = r["sop_uid"]
            sop_to_series[uid] = r.get("series_id", "")
            sop_to_dataset[uid] = ds_name

    series_ids = np.array([sop_to_series.get(uid, "") for uid in sop_uids])
    ds_arr = np.array([sop_to_dataset.get(uid, dataset_name) for uid in sop_uids])
    label_cols = getattr(dataset, "label_cols", ["ich"]) if not getattr(dataset, "binary", True) else ["ich"]

    return Predictions(
        sop_uid=np.array(sop_uids),
        series_id=series_ids,
        dataset=ds_arr,
        labels=all_targets.reshape(len(sop_uids), -1),
        probs=all_probs.reshape(len(sop_uids), -1),
        class_names=[c.replace("label_", "") for c in label_cols],
    )

