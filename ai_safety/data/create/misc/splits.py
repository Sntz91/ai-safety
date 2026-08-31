import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from ai_safety.data import DATASET_REGISTRY


def stratified_split(patients, label_cols, test_size, seed):
    """Splits a patient DataFrame into two stratified subsets."""
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_a, idx_b = next(splitter.split(patients, patients[label_cols].to_numpy(dtype=int)))
    return patients.iloc[idx_a], patients.iloc[idx_b]


def split_dataframe(df, out_dir, prefix="", val_ratio=0.10, test_ratio=0.10, seed=42):
    """Stratifies any slice DataFrame at patient level and writes clean Parquet splits."""
    group_col = "patient_id" if "patient_id" in df.columns else "series_id"
    label_cols = sorted([c for c in df.columns if c.startswith("label_")])

    # 1. Aggregate to patient level for stratification
    patients = df.groupby(group_col)[label_cols].max().reset_index()

    # 2. Perform 3-way or 2-way stratified splitting
    if test_ratio:
        train, heldout = stratified_split(patients, label_cols, val_ratio + test_ratio, seed)
        val, test = stratified_split(heldout, label_cols, test_ratio / (val_ratio + test_ratio), seed)
        splits = {"train": train, "val": val, "test": test}
    else:
        train, val = stratified_split(patients, label_cols, val_ratio, seed)
        splits = {"train": train, "val": val}

    # 3. Save clean single-column Parquet files
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}-" if prefix else ""

    for name, split_df in splits.items():
        pids = set(split_df[group_col])
        df.loc[df[group_col].isin(pids), ["sop_uid"]].to_parquet(
            out_dir / f"{pfx}{name}.parquet",
            index=False,
        )
    df[["sop_uid"]].to_parquet(out_dir / f"{pfx}all.parquet", index=False)


def split_single_dataset(name, splits_root, val_ratio=0.10, test_ratio=0.10, seed=42):
    """Creates standard train/val/test/all splits for a single dataset."""
    path = DATASET_REGISTRY.get(name, Path(name))
    df = pd.read_parquet(path / "index.parquet")
    split_dataframe(df, splits_root / name, prefix="", val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)


def split_paired_datasets(dataset_a, dataset_b, mapping_csv, splits_root, val_ratio=0.10, test_ratio=0.10, seed=42):
    """ Creates match-<counterpart> and no-match-<counterpart> splits for two overlapping datasets. """
    mapping = pd.read_csv(Path(mapping_csv))
    col_a = [c for c in mapping.columns if dataset_a in c and "sop_uid" in c]
    col_b = [c for c in mapping.columns if dataset_b in c and "sop_uid" in c]
    col_a_name = col_a[0] if col_a else mapping.columns[0]
    col_b_name = col_b[0] if col_b else mapping.columns[1]

    matched_sops_a = set(mapping[col_a_name])
    matched_sops_b = set(mapping[col_b_name])

    # 1. Dataset A (match-<B> and no-match-<B>)
    path_a = DATASET_REGISTRY.get(dataset_a, Path(dataset_a))
    if path_a.exists():
        df_a = pd.read_parquet(path_a / "index.parquet")
        
        # Save complete all.parquet
        out_dir_a = splits_root / dataset_a
        out_dir_a.mkdir(parents=True, exist_ok=True)
        df_a[["sop_uid"]].to_parquet(out_dir_a / "all.parquet", index=False)
        
        matched_a = df_a[df_a["sop_uid"].isin(matched_sops_a)]
        nomatch_a = df_a[~df_a["sop_uid"].isin(matched_sops_a)]

        split_dataframe(matched_a, out_dir_a, prefix=f"match-{dataset_b}", val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        if len(nomatch_a) > 0:
            split_dataframe(nomatch_a, out_dir_a, prefix=f"no-match-{dataset_b}", val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    # 2. Dataset B (match-<A> and no-match-<A>)
    path_b = DATASET_REGISTRY.get(dataset_b, Path(dataset_b))
    if path_b.exists():
        df_b = pd.read_parquet(path_b / "index.parquet")
        
        # Save complete all.parquet
        out_dir_b = splits_root / dataset_b
        out_dir_b.mkdir(parents=True, exist_ok=True)
        df_b[["sop_uid"]].to_parquet(out_dir_b / "all.parquet", index=False)
        
        matched_b = df_b[df_b["sop_uid"].isin(matched_sops_b)]
        nomatch_b = df_b[~df_b["sop_uid"].isin(matched_sops_b)]

        split_dataframe(matched_b, splits_root / dataset_b, prefix=f"match-{dataset_a}", val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
        if len(nomatch_b) > 0:
            split_dataframe(nomatch_b, splits_root / dataset_b, prefix=f"no-match-{dataset_a}", val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Create patient-stratified Parquet splits.")
    parser.add_argument("--dataset", nargs="+", required=True, help="One dataset name (e.g. 'bhx') or two paired datasets (e.g. 'rsna sinoct')")
    parser.add_argument("--mapping", help="Path to cross-dataset mapping CSV (required when two datasets are provided)")
    parser.add_argument("--out-dir", default="splits", help="Output directory for splits/ (default: splits/)")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio (default: 0.10)")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test ratio (default: 0.10, set 0 or omit for train/val only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    test_ratio = args.test_ratio if args.test_ratio > 0 else None
    splits_root = Path(args.out_dir)

    if len(args.dataset) == 1:
        split_single_dataset(
            name=args.dataset[0],
            splits_root=splits_root,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
            seed=args.seed,
        )
    elif len(args.dataset) == 2:
        if not args.mapping:
            raise ValueError("When providing two datasets, --mapping <path_to_mapping_csv> is required.")
        split_paired_datasets(
            dataset_a=args.dataset[0],
            dataset_b=args.dataset[1],
            mapping_csv=Path(args.mapping),
            splits_root=splits_root,
            val_ratio=args.val_ratio,
            test_ratio=test_ratio,
            seed=args.seed,
        )
    else:
        raise ValueError("Please provide either 1 dataset name or 2 paired dataset names to --dataset.")


if __name__ == "__main__":
    main()
