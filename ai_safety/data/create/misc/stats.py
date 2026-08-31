import argparse
from pathlib import Path
import pandas as pd
import json

from ai_safety.data import DATASET_REGISTRY

def print_stats(splits_dir):
    splits_dir = Path(splits_dir)
    if not splits_dir.exists():
        print(f"Directory {splits_dir} does not exist.")
        return
        
    dataset_stats = {}

    # Load all index.parquet files into memory
    indices = {}
    for name, path in DATASET_REGISTRY.items():
        index_path = path / "index.parquet"
        if index_path.exists():
            indices[name] = pd.read_parquet(index_path)

    for dataset_dir in sorted(splits_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        if dataset_name not in indices:
            print(f"\n[{dataset_name}] No index.parquet found in registry, skipping.")
            continue
            
        print(f"\n{'='*50}\nDATASET: {dataset_name}\n{'='*50}")
        index_df = indices[dataset_name]
        label_cols = sorted([c for c in index_df.columns if c.startswith('label_')])
        
        dataset_stats[dataset_name] = {}
        
        for split_file in sorted(dataset_dir.glob("*.parquet")):
            split_df = pd.read_parquet(split_file)
            
            # Merge to get full metadata
            merged_df = index_df.merge(split_df[['sop_uid']], on='sop_uid', how='inner')
            
            n_slices = len(merged_df)
            n_series = merged_df['series_id'].nunique() if 'series_id' in merged_df.columns else 0
            n_patients = merged_df['patient_id'].nunique() if 'patient_id' in merged_df.columns else 0
            
            print(f"\n--- Split: {split_file.name} ---")
            print(f"Patients: {n_patients:,} | Series: {n_series:,} | Slices: {n_slices:,}")
            
            split_stats = {
                "patients": int(n_patients),
                "series": int(n_series),
                "slices": int(n_slices),
                "labels": {}
            }
            
            if n_slices > 0:
                print("Label Distribution:")
                for col in label_cols:
                    count = int(merged_df[col].sum())
                    prevalence = count / n_slices
                    label_name = col.replace('label_', '')
                    print(f"  {label_name:<12}: {count:>7.0f} ({prevalence:.1%})")
                    split_stats["labels"][label_name] = {"count": count, "prevalence": prevalence}
                    
            dataset_stats[dataset_name][split_file.name] = split_stats

    # Save to JSON
    out_path = splits_dir / "stats.json"
    with open(out_path, "w") as f:
        json.dump(dataset_stats, f, indent=4)
    print(f"\nSaved statistics to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print statistics for generated splits.")
    parser.add_argument("--splits-dir", default="splits", help="Path to splits directory (default: splits/)")
    args = parser.parse_args()

    print_stats(args.splits_dir)
