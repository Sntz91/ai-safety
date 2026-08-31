import io
import hashlib
import tarfile
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from ai_safety.data import DATASET_REGISTRY

def _hash_from_tar(tar_path, df_tar):
    """Yields (md5_hash, row) for all slices in a single tar shard in one sequential pass."""
    with tarfile.open(tar_path, "r") as tf:
        for _, row in df_tar.iterrows():
            filename = f"{row['sop_uid']}.npy"
            f = tf.extractfile(filename)
            hu = np.load(io.BytesIO(f.read()))
            md5 = hashlib.md5(hu.tobytes()).hexdigest()
            yield md5, row

def match_datasets(dataset_a_root, dataset_b_root, name_a, name_b, out_csv):
    """Finds exact slice matches between two datasets via pixel array hashing."""
    df_a = pd.read_parquet(dataset_a_root / "index.parquet")
    df_b = pd.read_parquet(dataset_b_root / "index.parquet")
    # 1. Hash all slices in Dataset A into lookup map
    hash_map_a = defaultdict(list)
    for tar_name, grp in tqdm(df_a.groupby("tar"), desc=f"Hashing {name_a.upper()}"):
        tar_path = dataset_a_root / tar_name
        for md5, row in _hash_from_tar(tar_path, grp):
            hash_map_a[md5].append(row)

    # 2. Hash Dataset B and match against Dataset A
    matches = []
    for tar_name, grp in tqdm(df_b.groupby("tar"), desc=f"Matching {name_b.upper()}"):
        tar_path = dataset_b_root / tar_name
        for md5, row_b in _hash_from_tar(tar_path, grp):
            if md5 in hash_map_a:
                for row_a in hash_map_a[md5]:
                    matches.append({
                        f"{name_a}_sop_uid": row_a["sop_uid"],
                        f"{name_a}_series_id": row_a.get("series_id", ""),
                        f"{name_a}_patient_id": row_a.get("patient_id", ""),
                        f"{name_b}_sop_uid": row_b["sop_uid"],
                        f"{name_b}_series_id": row_b.get("series_id", ""),
                        f"{name_b}_patient_id": row_b.get("patient_id", ""),
                    })
    out_df = pd.DataFrame(matches)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Match CT slices across two datasets by pixel hash.")
    parser.add_argument("--dataset-a", required=True, help="Name or path of dataset A")
    parser.add_argument("--dataset-b", required=True, help="Name or path of dataset B")
    parser.add_argument("--out-csv", required=True, help="Output mapping CSV path")
    args = parser.parse_args()

    path_a = DATASET_REGISTRY.get(args.dataset_a, Path(args.dataset_a))
    path_b = DATASET_REGISTRY.get(args.dataset_b, Path(args.dataset_b))

    match_datasets(
        dataset_a_root=Path(path_a),
        dataset_b_root=Path(path_b),
        name_a=args.dataset_a,
        name_b=args.dataset_b,
        out_csv=Path(args.out_csv),
    )


if __name__ == "__main__":
    main()
