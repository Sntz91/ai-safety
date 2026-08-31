import argparse
from pathlib import Path
import pandas as pd

from ai_safety.data.create.base import BaseDatasetBuilder
from ai_safety.data.create.cq500 import CQ500Scanner, ICH_CLASSES, LABEL_COLS


class BHXHandler:
    """Parses BHX CSV annotations and encodes subtype labels."""

    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)

    def load_positive_slices(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df["labelName_lower"] = df["labelName"].str.lower()

        # 1-hot encode subtypes and map chronic -> subdural
        dummies = pd.get_dummies(df["labelName_lower"]).astype(int)
        dummies["subdural"] = (dummies.get("subdural", 0) | dummies.get("chronic", 0))

        for c in ICH_CLASSES:
            dummies[f"label_{c}"] = dummies.get(c, 0)

        dummies["sop_uid"] = df["SOPInstanceUID"].values
        target_cols = [f"label_{c}" for c in ICH_CLASSES]
        return dummies.groupby("sop_uid")[target_cols].max().reset_index()


class BHXCQ500DatasetBuilder(BaseDatasetBuilder):
    """Ingests BHX slice-level annotations overlaying CQ500 DICOMs."""

    def __init__(self, cq500_dir: str | Path, bhx_csv: str | Path):
        super().__init__(raw_dir=cq500_dir)
        self.bhx_csv = Path(bhx_csv)
        if not self.bhx_csv.exists():
            raise FileNotFoundError(f"BHX CSV not found: {self.bhx_csv}")

        # Reuses shared CQ500 scanner (Zero code duplication)
        self.scanner = CQ500Scanner(self.raw_dir)
        self.bhx_handler = BHXHandler(self.bhx_csv)

    def build_manifest(self) -> pd.DataFrame:
        scan_df = self.scanner.scan()
        series_df = self.scanner.summarize_series(scan_df)

        pos_labels = self.bhx_handler.load_positive_slices()
        positives = scan_df.merge(pos_labels, on="sop_uid", how="inner")
        bhx_norm_pids = set(positives["norm_pid"])

        # 1. Select negative series (1 per negative patient)
        neg_patients = self.scanner.get_negative_patients(exclude_norm_pids=bhx_norm_pids)
        chosen_neg_series = self.scanner.select_series(series_df, neg_patients)
        negatives = scan_df[scan_df["series_id"].isin(chosen_neg_series["series_id"])].copy()
        for col in LABEL_COLS:
            negatives[col] = 0.0

        # 2. Add non-annotated negative slices from positive series
        pos_series = set(positives["series_id"])
        series_slices = scan_df[scan_df["series_id"].isin(pos_series)].copy()
        annotated_sops = set(pos_labels["sop_uid"])
        non_annotated = series_slices[~series_slices["sop_uid"].isin(annotated_sops)].copy()
        for col in LABEL_COLS:
            non_annotated[col] = 0.0

        result = pd.concat([positives, negatives, non_annotated], ignore_index=True)
        result = result.sort_values(["series_id", "z_coord"]).reset_index(drop=True)
        result["slice_pos"] = result.groupby("series_id").cumcount()

        print(f"(BHX) Manifest: {len(result)} slices ({len(positives)} positive, "
              f"{len(negatives)} negative series, {len(non_annotated)} non-annotated negative)")
        for col in LABEL_COLS:
            print(f"  {col:<28} {int(result[col].sum())} slices")

        return result.drop(columns=["norm_pid"], errors="ignore")


def create_bhx(cq500_dir, bhx_csv, output_dir, max_shard_size=1000):
    builder = BHXCQ500DatasetBuilder(cq500_dir=cq500_dir, bhx_csv=bhx_csv)
    builder.create(output_dir, max_shard_size=max_shard_size)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create sharded BHX dataset.")
    parser.add_argument("--cq500-dir", required=True, help="Path to raw CQ500 DICOM directory")
    parser.add_argument("--bhx-csv", required=True, help="Path to bhx.csv annotations")
    parser.add_argument("--output-dir", required=True, help="Target output directory")
    parser.add_argument("--max-shard-size", type=int, default=1000)
    args = parser.parse_args()

    create_bhx(args.cq500_dir, args.bhx_csv, args.output_dir, max_shard_size=args.max_shard_size)
