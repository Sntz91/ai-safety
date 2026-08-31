import re
import argparse
import pydicom
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from ai_safety.data.create.base import BaseDatasetBuilder

ICH_CLASSES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
LABEL_COLS = [f"label_{c}" for c in ICH_CLASSES]

CQ500_SUBTYPE_READERS = {
    "label_epidural": ["R1:EDH", "R2:EDH", "R3:EDH"],
    "label_intraparenchymal": ["R1:IPH", "R2:IPH", "R3:IPH"],
    "label_intraventricular": ["R1:IVH", "R2:IVH", "R3:IVH"],
    "label_subarachnoid": ["R1:SAH", "R2:SAH", "R3:SAH"],
    "label_subdural": ["R1:SDH", "R2:SDH", "R3:SDH"],
}


def normalize_pid(x: str) -> str:
    """Normalizes CQ500 patient IDs (e.g. CQ500-CT-042 -> CQ500CT042)."""
    return re.sub(r"[^A-Za-z0-9]", "", str(x)).upper()


class CQ500Scanner:
    """Reusable scanner and filter for raw CQ500 DICOM archives and reads.csv."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.csv_path = self.raw_dir / "reads.csv"
        self.contrast_keywords = ["CONTRAST", "CE", "ANGIO", "CTA", "POST"]
        self.bone_kernel_keywords = ["BONE", "B70", "B80", "EC"]
        self.reformat_keywords = ["COR", "SAG"]

    def scan(self) -> pd.DataFrame:
        """Scans all DICOM files and extracts spatial/acquisition metadata."""
        rows = []
        skipped = 0
        for path in tqdm(self.raw_dir.rglob("*.dcm"), desc="Scanning CQ500 DICOMs"):
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception:
                skipped += 1
                continue

            z_val = getattr(ds, "SliceLocation", None)
            if z_val is None:
                img_pos = getattr(ds, "ImagePositionPatient", None)
                z_val = img_pos[2] if img_pos is not None and len(img_pos) >= 3 else 0.0

            rows.append({
                "img_path": str(path),
                "patient_id": str(getattr(ds, "PatientID", "")),
                "sop_uid": str(getattr(ds, "SOPInstanceUID")),
                "series_id": str(getattr(ds, "SeriesInstanceUID", "")),
                "SeriesDescription": str(getattr(ds, "SeriesDescription", "")),
                "SliceThickness": float(getattr(ds, "SliceThickness", 999.0)),
                "ConvolutionKernel": str(getattr(ds, "ConvolutionKernel", "")),
                "ImageType": "/".join(getattr(ds, "ImageType", [])).upper(),
                "z_coord": float(z_val),
            })

        print(f"(CQ500) Scan complete: {len(rows)} slices read (skipped {skipped} unreadable).")
        df = pd.DataFrame(rows)
        df["norm_pid"] = df["patient_id"].map(normalize_pid)
        df = df.sort_values(["series_id", "z_coord"]).reset_index(drop=True)
        df["slice_pos"] = df.groupby("series_id").cumcount()
        return df

    def summarize_series(self, scan_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates series metadata and applies clinical acquisition filters."""
        agg = scan_df.groupby("series_id").agg(
            patient_id=("patient_id", "first"),
            norm_pid=("norm_pid", "first"),
            SeriesDescription=("SeriesDescription", "first"),
            SliceThickness=("SliceThickness", "first"),
            ConvolutionKernel=("ConvolutionKernel", "first"),
            ImageType=("ImageType", "first"),
            num_slices=("sop_uid", "count"),
        ).reset_index()

        agg["is_contrast"] = agg["SeriesDescription"].apply(
            lambda d: any(k in d for k in self.contrast_keywords)
        )
        agg["is_bone_kernel"] = agg["ConvolutionKernel"].apply(
            lambda k: any(b in k for b in self.bone_kernel_keywords)
        )
        agg["is_reformat"] = agg["SeriesDescription"].apply(
            lambda d: any(k in d for k in self.reformat_keywords)
        )
        agg["is_scout"] = agg["ImageType"].str.contains("LOCALIZER") | (agg["num_slices"] < 10)
        agg["is_thin"] = agg["SliceThickness"] <= 1.5
        return agg

    def get_negative_patients(self, exclude_norm_pids: set = None) -> set:
        """Identifies patients with 0 hemorrhage based on 3-radiologist majority vote."""
        df = pd.read_csv(self.csv_path).rename(columns={"name": "patient_id"})
        vote_cols = ["R1:ICH", "R2:ICH", "R3:ICH"]
        votes = df[vote_cols].fillna(0).sum(axis=1)
        df["study_label"] = (votes >= 2).astype(int)
        df["norm_pid"] = df["patient_id"].map(normalize_pid)
        negatives = set(df.loc[df["study_label"] == 0, "norm_pid"])
        if exclude_norm_pids:
            negatives -= exclude_norm_pids
        return negatives

    def select_series(self, series_df: pd.DataFrame, target_norm_pids: set) -> pd.DataFrame:
        """Selects 1 standard thin non-contrast axial series per target patient."""
        candidates = series_df[
            series_df["norm_pid"].isin(target_norm_pids)
            & ~series_df["is_contrast"]
            & ~series_df["is_bone_kernel"]
            & ~series_df["is_scout"]
            & ~series_df["is_reformat"]
            & series_df["is_thin"]
        ].copy()
        candidates = candidates.sort_values(["SliceThickness", "num_slices"], ascending=[True, False])
        return candidates.groupby("norm_pid").first().reset_index()


class CQ500Builder(BaseDatasetBuilder):
    """Ingests full canonical CQ500 dataset with 3-radiologist majority vote labels."""

    def __init__(self, raw_dir: str | Path):
        super().__init__(raw_dir)
        self.scanner = CQ500Scanner(self.raw_dir)
        if not self.scanner.csv_path.exists():
            raise FileNotFoundError(f"Missing reads.csv in {self.raw_dir}")

    def load_annotations(self) -> pd.DataFrame:
        """Computes 3-reader consensus majority vote (>= 2/3) for all 5 ICH subtypes."""
        df = pd.read_csv(self.scanner.csv_path).rename(columns={"name": "patient_id"})
        df["norm_pid"] = df["patient_id"].map(normalize_pid)

        for target_col, reader_cols in CQ500_SUBTYPE_READERS.items():
            present_cols = [c for c in reader_cols if c in df.columns]
            df[target_col] = (df[present_cols].fillna(0).sum(axis=1) >= 2).astype(float)

        return df[["norm_pid"] + LABEL_COLS]

    def build_manifest(self) -> pd.DataFrame:
        scan_df = self.scanner.scan()
        series_df = self.scanner.summarize_series(scan_df)
        labels_df = self.load_annotations()

        # Select 1 standard series per patient across all CQ500 patients
        all_pids = set(labels_df["norm_pid"])
        chosen_series = self.scanner.select_series(series_df, all_pids)
        selected_slices = scan_df[scan_df["series_id"].isin(chosen_series["series_id"])].copy()

        # Merge with 3-reader consensus labels
        merged = selected_slices.merge(labels_df, on="norm_pid", how="inner")
        merged = merged.sort_values(["series_id", "z_coord"]).reset_index(drop=True)
        merged["slice_pos"] = merged.groupby("series_id").cumcount()

        print(f"(CQ500) Manifest: {len(merged)} slices across {merged['series_id'].nunique()} series.")
        for col in LABEL_COLS:
            print(f"  {col:<28} {int(merged.groupby('series_id')[col].max().sum())} positive scans")

        return merged.drop(columns=["norm_pid"], errors="ignore")


def create_cq500(raw_dir, output_dir, max_shard_size=1000):
    builder = CQ500Builder(raw_dir)
    builder.create(output_dir, max_shard_size=max_shard_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sharded canonical CQ500 dataset.")
    parser.add_argument("--raw-dir", required=True, help="Path to raw CQ500 directory (containing reads.csv)")
    parser.add_argument("--output-dir", required=True, help="Target output directory")
    parser.add_argument("--max-shard-size", type=int, default=1000)
    args = parser.parse_args()

    create_cq500(args.raw_dir, args.output_dir, max_shard_size=args.max_shard_size)
