import logging
import warnings
import argparse
import pydicom
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from ai_safety.data.create.base import BaseDatasetBuilder
# Get rid of warnings because they are not valid sopuids etc.
logging.getLogger('pydicom').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='pydicom')

ICH_CLASSES = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
LABEL_COLS = [f'label_{c}' for c in ICH_CLASSES]


class RSNABuilder(BaseDatasetBuilder):
    """Ingests raw RSNA-IHD dataset into a sharded archive with standardized labels."""

    def __init__(self, raw_dir):
        super().__init__(raw_dir)
        self.dicom_dir = self.raw_dir / 'stage_2_train'
        self.csv_path = self.raw_dir / 'stage_2_train.csv'

        if not self.dicom_dir.exists():
            raise FileNotFoundError(f"RSNA DICOM directory not found: {self.dicom_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"RSNA annotations CSV not found: {self.csv_path}")

    def load_annotations(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        def _extract_sop_uid(id_str):
            for lt in ICH_CLASSES + ['any']:
                if id_str.endswith('_' + lt):
                    return id_str[:-(len(lt) + 1)]
            return id_str

        df['sop_uid'] = df['ID'].apply(_extract_sop_uid)
        df['label_type'] = df['ID'].apply(lambda x: x.rsplit('_', 1)[-1])
        # a few duplicated images exist...
        df = df.drop_duplicates(subset=['sop_uid', 'label_type'])

        pivoted = df.pivot(index='sop_uid', columns='label_type', values='Label').reset_index()

        # Standardize target columns with label_ prefix
        for col in ICH_CLASSES:
            pivoted[f'label_{col}'] = pivoted.get(col, 0)

        return pivoted[['sop_uid'] + LABEL_COLS]

    def build_manifest(self) -> pd.DataFrame:
        """Scans raw RSNA DICOMs and merges with subtype annotations."""
        labels = self.load_annotations()
        rows = []

        dcm_files = [p for p in self.dicom_dir.iterdir() if p.suffix == '.dcm']
        for dcm in tqdm(dcm_files, desc="Scanning RSNA DICOMs"):
            try:
                ds = pydicom.dcmread(dcm, stop_before_pixels=True)
            except Exception:
                continue

            # Safe z_coord extraction
            img_pos = getattr(ds, 'ImagePositionPatient', None)
            z_coord = float(img_pos[2]) if img_pos is not None and len(img_pos) >= 3 else float(getattr(ds, 'SliceLocation', 0.0))

            rows.append({
                'img_path': str(dcm),
                'sop_uid': str(ds.SOPInstanceUID),
                'patient_id': str(ds.PatientID),
                'series_id': str(ds.SeriesInstanceUID),
                'z_coord': z_coord,
            })

        df = pd.DataFrame(rows)
        df = df.merge(labels, on='sop_uid', how='inner')
        df = df.sort_values(['series_id', 'z_coord']).reset_index(drop=True)
        df['slice_pos'] = df.groupby('series_id').cumcount()

        return df


def create_dataset(raw_dir, output_dir, max_shard_size=1000):
    builder = RSNABuilder(raw_dir)
    builder.create(output_dir, max_shard_size=max_shard_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create sharded RSNA-IHD dataset.")
    parser.add_argument('--raw-dir', required=True, help="Path to raw RSNA root (containing stage_2_train/ and stage_2_train.csv)")
    parser.add_argument('--output-dir', required=True, help="Target output directory for index.parquet and tars")
    parser.add_argument('--max-shard-size', type=int, default=1000, help="Slices per tar shard (default: 1000)")
    args = parser.parse_args()

    create_dataset(args.raw_dir, args.output_dir, max_shard_size=args.max_shard_size)
