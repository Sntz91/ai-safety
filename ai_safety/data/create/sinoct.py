import argparse
import pydicom
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ai_safety.data.create.base import BaseDatasetBuilder

class SinoCTBuilder(BaseDatasetBuilder):
    def __init__(self, raw_dir):
        super().__init__(raw_dir)
        self.csv_path = self.raw_dir / 'labels.csv'
        
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"SinoCT root directory not found: {self.raw_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"SinoCT labels.csv not found: {self.csv_path}")

    def load_annotations(self):
        labels = pd.read_csv(self.csv_path)
        labels = labels.rename(columns={'patient_id': 'patient_id_'})
        labels['label_normal'] = labels['label'].apply(
            lambda x: int(x.split(",")[0])
        )
        labels['label_abnormal'] = labels['label'].apply(
            lambda x: int(x.split(",")[1])
        )
        labels = labels.drop(columns=['label'])
        return labels

    def build_manifest(self) -> pd.DataFrame:
        labels = self.load_annotations()
        rows = []
        for dcm in tqdm(sorted(self.raw_dir.rglob('*.dcm')), desc="Processing SinoCT DICOMs"):
            try:
                ds = pydicom.dcmread(dcm, stop_before_pixels=True)
            except Exception:
                continue
                
            slice_pos = int(''.join(filter(str.isdigit, dcm.stem)) or '0')
            z_coord = float(getattr(ds, 'SliceLocation', slice_pos))
            
            rows.append({
                'key': str(ds.SOPInstanceUID).replace(".", "_"),
                'img_path': str(dcm),
                'patient_id': str(ds.PatientID),
                'patient_id_': dcm.parents[1].name,
                'sop_uid': str(ds.SOPInstanceUID),
                'series_id': str(ds.SeriesInstanceUID),
                'slice_pos': slice_pos,
                'z_coord': z_coord
            })
            
        df = pd.DataFrame(rows)
        df = df.merge(labels, on='patient_id_', how='inner')
        df = df.sort_values(['series_id', 'slice_pos']).reset_index(drop=True)
        return df


def create_dataset(sinoct_root, output_dir, max_shard_size):
    builder = SinoCTBuilder(sinoct_root)
    builder.create(output_dir, max_shard_size)
    print(f"SinoCT Dataset successfully written to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create sharded SinoCT dataset.")
    parser.add_argument('--raw-dir', required=True, help="Path to raw SinoCT root (containing head_ct_dataset_anon/ and labels.csv)")
    parser.add_argument('--output-dir', required=True, help="Target output directory for index.parquet and tars")
    parser.add_argument('--max-shard-size', type=int, default=1000, help="Slices per tar shard (default: 1000)")
    args = parser.parse_args()

    create_dataset(args.raw_dir, args.output_dir, args.max_shard_size)
