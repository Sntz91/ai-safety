import io
import tarfile
import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from abc import ABC, abstractmethod

REQUIRED_MANIFEST_COLS = {'img_path', 'sop_uid', 'series_id', 'patient_id', 'slice_pos', 'z_coord'}

class BaseDatasetBuilder(ABC):
    """ Abstract blueprint for datasets. """
    def __init__(self, raw_dir):
        self.raw_dir = Path(raw_dir)
        
    @abstractmethod
    def build_manifest(self) -> pd.DataFrame:
        """ Scans raw DICOMs and CSVs.
        Must return a DataFrame containing:
            - 'img_path'
            - 'sop_uid'
            - 'series_id'
            - 'patient_id'
            - 'z_coord'
            - 'slice_pos'
            - 'label_*'
        """
        pass

    def write_tars(self, manifest_df, output_dir, max_shard_size=1000):
        """ 
        Reads DICOMs from manifest_df['img_path'], writes 
        {sop_uid}.npy into tars,
        and returns the manifest augmented 
        with ['tar', 'offset', 'size'] to build index.parquet
        """
        tar_records = []
        shard_idx = 0
        current_tar = None
        current_tar_path = None
        count_in_shard = 0
        skipped = 0

        for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc=f'Writing {output_dir.name} tars'):
            # Read DICOM 
            try: 
                ds = pydicom.dcmread(row['img_path'])
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                hu = (ds.pixel_array.astype(np.float32) * slope + intercept).astype(np.int16)
            except Exception as e:
                print(f'Failed to read pydicom: {e}, skipped.')
                skipped += 1
                continue
            # full shard; open new one
            if count_in_shard >= max_shard_size or current_tar is None:
                if current_tar is not None:
                    current_tar.close()
                # create new shard
                current_tar_path = output_dir / f'shard-{shard_idx:06d}.tar'
                current_tar = tarfile.open(current_tar_path, 'w')
                shard_idx += 1
                count_in_shard = 0
            # Create npy for image
            buf = io.BytesIO()
            np.save(buf, hu)
            buf.seek(0)
            # Store slice within tar as {sop_uid}.npy
            ti = tarfile.TarInfo(name=f"{row['sop_uid']}.npy")
            ti.size = len(buf.getvalue())
            
            # Write header and data
            offset = current_tar.fileobj.tell() # Legacy header offset
            current_tar.addfile(ti, buf)
            
            # Mathematically compute the exact byte offset of the data payload
            end_offset = current_tar.fileobj.tell()
            padded_size = ti.size + (512 - ti.size % 512) if ti.size % 512 else ti.size
            data_offset = end_offset - padded_size

            tar_records.append({
                'sop_uid': row['sop_uid'],
                'tar': current_tar_path.name,
                'offset': offset,
                'data_offset': data_offset,
                'size': ti.size,
            })
            count_in_shard += 1
        if current_tar is not None:
            current_tar.close()
        if skipped > 0:
            print(f'Skipped {skipped} corrupted DICOM slices.')
        tar_df = pd.DataFrame(tar_records)
        return manifest_df.merge(tar_df, on='sop_uid', how='inner')

    def write_index(self, manifest_with_tars, output_dir):
        """ Save the index.parquet file into new dataset root. """
        index_path = output_dir / 'index.parquet'
        manifest_with_tars.drop(
            columns=['img_path']
        ).to_parquet(index_path, index=False)
        return index_path

    def create(self, output_dir, max_shard_size):
        """ Main method to create the dataset. """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # 1. Build manifest
        manifest_df = self.build_manifest()
        self._validate_manifest(manifest_df)
        # 2. Write Tars
        manifest_with_tars = self.write_tars(manifest_df, output_dir, max_shard_size=max_shard_size)
        # 3. Write index.parquet
        index_path = self.write_index(manifest_with_tars, output_dir)
        return index_path

    def _validate_manifest(self, df: pd.DataFrame):
        """ Validate if the manifest is correct. """
        assert REQUIRED_MANIFEST_COLS.issubset(set(df.columns)), 'Missing manifest columns.'
        label_cols = [c for c in df.columns if c.startswith('label_')]
        assert len(label_cols) > 0, 'No target column found (label_*)'


