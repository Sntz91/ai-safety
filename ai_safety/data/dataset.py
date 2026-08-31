import os
import io
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset

from ai_safety.data import DATASET_REGISTRY
from ai_safety.utils.io import Predictions

class BaseShardedDataset(Dataset):
    """ Base class to handle tar- and corresponding parquet files. """
    def __init__(self, tar_root, split_path=None, binary=False):
        self._tar_handles = {}
        self.binary = binary
        index = self._load_index(tar_root, split_path)
        self.records = index.to_dict('records')
        self.label_cols = sorted([c for c in index.columns if c.startswith("label_")])

    def __getstate__(self):
        """ Before a new worker is initialized; reset the tar handles. """
        state = self.__dict__.copy()
        state['_tar_handles'] = {}
        return state

    def _load_index(self, tar_root, split_path):
        """ load the corresponding index.parquet file. """
        tar_root = Path(tar_root)
        index = pd.read_parquet(tar_root / 'index.parquet')
        if split_path is not None:
            split_ids = pd.read_parquet(split_path)["sop_uid"]
            index = index[index["sop_uid"].isin(split_ids)].reset_index(drop=True)
        index['tar_full_path'] = tar_root / index['tar']
        return index

    def __len__(self):
        return len(self.records)

    def _load_tar(self, tar_path):
        """ Loads a tar and adds it to the worker's tar handles. """
        if tar_path not in self._tar_handles:
            self._tar_handles[tar_path] = os.open(tar_path, os.O_RDONLY)
        return self._tar_handles[tar_path]

    def get_labels(self, row=None):
        """ return labels depending on the task given index entry (or for all records if row is None). """
        if row is not None:
            if not self.binary:
                return np.array([row[c] for c in self.label_cols], dtype=np.float32)
            return np.array([any([row[c] for c in self.label_cols]) > 0], dtype=np.float32)
        return np.array([self.get_labels(r) for r in self.records], dtype=np.float32)

    def get_image(self, row):
        """ return the raw HU image given index entry in O(1) time. """
        fd = self._load_tar(row['tar_full_path'])
        
        npy_bytes = os.pread(
            fd,
            int(row['size']),
            int(row['data_offset'])
        )
        
        if len(npy_bytes) != int(row['size']):
            raise IOError(f"Short read for {row.get('sop_uid')}: {len(npy_bytes)} != {row.get('size')}")
            
        image = np.load(io.BytesIO(npy_bytes)).astype(np.float32)
        return image

    def filter_by_sop_uids(self, sop_uids):
        """ 
        filter the dataset by a set of sop_uids. (necessary for monitor 
        dataset only loading the predicted sop_uids) 
        """
        sop_set = set(sop_uids)
        self.records = [r for r in self.records if r["sop_uid"] in sop_set]

    def compute_pos_weights(self, cap=None):
        """Calculates positive class weight vector for BCE / Focal loss."""
        labels = np.array([self.get_labels(r) for r in self.records], dtype=np.float32)
        n = labels.shape[0]
        pos_counts = labels.sum(axis=0)
        neg_counts = n - pos_counts
        pos_weights = neg_counts / np.maximum(pos_counts, 1.0)
        if cap is not None:
            pos_weights = np.minimum(pos_weights, cap)
        return torch.tensor(pos_weights, dtype=torch.float32)

    def compute_sample_weights(self, cap=10.0):
        """Calculates balanced sample weights for WeightedRandomSampler."""
        labels = np.array([self.get_labels(r) for r in self.records], dtype=np.float32)
        pos_counts = labels.sum(axis=0)
        max_count = max(pos_counts.max(), 1.0)
        class_weight = np.minimum(max_count / np.maximum(pos_counts, 1.0), cap)

        weights = np.ones(len(labels), dtype=np.float64)
        has_any = labels.sum(axis=1) > 0
        if has_any.any():
            weights[has_any] = (labels[has_any] * class_weight).max(axis=1)
        return torch.tensor(weights, dtype=torch.double), labels


class CTSliceDataset(BaseShardedDataset):
    """ Standard CT Dataset on slice-level. """
    def __init__(self, tar_root, split_path=None, binary=False, transform=None, return_sopuid=False):
        super().__init__(tar_root, split_path, binary)
        self.transform = transform
        self.return_sopuid = return_sopuid

    def __getitem__(self, idx):
        row = self.records[idx]
        image = self.get_image(row)
        label = self.get_labels(row)
        if self.transform is not None:
            image, label = self.transform(image, label)
        if self.return_sopuid:
            return image, label, row["sop_uid"]
        return image, label


class MonitorDataset(ConcatDataset):
    """ Create monitor dataset based on predictions - can have different base datasets. """
    def __init__(self, predictions_path, thresholds, binary=True, subtype=False, transform=None):
        self.binary = binary
        self.subtype = subtype
        data = Predictions.from_csv(predictions_path)
        sop_uids = data.sop_uid
        labels = data.labels
        probs = data.probs
        classes = data.class_names
        datasets = data.dataset
        errors = self._get_failure_cases(probs, labels, thresholds)
        self.labels = {uid: errors[i] for i, uid in enumerate(sop_uids)}

        subsets = []
        
        # Create CT datasets; can be multi-institutional
        for dataset_name in sorted(set(datasets)):
            ds = CTSliceDataset(DATASET_REGISTRY[dataset_name], return_sopuid=True, transform=transform)
            ds.dataset_name = dataset_name
            ds.filter_by_sop_uids(sop_uids)
            subsets.append(ds)

        super().__init__(subsets)

    def _get_failure_cases(self, probs, labels, thresholds):
        """ Return errors based on mode -> binary / subtype. """
        preds = (probs >= np.asarray(thresholds, dtype=np.float32)).astype(np.float32)
        if self.binary:
            if not self.subtype:
                return (labels != preds).any(axis=1)[:, None].astype(np.float32)
            else:
                return (labels != preds).astype(np.float32)
        else:
            if not self.subtype:
                is_fp = ((labels == 0) & (preds == 1)).any(axis=1).astype(np.float32)
                is_fn = ((labels == 1) & (preds == 0)).any(axis=1).astype(np.float32)
                return np.stack([is_fp, is_fn], axis=1)
            else:
                raise NotImplementedError()

    def __getitem__(self, idx):
        image, _, sop_uid = super().__getitem__(idx)
        error_target = self.labels[sop_uid]
        if getattr(self, "return_sopuid", False):
            return image, error_target, sop_uid
        return image, error_target

    @property
    def records(self):
        """Aggregate records from underlying datasets to provide metadata like series_id."""
        recs = []
        for ds in self.datasets:
            recs.extend(ds.records)
        return recs

    def get_labels(self):
        """ Returns failure target labels for all samples in the dataset. """
        uids = []
        for ds in self.datasets:
            uids.extend([r["sop_uid"] for r in ds.records])
        return np.array([self.labels[uid] for uid in uids], dtype=np.float32)

    def compute_pos_weights(self, cap=None):
        labels = self.get_labels()
        if labels.ndim == 1:
            labels = labels[:, None]
        n = labels.shape[0]
        pos_counts = labels.sum(axis=0)
        neg_counts = n - pos_counts
        pos_weights = neg_counts / np.maximum(pos_counts, 1.0)
        if cap is not None:
            pos_weights = np.minimum(pos_weights, cap)
        return torch.tensor(pos_weights, dtype=torch.float32)


       
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ai_safety.data.transforms import Transform, IMAGENET_MEAN, IMAGENET_STD

    t_train = Transform(train=True, image_size=224)
    ds = CTSliceDataset(DATASET_REGISTRY['rsna'], transform=t_train)
    img_tensor, label = ds[100]
    print(label)
    img_rgb = img_tensor.permute(1, 2, 0).numpy()
    img_rgb = np.clip(img_rgb * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0.0, 1.0)
    
    plt.imshow(img_rgb)
    plt.show()

    ds = MonitorDataset('predictions-test.csv', thresholds=[0.5, 0.5, 0.2, 0.1, 0.1])
    for (img, label) in ds:
        print(label)
        plt.imshow(img)
        plt.show()



