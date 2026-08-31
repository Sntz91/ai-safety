import os
from pathlib import Path

BASE_DATA_DIR = Path(os.environ.get('DATA_DIR', '/run/media/tobias/backup/data'))

DATASET_REGISTRY = {
    'rsna': BASE_DATA_DIR / 'rsna-processed',
    'bhx': BASE_DATA_DIR / 'bhx-processed',
    #'cq500': BASE_DATA_DIR / 'cq500-processed',
    'sinoct': BASE_DATA_DIR / 'sinoct-processed',
}

def get_dataset_root(name):
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' not found. Available: {list(DATASET_REGISTRY.keys())}")
    return Path(DATASET_REGISTRY[name])
