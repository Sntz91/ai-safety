from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass
class Predictions:
    """Standardized container for slice-level model predictions."""
    sop_uid: np.ndarray
    series_id: np.ndarray
    dataset: np.ndarray
    labels: np.ndarray
    probs: np.ndarray
    class_names: list[str]

    def __len__(self) -> int:
        return len(self.sop_uid)

    def __getitem__(self, item: str):
        """Allows dictionary-style access for backwards compatibility."""
        return getattr(self, item)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts predictions dataclass into a pandas DataFrame."""
        n = len(self.sop_uid)
        data = {
            "sop_uid": self.sop_uid,
            "series_id": self.series_id,
            "dataset": self.dataset,
        }
        labels_2d = self.labels.reshape(n, -1)
        probs_2d = self.probs.reshape(n, -1)

        assert labels_2d.shape[1] == len(self.class_names), f"labels shape {labels_2d.shape} does not match class_names len ({len(self.class_names)})"
        assert probs_2d.shape[1] == len(self.class_names), f"probs shape {probs_2d.shape} does not match class_names len ({len(self.class_names)})"

        for i, c in enumerate(self.class_names):
            name = c.replace("label_", "")
            data[f"label_{name}"] = labels_2d[:, i]
            data[f"prob_{name}"] = probs_2d[:, i]
        return pd.DataFrame(data)

    def to_csv(self, path: Path | str):
        """Saves predictions to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):>7} predictions -> {path}")

    @classmethod
    def from_csv(cls, path: Path | str) -> "Predictions":
        """Reads a standardized prediction CSV file into a Predictions dataclass."""
        df = pd.read_csv(path)
        label_cols = sorted([c for c in df.columns if c.startswith("label_")])
        prob_cols = sorted([c for c in df.columns if c.startswith("prob_")])

        return cls(
            sop_uid=df["sop_uid"].to_numpy(),
            series_id=df["series_id"].to_numpy(),
            dataset=df["dataset"].to_numpy(),
            labels=df[label_cols].to_numpy(dtype=np.float32),
            probs=df[prob_cols].to_numpy(dtype=np.float32),
            class_names=[c.replace("label_", "") for c in label_cols],
        )

    def get_errors(self, thresholds) -> np.ndarray:
        """Computes binary failure indicators: 1 if prediction != ground_truth, else 0."""
        t_vals = np.asarray(thresholds, dtype=np.float32)
        preds = (self.probs >= t_vals).astype(np.float32)
        return (self.labels != preds).any(axis=-1).astype(np.float32)
