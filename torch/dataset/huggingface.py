"""HuggingFace datasets loader."""

from typing import List, Tuple, Union

import numpy as np

from .base import DataLoader


class HuggingFaceDataLoader(DataLoader):
    """Load data from HuggingFace datasets."""

    def load_data(self) -> Tuple[List[str], Union[np.ndarray, List[str]]]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for HuggingFace datasets. "
                "Install with: pip install datasets"
            )

        dataset_name = self.config.hf_dataset_name
        if not dataset_name:
            raise ValueError("hf_dataset_name must be specified for HuggingFace data format")

        # Load dataset
        load_kwargs = {}
        if self.config.hf_subset:
            load_kwargs["name"] = self.config.hf_subset

        print(f"Loading HuggingFace dataset: {dataset_name} ...")
        ds = load_dataset(dataset_name, **load_kwargs)

        # Combine train and test splits into one list (trainer handles splitting)
        all_rows = []
        for split_name in [self.config.hf_split_train, self.config.hf_split_test]:
            if split_name and split_name in ds:
                all_rows.extend(ds[split_name])

        if not all_rows:
            raise ValueError(
                f"No data found. Available splits: {list(ds.keys())}"
            )

        print(f"Loaded {len(all_rows)} examples from HuggingFace dataset")

        # Extract text column
        text_col = self.config.text_column
        if not text_col:
            raise ValueError("text_column must be specified for HuggingFace data format")

        texts = [str(row[text_col]) for row in all_rows]

        # Extract target or label column
        if self.config.target_column:
            targets = [str(row[self.config.target_column]) for row in all_rows]
            return texts, targets
        else:
            label_col = self.config.label_column
            if label_col and label_col in all_rows[0]:
                labels = [row[label_col] for row in all_rows]
                labels = np.array(labels, dtype=np.float32)
            else:
                labels = np.zeros(len(texts))
            return texts, labels
