"""Parquet data loader."""

from typing import List, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array

from .base import DataLoader


class ParquetDataLoader(DataLoader):
    """Load data from Parquet files."""

    def load_data(self) -> Tuple[List[str], Union[Array, List[str]]]:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for Parquet loading. Install with: pip install pandas pyarrow")

        df = pd.read_parquet(self.config.data_path)

        # Extract text
        if self.config.text_column:
            texts = df[self.config.text_column].astype(str).tolist()
        elif self.config.parquet_columns:
            # Concatenate multiple columns
            texts = df[self.config.parquet_columns].apply(
                lambda row: " | ".join(row.astype(str)), axis=1
            ).tolist()
        else:
            raise ValueError("Must specify text_column or parquet_columns for Parquet data")

        # Extract labels or targets
        if self.config.target_column:  # Generation task
            targets = df[self.config.target_column].astype(str).tolist()
            return texts, targets
        else:  # Classification/Regression
            labels = df[self.config.label_column].values
            # Convert to appropriate type
            if labels.dtype == object:
                # Try to encode string labels
                try:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    labels = le.fit_transform(labels)
                    labels = jnp.array(labels, dtype=jnp.int32)
                except:
                    raise ValueError("Could not encode string labels. Ensure labels are numeric.")
            else:
                labels = jnp.array(labels, dtype=jnp.float32)

            return texts, labels
