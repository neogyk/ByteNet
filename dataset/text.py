"""Text/CSV/TSV data loader."""

from pathlib import Path
from typing import List, Tuple, Union

import jax.numpy as jnp
from jaxtyping import Array

from .base import DataLoader


class TextDataLoader(DataLoader):
    """Load data from text files."""

    def load_data(self) -> Tuple[List[str], Union[Array, List[str]]]:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required. Install with: pip install pandas")

        # Support CSV and TSV
        if self.config.data_path.endswith('.csv'):
            df = pd.read_csv(self.config.data_path)
        elif self.config.data_path.endswith('.tsv'):
            df = pd.read_csv(self.config.data_path, sep='\t')
        elif self.config.data_path.endswith('.txt'):
            # Simple text file with one example per line
            with open(self.config.data_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            # Assume unlabeled or labels in separate file
            if self.config.label_column and Path(self.config.label_column).exists():
                with open(self.config.label_column, 'r') as f:
                    labels = [float(line.strip()) for line in f]
                return lines, jnp.array(labels)
            return lines, jnp.zeros(len(lines))  # Dummy labels
        else:
            raise ValueError(f"Unsupported text file format: {self.config.data_path}")

        # Same logic as Parquet
        texts = df[self.config.text_column].astype(str).tolist() if self.config.text_column else df.iloc[:, 0].astype(str).tolist()

        if self.config.target_column:
            targets = df[self.config.target_column].astype(str).tolist()
            return texts, targets
        else:
            labels = df[self.config.label_column].values if self.config.label_column in df else df.iloc[:, 1].values
            labels = jnp.array(labels, dtype=jnp.float32)
            return texts, labels
