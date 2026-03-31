"""Data loading configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data loading configuration."""
    data_path: str = ""
    data_format: str = "parquet"  # "root", "parquet", "image", "text", "huggingface"

    # Column/field specifications
    text_column: Optional[str] = None
    label_column: str = "label"
    target_column: Optional[str] = None  # For generation

    # ROOT-specific
    root_tree_name: Optional[str] = None
    root_branches: List[str] = field(default_factory=list)

    # Parquet-specific
    parquet_columns: List[str] = field(default_factory=list)

    # Image-specific
    image_dir: Optional[str] = None
    image_label_file: Optional[str] = None
    image_preprocessing: str = "raw"  # "raw", "normalize", "augment"

    # HuggingFace-specific
    hf_dataset_name: Optional[str] = None  # e.g. "codeparrot/apps"
    hf_subset: Optional[str] = None  # Dataset config/subset name
    hf_split_train: str = "train"
    hf_split_test: str = "test"
    hf_difficulties: Optional[List[str]] = None  # For APPS: filter by difficulty
    hf_solutions_column: Optional[str] = None  # Column with JSON-encoded solutions

    # Task-specific
    num_classes: Optional[int] = None
    max_target_length: int = 128
