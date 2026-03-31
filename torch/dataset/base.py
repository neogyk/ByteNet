"""Abstract base class for data loaders."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np

from .config import DataConfig


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    def __init__(self, config: DataConfig):
        self.config = config

    @abstractmethod
    def load_data(self) -> Tuple[List[str], Union[np.ndarray, List[str]]]:
        """
        Load and return data.

        Returns:
            For classification/regression: (texts, labels)
            For generation: (source_texts, target_texts)
        """
        pass
