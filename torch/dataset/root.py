"""ROOT file data loader (HEP format)."""

from typing import List, Tuple

import numpy as np

from .base import DataLoader


class ROOTDataLoader(DataLoader):
    """Load data from ROOT files (HEP format)."""

    def load_data(self) -> Tuple[List[str], np.ndarray]:
        try:
            import uproot
        except ImportError:
            raise ImportError("uproot is required for ROOT files. Install with: pip install uproot awkward")

        # Open ROOT file
        file = uproot.open(self.config.data_path)

        if not self.config.root_tree_name:
            # Use first tree
            self.config.root_tree_name = file.keys()[0].split(';')[0]

        tree = file[self.config.root_tree_name]

        # Read branches
        branches = self.config.root_branches if self.config.root_branches else tree.keys()
        arrays = tree.arrays(branches, library="np")

        # Convert arrays to text representations
        texts = []
        n_events = len(arrays[branches[0]])

        for i in range(n_events):
            # Concatenate all branch values for this event
            event_str = " | ".join([f"{branch}:{arrays[branch][i]}" for branch in branches if branch != self.config.label_column])
            texts.append(event_str)

        # Extract labels
        if self.config.label_column and self.config.label_column in arrays:
            labels = np.array(arrays[self.config.label_column], dtype=np.float32)
        else:
            labels = np.zeros(n_events)

        return texts, labels
