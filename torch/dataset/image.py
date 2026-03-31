"""Image data loader."""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .base import DataLoader


class ImageDataLoader(DataLoader):
    """Load image data."""

    def load_data(self) -> Tuple[List[str], np.ndarray]:
        try:
            from PIL import Image
            import pandas as pd
        except ImportError:
            raise ImportError("PIL is required for image loading. Install with: pip install Pillow pandas")

        # Load image paths and labels from file
        if self.config.image_label_file:
            df = pd.read_csv(self.config.image_label_file)
            image_paths = df.iloc[:, 0].tolist()
            labels = df.iloc[:, 1].values
        else:
            # List all images in directory
            image_dir = Path(self.config.image_dir or self.config.data_path)
            image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
            labels = np.zeros(len(image_paths))  # Dummy labels

        # Convert images to byte sequences
        texts = []
        for img_path in image_paths:
            full_path = Path(self.config.image_dir) / img_path if self.config.image_dir else Path(img_path)

            if self.config.image_preprocessing == "raw":
                # Read raw image bytes
                with open(full_path, 'rb') as f:
                    img_bytes = f.read()
                # Convert to string representation
                byte_str = " ".join([str(b) for b in img_bytes[:1000]])  # Truncate if too long
                texts.append(byte_str)
            else:
                # Load and process image
                img = Image.open(full_path).convert("RGB")
                img_array = np.array(img)

                if self.config.image_preprocessing == "normalize":
                    img_array = img_array / 255.0

                # Flatten and convert to string
                pixels = img_array.flatten()[:1000]  # Truncate
                pixel_str = " ".join([str(int(p)) if self.config.image_preprocessing == "raw" else str(float(p)) for p in pixels])
                texts.append(pixel_str)

        return texts, np.array(labels)
