import os
import glob
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

class DataLoader:
    def __init__(self, data_dir, image_size=(32, 32)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(image_size),
            transforms.ToTensor() # Converts to [0, 1]
        ])

    def load_authentic_images(self, limit=None):
        """
        Loads authentic images from the directory.
        Returns a list of image tensors.
        """
        image_paths = glob.glob(os.path.join(self.data_dir, "*"))
        # Filter for image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in valid_extensions]
        
        if limit:
            image_paths = image_paths[:limit]

        images = []
        print(f"[DataLoader] Found {len(image_paths)} images in {self.data_dir}")
        for p in image_paths:
            try:
                img = Image.open(p)
                tensor = self.transform(img)
                # Flatten for quantum encoding compatibility later if needed, 
                # but keep as tensor (C, H, W) for now.
                images.append(tensor)
            except Exception as e:
                print(f"[DataLoader] Skipped {p}: {e}")
        
        if not images:
             print("[DataLoader] Warning: No images loaded.")
             return torch.empty(0)

        return torch.stack(images)

    def load_single_image(self, image_path):
        """Loads and processes a single image."""
        try:
            img = Image.open(image_path)
            return self.transform(img).unsqueeze(0) # Batch dim
        except Exception as e:
            print(f"[DataLoader] Failed to load {image_path}: {e}")
            return None
