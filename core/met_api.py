import requests
import os
import time
import concurrent.futures
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO
import random
import numpy as np

# Configuration
MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

class MetArtFetcher:
    def __init__(self, save_dir="data/real_art"):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_painting_ids(self, count=100):
        """Fetch object IDs for public domain paintings."""
        print("Searching for public domain paintings...")
        search_url = f"{MET_API_BASE}/search"
        params = {
            "q": "painting",
            "medium": "Paintings",
            "hasImages": "true",
            "isPublicDomain": "true"
        }
        
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            object_ids = data.get('objectIDs', [])
            print(f"Found {len(object_ids)} paintings. Selecting top {count}.")
            return object_ids[:count]
        except Exception as e:
            print(f"Error searching Met API: {e}")
            return []

    def download_image(self, object_id):
        """Download a single image."""
        object_url = f"{MET_API_BASE}/objects/{object_id}"
        try:
            resp = requests.get(object_url, timeout=10)
            if resp.status_code != 200: return None
            
            data = resp.json()
            img_url = data.get('primaryImageSmall')
            if not img_url: return None
            
            img_resp = requests.get(img_url, timeout=10)
            if img_resp.status_code != 200: return None
            
            img = Image.open(BytesIO(img_resp.content)).convert('RGB')
            save_path = os.path.join(self.save_dir, f"{object_id}.jpg")
            img.save(save_path, "JPEG", quality=85)
            return save_path
        except Exception:
            return None

    def fetch_authentic_samples(self, count=10):
        """Fetch a batch of authentic images."""
        ids = self.get_painting_ids(count)
        downloaded_paths = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self.download_image, ids))
        
        return [r for r in results if r is not None]

class FakeGenerator:
    """Generates synthetic fakes from authentic images for testing."""
    
    @staticmethod
    def generate_fake(image_path, output_path=None):
        """
        Applies distortions to create a 'fake' version.
        Distortions: Blur, Noise, Color Shift, Rotation.
        """
        try:
            img = Image.open(image_path).convert('RGB')
            
            # 1. Gaussian Blur (simulate loss of detail)
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.0)))
            
            # 2. Color Jitter (simulate cheap pigment)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
            
            # 3. Add Noise
            img_np = np.array(img)
            noise = np.random.normal(0, 5, img_np.shape).astype(np.uint8)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            
            if output_path:
                img.save(output_path)
            
            return img
        except Exception as e:
            print(f"Error generating fake: {e}")
            return None
