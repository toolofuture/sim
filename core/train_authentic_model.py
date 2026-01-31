import os
import sys
# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import torch
import numpy as np
import pickle
from PIL import Image
from core.pipeline import VerificationPipeline
from tqdm import tqdm

def train_authentic_model(data_dir="data/authentic", output_path="models/quantum_projection.pkl"):
    print(f"Loading Authentic Data from {data_dir}...")
    
    # Initialize Pipeline (Model only)
    pipeline = VerificationPipeline()
    model = pipeline.siamese
    model.eval()
    transform = pipeline.transform
    device = pipeline.device
    
    # Collect Features
    features = []
    
    # Support multiple extensions
    exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    print(f"Found {len(files)} images.")
    
    if len(files) == 0:
        print("Error: No images found!")
        return

    # Batch processing could be faster but let's keep it simple
    with torch.no_grad():
        for fpath in tqdm(files):
            try:
                img = Image.open(fpath).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                feature = model.forward_one(tensor).cpu().numpy().flatten()
                features.append(feature)
            except Exception as e:
                print(f"Skipping {fpath}: {e}")
                
    X = np.array(features) # (N, 128)
    print(f"Feature Matrix Shape: {X.shape}")
    
    # PCA using torch
    # We want to reduce to 4 dimensions (for 2-qubit state)
    k = 4
    X_torch = torch.from_numpy(X)
    
    # 1. Compute Mean
    mean_vec = torch.mean(X_torch, dim=0)
    
    # 2. Centering is usually good for PCA, but for Amplitude Encoding we need a vector that we can normalize.
    # If we center, the mean becomes 0, which cannot be normalized.
    # However, we want the "Authentic Essence" to be captured.
    # The First Principal Component of uncentered data captures the "Average Direction".
    # So we will perform SVD on the raw data (or uncentered PCA).
    
    # Compute SVD of X^T (128, N) or X (N, 128) to find dominant directions
    # U, S, V = torch.pca_lowrank(X_torch, q=k, center=False)
    # V is (128, k) -> Projection matrix
    
    # Let's use sklearn-style PCA components logic manually with torch
    # Calculate covariance matrix if N is large, or just SVD
    # torch.pca_lowrank with center=False does SVD on X.
    
    U, S, V = torch.pca_lowrank(X_torch, q=k, center=False)
    
    projection_matrix = V.cpu().numpy() # (128, 4)
    
    # Authentic "State" is the Mean Vector projected to this subspace
    mean_vec_np = mean_vec.cpu().numpy()
    projected_mean = mean_vec_np @ projection_matrix # (4,)
    
    # Normalize to create the Target Quantum State
    norm = np.linalg.norm(projected_mean)
    authentic_state = projected_mean / norm
    
    print(f"Authentic State (4-dim): {authentic_state}")
    
    # Save Artifacts
    data = {
        "projection_matrix": projection_matrix,
        "authentic_state": authentic_state,
        "mean_vector_original": mean_vec_np
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
        
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Use the absolute path provided by user
    data_path = "/Users/kangsikseo/Downloads/mt"
    train_authentic_model(data_path)
