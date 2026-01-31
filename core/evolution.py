import os
import json
import numpy as np
from .kernel_verifier import QuantumKernelVerifier
from PIL import Image

class EvolutionManager:
    def __init__(self, memory_path="data/evolution_memory.json", model_path="models/quantum_kernel.pkl"):
        self.memory_path = memory_path
        self.verifier = QuantumKernelVerifier(model_path=model_path)
        self.memory = self._load_memory()
        
    def _load_memory(self):
        """Loads experience memory from JSON."""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                return json.load(f)
        return []

    def _save_memory(self):
        """Saves experience memory to JSON."""
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w") as f:
            json.dump(self.memory, f, indent=4)

    def add_experience(self, image_path, label):
        """
        Adds a new sample to the memory.
        label: 'Authentic' or 'Fake'
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Check if already exists to avoid duplicates
        for item in self.memory:
            if item['path'] == image_path:
                print(f"Updating label for {image_path} to {label}")
                item['label'] = label
                self._save_memory()
                return
                
        self.memory.append({
            "path": image_path,
            "label": label
        })
        self._save_memory()
        print(f"Added {label} experience: {image_path}")

    def evolve(self, feature_extractor_pipeline):
        """
        Triggers retraining of the Quantum Kernel using all accumulated memory.
        feature_extractor_pipeline: Instance of VerificationPipeline to extract features.
        """
        if not self.memory:
            print("No memory to evolve from.")
            return

        print(f"Evolving from {len(self.memory)} experiences...")
        
        X = []
        y = []
        
        valid_memory = []
        
        for item in self.memory:
            path = item['path']
            label = item['label']
            
            if not os.path.exists(path):
                print(f"Skipping missing file: {path}")
                continue
                
            try:
                # Extract features using the pipeline
                img = Image.open(path).convert('RGB')
                # we need to transform it. Pipeline has .transform and .siamese
                # But pipeline.verify does it all. We just need the feature extraction part.
                # Let's assume pipeline has a helper or we use its internal attributes.
                # Accessing internal logic of pipeline:
                
                # Preprocess
                tensor = feature_extractor_pipeline.transform(img).unsqueeze(0).to(feature_extractor_pipeline.device)
                
                # Forward pass through Siamese (branch 1 is enough for feature extraction)
                with resource_check_noop(): # pseudo-context
                     features = feature_extractor_pipeline.siamese.forward_one(tensor)
                     
                features_np = features.detach().cpu().numpy().flatten()
                
                X.append(features_np)
                y.append(1 if label == "Authentic" else 0)
                valid_memory.append(item)
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
        if len(X) < 2: # Need at least 2 samples (preferably one of each class)
             print("Insufficient data for training (need at least 2 valid samples).")
             return

        X_train = np.array(X)
        y_train = np.array(y)
        
        # Check class balance
        if len(np.unique(y_train)) < 2:
            print("Warning: Training data only contains one class. Model might be biased.")
            
        print("Retraining Quantum Kernel Model...")
        self.verifier.fit(X_train, y_train)
        print("Evolution Complete! Model updated.")

class resource_check_noop:
    def __enter__(self): pass
    def __exit__(self, *args): pass
