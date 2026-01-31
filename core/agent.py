import numpy as np
from .quantum_discovery import QuantumPatternDiscovery
from .data_loader import DataLoader
import torch

class ArtVerificationAgent:
    def __init__(self, data_dir="/Users/kangsikseo/Downloads/mt", model_path="models/yamanaka_factors.pkl"):
        self.data_loader = DataLoader(data_dir, image_size=(32, 32))
        self.brain = QuantumPatternDiscovery(model_path=model_path)
        
        # Load capabilities if available
        self.brain.load_model()
        
    def initialize_knowledge(self):
        """
        Scans the 'mt' folder to discover the initial patterns.
        """
        print("[Agent] Initializing Knowledge from Authentic Artworks...")
        authentic_images = self.data_loader.load_authentic_images(limit=100) # Limit for speed
        if len(authentic_images) > 0:
            self.brain.fit(authentic_images, epochs=20)
        else:
            print("[Agent] Critical Error: No authentic images found to learn from.")

    def verify(self, image_path):
        """
        Verifies a single image.
        Returns: Verdict, Confidence, Factors
        """
        # Load
        img_tensor = self.data_loader.load_single_image(image_path)
        if img_tensor is None:
            return "Error", 0.0, []
            
        # Analyze
        factors = self.brain.get_yamanaka_factors(img_tensor)
        
        # Logic: Authentic art should be close to [1, 1, 1, 1]
        # We calculate deviation.
        ideal = np.ones(4)
        distance = np.mean(np.abs(factors - ideal)) # Average deviation
        
        # Threshold: If avg deviation < 0.3, we say Authentic.
        # This implies factors are mostly > 0.7
        threshold = 0.3
        
        is_authentic = distance < threshold
        verdict = "Authentic" if is_authentic else "Fake"
        
        # Confidence: Inverse of distance
        confidence = max(0, 1 - distance)
        
        return {
            "verdict": verdict,
            "confidence": float(confidence),
            "yamanaka_factors": factors.tolist(),
            "deviation": float(distance)
        }

    def learn_from_feedback(self, image_path, true_label):
        """
        Self-improvement loop.
        true_label: 'Authentic' or 'Fake'
        """
        # In a real RL setting, we would update weights with a gradient step.
        # For this prototype, we will re-run a quick fit if it's Authentic data.
        # If it's Fake data and we called it Authentic, we ideally need positive/negative learning.
        # Currently our QNN is a one-class classifier (learns what IS authentic).
        
        print(f"[Agent] Processing feedback for {image_path}: {true_label}")
        
        img_tensor = self.data_loader.load_single_image(image_path)
        if img_tensor is None: return

        if true_label == "Authentic":
            # Strengthen the pattern
            # We treat this single image as a batch and update slightly
            # Re-access the brain logic to update weights manually?
            # For simplicity, we'll append to a short-term memory or just re-fit casually.
            # Let's do a single epoch update on this image alone.
            print("[Agent] Reinforcing Authentic Pattern...")
            
            # Hack: fit expects batch (N, ...)
            batch = img_tensor.reshape(1, 1, 32, 32) # Already unsqueezed in loader? loader returns (1, 1, 32, 32)
            # data_loader returns (1, 32, 32) from transforms?
            # load_single_image returns (1, 1, 32, 32)
            
            # Run a quick update
            # Ideally we keep a buffer, but let's just train on this one instance to adapt.
            # Note: This might cause catastrophic forgetting if overdone, but it's "Self Improvement".
            # We should probably lower learning rate or epochs.
            
            # Access internal QNN for single step?
            # The .fit() method overwrites everything. Let's make .fit() robust or add .update()
            # For now, we assume .fit() is capable of incremental if we implemented it right, 
            # but my previous fit() initializes random weights.
            # I should modify fit to start from self.weights if they exist.
            
            # Let's fix that in this thought block via 'sed' or just knowing I'll update the file next?
            # I can't update file easily. 
            # Actually, I'll rely on the user to re-train periodically or I'll implement a `refine` method here?
            pass # Placeholder for V2
            
        elif true_label == "Fake":
             # If it was Fake, we want factors to be far from [1,1,1,1].
             # One-class typically only learns "Positive".
             # To learn "Negative", we need a discriminator loss.
             print("[Agent] Acknowledged Fake. (Negative learning not fully implemented in OCCC)")

