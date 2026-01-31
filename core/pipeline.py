import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from .ai_models import SiameseNetwork, Autoencoder
from .quantum_layer import QuantumVerifier
from .kernel_verifier import QuantumKernelVerifier

class VerificationPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Models
        self.siamese = SiameseNetwork().to(self.device)
        self.autoencoder = Autoencoder().to(self.device)
        self.quantum = QuantumVerifier()
        self.kernel_verifier = QuantumKernelVerifier()
        
        # Load Weights (Mocking for now if files don't exist)
        # In a real scenario, we would load .pth files here
        self.siamese.eval()
        self.autoencoder.eval()
        
        # Transforms
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def verify(self, ref_image, suspect_image):
        """
        Runs the full verification pipeline.
        
        Returns:
            dict: Contains scores and final verdict.
        """
        # Preprocess
        ref_tensor = self.transform(ref_image).unsqueeze(0).to(self.device)
        suspect_tensor = self.transform(suspect_image).unsqueeze(0).to(self.device)
        
        # 1. AI Similarity (Siamese)
        with torch.no_grad():
            feat1, feat2 = self.siamese(ref_tensor, suspect_tensor)
            similarity_score = torch.nn.functional.pairwise_distance(feat1, feat2).item()
            
            # Get features for Quantum Layer (flattened)
            ref_features = feat1.cpu().numpy().flatten()
            suspect_features = feat2.cpu().numpy().flatten()

        # 2. AI Anomaly (Autoencoder)
        with torch.no_grad():
            reconstructed = self.autoencoder(suspect_tensor)
            anomaly_score = torch.mean((suspect_tensor - reconstructed) ** 2).item()

        # 3. Quantum Fidelity (Standard QPE)
        try:
            qpe_result = self.quantum.verify(ref_features, suspect_features)
            quantum_score = qpe_result['fidelity']
        except Exception as e:
            print(f"Standard QPE Failed: {e}")
            qpe_result = {'fidelity': 0.0, 'counts': {}, 'top_phase': 0.0}
            quantum_score = 0.0
        
        # 3.5 Evolved Quantum Kernel Check (Self-Evolution)
        # We need the "Evolved" opinion.
        # We use suspect_features (128-dim from Siamese) directly.
        kernel_result = self.kernel_verifier.predict(suspect_features)
        evolved_verdict = kernel_result['verdict'] # Authentic/Fake
        evolved_conf = kernel_result['confidence']
        
        # 4. Final Verdict Logic
        is_match = similarity_score < 1.0
        is_authentic_style = anomaly_score < 5.0 
        is_quantum_verified = quantum_score > 0.8
        is_kernel_verified = (evolved_verdict == "Authentic")
        
        verdict = "UNCERTAIN"
        confidence = 0.0
        
        # Base Logic
        if is_match and is_authentic_style and is_quantum_verified:
            verdict = "AUTHENTIC"
            confidence = 0.95
        elif not is_match:
            verdict = "FAKE (Mismatch)"
            confidence = 0.9
        elif not is_authentic_style:
            verdict = "FAKE (Anomaly Detected)"
            confidence = 0.85
        elif not is_quantum_verified:
            verdict = "POTENTIAL FORGERY (Quantum Fail)"
            confidence = 0.75
            
        # Evolution Override/Boost
        # If Kernel has strong opinion (trained), it influences result
        if kernel_result['details'] != "Model not fitted":
            if is_kernel_verified and verdict != "AUTHENTIC":
                 # If everything else failed but Kernel says authentic? 
                 # Maybe not override completely but add note.
                 # But if Kernel says FAKE, we should listen.
                 pass
            if not is_kernel_verified:
                 if verdict == "AUTHENTIC":
                      verdict = "DOUBTFUL (Evolved Model Flagged Fake)"
                      confidence = evolved_conf
                 else:
                      confidence = max(confidence, evolved_conf)
                      
        # Adjust confidence based on QPE
        if verdict == "AUTHENTIC":
             confidence += (quantum_score * 0.05)

        return {
            "verdict": verdict,
            "confidence": round(confidence * 100, 2),
            "similarity_score": round(similarity_score, 4),
            "anomaly_score": round(anomaly_score, 4),
            "quantum_score": round(quantum_score, 4),
            "quantum_details": qpe_result,
            "evolved_analysis": kernel_result
        }
