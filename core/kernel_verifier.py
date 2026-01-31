import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_aer import Aer
from dotenv import load_dotenv

# Ensure environment variables are loaded if needed for Qiskit Runtime
load_dotenv()

class QuantumKernelVerifier:
    def __init__(self, n_qubits=4, model_path="models/quantum_kernel.pkl"):
        self.n_qubits = n_qubits
        self.model_path = model_path
        
        # Quantum Components
        self.feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
        self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        
        # Classical Preprocessing & Classifier
        self.pca = PCA(n_components=n_qubits)
        self.scaler = MinMaxScaler(feature_range=(0, np.pi)) 
        self.svc = SVC(kernel='precomputed', probability=True)
        
        self.is_fitted = False
        self.train_data_transformed = None
        
        # Attempt to load existing model
        self.load_model()

    def _preprocess(self, data, fit=False):
        """Reduces dimension and scales data for quantum embedding."""
        # Ensure data is 2D
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
            
        if fit:
            # Handle case where n_samples < n_components
            if data.shape[0] < self.n_qubits:
                print(f"[QuantumKernel] Warning: Not enough samples ({data.shape[0]}) for PCA-{self.n_qubits}. Using slice fallback.")
                # Fallback: just take first n_qubits columns or pad
                # We assume input dim >> n_qubits (128 >> 4)
                data_pca = data[:, :self.n_qubits]
                
                # We can't really "fit" PCA here properly to preserve variance.
                # We will mark PCA as "not fitted" or just bypass it for now.
                # But self.transform needs PCA.
                # Hack: fit PCA on duplicated data? No.
                # Better: Just Mock PCA logic or store a flag.
                
                # To keep it compatible with 'transform' later (which expects a fitted PCA),
                # we ideally need to persist this decision.
                # For now, let's just FORCE usage of first n cols if PCA is not fitted/possible.
                self.use_pca_fallback = True
            else:
                self.pca.fit(data)
                self.use_pca_fallback = False
                data_pca = self.pca.transform(data)
                
            data_scaled = self.scaler.fit_transform(data_pca)
        else:
            if getattr(self, 'use_pca_fallback', False):
                 data_pca = data[:, :self.n_qubits]
            elif not hasattr(self.pca, 'components_'):
                 # Try to fallback if not fitted (should happen via load, but if load failed...)
                 data_pca = data[:, :self.n_qubits]
            else:
                data_pca = self.pca.transform(data)
            
            data_scaled = self.scaler.transform(data_pca)
            
        return data_scaled

    def fit(self, X_train, y_train):
        """
        Trains the Quantum Kernel SVC and saves the model.
        """
        print("[QuantumKernel] Preprocessing training data...")
        self.train_data_transformed = self._preprocess(X_train, fit=True)
        self.is_fitted = True
        
        print("[QuantumKernel] Computing Quantum Kernel Matrix...")
        kernel_matrix = self.kernel.evaluate(x_vec=self.train_data_transformed)
        
        print("[QuantumKernel] Training SVC...")
        self.svc.fit(kernel_matrix, y_train)
        print("[QuantumKernel] Training Complete.")
        
        self.save_model()
        return kernel_matrix

    def predict(self, image_features):
        """Returns verdict dict for a single image feature vector."""
        if not self.is_fitted:
            return {"verdict": "Uncertain", "confidence": 0.0, "details": "Model not fitted"}
            
        # Preprocess
        target_transformed = self._preprocess(image_features, fit=False)
        
        # Compute Kernel (similarity to support vectors)
        kernel_vec = self.kernel.evaluate(x_vec=target_transformed, y_vec=self.train_data_transformed)
        
        # Predict
        prediction = self.svc.predict(kernel_vec)[0]
        probs = self.svc.predict_proba(kernel_vec)[0] # [Prob(Fake), Prob(Authentic)]
        
        label = "Authentic" if prediction == 1 else "Fake"
        confidence = probs[1] if prediction == 1 else probs[0]
        
        return {
            "verdict": label,
            "confidence": float(confidence),
            "probability_authentic": float(probs[1]),
            "probability_fake": float(probs[0])
        }

    def save_model(self):
        """Saves the entire state (PCA, Scaler, SVC, Training Data) to pickle."""
        state = {
            "pca": self.pca,
            "scaler": self.scaler,
            "svc": self.svc,
            "train_data_transformed": self.train_data_transformed,
            "is_fitted": self.is_fitted,
            "use_pca_fallback": getattr(self, 'use_pca_fallback', False)
        }
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(state, f)
        print(f"[QuantumKernel] Model saved to {self.model_path}")

    def load_model(self):
        """Loads state from pickle if exists."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    state = pickle.load(f)
                self.pca = state["pca"]
                self.scaler = state["scaler"]
                self.svc = state["svc"]
                self.train_data_transformed = state["train_data_transformed"]
                self.is_fitted = state["is_fitted"]
                self.use_pca_fallback = state.get("use_pca_fallback", False)
                print(f"[QuantumKernel] Loaded model from {self.model_path}")
            except Exception as e:
                print(f"[QuantumKernel] Failed to load model: {e}")
                self.is_fitted = False
        else:
            print(f"[QuantumKernel] No existing model found at {self.model_path}. Need training.")
