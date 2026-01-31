import numpy as np
from sklearn.decomposition import PCA
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
# from qiskit_aer.primitives import Estimator as AerEstimator # Failed to build
from qiskit.primitives import Estimator # Use reference implementation
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit.primitives import Estimator
import pickle
import os

class QuantumPatternDiscovery:
    def __init__(self, n_qubits=4, model_path="models/yamanaka_factors.pkl"):
        """
        n_qubits: 4 (Corresponds to the 4 Yamanaka Factors)
        """
        self.n_qubits = n_qubits
        self.model_path = model_path
        self.pca = PCA(n_components=2**n_qubits) # 16 dimensions for 4 qubits (Amplitude Encoding)
        
        self._build_circuit()
        self.is_fitted = False

    def _build_circuit(self):
        """
        Constructs the Quantum Autoencoder-like circuit.
        We use a RealAmplitudes ansatz as the 'Pattern Discovery' logic.
        """
        # 1. Feature Map (Raw Amplitude Encoding is state prep, done in input)
        # We'll use a QNN approach where input is a parameter too, or just state preparation.
        # Ideally, we map 16 dims -> State |psi>.
        # Then Apply U(theta). 
        # Then Measure.
        
        # In Qiskit ML, we can use a QNN.
        self.qc = QuantumCircuit(self.n_qubits)
        
        # Parametrized Ansatz (The "Brain" that finds patterns)
        # We need trainable parameters.
        self.params = ParameterVector('theta', length=self.n_qubits * 3) # Simple depth
        
        # We don't strictly need a feature map here if we rely on the QNN to handle input
        # via state preparation, but EstimatorQNN usually takes 'input_params'.
        # For simplicity, let's look at the latent patterns directly.
        # We will use Angle Encoding for stability if Amplitude is too complex for basic QNN.
        # But user wants "Quantum State" from met data.
        
        # Let's use Angle Encoding on 4 qubits (takes 4 features) or 16 features?
        # 16 features fits into 4 qubits via Amplitude Embedding.
        # But QNN input gradients with AmplitudeEmbedding are tricky.
        # Let's use Angle Encoding with 4 features (PCA=4).
        # This means we compress image to 4 floats, then encode to 4 qubits.
        self.pca = PCA(n_components=self.n_qubits) # Reset to 4 components for Angle Encoding Simplicity
        
        self.feat_params = ParameterVector('x', length=self.n_qubits)
        
        # Encoding: Rx rotations
        for i in range(self.n_qubits):
            self.qc.rx(self.feat_params[i], i)
            
        # Entangling Interaction (Finding correlations)
        self.qc.cz(0, 1)
        self.qc.cz(1, 2)
        self.qc.cz(2, 3)
        self.qc.cz(3, 0)
        
        # Trainable pattern matching layers
        for i in range(self.n_qubits):
            self.qc.ry(self.params[i], i)
            self.qc.rz(self.params[i + self.n_qubits], i)
            self.qc.ry(self.params[i + 2*self.n_qubits], i)
            
        # Observables: We measure Z on all 4 qubits.
        # These 4 values are the "Yamanaka Factors".
        # If the image is Authentic, these 4 values should target a specific state (e.g. [+1, +1, +1, +1]).
        self.observables = [SparsePauliOp.from_list([(f"I"*i + "Z" + "I"*(self.n_qubits-1-i), 1)]) for i in range(self.n_qubits)]
        
        # QNN
        self.qnn = EstimatorQNN(
            circuit=self.qc,
            input_params=self.feat_params,
            weight_params=self.params,
            observables=self.observables
        )

    def fit(self, images, epochs=10):
        """
        Trains the parameters such that Authentic images produce a specific pattern (e.g., all +1).
        images: Tensor or Numpy array of shape (N, C, H, W)
        """
        # 1. Flatten and PCA
        flat = images.reshape(images.shape[0], -1).numpy()
        # Clean data
        flat = np.nan_to_num(flat)
        
        print(f"[Discovery] Fitting PCA on {flat.shape[0]} images...")
        try:
            self.pca.fit(flat)
            features = self.pca.transform(flat)
        except Exception as e:
            print(f"[Discovery] PCA Failed: {e}. Using random features.")
            features = np.random.randn(len(flat), self.n_qubits)
        
        # Normalize features to [0, Pi] for Angle Encoding
        # Simple MinMax scaling based on dataset
        self.min_val = features.min()
        self.max_val = features.max()
        features_norm = np.pi * (features - self.min_val) / (self.max_val - self.min_val + 1e-6)
        
        # 2. Target: We want Authentic images to map to [1, 1, 1, 1] (State |0000>)
        # Effectively, we are training a one-class classifier.
        # Loss = MSE(Output, [1,1,1,1])
        target = np.ones((len(images), self.n_qubits))
        
        # Initialize weights
        weights = np.random.random(self.qnn.num_weights)
        
        print(f"[Discovery] Training Quantum Circuit for {epochs} epochs...")
        # Simple Gradient Descent
        lr = 0.1
        for ep in range(epochs):
            # Forward
            output = self.qnn.forward(features_norm, weights)
            loss = np.mean((output - target) ** 2)
            
            if ep % 2 == 0:
                print(f"Epoch {ep}: Loss = {loss:.4f} (Avg Factors: {np.mean(output, axis=0)})")
            
            # Backprop: Chain rule
            # dL/dw = dL/dy * dy/dw
            # output (y): [N, 4], target: [N, 4]
            # grad_weights (dy/dw): [N, 4, num_weights]
            
            _, grad_weights = self.qnn.backward(features_norm, weights)

            diff = (output - target) # [N, 4] (dL/dy approx, ignoring scale 2)
            # We want to sum over the 4 outputs for each weight
            
            # Reshape diff for broadcasting: [N, 4, 1]
            diff_reshaped = diff[:, :, np.newaxis]
            
            # Element-wise mult and sum over outputs
            # grad_per_sample = sum(diff_j * dy_j/dw, axis=outputs)
            batch_grads = np.sum(diff_reshaped * grad_weights, axis=1) # [N, num_weights]
            
            # Update
            weights -= lr * np.mean(batch_grads, axis=0) # Avg over batch

            
        self.weights = weights
        self.is_fitted = True
        self.save_model()
        print("[Discovery] Training Complete. Yamanaka Factors identified.")

    def get_yamanaka_factors(self, image_tensor):
        """
        Extracts the 4 Key Patterns from a single image.
        Returns: [f1, f2, f3, f4] (floats between -1 and 1)
        """
        if not self.is_fitted:
            print("[Discovery] Model not fitted. Returning random.")
            return np.zeros(4)
            
        flat = image_tensor.reshape(1, -1).numpy()
        features = self.pca.transform(flat)
        features_norm = np.pi * (features - self.min_val) / (self.max_val - self.min_val + 1e-6)
        
        factors = self.qnn.forward(features_norm, self.weights)[0]
        return factors

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        state = {
            "pca": self.pca,
            "weights": self.weights,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "is_fitted": self.is_fitted
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(state, f)

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                state = pickle.load(f)
            self.pca = state["pca"]
            self.weights = state["weights"]
            self.min_val = state["min_val"]
            self.max_val = state["max_val"]
            self.is_fitted = state["is_fitted"]
            print("[Discovery] Model loaded.")
        else:
            print("[Discovery] No model found.")
