import os
import pickle
import numpy as np
from dotenv import load_dotenv

# Qiskit Imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, StatePreparation
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

class QuantumVerifier:
    def __init__(self, model_path="models/quantum_projection.pkl", force_simulator=False):
        load_dotenv()
        self.force_simulator = force_simulator
        self.service = self._connect_service() if not force_simulator else None
        self.model = self._load_model(model_path)
        self.backend = self._get_least_busy_backend() if not force_simulator else None
        
    def _connect_service(self):
        token = os.getenv("IBM_QUANTUM_TOKEN")
        crn = os.getenv("IBM_QUANTUM_CRN")
        
        # 1. Try Credentials from Env
        if token:
            try:
                if crn:
                    print("Connecting via IBM Cloud (CRN provided)...")
                    return QiskitRuntimeService(channel="ibm_cloud", instance=crn, token=token)
                else:
                    print("Connecting via IBM Quantum Platform...")
                    return QiskitRuntimeService(channel="ibm_quantum", token=token)
            except Exception as e:
                print(f"Env Auth Failed: {e}")

        # 2. Try Saved Account
        try:
             print("Attempting to load saved Qiskit account...")
             service = QiskitRuntimeService()
             print("Saved account loaded successfully.")
             return service
        except Exception as e:
             print(f"Saved Account Load Failed: {e}")
             
        print("No valid Qiskit credentials found.")
        return None

    def _load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _get_least_busy_backend(self):
        if not self.service:
            return None
        
        try:
            # Filter for real quantum hardware with >= 5 qubits
            service = self.service
            backends = service.backends(simulator=False, operational=True, min_num_qubits=5)
            if not backends:
                print("No real backends available. Using simulator.")
                return None
            
            # Find least busy
            least_busy = min(backends, key=lambda b: b.status().pending_jobs)
            print(f"Selected Backend: {least_busy.name}")
            return least_busy
        except Exception as e:
            print(f"Backend Selection Error: {e}")
            return None

    def _build_oracle(self, target_state):
        """
        Constructs an Oracle U such that U|target> = -|target> and U|other> = |other>.
        This is a Grover Diffusion Operator relative to the target state.
        S_target = I - 2|target><target|
        """
        dim = len(target_state)
        # Ensure normalization for unitarity
        norm = np.linalg.norm(target_state)
        if norm > 1e-9:
            target_state = target_state / norm
            
        # Outer product
        projector = np.outer(target_state, np.conj(target_state))
        identity = np.eye(dim)
        oracle_matrix = identity - 2 * projector
        
        # Convert to Operator
        return Operator(oracle_matrix)

    def run_qpe(self, feature_vector):
        """
        Runs Quantum Phase Estimation on the suspect image features.
        
        Args:
            feature_vector (np.array): 128-dim Feature Vector from ResNet.
            
        Returns:
            dict: {
                'fidelity': float (0-1),
                'counts': dict (raw measurement counts),
                'top_phase': float (0-1)
            }
        """
        # 1. Project Feature Vector to 4-dim Subspace
        proj_matrix = self.model["projection_matrix"] # (128, 4)
        mean_vec = self.model["mean_vector_original"]
        
        # We project the *direction* relative to the mean? 
        # Or just project the raw vector? 
        # Train script did essentially Raw -> PCA.
        # So we project raw.
        
        projected = feature_vector @ proj_matrix # (4,)
        # Use complex128 for high precision to satisfy Qiskit strict checks
        suspect_state = np.array(projected, dtype=np.complex128)
        
        norm = np.linalg.norm(suspect_state)
        if norm < 1e-9:
             suspect_state = np.array([1, 0, 0, 0], dtype=np.complex128)
        else:
             suspect_state = suspect_state / norm
        
        # Convert to list
        suspect_state = suspect_state.tolist()
            
        # 2. Define Circuits
        # Precision Qubits: 3 (q0, q1, q2)
        # State Qubits: 2 (q3, q4) - needed for 4-dim state
        qr_precision = QuantumRegister(3, 'precision')
        qr_state = QuantumRegister(2, 'state')
        cr = ClassicalRegister(3, 'meas')
        qc = QuantumCircuit(qr_precision, qr_state, cr)
        
        # A. Initialization
        # Encode Suspect State into State Qubits
        init_gate = StatePreparation(suspect_state)
        qc.append(init_gate, qr_state)
        
        # Hadoop on Precision (Superposition)
        qc.h(qr_precision)
        
        # B. Controlled Unitary Operations
        # Target State (Authentic)
        authentic_state = self.model['authentic_state']
        oracle_op = self._build_oracle(authentic_state)
        
        # We need phase 0.5 for Authentic.
        # U = S_target. eigenvalues: -1 (phase 0.5) for target, 1 (phase 0) for orthogonal.
        # CU^1 -> apply once
        # CU^2 -> apply twice (Identity)
        # CU^4 -> apply four times (Identity)
        
        # CU^1 controlled by q0 (LSB of precision? No, MSB power depends on convention)
        # Qiskit QPE: 
        # q0 controls U^(2^0)
        # q1 controls U^(2^1)
        # q2 controls U^(2^2)
        
        # U = Oracle. U^2 = I.
        # So q1 and q2 control Identity (do nothing).
        # Only q0 controls U.
        
        # Wait, using 3 precision qubits:
        # q0 (2^2 = 4) ??? No.
        # Standard QPE:
        # qubit k controls U^(2^k).
        # k=0 (LSB): U^1.
        # k=1: U^2 = I.
        # k=2: U^4 = I.
        
        # So effective circuit:
        # q0 (LSB) --o-- U -- 
        # q1 -------I-------
        # q2 -------I-------
        # Then Inverse QFT.
        
        # Let's add the Controlled-U gate
        # We need to turn Operator into Gate to control it
        controlled_U = oracle_op.to_instruction().control(1)
        
        # Apply C-U from q0 to state
        qc.append(controlled_U, [qr_precision[0]] + list(qr_state))
        
        # C. Inverse QFT
        iqft = QFT(3, inverse=True).to_gate()
        qc.append(iqft, qr_precision)
        
        # D. Measurement
        qc.measure(qr_precision, cr)
        
        # 3. Execution
        if self.backend:
            print("Submitting to Real Backend...")
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=3)
            isa_circuit = pm.run(qc)
            
            # 2048 shots for good statistics
            sampler = Sampler(mode=self.backend)
            job = sampler.run([isa_circuit], shots=2048)
            print(f"Job ID: {job.job_id()}")
            result = job.result()
            # Qiskit Runtime returns BitArray or similar structure. 
            # pub_result.data.meas.get_counts()
            pub_result = result[0]
            counts = pub_result.data.meas.get_counts()
        else:
            print("Simulating locally (Aer)...")
            from qiskit_aer import Aer
            backend = Aer.get_backend('qasm_simulator')
            # Transpile for simulator
            from qiskit import transpile
            # Decompose High Level objects like StatePreparation
            qc = qc.decompose(gates_to_decompose=["state_preparation"])
            tqc = transpile(qc, backend)
            job = backend.run(tqc, shots=2048)
            counts = job.result().get_counts()
            
        return self._process_results(counts)

    def _process_results(self, counts):
        total = sum(counts.values())
        
        # Ideal Authentic Phase = 0.5
        # 3 qubits. 0.5 = 1/2. 
        # Binary fraction 0.100
        # Qiskit QFT mapping:
        # |y_1 y_2 ... y_n> -> 0.y_1...y_n
        # If result is '100' (binary 4 in decimal if read standard)?
        # Little Endian (q2 q1 q0):
        # 100 -> q2=1, q1=0, q0=0.
        # Phase = 1*2^-1 + 0 + 0 = 0.5.
        # So we look for key '100'.
        
        target_key = '100'
        
        # Handling potentially missing key
        target_counts = counts.get(target_key, 0)
        fidelity = target_counts / total
        
        # Find top phase
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_bitstring = sorted_counts[0][0]
        # Convert bitstring to phase
        # '100' -> 0.5
        # '000' -> 0.0
        # '010' -> 0.25 (q1=1 -> 2^-2)
        phase_val = 0.0
        # Reverse string because qiskit key is q_n...q_0
        # q2 is index 0 in string '100'
        for i, bit in enumerate(top_bitstring):
            if bit == '1':
                phase_val += 2**(-(i+1))
                
        return {
            "fidelity": fidelity,
            "counts": counts, # Pass raw counts for plotting
            "top_phase": phase_val
        }

    def verify(self, ref_features_unused, suspect_features):
        # We ignore ref_features provided by pipeline because we use the pre-trained model
        return self.run_qpe(suspect_features)
