import tenseal as ts
import numpy as np
import torch
import time
import pickle
import os

class HomomorphicEncryptor:
    """
    Homomorphic encryption class using TenSEAL for privacy-preserving facial recognition
    """
    def __init__(self, poly_modulus_degree=8192, context=None):
        """
        Initialize the homomorphic encryptor
        
        Args:
            poly_modulus_degree (int): Polynomial modulus degree for the encryption scheme
            context (ts.Context): Optional pre-created TenSEAL context
        """
        self.poly_modulus_degree = poly_modulus_degree
        
        if context is None:
            # Create a new context for CKKS scheme
            self.context = self._create_context()
        else:
            self.context = context
    
    def _create_context(self):
        """
        Create a TenSEAL context for the CKKS encryption scheme
        
        Returns:
            ts.Context: TenSEAL context for CKKS
        """
        # Create TenSEAL context
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        
        # Set scale
        context.global_scale = 2**40
        
        # Generate galois keys for vector rotation
        context.generate_galois_keys()
        
        return context
    
    def encrypt_vector(self, vector):
        """
        Encrypt a vector using CKKS
        
        Args:
            vector (np.ndarray): Vector to encrypt
        
        Returns:
            ts.CKKSVector: Encrypted vector
        """
        # Convert to numpy if it's a PyTorch tensor
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        
        # Ensure the vector is flattened
        vector = vector.flatten()
        
        # Encrypt the vector
        encrypted_vector = ts.ckks_vector(self.context, vector)
        
        return encrypted_vector
    
    def decrypt_vector(self, encrypted_vector):
        """
        Decrypt an encrypted vector
        
        Args:
            encrypted_vector (ts.CKKSVector): Encrypted vector
        
        Returns:
            np.ndarray: Decrypted vector
        """
        # Decrypt the vector
        decrypted_vector = encrypted_vector.decrypt()
        
        return np.array(decrypted_vector)
    
    def linear_transform(self, encrypted_vector, weight_matrix, bias=None):
        """
        Perform privacy-preserving linear transformation on encrypted data
        
        Args:
            encrypted_vector (ts.CKKSVector): Encrypted input vector
            weight_matrix (np.ndarray): Weight matrix for linear transformation
            bias (np.ndarray, optional): Bias vector
        
        Returns:
            ts.CKKSVector: Encrypted result after linear transformation
        """
        # Number of neurons in the output layer
        n_outputs = weight_matrix.shape[0]
        
        # Initialize list to store results for each neuron
        results = []
        
        # Process each output neuron
        for i in range(n_outputs):
            # Get weights for the current neuron
            weights = weight_matrix[i]
            
            # Perform dot product (weights * input)
            result = encrypted_vector * weights  # Element-wise multiplication
            result = result.sum()  # Sum all elements (dot product)
            
            # Add bias if provided
            if bias is not None:
                result = result + bias[i]
            
            results.append(result)
        
        # Combine results into a single encrypted vector
        encrypted_result = ts.ckks_vector(self.context, [r for r in results])
        
        return encrypted_result
    
    def save_context(self, path):
        """
        Save the encryption context to a file
        
        Args:
            path (str): Path to save the context
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save context
        with open(path, 'wb') as f:
            pickle.dump(self.context, f)
    
    @classmethod
    def load_context(cls, path):
        """
        Load an encryption context from a file
        
        Args:
            path (str): Path to the saved context
            
        Returns:
            HomomorphicEncryptor: A new encryptor with the loaded context
        """
        with open(path, 'rb') as f:
            context = pickle.load(f)
        
        return cls(context=context)
    
    def encrypt_model_input(self, model_input):
        """
        Encrypt input data for a model
        
        Args:
            model_input (np.ndarray or torch.Tensor): Input data
            
        Returns:
            ts.CKKSVector: Encrypted input data
        """
        return self.encrypt_vector(model_input)
    
    def private_inference(self, encrypted_input, model_weights, model_biases):
        """
        Perform privacy-preserving inference on encrypted data
        
        Args:
            encrypted_input (ts.CKKSVector): Encrypted input data
            model_weights (list): List of weight matrices for each layer
            model_biases (list): List of bias vectors for each layer
            
        Returns:
            ts.CKKSVector: Encrypted model output
        """
        # Current state starts with the input
        current = encrypted_input
        
        # Process each layer
        for i, (weights, biases) in enumerate(zip(model_weights, model_biases)):
            # Linear transform (weights * input + bias)
            current = self.linear_transform(current, weights, biases)
            
            # Note: Non-linear activations like ReLU are not directly applicable in HE
            # For a real implementation, approximations like polynomial approximations would be used
            
        return current
    
    def benchmark_encryption(self, vector_size=512, n_trials=5):
        """
        Benchmark encryption and decryption performance
        
        Args:
            vector_size (int): Size of the vector to encrypt
            n_trials (int): Number of trials for averaging
            
        Returns:
            dict: Dictionary with benchmark results
        """
        results = {
            'encryption_time': [],
            'decryption_time': [],
            'vector_size': vector_size
        }
        
        # Generate a random vector
        vector = np.random.rand(vector_size)
        
        # Benchmark encryption
        for _ in range(n_trials):
            start_time = time.time()
            encrypted = self.encrypt_vector(vector)
            results['encryption_time'].append(time.time() - start_time)
            
            # Benchmark decryption
            start_time = time.time()
            self.decrypt_vector(encrypted)
            results['decryption_time'].append(time.time() - start_time)
        
        # Calculate averages
        results['avg_encryption_time'] = sum(results['encryption_time']) / n_trials
        results['avg_decryption_time'] = sum(results['decryption_time']) / n_trials
        
        return results 