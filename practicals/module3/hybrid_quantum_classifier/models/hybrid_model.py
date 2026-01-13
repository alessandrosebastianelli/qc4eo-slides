"""
Hybrid Quantum-Classical Classifier

This module combines classical CNN feature extraction with quantum processing
and classical classification into a single end-to-end trainable model.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import pennylane as qml

import sys

sys.path += ['..', '.']

from .cnn_extractor import CNNFeatureExtractor
from quantum.encoding import ENCODING_MAP
from quantum.circuits import CIRCUIT_MAP, get_weight_shape


def create_quantum_circuit(device, n_qubits):
    """
    Create a quantum circuit as a PennyLane QNode.
    
    A QNode is a quantum function that can be called like a regular function
    but executes on a quantum device. It's differentiable and integrates
    seamlessly with PyTorch.
    
    Args:
        device: PennyLane quantum device
        n_qubits: Number of qubits in the circuit
        
    Returns:
        QNode: A quantum circuit function
    """
    @qml.qnode(device, interface='torch', diff_method='backprop')
    def quantum_circuit(inputs, weights, encoding_type, circuit_type):
        """
        Complete quantum circuit with encoding + variational layers.
        
        Args:
            inputs: Classical features to encode (tensor)
            weights: Variational parameters (tensor)
            encoding_type: Type of encoding to use ('angle', 'amplitude', etc.)
            circuit_type: Type of circuit to use ('strongly_entangling', etc.)
            
        Returns:
            list: Expectation values of Pauli-Z on each qubit
        """
        wires = range(n_qubits)
        
        # Step 1: Encode classical data into quantum state
        encoding_func = ENCODING_MAP[encoding_type]
        encoding_func(inputs, wires)
        
        # Step 2: Apply variational quantum circuit
        circuit_func = CIRCUIT_MAP[circuit_type]
        circuit_func(weights, wires)
        
        # Step 3: Measure expectation values (quantum → classical)
        # Returns one value per qubit in range [-1, 1]
        return [qml.expval(qml.PauliZ(wire)) for wire in wires]
    
    return quantum_circuit


class HybridQuantumClassifier(pl.LightningModule):
    """
    Hybrid Quantum-Classical Classifier using PyTorch Lightning.
    
    Architecture:
        Input Image → CNN Feature Extractor → Projection Layer → 
        Quantum Circuit (batch processing) → Classical Classifier → Output
    
    The model processes images through 4 stages:
    1. CNN: Extracts spatial features from images
    2. Projection: Maps features to quantum input dimension
    3. Quantum: Processes features through quantum circuit
    4. Classifier: Maps quantum outputs to class predictions
    
    Args:
        config: Configuration object with all hyperparameters
        quantum_device: PennyLane quantum device
    """
    
    def __init__(self, config, quantum_device):
        """
        Initialize the hybrid model.
        
        Args:
            config: Configuration object (from config.py)
            quantum_device: PennyLane quantum device
        """
        super(HybridQuantumClassifier, self).__init__()
        
        # Save hyperparameters (useful for checkpointing)
        self.save_hyperparameters(ignore=['quantum_device'])
        
        self.config = config
        self.quantum_device = quantum_device
        
        # ==================== CLASSICAL COMPONENTS ====================
        
        # CNN Feature Extractor
        # Input: (batch, 3, 64, 64) → Output: (batch, 256)
        self.cnn = CNNFeatureExtractor(output_dim=256)
        
        # Projection layer to quantum input dimension
        # Maps CNN features to the size expected by quantum circuit
        quantum_input_dim = config.quantum_input_dim
        self.quantum_projection = nn.Linear(256, quantum_input_dim)
        
        # ==================== QUANTUM COMPONENTS ====================
        
        # Calculate quantum circuit weight shape
        weight_shape = get_weight_shape(
            config.CIRCUIT_TYPE,
            config.N_QUBITS,
            config.N_LAYERS
        )
        
        # Initialize quantum circuit weights as trainable parameters
        # These are the variational parameters that get optimized
        self.quantum_weights = nn.Parameter(
            torch.randn(weight_shape) * 0.1  # Small random initialization
        )
        
        # Create the quantum circuit
        self.quantum_circuit = create_quantum_circuit(
            quantum_device,
            config.N_QUBITS
        )
        
        # ==================== CLASSICAL HEAD ====================
        
        # Final classifier: quantum outputs → class predictions
        # Input: n_qubits features → Output: n_classes logits
        self.classifier = nn.Sequential(
            nn.Linear(config.N_QUBITS, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(64, config.N_CLASSES)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Storage for validation and test outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x):
        """
        Forward pass through the entire hybrid model.
        
        Args:
            x: Input images tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Tensor of shape (batch_size, n_classes) with class logits
        """
        # Stage 1: CNN Feature Extraction
        # (batch, 3, 64, 64) → (batch, 256)
        features = self.cnn(x)
        
        # Stage 2: Project to quantum input dimension
        # (batch, 256) → (batch, quantum_input_dim)
        quantum_input = self.quantum_projection(features)
        
        # Stage 3: Quantum Processing (sample by sample)
        # Process each sample in the batch through the quantum circuit
        batch_size = x.shape[0]
        quantum_outputs = []
        
        for i in range(batch_size):
            # Execute quantum circuit for this sample
            q_out = self.quantum_circuit(
                quantum_input[i],
                self.quantum_weights,
                self.config.ENCODING,
                self.config.CIRCUIT_TYPE
            )
            # Stack the expectation values
            quantum_outputs.append(torch.stack(q_out))
        
        # Stack all quantum outputs into a batch
        # (batch, n_qubits)
        quantum_outputs = torch.stack(quantum_outputs).float()
        
        # Stage 4: Classical Classification
        # (batch, n_qubits) → (batch, n_classes)
        output = self.classifier(quantum_outputs)
        
        return output
    
    def training_step(self, batch, batch_idx):
        """
        Training step (called by PyTorch Lightning).
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch
            
        Returns:
            Loss value for this batch
        """
        x, y = batch
        
        # Forward pass
        y_hat = self(x)
        
        # Compute loss
        loss = self.criterion(y_hat, y)
        
        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics (visible in progress bar and logger)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step (called by PyTorch Lightning).
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch
            
        Returns:
            Loss value for this batch
        """
        x, y = batch
        
        # Forward pass
        y_hat = self(x)
        
        # Compute loss
        loss = self.criterion(y_hat, y)
        
        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        # Store predictions for later analysis
        self.validation_step_outputs.append({
            'preds': preds,
            'targets': y
        })
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step (called by PyTorch Lightning).
        
        Args:
            batch: Tuple of (images, labels)
            batch_idx: Index of the current batch
            
        Returns:
            Loss value for this batch
        """
        x, y = batch
        
        # Forward pass
        y_hat = self(x)
        
        # Compute loss
        loss = self.criterion(y_hat, y)
        
        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        # Store predictions for confusion matrix and report
        self.test_step_outputs.append({
            'preds': preds,
            'targets': y
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch to clean up"""
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch to aggregate results"""
        # Concatenate all predictions and targets
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs])
        
        # Store for later use (e.g., confusion matrix)
        self.test_predictions = all_preds.cpu().numpy()
        self.test_targets = all_targets.cpu().numpy()
        
        # Clear outputs
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            dict: Optimizer and scheduler configuration
        """
        # Adam optimizer (works well for both classical and quantum parameters)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        # Learning rate scheduler: reduce LR when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,        # Reduce LR by half
            patience=5        # Wait 5 epochs before reducing
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'  # Monitor validation loss
            }
        }
    
    def get_num_parameters(self):
        """
        Get the total number of parameters in the model.
        
        Returns:
            dict: Parameter counts for each component
        """
        cnn_params = sum(p.numel() for p in self.cnn.parameters())
        proj_params = sum(p.numel() for p in self.quantum_projection.parameters())
        quantum_params = self.quantum_weights.numel()
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'cnn': cnn_params,
            'projection': proj_params,
            'quantum': quantum_params,
            'classifier': classifier_params,
            'total': total_params
        }
    
    def print_model_info(self):
        """Print detailed information about the model"""
        print("\n" + "=" * 60)
        print("HYBRID QUANTUM-CLASSICAL MODEL")
        print("=" * 60)
        print(f"Encoding: {self.config.ENCODING}")
        print(f"Circuit: {self.config.CIRCUIT_TYPE}")
        print(f"Qubits: {self.config.N_QUBITS}")
        print(f"Layers: {self.config.N_LAYERS}")
        print(f"Classes: {self.config.N_CLASSES}")
        print("\nParameter counts:")
        params = self.get_num_parameters()
        for component, count in params.items():
            print(f"  {component:15s}: {count:,}")
        print("=" * 60 + "\n")
