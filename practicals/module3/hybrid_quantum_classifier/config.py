"""
Configuration file for Hybrid Quantum Classifier
Contains all hyperparameters and settings for the project
"""

class Config:
    """
    Central configuration class for the Hybrid Quantum Classifier.
    Modify these settings to experiment with different configurations.
    """
    
    # ==================== QUANTUM CONFIGURATION ====================
    
    # Quantum device: 'default.qubit' or 'lightning.qubit' (2x faster!)
    DEVICE = 'default.qubit'
    
    # Number of qubits in the quantum circuit
    N_QUBITS = 8
    
    # Number of variational layers in the quantum circuit
    N_LAYERS = 3
    
    # Encoding strategy: 'angle', 'amplitude', 'iqp', 'basis'
    # - angle: Each feature becomes a rotation angle (simple, works well)
    # - amplitude: Features encode quantum state amplitudes (requires 2^n features)
    # - iqp: Instantaneous Quantum Polynomial encoding (good expressiveness)
    # - basis: Binary features encode basis states (simple, discrete)
    ENCODING = 'angle'
    
    # Circuit type: 'strongly_entangling', 'basic_entangling', 
    #               'hardware_efficient', 'custom_alternating'
    # - strongly_entangling: All-to-all qubit connectivity (most expressive)
    # - basic_entangling: Nearest-neighbor connectivity (hardware-friendly)
    # - hardware_efficient: Optimized for real quantum hardware
    # - custom_alternating: Custom design with alternating gates
    CIRCUIT_TYPE = 'strongly_entangling'
    
    # ==================== TRAINING CONFIGURATION ====================
    
    # Batch size for training
    BATCH_SIZE = 32
    
    # Learning rate for optimizer
    LEARNING_RATE = 0.001
    
    # Maximum number of training epochs
    MAX_EPOCHS = 50
    
    # Early stopping patience (stop if no improvement for N epochs)
    PATIENCE = 10
    
    # ==================== DATA CONFIGURATION ====================
    
    # Maximum images per class to load (for faster experimentation)
    MAX_IMAGES_PER_CLASS = 300
    
    # Image size after resizing
    IMAGE_SIZE = (64, 64)
    
    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Number of worker processes for data loading
    NUM_WORKERS = 2
    
    # ==================== MODEL CONFIGURATION ====================
    
    # CNN feature extractor output dimension
    CNN_OUTPUT_DIM = 256
    
    # Dropout rate
    DROPOUT_RATE = 0.3
    
    # Number of classes (will be set automatically from dataset)
    N_CLASSES = 10
    
    # ==================== COMPUTED PROPERTIES ====================
    
    @property
    def quantum_input_dim(self):
        """
        Calculate quantum input dimension based on encoding type.
        Amplitude encoding requires 2^n_qubits features, others need n_qubits.
        """
        if self.ENCODING == 'amplitude':
            return 2 ** self.N_QUBITS
        else:
            return self.N_QUBITS
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as a dictionary"""
        config = cls()
        return {
            'device': cls.DEVICE,
            'n_qubits': cls.N_QUBITS,
            'n_layers': cls.N_LAYERS,
            'encoding': cls.ENCODING,
            'circuit_type': cls.CIRCUIT_TYPE,
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'max_epochs': cls.MAX_EPOCHS,
            'patience': cls.PATIENCE,
            'quantum_input_dim': config.quantum_input_dim,
            'n_classes': cls.N_CLASSES,
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        config = cls.get_config_dict()
        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        for key, value in config.items():
            print(f"{key:25s}: {value}")
        print("=" * 60 + "\n")


# Create a global config instance
config = Config()
