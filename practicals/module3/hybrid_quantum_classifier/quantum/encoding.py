"""
Quantum encoding strategies for classical data

This module contains different methods to encode classical data into quantum states.
Each encoding method has different properties and is suitable for different types of data.
"""

import torch
import pennylane as qml


def angle_encoding(features, wires):
    """
    Angle Encoding (also called Rotation Encoding).
    
    Each feature value is encoded as a rotation angle on a qubit.
    This is one of the simplest and most commonly used encodings.
    
    Mathematical representation:
        |ψ⟩ = ⊗ᵢ RY(θᵢ)|0⟩
        
    Where θᵢ are the feature values.
    
    Properties:
    - Simple and intuitive
    - Works well for continuous features
    - Each qubit encodes one feature
    - Good starting point for most problems
    
    Args:
        features: Tensor of feature values (length should match number of wires)
        wires: List of wire indices to encode on
        
    Example:
        >>> features = torch.tensor([0.1, 0.5, -0.3])
        >>> wires = [0, 1, 2]
        >>> angle_encoding(features, wires)
    """
    for i, wire in enumerate(wires):
        if i < len(features):
            # Apply RY rotation with feature as angle
            qml.RY(features[i], wires=wire)


def amplitude_encoding(features, wires):
    """
    Amplitude Encoding (also called Quantum State Preparation).
    
    Features become the amplitudes of a quantum state. This encoding
    can represent 2^n features using n qubits, making it very efficient.
    
    Mathematical representation:
        |ψ⟩ = Σᵢ αᵢ|i⟩
        
    Where αᵢ are the normalized feature values.
    
    Properties:
    - Most data-efficient: 2^n features in n qubits
    - Requires feature normalization (done automatically)
    - More complex to implement in hardware
    - Good for high-dimensional data
    
    IMPORTANT: Requires exactly 2^n features for n qubits!
    - 4 qubits → 16 features
    - 8 qubits → 256 features
    
    Args:
        features: Tensor of feature values (length must be 2^n)
        wires: List of wire indices to encode on
        
    Example:
        >>> features = torch.randn(16)  # For 4 qubits
        >>> wires = [0, 1, 2, 3]
        >>> amplitude_encoding(features, wires)
    """
    # Normalize features to unit vector
    norm = torch.sqrt(torch.sum(features ** 2))
    normalized_features = features / (norm + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Use PennyLane's built-in amplitude embedding
    # This automatically handles the complex state preparation
    qml.AmplitudeEmbedding(
        features=normalized_features,
        wires=wires,
        normalize=True
    )


def iqp_encoding(features, wires):
    """
    IQP Encoding (Instantaneous Quantum Polynomial).
    
    This encoding creates a quantum state that is hard to simulate classically,
    potentially providing quantum advantage. It uses diagonal unitaries and
    entangling operations.
    
    Mathematical representation:
        |ψ⟩ = U_ent · U_diag(θ²) · U_ent · U_diag(θ) · H⊗ⁿ|0⟩
        
    Where:
    - H⊗ⁿ: Hadamard on all qubits
    - U_diag: Diagonal rotations with features
    - U_ent: Entangling CZ gates
    
    Properties:
    - Hard to simulate classically (potential quantum advantage)
    - Good expressiveness for complex patterns
    - Uses both linear and quadratic features
    - More quantum gates required
    
    Args:
        features: Tensor of feature values (length should match number of wires)
        wires: List of wire indices to encode on
        
    Example:
        >>> features = torch.tensor([0.2, 0.5, -0.1, 0.8])
        >>> wires = [0, 1, 2, 3]
        >>> iqp_encoding(features, wires)
    """
    # Step 1: Apply Hadamard to all qubits (create superposition)
    for wire in wires:
        qml.Hadamard(wires=wire)
    
    # Step 2: First diagonal layer - linear encoding
    for i, wire in enumerate(wires):
        if i < len(features):
            qml.RZ(features[i], wires=wire)
    
    # Step 3: Entangling layer with CZ gates
    for i in range(len(wires) - 1):
        qml.CZ(wires=[wires[i], wires[i + 1]])
    
    # Step 4: Second diagonal layer - quadratic encoding
    # Using squared features creates higher-order interactions
    for i, wire in enumerate(wires):
        if i < len(features):
            qml.RZ(features[i] ** 2, wires=wire)


def basis_encoding(features, wires):
    """
    Basis Encoding (also called Digital or Computational Basis Encoding).
    
    Binary features directly encode computational basis states.
    Each qubit represents a binary digit (0 or 1).
    
    Mathematical representation:
        |ψ⟩ = |x₁x₂...xₙ⟩
        
    Where xᵢ ∈ {0, 1} are binary features.
    
    Properties:
    - Simplest encoding (just X gates)
    - Best for binary/categorical data
    - Very easy to implement on hardware
    - No trainable parameters in encoding
    
    Note: This binarizes continuous features (feature > 0 → 1, else → 0)
    
    Args:
        features: Tensor of feature values (will be binarized)
        wires: List of wire indices to encode on
        
    Example:
        >>> features = torch.tensor([0.5, -0.2, 0.8, -0.1])
        >>> wires = [0, 1, 2, 3]
        >>> basis_encoding(features, wires)
        # Result: |1010⟩ (qubit 0=1, qubit 1=0, qubit 2=1, qubit 3=0)
    """
    # Convert to binary (1 if feature > 0, else 0)
    binary_features = (features > 0).int()
    
    # Apply X gate where feature is 1 (flips |0⟩ to |1⟩)
    for i, wire in enumerate(wires):
        if i < len(features) and binary_features[i] == 1:
            qml.PauliX(wires=wire)


# Dictionary mapping encoding names to functions
# This makes it easy to select encodings by name in configuration
ENCODING_MAP = {
    'angle': angle_encoding,
    'amplitude': amplitude_encoding,
    'iqp': iqp_encoding,
    'basis': basis_encoding
}


def get_encoding_info(encoding_name):
    """
    Get information about a specific encoding method.
    
    Args:
        encoding_name: Name of the encoding ('angle', 'amplitude', 'iqp', 'basis')
        
    Returns:
        str: Description of the encoding method
    """
    descriptions = {
        'angle': "Angle Encoding: Each feature → rotation angle. Simple, works well for continuous data.",
        'amplitude': "Amplitude Encoding: Features → quantum amplitudes. Very efficient (2^n features in n qubits).",
        'iqp': "IQP Encoding: Creates hard-to-simulate states. Good for complex patterns.",
        'basis': "Basis Encoding: Binary features → basis states. Simple, good for categorical data."
    }
    return descriptions.get(encoding_name, "Unknown encoding")


def print_encoding_options():
    """Print all available encoding options with descriptions"""
    print("\n" + "=" * 60)
    print("AVAILABLE ENCODING STRATEGIES")
    print("=" * 60)
    for name in ENCODING_MAP.keys():
        print(f"\n{name.upper()}:")
        print(f"  {get_encoding_info(name)}")
    print("\n" + "=" * 60 + "\n")
