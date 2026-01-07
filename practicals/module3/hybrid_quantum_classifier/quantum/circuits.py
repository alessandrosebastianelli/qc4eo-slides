"""
Quantum circuit architectures (Ansätze)

This module contains different variational quantum circuit designs.
Each circuit type has different entanglement patterns and gate structures.
"""

import pennylane as qml


def strongly_entangling_circuit(weights, wires):
    """
    Strongly Entangling Layers (All-to-All Connectivity).
    
    This is one of the most expressive ansätze, where each qubit can
    become entangled with every other qubit. Uses PennyLane's built-in
    StronglyEntanglingLayers template.
    
    Structure per layer:
    1. Apply 3 rotations (RX, RY, RZ) on each qubit
    2. Apply CNOT gates in all-to-all pattern
    
    Properties:
    - Most expressive (can represent more quantum states)
    - Highest qubit connectivity
    - More quantum gates required
    - Good for small numbers of qubits (< 15)
    - May be hard to implement on real quantum hardware
    
    Args:
        weights: Tensor of shape (n_layers, n_qubits, 3)
                Each qubit gets 3 rotation angles per layer
        wires: List of wire indices
        
    Example:
        >>> import torch
        >>> weights = torch.randn(3, 4, 3)  # 3 layers, 4 qubits, 3 rotations
        >>> wires = [0, 1, 2, 3]
        >>> strongly_entangling_circuit(weights, wires)
    """
    qml.StronglyEntanglingLayers(weights, wires=wires)


def basic_entangling_circuit(weights, wires):
    """
    Basic Entangling Layers (Nearest-Neighbor Connectivity).
    
    A simpler circuit where qubits only interact with their neighbors.
    This is more hardware-friendly as most quantum computers have
    limited connectivity.
    
    Structure per layer:
    1. Apply one rotation on each qubit
    2. Apply CNOT gates between nearest neighbors
    
    Properties:
    - Hardware-friendly (linear connectivity)
    - Fewer quantum gates
    - Faster to simulate
    - Good for hardware implementation
    - Less expressive than strongly entangling
    
    Args:
        weights: Tensor of shape (n_layers,)
                One rotation angle per layer (shared across qubits)
        wires: List of wire indices
        
    Example:
        >>> import torch
        >>> weights = torch.randn(3)  # 3 layers
        >>> wires = [0, 1, 2, 3]
        >>> basic_entangling_circuit(weights, wires)
    """
    qml.BasicEntanglerLayers(weights, wires=wires)


def hardware_efficient_circuit(weights, wires):
    """
    Hardware-Efficient Ansatz.
    
    Designed to be easy to implement on real quantum hardware.
    Uses only single-qubit rotations and nearest-neighbor CNOTs
    in a structured pattern.
    
    Structure per layer:
    1. RY and RZ rotations on each qubit
    2. CNOT gates in a ring pattern (nearest neighbors + wrap-around)
    
    Properties:
    - Optimized for real quantum hardware
    - Two-qubit gates only between neighbors
    - Balanced expressiveness and implementability
    - Ring connectivity (includes wrap-around)
    - Good practical performance
    
    Args:
        weights: Tensor of shape (n_layers, n_qubits, 2)
                Each qubit gets 2 rotation angles (RY, RZ) per layer
        wires: List of wire indices
        
    Example:
        >>> import torch
        >>> weights = torch.randn(3, 4, 2)  # 3 layers, 4 qubits, 2 rotations
        >>> wires = [0, 1, 2, 3]
        >>> hardware_efficient_circuit(weights, wires)
    """
    n_layers = weights.shape[0]
    n_wires = len(wires)
    
    for layer in range(n_layers):
        # Rotation layer: Apply RY and RZ to each qubit
        for i, wire in enumerate(wires):
            qml.RY(weights[layer, i, 0], wires=wire)
            qml.RZ(weights[layer, i, 1], wires=wire)
        
        # Entangling layer: CNOTs in a ring pattern
        for i in range(n_wires - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
        
        # Wrap-around CNOT (creates ring connectivity)
        if n_wires > 2:
            qml.CNOT(wires=[wires[-1], wires[0]])


def custom_alternating_circuit(weights, wires):
    """
    Custom Alternating Circuit.
    
    A custom design that alternates between different entangling
    patterns in different layers. This can help the circuit learn
    more diverse quantum states.
    
    Structure per layer:
    1. Apply 3 rotations (RX, RY, RZ) on each qubit
    2. Even layers: CNOT gates on even pairs (0-1, 2-3, ...)
    3. Odd layers: CZ gates on odd pairs (1-2, 3-4, ...) + wrap-around
    
    Properties:
    - Alternating entanglement patterns
    - Uses both CNOT and CZ gates
    - Good expressiveness with reasonable depth
    - Custom design (not a standard template)
    - Interesting for research/experimentation
    
    Args:
        weights: Tensor of shape (n_layers, n_qubits, 3)
                Each qubit gets 3 rotation angles per layer
        wires: List of wire indices
        
    Example:
        >>> import torch
        >>> weights = torch.randn(3, 4, 3)  # 3 layers, 4 qubits, 3 rotations
        >>> wires = [0, 1, 2, 3]
        >>> custom_alternating_circuit(weights, wires)
    """
    n_layers = weights.shape[0]
    n_wires = len(wires)
    
    for layer in range(n_layers):
        # Rotation layer: Apply RX, RY, RZ to each qubit
        for i, wire in enumerate(wires):
            qml.RX(weights[layer, i, 0], wires=wire)
            qml.RY(weights[layer, i, 1], wires=wire)
            # Use third weight or fall back to first weight if shape[2] < 3
            rz_angle = weights[layer, i, 2] if weights.shape[2] > 2 else weights[layer, i, 0]
            qml.RZ(rz_angle, wires=wire)
        
        # Alternating entangling patterns
        if layer % 2 == 0:
            # Even layers: CNOT on even pairs (0-1, 2-3, 4-5, ...)
            for i in range(0, n_wires - 1, 2):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
        else:
            # Odd layers: CZ on odd pairs (1-2, 3-4, 5-6, ...)
            for i in range(1, n_wires - 1, 2):
                qml.CZ(wires=[wires[i], wires[i + 1]])
            
            # Wrap-around CZ for connectivity
            if n_wires > 2:
                qml.CZ(wires=[wires[0], wires[-1]])


# Dictionary mapping circuit names to functions
CIRCUIT_MAP = {
    'strongly_entangling': strongly_entangling_circuit,
    'basic_entangling': basic_entangling_circuit,
    'hardware_efficient': hardware_efficient_circuit,
    'custom_alternating': custom_alternating_circuit
}


def get_weight_shape(circuit_type, n_qubits, n_layers):
    """
    Calculate the required weight shape for a given circuit type.
    
    Different circuit types require different numbers of parameters.
    This function calculates the exact shape needed.
    
    Args:
        circuit_type: Name of the circuit ('strongly_entangling', etc.)
        n_qubits: Number of qubits in the circuit
        n_layers: Number of variational layers
        
    Returns:
        tuple: Shape of the weight tensor
        
    Example:
        >>> get_weight_shape('strongly_entangling', 4, 3)
        (3, 4, 3)  # 3 layers, 4 qubits, 3 rotations per qubit
        
        >>> get_weight_shape('basic_entangling', 4, 3)
        (3,)  # 3 layers, 1 rotation per layer
    """
    weight_shapes = {
        'strongly_entangling': (n_layers, n_qubits, 3),
        'basic_entangling': (n_layers,),
        'hardware_efficient': (n_layers, n_qubits, 2),
        'custom_alternating': (n_layers, n_qubits, 3)
    }
    
    if circuit_type not in weight_shapes:
        raise ValueError(
            f"Unknown circuit type: {circuit_type}\n"
            f"Available types: {list(weight_shapes.keys())}"
        )
    
    return weight_shapes[circuit_type]


def get_circuit_info(circuit_name):
    """
    Get information about a specific circuit type.
    
    Args:
        circuit_name: Name of the circuit
        
    Returns:
        str: Description of the circuit
    """
    descriptions = {
        'strongly_entangling': "Strongly Entangling: All-to-all connectivity. Most expressive, good for small systems.",
        'basic_entangling': "Basic Entangling: Nearest-neighbor connectivity. Hardware-friendly, faster.",
        'hardware_efficient': "Hardware-Efficient: Optimized for real quantum hardware. Balanced performance.",
        'custom_alternating': "Custom Alternating: Alternating gate patterns. Good for experimentation."
    }
    return descriptions.get(circuit_name, "Unknown circuit")


def print_circuit_options():
    """Print all available circuit options with descriptions"""
    print("\n" + "=" * 60)
    print("AVAILABLE CIRCUIT ARCHITECTURES")
    print("=" * 60)
    for name in CIRCUIT_MAP.keys():
        print(f"\n{name.upper()}:")
        print(f"  {get_circuit_info(name)}")
        # Show weight shape for 8 qubits, 3 layers as example
        shape = get_weight_shape(name, 8, 3)
        print(f"  Weight shape (8 qubits, 3 layers): {shape}")
    print("\n" + "=" * 60 + "\n")
