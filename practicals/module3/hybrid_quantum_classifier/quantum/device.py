"""
Quantum device initialization and management
"""

import pennylane as qml


def create_quantum_device(n_qubits, device_name='default.qubit'):
    """
    Create a PennyLane quantum device for simulation.
    
    PennyLane supports different simulator backends:
    - 'default.qubit': Standard simulator (slower but always available)
    - 'lightning.qubit': Fast C++ simulator (2x faster, requires installation)
    
    Args:
        n_qubits: Number of qubits in the quantum device
        device_name: Name of the device ('default.qubit' or 'lightning.qubit')
        
    Returns:
        qml.Device: A PennyLane quantum device
        
    Raises:
        Exception: If lightning.qubit is requested but not available
    """
    
    print("\n" + "=" * 60)
    print("QUANTUM DEVICE SETUP")
    print("=" * 60)
    
    if device_name == 'lightning.qubit':
        try:
            # Try to create lightning.qubit device (faster)
            dev = qml.device('lightning.qubit', wires=n_qubits)
            print(f"✓ Quantum device created: {n_qubits} qubits")
            print(f"✓ Backend: lightning.qubit (FAST - 2x speedup!)")
            print(f"  Using PennyLane Lightning for accelerated simulation")
            
        except Exception as e:
            # Fall back to default.qubit if lightning not available
            print(f"⚠ Warning: lightning.qubit not available")
            print(f"  Error: {e}")
            print(f"  Install with: pip install pennylane-lightning")
            print(f"  Falling back to default.qubit")
            dev = qml.device('default.qubit', wires=n_qubits)
            print(f"✓ Quantum device created: {n_qubits} qubits")
            print(f"✓ Backend: default.qubit")
            
    else:
        # Create standard default.qubit device
        dev = qml.device('default.qubit', wires=n_qubits)
        print(f"✓ Quantum device created: {n_qubits} qubits")
        print(f"✓ Backend: default.qubit")
    
    # Print device information
    print(f"✓ Number of wires: {n_qubits}")
    print(f"✓ Hilbert space dimension: {2**n_qubits}")
    print("=" * 60 + "\n")
    
    return dev


def get_device_info(device):
    """
    Get information about a quantum device.
    
    Args:
        device: A PennyLane quantum device
        
    Returns:
        dict: Dictionary containing device information
    """
    return {
        'name': device.name,
        'short_name': device.short_name,
        'num_wires': len(device.wires),
        'wires': list(device.wires),
        'shots': device.shots,
    }


def print_device_info(device):
    """
    Print detailed information about a quantum device.
    
    Args:
        device: A PennyLane quantum device
    """
    info = get_device_info(device)
    
    print("\nQuantum Device Information:")
    print("-" * 40)
    for key, value in info.items():
        print(f"  {key:15s}: {value}")
    print("-" * 40 + "\n")
