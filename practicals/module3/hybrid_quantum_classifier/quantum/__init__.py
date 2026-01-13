"""
Quantum module containing all quantum computing components
"""

from .device import create_quantum_device
from .encoding import (
    angle_encoding,
    amplitude_encoding,
    iqp_encoding,
    basis_encoding,
    ENCODING_MAP,
    print_encoding_options
)
from .circuits import (
    strongly_entangling_circuit,
    basic_entangling_circuit,
    hardware_efficient_circuit,
    custom_alternating_circuit,
    CIRCUIT_MAP,
    get_weight_shape,
    print_circuit_options
)

__all__ = [
    'create_quantum_device',
    'angle_encoding',
    'amplitude_encoding',
    'iqp_encoding',
    'basis_encoding',
    'ENCODING_MAP',
    'strongly_entangling_circuit',
    'basic_entangling_circuit',
    'hardware_efficient_circuit',
    'custom_alternating_circuit',
    'CIRCUIT_MAP',
    'get_weight_shape',
    'print_encoding_options',
    'print_circuit_options'
]
