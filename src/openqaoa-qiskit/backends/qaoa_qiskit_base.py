import numpy as np
from copy import deepcopy
from typing import Union, List, Tuple, Optional

# IBM Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter


from openqaoa.backends.cost_function import cost_function
from openqaoa.qaoa_components.ansatz_constructor import (
    XGateMap,
    RXGateMap,
    RYGateMap,
    RZGateMap,
    RXXGateMap,
    RYYGateMap,
    RZZGateMap,
    RZXGateMap,
)


class QAOAQiskitBaseBackend():
    """
    
    """

    QISKIT_GATEMAP_LIBRARY = [
        XGateMap,
        RXGateMap,
        RYGateMap,
        RZGateMap,
        RXXGateMap,
        RYYGateMap,
        RZZGateMap,
        RZXGateMap,
    ]

    
    def from_abstract_to_real(self, abstract_circuit) -> QuantumCircuit:
        """
        Creates a qiskit circuit, given an abstract one. 
        """
        qureg = QuantumRegister(self.n_qubits)
        parametric_circuit = QuantumCircuit(qureg)

        qiskit_parameter_list = []
        
        for each_gate in abstract_circuit:
            
            decomposition = each_gate.decomposition("standard")
                
            # Create Circuit
            for each_tuple in decomposition:
                low_gate = each_tuple[0]()
                parametric_circuit = low_gate.apply_ibm_gate(
                    *each_tuple[1], parametric_circuit
                )
        
        
        return parametric_circuit

