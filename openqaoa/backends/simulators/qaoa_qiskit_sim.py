#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# General Imports
from ...basebackend import QAOABaseBackendParametric, QAOABaseBackendShotBased, QAOABaseBackendStatevector
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from ...utilities import flip_counts
from ...cost_function import cost_function
from ...qaoa_parameters.gatemap import (RXGateMap, RYGateMap, RZGateMap, RXXGateMap,
                                          RYYGateMap, RZZGateMap, RZXGateMap)

import numpy as np
from typing import Union, List, Tuple, Optional

# IBM Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter

"""
QASM Simluator can be used for different simulation purposes with different Simulation methods
    - supports different error models
    - supports incluing real IBM backend error models
"""

class QAOAQiskitBackendShotBasedSimulator(QAOABaseBackendShotBased, QAOABaseBackendParametric):
    """
    Local Shot-based simulators offered by Qiskit

    Parameters
    ----------
    circuit_params: `QAOACircuitParams`
        An object of the class ``QAOACircuitParams`` which contains information on 
        circuit construction and depth of the circuit.
    
    n_shots: `int`
        The number of shots to be taken for each circuit.
    
    seed_simulator: `int`
        Pseudorandom list of numbers of a seed

    prepend_state: `QuantumCircuit`
        The state prepended to the circuit.

    append_state: `QuantumCircuit`
        The state appended to the circuit.

    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the 
        QAOA part of the circuit.

    cvar_alpha: `float`
        The value of alpha for the CVaR cost function.
    
    qiskit_simulation_method: `str`
        The method to be used for the simulation.

    noise_model: `NoiseModel`
        The Qiskit noise model to be used for the simulation.
    """    
    QISKIT_GATEMAP_LIBRARY = [RXGateMap, RYGateMap, RZGateMap, RXXGateMap,
                                RYYGateMap, RZZGateMap, RZXGateMap]


    def __init__(self,
                 circuit_params: QAOACircuitParams,
                 n_shots: int,
                 prepend_state: Optional[QuantumCircuit],
                 append_state: Optional[QuantumCircuit],
                 init_hadamard: bool,
                 cvar_alpha: float,
                 qiskit_simulation_method: str = 'automatic',
                 seed_simulator: Optional[int] = None,
                 noise_model: Optional[NoiseModel] = None):
        
        QAOABaseBackendShotBased.__init__(self,circuit_params,
                                          n_shots,
                                          prepend_state,
                                          append_state,
                                          init_hadamard,
                                          cvar_alpha)

        self.qureg = QuantumRegister(self.n_qubits)
        self.qubit_layout = self.circuit_params.qureg
        
        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.qubits), "Cannot attach a bigger circuit " \
                                                                "to the QAOA routine"
        # options = {"seed_simulator":1}
        self.backend_simulator = AerSimulator(method=qiskit_simulation_method.lower(),
                                              noise_model = noise_model, seed_simulator=seed_simulator)
        # For parametric circuits
        self.parametric_circuit = self.parametric_qaoa_circuit

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> QuantumCircuit:
        """
        The final QAOA circuit to be executed on the QPU.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
        
        Returns
        -------
        qaoa_circuit: `QuantumCircuit`
            The final QAOA circuit after binding angles from variational parameters.
        """
        angles_list = self.obtain_angles_for_pauli_list(self.abstract_circuit, params)
        memory_map = dict(zip(self.qiskit_parameter_list, angles_list))
        new_parametric_circuit = self.parametric_circuit.bind_parameters(memory_map)
        return new_parametric_circuit
    
    @property
    def parametric_qaoa_circuit(self) -> QuantumCircuit:
        """
        Creates a parametric QAOA circuit, given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit.
        """
        # self.reset_circuit()
        parametric_circuit = QuantumCircuit(self.qureg)

        if self.prepend_state:
            parametric_circuit = parametric_circuit.compose(self.prepend_state)
        # Initial state is all |+>
        if self.init_hadamard:
            parametric_circuit.h(self.qureg)
        
        self.qiskit_parameter_list=[]
        for each_gate in self.abstract_circuit:
            angle_param = Parameter(str(each_gate.pauli_label))
            self.qiskit_parameter_list.append(angle_param)
            each_gate.rotation_angle = angle_param
            if type(each_gate) in self.QISKIT_GATEMAP_LIBRARY:
                decomposition = each_gate.decomposition('trivial')
            else: 
                decomposition = each_gate.decomposition('standard')
            # Create Circuit
            for each_tuple in decomposition:
                low_gate = each_tuple[0]()
                parametric_circuit = low_gate.apply_ibm_gate(*each_tuple[1],parametric_circuit)
        
        if self.append_state:
            parametric_circuit = parametric_circuit.compose(self.append_state)
        parametric_circuit.measure_all()

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams) -> dict:
        """
        Returns the counts of the final QAOA circuit after binding angles from variational parameters.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
        
        Returns
        -------
        counts: `dict`
            The counts of the final QAOA circuit after binding angles from variational parameters.
        """
        qaoa_circuit = self.qaoa_circuit(params)
        counts = self.backend_simulator.run(qaoa_circuit, shots=self.n_shots).result().get_counts()
        flipped_counts = flip_counts(counts)
        self.measurement_outcomes = flipped_counts
        return flipped_counts

    def circuit_to_qasm(self):
        """
        A method to convert the QAOA circuit to QASM.
        """
        raise NotImplementedError()
#         qasm_circuit = self.parametric_circuit.qasm()
#         return qasm_circuit

    def reset_circuit(self):
        raise NotImplementedError()
    

class QAOAQiskitBackendStatevecSimulator(QAOABaseBackendStatevector, QAOABaseBackendParametric):
    """
    Local Shot-based simulators offered by Qiskit

    Parameters
    ----------
    circuit_params: `QAOACircuitParams`
        An object of the class ``QAOACircuitParams`` which contains information on 
        circuit construction and depth of the circuit.
    
    n_shots: `int`
        The number of shots to be taken for each circuit.

    prepend_state: `np.ndarray` or `QuantumCircuit`
        The state prepended to the circuit.

    append_state: `QuantumCircuit or np.ndarray`
        The state appended to the circuit.

    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the 
        QAOA part of the circuit.

    cvar_alpha: `float`
        The value of alpha for the CVaR cost function.
    """
    QISKIT_GATEMAP_LIBRARY = [RXGateMap, RYGateMap, RZGateMap, RXXGateMap,
                                RYYGateMap, RZZGateMap, RZXGateMap]

    def __init__(self,
                 circuit_params: QAOACircuitParams,
                 prepend_state: Optional[Union[np.ndarray,QuantumCircuit]],
                 append_state: Optional[Union[np.ndarray,QuantumCircuit]],
                 init_hadamard: bool,
                 cvar_alpha: float = 1):
        
        QAOABaseBackendStatevector.__init__(self, circuit_params,
                                            prepend_state,
                                            append_state,
                                            init_hadamard,
                                            cvar_alpha)

        assert cvar_alpha == 1,  "Please use the shot-based simulator for simulations with cvar_alpha < 1"

        self.qureg = QuantumRegister(self.n_qubits)
        self.qubit_layout = self.circuit_params.qureg
        
        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.qubits), "Cannot attach a bigger circuit " \
                                                                "to the QAOA routine"

        # For parametric circuits
        self.parametric_circuit = self.parametric_qaoa_circuit
        self.qiskit_cost_hamil = self.qiskit_cost_hamiltonian
        self.qiskit_cost_hamil_sq = self.qiskit_cost_hamil**2

    @property
    def qiskit_cost_hamiltonian(self):
        """
        The qiskit cost hamiltonian for the QAOA circuit represented
        as a `PauliSumOp` object.
        """
        cost_hamil = self.cost_hamiltonian
        n_qubits = cost_hamil.n_qubits
        pauli_strings_list = ['I'*n_qubits]*len(cost_hamil.terms)
        for i,pauli_op in enumerate(cost_hamil.terms):
            pauli_term = list(pauli_strings_list[i])
            for pauli,qubit in zip(pauli_op.pauli_str,pauli_op.qubit_indices):
                pauli_term[qubit] = pauli
            #reverse pauli_strings because qiskit supports little endian notation
            pauli_strings_list[i] = ''.join(str(term) for term in pauli_term)[::-1]
                
        pauli_strings_list.append('I'*n_qubits)
        pauli_coeffs = cost_hamil.coeffs
            
        qiskit_pauli_op = [[pauli_strings,coeff] for pauli_strings,coeff in zip(pauli_strings_list,pauli_coeffs)]
        qiskit_pauli_op.append(['I'*n_qubits,cost_hamil.constant])
        qiskit_cost_hamil = PauliSumOp.from_list(qiskit_pauli_op)
        return qiskit_cost_hamil

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> QuantumCircuit:
        """
        The final QAOA circuit to be executed on the QPU.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
        
        Returns
        -------
        qaoa_circuit: `QuantumCircuit`
            The final QAOA circuit after binding angles from variational parameters.
        """
        angles_list = self.obtain_angles_for_pauli_list(self.abstract_circuit, params)
        memory_map = dict(zip(self.qiskit_parameter_list, angles_list))
        new_parametric_circuit = self.parametric_circuit.bind_parameters(memory_map)
        return new_parametric_circuit
    
    @property
    def parametric_qaoa_circuit(self) -> QuantumCircuit:
        """
        Creates a parametric QAOA circuit, given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit. To do this, you will need to subsequently execute the command self.eng.flush().

        Parameters
        ----------
            params:
                Object of type QAOAVariationalBaseParams
        """
        # self.reset_circuit()
        parametric_circuit = QuantumCircuit(self.qureg)

        if self.prepend_state:
            parametric_circuit = parametric_circuit.compose(self.prepend_state)
        # Initial state is all |+>
        if self.init_hadamard:
            parametric_circuit.h(self.qureg)
        
        self.qiskit_parameter_list=[]
        for each_gate in self.abstract_circuit:
            angle_param = Parameter(str(each_gate.pauli_label))
            self.qiskit_parameter_list.append(angle_param)
            each_gate.rotation_angle = angle_param
            if type(each_gate) in self.QISKIT_GATEMAP_LIBRARY:
                decomposition = each_gate.decomposition('trivial')
            else: 
                decomposition = each_gate.decomposition('standard')
            # Create Circuit
            for each_tuple in decomposition:
                low_gate = each_tuple[0]()
                parametric_circuit = low_gate.apply_ibm_gate(*each_tuple[1],parametric_circuit)

        if self.append_state:
            parametric_circuit = parametric_circuit.compose(self.append_state)

        return parametric_circuit

    def wavefunction(self,
                     params: QAOAVariationalBaseParams) -> Union[List[complex],np.ndarray]:
        """
        Get the wavefunction of the state produced by the parametric circuit.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        wf: `List[complex]` or `np.ndarray[complex]`
            A list of the wavefunction amplitudes.
         """
        ckt = self.qaoa_circuit(params)
        wf = Statevector(ckt).data
        self.measurement_outcomes = wf
        return wf

    def expectation(self,
                    params: QAOAVariationalBaseParams) -> float:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing 
            variable parameters.

        Returns
        -------
        `float`
            expectation value of cost operator wrt to quantum state produced by QAOA circuit
        """
        ckt = self.qaoa_circuit(params)
        output_wf = Statevector(ckt)
        self.measurement_outcomes = output_wf.data
        cost = np.real(output_wf.expectation_value(self.qiskit_cost_hamil))
        return cost

    def expectation_w_uncertainty(self, 
                                  params: QAOAVariationalBaseParams) -> Tuple[float, float]:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian and its uncertainty

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing 
            variable parameters.

        Returns
        -------
        `Tuple[float]`
            expectation value and its uncertainty of cost operator wrt 
            to quantum state produced by QAOA circuit.
        """
        ckt = self.qaoa_circuit(params)
        output_wf = Statevector(ckt)
        self.measurement_outcomes = output_wf.data
        cost = np.real(Statevector(ckt).expectation_value(self.qiskit_cost_hamil))
        cost_sq = np.real(Statevector(ckt).expectation_value(self.qiskit_cost_hamil_sq))
        
        uncertainty = np.sqrt(cost_sq - cost**2)
        return (cost, uncertainty)

    def reset_circuit(self):
        """
        Reset self.circuit after performing a computation

        TODO: Check if only the first instruction is needed. Two might
              be redundant
        """
        raise NotImplementedError()

    def circuit_to_qasm(self):
        """
        """
        raise NotImplementedError()
