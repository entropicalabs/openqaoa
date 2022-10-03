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

from collections import Counter
import numpy as np
from sympy import Symbol
from copy import deepcopy
from pytket.circuit import Circuit, OpType

from ...basebackend import QAOABaseBackendShotBased, QAOABaseBackendCloud, QAOABaseBackendParametric
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from ...devices import DevicePyTket, DeviceBase
from ...qaoa_parameters.gatemap import RZZGateMap, SWAPGateMap
    

class QAOAPyTketBackend(QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased):
    """
    A QAOA backend object for real Rigetti QPUs

    Parameters
    ----------
    device: `DevicePyquil`
        The device object to access pyquil devices with credentials.
    circuit_params: `QAOACircuitParams`
        An object of the class ``QAOACircuitParams`` which contains information on 
        circuit construction and depth of the circuit.
    n_shots: `int`
        The number of shots to be taken for each circuit.
    prepend_state: `pyquil.Program`
        The state prepended to the circuit.
    append_state: `pyquil.Program`
        The state appended to the circuit.
    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the QAOA part of the circuit.
    cvar_alpha: `float`
        Conditional Value-at-Risk (CVaR) – a measure that takes into account only the tail of the
        probability distribution arising from the circut's count dictionary. Must be between 0 and 1. Check
        https://arxiv.org/abs/1907.04769 for further details.
    circuit_optimization_lvl: `int`
        Specify an integer in the range [0,2] to select the strength of the circuit optimization performed
        on the logical circuit to map it to physical chip topology.
    """

    def __init__(self,
                 circuit_params: QAOACircuitParams,
                 device: DeviceBase,
                 n_shots: int,
                 prepend_state: Circuit,
                 append_state: Circuit,
                 init_hadamard: bool,
                 cvar_alpha: float, 
                 circuit_optimization_lvl: int = 1
                 ):

        QAOABaseBackendShotBased.__init__(self,
                                          circuit_params,
                                          n_shots,
                                          prepend_state,
                                          append_state,
                                          init_hadamard,
                                          cvar_alpha)
        
        #wrap the user-selected device as PyTket Device
        device = DevicePyTket(device)
        QAOABaseBackendCloud.__init__(self, device)
        
        self.tket_backend = self.device.backend_device
        self.qureg = self.circuit_params.qureg
        self.circuit_optimization_lvl = circuit_optimization_lvl

        # self.qureg_placeholders = QubitPlaceholder.register(self.n_qubits)
        # self.qubit_mapping = dict(zip(self.qureg, self.qubit_layout))
        
        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.get_qubits()), "Cannot attach a bigger circuit " \
                                                                "to the QAOA routine"

        self.parametric_circuit = self.parametric_qaoa_circuit
        self.optimized_circuit = self.optimized_parametric_circuit

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> Circuit:
        """
        Injects angles into created executable parametric circuit.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        `pyquil.Program`
            A pyquil.Program (executable) object.
        """
        #NOTE: While appending angles to the circuit parameters, divide by pi for
        #conversion to units of pi for PyTket

        angles_list = np.array(self.obtain_angles_for_pauli_list(
            self.abstract_circuit, params), dtype=float)
        angles_symbols = self.optimized_circuit.free_symbols()

        angles_map = {symbol:angle/np.pi for symbol,angle in zip(angles_symbols,angles_list)}

        angle_assigned_circuit = deepcopy(self.optimized_circuit)
        angle_assigned_circuit.symbol_substitution(angles_map)

        assert not angle_assigned_circuit.is_symbolic(), ValueError("Angle assignment is incomplete")

        return angle_assigned_circuit

    @property
    def optimized_parametric_circuit(self) -> Circuit:
        """
        Perform circuit optimization on the parametric circuit created using 
        circuit instructions derived from the problem statement

        Returns
        -------
        `pytket.circuit.Circuit`
            The optimized Pyktet circuit object
        """
        
        optimized_ckt = self.tket_backend.get_compiled_circuit(
                                                self.parametric_circuit, 
                                                optimisation_level = self.circuit_optimization_lvl
                                            )
        return optimized_ckt

    @property
    def parametric_qaoa_circuit(self) -> Circuit:
        """
        Creates a parametric QAOA circuit (pytket.circuit.Circuit object), given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit. 
        
        Returns
        -------
        `pytket.circuit.Circuit`
            A Pyktet circuit object
        """
        parametric_circuit = Circuit()
        self.qubit_mapping = parametric_circuit.add_q_register("q", self.n_qubits) 
        creg = parametric_circuit.add_c_register("z", self.n_qubits)

        if self.prepend_state:
            parametric_circuit.append(self.prepend_state)
            
        # Initial state is all |+>
        if self.init_hadamard:
            for i in self.qureg:
                parametric_circuit.add_gate(OpType.Rz, 1, [self.qubit_mapping[i]])
                parametric_circuit.add_gate(OpType.Rx, 1/2, [self.qubit_mapping[i]])
                parametric_circuit.add_gate(OpType.Rz, 1/2, [self.qubit_mapping[i]])
                parametric_circuit.add_gate(OpType.Rx, -1/2, [self.qubit_mapping[i]])

        # create a list of gates in order of application on quantum circuit
        for each_gate in self.abstract_circuit:
            gate_label = ''.join(str(label) for label in each_gate.pauli_label)
            angle_param = Symbol(gate_label)
            #     f'pauli{gate_label}', 'REAL', 1)
            each_gate.rotation_angle = angle_param
            if isinstance(each_gate, RZZGateMap) or isinstance(each_gate, SWAPGateMap):
                decomposition = each_gate.decomposition('standard')
            else:
                decomposition = each_gate.decomposition('standard')
                
            # using the list above, construct the circuit
            for each_tuple in decomposition:
                gate = each_tuple[0]()
                gate.apply_pytket_gate(*each_tuple[1],parametric_circuit)

        if self.append_state:
            parametric_circuit.append(self.append_state)
            
        # Measurement instructions
        for i, qubit in enumerate(self.qureg):
            parametric_circuit.Measure(self.qubit_mapping[qubit], creg[i])

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams) -> dict:
        """
        Execute the circuit and obtain the counts.

        Parameters
        ----------
        params: QAOAVariationalBaseParams
            The QAOA parameters - an object of one of the parameter classes, containing 
            variable parameters.

        Returns
        -------
        counts : dictionary
            A dictionary with the bitstring as the key and the number of counts as its value.
        """

        #before execution on the selected hardware, make sure all the Predicates are satisfied
        circuit_to_execute = self.qaoa_circuit(params)
        assert self.tket_backend.valid_circuit(circuit_to_execute), "The compiled Circuit cannot run on selected hardware"

        meas_handle = self.tket_backend.process_circuit(circuit_to_execute, n_shots = self.n_shots)
        meas_outcomes = self.tket_backend.get_result(meas_handle).get_counts()

        # TODO: check the endian (big or little) ordering of measurement outcomes
        # Expose counts
        counts = {''.join(str(i) for i in basis):prob for basis,prob in meas_outcomes.items()}
        self.measurement_outcomes = counts
        return counts

    def circuit_to_qasm(self, params: QAOAVariationalBaseParams) -> str:
        """
        A method to convert the pyQuil program to a OpenQASM string.
        """
        raise NotImplementedError()
    
    def reset_circuit(self):
        """
        Reset self.program after performing a computation. Also handle active reset and rewirings.
        """
        pass
