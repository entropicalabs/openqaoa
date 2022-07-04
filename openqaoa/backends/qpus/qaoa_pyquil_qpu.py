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
from pyquil import Program, gates, quilbase

from ...basebackend import QAOABaseBackendShotBased, QAOABaseBackendCloud, QAOABaseBackendParametric
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from ...devices import DevicePyquil
from ...qaoa_parameters.pauligate import RZZPauliGate, SWAPGate


def check_edge_connectivity(executable: Program, device: DevicePyquil):

    '''
    Check that the program does not contain 2-qubit terms that is not present in the QPU's topology (to prevent quilc from crashing).
    
    Parameters
    ----------
    executable: `Program`
        pyQuil executable program.
    device: `DevicePyquil`
        An object of the class ``DevicePyquil`` which contains information on pyQuil's `QuantumComputer` object, used to extract the selected QPU's topology.
    
    Returns
    -------
        None
    """
    '''

    qpu_graph = device.quantum_computer.qubit_topology()
    
    instrs = [instr for instr in executable if type(instr) == quilbase.Gate]
    pair_instrs = [list(instr.get_qubits()) for instr in instrs if len(instr.get_qubits()) == 2]

    for term in pair_instrs:
        if len(term) == 2:

            assert term in qpu_graph.edges(), f"Term {term} is not an edge on the QPU graph of {device.device_name}."
    
    

class QAOAPyQuilQPUBackend(QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased):
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
    active_reset: 
        Whether to use the pyQuil's active reset scheme to reset qubits between shots. 
    rewiring:
        Rewiring scheme to be used for Pyquil. 
        Either PRAGMA INITIAL_REWIRING "NAIVE" or PRAGMA INITIAL_REWIRING "PARTIAL". 
        If None, pyquil defaults according to:
        NAIVE: The qubits used in all instructions in the program satisfy the topological constraints of the device.
        PARTIAL: Otherwise.
    """

    def __init__(self,
                 device: DevicePyquil,
                 circuit_params: QAOACircuitParams,
                 n_shots: int,
                 prepend_state: Program,
                 append_state: Program,
                 init_hadamard: bool,
                 cvar_alpha: float,
                 active_reset: bool = False,
                 rewiring: str = '',
                 qubit_layout: list = []
                 ):

        QAOABaseBackendShotBased.__init__(self,
                                          circuit_params,
                                          n_shots,
                                          prepend_state,
                                          append_state,
                                          init_hadamard,
                                          cvar_alpha)
        QAOABaseBackendCloud.__init__(self, device)

        self.active_reset = active_reset
        self.rewiring = rewiring
        self.qureg = self.circuit_params.qureg

        # self.qureg_placeholders = QubitPlaceholder.register(self.n_qubits)
        self.qubit_layout = self.qureg if qubit_layout == [] else qubit_layout
        self.qubit_mapping = dict(zip(self.qureg, self.qubit_layout))
        
        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.get_qubits()), "Cannot attach a bigger circuit " \
                                                                "to the QAOA routine"

        self.parametric_circuit = self.parametric_qaoa_circuit
        native_prog = self.device.quantum_computer.compiler.quil_to_native_quil(
            self.parametric_circuit)
        self.prog_exe = self.device.quantum_computer.compiler.native_quil_to_executable(
            native_prog)
        
        # Check program connectivity against QPU connectivity
        # TODO: reconcile with PRAGMA PRESERVE

        # check_edge_connectivity(self.prog_exe, device)


    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> Program:
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
        angles_list = np.array(self.obtain_angles_for_pauli_list(
            self.pseudo_circuit, params), dtype=float)
        angle_declarations = list(self.prog_exe.declarations.keys())
        angle_declarations.remove('ro')
        for i, param_name in enumerate(angle_declarations):
            self.prog_exe.write_memory(
                region_name=param_name, value=angles_list[i])

        return self.prog_exe

    @property
    def parametric_qaoa_circuit(self) -> Program:
        """
        Creates a parametric QAOA circuit (pyquil.Program object), given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit. 

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
        
        Returns
        -------
        `pyquil.Program`
            A pyquil.Program object.
        """
        if self.active_reset:
            parametric_circuit = Program(gates.RESET())
        else:
            parametric_circuit = Program()
        
        if self.rewiring != None:
            if self.rewiring in ['PRAGMA INITIAL_REWIRING "NAIVE"', 'PRAGMA INITIAL_REWIRING "PARTIAL"', '']:
                parametric_circuit += Program(self.rewiring)
            else:
                raise ValueError('Rewiring command not recognized. Please use ''PRAGMA INITIAL_REWIRING "NAIVE"'' or ''PRAGMA INITIAL_REWIRING "PARTIAL"''')
        
        # declare the read-out register
        ro = parametric_circuit.declare('ro', 'BIT', self.n_qubits)

        if self.prepend_state:
            parametric_circuit += self.prepend_state
            
        # Initial state is all |+>
        if self.init_hadamard:
            for i in self.qureg:
                parametric_circuit += gates.RZ(np.pi, self.qubit_mapping[i]) 
                parametric_circuit += gates.RX(np.pi/2, self.qubit_mapping[i]) 
                parametric_circuit += gates.RZ(np.pi/2, self.qubit_mapping[i]) 
                parametric_circuit += gates.RX(-np.pi/2, self.qubit_mapping[i]) 

        # create a list of gates in order of application on quantum circuit
        for each_gate in self.pseudo_circuit:
            gate_label = ''.join(str(label) for label in each_gate.pauli_label)
            angle_param = parametric_circuit.declare(
                f'pauli{gate_label}', 'REAL', 1)
            each_gate.rotation_angle = angle_param
            if isinstance(each_gate, RZZPauliGate) or isinstance(each_gate, SWAPGate):
                decomposition = each_gate.decomposition('standard2')
            else:
                decomposition = each_gate.decomposition('standard')
                
            # using the list above, construct the circuit
            for each_tuple in decomposition:
                gate = each_tuple[0]()
                qubits, rotation_angle = each_tuple[1]
                if isinstance(qubits,list):
                    new_qubits = [self.qubit_mapping[qubit] for qubit in qubits]
                else:
                    new_qubits = self.qubit_mapping[qubits]
                parametric_circuit = gate.apply_pyquil_gate(new_qubits,rotation_angle,parametric_circuit)

        if self.append_state:
            parametric_circuit += self.append_state
            
        # Measurement instructions
        for i, qubit in enumerate(self.qureg):
            parametric_circuit += gates.MEASURE(self.qubit_mapping[qubit], ro[i])

        parametric_circuit.wrap_in_numshots_loop(self.n_shots)

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
        executable_program = self.qaoa_circuit(params)

        result = self.device.quantum_computer.run(executable_program)

        # TODO: check the endian (big or little) ordering of measurement outcomes
        meas_list = [''.join(str(bit) for bit in bitstring)
                     for bitstring in result.readout_data['ro']]
        
        # Expose counts
        counts = Counter(list(meas_list))
        self.measurement_outcomes = counts
        return counts

    def circuit_to_qasm(self, params: QAOAVariationalBaseParams) -> str:
        """
        A method to convert the pyQuil program to a OpenQASM string.
        """
        raise NotImplementedError()
        # qasm_program = self.device.quantum_computer.compiler.quil_to_qasm(self.qaoa_circuit(params))
        # return qasm_program

    def reset_circuit(self):
        """
        Reset self.program after performing a computation. Also handle active reset and rewirings.
        """
        pass
