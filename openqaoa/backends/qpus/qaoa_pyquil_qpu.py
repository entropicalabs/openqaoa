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
from pyquil import Program, gates

from ...basebackend import QAOABaseBackendShotBased, QAOABaseBackendCloud, QAOABaseBackendParametric
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from ...backends.qpus.qpu_auth import AccessObjectPyQuil
from ...qaoa_parameters.pauligate import RZZPauliGate, SWAPGate


class QAOAPyQuilQPUBackend(QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased):
    """
    A QAOA backend object for real Rigetti QPUs

    Parameters
    ----------
    access_object: `AccessObjectPyquil`
        An access object for the Rigetti QPUs.
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
    rewiring:
        Rewiring scheme to be used for Pyquil. 
        Either 'PRAGMA INITIAL_REWIRING "NAIVE"' or 
        'PRAGMA INITIAL_REWIRING "PARTIAL"'. If None, defaults to NAIVE.
    active_reset: 
        #TODO
    """

    def __init__(self,
                 access_object: AccessObjectPyQuil,
                 circuit_params: QAOACircuitParams,
                 n_shots: int,
                 prepend_state: Program,
                 append_state: Program,
                 init_hadamard: bool,
                 cvar_alpha: float,
                 active_reset: bool = False,
                 rewiring=None
                 ):

        QAOABaseBackendShotBased.__init__(self,
                                          circuit_params,
                                          n_shots,
                                          prepend_state,
                                          append_state,
                                          init_hadamard,
                                          cvar_alpha)
        QAOABaseBackendCloud.__init__(self, access_object)

        self.active_reset = active_reset
        self.rewiring = rewiring
        self.qureg = self.circuit_params.qureg

        # TODO: access_object implementation for PyQuil
        self.parametric_circuit = self.parametric_qaoa_circuit
        native_prog = self.access_object.quantum_computer.compiler.quil_to_native_quil(
            self.parametric_circuit)
        self.prog_exe = self.access_object.quantum_computer.compiler.native_quil_to_executable(
            native_prog)
        # self.qaoa_circuit()

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> Program:
        """
        Creates a QAOA circuit (pyquil.Program object), given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit. To do this, you will need to subsequently execute the command self.eng.flush().

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        `pyquil.Program`
            A pyquil.Program object.
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
        Creates a QAOA circuit (pyquil.Program object), given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit. To do this, you will need to subsequently execute the command self.eng.flush().

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        `pyquil.Program`
            A pyquil.Program object.
        """
        parametric_circuit = Program()
        # declare the read-out register
        ro = parametric_circuit.declare('ro', 'BIT', self.n_qubits)

        if self.prepend_state:
            parametric_circuit += self.prepend_state
        # Initial state is all |+>
        if self.init_hadamard:
            for i in range(self.n_qubits):
                parametric_circuit += gates.H(i)

        # create a list of gates in order of application on quantum circuit
        low_level_gate_list = []  # TOCHECK - Variable is not used anywhere
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
                gate = each_tuple[0](*each_tuple[1])
                parametric_circuit = gate.apply_gate(
                    parametric_circuit, 'pyquil')

        if self.append_state:
            parametric_circuit += self.append_state
        # Measurement instructions
        for i, qubit in enumerate(self.qureg):
            parametric_circuit += gates.MEASURE(qubit, ro[i])

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

        result = self.access_object.quantum_computer.run(executable_program)

        # TODO: check the endian (big or little) ordering of measurement outcomes
        meas_list = [''.join(str(bit) for bit in bitstring)
                     for bitstring in result.readout_data['ro']]
        counts = Counter(list(meas_list))
        return counts

    def circuit_to_qasm(self, params: QAOAVariationalBaseParams) -> str:
        """
        A method to convert the pyQuil program to a OpenQASM string.
        """
        raise NotImplementedError()
        # qasm_program = self.access_object.quantum_computer.compiler.quil_to_qasm(self.qaoa_circuit(params))
        # return qasm_program

    def reset_circuit(self):
        """
        Reset self.program after performing a computation. Also handle active reset and rewirings.
        """
        pass
