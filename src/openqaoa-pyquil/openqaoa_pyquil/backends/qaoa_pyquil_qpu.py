from collections import Counter
import numpy as np
from copy import deepcopy
from pyquil import Program, gates, quilbase
from typing import List, Optional
import warnings

from .devices import DevicePyquil
from .gates_pyquil import PyquilGateApplicator
from openqaoa.backends.basebackend import (
    QAOABaseBackendShotBased,
    QAOABaseBackendCloud,
    QAOABaseBackendParametric,
)
from openqaoa.qaoa_components import QAOADescriptor
from openqaoa.qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)
from openqaoa.qaoa_components.ansatz_constructor.gatemap import RZZGateMap, SWAPGateMap
from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle
from openqaoa.utilities import generate_uuid


def check_edge_connectivity(executable: Program, device: DevicePyquil):
    '''
    Check that the program does not contain 2-qubit terms that is not present
    in the QPU's topology (to prevent quilc from crashing).

    Parameters
    ----------
    executable: `Program`
        pyQuil executable program.
    device: `DevicePyquil`
        An object of the class ``DevicePyquil`` which contains information on
        pyQuil's `QuantumComputer` object, used to extract the selected QPU's topology.

    Returns
    -------
        None
    """
    '''

    qpu_graph = device.quantum_computer.qubit_topology()

    instrs = [instr for instr in executable if type(instr) == quilbase.Gate]
    pair_instrs = [
        list(instr.get_qubits()) for instr in instrs if len(instr.get_qubits()) == 2
    ]

    for term in pair_instrs:
        if len(term) == 2:
            assert (
                term in qpu_graph.edges()
            ), f"Term {term} is not an edge on the QPU graph of {device.device_name}."


class QAOAPyQuilQPUBackend(
    QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased
):
    """
    A QAOA backend object for real Rigetti QPUs

    Parameters
    ----------
    device: `DevicePyquil`
        The device object to access pyquil devices with credentials.
    qaoa_descriptor: `QAOADescriptor`
        An object of the class ``QAOADescriptor`` which contains information on
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
        NAIVE: The qubits used in all instructions in the program satisfy the
        topological constraints of the device.
        PARTIAL: Otherwise.
    """

    def __init__(
        self,
        device: DevicePyquil,
        qaoa_descriptor: QAOADescriptor,
        n_shots: int,
        prepend_state: Program,
        append_state: Program,
        init_hadamard: bool,
        cvar_alpha: float,
        active_reset: bool = False,
        rewiring: str = "",
        initial_qubit_mapping: Optional[List[int]] = None,
    ):
        QAOABaseBackendShotBased.__init__(
            self,
            qaoa_descriptor,
            n_shots,
            prepend_state,
            append_state,
            init_hadamard,
            cvar_alpha,
        )
        QAOABaseBackendCloud.__init__(self, device)

        self.gate_applicator = PyquilGateApplicator()

        self.active_reset = active_reset
        self.rewiring = rewiring
        self.qureg = list(range(self.n_qubits))
        self.problem_reg = self.qureg[0 : self.problem_qubits]

        if self.initial_qubit_mapping is None:
            self.initial_qubit_mapping = (
                initial_qubit_mapping
                if initial_qubit_mapping is not None
                else list(range(self.n_qubits))
            )

        # self.qureg_placeholders = QubitPlaceholder.register(self.n_qubits)
        self.qubit_mapping = dict(zip(self.qureg, self.initial_qubit_mapping))

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.get_qubits()), (
                "Cannot attach a bigger circuit " "to the QAOA routine"
            )

        if self.device.n_qubits < self.n_qubits:
            raise Exception(
                "There are lesser qubits on the device than the number of qubits required for the circuit."
            )

        self.parametric_circuit = self.parametric_qaoa_circuit

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
        parametric_circuit = deepcopy(self.parametric_circuit)
        # declare the read-out register
        ro = parametric_circuit.declare("ro", "BIT", self.problem_qubits)

        if self.append_state:
            parametric_circuit += self.append_state

        if self.final_mapping is None:
            for i, qbit in enumerate(self.problem_reg):
                parametric_circuit += gates.MEASURE(self.qubit_mapping[qbit], ro[i])
        else:
            # Measurement instructions
            for i, qubit in enumerate(self.final_mapping[0 : len(self.problem_reg)]):
                cbit = ro[i]
                parametric_circuit += gates.MEASURE(self.qubit_mapping[qubit], cbit)
        parametric_circuit.wrap_in_numshots_loop(self.n_shots)

        native = self.device.quantum_computer.compiler.quil_to_native_quil(
            parametric_circuit
        )

        prog_exe = self.device.quantum_computer.compiler.native_quil_to_executable(
            native
        )

        angles_list = np.array(
            self.obtain_angles_for_pauli_list(self.abstract_circuit, params),
            dtype=float,
        )
        angle_declarations = list(parametric_circuit.declarations.keys())

        angle_declarations.remove("ro")

        for i, param_name in enumerate(angle_declarations):
            prog_exe.write_memory(region_name=param_name, value=angles_list[i])

        return prog_exe

    @property
    def parametric_qaoa_circuit(self) -> Program:
        """
        Creates a parametric QAOA circuit (pyquil.Program object),
        given the qubit pairs, single qubits with biases,
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
            if self.rewiring in [
                'PRAGMA INITIAL_REWIRING "NAIVE"',
                'PRAGMA INITIAL_REWIRING "PARTIAL"',
                "",
            ]:
                parametric_circuit += Program(self.rewiring)
            else:
                raise ValueError(
                    "Rewiring command not recognized. Please use "
                    'PRAGMA INITIAL_REWIRING "NAIVE"'
                    " or "
                    'PRAGMA INITIAL_REWIRING "PARTIAL"'
                    ""
                )

        if self.prepend_state:
            parametric_circuit += self.prepend_state

        # Initial state is all |+>
        if self.init_hadamard:
            for i in self.problem_reg:
                parametric_circuit += gates.RZ(np.pi, self.qubit_mapping[i])
                parametric_circuit += gates.RX(np.pi / 2, self.qubit_mapping[i])
                parametric_circuit += gates.RZ(np.pi / 2, self.qubit_mapping[i])
                parametric_circuit += gates.RX(-np.pi / 2, self.qubit_mapping[i])

        # create a list of gates in order of application on quantum circuit
        for each_gate in self.abstract_circuit:
            # if gate is of type mixer or cost gate, assign parameter to it
            if each_gate.gate_label.type.value in ["MIXER", "COST"]:
                gatelabel_pyquil = each_gate.gate_label.__repr__()
                gatelabel_pyquil = (
                    "one" + gatelabel_pyquil[1:]
                    if each_gate.gate_label.n_qubits == 1
                    else "two" + gatelabel_pyquil[1:]
                )
                angle_param = parametric_circuit.declare(gatelabel_pyquil, "REAL", 1)
                each_gate.angle_value = angle_param
            if isinstance(each_gate, RZZGateMap) or isinstance(each_gate, SWAPGateMap):
                decomposition = each_gate.decomposition("standard2")
            else:
                decomposition = each_gate.decomposition("standard")

            # using the list above, construct the circuit
            for each_tuple in decomposition:
                if type(each_tuple[1][-1]) == RotationAngle:
                    rotation_angle = each_tuple[1][-1]
                    qubits = each_tuple[1][:-1]
                else:
                    rotation_angle = None
                    qubits = each_tuple[1]
                if not isinstance(qubits, list):
                    qubits = [qubits]
                new_qubits = [self.qubit_mapping[qubit] for qubit in qubits]

                if rotation_angle is None:
                    gate = each_tuple[0](self.gate_applicator, *new_qubits)
                else:
                    gate = each_tuple[0](
                        self.gate_applicator, *new_qubits, rotation_angle
                    )
                gate.apply_gate(parametric_circuit)

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        """
        Execute the circuit and obtain the counts.

        Parameters
        ----------
        params: QAOAVariationalBaseParams
            The QAOA parameters - an object of one of the parameter classes, containing
            variable parameters.
        n_shots: int
            The number of times to run the circuit. If None, n_shots is set to the default: self.n_shots

        Returns
        -------
        counts : dictionary
            A dictionary with the bitstring as the key and the number of counts as its value.
        """

        executable_program = self.qaoa_circuit(params)

        # if n_shots is given change the number of shots
        if n_shots is not None:
            executable_program.wrap_in_numshots_loop(n_shots)

        result = self.device.quantum_computer.run(executable_program)
        # we create an uuid for the job
        self.job_id = generate_uuid()

        # TODO: check the endian (big or little) ordering of measurement outcomes
        meas_list = [
            "".join(str(bit) for bit in bitstring)
            for bitstring in result.readout_data["ro"]
        ]

        # Expose counts
        final_counts = Counter(list(meas_list))
        self.measurement_outcomes = final_counts
        return final_counts

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
