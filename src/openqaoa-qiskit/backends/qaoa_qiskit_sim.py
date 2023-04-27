import numpy as np
from typing import Union, List, Tuple, Optional

# IBM Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter

from .gates_qiskit import QiskitGateApplicator
from openqaoa.backends.basebackend import (
    QAOABaseBackendParametric,
    QAOABaseBackendShotBased,
    QAOABaseBackendStatevector,
)
from openqaoa.qaoa_components import QAOADescriptor
from openqaoa.qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)
from openqaoa.utilities import (
    flip_counts,
    generate_uuid,
    round_value,
)
from openqaoa.backends.cost_function import cost_function
from openqaoa.qaoa_components.ansatz_constructor import (
    RXGateMap,
    RYGateMap,
    RZGateMap,
    RXXGateMap,
    RYYGateMap,
    RZZGateMap,
    RZXGateMap,
)


class QAOAQiskitBackendShotBasedSimulator(
    QAOABaseBackendShotBased, QAOABaseBackendParametric
):
    """
    Local Shot-based simulators offered by Qiskit

    Parameters
    ----------
    qaoa_descriptor: `QAOADescriptor`
        An object of the class ``QAOADescriptor`` which contains information on
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

    QISKIT_GATEMAP_LIBRARY = [
        RXGateMap,
        RYGateMap,
        RZGateMap,
        RXXGateMap,
        RYYGateMap,
        RZZGateMap,
        RZXGateMap,
    ]

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        n_shots: int,
        prepend_state: Optional[QuantumCircuit],
        append_state: Optional[QuantumCircuit],
        init_hadamard: bool,
        cvar_alpha: float,
        qiskit_simulation_method: str = "automatic",
        seed_simulator: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
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
        self.gate_applicator = QiskitGateApplicator()
        self.qureg = QuantumRegister(self.n_qubits)

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.qubits), (
                "Cannot attach a bigger circuit " "to the QAOA routine"
            )
        # options = {"seed_simulator":1}
        self.backend_simulator = AerSimulator(
            method=qiskit_simulation_method.lower(),
            noise_model=noise_model,
            seed_simulator=seed_simulator,
        )
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
        circuit_with_angles = self.parametric_circuit.bind_parameters(memory_map)

        if self.append_state:
            circuit_with_angles = circuit_with_angles.compose(self.append_state)
        circuit_with_angles.measure_all()

        return circuit_with_angles

    @property
    def parametric_qaoa_circuit(self) -> QuantumCircuit:
        """
        Creates a parametric QAOA circuit, given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit.
        """
        # self.reset_circuit()
        parametric_circuit = QuantumCircuit(
            self.qureg
        )  # consider changing this too with my new function

        if self.prepend_state:
            parametric_circuit = parametric_circuit.compose(self.prepend_state)
        # Initial state is all |+>
        if self.init_hadamard:
            parametric_circuit.h(self.qureg)

        self.qiskit_parameter_list = []
        for each_gate in self.abstract_circuit:
            # if gate is of type mixer or cost gate, assign parameter to it
            if each_gate.gate_label.type.value in ["MIXER", "COST"]:
                angle_param = Parameter(each_gate.gate_label.__repr__())
                self.qiskit_parameter_list.append(angle_param)
                each_gate.angle_value = angle_param
            if (
                type(each_gate)
                in QAOAQiskitBackendShotBasedSimulator.QISKIT_GATEMAP_LIBRARY
            ):
                decomposition = each_gate.decomposition("trivial")
            else:
                decomposition = each_gate.decomposition("standard")
            # Create Circuit
            for each_tuple in decomposition:
                gate = each_tuple[0](self.gate_applicator, *each_tuple[1])
                gate.apply_gate(parametric_circuit)

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        """
        Returns the counts of the final QAOA circuit after binding angles from variational parameters.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing variable parameters.
        n_shots: `int`
            The number of times to run the circuit. If None, n_shots is set to the default: self.n_shots

        Returns
        -------
        counts: `dict`
            The counts of the final QAOA circuit after binding angles from variational parameters.
        """
        # generate a job id for the wavefunction evaluation
        self.job_id = generate_uuid()

        # set the number of shots, if not specified take the default
        n_shots = self.n_shots if n_shots == None else n_shots

        qaoa_circuit = self.qaoa_circuit(params)
        counts = (
            self.backend_simulator.run(qaoa_circuit, shots=n_shots)
            .result()
            .get_counts()
        )

        final_counts = flip_counts(counts)
        self.measurement_outcomes = final_counts
        return final_counts

    def circuit_to_qasm(self):
        """
        A method to convert the QAOA circuit to QASM.
        """
        raise NotImplementedError()

    #         qasm_circuit = self.parametric_circuit.qasm()
    #         return qasm_circuit

    def reset_circuit(self):
        raise NotImplementedError()


class QAOAQiskitBackendStatevecSimulator(
    QAOABaseBackendStatevector, QAOABaseBackendParametric
):
    """
    Local Statevector-based simulators offered by Qiskit

    Parameters
    ----------
    qaoa_descriptor: `QAOADescriptor`
        An object of the class ``QAOADescriptor`` which contains information on
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

    QISKIT_GATEMAP_LIBRARY = [
        RXGateMap,
        RYGateMap,
        RZGateMap,
        RXXGateMap,
        RYYGateMap,
        RZZGateMap,
        RZXGateMap,
    ]

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        prepend_state: Optional[Union[np.ndarray, QuantumCircuit]],
        append_state: Optional[Union[np.ndarray, QuantumCircuit]],
        init_hadamard: bool,
        cvar_alpha: float = 1,
    ):
        QAOABaseBackendStatevector.__init__(
            self,
            qaoa_descriptor,
            prepend_state,
            append_state,
            init_hadamard,
            cvar_alpha,
        )

        assert (
            cvar_alpha == 1
        ), "Please use the shot-based simulator for simulations with cvar_alpha < 1"

        self.qureg = QuantumRegister(self.n_qubits)
        self.gate_applicator = QiskitGateApplicator()

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.qubits), (
                "Cannot attach a bigger circuit " "to the QAOA routine"
            )

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
        pauli_strings_list = ["I" * n_qubits] * len(cost_hamil.terms)
        for i, pauli_op in enumerate(cost_hamil.terms):
            pauli_term = list(pauli_strings_list[i])
            for pauli, qubit in zip(pauli_op.pauli_str, pauli_op.qubit_indices):
                pauli_term[qubit] = pauli
            # reverse pauli_strings because qiskit supports little endian notation
            pauli_strings_list[i] = "".join(str(term) for term in pauli_term)[::-1]

        pauli_strings_list.append("I" * n_qubits)
        pauli_coeffs = cost_hamil.coeffs

        qiskit_pauli_op = [
            [pauli_strings, coeff]
            for pauli_strings, coeff in zip(pauli_strings_list, pauli_coeffs)
        ]
        qiskit_pauli_op.append(["I" * n_qubits, cost_hamil.constant])
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
        # generate a job id for the wavefunction evaluation
        self.job_id = generate_uuid()

        angles_list = self.obtain_angles_for_pauli_list(self.abstract_circuit, params)
        memory_map = dict(zip(self.qiskit_parameter_list, angles_list))
        circuit_with_angles = self.parametric_circuit.bind_parameters(memory_map)
        return circuit_with_angles

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

        self.qiskit_parameter_list = []
        for each_gate in self.abstract_circuit:
            # if gate is of type mixer or cost gate, assign parameter to it
            if each_gate.gate_label.type.value in ["MIXER", "COST"]:
                angle_param = Parameter(each_gate.gate_label.__repr__())
                self.qiskit_parameter_list.append(angle_param)
                each_gate.angle_value = angle_param
            if (
                type(each_gate)
                in QAOAQiskitBackendStatevecSimulator.QISKIT_GATEMAP_LIBRARY
            ):
                decomposition = each_gate.decomposition("trivial")
            else:
                decomposition = each_gate.decomposition("standard")
            # Create Circuit
            for each_tuple in decomposition:
                gate = each_tuple[0](self.gate_applicator, *each_tuple[1])
                gate.apply_gate(parametric_circuit)

        if self.append_state:
            parametric_circuit = parametric_circuit.compose(self.append_state)

        return parametric_circuit

    def wavefunction(
        self, params: QAOAVariationalBaseParams
    ) -> Union[List[complex], np.ndarray]:
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

    @round_value
    def expectation(self, params: QAOAVariationalBaseParams) -> float:
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

    @round_value
    def expectation_w_uncertainty(
        self, params: QAOAVariationalBaseParams
    ) -> Tuple[float, float]:
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
        """ """
        raise NotImplementedError()
