import time
from typing import Optional, List
import warnings

# IBM Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_provider.job.exceptions import (
    IBMJobApiError,
    IBMJobInvalidStateError,
    IBMJobFailureError,
    IBMJobTimeoutError,
)
from qiskit.circuit import Parameter

from .devices import DeviceQiskit
from .gates_qiskit import QiskitGateApplicator
from openqaoa.backends.basebackend import (
    QAOABaseBackendShotBased,
    QAOABaseBackendCloud,
    QAOABaseBackendParametric,
)
from openqaoa.qaoa_components import QAOADescriptor
from openqaoa.qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)
from openqaoa.utilities import flip_counts


class QAOAQiskitQPUBackend(
    QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased
):
    """
    A QAOA simulator as well as for real QPU using qiskit as the backend

    Parameters
    ----------
    device: `DeviceQiskit`
        An object of the class ``DeviceQiskit`` which contains the credentials
        for accessing the QPU via cloud and the name of the device.
    qaoa_descriptor: `QAOADescriptor`
        An object of the class ``QAOADescriptor`` which contains information on
        circuit construction and depth of the circuit.
    n_shots: `int`
        The number of shots to be taken for each circuit.
    prepend_state: `QuantumCircuit`
        The state prepended to the circuit.
    append_state: `QuantumCircuit`
        The state appended to the circuit.
    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the
        QAOA part of the circuit.
    cvar_alpha: `float`
        The value of alpha for the CVaR method.
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        device: DeviceQiskit,
        n_shots: int,
        prepend_state: Optional[QuantumCircuit],
        append_state: Optional[QuantumCircuit],
        init_hadamard: bool,
        initial_qubit_mapping: Optional[List[int]] = None,
        cvar_alpha: float = 1,
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

        self.qureg = QuantumRegister(self.n_qubits)
        self.problem_reg = self.qureg[0 : self.problem_qubits]
        if self.initial_qubit_mapping is None:
            self.initial_qubit_mapping = (
                initial_qubit_mapping
                if initial_qubit_mapping is not None
                else list(range(self.n_qubits))
            )

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.qubits), (
                "Cannot attach a bigger circuit" "to the QAOA routine"
            )

        if self.device.provider_connected and self.device.qpu_connected:
            self.backend_qpu = self.device.backend_device
        elif self.device.provider_connected and self.device.qpu_connected in [
            False,
            None,
        ]:
            raise Exception(
                "Connection to {} was made. Error connecting to the specified backend.".format(
                    self.device.device_location.upper()
                )
            )

        else:

            raise Exception(
                "Error connecting to {}.".format(self.device.device_location.upper())
            )

        if self.device.n_qubits < self.n_qubits:
            raise Exception(
                "There are lesser qubits on the device than the number of qubits required for the circuit."
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
        if self.qaoa_descriptor.routed == True:
            transpiled_circuit = transpile(
                circuit_with_angles,
                self.backend_qpu,
                initial_layout=self.initial_qubit_mapping,
                optimization_level=0,
                routing_method="none",
            )
        else:
            transpiled_circuit = transpile(
                circuit_with_angles,
                self.backend_qpu,
                initial_layout=self.initial_qubit_mapping,
            )
        return transpiled_circuit

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
        creg = ClassicalRegister(len(self.problem_reg))
        parametric_circuit = QuantumCircuit(self.qureg, creg)
        gate_applicator = QiskitGateApplicator()

        if self.prepend_state:
            parametric_circuit = parametric_circuit.compose(self.prepend_state)
        # Initial state is all |+>
        if self.init_hadamard:
            parametric_circuit.h(self.problem_reg)

        self.qiskit_parameter_list = []
        for each_gate in self.abstract_circuit:
            # if gate is of type mixer or cost gate, assign parameter to it
            if each_gate.gate_label.type.value in ["MIXER", "COST"]:
                angle_param = Parameter(each_gate.gate_label.__repr__())
                self.qiskit_parameter_list.append(angle_param)
                each_gate.angle_value = angle_param
            decomposition = each_gate.decomposition("standard")
            # using the list above, construct the circuit
            for each_tuple in decomposition:
                gate = each_tuple[0](gate_applicator, *each_tuple[1])
                gate.apply_gate(parametric_circuit)

        if self.append_state:
            parametric_circuit = parametric_circuit.compose(self.append_state)

        # only measure the problem qubits
        if self.final_mapping is None:
            parametric_circuit.measure(self.problem_reg, creg)
        else:
            for idx, qubit in enumerate(self.final_mapping[0 : len(self.problem_reg)]):
                cbit = creg[idx]
                parametric_circuit.measure(qubit, cbit)

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        """
        Execute the circuit and obtain the counts

        Parameters
        ----------
        params: QAOAVariationalBaseParams
            The QAOA parameters - an object of one of the parameter classes, containing
            variable parameters.
        n_shots: int
            The number of times to run the circuit. If None, n_shots is set to the default: self.n_shots

        Returns
        -------
            A dictionary with the bitstring as the key and the number of counts
            as its value.
        """

        n_shots = self.n_shots if n_shots == None else n_shots

        circuit = self.qaoa_circuit(params)

        job_state = False
        no_of_job_retries = 0
        max_job_retries = 5

        while job_state == False:
            # initial_layout only passed if not azure device
            # if type(self.device).__name__ == "DeviceAzure":
            # job = self.backend_qpu.run(circuit, **input_items)
            job = self.backend_qpu.run(circuit, shots=n_shots)

            api_contact = False
            no_of_api_retries = 0
            max_api_retries = 5

            while api_contact == False:
                try:
                    self.job_id = job.job_id()
                    counts = job.result().get_counts()
                    api_contact = True
                    job_state = True
                except (IBMJobApiError, IBMJobTimeoutError):
                    print("There was an error when trying to contact the IBMQ API.")
                    job_state = True
                    no_of_api_retries += 1
                    time.sleep(5)
                except (IBMJobFailureError, IBMJobInvalidStateError):
                    print("There was an error with the state of the Job in IBMQ.")
                    no_of_job_retries += 1
                    break

                if no_of_api_retries >= max_api_retries:
                    raise ConnectionError(
                        "Number of API Retries exceeded Maximum allowed."
                    )

            if no_of_job_retries >= max_job_retries:
                raise ConnectionError("An Error Occurred with the Job(s) sent to IBMQ.")

        # Expose counts
        final_counts = flip_counts(counts)
        self.measurement_outcomes = final_counts
        return final_counts

    def circuit_to_qasm(self, params: QAOAVariationalBaseParams) -> str:
        """
        A method to convert the entire QAOA `QuantumCircuit` object into
        a OpenQASM string
        """
        raise NotImplementedError()
        # qasm_string = self.qaoa_circuit(params).qasm(formatted=True)
        # return qasm_string

    def reset_circuit(self):
        """
        Reset self.circuit after performing a computation
        """
        raise NotImplementedError()
