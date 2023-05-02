import os
from copy import deepcopy
from typing import Optional, List
import warnings

from braket.circuits import Circuit
from braket.circuits.gates import H
from braket.circuits.result_types import Probability
from braket.circuits.free_parameter import FreeParameter
from braket.jobs.metrics import log_metric

from .devices import DeviceAWS
from .gates_braket import BraketGateApplicator
from openqaoa.backends.basebackend import (
    QAOABaseBackendShotBased,
    QAOABaseBackendCloud,
    QAOABaseBackendParametric,
)
from openqaoa.qaoa_components import QAOADescriptor
from openqaoa.qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)


class QAOAAWSQPUBackend(
    QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased
):
    """
    A QAOA simulator as well as for real QPU using Amazon Braket as backend

    Parameters
    ----------
    device: `DeviceAWS`
        An object of the class ``DeviceAWS`` which contains the credentials
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
    disable_qubit_rewiring: `bool`
        A boolean that determines whether qubit routing on the provider's end is
        used. This is False by default. Not all providers provide this feature.
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        device: DeviceAWS,
        n_shots: int,
        prepend_state: Optional[Circuit],
        append_state: Optional[Circuit],
        init_hadamard: bool,
        cvar_alpha: float,
        disable_qubit_rewiring: bool = False,
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
        self.gate_applicator = BraketGateApplicator()

        self.qureg = list(range(self.n_qubits))
        self.problem_reg = self.qureg[0 : self.problem_qubits]
        if self.initial_qubit_mapping is None:
            self.initial_qubit_mapping = (
                initial_qubit_mapping
                if initial_qubit_mapping is not None
                else list(range(self.n_qubits))
            )
        else:
            warnings.warn(
                "Ignoring the initial_qubit_mapping since the routing algorithm chose one"
            )
        self.disable_qubit_rewiring = disable_qubit_rewiring

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.qubits), (
                "Cannot attach a bigger circuit" " to the QAOA routine"
            )

        if self.device.provider_connected and self.device.qpu_connected:
            self.backend_qpu = self.device.backend_device
        elif (
            self.device.provider_connected is True
            and self.device.qpu_connected is False
        ):
            raise Exception(
                "Connection to AWS was made. Error connecting to the specified backend."
            )
        elif (
            self.device.provider_connected is True and self.device.qpu_connected is None
        ):
            raise Exception(
                "Connection to AWS was made. A device name was not specified."
            )
        else:
            raise Exception("Error connecting to AWS.")

        if self.device.n_qubits < self.n_qubits:
            raise Exception(
                "There are lesser qubits on the device than the number of qubits required for the circuit."
            )

        self.parametric_circuit = self.parametric_qaoa_circuit

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> Circuit:
        """
        The final QAOA circuit to be executed on the QPU.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        qaoa_circuit: `Circuit`
            The final QAOA circuit constructed using the angles from variational params.
        """
        parametric_circuit = deepcopy(self.parametric_circuit)

        if self.append_state:
            parametric_circuit += self.append_state

        # TODO: needs to be fixed --> measurement operations on problem qubits
        parametric_circuit += Probability.probability()

        angles_list = self.obtain_angles_for_pauli_list(self.abstract_circuit, params)
        memory_map = dict(
            zip(
                [
                    each_free_param_obj.name
                    for each_free_param_obj in self.braket_parameter_list
                ],
                angles_list,
            )
        )
        circuit_with_angles = parametric_circuit.make_bound_circuit(memory_map)

        return circuit_with_angles

    @property
    def parametric_qaoa_circuit(self) -> Circuit:
        """
        Creates a parametric QAOA circuit, given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit.
        """
        parametric_circuit = Circuit()

        if self.prepend_state:
            parametric_circuit += self.prepend_state

        # Initial state is all |+>
        if self.init_hadamard:
            for each_qubit in self.problem_reg:
                parametric_circuit += H.h(each_qubit)

        self.braket_parameter_list = []
        for each_gate in self.abstract_circuit:
            # if gate is of type mixer or cost gate, assign parameter to it
            if each_gate.gate_label.type.value in ["MIXER", "COST"]:
                angle_param = FreeParameter(each_gate.gate_label.__repr__())
                self.braket_parameter_list.append(angle_param)
                each_gate.angle_value = angle_param
            decomposition = each_gate.decomposition("standard")
            # using the list above, construct the circuit
            for each_tuple in decomposition:
                gate = each_tuple[0](self.gate_applicator, *each_tuple[1])
                gate.apply_gate(parametric_circuit)

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
            The number of times to run the circuit. If None, n_shots is set to the default: self.n_shots.

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
            job = self.backend_qpu.run(
                circuit,
                (self.device.s3_bucket_name, self.device.folder_name),
                shots=n_shots,
                disable_qubit_rewiring=self.disable_qubit_rewiring,
            )

            try:
                self.job_id = job.id

                job_result = job.result()

                # If there was an issue with the job sent, send again.
                if job.state() in ["FAILED", "CANCELLED"] or job_result == None:
                    raise ValueError

                counts = job_result.measurement_counts

            except ValueError:
                print("The task has failed or was cancelled by AWS. Resending task.")
                no_of_job_retries += 1

            except Exception as e:
                print(e, "\n")
                print(
                    "An unknown error occurred while trying to retrieve task results. Resending task."
                )
                no_of_job_retries += 1

            else:
                job_state = True

            finally:
                if no_of_job_retries >= max_job_retries:
                    raise ConnectionError(
                        "An Error Occurred with the Task(s) sent to AWS."
                    )

        final_counts = counts
        # if self.final_mapping is not None:
        #     final_counts = permute_counts_dictionary(final_counts,
        #                                             self.final_mapping)
        # # Expose counts
        self.measurement_outcomes = final_counts
        return final_counts

    def log_with_backend(self, metric_name: str, value, iteration_number) -> None:
        """
        If using AWS Jobs, these values will be logged.
        """

        try:
            if os.environ["AMZN_BRAKET_JOB_NAME"] is not None:
                in_jobs = True
        except KeyError:
            in_jobs = False

        if in_jobs:
            log_metric(
                metric_name=metric_name,
                value=value,
                iteration_number=iteration_number,
            )

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
