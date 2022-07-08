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
import numpy as np
from time import time
from typing import Optional, List
# IBM Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit.providers.ibmq.job import (IBMQJobApiError, IBMQJobInvalidStateError,
                                       IBMQJobFailureError, IBMQJobTimeoutError)
from qiskit.circuit import Parameter

from ...devices import DeviceQiskit
from ...basebackend import QAOABaseBackendShotBased, QAOABaseBackendCloud, QAOABaseBackendParametric
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from ...utilities import flip_counts


# add support to perform error mitigation for a future version
# from qiskit.ignis.mitigation.measurement import (complete_meas_cal,
#                                                  CompleteMeasFitter)


class QAOAQiskitQPUBackend(QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased):
    """
    A QAOA simulator as well as for real QPU using qiskit as the backend

    Parameters
    ----------
    device: `DeviceQiskit`
        An object of the class ``DeviceQiskit`` which contains the credentials
        for accessing the QPU via cloud and the name of the device.
    circuit_params: `QAOACircuitParams`
        An object of the class ``QAOACircuitParams`` which contains information on 
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

    def __init__(self,
                 circuit_params: QAOACircuitParams,
                 device: DeviceQiskit,
                 n_shots: int,
                 prepend_state: Optional[QuantumCircuit],
                 append_state: Optional[QuantumCircuit],
                 init_hadamard: bool,
                 qubit_layout: List[int] = [],
                 cvar_alpha: float = 1):

        QAOABaseBackendShotBased.__init__(self,
                                          circuit_params,
                                          n_shots,
                                          prepend_state,
                                          append_state,
                                          init_hadamard,
                                          cvar_alpha)
        QAOABaseBackendCloud.__init__(self, device)

        self.qureg = QuantumRegister(self.n_qubits)
        self.qubit_layout = self.circuit_params.qureg if qubit_layout == [] else qubit_layout

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.qubits), "Cannot attach a bigger circuit" \
                                                               "to the QAOA routine"

        if self.device.provider_connected and self.device.qpu_connected:
            self.backend_qpu = self.device.backend_device
        elif self.device.provider_connected and self.device.qpu_connected in [False, None]:
            raise Exception(
                'Connection to IBMQ was made. Error connecting to the specified backend.')
        else:
            raise Exception('Error connecting to IBMQ.')
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

        angles_list = self.obtain_angles_for_pauli_list(
            self.pseudo_circuit, params)
        memory_map = dict(zip(self.qiskit_parameter_list, angles_list))
        new_parametric_circuit = self.parametric_circuit.bind_parameters(
            memory_map)
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

        self.qiskit_parameter_list = []
        for each_gate in self.pseudo_circuit:
            angle_param = Parameter(str(each_gate.pauli_label))
            self.qiskit_parameter_list.append(angle_param)
            each_gate.rotation_angle = angle_param
            decomposition = each_gate.decomposition('standard')
            # using the list above, construct the circuit
            for each_tuple in decomposition:
                gate = each_tuple[0]()
                parametric_circuit = gate.apply_ibm_gate(*each_tuple[1],parametric_circuit)

        if self.append_state:
            parametric_circuit = parametric_circuit.compose(self.append_state)
        parametric_circuit.measure_all()

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams) -> dict:
        """
        Execute the circuit and obtain the counts

        Parameters
        ----------
        params: QAOAVariationalBaseParams
            The QAOA parameters - an object of one of the parameter classes, containing 
            variable parameters.

        Returns
        -------
            A dictionary with the bitstring as the key and the number of counts 
            as its value.
        """

        circuit = self.qaoa_circuit(params)

        job_state = False
        no_of_job_retries = 0
        max_job_retries = 5

        while job_state == False:
            job = execute(circuit, backend=self.backend_qpu,
                          shots=self.n_shots, initial_layout=self.qubit_layout)

            api_contact = False
            no_of_api_retries = 0
            max_api_retries = 5

            while api_contact == False:
                try:
                    counts = job.result().get_counts()
                    api_contact = True
                    job_state = True
                except (IBMQJobApiError, IBMQJobTimeoutError):
                    print("There was an error when trying to contact the IBMQ API.")
                    job_state = True
                    no_of_api_retries += 1
                    time.sleep(5)
                except (IBMQJobFailureError, IBMQJobInvalidStateError):
                    print(
                        "There was an error with the state of the Job in IBMQ.")
                    no_of_job_retries += 1
                    break

                if no_of_api_retries >= max_api_retries:
                    raise ConnectionError(
                        "Number of API Retries exceeded Maximum allowed.")

            if no_of_job_retries >= max_job_retries:
                raise ConnectionError(
                    "An Error Occurred with the Job(s) sent to IBMQ.")

        # Expose counts
        counts_flipped = flip_counts(counts)
        self.measurement_outcomes = counts_flipped
        return counts_flipped

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
