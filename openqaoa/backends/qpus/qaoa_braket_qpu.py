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
import os
from typing import Optional, List
# AWS Braket imports

from braket.circuits import Circuit
from braket.circuits.gates import H
from braket.circuits.result_types import Probability
from braket.circuits.free_parameter import FreeParameter
from braket.jobs.metrics import log_metric

from ...devices import DeviceAWS
from ...basebackend import QAOABaseBackendShotBased, QAOABaseBackendCloud, QAOABaseBackendParametric
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams

class QAOAAWSQPUBackend(QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased):
    """
    A QAOA simulator as well as for real QPU using Amazon Braket as backend

    Parameters
    ----------
    device: `DeviceAWS`
        An object of the class ``DeviceAWS`` which contains the credentials
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
    disable_qubit_rewiring: `bool`
        A boolean that determines whether qubit routing on the provider's end is 
        used. This is False by default. Not all providers provide this feature.
    """

    def __init__(self,
                 circuit_params: QAOACircuitParams,
                 device: DeviceAWS,
                 n_shots: int,
                 prepend_state: Optional[Circuit],
                 append_state: Optional[Circuit],
                 init_hadamard: bool,
                 cvar_alpha: float,
                 qubit_layout: List[int] = [], 
                 disable_qubit_rewiring: bool = False):

        QAOABaseBackendShotBased.__init__(self,
                                          circuit_params,
                                          n_shots,
                                          prepend_state,
                                          append_state,
                                          init_hadamard,
                                          cvar_alpha)
        QAOABaseBackendCloud.__init__(self, device)

        self.qureg = self.circuit_params.qureg
        self.qubit_layout = self.qureg if qubit_layout == [] else qubit_layout
        self.disable_qubit_rewiring = disable_qubit_rewiring

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.qubits), "Cannot attach a bigger circuit" \
                                                               "to the QAOA routine"

        if self.device.provider_connected and self.device.qpu_connected:
            self.backend_qpu = self.device.backend_device
        elif self.device.provider_connected and self.device.qpu_connected in [False, None]:
            raise Exception(
                'Connection to AWS was made. Error connecting to the specified backend.')
        else:
            raise Exception('Error connecting to AWS.')
            
        self.parametric_circuit = self.parametric_qaoa_circuit()

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
        
        angles_list = self.obtain_angles_for_pauli_list(
            self.abstract_circuit, params)
        memory_map = dict(zip([each_free_param_obj.name for each_free_param_obj in self.braket_parameter_list], angles_list))
        new_parametric_circuit = self.parametric_circuit.make_bound_circuit(
            memory_map)
        return new_parametric_circuit

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
            for each_qubit in self.qubit_layout:
                parametric_circuit += H.h(each_qubit)

        self.braket_parameter_list = []
        for each_gate in self.abstract_circuit:
            angle_param = FreeParameter(str(each_gate.pauli_label))
            self.braket_parameter_list.append(angle_param)
            each_gate.rotation_angle = angle_param
            decomposition = each_gate.decomposition('standard')
            # using the list above, construct the circuit
            for each_tuple in decomposition:
                gate = each_tuple[0]()
                parametric_circuit = gate.apply_braket_gate(*each_tuple[1], parametric_circuit)

        if self.append_state:
            parametric_circuit += self.append_state
            
        parametric_circuit += Probability.probability()

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
            job = self.backend_qpu.run(circuit, 
                                       (self.device.s3_bucket_name, 
                                        self.device.folder_name), 
                                       shots = self.n_shots, 
                                       disable_qubit_rewiring = self.disable_qubit_rewiring)
            
            try:
                self.job_id = job.id
                
                job_result = job.result()
                
                # If there was an issue with the job sent, send again.
                if job.state() in ['FAILED', 'CANCELLED'] or job_result == None:
                    raise ValueError
                    
                counts = job_result.measurement_counts
                    
            except ValueError:
                print('The task has failed or was cancelled by AWS. Resending task.')
                no_of_job_retries += 1
                    
            except Exception as e:
                print(e, '\n')
                print("An unknown error occurred while trying to retrieve task results. Resending task.")
                no_of_job_retries += 1
                
            else:
                job_state = True
                
            finally:
                if no_of_job_retries >= max_job_retries:
                    raise ConnectionError(
                        "An Error Occurred with the Task(s) sent to AWS.")

        # Expose counts
        self.measurement_outcomes = counts
        return counts
    
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
