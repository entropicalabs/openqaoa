import unittest
from unittest.mock import Mock
import json
import numpy as np
import pytest
import subprocess

import openqaoa
from openqaoa_azure.backends import DeviceAzure
from openqaoa.qaoa_components import (
    create_qaoa_variational_params,
    PauliOp,
    Hamiltonian,
    QAOADescriptor,
    QAOAVariationalStandardParams,
)
from openqaoa_qiskit.backends import (
    QAOAQiskitQPUBackend,
)
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.problems import NumberPartition
from openqaoa import QAOA


class TestingQAOAQiskitQPUBackendAzure(unittest.TestCase):
    
    """This Object tests the QAOA Qiskit QPU Backend object with the Azure 
    Device object. This checks that the use of qiskit to send circuits to Azure
    is working as intended.
    
    The Azure CLI has to be configured beforehand to run these tests.
    """
    
    @pytest.mark.api
    def setUp(self):
        
        bashCommand = "az resource list"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        
        if error is not None:
            print(error)
            raise Exception('You must have the Azure CLI installed and must be logged in to use the Azure Quantum Backends')
        else:
            output_json = json.loads(output)
            output_json_s = [each_json for each_json in output_json if each_json['name'] == 'TestingOpenQAOA'][0]
            self.RESOURCE_ID = output_json_s['id']
            self.AZ_LOCATION = output_json_s['location']
            
    @pytest.mark.sim
    def check_shots_tally(self):
        
        """There is a known bug in the qiskit backend for azure where if the shots 
        argument might be ignored. This test checks that the output from the azure 
        computation matches the input requirements.
        """
        
        shots = 1024
        problem_qubo = NumberPartition([1, 2, 3]).qubo
        azure_device = create_device(location='azure', name='rigetti.sim.qvm', 
                                     resource_id=self.RESOURCE_ID, 
                                     az_location=self.AZ_LOCATION)
        
        q = QAOA()
        q.set_device(azure_device)
        q.set_backend_properties(n_shots=shots)
        q.set_classical_optimizer(maxiter = 1)
        q.compile(problem_qubo)
        q.optimize()
        
        comp_shots = sum(q.result.optimized['optimized measurement outcomes'].values())
        
        self.assertEqual(shots, comp_shots)
        
    @pytest.mark.api
    def test_expectations_in_init(self):
        
        """
        Testing the Exceptions in the init function of the QiskitQPUShotBasedBackend
        """
        
        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1/8*np.pi]
        betas = [1/8*np.pi]
        shots = 10000

        cost_hamil = Hamiltonian([PauliOp('ZZ', (0, 1)), PauliOp('ZZ', (1, 2)),
                                  PauliOp('ZZ', (0, 2))], weights, 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor,
                                                       betas, gammas)
        
        # We mock the potential Exception that could occur in the Device class
        azure_device = DeviceAzure('', '', '')
        azure_device._check_provider_connection = Mock(return_value=False)
        
        try:
            QAOAQiskitQPUBackend(qaoa_descriptor, azure_device, 
                                 shots, None, None, True)
        except Exception as e:
            self.assertEqual(str(e), 'Error connecting to AZURE.')
        
        with self.assertRaises(Exception):
            QAOAQiskitQPUBackend(qaoa_descriptor, azure_device, shots, None, None, True)
        

        azure_device = DeviceAzure(device_name='', resource_id=self.RESOURCE_ID, 
                                   az_location=self.AZ_LOCATION)
        
        try:
            QAOAQiskitQPUBackend(qaoa_descriptor, azure_device, 
                                 shots, None, None, True)
        except Exception as e:
            self.assertEqual(str(e), 'Connection to AZURE was made. Error connecting to the specified backend.')
        
        with self.assertRaises(Exception):
            QAOAQiskitQPUBackend(qaoa_descriptor, azure_device, shots, None, None, True)
        
    @pytest.mark.api
    def test_remote_qubit_overflow(self):
        
        """
        If the user creates a circuit that is larger than the maximum circuit size
        that is supported by the QPU. An Exception should be raised with the 
        appropriate error message alerting the user to the error.
        """
        
        shots = 100
        
        set_of_numbers = np.random.randint(1, 10, 6).tolist()
        qubo = NumberPartition(set_of_numbers).qubo

        mixer_hamil = X_mixer_hamiltonian(n_qubits=6)
        qaoa_descriptor = QAOADescriptor(qubo.hamiltonian, mixer_hamil, p=1)
        variate_params = create_qaoa_variational_params(qaoa_descriptor, 'standard', 'rand')

        azure_device = DeviceAzure('rigetti.sim.qvm', self.RESOURCE_ID, self.AZ_LOCATION)
        
        try:
            QAOAQiskitQPUBackend(qaoa_descriptor, azure_device, 
                                 shots, None, None, True)
        except Exception as e:
            self.assertEqual(str(e), 'There are lesser qubits on the device than the number of qubits required for the circuit.')


if __name__ == "__main__":
    unittest.main()
