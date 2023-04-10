import unittest
import numpy as np
import json
import os
import pytest
import subprocess

from openqaoa.backends.qaoa_backend import (
    get_qaoa_backend,
    DEVICE_NAME_TO_OBJECT_MAPPER,
    DEVICE_ACCESS_OBJECT_MAPPER,
)
from openqaoa.qaoa_components import (
    Hamiltonian,
    create_qaoa_variational_params,
    QAOADescriptor,
)

from openqaoa_qiskit.backends import (
    QAOAQiskitBackendShotBasedSimulator,
)


from openqaoa.backends.wrapper import SPAMTwirlingWrapper

from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.backends.qaoa_device import create_device
from openqaoa.backends.basebackend import QAOABaseBackendShotBased


def get_params():
    cost_hamil = Hamiltonian.classical_hamiltonian([[0, 1]], [1], constant=0)
    mixer_hamil = X_mixer_hamiltonian(2)

    qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)
    variational_params_std = create_qaoa_variational_params(
        qaoa_descriptor, "standard", "ramp"
    )

    return qaoa_descriptor, variational_params_std


class TestingBaseWrapper(unittest.TestCase):
    """
    These tests check that the methods of the wrapper around the backend are working properly.
    """

    def test_setup(self):
        """
        Check that we can wrap around any backend, i.e. that the wrapper is backend-agnostic. 
        """
        qaoa_descriptor, variational_params_std = get_params()
        
        
        
        rigetti_args ={
    'as_qvm':True, 
    'execution_timeout':10,
    'compiler_timeout':100
}
        device = create_device(location='qcs', name='7q-noisy-qvm', **rigetti_args)
        
        device_name = 'qiskit.qasm_simulator'  # TODO for loop for all devices
        device = create_device(location="local", name=device_name)
        
        
        backend = get_qaoa_backend(
            qaoa_descriptor=qaoa_descriptor, device=device, wrapper=SPAMTwirlingWrapper, wrapper_options={'n_batches' : 6, 'calibration_data_location':'./tests/qpu_calibration_data/spam_twirling_mock.json'}, n_shots = 42
        )

        assert (
            sum(
                backend.get_counts(
                    params=variational_params_std, n_shots=42
                ).values()
            )
            == 42
        ), "`n_shots` is not being respected for the local simulator `{}` when calling backend.get_counts(n_shots=42).".format(
            device_name
        )
            

class TestingSPAMTwirlingWrapper(unittest.TestCase):
    """
    These tests check methods of the SPAM Twirling wrapper.

    """

    def test_get_counts(self):

        """
        Test get_counts in the spam twirling wrapper.
        """
        shots = 10000
        qaoa_descriptor, variate_params = get_params()
        qiskit_shot_backend = QAOAQiskitBackendShotBasedSimulator(qaoa_descriptor, shots, None, None, True, 1.0)

        shot_result = qiskit_shot_backend.get_counts(variate_params)

        self.assertEqual(type(shot_result), dict)

if __name__ == "__main__":
    unittest.main()
