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


class TestingSPAMTwirlingWrapper(unittest.TestCase):
    """
    These tests check methods of the SPAM Twirling wrapper.

    """

    def setUp(
        self,
    ):
        self.n_batches = 6
        self.calibration_data_location = (
            "./tests/qpu_calibration_data/spam_twirling_mock.json"
        )
        self.qaoa_descriptor, self.variate_params = get_params()
        qiskit_shot_backend = QAOAQiskitBackendShotBasedSimulator(
            self.qaoa_descriptor, 100, None, None, True, 1.0
        )
        self.wrapped_obj = SPAMTwirlingWrapper(
            qiskit_shot_backend,
            n_batches=self.n_batches,
            calibration_data_location=self.calibration_data_location,
        )

    def test_wrap_any_backend(self):
        """
        Check that we can wrap around any backend, i.e. that the wrapper is backend-agnostic.
        """
        qaoa_descriptor, variational_params_std = get_params()

        rigetti_args = {
            "as_qvm": True,
            "execution_timeout": 10,
            "compiler_timeout": 100,
        }
        device_list = [
            create_device(location="local", name="qiskit.qasm_simulator"),
            create_device(location="qcs", name="7q-noisy-qvm", **rigetti_args),
        ]
        for device in device_list:
            backend = get_qaoa_backend(
                    qaoa_descriptor=self.qaoa_descriptor,
                    device=device,
                    n_shots=42,
                )
            try:
                SPAMTwirlingWrapper(
            backend,
            n_batches=self.n_batches,
            calibration_data_location=self.calibration_data_location,
        )
            except:
                raise ValueError("The {} backend cannot be wrapped.".format(backend))

    def test_setUp(self):
        assert (
            self.wrapped_obj.n_batches == self.n_batches
        ), "The number of batches hasn't been set correctly."
        assert (
            self.wrapped_obj.calibration_data_location == self.calibration_data_location
        ), "The location of the calibration file hasn't been set correctly."

    def test_get_counts(self):
        """
        Test get_counts in the spam twirling wrapper.
        """

        self.wrapped_obj.get_counts(
            self.variate_params
        )  # TODO how to check if this outputs the right counts dict and it's not the same as the one without the wrapper ???


if __name__ == "__main__":
    unittest.main()
