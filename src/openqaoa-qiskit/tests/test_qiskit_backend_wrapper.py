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

from openqaoa.utilities import X_mixer_hamiltonian, bitstring_energy
from openqaoa.backends.qaoa_device import create_device
from openqaoa.backends.basebackend import QAOABaseBackendShotBased


def get_params():
    cost_hamil = Hamiltonian.classical_hamiltonian(
        terms=[[0, 1], [0], [1]], coeffs=[1, 1, 1], constant=0
    )
    mixer_hamil = X_mixer_hamiltonian(2)

    qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)
    variational_params_std = create_qaoa_variational_params(
        qaoa_descriptor, "standard", "ramp"
    )

    return qaoa_descriptor, variational_params_std


class TestingSPAMTwirlingWrapper(unittest.TestCase):
    """
    These tests check methods of the SPAM Twirling wrapper.
    """

    def setUp(
        self,
    ):
        self.n_batches = 5
        self.calibration_data_location = (
            "./tests/qpu_calibration_data/spam_twirling_mock.json"
        )
        self.qaoa_descriptor, self.variate_params = get_params()

        self.qiskit_shot_backend = QAOAQiskitBackendShotBasedSimulator(
            qaoa_descriptor=self.qaoa_descriptor,
            n_shots=100,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1.0,
            qiskit_simulation_method="automatic",
            seed_simulator=2642,
            noise_model=None,
        )

        self.wrapped_obj = SPAMTwirlingWrapper(
            self.qiskit_shot_backend,
            n_batches=self.n_batches,
            calibration_data_location=self.calibration_data_location,
        )

        self.wrapped_obj_trivial = SPAMTwirlingWrapper(
            self.qiskit_shot_backend,
            n_batches=1,
            calibration_data_location=self.calibration_data_location,
        )

    @pytest.mark.qpu
    def test_wrap_any_backend(self):
        """
        Testing if the wrapper is backend-agnostic by checking if it can take
        any of the relevant backend objects as an argument.
        """

        rigetti_args = {
            "as_qvm": True,
            "execution_timeout": 10,
            "compiler_timeout": 100,
        }
        device_list = []
        device_list.append(
            create_device(location="local", name="qiskit.qasm_simulator")
        )
        device_list.append(
            create_device(location="ibmq", name="ibm_kyoto"),
        )

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
            except Exception as e:
                raise ValueError("The {} backend cannot be wrapped.".format(backend), e)

    def test_setUp(self):
        assert (
            self.wrapped_obj.n_batches == self.n_batches
        ), "The number of batches hasn't been set correctly."
        assert (
            self.wrapped_obj.calibration_data_location == self.calibration_data_location
        ), "The location of the calibration file hasn't been set correctly."

    def test_get_counts(self):
        """
        Testing the get_counts method of the SPAM Twirling wrapper.

        The test compares the probability dictionary returned from the
        get_counts method as overriden by the wrapper to the hardcoded one and
        the unwrapped backend. The later is only true for the very trivial case
        of a using a single batch and seed 1 which translates to not adding
        any X gates to the circuit. The test also compares the outcome
        probabilitites obtained within the wrapper to the ones obtained
        form the original backend. These should be almost equal given that
        there's no measurement noise and enough shots.
        """
        try:
            self.wrapped_obj
            self.wrapped_obj_trivial
        except NameError:
            print("Skipping Qiskit tests as Qiskit is not installed.")
            return None

        assert self.wrapped_obj.get_counts(
            self.variate_params, n_shots=100, seed=13
        ) == {
            "11": 27,
            "10": 36,
            "01": 35,
            "00": 2,
        }, "The get_counts function in the wrapper didn't return the expected counts."

        assert self.wrapped_obj_trivial.get_counts(
            self.variate_params, n_shots=100, seed=1
        ) == self.qiskit_shot_backend.get_counts(
            self.variate_params, n_shots=100
        ), "The get_counts function in the trivial wrapper didn't return the original backend counts."

        # Obtain counts for the trivial case with n_batches=1 but seed=1 which means applying X gates to both qubits
        n_shots = 10**6  # more counts for accurate statistics
        wrapper_counts = self.wrapped_obj_trivial.get_counts(
            self.variate_params, n_shots=n_shots, seed=2
        )
        backend_counts = self.qiskit_shot_backend.get_counts(
            self.variate_params, n_shots=n_shots
        )

        # convert counts to probabilities
        wrapper_probs = {key: value / n_shots for key, value in wrapper_counts.items()}
        backend_probs = {key: value / n_shots for key, value in backend_counts.items()}

        for key in wrapper_probs:
            assert np.isclose(
                wrapper_probs[key], backend_probs[key], rtol=1e-1
            ), "The get_counts function in the trivial wrapper when negating both qubits returned a value too far from the original backend."

        # Obtain counts for the trivial case with n_batches=5 and seed=26 which means applying X gates at random at every batch
        n_shots = 10**6  # more counts for accurate statistics
        wrapper_counts = self.wrapped_obj.get_counts(
            self.variate_params, n_shots=n_shots, seed=26
        )
        backend_counts = self.qiskit_shot_backend.get_counts(
            self.variate_params, n_shots=n_shots
        )

        # convert counts to probabilities
        wrapper_probs = {key: value / n_shots for key, value in wrapper_counts.items()}
        backend_probs = {key: value / n_shots for key, value in backend_counts.items()}

        for key in wrapper_probs:
            assert np.isclose(
                wrapper_probs[key], backend_probs[key], rtol=1e-1
            ), "The get_counts function in the wrapper when negating random qubits returned a value too far from the original backend."

    def test_expectation_value_spam_twirled(self):
        """
        Testing the expectation_value_spam_twirled method of the SPAM Twirling
        wrapper in the following scenarious:
        Given very trivial counts where only the 00 state is present and
        calibration factors 1.0, meaning no errors, the expectation value
        of the energy must be the energy of the 00 bitstring.
        Given the same counts but calibration factors 0.5, the expectation
        value of the energy must be twice the energy of the 00 bitstring
        due to corrections coming from the calibration factors.

        """
        try:
            self.wrapped_obj
            self.wrapped_obj_trivial
        except NameError:
            print("Skipping Qiskit tests as Qiskit is not installed.")
            return None

        counts = {"00": 100, "01": 0, "10": 0, "11": 0}
        hamiltonian = Hamiltonian.classical_hamiltonian(
            terms=[[0, 1], [0], [1]], coeffs=[1, 1, 1], constant=0
        )
        calibration_factors = {(1,): 1.0, (0,): 1.0, (0, 1): 1.0}
        assert self.wrapped_obj.expectation_value_spam_twirled(
            counts, hamiltonian, calibration_factors
        ) == bitstring_energy(
            hamiltonian, "00"
        ), "The function which computes the expectation value when using spam twirling with trivial calibration factors doesn't give the correct energy."

        calibration_factors = {(1,): 0.5, (0,): 0.5, (0, 1): 0.5}
        assert self.wrapped_obj.expectation_value_spam_twirled(
            counts, hamiltonian, calibration_factors
        ) == 2 * bitstring_energy(
            hamiltonian, "00"
        ), "The function which computes the expectation value when using spam twirling with calibration factors = 0.5 doesn't give the correct energy."

        counts = {"00": 25, "01": 25, "10": 25, "11": 25}
        calibration_factors = {(1,): 1.0, (0,): 1.0, (0, 1): 1.0}
        assert (
            self.wrapped_obj.expectation_value_spam_twirled(
                counts, hamiltonian, calibration_factors
            )
            == 0
        ), "The function which computes the expectation value when using spam twirling when equal superposition of states doesn't give the correct energy."

        counts = {"00": 13, "01": 26, "10": 42, "11": 66}
        calibration_factors = {(1,): 0.9, (0,): 0.8, (0, 1): 0.7}
        assert (
            self.wrapped_obj.expectation_value_spam_twirled(
                counts, hamiltonian, calibration_factors
            )
            == -0.759502213584
        ), "The function which computes the expectation value when using spam twirling with random counts and factors doesn't give the correct energy."

    def test_expectation(self):
        """
        Testing the expectation method of the SPAM Twirling wrapper which overrides the backend function as defined in basebackends.
        """
        try:
            self.wrapped_obj
            self.wrapped_obj_trivial
        except NameError:
            print("Skipping Qiskit tests as Qiskit is not installed.")
            return None

        self.assertAlmostEqual(
            self.wrapped_obj.expectation(self.variate_params, n_shots=10000), -0.94, 1
        )


if __name__ == "__main__":
    unittest.main()
