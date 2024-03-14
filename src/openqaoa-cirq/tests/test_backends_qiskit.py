import unittest
import pytest

from openqaoa.backends.qaoa_backend import DEVICE_ACCESS_OBJECT_MAPPER
from openqaoa.qaoa_components import (
    Hamiltonian,
    create_qaoa_variational_params,
    QAOADescriptor,
)
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa_qiskit.backends import DeviceQiskit


def get_params():
    cost_hamil = Hamiltonian.classical_hamiltonian([[0, 1]], [1], constant=0)
    mixer_hamil = X_mixer_hamiltonian(2)

    qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)
    variational_params_std = create_qaoa_variational_params(
        qaoa_descriptor, "standard", "ramp"
    )

    return qaoa_descriptor, variational_params_std


class TestingBackendQPUs(unittest.TestCase):
    """
    These tests check methods of the QPU backends.
    """

    @pytest.mark.qpu
    def setUp(self):
        self.HUB = "ibm-q"
        self.GROUP = "open"
        self.PROJECT = "main"

    @pytest.mark.qpu
    def test_get_counts_and_expectation_n_shots(self):
        """
        TODO: test needs to be updated as DEVICE_ACCESS_OBJECT_MAPPER is now dynamically filled based on whether a module exists.

        Check that the .get_counts, .expectation and .expecation_w_uncertainty methods admit n_shots as an argument for the backends of all QPUs.
        """

        list_device_attributes = [
            {
                "QPU": "Qiskit",
                "device_name": "ibmq_qasm_simulator",
                "hub": self.HUB,
                "group": self.GROUP,
                "project": self.PROJECT,
            }
        ]

        assert DeviceQiskit in DEVICE_ACCESS_OBJECT_MAPPER.keys()

        device = DeviceQiskit
        backend = DEVICE_ACCESS_OBJECT_MAPPER[DeviceQiskit]
        device_attributes = list_device_attributes[0]

        qaoa_descriptor, variational_params_std = get_params()

        QPU_name = device_attributes.pop("QPU")
        print("Testing {} backend.".format(QPU_name))

        # if QPU_name in ["AWS", 'Qiskit']:
        #     print(f"Skipping test for {QPU_name} backend.")
        #     continue

        try:
            print(device, device_attributes)
            device = device(**device_attributes)
            backend = backend(
                qaoa_descriptor=qaoa_descriptor,
                device=device,
                cvar_alpha=1,
                n_shots=100,
                prepend_state=None,
                append_state=None,
                init_hadamard=True,
            )

            # Check that the .get_counts, .expectation and .expectation_w_variance methods admit n_shots as an argument
            assert (
                sum(
                    backend.get_counts(
                        params=variational_params_std, n_shots=58
                    ).values()
                )
                == 58
            ), "`n_shots` is not being respected when calling .get_counts(n_shots=58).".format(
                QPU_name
            )
            backend.expectation(params=variational_params_std, n_shots=58)
            backend.expectation_w_uncertainty(params=variational_params_std, n_shots=58)

        except Exception as e:
            raise e from type(e)(f"Error raised for `{QPU_name}`: " + str(e))

        print("Test passed for {} backend.".format(QPU_name))


if __name__ == "__main__":
    unittest.main()
