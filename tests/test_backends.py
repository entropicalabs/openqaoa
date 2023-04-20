import unittest
import json
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


class TestingBackendLocal(unittest.TestCase):
    """
    These tests check that the methods of the local backends are working properly.
    """

    def test_get_counts_and_expectation_n_shots(self):
        """
        Check that the .get_counts admit n_shots as an argument, and works properly for
          the backend of all local devices.
        Also check that .expectation and .expecation_w_uncertainty methods admit n_shots
         as an argument for the QAOABaseBackendShotBased backends.
        """

        for device_name in DEVICE_NAME_TO_OBJECT_MAPPER.keys():
            # Analytical device doesn't have any of those so we are skipping it in the tests.
            if device_name in ["analytical_simulator"]:
                continue

            qaoa_descriptor, variational_params_std = get_params()

            device = create_device(location="local", name=device_name)
            backend = get_qaoa_backend(
                qaoa_descriptor=qaoa_descriptor, device=device, n_shots=1000
            )

            assert (
                sum(
                    backend.get_counts(
                        params=variational_params_std, n_shots=58
                    ).values()
                )
                == 58
            ), "`n_shots` is not being respected for the local simulator `{}` when \
                calling backend.get_counts(n_shots=58).".format(
                device_name
            )
            if isinstance(backend, QAOABaseBackendShotBased):
                try:
                    backend.expectation(params=variational_params_std, n_shots=58)
                except Exception:
                    raise Exception(
                        "backend.expectation does not admit `n_shots` as an argument \
                            for the local simulator `{}`.".format(
                            device_name
                        )
                    )
                try:
                    backend.expectation_w_uncertainty(
                        params=variational_params_std, n_shots=58
                    )
                except Exception:
                    raise Exception(
                        "backend.expectation_w_uncertainty does not admit `n_shots` \
                            as an argument for the local simulator `{}`.".format(
                            device_name
                        )
                    )


class TestingBackendQPUs(unittest.TestCase):
    """
    These tests check methods of the QPU backends.

    For all of these tests, credentials.json MUST be filled with the appropriate
    credentials.
    """

    @pytest.mark.qpu
    def setUp(self):
        self.HUB = "ibm-q"
        self.GROUP = "open"
        self.PROJECT = "main"

        bashCommand = "az resource list"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error is not None:
            print(error)
            raise Exception(
                "You must have the Azure CLI installed and must be logged in to use the Azure Quantum Backends"
            )
        else:
            output_json = json.loads(output)
            output_json_s = [
                each_json
                for each_json in output_json
                if each_json["name"] == "TestingOpenQAOA"
            ][0]
            self.RESOURCE_ID = output_json_s["id"]
            self.AZ_LOCATION = output_json_s["location"]

    @pytest.mark.qpu
    def test_get_counts_and_expectation_n_shots(self):
        """
        TODO: test needs to be updated as DEVICE_ACCESS_OBJECT_MAPPER is now dynamically filled
        based on whether a module exists.

        Check that the .get_counts, .expectation and .expecation_w_uncertainty methods
        admit n_shots as an argument for the backends of all QPUs.
        """

        list_device_attributes = [
            {
                "QPU": "Azure",
                "device_name": "rigetti.sim.qvm",
                "resource_id": self.RESOURCE_ID,
                "az_location": self.AZ_LOCATION,
            },
            {
                "QPU": "AWS",
                "device_name": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            },
            {
                "QPU": "Pyquil",
                "device_name": "2q-qvm",
                "as_qvm": True,
                "execution_timeout": 3,
                "compiler_timeout": 3,
            },
            {
                "QPU": "Qiskit",
                "device_name": "ibmq_qasm_simulator",
                "hub": self.HUB,
                "group": self.GROUP,
                "project": self.PROJECT,
            },
        ]

        assert len(list_device_attributes) == len(
            DEVICE_ACCESS_OBJECT_MAPPER
        ), "The number of QPUs in the list of tests is not the same as the number of QPUs in \
             the DEVICE_ACCESS_OBJECT_MAPPER. The list should be updated."
        print(DEVICE_ACCESS_OBJECT_MAPPER.items(), list_device_attributes)
        for (device, backend), device_attributes in zip(
            DEVICE_ACCESS_OBJECT_MAPPER.items(), list_device_attributes
        ):
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

                # Check that the .get_counts, .expectation and .expectation_w_variance methods
                # admit n_shots as an argument
                assert (
                    sum(
                        backend.get_counts(
                            params=variational_params_std, n_shots=58
                        ).values()
                    )
                    == 58
                ), "`n_shots` is not being respected \
                    when calling .get_counts(n_shots=58) for QPU `{}`.".format(
                    QPU_name
                )
                backend.expectation(params=variational_params_std, n_shots=58)
                backend.expectation_w_uncertainty(
                    params=variational_params_std, n_shots=58
                )

            except Exception as e:
                raise e from type(e)(f"Error raised for `{QPU_name}`: " + str(e))

            print("Test passed for {} backend.".format(QPU_name))


if __name__ == "__main__":
    unittest.main()
