import unittest

from openqaoa.backends.qaoa_backend import (
    get_qaoa_backend,
    DEVICE_NAME_TO_OBJECT_MAPPER,
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
        Check that the .get_counts admit n_shots as an argument, and works properly for the backend of all local devices.
        Also check that .expectation and .expecation_w_uncertainty methods admit n_shots as an argument for the QAOABaseBackendShotBased backends.
        """
        print(DEVICE_NAME_TO_OBJECT_MAPPER)
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
            ), "`n_shots` is not being respected for the local simulator `{}` when calling backend.get_counts(n_shots=58).".format(
                device_name
            )
            if isinstance(backend, QAOABaseBackendShotBased):
                try:
                    backend.expectation(params=variational_params_std, n_shots=58)
                except Exception:
                    raise Exception(
                        "backend.expectation does not admit `n_shots` as an argument for the local simulator `{}`.".format(
                            device_name
                        )
                    )
                try:
                    backend.expectation_w_uncertainty(
                        params=variational_params_std, n_shots=58
                    )
                except Exception:
                    raise Exception(
                        "backend.expectation_w_uncertainty does not admit `n_shots` as an argument for the local simulator `{}`.".format(
                            device_name
                        )
                    )


if __name__ == "__main__":
    unittest.main()
