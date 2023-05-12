"""
Energy expectation as a function of angles computed accordingly to the analytical expression for p=1.
"""
from .basebackend import QAOABaseBackend
from ..qaoa_components import (
    QAOADescriptor,
    QAOAVariationalStandardParams,
)
from ..qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)
from ..utilities import energy_expectation_analytical, generate_uuid, round_value


class QAOABackendAnalyticalSimulator(QAOABaseBackend):
    r"""
    A simulator class, specific for QAOA with a single layer, p=1, starting with
    a layer of Hadamards and using the X mixer. Works by calculating the expectation value
    of the given quantum circuit (specificied with beta and gamma angles) from the
    analytical formula derived in arXiv:2011.13420v2.

    Parameters
    ----------
    qaoa_descriptor: QAOADescriptor
        An object of the class ``QAOADescriptor`` which contains information on
        circuit construction and depth of the circuit.
        Note that it only works for p=1 and the X Mixer.
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        prepend_state=None,
        append_state=None,
        init_hadamard=True,
        cvar_alpha=1,
    ):
        # checking if not supported parameters are passed
        for k, val in {
            "Prepend_state": (prepend_state, None),
            "append_state": (append_state, None),
            "init_hadamard": (init_hadamard, True),
            "cvar_alpha": (cvar_alpha, 1),
        }.items():
            if val[0] != val[1]:
                print(
                    f"{k} is not supported for the analytical backend. {k} is set to None."
                )

        QAOABaseBackend.__init__(
            self,
            qaoa_descriptor,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1,
        )

        self.measurement_outcomes = (
            {}
        )  # passing empty dict for the logger since measurements are irrelevant for this backend.

        # check if conditions for the analytical formula are met
        assert self.qaoa_descriptor.p == 1, "Analytical formula only holds for p=1."

        for gatemap in self.qaoa_descriptor.mixer_qubits_singles:
            assert gatemap == "RXGateMap", "Analytical formula only holds for X mixer."

        assert (
            self.qaoa_descriptor.mixer_qubits_pairs == []
        ), "Analytical formula only holds for X mixer."

    def assign_angles(self):
        raise NotImplementedError("This method is irrelevant for this backend")

    def obtain_angles_for_pauli_list(self):
        raise NotImplementedError("This method is irrelevant for this backend")

    def qaoa_circuit(self):
        raise NotImplementedError("This method is irrelevant for this backend")

    def get_counts(self):
        raise NotImplementedError("This method is irrelevant for this backend")

    @round_value
    def expectation(self, params: QAOAVariationalBaseParams) -> float:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian analytically.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing
            variable parameters.

        Returns
        -------
        float:
            Expectation value of cost operator wrt to the QAOA parameters
            according to the analytical expression for p=1.
        """
        # generate a job id
        self.job_id = generate_uuid()

        assert isinstance(
            params, QAOAVariationalStandardParams
        ), "Analytical formula only holds for standard parametrization (for now)."

        betas = params.betas
        gammas = params.gammas

        cost = energy_expectation_analytical([betas, gammas], self.cost_hamiltonian)
        return cost

    def expectation_w_uncertainty(self, params):
        raise NotImplementedError("Not implemented yet. In progress.")

    def reset_circuit(self):
        raise NotImplementedError("This method is irrelevant for this backend")

    def circuit_to_qasm(self):
        raise NotImplementedError("This method is irrelevant for this backend")
