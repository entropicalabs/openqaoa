import random
import numpy as np
import json

from .basebackend import VQABaseBackend

from ..qaoa_components.ansatz_constructor.gates import X

from ..qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)

from ..qaoa_components import Hamiltonian

from ..utilities import (
    exp_val_pair,
    exp_val_single,
    negate_counts_dictionary,
    calculate_calibration_factors,
    round_value,
)


class BaseWrapper(VQABaseBackend):
    """
    This is the Abstract Base Class over which other classes will be built.
    It is designed to take the backend object and wrap around it in order
    to override the necessary methods.

    Parameters
    ----------
    backend: `VQABaseBackend`
        The backend object to be wrapped.
    """

    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def expectation(self, *args, **kwargs):
        return self.backend.expectation(*args, **kwargs)

    def expectation_w_uncertainty(self, *args, **kwargs):
        return self.backend.expectation_w_uncertainty(*args, **kwargs)

    def exact_solution(self, *args, **kwargs):
        return self.backend.exact_solution(*args, **kwargs)


class SPAMTwirlingWrapper(BaseWrapper):
    """
    This class inherits from the BaseWrapper and needs to be backend
    agnostic to QAOA implementations on different devices and their
    respective SDKs. It implements State Preparation and Measurement (SPAM)
    Twirling which is a simple error mitigation technique used to remove any
    preferred direction due to readout noise.

    Parameters
    ----------
    n_batches: `int`
        Number of batches.
    calibration_data_location: `str`
        The location of the calibration data file.
    """

    def __init__(self, backend, n_batches, calibration_data_location):
        super().__init__(backend)
        self.n_batches = n_batches
        self.calibration_data_location = calibration_data_location

        with open(self.calibration_data_location, "r") as f:
            calibration_data = json.load(f)

        calibration_measurements = calibration_data["results"]["measurement_outcomes"]
        calibration_registers = calibration_data["register"]

        assert (
            len(calibration_registers) >= self.backend.n_qubits
        ), "Problem requires more qubits than present in the calibration data."
        "Make sure that the calibration data file is for the correct device."

        qubit_mapping = self.backend.initial_qubit_mapping

        self.calibration_factors = calculate_calibration_factors(
            self.backend.qaoa_descriptor.cost_hamiltonian,
            calibration_measurements,
            calibration_registers,
            qubit_mapping,
        )

    def get_counts(
        self, params: QAOAVariationalBaseParams, n_shots=None, seed=None
    ) -> dict:
        """
        This method overrides the get_counts method of the backend object
        to obtain the measurement outcomes under the bit flip averaging
        technique, which is applying an X gate just before the measurement
        and negating the classical outcomes for a set of qubits selected at
        random. Every such set we call a batch, and the total number of shots
        is distributed amongst the number of batches. The procedure is
        as follows: per batch, generate a random integer, obtain its
        binary representation and use the positions of 1s in the bitstring to
        mark the qubits to which an X gate will be applied. Then, create an
        append state which is a set of all these X gates on the chosen qubits.
        Run the modified circuit and obtain the measurement outcomes using the
        original backend get_counts method. Negate the counts classically for
        the selected qubits. Lastly, combine the counts from all batches.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes,
            containing variable parameters.
        n_shots: `int`
            The number of shots to be used for the measurement. If None,
            the backend default.
        seed: `int`
            The seed controlling the random number generator, which selects
            the set of qubits to be negated at every batch. Default is None.

        Returns
        -------
        dict:
            The dictionary containing measurement outcomes (counts) of a given
            QAOA circit under the bit-flip-averaging (BFA) technique.
        """
        if seed is not None:
            random.seed(seed)
        # list of integers whose binary representation signifies which qubits to be flipped at every batch
        s_list = []
        for _ in range(0, self.n_batches):
            s_list.append(random.getrandbits(self.backend.n_qubits))

        n_shots = self.backend.n_shots if n_shots == None else n_shots
        n_shots_batch = int(n_shots / self.n_batches)

        counts = {}

        for batch in range(0, self.n_batches):
            s = s_list[batch]
            s_binary = format(s, "b").zfill(self.backend.n_qubits)  # convert to binary
            arr = np.fromiter(s_binary, dtype=int)
            negated_qubits = np.where(arr == 1)[0]  # where the syndrome has a 1

            circuit_to_append = self.backend.gate_applicator.create_quantum_circuit(
                self.backend.n_qubits
            )

            for negated_qubit in negated_qubits:
                negated_qubit = negated_qubit.item()
                negation_gate = X(self.backend.gate_applicator, negated_qubit)
                circuit_to_append = self.backend.gate_applicator.apply_gate(
                    negation_gate, negated_qubit, circuit_to_append
                )

            self.backend.append_state = circuit_to_append

            counts_batch = self.backend.get_counts(
                params, n_shots_batch
            )  # calls the get_counts method of the specific backend

            negated_counts = negate_counts_dictionary(
                counts_dictionary=counts_batch, s=s
            )

            # Add to the final counts dict
            for key in negated_counts:
                if key in counts:
                    counts[key] = counts[key] + negated_counts[key]
                else:
                    counts.update([(key, negated_counts[key])])

        self.measurement_outcomes = counts
        self.backend.append_state = []  # reset

        return counts

    @round_value
    def expectation_value_spam_twirled(
        self, counts: dict, hamiltonian: Hamiltonian, calibration_factors: dict
    ) -> float:
        """
        This method computes the expectation value of a Hamiltonian termwise
        while correcting for measurement errors by dividing by the
        hardware-specific calibration factors.

        Parameters
        ----------
        counts: `dict`
            A dictionary of the measurement outcomes of a given QAOA circuit.
        hamiltonian: `Hamiltonian`
            Hamiltonian object containing the problem statement.
        calibration_factors: `dict`
            Dictionary containing the relevant calibration factors computed
            given calibration data for the specific hardware.

        Returns
        -------
        float:
            The error mitigated expectation value of cost operator wrt a
            measurement outcomes dictionary (counts).
        """

        terms = [term.qubit_indices for term in hamiltonian.terms]
        hamiltonian_as_dict = hamiltonian.hamiltonian_dict()

        energy = 0

        # Compute expectation values and correlations of terms present in the Hamiltonian
        for term in hamiltonian.terms:
            # If bias term compute expectation value
            if len(term) == 1:
                i = term.qubit_indices[0]
                exp_vals_z = exp_val_single(i, counts)
                exp_vals_z /= calibration_factors[(i,)]

                energy += exp_vals_z * hamiltonian_as_dict[(i,)]

            # If two-body term compute correlation
            elif len(term) == 2:
                i, j = term.qubit_indices
                exp_vals_zz = exp_val_pair((i, j), counts)
                exp_vals_zz /= calibration_factors[(i, j)]

                energy += exp_vals_zz * hamiltonian_as_dict[(i, j)]

            # If constant term, ignore
            if len(term) == 0:
                continue

        energy += hamiltonian.constant

        return energy

    def expectation(self, params: QAOAVariationalBaseParams, n_shots=None) -> float:
        """
        This method overrrides the one from the basebackend to allow for
        correcting the expectation values of each term in the Hamiltonian
        before providing the energy to the optimizer. It does this by first
        obtaining the measurement outcomes through the modified get_counts
        method and then calculating the cost with the wrapper's own
        expectation_value_spam_twirled method.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes,
            containing variable parameters.
        n_shots: `int`
            The number of shots to be used for the measurement. If None,
            the backend default.

        Returns
        -------
        float:
            The error mitigated expectation value of cost operator wrt to
            quantum state produced by QAOA circuit
        """
        counts = self.get_counts(params, n_shots)
        cost = self.expectation_value_spam_twirled(
            counts,
            self.backend.qaoa_descriptor.cost_hamiltonian,
            self.calibration_factors,
        )

        return cost
