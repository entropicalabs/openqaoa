import random
import numpy as np
from typing import Dict
import json
import time
import sys

from .basebackend import VQABaseBackend

from ..qaoa_components.ansatz_constructor.gates import X

from ..qaoa_components import Hamiltonian

from ..utilities import (
    exp_val_pair,
    exp_val_single,
    negate_counts_dictionary,
    calculate_calibration_factors,
    round_value
)


class BaseWrapper(VQABaseBackend):
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
    def __init__(self, backend, n_batches, calibration_data_location):
        super().__init__(backend)
        self.n_batches = n_batches
        self.calibration_data_location = calibration_data_location

        with open(self.calibration_data_location, "r") as f:
            calibration_data = json.load(f)

        if (
            self.calibration_data_location
            == "calibration_data/8bc00556-2824-4d38-b563-706a803c6401--calibration.json"
            or self.calibration_data_location
            == "calibration_data/ac469846-25e7-40be-b0d0-2d60ae29d973--calibration.json"
        ):
            # Rigetti's data:
            out = calibration_data["results"]["measurement_outcomes"]

            # convert list to strings
            calibration_measurements = {}
            for outcomes in out:
                for state, counts in outcomes.items():
                    state = "".join([str(e) for e in eval(state)])
                    if calibration_measurements.get(state) is None:
                        calibration_measurements.update({state: counts})
                    else:
                        calibration_measurements[state] += counts
        else:
            calibration_measurements = calibration_data["results"][
                "measurement_outcomes"
            ]

        calibration_registers = calibration_data["register"]

        qubit_mapping = self.backend.initial_qubit_mapping

        self.calibration_factors = calculate_calibration_factors(
            self.backend.qaoa_descriptor.cost_hamiltonian,
            calibration_measurements,
            calibration_registers,
            qubit_mapping,
        )

        """
        This doesn't work because keys are tuples  =(
        # save calibration factors just in case
        with open('test.json', "w") as fp:
            json.dump(self.calibration_factors, fp)
        """

    def get_counts(self, params, n_shots=None, seed=42):
        """
        Overrides the get_counts function of the backend object.
        Modified function to...
            divide into batches
            change the self.append_state according to the schedule, s.
            get the counts and classically negate them
            combine all batches into a count dict under BFA
        """
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
            
            print(negated_qubits)

            circuit_to_append = self.backend.gate_applicator.create_quantum_circuit(
                self.backend.n_qubits
            )

            for negated_qubit in negated_qubits:
                negated_qubit = (
                    negated_qubit.item()
                )
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

        return counts

    @round_value
    def expectation_value_spam_twirled(
        self, counts: Dict, hamiltonian: Hamiltonian, calibration_factors: dict
    ):
        """
        TODO
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

    def expectation(self, params, n_shots=None) -> float:
        """
        use the total counts under BFA to compute expectation values Zi and ZiZj
        correct these expectation values with the calibration factors, lambda_i, lambda_ij
        combine all corrected expectation values into the energy = cost fn to be given to the optimizer every time it calls expectation
        """
        counts = self.get_counts(params, n_shots)
        cost = self.expectation_value_spam_twirled(
            counts,
            self.backend.qaoa_descriptor.cost_hamiltonian,
            self.calibration_factors,
        )

        return cost
