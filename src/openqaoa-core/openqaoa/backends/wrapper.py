import random
import numpy as np
import json
from mitiq import QPROGRAM, Executor, Observable, QuantumResult
from mitiq.zne.inference import RichardsonFactory, LinearFactory, PolyExpFactory, PolyFactory, AdaExpFactory, FakeNodesFactory, ExpFactory
from mitiq.zne.scaling import fold_gates_at_random, fold_gates_from_left, fold_gates_from_right
from mitiq.zne import execute_with_zne
from cirq import DensityMatrixSimulator, depolarize
from mitiq.interface import convert_to_mitiq, mitiq_cirq

from qiskit import QuantumCircuit, transpile #,qasm_simulator, shot_simulator, statevector_simulator
from qiskit.quantum_info import DensityMatrix
from .qaoa_backend import DEVICE_NAME_TO_OBJECT_MAPPER

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

from .cost_function import cost_function
from openqaoa_qiskit.backends.qaoa_qiskit_qpu import QAOAQiskitQPUBackend


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


#------------------------ZNE WRAPPER---------------------------------
available_factories = [
    "Richardson",
    "Linear",
    "Poly",
    "Exp",
    "PolyExp",
    "AdaExp",
    "FakeNodes"
]
available_scaling = [
    "fold_gates_at_random",
    "fold_gates_from_right",
    "fold_gates_from_left"
]
class ZNEWrapper(BaseWrapper):
    """
    This class inherits from the BaseWrapper and need to be backend agnostic
    to QAOA implementations on different devices and their respectives SDKs. It 
    implements Zero Noise Extrapolation (ZNE) from Mitiq framework. ZNE is an 
    error mitigation technique used to extrapolate the noiseless expectation
    value of an observable from a range of expectation values computed at 
    different noise levels.

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


        # only qiskit backends are supported
        assert( type(backend) == DEVICE_NAME_TO_OBJECT_MAPPER['qiskit.qasm_simulator'] or 
                type(backend) == DEVICE_NAME_TO_OBJECT_MAPPER['qiskit.shot_simulator'] or
                type(backend) == DEVICE_NAME_TO_OBJECT_MAPPER['qiskit.statevector_simulator']
                or type(backend) == QAOAQiskitQPUBackend), "Only Qiskit backends are supported."


        with open(self.calibration_data_location, "r") as f:
            calibration_data = json.load(f)

        factory = calibration_data["factory"]
        scaling = calibration_data["scaling"]
        scale_factor = calibration_data["scale_factor"]

        # preconditions over calibration data
        assert(factory in available_factories), "Supported factories are: Poly, Richardson, Exp, FakeNodes, Linear, PolyExp, AdaExp"
        assert(scaling in available_scaling), "Supported scaling methods are: fold_gates_at_random, fold_gates_from_right, fold_gates_from_left"
        assert(
            type(scale_factor) == list and
            (type(x) == float and x >=1) for x in scale_factor
            ), "Scale factor must be a list of floats greater or equal to 1"
            
            try:
                seed = calibration_data["seed"]
                assert(type(seed) == int), "Seed must be an integer"
                assert(seed >= 0), "Seed must be >= 0"
            except:
                seed = self.seed

            self.factory_obj = None
            if factory == "Richardson":
                self.factory_obj = RichardsonFactory(scale_factors = scale_factor, seed = seed)
            elif factory == "Linear":
                self.factory_obj = LinearFactory(scale_factors = scale_factor, seed = seed)
            elif factory == "Exp":
                self.factory_obj = ExpFactory(scale_factors = scale_factor, seed = seed)
            elif factory == "Poly":
                self.factory_obj = PolyFactory(scale_factors = scale_factor, order = calibration_data["order"], seed = seed)
            elif factory == "PolyExp":
                self.factory_obj = PolyExpFactory(scale_factors = scale_factor, order = calibration_data["order"], seed = seed)
            elif factory == "AdaExp":
                self.factory_obj = AdaExpFactory(cale_factors = scale_factor, steps = calibration_data["steps"], seed = seed)
            elif factory == "FakeNodes":
                self.factory_obj = FakeNodesFactory(scale_factors = scale_factor, seed = seed)

        # setting the scaling
        self.scale_noise = None
        if scaling == "fold_gates_at_random":
            self.scale_noise = fold_gates_at_random
        elif scaling == "fold_gates_from_left":
            self.scale_noise = fold_gates_from_left
        elif scaling == "fold_gates_from_right":
            self.scale_noise = fold_gates_from_right

        #setting the scale_factor  
        self.scale_factor = scale_factor
        

    def expectation(self, params: QAOAVariationalBaseParams, n_shots=None) -> float:
        """
        This method overrrides the one from the basebackend to allow for
        correcting the expectation values of each term in the Hamiltonian
        before providing the energy to the optimized. It does this by using
        execute_with_zne() method from Mitiq. This method estimates the
        error-mitigated expectation value associated with a circuit, via
        the application of ZNE.

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

        # executor used by Mitiq
        def executor(qc: QuantumCircuit) -> float:                
                # calculate the counts
                counts = self.backend.get_counts(params, n_shots)
                self.measurement_outcomes = counts
                
                # calculate and return the cost
                cost = cost_function(
                    counts,
                    self.backend.qaoa_descriptor.cost_hamiltonian)
                return cost

        # this quantum circuit is NOT used.
        # this quantum circuit is just created because of the parameter needs of execute_with_zne()
        # the quantum circuit to be used is inside the executor's get_counts()
        qc = self.backend.qaoa_circuit(params)
        qc = transpile(qc, basis_gates=["u1", "u2", "u3", "cx"])

        return execute_with_zne(
            circuit = qc,
            executor = executor,
            observable = None,
            factory = self.factory_obj,
            scale_noise = self.scale_noise)


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
