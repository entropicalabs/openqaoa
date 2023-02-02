from __future__ import annotations
from abc import ABC
from typing import List, Union, Tuple, Any, Callable, Iterable
import numpy as np

from .operators import Hamiltonian
from .hamiltonianmapper import HamiltonianMapper


def _is_iterable_empty(in_iterable):
    if isinstance(in_iterable, Iterable):  # Is Iterable
        return all(map(_is_iterable_empty, in_iterable))
    return False  # Not an Iterable


class shapedArray:
    """Decorator-Descriptor for arrays that have a fixed shape.

    This is used to facilitate automatic input checking for all the different
    internal parameters. Each instance of this class can be removed without
    replacement and the code should still work, provided the user provides
    only correct angles to below parameter classes

    Parameters
    ----------
    shape: `Callable[[Any], Tuple]`
        Returns the shape for self.values

    Example
    -------
    With this descriptor, the two following are equivalent:

    .. code-block:: python

        class foo():
            def __init__(self):
                self.shape = (n, m)
                self._my_attribute = None

            @property
            def my_attribute(self):
                return _my_attribute

            @my_attribute.setter
            def my_attribute(self):
                try:
                    self._my_attribute = np.reshape(values, self.shape)
                except ValueError:
                    raise ValueError("my_attribute must have shape "
                                    f"{self.shape}")


    can be simplified to

    .. code-block:: python

        class foo():
            def __init__(self):
                self.shape = (n, m)

            @shapedArray
            def my_attribute(self):
                return self.shape
    """

    def __init__(self, shape: Callable[[Any], Tuple]):
        """The constructor. See class documentation"""
        self.name = shape.__name__
        self.shape = shape

    def __set__(self, obj, values):
        """The setter with input checking."""
        try:
            # also round to 12 decimal places to avoid floating point errors
            setattr(
                obj, f"__{self.name}", np.round(np.reshape(values, self.shape(obj)), 12)
            )
        except ValueError:
            raise ValueError(f"{self.name} must have shape {self.shape(obj)}")

    def __get__(self, obj, objtype):
        """The getter."""
        return getattr(obj, f"__{self.name}")


class AnsatzDescriptor(ABC):
    """
    Parameters class to construct a specific quantum ansatz to attack
    a problem

    Parameters
    ----------
    algorithm: `str`

    Attributes
    ----------
    algorithm: `str`
    """

    def __init__(self, algorithm: str):
        self.algorithm = algorithm


class QAOADescriptor(AnsatzDescriptor):

    """
    Create the problem attributes consisting of the Hamiltonian, QAOA 'p'
    value and other specific parameters.

    Parameters
    ----------
    cost_hamiltonian: `Hamiltonian`
        The cost hamiltonian of the problem the user is trying to solve.

    mixer_block: `Union[List[RotationGateMap], Hamiltonian]`
        The mixer hamiltonian or a list of initialised RotationGateMap objects
        that defines the gates to be used within the "mixer part" of the circuit.

    p: `int`
        Number of QAOA layers; defaults to 1 if not specified

    mixer_coeffs: `List[float]`
        A list containing coefficients for each mixer GateMap. The order of the
        coefficients should follow the order of the GateMaps provided in the relevant gate block.
        This input isnt required if the input mixer block is of type Hamiltonian.

    Attributes
    ----------
    cost_hamiltonian: `Hamiltonian`

    qureg: `List[int]`

    cost_block_coeffs: `List[float]`

    cost_single_qubit_coeffs: `List[float]`

    cost_qubits_singles: `List[str]`

    cost_pair_qubit_coeffs: `List[float]`

    cost_qubits_pairs: `List[str]`

    mixer_block_coeffs: `List[float]`

    Properties
    ----------
    cost_block: `List[RotationGateMap]`

    mixer_block: `List[RotationGateMap]`

    abstract_circuit: `List[RotationGateMap]`
    """

    def __init__(
        self,
        cost_hamiltonian: Hamiltonian,
        mixer_block: Union[List[RotationGateMap], Hamiltonian],
        p: int,
        mixer_coeffs: List[float] = [],
    ):

        super().__init__(algorithm="QAOA")

        self.p = p

        self.qureg = cost_hamiltonian.qureg
        self.cost_block_coeffs = cost_hamiltonian.coeffs

        try:
            self.mixer_block_coeffs = mixer_block.coeffs
        except AttributeError:
            self.mixer_block_coeffs = mixer_coeffs

        # Needed in the BaseBackend to compute exact_solution, cost_funtion method
        # and bitstring_energy
        self.cost_hamiltonian = cost_hamiltonian
        self.cost_block = cost_hamiltonian
        self.mixer_block = mixer_block

        (
            self.cost_single_qubit_coeffs,
            self.cost_pair_qubit_coeffs,
            self.cost_qubits_singles,
            self.cost_qubits_pairs,
        ) = self._assign_coefficients(self.cost_block[0], self.cost_block_coeffs)

        (
            self.mixer_single_qubit_coeffs,
            self.mixer_pair_qubit_coeffs,
            self.mixer_qubits_singles,
            self.mixer_qubits_pairs,
        ) = self._assign_coefficients(self.mixer_block[0], self.mixer_block_coeffs)

    def _assign_coefficients(
        self, input_block: List[RotationGateMap], input_coeffs: List[float]
    ) -> None:

        """
        Splits the coefficients and gatemaps into qubit singles and qubit pairs.
        """

        single_qubit_coeffs = []
        pair_qubit_coeffs = []
        qubit_singles = []
        qubit_pairs = []

        if len(input_block) != len(input_coeffs):

            raise ValueError(
                "The number of terms/gatemaps must match the number of coefficients provided."
            )

        for each_gatemap, each_coeff in zip(input_block, input_coeffs):

            if each_gatemap.pauli_label[0] == "1q":
                single_qubit_coeffs.append(each_coeff)
                # Giving a string name to each gatemap (?)
                qubit_singles.append(type(each_gatemap).__name__)
            elif each_gatemap.pauli_label[0] == "2q":
                pair_qubit_coeffs.append(each_coeff)
                qubit_pairs.append(type(each_gatemap).__name__)

        return (single_qubit_coeffs, pair_qubit_coeffs, qubit_singles, qubit_pairs)

    @property
    def cost_block(self):

        return self._cost_block

    @cost_block.setter
    def cost_block(self, input_object: Hamiltonian) -> None:

        self._cost_block = HamiltonianMapper.repeat_gate_maps(
            input_object, "cost", self.p
        )

    @property
    def mixer_block(self):

        return self._mixer_block

    @mixer_block.setter
    def mixer_block(
        self, input_object: Union[List[RotationGateMap], Hamiltonian]
    ) -> None:

        if type(input_object) == Hamiltonian:

            self._mixer_block = HamiltonianMapper.repeat_gate_maps(
                input_object, "mixer", self.p
            )

        else:

            self._mixer_block = HamiltonianMapper.repeat_gate_maps_from_gate_map_list(
                input_object, "mixer", self.p
            )

    def __repr__(self):

        """Return an overview over the parameters and hyperparameters

        Todo
        ----
        Split this into ``__repr__`` and ``__str__`` with a more verbose
        output in ``__repr__``.
        """

        string = "Circuit Parameters:\n"
        string += "\tp: " + str(self.p) + "\n"
        string += "\tregister: " + str(self.qureg) + "\n" + "\n"

        string += "Cost Hamiltonian:\n"
        string += "\tcost_qubits_singles: " + str(self.cost_qubits_singles) + "\n"
        string += (
            "\tcost_single_qubit_coeffs: " + str(self.cost_single_qubit_coeffs) + "\n"
        )
        string += "\tcost_qubits_pairs: " + str(self.cost_qubits_pairs) + "\n"
        string += (
            "\tcost_pair_qubit_coeffs: "
            + str(self.cost_pair_qubit_coeffs)
            + "\n"
            + "\n"
        )

        string += "Mixer Hamiltonian:\n"
        string += "\tmixer_qubits_singles: " + str(self.mixer_qubits_singles) + "\n"
        string += (
            "\tmixer_single_qubit_coeffs: " + str(self.mixer_single_qubit_coeffs) + "\n"
        )
        string += "\tmixer_qubits_pairs: " + str(self.mixer_qubits_pairs) + "\n"
        string += (
            "\tmixer_pair_qubit_coeffs: " + str(self.mixer_pair_qubit_coeffs) + "\n"
        )

        return string

    @property
    def abstract_circuit(self):

        _abstract_circuit = []
        for each_p in range(self.p):
            _abstract_circuit.extend(self.cost_block[each_p])
            _abstract_circuit.extend(self.mixer_block[each_p])

        return _abstract_circuit
