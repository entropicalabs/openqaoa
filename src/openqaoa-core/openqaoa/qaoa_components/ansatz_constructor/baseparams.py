from __future__ import annotations
from abc import ABC, abstractproperty
from typing import List, Union, Tuple, Any, Callable, Iterable, Optional
import numpy as np
from enum import Enum
import copy

from .operators import Hamiltonian
from .hamiltonianmapper import HamiltonianMapper
from .gatemap import RotationGateMap, SWAPGateMap
from .gatemaplabel import GateMapType


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
        The algorithm corresponding to the ansatz
    Attributes
    ----------
    algorithm: `str`
    """

    def __init__(self, algorithm: str):
        self.algorithm = algorithm

    @abstractproperty
    def n_qubits(self) -> int:
        pass


class QAOADescriptor(AnsatzDescriptor):

    """
    Create the problem attributes consisting of the Hamiltonian, QAOA 'p'
    value and other specific parameters.

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

    cost_blocks: `List[RotationGateMap]`

    mixer_blocks: `List[RotationGateMap]`

    Properties
    ----------
    n_qubits: `int`

    abstract_circuit: `List[RotationGateMap]`
    """

    def __init__(
        self,
        cost_hamiltonian: Hamiltonian,
        mixer_block: Union[List[RotationGateMap], Hamiltonian],
        p: int,
        mixer_coeffs: List[float] = [],
        routing_function: Optional[Callable] = None,
        device: Optional["DeviceBase"] = None,
    ):
        """
        Parameters
        ----------
        cost_hamiltonian: `Hamiltonian`
            The cost hamiltonian of the problem the user is trying to solve.
        mixer_block: Union[List[RotationGateMap], Hamiltonian]
            The mixer hamiltonian or a list of initialised RotationGateMap objects
            that defines the gates to be used within the "mixer part" of the circuit.
        p: `int`
            Number of QAOA layers; defaults to 1 if not specified
        mixer_coeffs: `List[float]`
            A list containing coefficients for each mixer GateMap. The order of the
            coefficients should follow the order of the GateMaps provided in the relevant gate block.
            This input isnt required if the input mixer block is of type Hamiltonian.
        routing_function Optional[Callable]
            A callable function running the routing algorithm on the problem
        device: DeviceBase
            The device on which to run the Quantum Circuit
        """

        super().__init__(algorithm="QAOA")

        self.p = p
        self.cost_block_coeffs = cost_hamiltonian.coeffs

        try:
            self.mixer_block_coeffs = mixer_block.coeffs
        except AttributeError:
            self.mixer_block_coeffs = mixer_coeffs

        # Needed in the BaseBackend to compute exact_solution, cost_funtion method
        # and bitstring_energy
        self.cost_hamiltonian = cost_hamiltonian
        self.cost_block = self.block_setter(cost_hamiltonian, GateMapType.COST)
        (
            self.cost_single_qubit_coeffs,
            self.cost_pair_qubit_coeffs,
            self.cost_qubits_singles,
            self.cost_qubits_pairs,
        ) = self._assign_coefficients(self.cost_block, self.cost_block_coeffs)

        # route the cost block and append SWAP gates
        if isinstance(routing_function, Callable):
            try:
                (
                    self.cost_block,
                    self.initial_mapping,
                    self.final_mapping,
                ) = self.route_gates_list(self.cost_block, device, routing_function)
                self.routed = True
            except TypeError:
                raise TypeError(
                    "The specified function can has a set signature that accepts"
                    " device, problem, and initial_mapping"
                )
            except Exception as e:
                raise e
        elif routing_function == None:
            self.routed = False
        else:
            raise ValueError(
                f"Routing function can only be a Callable not {type(routing_function)}"
            )

        self.mixer_block = self.block_setter(mixer_block, GateMapType.MIXER)
        (
            self.mixer_single_qubit_coeffs,
            self.mixer_pair_qubit_coeffs,
            self.mixer_qubits_singles,
            self.mixer_qubits_pairs,
        ) = self._assign_coefficients(self.mixer_block, self.mixer_block_coeffs)

        self.mixer_blocks = HamiltonianMapper.repeat_gate_maps(self.mixer_block, self.p)
        self.cost_blocks = HamiltonianMapper.repeat_gate_maps(self.cost_block, self.p)
        self.qureg = list(range(self.n_qubits))

    @property
    def n_qubits(self) -> int:
        if self.routed == True:
            return len(self.final_mapping)
        else:
            return self.cost_hamiltonian.n_qubits

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
            if each_gatemap.gate_label.n_qubits == 1:
                single_qubit_coeffs.append(each_coeff)
                # Giving a string name to each gatemap (?)
                qubit_singles.append(type(each_gatemap).__name__)
            elif each_gatemap.gate_label.n_qubits == 2:
                pair_qubit_coeffs.append(each_coeff)
                qubit_pairs.append(type(each_gatemap).__name__)

        return (single_qubit_coeffs, pair_qubit_coeffs, qubit_singles, qubit_pairs)

    @staticmethod
    def block_setter(
        input_object: Union[List["RotationGateMap"], Hamiltonian], block_type: Enum
    ) -> List["RotationGateMap"]:
        """
        Converts a Hamiltonian Object into a List of RotationGateMap Objects with
        the appropriate block_type and sequence assigned to the GateLabel

        OR

        Remaps a list of RotationGateMap Objects with a block_type and sequence
        implied from its position in the list.

        Parameters
        ----------
        input_object: `Union[List[RotationGateMap], Hamiltonian]`
            A Hamiltonian Object or a list of RotationGateMap Objects (Ordered
            according to their application order in the final circuit)
        block_type: Enum
            The type to be assigned to all the RotationGateMap Objects generated
            from input_object

        Returns
        -------
        `List[RotationGateMap]`
        """

        if isinstance(input_object, Hamiltonian):
            block = HamiltonianMapper.generate_gate_maps(input_object, block_type)
        elif isinstance(input_object, list):
            input_object = QAOADescriptor.set_block_sequence(input_object)
            for each_gate in input_object:
                if isinstance(each_gate, RotationGateMap):
                    each_gate.gate_label.update_gatelabel(new_gatemap_type=block_type)
                else:
                    raise TypeError(
                        f"Input gate is of unsupported type {type(each_gate)}."
                        "Only RotationGateMaps are supported"
                    )
            block = input_object
        else:
            raise ValueError(
                "The input object defining mixer should be a List of RotationGateMaps or type Hamiltonian"
            )
        return block

    @staticmethod
    def set_block_sequence(
        input_gatemap_list: List["RotationGateMap"],
    ) -> List["RotationGateMap"]:
        """
        This method assigns the sequence attribute to all RotationGateMap objects in the list.
        The sequence of the GateMaps are implied based on their positions in the list.

        Parameters
        ----------
        input_gatemap_list: `List[RotationGateMap]`
            A list of RotationGateMap Objects

        Returns
        -------
        `List[RotationGateMap]`
        """

        one_qubit_count = 0
        two_qubit_count = 0

        for each_gate in input_gatemap_list:
            if isinstance(each_gate, RotationGateMap):
                if each_gate.gate_label.n_qubits == 1:
                    each_gate.gate_label.update_gatelabel(
                        new_application_sequence=one_qubit_count,
                    )
                    one_qubit_count += 1
                elif each_gate.gate_label.n_qubits == 2:
                    each_gate.gate_label.update_gatelabel(
                        new_application_sequence=two_qubit_count,
                    )
                    two_qubit_count += 1
            else:
                raise TypeError(
                    f"Input gate is of unsupported type {type(each_gate)}."
                    "Only RotationGateMaps are supported"
                )
        return input_gatemap_list

    def reorder_gates_block(self, gates_block, layer_number):
        """Update the qubits that the gates are acting on after application
        of SWAPs in the cost layer
        """
        for gate in gates_block:
            if layer_number % 2 == 0:
                mapping = self.final_mapping
                gate.qubit_1 = mapping[gate.qubit_1]
                if gate.gate_label.n_qubits == 2:
                    gate.qubit_2 = mapping[gate.qubit_2]
            else:
                pass

        return gates_block

    @staticmethod
    def route_gates_list(
        gates_to_route: List["GateMap"],
        device: "DeviceBase",
        routing_function: Callable,
    ) -> List["GateMap"]:
        """
        Apply qubit routing to the abstract circuit gate list
        based on device information

        Parameters
        ----------
        gates_to_route: `List[GateMap]`
            The gates to route
        device: `DeviceBase`
            The device on which to run the circuit
        routing_function: `Callable`
            The function that accepts as input the device, problem, initial_mapping and
            outputs the list of gates with swaps
        """
        original_qubits_to_gate_mapping = {
            (gate.qubit_1, gate.qubit_2): gate
            for gate in gates_to_route
            if gate.gate_label.n_qubits == 2
        }
        problem_to_solve = list(original_qubits_to_gate_mapping.keys())
        (
            gate_list_indices,
            swap_mask,
            initial_physical_to_logical_mapping,
            final_mapping,
        ) = routing_function(device, problem_to_solve)

        gates_list = [gate for gate in gates_to_route if gate.gate_label.n_qubits == 1]
        swapped_history = []
        for idx, pair_ij in enumerate(gate_list_indices):
            mask = swap_mask[idx]
            qi, qj = pair_ij
            if mask == True:
                swapped_history.append(pair_ij)
                gates_list.append(SWAPGateMap(qi, qj))
            elif mask == False:
                old_qi, old_qj = qi, qj
                # traverse each SWAP application in reverse order to obtain
                # the original location of the current qubit
                for swap_pair in swapped_history[::-1]:
                    if old_qi in swap_pair:
                        old_qi = (
                            swap_pair[0] if swap_pair[1] == old_qi else swap_pair[1]
                        )
                    if old_qj in swap_pair:
                        old_qj = (
                            swap_pair[0] if swap_pair[1] == old_qj else swap_pair[1]
                        )
                try:
                    ising_gate = original_qubits_to_gate_mapping[
                        tuple([old_qi, old_qj])
                    ]
                except KeyError:
                    ising_gate = original_qubits_to_gate_mapping[
                        tuple([old_qj, old_qi])
                    ]
                except Exception as e:
                    raise e
                ising_gate.qubit_1, ising_gate.qubit_2 = qi, qj
                gates_list.append(ising_gate)

        return (
            gates_list,
            list(initial_physical_to_logical_mapping.keys()),
            final_mapping,
        )

    @property
    def abstract_circuit(self):
        # even layer inversion if the circuit contains SWAP gates
        even_layer_inversion = -1 if self.routed == True else 1
        _abstract_circuit = []
        for each_p in range(self.p):
            # apply each cost_block with reversed order to maintain the SWAP sequence
            _abstract_circuit.extend(
                self.cost_blocks[each_p][:: (even_layer_inversion) ** each_p]
            )
            # apply the mixer block
            if self.routed == True:
                mixer_block = self.reorder_gates_block(
                    self.mixer_blocks[each_p], each_p
                )
            else:
                mixer_block = self.mixer_blocks[each_p]
            _abstract_circuit.extend(mixer_block)

        return _abstract_circuit
