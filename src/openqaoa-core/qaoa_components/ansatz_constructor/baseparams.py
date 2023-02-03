from __future__ import annotations
from abc import ABC
from typing import List, Union, Tuple, Any, Callable, Iterable
import numpy as np

from .operators import Hamiltonian
from .hamiltonianmapper import HamiltonianMapper
from .gatemap import GateMap, RotationGateMap, SWAPGateMap
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
        routed_gate_list_indices:List = None,
        swap_mask:List[bool] = None
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
        """
        
        super().__init__(algorithm = "QAOA")
        
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
        
        self.routed_gate_list_indices = routed_gate_list_indices
        self.swap_mask = swap_mask
        self.routed = False
        
        (self.cost_single_qubit_coeffs, self.cost_pair_qubit_coeffs, self.cost_qubits_singles,
         self.cost_qubits_pairs) = self._assign_coefficients(self.cost_block, self.cost_block_coeffs)
        
        (self.mixer_single_qubit_coeffs, self.mixer_pair_qubit_coeffs, self.mixer_qubits_singles,
         self.mixer_qubits_pairs) = self._assign_coefficients(self.mixer_block, self.mixer_block_coeffs)

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

    @property
    def cost_block(self):

        return self._cost_block

    @cost_block.setter
    def cost_block(self, input_object: Hamiltonian) -> None:

        self._cost_block = HamiltonianMapper.generate_gate_maps(input_object, GateMapType.COST)

    @property
    def mixer_block(self):

        return self._mixer_block

    @mixer_block.setter
    def mixer_block(self, input_object: Union[List[RotationGateMap], Hamiltonian]) -> None:
        
        if isinstance(input_object, Hamiltonian):            
            self._mixer_block = HamiltonianMapper.generate_gate_maps(input_object,GateMapType.MIXER)
        elif isinstance(input_object, list):
            for each_gate in input_object:
                if isinstance(each_gate, RotationGateMap):
                    each_gate.gate_label.update_gatelabel(gatemap_type=GateMapType.MIXER)
                else:
                    raise TypeError(f"Input gate is of unsupported type {type(each_gate)}."\
                                    "Only RotationGateMaps are supported")
            self._mixer_block = input_object
        else:
            raise ValueError("The input object defining mixer should be a List of RotationGateMaps or type Hamiltonian")

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
        
        #route the cost block and append SWAP gates
        if (self.routed_gate_list_indices is not None and\
            self.swap_mask is not None and self.routed is False
        ):
            self.route_cost_block()
        
        self.mixer_blocks = HamiltonianMapper.repeat_gate_maps(self.mixer_block,self.p)
        self.cost_blocks = HamiltonianMapper.repeat_gate_maps(self.cost_block,self.p) 

        _abstract_circuit = []
        for each_p in range(self.p):
            #apply each cost_block with reversed order to maintain the SWAP sequence
            _abstract_circuit.extend(self.cost_blocks[each_p][::(-1)**each_p])
            _abstract_circuit.extend(self.mixer_blocks[each_p])

        return _abstract_circuit
    
    def route_cost_block(self) -> List[GateMap]:
        """
        Apply qubit routing to the abstract circuit gate list
        based on device information
        """
        for i,(gate_ind,mask) in enumerate(zip(self.routed_gate_list_indices, self.swap_mask)):
            if mask == True:
                self.cost_block.insert(i,SWAPGateMap(gate_ind[0],gate_ind[1]))
        self.routed = True