#   Copyright 2022 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from abc import ABC
from typing import List, Union, Tuple, Any, Callable, Iterable, Type
import numpy as np

from .operators import Hamiltonian
from .hamiltonianmapper import HamiltonianMapper



def _is_iterable_empty(in_iterable):
    if isinstance(in_iterable, Iterable):    # Is Iterable
        return all(map(_is_iterable_empty, in_iterable))
    return False    # Not an Iterable


class shapedArray:
    """Decorator-Descriptor for arrays that have a fixed shape.

    This is used to facilitate automatic input checking for all the different
    internal parameters. Each instance of this class can be removed without
    replacement and the code should still work, provided the user provides
    only correct angles to below parameter classes

    Parameters
    ----------
    shape: Callable[[Any], Tuple]:
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
            setattr(obj, f"__{self.name}", np.reshape(values, self.shape(obj)))
        except ValueError:
            raise ValueError(f"{self.name} must have shape {self.shape(obj)}")

    def __get__(self, obj, objtype):
        """The getter."""
        return getattr(obj, f"__{self.name}")


class VQACircuitParams(ABC):
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

    def __init__(self,
                 algorithm: str):
        self.algorithm = algorithm


class QAOACircuitParams(VQACircuitParams):
    """
    Create the problem attributes consisting of the Hamiltonian and Hamiltonian
    squared as dictionaries, QAOA 'p' value and other specific parameters.

    Parameters
    ----------    
    cost_hamiltonian: `Hamiltonian` 

    mixer_hamiltonian: `Hamiltonian`

    p: 
        Number of QAOA layers; defaults to 1 if not specified

    Attributes
    ----------
        All the parameters specified above are initialised as attributes in __init__

    Properties
    ----------
    hamiltonian: 
        Returns the problem Hamiltonian as a dictionary
    hamiltonian_squared: 
        Returns the problem Hamiltonian Squared as a dictionary
    """

    def __init__(self,
                 cost_hamiltonian: Hamiltonian,
                 mixer_hamiltonian: Hamiltonian,
                 p: int):

        super().__init__(algorithm='QAOA')
        self.cost_hamiltonian = cost_hamiltonian
        self.mixer_hamiltonian = mixer_hamiltonian
        self.qureg = cost_hamiltonian.qureg
        self.p = p

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
        string += "\tcost_qubits_singles: " + \
            str(self.cost_hamiltonian.qubits_singles) + "\n"
        string += "\tcost_single_qubit_coeffs: " + \
            str(self.cost_hamiltonian.single_qubit_coeffs) + "\n"
        string += "\tcost_qubits_pairs: " + \
            str(self.cost_hamiltonian.qubits_pairs) + "\n"
        string += "\tcost_pair_qubit_coeffs: " + \
            str(self.cost_hamiltonian.pair_qubit_coeffs) + "\n" + "\n"

        string += "Mixer Hamiltonian:\n"
        string += "\tmixer_qubits_singles: " + \
            str(self.mixer_hamiltonian.qubits_singles) + "\n"
        string += "\tmixer_single_qubit_coeffs: " + \
            str(self.mixer_hamiltonian.single_qubit_coeffs) + "\n"
        string += "\tmixer_qubits_pairs: " + \
            str(self.mixer_hamiltonian.qubits_pairs) + "\n"
        string += "\tmixer_pair_qubit_coeffs: " + \
            str(self.mixer_hamiltonian.pair_qubit_coeffs) + "\n"

        return string

    @property
    def pseudo_circuit(self) -> List:

        cost_pauli_gate_list = HamiltonianMapper.repeat_paulis(
            self.cost_hamiltonian, 'cost', self.p)
        mixer_pauli_gate_list = HamiltonianMapper.repeat_paulis(
            self.mixer_hamiltonian, 'mixer', self.p)

        _pseudo_circuit = []
        for each_p in range(self.p):
            _pseudo_circuit.extend(cost_pauli_gate_list[each_p])
            _pseudo_circuit.extend(mixer_pauli_gate_list[each_p])

        return _pseudo_circuit


class QAOAVariationalBaseParams(ABC):
    """
    A class that initialises and keeps track of the Variational
    parameters 

    Parameters
    ----------
    qaoa_circuit_params:
        Specify the circuit parameters to construct circuit angles to be 
        used for training

    Attributes
    ----------

    """

    def __init__(self, qaoa_circuit_params: Type[QAOACircuitParams]):

        self.qaoa_circuit_params = qaoa_circuit_params
        self.p = self.qaoa_circuit_params.p
        self.cost_1q_coeffs = qaoa_circuit_params.cost_hamiltonian.single_qubit_coeffs
        self.cost_2q_coeffs = qaoa_circuit_params.cost_hamiltonian.pair_qubit_coeffs
        self.mixer_1q_coeffs = qaoa_circuit_params.mixer_hamiltonian.single_qubit_coeffs
        self.mixer_2q_coeffs = qaoa_circuit_params.mixer_hamiltonian.pair_qubit_coeffs

    def __len__(self):
        """
        Returns
        -------
        int:
                the length of the data produced by self.raw() and accepted by
                self.update_from_raw()
        """
        raise NotImplementedError()

    def __repr__(self):

        raise NotImplementedError()

    def __str__(self):

        return self.__repr__()

    @property
    def mixer_1q_angles(self) -> np.ndarray:
        """2D array with the X-rotation angles.

        1st index goes over p and the 2nd index over the qubits to
        apply X-rotations on. These are needed by ``qaoa.cost_function.make_qaoa_memory_map``
        """
        raise NotImplementedError()

    @property
    def mixer_2q_angles(self) -> np.ndarray:
        """2D array with the X-rotation angles.

        1st index goes over p and the 2nd index over the qubits to
        apply X-rotations on. These are needed by ``qaoa.cost_function.make_qaoa_memory_map``
        """
        raise NotImplementedError()

    @property
    def cost_1q_angles(self) -> np.ndarray:
        """2D array with the ZZ-rotation angles.

        1st index goes over the p and the 2nd index over the qubit
        pairs, to apply ZZ-rotations on. These are needed by ``qaoa.cost_function.make_qaoa_memory_map``
        """
        raise NotImplementedError()

    @property
    def cost_2q_angles(self) -> np.ndarray:
        """2D array with Z-rotation angles.

        1st index goes over the p and the 2nd index over the qubit
        pairs, to apply Z-rotations on. These are needed by ``qaoa.cost_function.make_qaoa_memory_map``
        """
        raise NotImplementedError()

    def update_from_raw(self, new_values: Union[list, np.array]):
        """
        Update all the parameters from a 1D array.

        The input has the same format as the output of ``self.raw()``.
        This is useful for ``scipy.optimize.minimize`` which expects
        the parameters that need to be optimized to be a 1D array.

        Parameters
        ----------
        new_values:
                A 1D array with the new parameters. Must have length  ``len(self)``
                and the ordering of the flattend ``parameters`` in ``__init__()``.

        """
        raise NotImplementedError()

    def raw(self) -> np.ndarray:
        """
        Return the parameters in a 1D array.

        This 1D array is needed by ``scipy.optimize.minimize`` which expects
        the parameters that need to be optimized to be a 1D array.

        Returns
        -------
        np.array:
                The parameters in a 1D array. Has the same output
                format as the expected input of ``self.update_from_raw``. Hence
                corresponds to the flattened `parameters` in `__init__()`

        """
        raise NotImplementedError()

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     qaoa_circuit_params: QAOACircuitParams,
                                     time: float = None):
        """Alternative to ``__init__`` that already fills ``parameters``.

        Calculate initial parameters from register, terms, weights (specifiying a Hamiltonian), corresponding to a
        linear ramp annealing schedule and return a ``QAOAParams`` object.

        Parameters
        ----------
        qaoa_circuit_params:
                QAOACircuitParams object containing information about terms,weights,register and p

        time:
                Total annealing time. Defaults to ``0.7*p``.

        Returns
        -------
        VariationalBaseParams
                The initial parameters for a linear ramp for ``hamiltonian``.

        """
        raise NotImplementedError()

    @classmethod
    def random(cls, qaoa_circuit_params: QAOACircuitParams, seed: int = None):
        """
        Initialise parameters randomly

        Parameters
        ----------
        qaoa_circuit_params:
                QAOACircuitParams object containing information about terms,weights,register and p

        seed:
                Use a fixed seed for reproducible random numbers

        Returns
        -------
        VariationalBaseParams
                Randomly initialiased parameters
        """
        raise NotImplementedError()

    @classmethod
    def empty(cls, qaoa_circuit_params: QAOACircuitParams):
        """
        Alternative to ``__init__`` that only takes ``qaoa_circuit_params`` and
        fills ``parameters`` via ``np.empty``

        Parameters
        ----------
        qaoa_circuit_params:
            QAOACircuitParams object containing information about terms,weights,register and p

        Returns
        -------
        VariationalBaseParams:
            A Parameter object with the parameters filled by ``np.empty``
        """
        raise NotImplementedError()

    @classmethod
    def from_other_parameters(cls, params):
        """Alternative to ``__init__`` that takes parameters with less degrees
        of freedom as the input.
        Parameters
        ----------
        params: QAOAVaritionalBaseParams
            The input parameters object to construct the new parameters object from.
        Returns
        -------
        QAOAVariationalBaseParams:
            The converted paramters s.t. all the rotation angles of the in
            and output parameters are the same.
        """
        from . import converter
        return converter(params, cls)

    def raw_rotation_angles(self) -> np.ndarray:
        """
        Flat array of the rotation angles for the memory map for the
        parametric circuit.

        Returns
        -------
        np.array:
                Returns all single rotation angles in the ordering
                ``(x_rotation_angles, gamma_singles, zz_rotation_angles)`` where
                ``x_rotation_angles = (beta_q0_t0, beta_q1_t0, ... , beta_qn_tp)``
                and the same for ``z_rotation_angles`` and ``zz_rotation_angles``

        """
        raw_data = np.concatenate((self.mixer_1q_angles.flatten(),
                                   self.mixer_2q_angles.flatten(),
                                   self.cost_1q_angles.flatten(),
                                   self.cost_1q_angles.flatten(),))
        return raw_data

    def plot(self, ax=None, **kwargs):
        """
        Plots ``self`` in a sensible way to the canvas ``ax``, if provided.

        Parameters
        ----------
        ax: matplotlib.axes._subplots.AxesSubplot
                The canvas to plot itself on
        kwargs:
                All remaining keyword arguments are passed forward to the plot
                function

        """
        raise NotImplementedError()

    # def QAOA_circuit_params(self):
    # 	"""
    # 	Returns parameters needed for QAOA circuit in tuple form.
    # 	"""

    # 	p = self.p
    # 	x_rotation_angles = self.x_rotation_angles
    # 	bias_qubits = self.qubits_singles
    # 	pairs = self.qubits_pairs
    # 	z_rotation_angles = self.z_rotation_angles
    # 	zz_rotation_angles = self.zz_rotation_angles

    # 	return (p,x_rotation_angles,bias_qubits,z_rotation_angles,pairs,zz_rotation_angles)
    # 	return (self.p, self.mixer_1q_angles, self.mixer_2q_angles,
    # 			self.cost_1q_angles, self.cost_2q_angles)


class QAOAParameterIterator:
    """An iterator to sweep one parameter over a range in a QAOAParameter object.

    Parameters
    ----------
    qaoa_params:
        The initial QAOA parameters, where one of them is swept over
    the_parameter:
        A string specifying, which parameter should be varied. It has to be
        of the form ``<attr_name>[i]`` where ``<attr_name>`` is the name
        of the _internal_ list and ``i`` the index, at which it sits. E.g.
        if ``qaoa_params`` is of type ``AnnealingParams``
        and  we want to vary over the second timestep, it is
        ``the_parameter = "times[1]"``.
    the_range:
        The range, that ``the_parameter`` should be varied over

    Todo
    ----
    - Add checks, that the number of indices in ``the_parameter`` matches
      the dimensions of ``the_parameter``
    - Add checks, that the index is not too large

    Example
    -------
    Assume qaoa_params is of type ``StandardWithBiasParams`` and
    has `p >= 2`. Then the following code produces a loop that
    sweeps ``gammas_singles[1]`` over the range ``(0, 1)`` in 4 layers:

    .. code-block:: python

        the_range = np.arange(0, 1, 0.4)
        the_parameter = "gammas_singles[1]"
        param_iterator = QAOAParameterIterator(qaoa_params, the_parameter, the_range)
        for params in param_iterator:
            # do what ever needs to be done.
            # we have type(params) == type(qaoa_params)
    """

    def __init__(self,
                 variational_params: QAOAVariationalBaseParams,
                 the_parameter: str,
                 the_range: Iterable[float]):
        """See class documentation for details"""
        self.params = variational_params
        self.iterator = iter(the_range)
        self.the_parameter, *indices = the_parameter.split("[")
        indices = [i.replace(']', '') for i in indices]
        if len(indices) == 1:
            self.index0 = int(indices[0])
            self.index1 = False
        elif len(indices) == 2:
            self.index0 = int(indices[0])
            self.index1 = int(indices[1])
        else:
            raise ValueError("the_parameter has to many indices")

    def __iter__(self):
        return self

    def __next__(self):
        # get next value from the_range
        value = next(self.iterator)

        # 2d list or 1d list?
        if self.index1 is not False:
            getattr(self.params, self.the_parameter)[
                self.index0][self.index1] = value
        else:
            getattr(self.params, self.the_parameter)[self.index0] = value

        return self.params
