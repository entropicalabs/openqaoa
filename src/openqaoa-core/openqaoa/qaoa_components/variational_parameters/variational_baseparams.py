from __future__ import annotations
from typing import Union, Iterable
from abc import ABC
import numpy as np

from ..ansatz_constructor import QAOADescriptor


class QAOAVariationalBaseParams(ABC):
    """
    A class that initialises and keeps track of the Variational
    parameters

    Parameters
    ----------
    qaoa_descriptor: `QAOADescriptor`
        Specify the circuit parameters to construct circuit angles to be
        used for training

    Attributes
    ----------
    qaoa_descriptor: `QAOADescriptor`
    p: `int`
    cost_1q_coeffs
    cost_2q_coeffs
    mixer_1q_coeffs
    mixer_2q_coeffs
    """

    def __init__(self, qaoa_descriptor: QAOADescriptor):
        self.qaoa_descriptor = qaoa_descriptor
        self.p = self.qaoa_descriptor.p

        try:
            self.cost_1q_coeffs = qaoa_descriptor.cost_single_qubit_coeffs
            self.cost_2q_coeffs = qaoa_descriptor.cost_pair_qubit_coeffs
            self.mixer_1q_coeffs = qaoa_descriptor.mixer_single_qubit_coeffs
            self.mixer_2q_coeffs = qaoa_descriptor.mixer_pair_qubit_coeffs
        except AttributeError:
            self.cost_1q_coeffs = qaoa_descriptor.cost_hamiltonian.single_qubit_coeffs
            self.cost_2q_coeffs = qaoa_descriptor.cost_hamiltonian.pair_qubit_coeffs
            self.mixer_1q_coeffs = qaoa_descriptor.mixer_hamiltonian.single_qubit_coeffs
            self.mixer_2q_coeffs = qaoa_descriptor.mixer_hamiltonian.pair_qubit_coeffs

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
        apply X-rotations on.
        """
        raise NotImplementedError()

    @property
    def mixer_2q_angles(self) -> np.ndarray:
        """2D array with the X-rotation angles.

        1st index goes over p and the 2nd index over the qubits to
        apply X-rotations on.
        """
        raise NotImplementedError()

    @property
    def cost_1q_angles(self) -> np.ndarray:
        """2D array with the ZZ-rotation angles.

        1st index goes over the p and the 2nd index over the qubit
        pairs, to apply ZZ-rotations on.
        """
        raise NotImplementedError()

    @property
    def cost_2q_angles(self) -> np.ndarray:
        """2D array with Z-rotation angles.

        1st index goes over the p and the 2nd index over the qubit
        pairs, to apply Z-rotations on. These are needed by
        ``qaoa.cost_function.make_qaoa_memory_map``
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
        new_values: `Union[list, np.array]`
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
            The parameters in a 1D array. Has the same output format as the
            expected input of ``self.update_from_raw``. Hence corresponds to
            the flattened `parameters` in `__init__()`

        """
        raise NotImplementedError()

    def update_from_dict(self, new_values: dict):
        """
        Update all the parameters from a dictionary.

        The input has the same format as the output of ``self.asdict()``.

        Parameters
        ----------
        new_values: `dict`
            A dictionary with the new parameters. Must have the same keys as
            the output of ``self.asdict()``.

        """

        assert isinstance(new_values, dict), f"Expected dict, got {type(new_values)}"

        for key, value in new_values.items():
            if key not in self.asdict().keys():
                raise KeyError(
                    f"'{key}' not in {self.__class__.__name__}, expected keys: {list(self.asdict().keys())}"
                )
            else:
                if getattr(self, key).shape != np.array(value).shape:
                    raise ValueError(
                        f"Shape of '{key}' does not match. Expected shape {getattr(self, key).shape}, got {np.array(value).shape}."
                    )

        raw_params = []
        for key, value in self.asdict().items():
            if key in new_values.keys():
                raw_params += list(np.array(new_values[key]).flatten())
            else:
                raw_params += list(np.array(value).flatten())

        self.update_from_raw(raw_params)

    def asdict(self) -> dict:
        """
        Return the parameters as a dictionary.

        Returns
        -------
        dict:
            The parameters as a dictionary. Has the same output format as the
            expected input of ``self.update_from_dict``.

        """
        return {k[2:]: v for k, v in self.__dict__.items() if k[0:2] == "__"}

    @classmethod
    def linear_ramp_from_hamiltonian(
        cls, qaoa_descriptor: QAOADescriptor, time: float = None
    ):
        """Alternative to ``__init__`` that already fills ``parameters``.

        Calculate initial parameters from register, terms, weights
        (specifiying a Hamiltonian), corresponding to a linear ramp
        annealing schedule and return a ``QAOAVariationalBaseParams`` object.

        Parameters
        ----------
        qaoa_descriptor: `QAOADescriptor`
            QAOADescriptor object containing information about terms,weights,register and p

        time: `float`
            Total annealing time. Defaults to ``0.7*p``.

        Returns
        -------
        QAOAVariationalBaseParams:
            The initial parameters for a linear ramp for ``hamiltonian``.

        """
        raise NotImplementedError()

    @classmethod
    def random(cls, qaoa_descriptor: QAOADescriptor, seed: int = None):
        """
        Initialise parameters randomly

        Parameters
        ----------
        qaoa_descriptor: `QAOADescriptor`
            QAOADescriptor object containing information about terms,
            weights, register and p.

        seed: `int`
                Use a fixed seed for reproducible random numbers

        Returns
        -------
        QAOAVariationalBaseParams:
            Randomly initialiased parameters
        """
        raise NotImplementedError()

    @classmethod
    def empty(cls, qaoa_descriptor: QAOADescriptor):
        """
        Alternative to ``__init__`` that only takes ``qaoa_descriptor`` and
        fills ``parameters`` via ``np.empty``

        Parameters
        ----------
        qaoa_descriptor: `QAOADescriptor`
            QAOADescriptor object containing information about terms,weights,register and p

        Returns
        -------
        QAOAVariationalBaseParams:
            A Parameter object with the parameters filled by ``np.empty``
        """
        raise NotImplementedError()

    @classmethod
    def from_other_parameters(cls, params):
        """Alternative to ``__init__`` that takes parameters with less degrees
        of freedom as the input.

        Parameters
        ----------
        params: `QAOAVaritionalBaseParams`
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
        raw_data = np.concatenate(
            (
                self.mixer_1q_angles.flatten(),
                self.mixer_2q_angles.flatten(),
                self.cost_1q_angles.flatten(),
                self.cost_1q_angles.flatten(),
            )
        )
        return raw_data

    def plot(self, ax=None, **kwargs):
        """
        Plots ``self`` in a sensible way to the canvas ``ax``, if provided.

        Parameters
        ----------
        ax: `matplotlib.axes._subplots.AxesSubplot`
                The canvas to plot itself on
        kwargs:
                All remaining keyword arguments are passed forward to the plot
                function

        """
        raise NotImplementedError()


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

    def __init__(
        self,
        variational_params: QAOAVariationalBaseParams,
        the_parameter: str,
        the_range: Iterable[float],
    ):
        """See class documentation for details"""
        self.params = variational_params
        self.iterator = iter(the_range)
        self.the_parameter, *indices = the_parameter.split("[")
        indices = [i.replace("]", "") for i in indices]
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
            getattr(self.params, self.the_parameter)[self.index0][self.index1] = value
        else:
            getattr(self.params, self.the_parameter)[self.index0] = value

        return self.params
