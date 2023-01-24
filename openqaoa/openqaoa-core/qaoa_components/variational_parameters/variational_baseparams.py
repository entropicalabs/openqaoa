from __future__ import annotations

from abc import ABC

class QAOAVariationalBaseParams(ABC):
    """
    A class that initialises and keeps track of the Variational
    parameters 

    Parameters
    ----------
    qaoa_circuit_params: `QAOACircuitParams`
        Specify the circuit parameters to construct circuit angles to be 
        used for training

    Attributes
    ----------
    qaoa_circuit_params: `QAOACircuitParams`
    p: `int`
    cost_1q_coeffs
    cost_2q_coeffs
    mixer_1q_coeffs
    mixer_2q_coeffs
    """

    def __init__(self, qaoa_circuit_params: QAOACircuitParams):

        self.qaoa_circuit_params = qaoa_circuit_params
        self.p = self.qaoa_circuit_params.p
        
        try:
            self.cost_1q_coeffs = qaoa_circuit_params.cost_single_qubit_coeffs
            self.cost_2q_coeffs = qaoa_circuit_params.cost_pair_qubit_coeffs
            self.mixer_1q_coeffs = qaoa_circuit_params.mixer_single_qubit_coeffs
            self.mixer_2q_coeffs = qaoa_circuit_params.mixer_pair_qubit_coeffs
        except AttributeError:
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

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     qaoa_circuit_params: QAOACircuitParams,
                                     time: float = None):
        """Alternative to ``__init__`` that already fills ``parameters``.

        Calculate initial parameters from register, terms, weights (specifiying a Hamiltonian), corresponding to a
        linear ramp annealing schedule and return a ``QAOAVariationalBaseParams`` object.

        Parameters
        ----------
        qaoa_circuit_params: `QAOACircuitParams`
            QAOACircuitParams object containing information about terms,weights,register and p

        time: `float`
            Total annealing time. Defaults to ``0.7*p``.

        Returns
        -------
        QAOAVariationalBaseParams: 
            The initial parameters for a linear ramp for ``hamiltonian``.

        """
        raise NotImplementedError()

    @classmethod
    def random(cls, qaoa_circuit_params: QAOACircuitParams, seed: int = None):
        """
        Initialise parameters randomly

        Parameters
        ----------
        qaoa_circuit_params: `QAOACircuitParams`
            QAOACircuitParams object containing information about terms, 
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
    def empty(cls, qaoa_circuit_params: QAOACircuitParams):
        """
        Alternative to ``__init__`` that only takes ``qaoa_circuit_params`` and
        fills ``parameters`` via ``np.empty``

        Parameters
        ----------
        qaoa_circuit_params: `QAOACircuitParams`
            QAOACircuitParams object containing information about terms,weights,register and p

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
        ax: `matplotlib.axes._subplots.AxesSubplot`
                The canvas to plot itself on
        kwargs:
                All remaining keyword arguments are passed forward to the plot
                function

        """
        raise NotImplementedError()