from typing import List, Optional, Union
import numpy as np
import json
from scipy.optimize._minimize import MINIMIZE_METHODS

from ..optimizers.training_vqa import CustomScipyGradientOptimizer, PennyLaneOptimizer
from ..backends.devices_core import SUPPORTED_LOCAL_SIMULATORS
from ..backends.basebackend import QuantumCircuitBase
from ..utilities import convert2serialize


ALLOWED_PARAM_TYPES = [
    "standard",
    "standard_w_bias",
    "extended",
    "fourier",
    "fourier_extended",
    "fourier_w_bias",
    "annealing",
]
ALLOWED_INIT_TYPES = ["rand", "ramp", "custom"]
ALLOWED_MIXERS = ["x", "xy"]

ALLOWED_MINIMIZATION_METHODS = (
    MINIMIZE_METHODS
    + CustomScipyGradientOptimizer.CUSTOM_GRADIENT_OPTIMIZERS
    + PennyLaneOptimizer.PENNYLANE_OPTIMIZERS
)

ALLOWED_QVM_DEVICES = ["Aspen-11", "Aspen-M-1"]
ALLOWED_QVM_DEVICES.extend(f"{n}q-qvm" for n in range(2, 80))

ALLOWED_LOCAL_SIMUALTORS = SUPPORTED_LOCAL_SIMULATORS
ALLOWED_IMBQ_GLOBAL = [
    "ibmq_qasm_simulator",
    "ibmq_armonk",
    "ibmq_santiago",
    "ibmq_bogota",
    "ibmq_lima",
    "ibmq_belem",
    "ibmq_quito",
    "simulator_statevector",
    "simulator_mps",
    "simulator_extended_stabilizer",
    "simulator_stabilizer",
    "ibmq_manila",
]
ALLOWED_DEVICES = ALLOWED_LOCAL_SIMUALTORS + ALLOWED_QVM_DEVICES + ALLOWED_IMBQ_GLOBAL


class WorkflowProperties:
    def __iter__(self):
        for key, value in self.__dict__.items():
            # remove "_" from the beginning of the key if it exists
            yield (key[1:] if key.startswith("_") else key, value)

    def asdict(self):
        return convert2serialize(dict(self))


class CircuitProperties(WorkflowProperties):
    """
    Tunable properties of the QAOA circuit to be specified by the user
    """

    def __init__(
        self,
        param_type: str = "standard",
        init_type: str = "ramp",
        qubit_register: List = [],
        p: int = 1,
        q: Optional[int] = 1,
        annealing_time: Optional[float] = None,
        linear_ramp_time: Optional[float] = None,
        variational_params_dict: Optional[dict] = {},
        mixer_hamiltonian: Optional[str] = "x",
        mixer_qubit_connectivity: Optional[Union[List[list], List[tuple], str]] = None,
        mixer_coeffs: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.param_type = param_type
        self.init_type = init_type
        self.qubit_register = qubit_register
        self.p = p
        self.q = (
            q
            if param_type.lower() in ["fourier", "fourier_extended", "fourier_w_bias"]
            else None
        )
        self.variational_params_dict = variational_params_dict
        self.annealing_time = (
            annealing_time if annealing_time is not None else 0.7 * self.p
        )
        self.linear_ramp_time = (
            linear_ramp_time if linear_ramp_time is not None else 0.7 * self.p
        )
        self.mixer_hamiltonian = mixer_hamiltonian
        if self.mixer_hamiltonian.lower() == "xy":
            self.mixer_qubit_connectivity = (
                mixer_qubit_connectivity
                if mixer_qubit_connectivity is not None
                else "full"
            )
        else:
            self.mixer_qubit_connectivity = None
        self.mixer_coeffs = mixer_coeffs
        self.seed = seed

    @property
    def param_type(self):
        return self._param_type

    @param_type.setter
    def param_type(self, value):
        if value not in ALLOWED_PARAM_TYPES:
            raise ValueError(
                f"param_type {value} is not recognised. Please use {ALLOWED_PARAM_TYPES}"
            )
        self._param_type = value

    @property
    def init_type(self):
        return self._init_type

    @init_type.setter
    def init_type(self, value):
        if value not in ALLOWED_INIT_TYPES:
            raise ValueError(
                f"init_type {value} is not recognised. Please use {ALLOWED_INIT_TYPES}"
            )
        self._init_type = value

    @property
    def mixer_hamiltonian(self):
        return self._mixer_hamiltonian

    @mixer_hamiltonian.setter
    def mixer_hamiltonian(self, value):
        if value not in ALLOWED_MIXERS:
            raise ValueError(
                f"mixer_hamiltonian {value} is not recognised. Please use {ALLOWED_MIXERS}"
            )
        self._mixer_hamiltonian = value

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if value <= 0:
            raise ValueError(
                f"Number of layers `p` cannot be smaller or equal to zero. Value {value} was provided"
            )
        self._p = value

    @property
    def annealing_time(self):
        return self._annealing_time

    @annealing_time.setter
    def annealing_time(self, value):
        if value <= 0:
            raise ValueError(
                f"The annealing time `annealing_time` cannot be smaller or equal to zero. Value {value} was provided"
            )
        self._annealing_time = value

    # @property
    # def mixer_qubit_connectivity(self):
    #     return self._mixer_qubit_connectivity

    # @annealing_time.setter
    # def mixer_qubit_connectivity(self, value):
    #     print(value)
    #     if (self.mixer_hamiltonian != 'xy') and (value != None):
    #         self._mixer_qubit_connectivity = None
    #         raise ValueError(f"mixer_qubit_connectivity can be used if and only if `mixer_hamiltonian` is set to `xy`")
    #     else:
    #         print(value)
    #         self._mixer_qubit_connectivity = value


class BackendProperties(WorkflowProperties):
    """
    Choose the backend on which to run the QAOA circuits

    Parameters
    ----------
    device: DeviceBase
        The device to use for the backend.
    prepend_state: Union[openqaoa.basebackend.QuantumCircuitBase,numpy.ndarray(complex)]
        The state prepended to the circuit.
    append_state: Union[QuantumCircuitBase,numpy.ndarray(complex)]
        The state appended to the circuit.
    init_hadamard: bool
        Whether to apply a Hadamard gate to the beginning of the
        QAOA part of the circuit.
    n_shots: int
        The number of shots to be used for the shot-based computation.
    cvar_alpha: float
        The value of the CVaR parameter.
    noise_model: NoiseModel
        The `qiskit` noise model to be used for the shot-based simulator.
    initial_qubit_mapping: Union[List[int], numpy.ndarray]
        Mapping from physical to logical qubit indices, used to eventually
        construct the quantum circuit.  For example, for a system composed by 3 qubits
       `qubit_layout=[1,3,2]`, maps `1<->0`, `3<->1`, `2<->2`, where the left hand side is the physical qubit
        and the right hand side is the logical qubits
    qiskit_simulation_method: str
        Specify the simulation method to use with the `qiskit.AerSimulator`
    qiskit_optimization_level: int, optional
        Specify the qiskit.transpile optimization level. Choose from 0,1,2,3
    seed_simulator: int
        Specify a seed for `qiskit` simulators
    active_reset: bool
        To use the active_reset functionality on Rigetti backends through QCS
    rewiring: str
        Specify the rewiring strategy for compilation for Rigetti QPUs through QCS
    disable_qubit_rewiring: bool
        enable/disbale qubit rewiring when accessing QPUs via the AWS `braket`
    """

    def __init__(
        self,
        prepend_state: Optional[
            Union[QuantumCircuitBase, List[complex], np.ndarray]
        ] = None,
        append_state: Optional[Union[QuantumCircuitBase, np.ndarray]] = None,
        init_hadamard: bool = True,
        n_shots: int = 100,
        cvar_alpha: float = 1,
        noise_model=None,
        initial_qubit_mapping: Optional[Union[List[int], np.ndarray]] = None,
        qiskit_simulation_method: Optional[str] = None,
        qiskit_optimization_level: Optional[int] = None,
        seed_simulator: Optional[int] = None,
        active_reset: Optional[bool] = None,
        rewiring: Optional[str] = None,
        disable_qubit_rewiring: Optional[bool] = None,
    ):
        self.init_hadamard = init_hadamard
        self.n_shots = n_shots
        self.prepend_state = prepend_state
        self.append_state = append_state
        self.cvar_alpha = cvar_alpha
        self.noise_model = noise_model
        self.initial_qubit_mapping = initial_qubit_mapping
        self.seed_simulator = seed_simulator
        self.qiskit_simulation_method = qiskit_simulation_method
        self.qiskit_optimization_level = qiskit_optimization_level
        self.active_reset = active_reset
        self.rewiring = rewiring
        self.disable_qubit_rewiring = disable_qubit_rewiring

    # @property
    # def cvar_alpha(self):
    #     return self._cvar_alpha

    # @cvar_alpha.setter
    # def cvar_alpha(self, value):
    #     if (value <0) or (value>1) :
    #         raise ValueError(
    #             f"cvar_alpha must be between 0 and 1. Received {value}.")
    #     self._cvar_alpha = value

class ErrorMitigationProperties(WorkflowProperties):
    """
    Optional, choose an error mitigation technique for the QAOA circuit.

    Parameters
    ----------
    error_mitigation_technique: str
        The name of the error mitigation technique. Currently supported values: "spam_twirling" for the Spam Twirling mitigation method, and "mitiq_zne" for the Zero-Noise Extrapolation (ZNE) mitigation method from Mitiq framework.
    """
    
    def __init__(
        self,
        error_mitigation_technique: Optional[str] = None,
    ):
        self.error_mitigation_technique = (
            error_mitigation_technique.lower()
            if type(error_mitigation_technique) == str
            else error_mitigation_technique
        )

class SpamProperties(ErrorMitigationProperties):
    """
    Class containing all the required parameters for the execution of the SPAM twirling mitigation technique.

    Parameters
    ----------
    error_mitigation_technique: str
        The name of the error mitigation technique.
    n_batches: Optional[int] = int
        Number of batches in which the total number of shots is divided to. For every batch, we choose a set of qubits at random to which we apply X gates and classical negating. The dafault value is set to 10 to be comparable with most problem sizes in NISQ without creating too much of an overhead.
    calibration_data_location: str
        The path to the file containing calibration data for the specific device.
    """

    def __init__(
        self,
        error_mitigation_technique: Optional[str] = None,
        n_batches: Optional[int] = 10,
        calibration_data_location: Optional[str] = None,
    ):         
        super().__init__(error_mitigation_technique)

        if isinstance(n_batches, int) and n_batches > 0:
            self.n_batches = n_batches
        else:
            raise ValueError("n_batches must be a positive integer.")

        if calibration_data_location != None:
            try:
                with open(calibration_data_location, "r") as file:
                    # Parse the JSON file
                    calibration_data = json.load(file)

                    # Check if the file has the expected structure
                    calibration_measurements = calibration_data["results"][
                        "measurement_outcomes"
                    ]
                    calibration_registers = calibration_data["register"]
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Calibration data file not found at specified location: {}".format(
                        calibration_data_location
                    )
                )
            except ValueError:
                raise ValueError(
                    "Calibration data file {} is not a valid JSON file".format(
                        calibration_data_location
                    )
                )
            except KeyError:
                raise KeyError(
                    "Calibration data file {} structure not as expected".format(
                        calibration_data_location
                    )
                )

        self.calibration_data_location = calibration_data_location

class MitiqZNEProperties(ErrorMitigationProperties):
    """
    Class containing all the required parameters for the execution of the Mitiq Zero-Noise Extrapolation mitigation technique.

    Parameters
    ----------
    error_mitigation_technique: str
        The name of the error mitigation technique.
    factory: str
        The name of the zero-noise extrapolation method. Supported values: "Richardson", "Linear", "Poly", "Exp", "PolyExp", "AdaExp", "FakeNodes".
    scaling: str
        The name of the function for scaling the noise of a quantum circuit. Supported values: "fold_gates_at_random" ("fold_gates_from_right", "fold_gates_from_left" not supported as of version 0.8).
    scale_factors: List[int]
        Sequence of noise scale factors at which expectation values should be measured.
        For factory = "AdaExp", just the first element of the list will be considered.
    order: int
        Extrapolation order (degree of the polynomial fit). It cannot exceed len(scale_factors) - 1.
        Only used for factory = "Poly" or "PolyExp".
    steps: int
        The number of optimization steps. At least 3 are necessary.
        Only used for factory = "AdaExp".
    """

    def __init__(
        self,
        error_mitigation_technique: Optional[str] = None,
        factory: str = 'Linear',
        scaling: str = 'fold_gates_at_random',
        scale_factors: List[int] = [1,2,3],
        order: int = 1, 
        steps: int = 4
    ):         
        super().__init__(error_mitigation_technique)
        self.factory = factory
        self.scaling = scaling
        self.scale_factors = scale_factors
        self.order = order
        self.steps = steps


class ClassicalOptimizer(WorkflowProperties):
    """
    The classical optimizer for the QAOA optimization routine
    of the QAOA circuit parameters.

    Parameters
    ----------
    optimize: bool
        Whether to perform optimization routine on the given QAOA problem
    method: str
        optimization method for QAOA e.g. 'COBYLA'
    maxiter : Optional[int]
        Maximum number of iterations.
    maxfev : Optional[int]
        Maximum number of function evaluations.
    jac: str
        Method to compute the gradient vector. Choose from:
            - ['finite_difference', 'param_shift', 'stoch_param_shift', 'grad_spsa']
    hess:
        Method to compute the hessian. Choose from:
            - ['finite_difference', 'param_shift', 'stoch_param_shift', 'grad_spsa']
    constraints: `scipy.optimize.LinearConstraints`, `scipy.optimize.NonlinearConstraints`
        Scipy-based constraints on parameters of optimization
    bounds: `scipy.scipy.optimize.Bounds`
        Scipy-based bounds on parameters of optimization
    tol : float
        Tolerance before the optimizer terminates; if `tol` is larger than
        the difference between two steps, terminate optimization.
    optimizer_options: dict
        Dictionary of optimiser-specific arguments, defaults to ``None``.
            stepsize : float
                Step size of each gradient descent step.
            decay : float
                Stepsize decay parameter of RMSProp.
            eps : float
                Small number to prevent division by zero for RMSProp.
            lambd : float
                Small number to regularize QFIM for Natural Gradient Descent.
    jac_options : dict
        Dictionary that specifies gradient-computation options according to method chosen in 'jac'.
    hess_options : dict
        Dictionary that specifies Hessian-computation options according to method chosen in 'hess'.
    optimization_progress : bool
        Returns history of measurement outcomes/wavefunction if `True`. Defaults to `False`.
    cost_progress : bool
        Returns history of cost values if `True`. Defaults to `True`.
    parameter_log : bool
        Returns history of angles if `True`. Defaults to `True`.
    save_intermediate: bool
        Outputs the jobids and parameters used for each circuit into
        seperate csv files. Defaults to `False`.
    """

    def __init__(
        self,
        optimize: bool = True,
        method: str = "cobyla",
        maxiter: int = 100,
        maxfev: int = None,
        jac: str = None,
        hess: str = None,
        constraints=None,
        bounds=None,
        tol=None,
        optimizer_options: dict = None,
        jac_options: dict = None,
        hess_options: dict = None,
        optimization_progress: bool = False,
        cost_progress: bool = True,
        parameter_log: bool = True,
        save_intermediate: bool = False,
    ):
        self.optimize = optimize
        self.method = method.lower()
        self.maxiter = maxiter
        self.maxfev = maxfev
        self.jac = jac.lower() if type(jac) == str else jac
        self.hess = hess.lower() if type(hess) == str else hess
        self.constraints = constraints
        self.bounds = bounds
        self.tol = tol
        self.optimizer_options = optimizer_options
        self.jac_options = jac_options
        self.hess_options = hess_options
        self.parameter_log = parameter_log
        self.optimization_progress = optimization_progress
        self.cost_progress = cost_progress
        self.parameter_log = parameter_log
        self.save_intermediate = save_intermediate

    # @property
    # def method(self):
    #     return self._method

    # @method.setter
    # def method(self, value):
    #     if value not in ALLOWED_MINIMIZATION_METHODS:
    #         raise ValueError(
    #             f"method `{value}` is not supported. Please choose between {ALLOWED_MINIMIZATION_METHODS}")
    #     self._method = value
