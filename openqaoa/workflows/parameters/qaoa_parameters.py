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

from typing import List, Dict, Optional, Union
import numpy as np

from openqaoa.basebackend import QuantumCircuitBase
from openqaoa.devices import SUPPORTED_LOCAL_SIMULATORS
from .parameters import Parameters
from scipy.optimize._minimize import MINIMIZE_METHODS


ALLOWED_PARAM_TYPES = ['standard', 'standard_w_bias', 'extended', 'fourier',
                       'fourier_extended', 'fourier_w_bias', 'annealing']
ALLOWED_INIT_TYPES = ['rand', 'ramp', 'custom']
ALLOWED_MIXERS = ['x', 'xy']
ALLOWED_MINIMIZATION_METHODS = MINIMIZE_METHODS

ALLOWED_QVM_DEVICES = ['Aspen-11', 'Aspen-M-1']
ALLOWED_QVM_DEVICES.extend(f'{n}q-qvm' for n in range(2, 80))

ALLOWED_LOCAL_SIMUALTORS = SUPPORTED_LOCAL_SIMULATORS
ALLOWED_IMBQ_GLOBAL =  ['ibmq_qasm_simulator', 'ibmq_armonk', 'ibmq_santiago', 'ibmq_bogota', 'ibmq_lima', 'ibmq_belem', 'ibmq_quito', 'simulator_statevector', 'simulator_mps', 'simulator_extended_stabilizer', 'simulator_stabilizer', 'ibmq_manila']
ALLOWED_DEVICES  = ALLOWED_LOCAL_SIMUALTORS + ALLOWED_QVM_DEVICES + ALLOWED_IMBQ_GLOBAL


class CircuitProperties(Parameters):
    """
    Tunable properties of the QAOA circuit to be specified by the user
    """

    def __init__(self,
                 param_type: str = 'standard',
                 init_type: str = 'ramp',
                 qubit_register: List = [],
                 p: int = 1,
                 q: Optional[int] = 1,
                 annealing_time: Optional[float] = None,
                 linear_ramp_time: Optional[float] = None,
                 variational_params_dict: Optional[dict] = {},
                 mixer_hamiltonian: Optional[str] = 'x',
                 mixer_qubit_connectivity: Optional[Union[List[list],
                                                          List[tuple], str]] = None,
                 mixer_coeffs: Optional[float] = None,
                 seed: Optional[int] = None):

        self.param_type = param_type
        self.init_type = init_type
        self.qubit_register = qubit_register
        self.p = p
        self.q = q if param_type.lower() in ['fourier','fourier_extended', 'fourier_w_bias'] else None
        self.variational_params_dict = variational_params_dict
        self.annealing_time = annealing_time if annealing_time is not None else 0.7*self.p
        self.linear_ramp_time = linear_ramp_time if linear_ramp_time is not None else 0.7*self.p
        self.mixer_hamiltonian = mixer_hamiltonian
        self.mixer_qubit_connectivity = mixer_qubit_connectivity
        self.mixer_coeffs = mixer_coeffs
        self.seed = seed

    @property
    def param_type(self):
        return self._param_type

    @param_type.setter
    def param_type(self, value):
        if value not in ALLOWED_PARAM_TYPES:
            raise ValueError(
                f"param_type {value} is not recognised. Please use {ALLOWED_PARAM_TYPES}")
        self._param_type = value

    @property
    def init_type(self):
        return self._init_type

    @init_type.setter
    def init_type(self, value):
        if value not in ALLOWED_INIT_TYPES:
            raise ValueError(
                f"init_type {value} is not recognised. Please use {ALLOWED_INIT_TYPES}")
        self._init_type = value

    @property
    def mixer_hamiltonian(self):
        return self._mixer_hamiltonian

    @mixer_hamiltonian.setter
    def mixer_hamiltonian(self, value):
        if value not in ALLOWED_MIXERS:
            raise ValueError(
                f"mixer_hamiltonian {value} is not recognised. Please use {ALLOWED_MIXERS}")
        self._mixer_hamiltonian = value

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if value <= 0:
            raise ValueError(
                f"Number of layers `p` cannot be smaller or equal to zero. Value {value} was provided")
        self._p = value

    @property
    def annealing_time(self):
        return self._annealing_time

    @annealing_time.setter
    def annealing_time(self, value):
        if value <= 0:
            raise ValueError(
                f"The annealing time `annealing_time` cannot be smaller or equal to zero. Value {value} was provided")
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

class BackendProperties(Parameters):
    """
    Choose the backend on which to run the QAOA circuits

    Parameters
    ----------
    device: `DeviceBase`
        The device to use for the backend.
    prepend_state: `Union[QuantumCircuitBase,np.ndarray(complex)]`
        The state prepended to the circuit.
    append_state: `Union[QuantumCircuitBase,np.ndarray(complex)]`
        The state appended to the circuit.
    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the 
        QAOA part of the circuit.
    n_shots: `int`
        The number of shots to be used for the shot-based computation.
    cvar_alpha: `float`
        The value of the CVaR parameter.
    noise_model: `NoiseModel`
        The noise model to be used for the shot-based simulator.
    qubit_layout: `Union[List[int], np.ndarray]`
        Mapping from physical to logical qubit indices, used to eventually 
        construct the quantum circuit.  For example, for a system composed by 3 qubits
       `qubit_layout=[1,3,2]`, maps `1<->0`, `3<->1`, `2<->2`, where the left hand side is the physical qubit 
        and the right hand side is the logical qubits
    """

    def __init__(self,
                 prepend_state: Optional[Union[QuantumCircuitBase,
                                               List[complex], np.ndarray]] = None,
                 append_state: Optional[Union[QuantumCircuitBase,
                                              np.ndarray]] = None,
                 init_hadamard: bool = True,
                 n_shots: int = 100,
                 cvar_alpha: float = 1,
                 noise_model = None,
                 qubit_layout: Optional[Union[List[int], np.ndarray]] = None):
        
        self.init_hadamard = init_hadamard
        self.n_shots = n_shots
        self.prepend_state = prepend_state
        self.append_state = append_state
        self.cvar_alpha = cvar_alpha
        self.noise_model = noise_model
        self.qubit_layout = qubit_layout

    # @property
    # def cvar_alpha(self):
    #     return self._cvar_alpha

    # @cvar_alpha.setter
    # def cvar_alpha(self, value):
    #     if (value <0) or (value>1) :
    #         raise ValueError(
    #             f"cvar_alpha must be between 0 and 1. Received {value}.")
    #     self._cvar_alpha = value


class ClassicalOptimizer(Parameters):
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
    jac: str
        Method to compute the gradient vector. Choose from:
        `['finite_difference', 'param_shift', 'stoch_param_shift', 'grad_spsa']       
    hess:
        Method to compute the hessian. Choose from:
        `['finite_difference', 'param_shift', 'stoch_param_shift', 'grad_spsa']
    constraints: scipy.optimize.LinearConstraints, scipy.optimize.NonlinearConstraints 
        Scipy-based constraints on parameters of optimization 
    bounds: scipy.scipy.optimize.Bounds
        Scipy-based bounds on parameters of optimization 
    tol : float
        Tolerance before the optimizer terminates; if `tol` is larger than
        the difference between two steps, terminate optimization.
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

    """

    def __init__(self,
                 optimize: bool = True,
                 method: str = 'cobyla',
                 maxiter: int = 100,
                 jac: str = None,
                 hess: str = None,
                 constraints=None,
                 bounds=None,
                 tol=None,
                 stepsize: float = None,
                 decay: float = None,
                 eps: float = None,
                 lambd: float = None,
                 jac_options: dict = None,
                 hess_options: dict = None,
                 optimization_progress: bool = False,
                 cost_progress: bool = True,
                 parameter_log: bool = True,
                 top_k_solutions: int = 1):
        self.optimize = optimize
        self.method = method.lower()
        self.maxiter = maxiter
        self.jac = jac.lower() if type(jac) == str else jac
        self.hess = hess.lower() if type(hess) == str else hess
        self.constraints = constraints
        self.bounds = bounds
        self.tol = tol
        self.stepsize = stepsize
        self.decay = decay
        self.eps = eps
        self.lambd = lambd
        self.jac_options = jac_options
        self.hess_options = hess_options
        self.parameter_log = parameter_log
        self.optimization_progress = optimization_progress
        self.cost_progress = cost_progress
        self.parameter_log = parameter_log
        self.top_k_solutions = top_k_solutions

    # @property
    # def method(self):
    #     return self._method

    # @method.setter
    # def method(self, value):
    #     if value not in ALLOWED_MINIMIZATION_METHODS:
    #         raise ValueError(
    #             f"method `{value}` is not supported. Please choose between {ALLOWED_MINIMIZATION_METHODS}")
    #     self._method = value


class ExtraResults(Parameters):
    """
    The classical optimizer for the QAOA optimization routine 
    of the QAOA circuit parameters.

    Parameters
    ----------
    extra_results: int
        Whether to perform optimization routine on the given QAOA problem
    """

    def __init__(self,
                 top_k_solutions: int = 1):
        self.top_k_solutions = top_k_solutions
