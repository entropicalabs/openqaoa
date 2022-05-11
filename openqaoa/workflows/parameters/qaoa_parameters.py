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
from pyquil.api._engagement_manager import EngagementManager
from qcs_api_client.client import QCSClientConfiguration
import numpy as np

from openqaoa.basebackend import QuantumCircuitBase
from openqaoa.backends.qpus.qpu_auth import AccessObjectBase, AccessObjectQiskit, AccessObjectPyQuil
from openqaoa.basebackend import QuantumCircuitBase
from .parameters import Parameters
from scipy.optimize._minimize import MINIMIZE_METHODS


ALLOWED_PARAM_TYPES = ['standard', 'standard_w_bias',
                       'extended', 'fourier', 'fourier_extended', 'fourier_w_bias']
ALLOWED_INIT_TYPES = ['rand', 'ramp', 'custom']
ALLOWED_MIXERS = ['x', 'xy']
ALLOWED_MINIMIZATION_METHODS = MINIMIZE_METHODS

ALLOWED_QVM_DEVICES = ['Aspen-11', 'Aspen-M-1', '2q-qvm', '3q-qvm', '4q-qvm', '5q-qvm', '6q-qvm', '7q-qvm', '8q-qvm', '9q-qvm', '10q-qvm', '11q-qvm', '12q-qvm', '13q-qvm', '14q-qvm', '15q-qvm',
                       '16q-qvm', '17q-qvm', '18q-qvm', '19q-qvm', '20q-qvm', '21q-qvm', '22q-qvm', '23q-qvm', '24q-qvm', '25q-qvm', '26q-qvm', '27q-qvm', '28q-qvm', '29q-qvm', '30q-qvm', '31q-qvm', '32q-qvm', '33q-qvm', '34q-qvm', '35q-qvm', '36q-qvm', '37q-qvm', '38q-qvm', '39q-qvm', '40q-qvm', '41q-qvm', '42q-qvm', '43q-qvm', '44q-qvm', '45q-qvm',
                       '46q-qvm', '47q-qvm', '48q-qvm', '49q-qvm', '50q-qvm', '51q-qvm', '52q-qvm', '53q-qvm', '54q-qvm', '55q-qvm', '56q-qvm', '57q-qvm', '58q-qvm', '59q-qvm', '60q-qvm',
                       '61q-qvm', '62q-qvm', '63q-qvm', '64q-qvm', '65q-qvm', '66q-qvm', '67q-qvm', '68q-qvm', '69q-qvm', '70q-qvm', '71q-qvm', '72q-qvm', '73q-qvm', '74q-qvm', '75q-qvm', '76q-qvm', '77q-qvm', '78q-qvm', '79q-qvm']
ALLOWED_LOCAL_SIMUALTORS = ['qiskit_shot_simulator', 'qiskit_statevec_simulator', 'qiskit_qasm_simulator', 'vectorized']
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
        self.q = q if param_type.lower() == 'fourier' else None
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


class Credentials(Parameters):
    """
    The general Credentials class for different QPU services
    """

    def __init__(self,
                 qpu_credentials: Dict = None):
        # TODO: once the credential code is ironed out put some proper try/except blocks
        if qpu_credentials['device_location'].lower() == 'ibmq':
            qpu_credentials.pop('device_location')
            # try:
            self.accessObject = IBMQCredentials(**qpu_credentials).credentials
            # except:
            #     raise ValueError(f"Cloud credentials {qpu_credentials} be wrong?")
        elif qpu_credentials['device_location'].lower() == 'qcs':
            # try:
            qpu_credentials.pop('device_location')
            self.accessObject = QCSCredentials(**qpu_credentials).credentials
            # except:
            #     raise ValueError("Could cloud credentials {qpu_credentials} be wrong?")
        else:
            pass


class IBMQCredentials(Parameters):
    """
    Implement IBMQ specific credentials
    """

    def __init__(self,
                 api_token: str,
                 hub: str = None,
                 group: str = None,
                 project: str = None,
                 selected_qpu: str = None,
                 provider: str = 'ibmq'):

        self.credentials = AccessObjectQiskit(
            api_token=api_token, hub=hub, group=group, project=project, selected_qpu=selected_qpu)


class QCSCredentials(Parameters):
    """
    Implement QCS specific credentials
    """

    def __init__(self,
                 name: str,
                 as_qvm: bool = None, noisy: bool = None,
                 compiler_timeout: float = 20.0,
                 execution_timeout: float = 20.0,
                 client_configuration: QCSClientConfiguration = None,
                 endpoint_id: str = None,
                 engagement_manager: EngagementManager = None):

        self.credentials = AccessObjectPyQuil(
            name, as_qvm=as_qvm, compiler_timeout=compiler_timeout, execution_timeout=execution_timeout,
            client_configuration=client_configuration, endpoint_id=endpoint_id, engagement_manager=engagement_manager
        )


class BackendProperties(Parameters):
    """
    Choose the backend on which to run the QAOA circuits
    """

    def __init__(self,
                 prepend_state: Optional[Union[QuantumCircuitBase,
                                               List[complex], np.ndarray]] = None,
                 append_state: Optional[Union[QuantumCircuitBase,
                                              List[complex], np.ndarray]] = None,
                 init_hadamard: bool = True,
                 n_shots: int = 100,
                 cvar_alpha: float = 1):
        """
        Parameters:
            provider: str
                The provider of the backend
            cost_std_dev: bool
                Return std dev of cost value along with cost value at each iteration
                **Only for simulators**
            init_hadamard: bool
                Set to `False`, to start the QAOA circuit without the initial Hadamard
                superposition
            init_prog: Optional[AbstractParams]
                Start the QAOA circuit in a pre-defined QAOA-like state, specified via 
                QAOA `AbstractParams`
            shots: Optional[int]
                If using, a `shot-based simulator` or a `QPU, specify the number of shots
                for each circuit execution
            credentials: Optional[Credentials]
                Provide, the service credentials for using `QPUs` offered by that service
                provider. 
        """

        self.init_hadamard = init_hadamard
        self.n_shots = n_shots
        self.prepend_state = prepend_state
        self.append_state = append_state
        self.cvar_alpha = cvar_alpha

    # @property
    # def cvar_alpha(self):
    #     return self._cvar_alpha

    # @cvar_alpha.setter
    # def cvar_alpha(self, value):
    #     if (value <0) or (value>1) :
    #         raise ValueError(
    #             f"cvar_alpha must be between 0 and 1. Received {value}.")
    #     self._cvar_alpha = value


class DeviceProperties(Parameters):
    """
    Choose the device properties.
    """

    def __init__(self,
                 device_location: str = 'local',
                 device_name: str = 'vectorized',
                 cloud_credentials: Optional[dict] = {},
                 device: Union[str, AccessObjectBase] = 'vectorized'):

        self.device_location = device_location
        self.device_name = device_name
        self.cloud_credentials = cloud_credentials
        self.device = device

    @property
    def device_location(self):
        return self._device_location

    @device_location.setter
    def device_location(self, value):
        if value not in ['local', 'ibmq', 'qcs']:
            raise ValueError(
                f"Device location {value} is not recognised. Please use ['local', 'ibmq', 'qcs']")
        self._device_location = value

    @property
    def device_name(self):
        return self._device_name

    @device_name.setter
    def device_name(self, value):
        if value not in ALLOWED_DEVICES:
            raise ValueError(
                f"Device name {value} is not recognised. Please use {ALLOWED_DEVICES}")
        self._device_name = value


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
        `['finite_difference', 'param-shift', 'stoch_param_shift', 'grad_spsa']        
    hess:
        Method to compute the hessian. Choose from:
        `['finite_difference', 'param-shift', 'stoch_param_shift', 'grad_spsa']
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
