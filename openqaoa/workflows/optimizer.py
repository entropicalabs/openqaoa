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
import numpy as np
from openqaoa.devices import DeviceLocal, DeviceBase

from openqaoa.rqaoa.rqaoa import custom_rqaoa
from openqaoa.problems.problem import QUBO
from openqaoa.problems.helper_functions import convert2serialize
from openqaoa.workflows.parameters.qaoa_parameters import CircuitProperties, BackendProperties, ClassicalOptimizer, ExtraResults
from openqaoa.workflows.parameters.rqaoa_parameters import RqaoaParameters
from openqaoa.qaoa_parameters import Hamiltonian, QAOACircuitParams, create_qaoa_variational_params
from openqaoa.utilities import get_mixer_hamiltonian
from openqaoa.backends.qaoa_backend import get_qaoa_backend, DEVICE_NAME_TO_OBJECT_MAPPER, DEVICE_ACCESS_OBJECT_MAPPER
from openqaoa.optimizers.qaoa_optimizer import get_optimizer
from openqaoa.rqaoa import adaptive_rqaoa, custom_rqaoa


class Optimizer(ABC):
    """
    Abstract class to represent an optimizer
    """

    def asdict(self):
        pass


class QAOA(Optimizer):
    """
    A class implementing a QAOA workflow end to end.

    It's basic usage consists of 
    1. Initialization
    2. Compilation
    3. Optimization

    .. warning::
        To all our dear beta testers: the setter functions will most likely change. Bear with us as we figure our the smoother way to create the workflows :-)


    .. note::
        The attributes of the QAOA class should be initialized using the set methods of QAOA. For example, to set the circuit's depth to 10 you should run `set_circuit_properties(p=10)`

    Attributes
    ----------
        circuit_properties: `CircuitProperties`
            The circuit properties of the QAOA workflow. Use to set depth `p`, choice of parametrisation, parameter initialisation strategies, mixer hamiltonians.
            For a complete list of its parameters and usage please see the method set_circuit_properties
        backend_properties: `BackendProperties`
            The backend properties of the QAOA workflow. Use to set the backend properties such as the number of shots and the cvar values.
            For a complete list of its parameters and usage please see the method set_backend_properties
        classical_optimizer: `ClassicalOptimizer`
            The classical optimiser properties of the QAOA workflow. Use to set the classical optimiser needed for the classical optimisation part of the QAOA routine.
            For a complete list of its parameters and usage please see the method set_classical_optimizer
        local_simulators: list[str]`
            A list containing the available local simulators
        mixer_hamil: Hamiltonian
            The desired mixer hamiltonian
        cost_hamil: Hamiltonian
            The desired mixer hamiltonian
        circuit_params: QAOACircuitParams
            the abstract and backend-agnostic representation of the underlying QAOA parameters
        variate_params: QAOAVariationalBaseParams
            The variational parameters. These are the parameters to be optimised by the classical optimiser
        backend: VQABaseBackend
            The openQAOA representation of the backend to be used to execute the quantum circuit
        optimizer: OptimizeVQA
            The classical optimiser
        results: `Result`
            Contains the logs of the optimisation process
        compiled: `Bool`
            A boolean flag to check whether the QAOA object has been correctly compiled at least once

    Examples
    --------
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> q = QAOA()
    >>> q.compile(QUBO)
    >>> q.optimise()

    Where `QUBO` is a an instance of `openqaoa.problems.problem.QUBO`

    If you want to use non-default parameters:

    >>> q_custom = QAOA()
    >>> q_custom.set_circuit_properties(p=10, param_type='extended', init_type='ramp', mixer_hamiltonian='x')
    >>> q_custom.set_device_properties(device_location='qcs', device_name='Aspen-11', cloud_credentials={'name' : "Aspen11", 'as_qvm':True, 'execution_timeout' : 10, 'compiler_timeout':10})
    >>> q_custom.set_backend_properties(n_shots=200, cvar_alpha=1)
    >>> q_custom.set_classical_optimizer(method='nelder-mead', maxiter=2)
    >>> q_custom.compile(qubo_problem)
    >>> q_custom.optimize()
    """

    def __init__(self, device=DeviceLocal('vectorized')):
        self.circuit_properties = CircuitProperties()
        self.device = device
        self.backend_properties = BackendProperties()
        self.classical_optimizer = ClassicalOptimizer()
        self.local_simulators = list(DEVICE_NAME_TO_OBJECT_MAPPER.keys())
        self.cloud_provider = list(DEVICE_ACCESS_OBJECT_MAPPER.keys())
        self.compiled = False

    def set_device(self, device: DeviceBase):
        """"
        Specify the device to be used by the QAOA.

        Parameters
        ----------
        location: `str`
            Can be either local, qcs, or ibmq
        name: `str`
            The name of the device to be used, for local simulators please refer to `q.local_simulators`.
            For cloud providers please refer to the provider's naming conventions
        """
        self.device = device

    def set_circuit_properties(self, **kwargs):
        """
        Specify the circuit properties to construct QAOA circuit

        Parameters
        -------------------
            qubit_register: `list`
                Select the desired qubits to run the QAOA program. Meant to be used as a qubit
                selector for qubits on a QPU. Defaults to a list from 0 to n-1 (n = number of qubits)
            p: `int`
                Depth `p` of the QAOA circuit
            q: `int`
                Analogue of `p` of the QAOA circuit in the Fourier parameterisation
            param_type: `str`
                Choose the QAOA circuit parameterisation. Currently supported parameterisations include:
                `'standard'`: Standard QAOA parameterisation
                `'standard_w_bias'`: Standard QAOA parameterisation with a separate parameter for single-qubit terms.
                `'extended'`: Individual parameter for each qubit and each term in the Hamiltonian.
                `'fourier'`: Fourier circuit parameterisation
                `'fourier_extended'`: Fourier circuit parameterisation with individual parameter for each qubit and term in Hamiltonian.
                `'fourier_w_bias'`: Fourier circuit parameterisation with aseparate parameter for single-qubit terms
            init_type: `str`
                Initialisation strategy for the QAOA circuit parameters. Allowed init_types:
                `'rand'`: Randomly initialise circuit parameters
                `'ramp'`: Linear ramp from Hamiltonian initialisation of circuit parameters (inspired from Quantum Annealing)
                `'custom'`: User specified initial circuit parameters
            mixer_hamiltonian: `str`
                Parameterisation of the mixer hamiltonian:
                `'x'`: Randomly initialise circuit parameters
                `'xy'`: Linear ramp from Hamiltonian initialisation of circuit 
            mixer_qubit_connectivity: `[Union[List[list],List[tuple], str]]`
                The connectivity of the qubits in the mixer Hamiltonian. Use only if `mixer_hamiltonian = xy`.
            annealing_time: `float`
                Total time to run the QAOA program in the Annealing parameterisation (digitised annealing)
            ramp_time: `float`
                The slope(rate) of linear ramp initialisation of QAOA parameters.
            trainable_params_dict: `dict`
                Dictionary object specifying the initial value of each circuit parameter for the chosen parameterisation, if the `init_type` is selected as `'custom'`.    
        """
        for key, value in kwargs.items():
            if hasattr(self.circuit_properties, key):
                pass
            else:
                raise ValueError(
                    "Specified argument is not supported by the circuit")
        self.circuit_properties = CircuitProperties(**kwargs)

        return None

    def set_backend_properties(self, **kwargs):
        """
        Set the backend properties

        Parameters
        -------------------
            device: DeviceBase
            prepend_state: [Union[QuantumCircuitBase,List[complex], np.ndarray]
                The state prepended to the circuit.
            append_state: [Union[QuantumCircuitBase,List[complex], np.ndarray]
                The state prepended to the circuit.
            init_hadamard: bool
            Whether to apply a Hadamard gate to the beginning of the 
                QAOA part of the circuit.. Defaults to `True`
            n_shots: int
            Optional argument to specify the number of shots required to run QAOA computations
                on shot-based simulators and QPUs. Defaults to 100.
            cvar_alpha: float
                The value of alpha for the CVaR cost function
            qiskit_simulation_method: str, optional
                The method to be used for the simulation.
            noise_model: `qiskit.providers.aer.noise.NoiseModel`
                    The Qiskit noise model to be used for the simulation.
            active_reset:
                #TODO
            rewiring:
                Rewiring scheme to be used for Pyquil. 
                Either 'PRAGMA INITIAL_REWIRING "NAIVE"' or 
                'PRAGMA INITIAL_REWIRING "PARTIAL"'. If None, defaults to NAIVE
        """

        for key, value in kwargs.items():
            if hasattr(self.backend_properties, key):
                pass# setattr(self.backend_properties, key, value)
            else:
                raise ValueError(
                    f'Specified argument `{value}` for `{key}` in set_backend_properties is not supported')

        self.backend_properties = BackendProperties(**kwargs)
        return None

    def set_classical_optimizer(self, **kwargs):
        """
        Set the parameters for the classical optimizer to be used in the QAOA workflow

        Parameters
        -------------------
            method: str
                The classical optimization method. Choose from:
                ['imfil','bobyqa','snobfit']
                ['vgd', 'sgd', 'rmsprop'] 
                ['nelder-mead','powell','cg','bfgs','newton-cg','l-bfgs-b','cobyla'] 
            maxiter : Optional[int]
                Maximum number of iterations.
            jac: str
                Method to compute the gradient vector. Choose from:
                ['finite_difference', 'param_shift', 'stoch_param_shift', 'grad_spsa']        
            hess: str
                Method to compute the hessian. Choose from:
                ['finite_difference', 'param_shift', 'stoch_param_shift', 'grad_spsa']
            constraints: scipy.optimize.LinearConstraints, scipy.optimize.NonlinearConstraints  
                Scipy-based constraints on parameters of optimization. Will be available soon
            bounds: scipy.optimize.Bounds
                Scipy-based bounds on parameters of optimization. Will be available soon
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
                Small number to prevent singularity of QFIM matrix for Natural Gradient Descent.
            ramp_time: float
                The slope(rate) of linear ramp initialisation of QAOA parameters.
            jac_options : dict
                Dictionary that specifies gradient-computation options according to method chosen in 'jac'.
            hess_options : dict
                Dictionary that specifies Hessian-computation options according to method chosen in 'hess'.
        """
        for key, value in kwargs.items():
            if hasattr(self.classical_optimizer, key):
                pass #setattr(self.classical_optimizer, key, value)
            else:
                raise ValueError(
                    'Specified argument is not supported by the Classical Optimizer')

        self.classical_optimizer = ClassicalOptimizer(**kwargs)
        return None

    def compile(self, problem: QUBO = None, verbose: bool = False):
        """
        Initialise the trainable parameters for QAOA according to the specified
        strategies and by passing the problem statement

        .. note::
            Compilation is necessary because it is the moment where the problem statement and the QAOA instructions are used to build the actual QAOA circuit.

        .. tip::
            Set Verbose to false if you are running batch computations! 

        Parameters
        ----------
        problem: `Problem`
            QUBO problem to be solved by QAOA
        verbose: bool
            Set True to have a summary of QAOA to displayed after compilation
        """
        self.mixer_hamil = get_mixer_hamiltonian(n_qubits=problem.n,
                                                 mixer_type=self.circuit_properties.mixer_hamiltonian,
                                                 qubit_connectivity=self.circuit_properties.mixer_qubit_connectivity,
                                                 coeffs=self.circuit_properties.mixer_coeffs)

        self.cost_hamil = Hamiltonian.classical_hamiltonian(
            terms=problem.terms, coeffs=problem.weights, constant=problem.constant)

        self.circuit_params = QAOACircuitParams(
            self.cost_hamil, self.mixer_hamil, p=self.circuit_properties.p)
        self.variate_params = create_qaoa_variational_params(qaoa_circuit_params=self.circuit_params,
                                                             params_type=self.circuit_properties.param_type,
                                                             init_type=self.circuit_properties.init_type, 
                                                             variational_params_dict=self.circuit_properties.variational_params_dict,
                                                             linear_ramp_time=self.circuit_properties.linear_ramp_time, 
                                                             q=self.circuit_properties.q, 
                                                             seed=self.circuit_properties.seed)

        self.backend = get_qaoa_backend(circuit_params=self.circuit_params,
                                        device=self.device,
                                        **self.backend_properties.__dict__)

        self.optimizer = get_optimizer(vqa_object=self.backend,
                                       variational_params=self.variate_params,
                                       optimizer_dict=self.classical_optimizer.asdict())

        self.compiled = True

        if verbose:
            print('\t \033[1m ### Summary ###\033[0m')
            print(f'OpenQAOA has ben compiled with the following properties')
            print(
                f'Solving QAOA with \033[1m {self.device.device_name} \033[0m on  \033[1m{self.device.device_location}\033[0m')
            print(f'Using p={self.circuit_properties.p} with {self.circuit_properties.param_type} parameters initialsied as {self.circuit_properties.init_type}')

            if self.device.device_name == 'vectorized':
                print(
                    f'OpenQAOA will optimize using \033[1m{self.classical_optimizer.method}\033[0m, with up to \033[1m{self.classical_optimizer.maxiter}\033[0m maximum iterations')

            else:
                print(
                    f'OpenQAOA will optimize using \033[1m{self.classical_optimizer.method}\033[0m, with up to \033[1m{self.classical_optimizer.maxiter}\033[0m maximum iterations. Each iteration will contain \033[1m{self.backend_properties.n_shots} shots\033[0m')
                print(
                    f'The total numner of shots is set to maxiter*shots = {self.classical_optimizer.maxiter*self.backend_properties.n_shots}')

        return None

    def optimize(self, verbose=False):
        '''
        A method running the classical optimisation loop
        '''

        if self.compiled == False:
            raise ValueError('Please compile the QAOA before optimizing it!')

        self.optimizer.optimize()
        # TODO: results and qaoa_results will differ
        self.results = self.optimizer.qaoa_result

        if verbose:
            print(f'optimization completed.')
        return


class RQAOA(Optimizer):
    """
    RQAOA optimizer class.

    Attributes
    ----------
    algorithm: `str`
        A string contaning the name of the algorithm, here fixed to `rqaoa`
    qaoa: 'QAOA'
        QAOA class instance containing all the relevant information for the
        QAOA runs at each recursive step.
    rqaoa_parameters: 'RqaoaParameters'
        Set of parameters containing all the relevant information for the 
        recursive procedure.
    result: `dict`
        Dictionary containing the solution and all of the RQAOA procedure 
        information
    """

    def __init__(self, rqaoa_type: str = 'adaptive', qaoa: QAOA = QAOA()):
        """
        Initializes the RQAOA optimizer class.

        Parameters
        ----------
        rqaoa_type: `str`
            Recursive scheme for RQAOA algorithm. Choose from `custom` or `adaptive`
        q: `QAOA`
            QAOA instance specificying how QAOA is run within RQAOA
        """
        self.algorithm = 'rqaoa'
        self.qaoa = qaoa
        self.rqaoa_parameters = RqaoaParameters(rqaoa_type=rqaoa_type)
        self.rqaoa_mixer = {'type': self.qaoa.circuit_properties.mixer_hamiltonian,
                            'connectivity': self.qaoa.circuit_properties.mixer_qubit_connectivity,
                            'coeffs': self.qaoa.circuit_properties.mixer_coeffs}

    def set_rqaoa_parameters(self, **kwargs):
        """
        Sets the parameters for the RQAOA class.
        """
        for key, value in kwargs.items():
            if hasattr(self.rqaoa_parameters, key):
                setattr(self.rqaoa_parameters, key, value)
            else:
                raise ValueError(
                    f'Specified part {key},  {value} is not supported by RQAOA')

        return None

    def compile(self, problem: QUBO = None, verbose: bool = True):
        """
        Initialize the trainable parameters for QAOA according to the specified
        strategies and by passing the problem statement.

        Parameters
        ----------
        problem: `Problem`
            QUBO problem to be solved by RQAOA
        verbose: bool
            !NotYetImplemented! Set true to have a summary of RQAOA to displayed after compilation
        """

        self.qaoa.compile(problem, verbose=False)

    def optimize(self):
        """
        Performs optimization using RQAOA with the `custom` method or the `adaptive` method.
        """
        if self.rqaoa_parameters.rqaoa_type == 'adaptive':
            res = adaptive_rqaoa(
                hamiltonian=self.qaoa.cost_hamil,
                mixer=self.rqaoa_mixer,
                p=self.qaoa.circuit_properties.p,
                n_max=self.rqaoa_parameters.n_max,
                n_cutoff=self.rqaoa_parameters.n_cutoff,
                device=self.qaoa.device,
                params_type=self.qaoa.circuit_properties.param_type,
                init_type=self.qaoa.circuit_properties.init_type,
                optimizer_dict=self.qaoa.classical_optimizer.asdict(),
                backend_properties=self.qaoa.backend_properties.__dict__)
        elif self.rqaoa_parameters.rqaoa_type == 'custom':
            res = custom_rqaoa(
                hamiltonian=self.qaoa.cost_hamil,
                mixer=self.rqaoa_mixer,
                p=self.qaoa.circuit_properties.p,
                n_cutoff=self.rqaoa_parameters.n_cutoff,
                steps=self.rqaoa_parameters.steps,
                device=self.qaoa.device,
                params_type=self.qaoa.circuit_properties.param_type,
                init_type=self.qaoa.circuit_properties.init_type,
                optimizer_dict=self.qaoa.classical_optimizer.asdict(),
                backend_properties=self.qaoa.backend_properties.__dict__)
        else:
            raise f'rqaoa_type {self.rqaoa_parameters.rqaoa_type} is not supported. Please selet either "adaptive" or "custom'

        self.result = res
        return None

    def asdict(self):
        attributes_dict = convert2serialize(self)
        return attributes_dict
