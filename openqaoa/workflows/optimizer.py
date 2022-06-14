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
from openqaoa.problems.problem import PUBO
from openqaoa.problems.helper_functions import convert2serialize
from openqaoa.workflows.parameters.qaoa_parameters import CircuitProperties, BackendProperties, ClassicalOptimizer, ExtraResults
from openqaoa.workflows.parameters.rqaoa_parameters import RqaoaParameters
from openqaoa.qaoa_parameters import Hamiltonian, QAOACircuitParams, create_qaoa_variational_params
from openqaoa.utilities import get_mixer_hamiltonian
from openqaoa.backends.qaoa_backend import get_qaoa_backend, DEVICE_NAME_TO_OBJECT_MAPPER
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

    It's basic usage cosinsts of 
    1. Initialisation
    2. Compilation
    3. Optimisation

    .. warning::
        To all our dear beta testers: the setter functions will most likely change. Bear with us as we figure our the smoother way to create the workflows :-)
        

    .. note::
        The attributes of the QAOA class should be initialised using the set methods of QAOA. For example, to set the circuit's depth to 10 you should run `set_circuit_properties(p=10)`

    Attributes
    ----------
        circuit_properties: CircuitProperties
            The circut properties of the QAOA workflow. Use to set depth `p`, choice of parametrisation, parameter initialisation strategies, mixer hamiltonians.
            For a complete list of its parameters and usage please see the method set_circuit_properties
        backend_properties: BackendProperties
            The backend properties of the QAOA workflow. Use to set the backend properties such as the number of shots and the cvar values.
            For a complete list of its parameters and usage please see the method set_backend_properties
        classical_optimizer: ClassicalOptimizer
            The classical optimiser properties of the QAOA workflow. Use to set the classical optimiser needed for the classical optimisation part of the QAOA routine.
            For a complete list of its parameters and usage please see the method set_classical_optimizer
        intialised_w_prob: bool
            Deprecated feature: TOBE REMOVED
        local_simulators: list[str]
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
        results_information: dict
            A dictonary containg the logs of the optimiser
        solution: list
            A list providing the most probable bitstring

    Examples
    --------
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> q = QAOA()
    >>> q.compile(PUBO)
    >>> q.optimise()

    Where `PUBO` is a an isntance of `openqaoa.problems.problem.PUBO`

    If you want to use non-default parameters:

    >>> q_custom = QAOA()
    >>> q_custom.set_circuit_properties(p=10, param_type='extended', init_type='ramp', mixer_hamiltonian='x')
    >>> q_custom.set_device_properties(device_location='qcs', device_name='Aspen-11', cloud_credentials={'name' : "Aspen11", 'as_qvm':True, 'execution_timeout' : 10, 'compiler_timeout':10})
    >>> q_custom.set_backend_properties(n_shots=200, cvar_alpha=1)
    >>> q_custom.set_classical_optimizer(method='nelder-mead', maxiter=2)
    >>> q_custom.compile(pubo_problem)
    >>> q_custom.optimize()
    """
    def __init__(self, device = DeviceLocal('vectorized')):
        self.circuit_properties = CircuitProperties()
        self.device = device
        self.backend_properties = BackendProperties()
        self.classical_optimizer = ClassicalOptimizer()
        self.intialised_w_prob = False
        self.local_simulators = list(DEVICE_NAME_TO_OBJECT_MAPPER.keys())

    def set_device(self, device: DeviceBase):
        self.device = device
        
    def set_circuit_properties(self, **kwargs):
        """
        Specify the circuit properties to construct QAOA circuit

        Parameters
        -------------------
            qubit_register: list
                Select the desired qubits to run the QAOA program. Meant to be used as a qubit
                selector for qubits on a QPU. Defaults to a list from 0 to n-1 (n = number of qubits)
            p: int
                Depth `p` of the QAOA circuit
            q: int
                Analogue of `p` of the QAOA circuit in the Fourier parameterisation
            param_type: `str`
                Choose the QAOA circuit parameterisation. Currently supported parameterisations include:
                `'standard'`: Standard QAOA parameterisation
                `'standard_w_bias'`: Standard QAOA parameterisation with a separate parameter for single-qubit terms.
                `'extended'`: Individual parameter for each qubit and each term in the Hamiltonian.
                `'fourier'`: Fourier circuit parameterisation
                `'fourier_extended'`: Fourier circuit parameterisation with individual parameter for each qubit and term in Hamiltonian.
                `'fourier_w_bias'`: Fourier circuit parameterisation with aseparate parameter for single-qubit terms
            init_type: str
                Initialisation strategy for the QAOA circuit parameters. Allowed init_types:
                `'rand'`: Randomly initialise circuit parameters
                `'ramp'`: Linear ramp from Hamiltonian initialisation of circuit parameters (inspired from Quantum Annealing)
                `'custom'`: User specified initial circuit parameters
            mixer_hamiltonian: str
                Parameterisation of the mixer hamiltonian:
                `'x'`: Randomly initialise circuit parameters
                `'xy'`: Linear ramp from Hamiltonian initialisation of circuit 
            annealing_time: `float`
                Total time to run the QAOA program in the Annealing parameterisation (digitised annealing)
            ramp_time: `float`
                The slope(rate) of linear ramp initialisation of QAOA parameters.
            trainable_params_dict: `dict`
                Dictionary object specifying the initial value of each circuit parameter for the chosen parameterisation, if the `init_type` is selected as `'custom'`.    
        """
        if self.intialised_w_prob:
            raise AttributeError(
                "Cannot change circuit properties after initialisation with the problem")
        else:
            for key, value in kwargs.items():
                if hasattr(self.circuit_properties, key):
                    setattr(self.circuit_properties, key, value)
                else:
                    raise ValueError(
                        "Specified argument is not supported by the circuit")

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
                setattr(self.backend_properties, key, value)
            else:
                raise ValueError(
                    f'Specified argument `{value}` for `{key}` in set_backend_properties is not supported')

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
                ['finite_difference', 'param-shift', 'stoch_param_shift', 'grad_spsa']        
            hess: str
                Method to compute the hessian. Choose from:
                ['finite_difference', 'param-shift', 'stoch_param_shift', 'grad_spsa']
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
                setattr(self.classical_optimizer, key, value)
            else:
                raise ValueError(
                    'Specified argument is not supported by the Classical Optimizer')

        return None

    def asdict(self):
        attributes_dict = convert2serialize(self)
        return attributes_dict

    def compile(self, problem: PUBO = None, verbose: bool = True):
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
                                                 qubit_connectivity=self.circuit_properties.mixer_qubit_connectivity,
                                                 coeffs=self.circuit_properties.mixer_coeffs)

        self.cost_hamil = Hamiltonian.classical_hamiltonian(
            terms=problem.terms, coeffs=problem.weights, constant=problem.constant)

        self.circuit_params = QAOACircuitParams(
            self.cost_hamil, self.mixer_hamil, p=self.circuit_properties.p)
        self.variate_params = create_qaoa_variational_params(qaoa_circuit_params=self.circuit_params, params_type=self.circuit_properties.param_type,
                                                             init_type=self.circuit_properties.init_type, variational_params_dict=self.circuit_properties.variational_params_dict,
                                                             linear_ramp_time=self.circuit_properties.linear_ramp_time, q=self.circuit_properties.q, seed=self.circuit_properties.seed)

        self.backend = get_qaoa_backend(circuit_params=self.circuit_params,
                                        device = self.device,
                                        **self.backend_properties.__dict__)

        self.optimizer = get_optimizer(vqa_object=self.backend,
                                       variational_params=self.variate_params,
                                       optimizer_dict=self.classical_optimizer.asdict())

        if verbose:
            print('\t \033[1m ### Summary ###\033[0m')
            print(f'OpenQAOA has ben compiled with the following properties')
            print(
                f'Solving QAOA with \033[1m {self.device.device_name} \033[0m on  \033[1m{self.device.device_location}\033[0m')
            print(f'Using p={self.circuit_properties.p} with {self.circuit_properties.param_type} parameters initialsied as {self.circuit_properties.init_type}')
            

            if self.device.device_name == 'vectorized':
                print(f'OpenQAOA will optimize using \033[1m{self.classical_optimizer.method}\033[0m, with up to \033[1m{self.classical_optimizer.maxiter}\033[0m maximum iterations')
            
            else:
                print(f'OpenQAOA will optimize using \033[1m{self.classical_optimizer.method}\033[0m, with up to \033[1m{self.classical_optimizer.maxiter}\033[0m maximum iterations. Each iteration will contain \033[1m{self.backend_properties.n_shots} shots\033[0m')
                print(
                    f'The total numner of shots is set to maxiter*shots = {self.classical_optimizer.maxiter*self.backend_properties.n_shots}')

        return None

    def optimize(self):
        '''
        A method running the classical optimisation loop
        '''

        self.optimizer.optimize()
        self.results_information = self.optimizer.results_information()



        print(f'optimization completed.')
        return None


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
        # print(f'\n\nThe default device inside the class is {self.qaoa.device_properties.device}\n\n')
        self.rqaoa_parameters = RqaoaParameters(rqaoa_type=rqaoa_type)

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

    def compile(self, problem: PUBO = None, verbose: bool = True):
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
                mixer_hamiltonian=self.qaoa.mixer_hamil,
                p=self.qaoa.circuit_properties.p,
                n_max=self.rqaoa_parameters.n_max,
                n_cutoff=self.rqaoa_parameters.n_cutoff,
                device=self.qaoa.device,
                params_type=self.qaoa.circuit_properties.param_type,
                init_type=self.qaoa.circuit_properties.init_type,
                shots=self.qaoa.backend_properties.n_shots,
                optimizer_dict=self.qaoa.classical_optimizer.asdict())
        elif self.rqaoa_parameters.rqaoa_type == 'custom':
            res = custom_rqaoa(
                hamiltonian=self.qaoa.cost_hamil,
                mixer_hamiltonian=self.qaoa.mixer_hamil,
                p=self.qaoa.circuit_properties.p,
                n_cutoff=self.rqaoa_parameters.n_cutoff,
                steps= self.rqaoa_parameters.steps,
                device=self.qaoa.device,
                params_type=self.qaoa.circuit_properties.param_type,
                init_type=self.qaoa.circuit_properties.init_type,
                shots=self.qaoa.backend_properties.n_shots,
                optimizer_dict=self.qaoa.classical_optimizer.asdict())
        else:
            raise f'rqaoa_type {self.rqaoa_parameters.rqaoa_type} is not supported. Please selet either "adaptive" or "custom'

        self.result = res
        return None

    def asdict(self):
        attributes_dict = convert2serialize(self)
        return attributes_dict


def format_output_results(optimizer_results: dict, classical_optimizer: dict, top_k_solutions: int) -> dict:
    """
    Formats the output result. The user is able to control the information that is returned
    from the optimization process with the flags (count_progress, cost_progress
    and parameter_log).

    Returns
    -------
        output:
            Appropriately formatted results of the optimization.
    """

    if classical_optimizer['optimization_progress']:
        lowest_cost_index = np.argmin(optimizer_results['cost progress list'])

        if len(optimizer_results['count progress list']) != 0:
            final_state = optimizer_results['count progress list'][lowest_cost_index]
        elif len(optimizer_results['probability progress list']) != 0:
            final_state = optimizer_results['probability progress list'][lowest_cost_index]
    else:
        if len(optimizer_results['count progress list']) != 0:
            final_state = optimizer_results['count progress list'][0]
        elif len(optimizer_results['probability progress list']) != 0:
            final_state = optimizer_results['probability progress list'][0]

    solutions = sorted(final_state, key=final_state.get,
                       reverse=True)[:top_k_solutions]

    output_solution = [[int(i) for i in each_solution]
                       for each_solution in solutions]

    output = {"cost": optimizer_results['optimal cost'],
              "solution": output_solution,
              "function evals": optimizer_results['cost function calls'],
              "optimizer raw result": optimizer_results['opt result'],
              "optimal angles": optimizer_results['final params'],
              "optimization method": classical_optimizer['method'],
              "eqaoa_version": '0.0.1'}  # TO CHANGE

    if classical_optimizer['optimization_progress']:
        output.update(
            {"count progress": optimizer_results['count progress list']})
        output.update(
            {"probability progress": optimizer_results['probability progress list']})

    if classical_optimizer['cost_progress']:
        output.update(
            {"cost progress": optimizer_results['cost progress list']})

    if classical_optimizer['parameter_log']:
        output.update({"parameter log": optimizer_results['parameter log']})

    return output
