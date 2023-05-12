import time
import numpy as np

from .rqaoa_workflow_properties import RqaoaParameters
from ..baseworkflow import Workflow, check_compiled
from ..qaoa import QAOA
from ..workflow_properties import CircuitProperties
from ...backends.devices_core import DeviceLocal, DeviceBase
from ...problems import QUBO
from ...utilities import (
    ground_state_hamiltonian,
    exp_val_hamiltonian_termwise,
    generate_timestamp,
)
from ...backends.qaoa_analytical_sim import QAOABackendAnalyticalSimulator
from . import rqaoa_utils
from .rqaoa_result import RQAOAResult


class RQAOA(Workflow):
    """
    A class implementing a RQAOA workflow end to end.

    It's basic usage consists of
    1. Initialization
    2. Compilation
    3. Optimization

    .. note::
        The attributes of the RQAOA class should be initialized using the set methods of QAOA.
        For example, to set the qaoa circuit's depth to 10 you should run `set_circuit_properties(p=10)`

    Attributes
    ----------
    device: `DeviceBase`
        Device to be used by the optimizer
    backend_properties: `BackendProperties`
        The backend properties of the RQAOA workflow. These properties will be used to
        run QAOA at each RQAOA step. Use to set the backend properties such as the
        number of shots and the cvar values. For a complete list of its parameters and
        usage please see the method set_backend_properties
    classical_optimizer: `ClassicalOptimizer`
        The classical optimiser properties of the RQAOA workflow.
        Use to set the classical optimiser needed for the classical optimisation part of the QAOA routine.
        For a complete list of its parameters and usage please see the method set_classical_optimizer
    local_simulators: `list[str]`
        A list containing the available local simulators
    cloud_provider: `list[str]`
        A list containing the available cloud providers
    compiled: `Bool`
        A boolean flag to check whether the optimizer object has been correctly compiled at least once
    circuit_properties: `CircuitProperties`
        The circuit properties of the RQAOA workflow. These properties will be used to run QAOA at each RQAOA step.
        Use to set depth `p`, choice of parametrisation, parameter initialisation strategies, mixer hamiltonians.
        For a complete list of its parameters and usage please see the method set_circuit_properties
    rqaoa_parameters: `RqaoaParameters`
        Set of parameters containing all the relevant information for the recursive procedure of RQAOA.
    results: `RQAOAResult`
        The results of the RQAOA optimization.
        Dictionary containing all the information about the RQAOA run: the
        solution states and energies (key: 'solution'), the output of the classical
        solver (key: 'classical_output'), the elimination rules for each step
        (key: 'elimination_rules'), the number of eliminations at each step (key: 'schedule'),
        total number of steps (key: 'number_steps'), the intermediate QUBO problems and the
        intermediate QAOA objects that have been optimized in each RQAOA step (key: 'intermediate_problems').
        This object (`RQAOAResult`) is a dictionary with some custom methods as
        RQAOAResult.get_hamiltonian_step(i) which get the hamiltonian of reduced problem of the i-th step.
        To see the full list of methods please see the RQAOAResult class.

    Examples
    --------
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> r = RQAOA()
    >>> r.compile(QUBO)
    >>> r.optimise()

    Where `QUBO` is a an instance of `openqaoa.problems.problem.QUBO`

    If you want to use non-default parameters:

    Standard/custom (default) type:

    >>> r = RQAOA()
    >>> r.set_circuit_properties(
            p=10,
            param_type='extended',
            init_type='ramp',
            mixer_hamiltonian='x'
        )
    >>> r.set_device_properties(
            device_location='qcs',
            device_name='Aspen-11',
            cloud_credentials={
                'name' : "Aspen11",
                'as_qvm':True,
                'execution_timeout' : 10,
                'compiler_timeout':10
            }
        )
    >>> r.set_backend_properties(n_shots=200, cvar_alpha=1)
    >>> r.set_classical_optimizer(method='nelder-mead', maxiter=2)
    >>> r.set_rqaoa_parameters(n_cutoff = 5, steps=[1,2,3,4,5])
    >>> r.compile(qubo_problem)
    >>> r.optimize()

    Ada-RQAOA:

    >>> r_adaptive = RQAOA()
    >>> r.set_circuit_properties(
            p=10,
            param_type='extended',
            init_type='ramp',
            mixer_hamiltonian='x'
        )
    >>> r.set_device_properties(
            device_location='qcs',
            device_name='Aspen-11',
            cloud_credentials={
                'name' : "Aspen11",
                'as_qvm':True,
                'execution_timeout' : 10,
                'compiler_timeout':10
            }
        )
    >>> r_adaptive.set_backend_properties(n_shots=200, cvar_alpha=1)
    >>> r_adaptive.set_classical_optimizer(method='nelder-mead', maxiter=2)
    >>> r_adaptive.set_rqaoa_parameters(rqaoa_type = 'adaptive', n_cutoff = 5, n_max=5)
    >>> r_adaptive.compile(qubo_problem)
    >>> r_adaptive.optimize()
    """

    results_class = RQAOAResult

    def __init__(self, device: DeviceBase = DeviceLocal("vectorized")):
        """
        Initialize the RQAOA class.

        Parameters
        ----------
            device: `DeviceBase`
                Device to be used by the optimizer. Default is using the local 'vectorized' simulator.
        """
        super().__init__(device)  # use the parent class to initialize
        self.circuit_properties = CircuitProperties()
        self.rqaoa_parameters = RqaoaParameters()

        # change algorithm name to rqaoa
        self.header["algorithm"] = "rqaoa"

    @check_compiled
    def set_circuit_properties(self, **kwargs):
        """
        Specify the circuit properties to construct the QAOA circuits

        Parameters
        ----------
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
                `'standard_w_bias'`: Standard QAOA parameterisation with a separate parameter
                for single-qubit terms.
                `'extended'`: Individual parameter for each qubit and each term in the Hamiltonian.
                `'fourier'`: Fourier circuit parameterisation
                `'fourier_extended'`: Fourier circuit parameterisation with individual parameter
                for each qubit and term in Hamiltonian.
                `'fourier_w_bias'`: Fourier circuit parameterisation with a separate parameter
                for single-qubit terms
            init_type: `str`
                Initialisation strategy for the QAOA circuit parameters. Allowed init_types:
                `'rand'`: Randomly initialise circuit parameters
                `'ramp'`: Linear ramp from Hamiltonian initialisation of circuit parameters
                (inspired from Quantum Annealing)
                `'custom'`: User specified initial circuit parameters
            mixer_hamiltonian: `str`
                Parameterisation of the mixer hamiltonian:
                `'x'`: Randomly initialise circuit parameters
                `'xy'`: Linear ramp from Hamiltonian initialisation of circuit
            mixer_qubit_connectivity: `[Union[List[list],List[tuple], str]]`
                The connectivity of the qubits in the mixer Hamiltonian. Use only if `mixer_hamiltonian = xy`.
                The user can specify the connectivity as a list of lists, a list of tuples,
                or a string chosen from ['full', 'chain', 'star'].
            mixer_coeffs: `list`
                The coefficients of the mixer Hamiltonian. By default all set to -1
            annealing_time: `float`
                Total time to run the QAOA program in the Annealing parameterisation (digitised annealing)
            linear_ramp_time: `float`
                The slope(rate) of linear ramp initialisation of QAOA parameters.
            variational_params_dict: `dict`
                Dictionary object specifying the initial value of each circuit parameter for
                the chosen parameterisation, if the `init_type` is selected as `'custom'`.
                For example, for standard params set {'betas': [0.1, 0.2, 0.3], 'gammas': [0.1, 0.2, 0.3]}
        """

        for key in kwargs.keys():
            if hasattr(self.circuit_properties, key):
                pass
            else:
                raise ValueError(
                    f"Specified argument {key} is not supported by the circuit"
                )

        self.circuit_properties = CircuitProperties(**kwargs)

        return None

    @check_compiled
    def set_rqaoa_parameters(self, **kwargs):
        """
        Specify the parameters to run a desired RQAOA program.

        Parameters
        ----------
        rqaoa_type: `int`
            String specifying the RQAOA scheme under which eliminations are computed. The two methods are 'custom' and
            'adaptive'. Defaults to 'custom'.
        n_max: `int`
            Maximum number of eliminations allowed at each step when using the adaptive method.
        steps: `Union[list,int]`
            Elimination schedule for the RQAOA algorithm. If an integer is passed, it sets the number of spins eliminated
            at each step. If a list is passed, the algorithm will follow the list to select how many spins to eliminate
            at each step. Note that the list needs enough elements to specify eliminations from the initial number of qubits
            up to the cutoff value. If the list contains more, the algorithm will follow instructions until the cutoff value
            is reached.
        n_cutoff: `int`
            Cutoff value at which the RQAOA algorithm obtains the solution classically.
        original_hamiltonian: `Hamiltonian`
            Hamiltonian encoding the original problem fed into the RQAOA algorithm.
        counter: `int`
            Variable to count the step in the schedule. If counter = 3 the next step is schedule[3].
            Default is 0, but can be changed to start in the position of the schedule that one wants.
        """

        for key in kwargs.keys():
            if hasattr(self.rqaoa_parameters, key):
                pass
            else:
                raise ValueError(f"Specified argument {key} is not supported by RQAOA")

        self.rqaoa_parameters = RqaoaParameters(**kwargs)

        return None

    def compile(self, problem: QUBO = None, verbose: bool = False):
        """
        Create a QAOA object and initialize it with the circuit properties, device, classical optimizer and
        backend properties specified by the user.
        This QAOA object will be used to run QAOA changing the problem to sove at each RQAOA step.
        Here, the QAOA is compiled passing the problem statement, so to check that the compliation of
        QAOA is correct. See the QAOA class.

        .. note::
            Compilation is necessary because it is the moment where the problem statement and
            the QAOA instructions are used to build the actual QAOA circuit.

        Parameters
        ----------
        problem: `Problem`
            QUBO problem to be solved by RQAOA
        verbose: bool
            !NotYetImplemented! Set true to have a summary of QAOA first step to displayed after compilation
        """

        # we compile the method of the parent class to genereate the id
        # and check the problem is a QUBO object and save it
        super().compile(problem=problem)

        # if type is custom and steps is an int, set steps correctly
        if (
            self.rqaoa_parameters.rqaoa_type == "custom"
            and self.rqaoa_parameters.n_cutoff <= problem.n
        ):
            n_cutoff = self.rqaoa_parameters.n_cutoff
            n_qubits = problem.n
            counter = self.rqaoa_parameters.counter

            # If schedule for custom RQAOA is not given, we create a schedule such that
            # n = self.rqaoa_parameters.steps spins is eliminated at a time
            if type(self.rqaoa_parameters.steps) is int:
                self.rqaoa_parameters.steps = [self.rqaoa_parameters.steps] * (
                    n_qubits - n_cutoff
                )

            # In case a schedule is given, ensure there are enough steps in the schedule
            assert np.abs(n_qubits - n_cutoff - counter) <= sum(
                self.rqaoa_parameters.steps
            ), f"Schedule is incomplete, add {np.abs(n_qubits - n_cutoff - counter) - sum(self.rqaoa_parameters.steps)} more eliminations"

        # Create the qaoa object with the properties
        self.__q = QAOA(self.device)
        self.__q.circuit_properties = self.circuit_properties
        self.__q.backend_properties = self.backend_properties
        self.__q.classical_optimizer = self.classical_optimizer
        self.__q.exp_tags = self.exp_tags

        # set the header of the qaoa object to be the same as the header of the rqaoa object
        self.__q.header = self.header.copy()
        self.__q.header[
            "algorithm"
        ] = "qaoa"  # change the algorithm name to qaoa, since this is a qaoa object

        # compile qaoa object
        self.__q.compile(problem, verbose=verbose)

        # set compiled boolean to true
        self.compiled = True

        return

    def optimize(
        self, dump: bool = False, dump_options: dict = {}, verbose: bool = False
    ):
        """
        Performs optimization using RQAOA with the `custom` method or the `adaptive` method.
        The elimination RQAOA loop will occur until the number of qubits is equal to the number of qubits specified in `n_cutoff`.
        In each loop, the QAOA will be run, then the eliminations will be computed, a new problem will be redefined
        and the QAOA will be recompiled with the new problem.
        Once the loop is complete, the final problem will be solved classically and the final solution will be reconstructed.
        Results will be stored in the `results` attribute.

        Parameters
        ----------
        dump: `bool`
            If true, the object will be dumped to a file. And at the end of each step,
            the qaoa object will be dumped to a file.
            Default is False.
        dump_options: `dict`
            Dictionary containing the options for the dump. To see the options,
            see the `dump` method of the `QAOA` or `RQAOA` class.
            Default is empty.
        verbose: `bool`
            TODO
        """

        # check if the object has been compiled (or already optimized)
        assert (
            self.compiled
        ), "RQAOA object has not been compiled. Please run the compile method first."

        # lists to append the eliminations, the problems, the qaoa results objects,
        # the correlation matrix, the expectation values z and a dictionary for the atomic ids
        elimination_tracker = []
        q_results_steps = []
        problem_steps = []
        exp_vals_z_steps = []
        corr_matrix_steps = []
        atomic_id_steps = {}

        # get variables
        problem = self.problem
        problem_metadata = self.problem.metadata
        n_cutoff = self.rqaoa_parameters.n_cutoff
        n_qubits = problem.n
        counter = self.rqaoa_parameters.counter

        # get the qaoa object
        q = self.__q

        # create a different max_terms function for each type
        if self.rqaoa_parameters.rqaoa_type == "adaptive":
            f_max_terms = rqaoa_utils.ada_max_terms
        else:
            f_max_terms = rqaoa_utils.max_terms

        # timestamp for the start of the optimization
        self.header["execution_time_start"] = generate_timestamp()

        # flag, set to true if the problem vanishes due to elimination before reaching cutoff
        total_elimination = False

        # If above cutoff, loop quantumly, else classically
        while n_qubits > n_cutoff:
            # put a tag to the qaoa object to know which step it is
            q.set_exp_tags({"rqaoa_counter": counter})

            # Run QAOA
            q.optimize()

            # save the results if dump is true
            if dump:
                q.dump(**dump_options)

            # Obtain statistical results
            exp_vals_z, corr_matrix = self.__exp_val_hamiltonian_termwise(q)
            # Retrieve highest expectation values according to adaptive method or schedule in custom method
            max_terms_and_stats = f_max_terms(
                exp_vals_z, corr_matrix, self.__n_step(n_qubits, n_cutoff, counter)
            )
            # Generate spin map
            spin_map = rqaoa_utils.spin_mapping(problem, max_terms_and_stats)
            # Eliminate spins and redefine problem
            new_problem, spin_map = rqaoa_utils.redefine_problem(problem, spin_map)

            # In case eliminations cancel out the whole graph, break the loop before reaching the predefined cutoff.
            if new_problem == problem:
                total_elimination = True
                break

            # Extract final set of eliminations with correct dependencies and update tracker
            eliminations = [
                {"singlet": (spin,), "bias": spin_map[spin][0]}
                if spin_map[spin][1] is None
                else {
                    "pair": (spin_map[spin][1], spin),
                    "correlation": spin_map[spin][0],
                }
                for spin in sorted(spin_map.keys())
                if spin != spin_map[spin][1]
            ]
            elimination_tracker.append(eliminations)
            # add the metadata to the problem
            new_problem.metadata = problem_metadata

            # Save qaoa object, correlation matrix and expectation values z
            q_results_steps.append(q.result)
            corr_matrix_steps.append(corr_matrix)
            exp_vals_z_steps.append(exp_vals_z)
            problem_steps.append(problem)
            atomic_id_steps[counter] = q.header["atomic_id"]

            # Extract new number of qubits
            n_qubits = new_problem.n

            # problem is updated
            problem = new_problem

            # Compile qaoa with the problem
            q.compile(problem, verbose=False)

            # Add one step to the counter
            counter += 1
            # TODO: do rqaoa dumps here if dump is true, so the user can still get the results in case the loop is interrupted.

        if total_elimination:
            # Solve the smallest non-vanishing problem by fixing spins arbitrarily or according to the correlations
            cl_energy, cl_ground_states = rqaoa_utils.solution_for_vanishing_instances(
                problem.hamiltonian, spin_map
            )
        else:
            # Solve the new problem classically
            cl_energy, cl_ground_states = ground_state_hamiltonian(problem.hamiltonian)

        # Retrieve full solutions including eliminated spins and their energies
        full_solutions = rqaoa_utils.final_solution(
            elimination_tracker, cl_ground_states, self.problem.hamiltonian
        )

        # timestamp for the end of the optimization
        self.header["execution_time_end"] = generate_timestamp()

        # Compute description dictionary containing all the information
        self.result = self.results_class()
        self.result["solution"] = full_solutions
        self.result["classical_output"] = {
            "minimum_energy": cl_energy,
            "optimal_states": cl_ground_states,
        }
        self.result["elimination_rules"] = elimination_tracker
        self.result["schedule"] = [
            len(eliminations) for eliminations in elimination_tracker
        ]
        self.result["number_steps"] = counter - self.rqaoa_parameters.counter
        self.result["intermediate_steps"] = [
            {
                "counter": counter,
                "problem": problem,
                "qaoa_results": q_results,
                "exp_vals_z": exp_vals_z,
                "corr_matrix": corr_matrix,
            }
            for counter, problem, q_results, exp_vals_z, corr_matrix in zip(
                range(self.rqaoa_parameters.counter, counter),
                problem_steps,
                q_results_steps,
                exp_vals_z_steps,
                corr_matrix_steps,
            )
        ]
        self.result["atomic_ids"] = atomic_id_steps

        # set compiled to false
        self.compiled = False

        # dump the object if dump is true
        if dump:
            self.dump(
                **{**dump_options, **{"options": {"intermediate_measurements": False}}}
            )

        if verbose:
            print(f"RQAOA optimization completed.")

        return

    def __exp_val_hamiltonian_termwise(self, q: QAOA):
        """
        Private method to call the exp_val_hamiltonian_termwise function taking the data from
        the QAOA object _q.
        It eturns what the exp_val_hamiltonian_termwise function returns.
        """

        variational_params = q.variate_params
        qaoa_backend = q.backend
        cost_hamiltonian = q.cost_hamil
        mixer_type = q.circuit_properties.mixer_hamiltonian
        p = q.circuit_properties.p
        qaoa_optimized_angles = q.result.optimized["angles"]
        qaoa_optimized_counts = q.result.get_counts(
            q.result.optimized["measurement_outcomes"]
        )
        analytical = isinstance(qaoa_backend, QAOABackendAnalyticalSimulator)

        return exp_val_hamiltonian_termwise(
            cost_hamiltonian,
            mixer_type,
            p,
            qaoa_optimized_angles,
            qaoa_optimized_counts,
            analytical=analytical,
        )

    def __n_step(self, n_qubits, n_cutoff, counter):
        """
        Private method that returns the n_max value in case of adaptive or the number of eliminations according
        to the schedule and the counter in case of custom method.
        """

        if self.rqaoa_parameters.rqaoa_type == "adaptive":
            # Number of spins to eliminate according the schedule
            n = self.rqaoa_parameters.n_max
        else:
            # max Number of spins to eliminate
            n = self.rqaoa_parameters.steps[counter]

        # If the step eliminates more spins than available, reduce step to match cutoff
        return (n_qubits - n_cutoff) if (n_qubits - n_cutoff) < n else n

    def _serializable_dict(
        self, complex_to_string: bool = False, intermediate_measurements: bool = True
    ):
        """
        Returns all values and attributes of the object that we want to
        return in `asdict` and `dump(s)` methods in a dictionary.

        Parameters
        ----------
        complex_to_string: bool
            If True, complex numbers are converted to strings.
            If False, complex numbers are converted to lists of real and imaginary parts.

        Returns
        -------
        serializable_dict: dict
            Dictionary containing all the values and attributes of the object
            that we want to return in `asdict` and `dump(s)` methods.
        intermediate_measurements: bool
            If True, intermediate measurements are included in the dump. If False,
            intermediate measurements are not included in the dump.
            Default is True.
        """
        # we call the _serializable_dict method of the parent class,
        # specifying the keys to delete from the results dictionary
        serializable_dict = super()._serializable_dict(
            complex_to_string, intermediate_measurements
        )

        # we add the keys of the RQAOA object that we want to return
        serializable_dict["data"]["input_parameters"]["circuit_properties"] = dict(
            self.circuit_properties
        )
        serializable_dict["data"]["input_parameters"]["rqaoa_parameters"] = dict(
            self.rqaoa_parameters
        )

        # include parameters in the header metadata
        serializable_dict["header"]["metadata"]["param_type"] = serializable_dict[
            "data"
        ]["input_parameters"]["circuit_properties"]["param_type"]
        serializable_dict["header"]["metadata"]["init_type"] = serializable_dict[
            "data"
        ]["input_parameters"]["circuit_properties"]["init_type"]
        serializable_dict["header"]["metadata"]["p"] = serializable_dict["data"][
            "input_parameters"
        ]["circuit_properties"]["p"]

        if (
            serializable_dict["data"]["input_parameters"]["circuit_properties"]["q"]
            is not None
        ):
            serializable_dict["header"]["metadata"]["q"] = serializable_dict["data"][
                "input_parameters"
            ]["circuit_properties"]["q"]

        serializable_dict["header"]["metadata"]["rqaoa_type"] = serializable_dict[
            "data"
        ]["input_parameters"]["rqaoa_parameters"]["rqaoa_type"]
        serializable_dict["header"]["metadata"]["rqaoa_n_max"] = serializable_dict[
            "data"
        ]["input_parameters"]["rqaoa_parameters"]["n_max"]
        serializable_dict["header"]["metadata"]["rqaoa_n_cutoff"] = serializable_dict[
            "data"
        ]["input_parameters"]["rqaoa_parameters"]["n_cutoff"]

        return serializable_dict
