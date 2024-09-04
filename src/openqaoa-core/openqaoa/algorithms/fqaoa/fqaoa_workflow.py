from typing import Callable, Optional, Tuple, List, Union, Dict
from copy import deepcopy
import numpy as np

from .fqaoa_utils import (
    get_analytical_fermi_orbitals,
    get_fermi_orbitals,
    get_statevector,
    get_givens_rotation_angle,
)

from ..workflow_properties import WorkflowProperties
from ..baseworkflow import Workflow, check_compiled

from ...backends.devices_core import DeviceBase, DeviceLocal
from ...backends.qaoa_backend import get_qaoa_backend
from ...backends.basebackend import QuantumCircuitBase
from ...problems import QUBO
from ...qaoa_components.ansatz_constructor.gatemap import GateMap
from ...qaoa_components.ansatz_constructor.gatemaplabel import GateMapType, GateMapLabel
from ...qaoa_components.ansatz_constructor.gates import RotationAngle, X, RY, RZ, CX
from ...qaoa_components import (
    Hamiltonian,
    QAOADescriptor,
    create_qaoa_variational_params,
)
from ...qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)
from ...utilities import (
    get_mixer_hamiltonian,
    generate_timestamp,
)
from ...optimizers.qaoa_optimizer import get_optimizer
from ...backends.wrapper import SPAMTwirlingWrapper,ZNEWrapper

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
ALLOWED_MIXERS = ["xy"]
ALLOWED_LATTICE = ["cyclic", "chain"]
ALLOWED_LOCAL_SIMUALTORS = [
    "vectorized",
    "pyquil.statevector_simulator",
    'qiskit.qasm_simulator',
    'qiskit.shot_simulator',
    'qiskit.statevector_simulator',
]
NOT_ALLOWED_LOCAL_SIMULATORS = ["analytical_simulator"]

class FQAOA(Workflow):
    """
    A class implementing a FQAOA workflow end to end.

    It's basic usage consists of
    1. Initialization
    2. Compilation
    3. Optimization

    .. note::
        The attributes of the FQAOA class should be initialized using the set methods of FQAOA.
        For example, to set the circuit's depth to 10 you should run `set_circuit_properties(p=10)`

    Attributes
    ----------
    device: `DeviceBase`
        Device to be used by the optimizer
    circuit_properties: `FermiCircuitProperties`
        The circuit properties of the FQAOA workflow. Use to set depth `p`,
        choice of parameterization, parameter initialisation strategies, mixer hamiltonians.
        For a complete list of its parameters and usage please see the method `set_circuit_properties`
    backend_properties: `BackendProperties`
        The backend properties of the FQAOA workflow. Use to set the backend properties
        such as the number of shots and the cvar values.
        For a complete list of its parameters and usage please see the method `set_backend_properties`
    classical_optimizer: `ClassicalOptimizer`
        The classical optimiser properties of the QAOA workflow. Use to set the
        classical optimiser needed for the classical optimisation part of the QAOA routine.
        For a complete list of its parameters and usage please see the method `set_classical_optimizer`
    local_simulators: `list[str]`
        A list containing the available local simulators
    cloud_provider: `list[str]`
        A list containing the available cloud providers
    mixer_hamil: Hamiltonian
        The desired mixer hamiltonian
    cost_hamil: Hamiltonian
        The desired mixer hamiltonian
    qaoa_descriptor: QAOADescriptor
        the abstract and backend-agnostic representation of the underlying QAOA parameters
    variate_params: QAOAVariationalBaseParams
        The variational parameters. These are the parameters to be optimised by the classical optimiser
    backend: VQABaseBackend
        The openQAOA representation of the backend to be used to execute the quantum circuit
    optimizer: OptimizeVQA
        The classical optimiser
    result: `Result`
        Contains the logs of the optimisation process
    compiled: `Bool`
        A boolean flag to check whether the QAOA object has been correctly compiled at least once

    Examples
    --------
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> fqaoa = FQAOA()
    >>> fqaoa.compile(problem, n_fermions)
    >>> fqaoa.optimize()

    Where `problem` is an instance of `openqaoa.problems.problem.QUBO`
    with hamming weight constant constraint, where `n_fermions` is a constraint value.

    If you want to use non-default parameters:

    >>> fqaoa_custom = FQAOA()
    >>> fqaoa_custom.set_circuit_properties(
            p=10,
            param_type='extended',
            init_type='ramp',
        )
    >>> device = create_device(
            location='aws',
            name='arn:aws:braket:::device/quantum-simulator/amazon/sv1', aws_region='us-east-1',
        )
    >>> fqaoa_custom.set_device(device)
    >>> fqaoa_custom.set_backend_properties(n_shots=200)
    >>> fqaoa_custom.set_classical_optimizer(method='nelder-mead', maxiter=2)
    >>> fqaoa_custom.compile(problem, n_fermions)
    >>> fqaoa_custom.optimize()
    """

    def __init__(self, device=DeviceLocal("vectorized")):
        """
        Initialize the QAOA class.

        Parameters
        ----------
        device: `DeviceBase`
            Device to be used by the optimizer. Default is using the local 'vectorized' simulator.
        """
        super().__init__(device)
        self.circuit_properties = FermiCircuitProperties()
        self.backend_properties = FermiBackendProperties()

        # Exception handling in FQAOA
        if device.device_name in NOT_ALLOWED_LOCAL_SIMULATORS:
            raise ValueError(f"FQAOA does not support {NOT_ALLOWED_LOCAL_SIMULATORS}.")

        # change header algorithm to fqaoa
        self.header["algorithm"] = "fqaoa"

    @check_compiled
    def set_device(self, device: DeviceBase):
        """
        Override set_device to add a check for unsupported devices in FQAOA.

        Parameters
        ----------
        device: `DeviceBase`
            Device to be used by the optimizer.
        """

        # Exception handling in FQAOA
        if device.device_name in NOT_ALLOWED_LOCAL_SIMULATORS:
            raise ValueError(f"FQAOA does not support {NOT_ALLOWED_LOCAL_SIMULATORS}.")

        # Call the parent class's set_device method to handle the rest
        super().set_device(device)

    @check_compiled
    def set_backend_properties(self, **kwargs):
        """
        Override set_backend_properties to use FermiBackendProperties.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments representing backend properties.

            - init_hadamard : bool
                Whether to apply the Hadamard gate during initialization. This will be overridden to False.
        """

        for key, value in kwargs.items():
            if hasattr(self.backend_properties, key):
                pass  # setattr(self.backend_properties, key, value)
            else:
                raise ValueError(
                    f"Specified argument `{value}` for `{key}` in set_backend_properties is not supported"
                )

        self.backend_properties = FermiBackendProperties(**kwargs)

        return None

    @check_compiled
    def set_circuit_properties(self, **kwargs):
        """
        Specify the circuit properties to construct QAOA circuit

        Parameters
        ----------
        qubit_register: `list`
            Select the desired qubits to run the QAOA program. Meant to be used as a qubit
            selector for qubits on a QPU. Defaults to a list from 0 to n-1 (n = number of qubits)
        p: `int`
            Depth `p` of the QAOA circuit
        q: `int`
            Analogue of `p` of the QAOA circuit in the Fourier parameterization
        param_type: `str`
            Choose the QAOA circuit parameterization. Currently supported parameterizations include:
            `'standard'`: Standard QAOA parameterization
            `'standard_w_bias'`: Standard QAOA parameterization with a separate parameter for single-qubit terms.
            `'extended'`: Individual parameter for each qubit and each term in the Hamiltonian.
            `'fourier'`: Fourier circuit parameterization
            `'fourier_extended'`: Fourier circuit parameterization with individual parameter
            for each qubit and term in Hamiltonian.
            `'fourier_w_bias'`: Fourier circuit parameterization with a separate
            parameter for single-qubit terms
        init_type: `str`
            Initialisation strategy for the QAOA circuit parameters. Allowed init_types:
            `'rand'`: Randomly initialise circuit parameters
            `'ramp'`: Linear ramp from Hamiltonian initialisation of circuit
            parameters (inspired from Quantum Annealing)
            `'custom'`: User specified initial circuit parameters
        mixer_hamiltonian: `str`
            Allowed mixer hamiltonian:
            `'xy'`: xy-mixer
        mixer_qubit_connectivity: `[Union[List[list],List[tuple], str]]` By default set to 'cyclic'
            The connectivity of the qubits in the mixer Hamiltonian. Use only if
            `mixer_hamiltonian = xy`. The user can specify the connectivity as a list of lists,
            a list of tuples, or a string chosen from ['cyclic', 'chain'].
        mixer_coeffs: `list`
            The coefficients of the mixer Hamiltonian. By default all set to -1
        annealing_time: `float`
            Total time to run the FQAOA program in the Annealing parameterization (digitised annealing)
        linear_ramp_time: `float`
            The slope(rate) of linear ramp initialisation of QAOA parameters.
        variational_params_dict: `dict`
            Dictionary object specifying the initial value of each circuit parameter for
            the chosen parameterization, if the `init_type` is selected as `'custom'`.
            For example, for standard params set {'betas': [0.1, 0.2, 0.3], 'gammas': [0.1, 0.2, 0.3]}
        """

        for key, value in kwargs.items():
            if hasattr(self.circuit_properties, key):
                pass
            else:
                raise ValueError("Specified argument is not supported by the circuit")
        self.circuit_properties = FermiCircuitProperties(**kwargs)

        return None

    def compile(
        self,
        problem: QUBO = None,
        n_fermions: int = None,
        hopping: float = 1.0,
        verbose: bool = False,
        routing_function: Optional[Callable] = None,
    ):
        """
        Initialise the trainable parameters for FQAOA according to the specified
        strategies and by passing the problem statement

        .. note::
            Compilation is necessary because it is the moment where the problem statement and
            the FQAOA instructions are used to build the actual FQAOA circuit.

        .. tip::
            Set Verbose to false if you are running batch computations!

        Parameters
        ----------
        problem: `QUBO`
            portfolio optimisation problems converted to QUBO using penalty methods
        n_fermions: `int`
            a constraint value, budgets in portfolio optimization problem.
        hopping: `float`, optional
            the coefficient of the fermionic mixer Hamiltonian
        verbose: bool
            Set True to have a summary of FQAOA to displayed after compilation
        """

        # connect to the QPU specified
        self.device.check_connection()
        # we compile the method of the parent class to generate the id and
        # check the problem is a QUBO object and save it
        super().compile(problem=problem)

        # check the n_fermions is given and save it
        if n_fermions is None:
            raise ValueError("In FQAOA, the 'n_fermions' argument must be specified")

        self.n_fermions = n_fermions
        self.hopping = hopping

        self.cost_hamil = Hamiltonian.classical_hamiltonian(
            terms=problem.terms, coeffs=problem.weights, constant=problem.constant
        )

        self.n_qubits = self.cost_hamil.n_qubits

        # Determine the coefficients of the mixer hamiltonian
        if self.circuit_properties.mixer_qubit_connectivity == "cyclic":
            self.circuit_properties.mixer_coeffs = [-0.5*hopping] * 2 * self.n_qubits
        elif self.circuit_properties.mixer_qubit_connectivity == "chain":
            self.circuit_properties.mixer_coeffs = [-0.5*hopping] * 2 * (self.n_qubits-1)

        self.mixer_hamil = get_mixer_hamiltonian(
            n_qubits=self.n_qubits,
            mixer_type=self.circuit_properties.mixer_hamiltonian,
            qubit_connectivity=self.circuit_properties.mixer_qubit_connectivity,
            coeffs=self.circuit_properties.mixer_coeffs,
        )

        self.qaoa_descriptor = QAOADescriptor(
            self.cost_hamil,
            self.mixer_hamil,
            p=self.circuit_properties.p,
            routing_function=routing_function,
            device=self.device,
        )

        self.variate_params = create_qaoa_variational_params(
            qaoa_descriptor=self.qaoa_descriptor,
            params_type=self.circuit_properties.param_type,
            init_type=self.circuit_properties.init_type,
            variational_params_dict=self.circuit_properties.variational_params_dict,
            linear_ramp_time=self.circuit_properties.linear_ramp_time,
            q=self.circuit_properties.q,
            seed=self.circuit_properties.seed,
            total_annealing_time=self.circuit_properties.annealing_time,
        )

        # Backend configuration required for initial state preparation in FQAOA.
        lattice = self.circuit_properties.mixer_qubit_connectivity

        # fermion orbitals
        orbitals = get_fermi_orbitals(self.n_qubits, self.n_fermions, lattice, hopping)

        # initial statevector or circuit
        if self.device.device_name in 'vectorized':
            self.backend_properties.prepend_state = get_statevector(orbitals)
        else:
            gate_applicator = self._gate_applicator()
            self.backend_properties.prepend_state = self._fermi_initial_circuit(orbitals, gate_applicator)

        backend_dict = self.backend_properties.__dict__.copy()

        self.backend = get_qaoa_backend(
            qaoa_descriptor=self.qaoa_descriptor,
            device=self.device,
            **backend_dict,
        )

        # Implementing SPAM Twirling and MITIQs error mitigation requires wrapping the backend.
        # However, the BaseWrapper can have many more use cases.
        if (
            self.error_mitigation_properties.error_mitigation_technique
            == "spam_twirling"
        ):
            self.backend = SPAMTwirlingWrapper(
                backend=self.backend,
                n_batches=self.error_mitigation_properties.n_batches,
                calibration_data_location=self.error_mitigation_properties.calibration_data_location,
            )
        elif(
            self.error_mitigation_properties.error_mitigation_technique
            == "mitiq_zne"
        ):
            self.backend = ZNEWrapper(
                backend=self.backend,
                factory=self.error_mitigation_properties.factory,
                scaling=self.error_mitigation_properties.scaling,
                scale_factors=self.error_mitigation_properties.scale_factors,
                order=self.error_mitigation_properties.order,
                steps=self.error_mitigation_properties.steps
            )

        self.optimizer = get_optimizer(
            vqa_object=self.backend,
            variational_params=self.variate_params,
            optimizer_dict=self.classical_optimizer.asdict(),
        )

        # Set the header properties
        self.header["target"] = self.device.device_name
        self.header["cloud"] = self.device.device_location

        metadata = {
            "p": self.circuit_properties.p,
            "param_type": self.circuit_properties.param_type,
            "init_type": self.circuit_properties.init_type,
            "optimizer_method": self.classical_optimizer.method,
        }

        self.set_exp_tags(tags=metadata)

        self.compiled = True

        if verbose:
            print("\t \033[1m ### Summary ###\033[0m")
            print("OpenQAOA has been compiled with the following properties")
            print(
                f"Solving FQAOA with \033[1m {self.device.device_name} \033[0m on"
                f"\033[1m{self.device.device_location}\033[0m"
            )
            print(
                f"Using p={self.circuit_properties.p} with {self.circuit_properties.param_type}"
                f"parameters initialized as {self.circuit_properties.init_type}"
            )

            if hasattr(self.backend, "n_shots"):
                print(
                    f"OpenQAOA will optimize using \033[1m{self.classical_optimizer.method}"
                    f"\033[0m, with up to \033[1m{self.classical_optimizer.maxiter}"
                    f"\033[0m maximum iterations. Each iteration will contain"
                    f"\033[1m{self.backend_properties.n_shots} shots\033[0m"
                )
            else:
                print(
                    f"OpenQAOA will optimize using \033[1m{self.classical_optimizer.method}\033[0m,"
                    "with up to \033[1m{self.classical_optimizer.maxiter}\033[0m maximum iterations"
                )

        return None

    def optimize(self, verbose=False):
        """
        A method running the classical optimisation loop
        """

        if self.compiled is False:
            raise ValueError("Please compile the FQAOA before optimizing it !")

        # timestamp for the start of the optimization
        self.header["execution_time_start"] = generate_timestamp()

        self.optimizer.optimize()
        # TODO: result and qaoa_result will differ
        self.result = self.optimizer.qaoa_result

        # timestamp for the end of the optimization
        self.header["execution_time_end"] = generate_timestamp()

        if verbose:
            print("Optimization completed.")
        return

    def evaluate_circuit(
        self,
        params: Union[List[float], Dict[str, List[float]], QAOAVariationalBaseParams],
    ):
        """
        A method to evaluate the QAOA circuit at a given set of parameters

        Parameters
        ----------
        params: list or dict or QAOAVariationalBaseParams or None
            List of parameters or dictionary of parameters. Which will be used to evaluate the QAOA circuit.
            If None, the variational parameters of the QAOA object will be used.

        Returns
        -------
        result: dict
            A dictionary containing the results of the evaluation:
            - "cost": the expectation value of the cost Hamiltonian
            - "uncertainty": the uncertainty of the expectation value of the cost Hamiltonian
            - "measurement_results": either the state of the QAOA circuit output (if the QAOA circuit is
            evaluated on a state simulator) or the counts of the QAOA circuit output
            (if the QAOA circuit is evaluated on a QPU or shot-based simulator)
        """
        # before evaluating the circuit we check that the QAOA object has been compiled
        if self.compiled is False:
            raise ValueError("Please compile the FQAOA before optimizing it!")

        # Check the type of the input parameters and save them as a
        # QAOAVariationalBaseParams object at the variable `params_obj`

        # if the parameters are passed as a dictionary we copy and update the variational parameters of the QAOA object
        if isinstance(params, dict):
            params_obj = deepcopy(self.variate_params)
            # we check that the dictionary contains all the parameters of the QAOA object that are not empty
            for key, value in params_obj.asdict().items():
                if value.size > 0:
                    assert (
                        key in params.keys()
                    ), f"The parameter `{key}` is missing from the input dictionary"
            params_obj.update_from_dict(params)

        # if the parameters are passed as a list we copy and update the variational parameters of the QAOA object
        elif isinstance(params, list) or isinstance(params, np.ndarray):
            assert len(params) == len(
                self.variate_params
            ), "The number of parameters does not match the number of parameters in the QAOA circuit"
            params_obj = deepcopy(self.variate_params)
            params_obj.update_from_raw(params)

        # if the parameters are passed as a QAOAVariationalBaseParams object we just take it as it is
        elif isinstance(params, QAOAVariationalBaseParams):
            # check whether the input params object is supported for circuit evaluation
            assert (
                len(self.variate_params.mixer_1q_angles) == len(params.mixer_1q_angles)
                and len(self.variate_params.mixer_2q_angles)
                == len(self.variate_params.mixer_2q_angles)
                and len(self.variate_params.cost_1q_angles)
                == len(self.variate_params.cost_1q_angles)
                and len(self.variate_params.cost_2q_angles)
                == len(self.variate_params.cost_2q_angles)
            ), "Specify a supported params object"
            params_obj = params

        # if the parameters are passed in a different format, we raise an error
        else:
            raise TypeError(
                f"The input params must be a list or a dictionary. Instead, received {type(params)}"
            )

        # Evaluate the QAOA circuit and return the results
        output_dict = {
            "cost": None,
            "uncertainty": None,
            "measurement_results": None,
        }
        # if the workflow implements SPAM Twirling,
        # we just return the expectation value of the cost Hamiltonian and the measurement outcomes
        if isinstance(self.backend, SPAMTwirlingWrapper):
            cost = self.backend.expectation(params_obj)
            measurement_results = (
                self.backend.measurement_outcomes
                if isinstance(self.backend.measurement_outcomes, dict)
                else self.backend.measurement_outcomes.tolist()
            )
            output_dict.update(
                {
                    "cost": cost,
                    "measurement_results": measurement_results,
                }
            )
            # in all other cases, we return the expectation value of the cost Hamiltonian,
            # the associated uncertainty and the measurement outcomes
        else:
            cost, uncertainty = self.backend.expectation_w_uncertainty(params_obj)
            measurement_results = (
                self.backend.measurement_outcomes
                if isinstance(self.backend.measurement_outcomes, dict)
                else self.backend.measurement_outcomes.tolist()
            )
            output_dict.update(
                {
                    "cost": cost,
                    "uncertainty": uncertainty,
                    "measurement_results": measurement_results,
                }
            )
        return output_dict

    def _serializable_dict(
        self, complex_to_string: bool = False, intermediate_measurements: bool = True
    ):
        """
        Returns all values and attributes of the object that we want to return in
        `asdict` and `dump(s)` methods in a dictionary.

        Parameters
        ----------
        complex_to_string: bool
            If True, complex numbers are converted to strings.
            This is useful for JSON serialization.

        Returns
        -------
        serializable_dict: dict
            A dictionary containing all the values and attributes of the object
            that we want to return in `asdict` and `dump(s)` methods.
        intermediate_measurements: bool
            If True, intermediate measurements are included in the dump.
            If False, intermediate measurements are not included in the dump.
            Default is True.
        """

        # we call the _serializable_dict method of the parent class,
        # specifying the keys to delete from the results dictionary
        serializable_dict = super()._serializable_dict(
            complex_to_string, intermediate_measurements
        )

        # we add the keys of the QAOA object that we want to return
        serializable_dict["data"]["input_parameters"]["circuit_properties"] = dict(
            self.circuit_properties
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

        return serializable_dict

    def _fermi_initial_circuit(self, orbitals: np.array, gate_applicator: object) -> object:
        """
        Constructs the initial quantum circuit for the FQAOA.

        This method initializes a quantum circuit for a system with a specified number of fermions and
        qubits. The method applies X gates to excite the number of fermions and then applies a series 
        of Givens rotation gates according to the provided orbital data.

        Parameters
        ----------
        orbitals : np.array
            A numpy array containing the orbital information needed to compute the Givens rotation angles.
        gate_applicator : object
            An object responsible for applying quantum gates to the circuit.

        Returns
        -------
        object
            A quantum circuit object initialized with the fermions and Givens rotations.
        """

        initial_circuit = gate_applicator.create_quantum_circuit(self.n_qubits)

        # excites `n_fermions` number of fermion
        for i in range(self.n_fermions):
            gate = X(gate_applicator, i)
            gate.apply_gate(initial_circuit)

        # apply `givens rotation gate`
        gtheta = get_givens_rotation_angle(orbitals)
        i = (self.n_qubits-self.n_fermions)*self.n_fermions
        for irow in range(self.n_fermions-1, -1, -1):
            for icol in range(irow+1, self.n_qubits-self.n_fermions+irow+1):
                i -= 1
                angle = gtheta[i]
                for each_tuple in GivensRotationGateMap(icol, icol-1, angle).decomposition('standard'):
                    gate = each_tuple[0](gate_applicator, *each_tuple[1])
                    gate.apply_gate(initial_circuit)

        return initial_circuit

    def _gate_applicator(self) -> object:
        """
        Set up and return the gate applicator for the specified device.

        This method temporarily sets the backend by calling the appropriate
        gate applicator based on the specified device properties.

        Returns
        -------
        object
            The gate applicator object associated with the specified device.
        """

        device_name = self.device.device_name
        backend_dict = self.backend_properties.__dict__.copy()
        self.backend = get_qaoa_backend(
            qaoa_descriptor=self.qaoa_descriptor,
            device = self.device,
            **backend_dict,)
        gate_applicator = self.backend.gate_applicator

        return(gate_applicator)

class GivensRotationGateMap(GateMap):
    """
    Returns the gate applicator object for the specified quantum backend.

    This method configures the quantum backend based on the device. 
    It then retrieves and returns the gate applicator, which is used to apply quantum gates in the 
    circuit construction process.

    Returns
    -------
    object
        An object representing the gate applicator for the current backend.
    """

    def __init__(self, qubit_1: int, qubit_2: int, angle: float):
        super().__init__(qubit_1)
        self.qubit_2 = qubit_2
        self.angle = angle
        self.gate_label = GateMapLabel(n_qubits=2, gatemap_type=GateMapType.FIXED)

    @property
    def _decomposition_standard(self) -> List[Tuple]:
        return[
            (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)]),
            (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)]),
            (RY, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)]),
            (X,  [self.qubit_1]),
            (CX, [self.qubit_1, self.qubit_2]),
            (RY, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, self.angle)]),
            (RY, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, self.angle)]),
            (CX, [self.qubit_1, self.qubit_2]),
            (RY, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)]),
            (X,  [self.qubit_1]),
            (RZ, [self.qubit_2, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)]),
            (RZ, [self.qubit_1, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)]),
        ]

class FermiBackendProperties(WorkflowProperties):
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
        enable/disable qubit rewiring when accessing QPUs via the AWS `braket`
    """

    def __init__(
        self,
        prepend_state: Optional[
            Union[QuantumCircuitBase, List[complex], np.ndarray]
        ] = None,
        append_state: Optional[
            Union[QuantumCircuitBase, np.ndarray]
        ] = None,
        init_hadamard: bool = False,
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
        if init_hadamard:
            raise ValueError("In FQAOA, init_hadamard is not recognized.")
        if prepend_state is not None:
            raise ValueError("In FQAOA, prepend_state is not recognized.")
        if append_state is not None:
            raise ValueError("In FQAOA, append_state is not recognized.")
        self.init_hadamard = False
        self.prepend_state = None
        self.append_state = append_state
        self.n_shots = n_shots
        self.cvar_alpha = cvar_alpha
        self.noise_model = noise_model
        self.initial_qubit_mapping = initial_qubit_mapping
        self.seed_simulator = seed_simulator
        self.qiskit_simulation_method = qiskit_simulation_method
        self.qiskit_optimization_level = qiskit_optimization_level
        self.active_reset = active_reset
        self.rewiring = rewiring
        self.disable_qubit_rewiring = disable_qubit_rewiring

class FermiCircuitProperties(WorkflowProperties):
    """
    Tunable properties of the FQAOA circuit to be specified by the user

    The only difference with CircuitProperties is that mixer_hamiltonian is limited to "xy"
    and mixer_qubit connetivity is limited to "cyclic" or "chain".
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
        mixer_hamiltonian: Optional[str] = "xy",
        mixer_qubit_connectivity: Optional[str] = "cyclic",
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
        if mixer_hamiltonian.lower() not in ALLOWED_MIXERS:
            raise ValueError(f"In FQAOA, mixer_hamiltonian {mixer_hamiltonian.lower()} is not recognized.")
        if mixer_qubit_connectivity not in ALLOWED_LATTICE:
            raise ValueError(f"In FQAOA, mixer_qubit_connectivity {mixer_qubit_connectivity} is not recognized.")
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
                f"param_type {value} is not recognized. Please use {ALLOWED_PARAM_TYPES}"
            )
        self._param_type = value

    @property
    def init_type(self):
        return self._init_type

    @init_type.setter
    def init_type(self, value):
        if value not in ALLOWED_INIT_TYPES:
            raise ValueError(
                f"init_type {value} is not recognized. Please use {ALLOWED_INIT_TYPES}"
            )
        self._init_type = value

    @property
    def mixer_hamiltonian(self):
        return self._mixer_hamiltonian

    @mixer_hamiltonian.setter
    def mixer_hamiltonian(self, value):
        if value not in ALLOWED_MIXERS:
            raise ValueError(
                f"mixer_hamiltonian {value} is not recognized. Please use {ALLOWED_MIXERS}"
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
