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
from typing import List
import json
import gzip
from os.path import exists

from .workflow_properties import (
    BackendProperties,
    ErrorMitigationProperties,
    ClassicalOptimizer,
)
from ..backends.devices_core import DeviceBase, DeviceLocal
from ..problems import QUBO
from ..utilities import delete_keys_from_dict, is_valid_uuid, generate_uuid
from ..backends.qaoa_backend import (
    DEVICE_NAME_TO_OBJECT_MAPPER,
    DEVICE_ACCESS_OBJECT_MAPPER,
)


def check_compiled(func):
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        if self.compiled:
            raise ValueError(
                "Cannot change properties of the object after compilation."
            )
        return result

    return wrapper


class Workflow(ABC):
    """
    Abstract class to represent an optimizer

    It's basic usage consists of

     #. Initialization
     #. Compilation
     #. Optimization

    Attributes
    ----------
    device: `DeviceBase`
        Device to be used by the optimizer
    backend_properties: `BackendProperties`
        The backend properties of the optimizer workflow. Use to set the
        backend properties such as the number of shots and the cvar values.
        For a complete list of its parameters and usage please see the method `set_backend_properties`
    classical_optimizer: `ClassicalOptimizer`
        The classical optimiser properties of the optimizer workflow.
        Use to set the classical optimiser needed for the classical optimisation part of the optimizer routine.
        For a complete list of its parameters and usage please see the method `set_classical_optimizer`
    local_simulators: `list[str]`
        A list containing the available local simulators
    cloud_provider: `list[str]`
        A list containing the available cloud providers
    compiled: `Bool`
        A boolean flag to check whether the optimizer object has been correctly compiled at least once
    id: TODO
    exp_tags: `dict`
        A dictionary containing the tags of the optimizer object. The user can set this
        value using the set_exp_tags method.
    problem: `Problem`
        The problem object that the optimizer will be optimizing.
    results: `Results`
        The results object that will contain the results of the optimization routine.
    """

    def __init__(self, device=DeviceLocal("vectorized")):
        """
        Initialize the optimizer class.

        Parameters
        ----------
        device: `DeviceBase`
            Device to be used by the optimizer. Default is using the local 'vectorized' simulator.
        """

        self.device = device
        self.backend_properties = BackendProperties()
        self.error_mitigation_properties = ErrorMitigationProperties()
        self.classical_optimizer = ClassicalOptimizer()
        self.local_simulators = list(DEVICE_NAME_TO_OBJECT_MAPPER.keys())
        self.cloud_provider = list(DEVICE_ACCESS_OBJECT_MAPPER.keys())
        self.available_error_mitigation_techniques = ["spam_twirling"]
        self.compiled = False

        # Initialize the identifier stamps, we initialize all the stamps needed to None
        self.header = {
            "atomic_id": None,  # the id of the run it is generated automatically in the compilation
            "experiment_id": generate_uuid(),  # the id of the experiment it is generated automatically here
            "project_id": None,
            "algorithm": None,  # qaoa or rqaoa
            "description": None,
            "run_by": None,
            "provider": None,
            "target": None,
            "cloud": None,
            "client": None,
            "execution_time_start": None,
            "execution_time_end": None,
        }

        # Initialize the experiment tags
        self.exp_tags = {}

        # Initialize the results and problem objects
        self.problem = None
        self.result = None

    def __setattr__(self, __name, __value):
        # check the attribute exp_tags is json serializable
        if __name == "exp_tags":
            try:
                json.dumps(__value)
            except:
                raise ValueError("The exp_tags attribute is not json serializable")

        return super().__setattr__(__name, __value)

    def set_header(
        self,
        project_id: str = None,
        description: str = None,
        run_by: str = None,
        provider: str = None,
        target: str = None,
        cloud: str = None,
        client: str = None,
        experiment_id: str = None,
    ):
        """
        Method to set the identification stamps of the optimizer object in self.header.

        Parameters
        ----------
        TODO : document the parameters
        """
        if project_id is not None:
            if not is_valid_uuid(project_id):
                raise ValueError(
                    "The project_id is not a valid uuid, example of a valid uuid: 8353185c-b175-4eda-9628-b4e58cb0e41b"
                )

        if not is_valid_uuid(project_id):
            raise ValueError(
                "The project_id is not a valid uuid, example of a valid uuid: 8353185c-b175-4eda-9628-b4e58cb0e41b"
            )

        if experiment_id is not None:
            if not is_valid_uuid(experiment_id):
                raise ValueError(
                    "The experiment_id is not a valid uuid, \
                        example of a valid uuid: 8353185c-b175-4eda-9628-b4e58cb0e41b"
                )
            else:
                self.header["experiment_id"] = experiment_id

        self.header["project_id"] = project_id
        self.header["description"] = description
        self.header["run_by"] = run_by
        self.header["provider"] = provider
        self.header["target"] = target
        self.header["cloud"] = cloud
        self.header["client"] = client

    def set_exp_tags(self, tags: dict):
        """
        Method to add tags to the experiment. Tags are stored in a dictionary
        (self.exp_tags) and can be used to identify the experiment.
        Name is a special tag that is used to identify the experiment in the results object,
        it will also be stored in the dictionary, and will overwrite any previous name.

        Parameters
        ----------
        name: `str`
            Name of the experiment. If None, the name will not be changed.
            If not None, the name will be changed to the new one.
        tags: `dict`
            Dictionary containing the tags to be added to the experiment.
            If the tag already exists, it will be overwritten.
        """

        self.exp_tags = {**self.exp_tags, **tags}

    @check_compiled
    def set_device(self, device: DeviceBase):
        """ "
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

    @check_compiled
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
            noise_model: `qiskit.providers.aer.noise.NoiseModel`
                    The Qiskit noise model to be used for the simulation.
            qiskit_simulation_method: str, optional
                The method to be used for the simulation.
            qiskit_optimization_level: int, optional
                The qiskit.transpile optimization level to use. Choose from 0,1,2,3
            seed_simulator: int
                Optional argument to initialize a pseudorandom solution. Default None
            active_reset:
                #TODO
            rewiring:
                Rewiring scheme to be used for Pyquil.
                Either 'PRAGMA INITIAL_REWIRING "NAIVE"' or
                'PRAGMA INITIAL_REWIRING "PARTIAL"'. If None, defaults to NAIVE
            disable_qubit_rewiring: `bool`
                Disable automatic qubit rewiring on AWS braket backend
        """

        for key, value in kwargs.items():
            if hasattr(self.backend_properties, key):
                pass  # setattr(self.backend_properties, key, value)
            else:
                raise ValueError(
                    f"Specified argument `{value}` for `{key}` in set_backend_properties is not supported"
                )

        self.backend_properties = BackendProperties(**kwargs)
        return None

    @check_compiled
    def set_error_mitigation_properties(self, **kwargs):
        """
        Set the error mitigation properties, if any.

        Parameters
        ----------
            error_mitigation_technique: str
                The specific technique used to mitigate the errors. Only a simple state preparation and measurement twirling with bitflip averages, under the name "spam_twirling" is currently supported.
            n_batches: int
                The number of batches specifies the different negating schedules at random. Total number of shots is distributed accordingly.
            calibration_data_location: str
                The location of the json file containing calibration data. For spam twirling this is the measurement outcomes of an empty circuit under the bit-flip averaging.

        """
        for key, value in kwargs.items():
            if hasattr(self.error_mitigation_properties, key) and (
                kwargs["error_mitigation_technique"].lower()
                in self.available_error_mitigation_techniques
            ):
                pass  # setattr(self.error_mitigation, key, value)
            else:
                raise ValueError(
                    f"Specified argument `{value}` for `{key}` in set_error_mitigation_properties is not supported"
                )

        self.error_mitigation_properties = ErrorMitigationProperties(**kwargs)
        return None

    @check_compiled
    def set_classical_optimizer(self, **kwargs):
        """
        Set the parameters for the classical optimizer to be used in the optimizers workflow

        Parameters
        ----------
            method: str
                The classical optimization method. To see the list of supported optimizers, refer
                to `available_optimizers` in openqaoa/optimizers/qaoa_optimizer.py
            maxiter : Optional[int]
                Maximum number of iterations.
            maxfev : Optional[int]
                Maximum number of function evaluations.
            jac: str
                Method to compute the gradient vector. Choose from:
                    - ['finite_difference', 'param_shift', 'stoch_param_shift', 'grad_spsa']
            hess: str
                Method to compute the hessian. Choose from:
                    - ['finite_difference', 'param_shift', 'stoch_param_shift', 'grad_spsa']
            constraints: scipy.optimize.LinearConstraints, scipy.optimize.NonlinearConstraints
                Scipy-based constraints on parameters of optimization. Will be available soon
            bounds: scipy.optimize.Bounds
                Scipy-based bounds on parameters of optimization. Will be available soon
            tol : float
                Tolerance before the optimizer terminates; if `tol` is larger than
                the difference between two steps, terminate optimization.
            optimizer_options : dict
                Dictionary of optimiser-specific arguments.
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
            optimization_progress : bool
                Returns history of measurement outcomes/wavefunction if `True`. Defaults to `False`.
            cost_progress : bool
                Returns history of cost values if `True`. Defaults to `True`.
            parameter_log : bool
                Returns history of angles if `True`. Defaults to `True`.
            save_intermediate: bool
                If True, the intermediate parameters of the optimization and job ids,
                if available, are saved throughout the run. This is set to False by default.
        """
        for key, value in kwargs.items():
            if hasattr(self.classical_optimizer, key):
                pass  # setattr(self.classical_optimizer, key, value)
            else:
                raise ValueError(
                    "Specified argument is not supported by the Classical Optimizer"
                )

        self.classical_optimizer = ClassicalOptimizer(**kwargs)
        return None

    def compile(self, problem: QUBO):
        """
        Method that will make sure that the problem is in the correct form for
        the optimizer to run and generate the atomic id.
        This method should be extended by the child classes to include
        the compilation of the problem into the correct form for the optimizer to run.

        Parameters
        ----------
        problem: QUBO
            The problem to be optimized. Must be in QUBO form.
        """

        # check and set problem
        assert isinstance(problem, QUBO), "The problem must be converted into QUBO form"
        self.problem = problem

        if hasattr(self.device, "n_qubits"):
            if self.device.n_qubits < self.problem.n:
                raise Exception(
                    f"The number of qubits {self.problem.n} is more than the number of qubits available on the device."
                    f"{self.device.name} features f{self.device.n_qubits} qubits"
                )

        # the atomic id is generated every time that it is compiled
        self.header["atomic_id"] = generate_uuid()

        # header is updated with the qubit number of the problem
        self.set_exp_tags({"qubit_number": self.problem.n})

    def optimize():
        raise NotImplementedError

    def _serializable_dict(
        self, complex_to_string: bool = False, intermediate_measurements: bool = True
    ):
        """
        Returns a dictionary with all values and attributes of the object that we want to
        return in `asdict` and `dump(s)` methods in a dictionary.
        The returned dictionary has two keys: header and data. The header contains all the data
        that can identify the experiment, while the data contains all the input and output data
        of the experiment (also the experiment tags).

        Parameters
        ----------
        complex_to_string: bool
            If True, converts all complex numbers to strings. This is useful for
            JSON serialization, for the `dump(s)` methods.
        intermediate_measurements: bool
            If True, intermediate measurements are included in the dump.
            If False, intermediate measurements are not included in the dump.
            Default is True.
        """

        # create the final data dictionary
        data = {}
        data["exp_tags"] = self.exp_tags.copy()
        data["input_problem"] = dict(self.problem) if self.problem is not None else None
        data["input_parameters"] = {
            "device": {
                "device_location": self.device.device_location,
                "device_name": self.device.device_name,
            },
            "backend_properties": dict(self.backend_properties),
            "classical_optimizer": dict(self.classical_optimizer),
        }
        # change the parameters that aren't serializable to strings
        for item in ["noise_model", "append_state", "prepend_state"]:
            if data["input_parameters"]["backend_properties"][item] is not None:
                data["input_parameters"]["backend_properties"][item] = str(
                    data["input_parameters"]["backend_properties"][item]
                )

        data["result"] = (
            self.result.asdict(False, complex_to_string, intermediate_measurements)
            if self.result not in [None, {}]
            else None
        )

        # create the final header dictionary
        header = self.header.copy()
        header["metadata"] = {
            **self.exp_tags.copy(),
            **(
                {
                    "problem_type": data["input_problem"]["problem_instance"][
                        "problem_type"
                    ]
                }
                if data["input_problem"] is not None
                else {}
            ),
            **(
                data["input_problem"]["metadata"].copy()
                if data["input_problem"] is not None
                else {}
            ),
            **{"n_shots": data["input_parameters"]["backend_properties"]["n_shots"]},
            **{
                prepend + key: data["input_parameters"]["classical_optimizer"][key]
                for prepend, key in zip(
                    ["optimizer_", "", ""], ["method", "jac", "hess"]
                )
                if not data["input_parameters"]["classical_optimizer"][key] is None
            },
        }

        # we return a dictionary (serializable_dict) that will have two keys: header and data
        serializable_dict = {
            # header is a dictionary containing all the data that can identify the experiment
            "header": header,
            # data is a dictionary containing all the input and output data of the experiment (also the experiment tags)
            "data": data,
        }

        return serializable_dict

    def asdict(self, exclude_keys: List[str] = [], options: dict = {}):
        """
        Returns a dictionary of the Optimizer object, where all objects are converted to dictionaries.

        Parameters
        ----------
        exclude_keys : List[str]
            A list of keys to exclude from the returned dictionary.
        options : dict
            A dictionary of options to pass to the method that creates the dictionary to dump.
                complex_to_string : bool
                    If True, converts complex numbers to strings. If False,
                    complex numbers are not converted to strings.
                intermediate_measurements : bool
                    If True, includes the intermediate measurements in the results.
                    If False, only the final measurements are included.

        Returns
        -------
        dict
        """

        options = {**{"complex_to_string": False}, **options}

        if exclude_keys == []:
            return self._serializable_dict(**options)
        else:
            return delete_keys_from_dict(
                obj=self._serializable_dict(**options), keys_to_delete=exclude_keys
            )

    def dumps(self, indent: int = 2, exclude_keys: List[str] = [], options: dict = {}):
        """
        Returns a json string of the Optimizer object.

        Parameters
        ----------
        indent : int
            The number of spaces to indent the result in the json file.
            If None, the result is not indented.
        exclude_keys : List[str]
            A list of keys to exclude from the json string.
        options : dict
            A dictionary of options to pass to the method that creates
            the dictionary to dump.
        intermediate_measurements : bool
            If True, includes the intermediate measurements in the results.
            If False, only the final measurements are included.

        Returns
        -------
        str
        """

        options = {**options, **{"complex_to_string": True}}

        if exclude_keys == []:
            return json.dumps(self._serializable_dict(**options), indent=indent)
        else:
            return json.dumps(
                delete_keys_from_dict(
                    obj=self._serializable_dict(**options), keys_to_delete=exclude_keys
                ),
                indent=indent,
            )

    def dump(
        self,
        file_name: str = "",
        file_path: str = "",
        prepend_id: bool = False,
        indent: int = 2,
        compresslevel: int = 0,
        exclude_keys: List[str] = [],
        overwrite: bool = False,
        options: dict = {},
    ):
        """
        Saves the Optimizer object as json file (if compresslevel is 0).
        If compresslevel is not 0, saves the Optimizer object as a .gz file
        (which should be decompressed before use).

        Parameters
        ----------
        file_name : str
            The name of the json file.
        file_path : str
            The path where the json file will be saved.
        prepend_id : bool
            If True, the name will have the following format: '{project_id}--{experiment_id}--{atomic_id}--{file_name}.json'.
            If False, the name will have the following format: '{file_name}.json'.
            Default is False.
        indent : int
            The number of spaces to indent the result in the json file.
            If None, the result is not indented.
        compresslevel : int
            The compression level to use. If 0, no compression is used and a json file is saved.
            If 1, the fastest compression method is used. If 9, the slowest
            but most effective compression method is used. And a .gz file is saved.
        exclude_keys : List[str]
            A list of keys that should not be included in the json file.
        overwrite : bool
            If True, overwrites the file if it already exists. If False,
            raises an error if the file already exists.
        options : dict
            A dictionary of options to pass to the method that creates the dictionary to dump.
        intermediate_measurements : bool
            If True, includes the intermediate measurements in the results.
            If False, only the final measurements are included.
        """

        options = {**options, **{"complex_to_string": True}}

        project_id = (
            self.header["project_id"]
            if not self.header["project_id"] is None
            else "None"
        )

        # get the full name
        if prepend_id is False and file_name == "":
            raise ValueError(
                "dump method missing argument: 'file_name'. Otherwise 'prepend_id' must be specified as True."
            )
        elif prepend_id is False:
            file = file_path + file_name
        elif file_name == "":
            file = (
                file_path
                + project_id
                + "--"
                + self.header["experiment_id"]
                + "--"
                + self.header["atomic_id"]
            )
        else:
            file = (
                file_path
                + project_id
                + "--"
                + self.header["experiment_id"]
                + "--"
                + self.header["atomic_id"]
                + "--"
                + file_name
            )

        # adding .json extension if not present and adding .gz extension if compresslevel is not 0 and not present
        file = file + ".json" if ".json" != file[-5:] else file
        if compresslevel != 0:
            file = file + ".gz" if ".gz" != file[-3:] else file

        # checking if the file already exists, and raising an error if it does and overwrite is False
        if overwrite is False and exists(file):
            raise FileExistsError(
                f"The file {file} already exists. Please change the name of the file or set overwrite=True."
            )

        # saving the file
        if compresslevel == 0:  # if compresslevel is 0, save as json file
            with open(file, "w") as f:
                if exclude_keys == []:
                    json.dump(self._serializable_dict(**options), f, indent=indent)
                else:
                    json.dump(
                        delete_keys_from_dict(
                            obj=self._serializable_dict(**options),
                            keys_to_delete=exclude_keys,
                        ),
                        f,
                        indent=indent,
                    )
        else:  # if compresslevel is not 0, save as .gz file (which should be decompressed before use)
            with gzip.open(file, "w", compresslevel=compresslevel) as f:
                f.write(
                    self.dumps(
                        indent=indent, exclude_keys=exclude_keys, options=options
                    ).encode("utf-8")
                )

        # print the file path and name
        if file_path == "":
            print(
                'Results saved as "{}" in the current directory.'.format(
                    file[len(file_path) :]
                )
            )
        else:
            print(
                'Results saved as "{}" in the folder "{}".'.format(
                    file[len(file_path) :], file_path
                )
            )

    @classmethod
    def from_dict(cls, dictionary: dict):
        """
        Creates an Optimizer object from a dictionary (which is the output of the asdict method)
        Parameters
        ----------
        dictionary : dict
            A dictionary with the information of the Optimizer object.
        Returns
        -------
        QAOA or RQAOA
        """

        # check if the class is correct
        algorithm = dictionary["header"]["algorithm"]
        assert (
            algorithm.lower() == cls.__name__.lower()
        ), f"The class {cls.__name__} does not match the algorithm ({algorithm}) of the dictionary."

        # create the object
        obj = cls()

        # header
        obj.header = dictionary["header"].copy()
        obj.header.pop("metadata", None)  # remove the metadata from the header

        # tags
        obj.exp_tags = dictionary["data"]["exp_tags"].copy()

        # problem
        obj.problem = (
            QUBO.from_dict(dictionary["data"]["input_problem"])
            if dictionary["data"]["input_problem"] is not None
            else None
        )

        # input parameters
        map_inputs = {
            "backend_properties": obj.set_backend_properties,
            "circuit_properties": obj.set_circuit_properties,
            "classical_optimizer": obj.set_classical_optimizer,
            "rqaoa_parameters": obj.set_rqaoa_parameters
            if algorithm == "rqaoa"
            else None,
        }
        for key, value in dictionary["data"]["input_parameters"].items():
            if key == "device":
                continue
            map_inputs[key](**value)

        # results
        if (
            "result" in dictionary["data"].keys()
            and dictionary["data"]["result"] is not None
        ):
            obj.result = obj.results_class.from_dict(
                dictionary["data"]["result"],
                **(
                    {"cost_hamiltonian": obj.problem.hamiltonian}
                    if algorithm == "qaoa"
                    else {}
                ),
            )

        # print a message when the object is loaded
        print(f"Loaded {cls.__name__} object.")
        print("The device has to be set manually using the set_device method.")
        print(
            f"Name of the device used was: {dictionary['data']['input_parameters']['device']}"
        )

        return obj

    @classmethod
    def loads(cls, string: str):
        """
        Creates an Optimizer object from a string (which is the output of the dumps method)
        Parameters
        ----------
        string : str
            A string with the information of the Optimizer object.
        Returns
        -------
        QAOA or RQAOA
        """
        return cls.from_dict(json.loads(string))

    @classmethod
    def load(cls, file_name: str, file_path: str = ""):
        """
        Creates an Optimizer object from a file (which is the output of the dump method)
        Parameters
        ----------
        file_name : str
            The name of the file.
        file_path : str
            The path of the file.
        Returns
        -------
        QAOA or RQAOA
        """
        file = file_path + file_name
        if ".gz" == file_name[-3:]:
            with gzip.open(file, "r") as f:
                return cls.loads(f.read().decode("utf-8"))
        else:
            with open(file, "r") as f:
                return cls.loads(f.read())
