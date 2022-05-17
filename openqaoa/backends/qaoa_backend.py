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
from typing import Union, Optional, List
import numpy as np

from ..backends import (QAOAQiskitQPUBackend, QAOAPyQuilQPUBackend, QAOAPyQuilWavefunctionSimulatorBackend,
                        QAOAQiskitBackendStatevecSimulator, QAOAQiskitBackendShotBasedSimulator, QAOAvectorizedBackendSimulator,
                        AccessObjectBase, AccessObjectQiskit, AccessObjectPyQuil)
from ..qaoa_parameters.baseparams import QAOACircuitParams
from ..basebackend import QuantumCircuitBase, QAOABaseBackend

DEVICE_NAME_TO_OBJECT_MAPPER = {
    'qiskit_shot_simulator': QAOAQiskitBackendShotBasedSimulator,
    'qiskit_statevec_simulator': QAOAQiskitBackendStatevecSimulator,
    'qiskit_qasm_simulator': QAOAQiskitBackendShotBasedSimulator,
    'vectorized': QAOAvectorizedBackendSimulator,
    'pyquil_statevec_simulator': QAOAPyQuilWavefunctionSimulatorBackend,
    'pyquil_wavefunction_simulator': QAOAPyQuilWavefunctionSimulatorBackend
}

DEVICE_ACCESS_OBJECT_MAPPER = {
    AccessObjectQiskit: QAOAQiskitQPUBackend,
    AccessObjectPyQuil: QAOAPyQuilQPUBackend
}


def _backend_arg_mapper(backend_obj: QAOABaseBackend,
                        n_shots: Optional[int] = None,
                        qiskit_simulation_method: Optional[str] = None,
                        noise_model = None,
                        active_reset: Optional[bool] = None,
                        rewiring = None,
                        qubit_layout = None):

    BACKEND_ARGS_MAPPER = {
        QAOAvectorizedBackendSimulator: {},
        QAOAQiskitBackendStatevecSimulator: {},
        QAOAPyQuilWavefunctionSimulatorBackend: {},
        QAOAQiskitBackendShotBasedSimulator: {'n_shots': n_shots,
                                              'qiskit_simulation_method': qiskit_simulation_method,
                                              'noise_model': noise_model},
        QAOAQiskitQPUBackend: {'n_shots': n_shots,
                               'qubit_layout':qubit_layout},
        QAOAPyQuilQPUBackend: {'n_shots': n_shots,
                               'active_reset': active_reset,
                               'rewiring': rewiring,
                               'qubit_layout':qubit_layout}
    }

    final_backend_kwargs = {key: value for key, value in BACKEND_ARGS_MAPPER[backend_obj].items()
                            if value is not None}
    return final_backend_kwargs


def backend_name_to_object_mapper(device: Union[str, AccessObjectBase]):
    """
    Return the correct `QAOABaseBackend` object corresponding to the 
    requested backend name
    """
    if isinstance(device, str):
        try:
            backend_class = DEVICE_NAME_TO_OBJECT_MAPPER[device]
        except KeyError:
            raise ValueError(f"The device {device} is not supported."
                             f"Please choose from {DEVICE_NAME_TO_OBJECT_MAPPER.keys()}")
        except:
            raise Exception(
                f"The device name {device} raised an unknown error")

    elif isinstance(device, AccessObjectBase):
        try:
            backend_class = DEVICE_ACCESS_OBJECT_MAPPER[type(device)]
        except KeyError:
            raise ValueError(f"The access object {device} is not supported."
                             f"Please choose from {DEVICE_ACCESS_OBJECT_MAPPER.keys()}")
        except:
            raise Exception(f"The specified {device} raised an unknown error")
    else:
        raise ValueError(f"Uknown device information type {type(device)}")

    return backend_class

# def available_backend_providers():
#     """
#     List of all available backends via OpenQAOA

#     TODO: The QPU list should be automatedly generated.
#     """
#     return list(BACKEND_PROVIDERS_MAPPER.keys())


def get_qaoa_backend(circuit_params: QAOACircuitParams,
                     device: Union[str, AccessObjectBase],
                     prepend_state: Optional[Union[QuantumCircuitBase,
                                                   List[complex], np.ndarray]] = None,
                     append_state: Optional[Union[QuantumCircuitBase,
                                                  List[complex], np.ndarray]] = None,
                     init_hadamard: bool = True,
                     cvar_alpha: float = 1,
                     **kwargs):
    """
    A wrapper function to return a QAOA backend object.

    Parameters
    ----------
    circuit_params: `QAOACircuitParams`
        An object of the class ``QAOACircuitParams`` which contains information on 
        circuit construction along with depth of the circuit.
    device: `str` or `AccessObjectBase`
        The device to be used: Specified as a string or an object of the class ``AccessObjectBase``.
    prepend_state: `Union[QuantumCircuitBase,np.ndarray(complex)]`
        The state prepended to the circuit.
    append_state: `Union[QuantumCircuitBase,np.ndarray(complex)]`
        The state appended to the circuit.
    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the 
        QAOA part of the circuit.
    cvar_alpha: `float`
        The value of the CVaR parameter.
    kwargs:
    Additional keyword arguments for the backend.
        qubit_layout: `list`
            A list of physical qubits to be used for the QAOA circuit.
        n_shots: `int`
            The number of shots to be used for the shot-based computation.

    Returns
    -------
    `QAOABaseBackend`
        The corresponding backend object.
    """
    if isinstance(device, str):
        simulator_name = device.lower()
        # try:
        backend_class = backend_name_to_object_mapper(simulator_name)
        backend_kwargs = _backend_arg_mapper(backend_class, **kwargs)
        backend_obj = backend_class(circuit_params=circuit_params, prepend_state=prepend_state,
                                    append_state=append_state, init_hadamard=init_hadamard,
                                    cvar_alpha=cvar_alpha, **backend_kwargs)
        # except TypeError:
        #     raise TypeError(
        #         'Please make sure the **kwargs are supported by the chosen backend')

        # except Exception as e:
        #     raise ValueError(
        #         f"{e} The backend {simulator_name} returned an error")

    elif isinstance(device, AccessObjectBase):
        device_access_object = device
        try:
            backend_class = backend_name_to_object_mapper(device_access_object)
            backend_kwargs = _backend_arg_mapper(backend_class, **kwargs)
            backend_obj = backend_class(circuit_params=circuit_params, access_object=device_access_object,
                                        prepend_state=prepend_state, append_state=append_state,
                                        init_hadamard=init_hadamard, cvar_alpha=cvar_alpha,
                                        **backend_kwargs)

        except TypeError:
            raise TypeError(
                'Please make sure the **kwargs are supported by the chosen backend')

        except Exception as e:
            raise ValueError(f"{e} The backend returned an error")

    else:
        raise ValueError(
            f"Specified device argument {device} is not a supported type")

    return backend_obj
