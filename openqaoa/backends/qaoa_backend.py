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
                        QAOAQiskitBackendStatevecSimulator, QAOAQiskitBackendShotBasedSimulator, 
                        QAOAvectorizedBackendSimulator)

from ..devices import DeviceBase, DeviceLocal, DevicePyquil, DeviceQiskit
from ..qaoa_parameters.baseparams import QAOACircuitParams
from ..basebackend import QuantumCircuitBase, QAOABaseBackend

DEVICE_NAME_TO_OBJECT_MAPPER = {
    'qiskit.qasm_simulator': QAOAQiskitBackendShotBasedSimulator,
    'qiskit.shot_simulator': QAOAQiskitBackendShotBasedSimulator,
    'qiskit.statevector_simulator': QAOAQiskitBackendStatevecSimulator,
    'vectorized': QAOAvectorizedBackendSimulator,
    'pyquil.statevector_simulator': QAOAPyQuilWavefunctionSimulatorBackend
}

DEVICE_ACCESS_OBJECT_MAPPER = {
    DeviceQiskit: QAOAQiskitQPUBackend,
    DevicePyquil: QAOAPyQuilQPUBackend
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


def device_to_backend_mapper(device: DeviceBase) -> QAOABaseBackend:
    """
    Return the correct `QAOABaseBackend` object corresponding to the 
    requested device
    """
    if isinstance(device, DeviceLocal):
        try:
            backend_class = DEVICE_NAME_TO_OBJECT_MAPPER[device.device_name]
        except KeyError:
            raise ValueError(f"The device {device} is not supported."
                             f"Please choose from {DEVICE_NAME_TO_OBJECT_MAPPER.keys()}")
        except:
            raise Exception(
                f"The device name {device} raised an unknown error")

    else:
        try:
            backend_class = DEVICE_ACCESS_OBJECT_MAPPER[type(device)]
        except KeyError:
            raise ValueError(f"The device {device} is not supported."
                             f"Please choose from {DEVICE_ACCESS_OBJECT_MAPPER.keys()}")
        except:
            raise Exception(f"The specified {device} raised an unknown error")

    return backend_class

def get_qaoa_backend(circuit_params: QAOACircuitParams,
                     device: DeviceBase,
                     prepend_state: Optional[Union[QuantumCircuitBase,
                                                   List[complex], np.ndarray]] = None,
                     append_state: Optional[Union[QuantumCircuitBase,
                                                  np.ndarray]] = None,
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
    device: `DeviceBase`
        The device to be used: Specified as an object of the class `DeviceBase`.
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

    backend_class = device_to_backend_mapper(device)
    backend_kwargs = _backend_arg_mapper(backend_class, **kwargs)

    try:
        if isinstance(device, DeviceLocal):
            backend_obj = backend_class(circuit_params=circuit_params, prepend_state=prepend_state,
                                        append_state=append_state, init_hadamard=init_hadamard,
                                        cvar_alpha=cvar_alpha, **backend_kwargs)

        else:
            backend_obj = backend_class(circuit_params=circuit_params, device=device,
                                        prepend_state=prepend_state, append_state=append_state,
                                        init_hadamard=init_hadamard, cvar_alpha=cvar_alpha,
                                        **backend_kwargs)
    except Exception as e:
        raise ValueError(f"The backend returned an error: {e}")

    return backend_obj
