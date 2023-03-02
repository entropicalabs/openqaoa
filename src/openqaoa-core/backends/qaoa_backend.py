from typing import Union, Optional, List
import numpy as np

from .plugin_finder import backend_boolean_list
from .qaoa_vectorized import QAOAvectorizedBackendSimulator
from .qaoa_analytical_sim import QAOABackendAnalyticalSimulator
from .devices_core import DeviceBase, DeviceLocal
from .basebackend import QuantumCircuitBase, QAOABaseBackend
from ..qaoa_components import QAOADescriptor

BACKEND_BOOLEAN_LIST = backend_boolean_list()

def _create_mappers(input_boolean_list: list) -> dict:
    
    DEVICE_NAME_TO_OBJECT_MAPPER = dict()
    DEVICE_ACCESS_OBJECT_MAPPER = dict()
    
    DEVICE_NAME_TO_OBJECT_MAPPER["vectorized"] = QAOAvectorizedBackendSimulator
    DEVICE_NAME_TO_OBJECT_MAPPER["analytical_simulator"] = QAOABackendAnalyticalSimulator
    
    if BACKEND_BOOLEAN_LIST[0]:
        from openqaoa_braket.utilities import device_access
        DEVICE_ACCESS_OBJECT_MAPPER.update(device_access)
        
    if BACKEND_BOOLEAN_LIST[1] and input_boolean_list[2]:
        from openqaoa_azure.utilities import device_access
        DEVICE_ACCESS_OBJECT_MAPPER.update(device_access)
    
    if BACKEND_BOOLEAN_LIST[2]:
        from openqaoa_qiskit.utilities import device_access, device_name_to_obj
        DEVICE_NAME_TO_OBJECT_MAPPER.update(device_name_to_obj)
        DEVICE_ACCESS_OBJECT_MAPPER.update(device_access)
        
    if BACKEND_BOOLEAN_LIST[3]:
        from openqaoa_pyquil.utilities import device_access, device_name_to_obj
        DEVICE_NAME_TO_OBJECT_MAPPER.update(device_name_to_obj)
        DEVICE_ACCESS_OBJECT_MAPPER.update(device_access)
        
    return DEVICE_NAME_TO_OBJECT_MAPPER, DEVICE_ACCESS_OBJECT_MAPPER

DEVICE_NAME_TO_OBJECT_MAPPER, DEVICE_ACCESS_OBJECT_MAPPER = _create_mappers(backend_boolean_list())

def _backend_arg_mapper(
    backend_obj: QAOABaseBackend,
    n_shots: Optional[int] = None,
    seed_simulator: Optional[int] = None,
    qiskit_simulation_method: Optional[str] = None,
    noise_model=None,
    active_reset: Optional[bool] = None,
    rewiring=None,
    disable_qubit_rewiring: Optional[bool] = None,
    initial_qubit_mapping=None,
):
    
    BACKEND_ARGS_MAPPER = {
        QAOABackendAnalyticalSimulator: {},
        QAOAvectorizedBackendSimulator: {},
    }

    if BACKEND_BOOLEAN_LIST[0]:
        from openqaoa_braket.backends import DeviceAWS, QAOAAWSQPUBackend
        BACKEND_ARGS_MAPPER[QAOAAWSQPUBackend] = {
            "n_shots": n_shots,
            "disable_qubit_rewiring": disable_qubit_rewiring,
            "initial_qubit_mapping": initial_qubit_mapping
        }
    if BACKEND_BOOLEAN_LIST[2]:
        from openqaoa_qiskit.backends import (
            DeviceQiskit,
            QAOAQiskitQPUBackend,
            QAOAQiskitBackendShotBasedSimulator,
            QAOAQiskitBackendStatevecSimulator,
        )
        BACKEND_ARGS_MAPPER[QAOAQiskitBackendStatevecSimulator] = {}
        BACKEND_ARGS_MAPPER[QAOAQiskitBackendShotBasedSimulator] = {
            "n_shots": n_shots,
            "seed_simulator": seed_simulator,
            "qiskit_simulation_method": qiskit_simulation_method,
            "noise_model": noise_model,
            "initial_qubit_mapping": initial_qubit_mapping,
        }
        BACKEND_ARGS_MAPPER[QAOAQiskitQPUBackend] = {
            "n_shots": n_shots,
            "initial_qubit_mapping": initial_qubit_mapping,
        }
    if BACKEND_BOOLEAN_LIST[3]:
        from openqaoa_pyquil.backends import (
            DevicePyquil,
            QAOAPyQuilQPUBackend,
            QAOAPyQuilWavefunctionSimulatorBackend
        )
        BACKEND_ARGS_MAPPER[QAOAPyQuilWavefunctionSimulatorBackend] = {}
        BACKEND_ARGS_MAPPER[QAOAPyQuilQPUBackend] = {
            "n_shots": n_shots,
            "active_reset": active_reset,
            "rewiring": rewiring,
            "initial_qubit_mapping": initial_qubit_mapping,
        }

    final_backend_kwargs = {
        key: value
        for key, value in BACKEND_ARGS_MAPPER[backend_obj].items()
        if value is not None
    }
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
            raise ValueError(
                f"The device {device} is not supported."
                f"Please choose from {DEVICE_NAME_TO_OBJECT_MAPPER.keys()}"
            )
        except:
            raise Exception(f"The device name {device} raised an unknown error")

    else:
        try:
            backend_class = DEVICE_ACCESS_OBJECT_MAPPER[type(device)]
        except KeyError:
            raise ValueError(
                f"The device {device} is not supported."
                f"Please choose from {DEVICE_ACCESS_OBJECT_MAPPER.keys()}"
            )
        except:
            raise Exception(f"The specified {device} raised an unknown error")

    return backend_class


def get_qaoa_backend(
    qaoa_descriptor: QAOADescriptor,
    device: DeviceBase,
    prepend_state: Optional[
        Union[QuantumCircuitBase, List[complex], np.ndarray]
    ] = None,
    append_state: Optional[Union[QuantumCircuitBase, np.ndarray]] = None,
    init_hadamard: bool = True,
    cvar_alpha: float = 1,
    **kwargs,
):
    """
    A wrapper function to return a QAOA backend object.

    Parameters
    ----------
    qaoa_descriptor: `QAOADescriptor`
        An object of the class ``QAOADescriptor`` which contains information on
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
        initial_qubit_mapping: `list`
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
            backend_obj = backend_class(
                qaoa_descriptor=qaoa_descriptor,
                prepend_state=prepend_state,
                append_state=append_state,
                init_hadamard=init_hadamard,
                cvar_alpha=cvar_alpha,
                **backend_kwargs,
            )

        else:
            backend_obj = backend_class(
                qaoa_descriptor=qaoa_descriptor,
                device=device,
                prepend_state=prepend_state,
                append_state=append_state,
                init_hadamard=init_hadamard,
                cvar_alpha=cvar_alpha,
                **backend_kwargs,
            )
    except Exception as e:
        raise ValueError(f"The backend returned an error: {e}")

    return backend_obj
