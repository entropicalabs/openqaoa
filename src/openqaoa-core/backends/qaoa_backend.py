from typing import Union, Optional, List
import numpy as np

from .plugin_finder import plugin_finder_dict
from .qaoa_vectorized import QAOAvectorizedBackendSimulator
from .qaoa_analytical_sim import QAOABackendAnalyticalSimulator
from .devices_core import DeviceBase, DeviceLocal
from .basebackend import QuantumCircuitBase, QAOABaseBackend
from ..qaoa_components import QAOADescriptor

PLUGIN_DICT = plugin_finder_dict()

def _create_mappers(input_plugin_dict: dict) -> dict:
    
    DEVICE_NAME_TO_OBJECT_MAPPER = dict()
    DEVICE_ACCESS_OBJECT_MAPPER = dict()
    
    DEVICE_NAME_TO_OBJECT_MAPPER["vectorized"] = QAOAvectorizedBackendSimulator
    DEVICE_NAME_TO_OBJECT_MAPPER["analytical_simulator"] = QAOABackendAnalyticalSimulator
    
    for each_entry_key, each_entry_value in input_plugin_dict.items():
        if hasattr(each_entry_value, 'device_access'):
            DEVICE_ACCESS_OBJECT_MAPPER.update(each_entry_value.device_access)
        if hasattr(each_entry_value, 'device_name_to_obj'):
            DEVICE_NAME_TO_OBJECT_MAPPER.update(each_entry_value.device_name_to_obj)
        
    return DEVICE_NAME_TO_OBJECT_MAPPER, DEVICE_ACCESS_OBJECT_MAPPER

DEVICE_NAME_TO_OBJECT_MAPPER, DEVICE_ACCESS_OBJECT_MAPPER = _create_mappers(PLUGIN_DICT)

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
    
    local_vars = locals()
    
    for each_plugin_entrypoint in PLUGIN_DICT.values():
        if hasattr(each_plugin_entrypoint, 'backend_args'):
            for each_key, each_value in each_plugin_entrypoint.backend_args.items():
                # Convert list of accepted parameters into a dictionary with
                # the name of the variable as a key and the local value of the 
                # variable
                var_values = [local_vars[each_name] for each_name in each_value]
                input_dict = {each_key: dict(zip(each_value, var_values))}
                BACKEND_ARGS_MAPPER.update(input_dict)

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
