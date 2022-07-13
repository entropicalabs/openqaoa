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

from typing import List, Union, Optional, Tuple
import numpy as np
from .baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from .annealingparams import QAOAVariationalAnnealingParams
from .fourierparams import (QAOAVariationalFourierParams, QAOAVariationalFourierExtendedParams,
                            QAOAVariationalFourierWithBiasParams)
from .extendedparams import QAOAVariationalExtendedParams
from .standardparams import QAOAVariationalStandardParams, QAOAVariationalStandardWithBiasParams

VARIATIONAL_PARAMS_DICT_KEYS = {'standard': ['betas', 'gammas'],
                                'standard_w_bias': ['betas', 'gammas_singles', 'gammas_pairs'],
                                'extended': ['betas_singles', 'betas_pairs', 'gammas_singles', 'gammas_pairs'],
                                'fourier': ['q', 'v', 'u'],
                                'fourier_extended': ['q', 'v_singles', 'v_pairs', 'u_singles', 'u_pairs'],
                                'fourier_w_bias': ['q', 'v', 'u_singles', 'u_pairs'],
                                'annealing': ['total_annealing_time', 'schedule']}

PARAMS_CLASSES_MAPPER = {'standard': QAOAVariationalStandardParams,
                         'standard_w_bias': QAOAVariationalStandardWithBiasParams,
                         'extended': QAOAVariationalExtendedParams,
                         'fourier': QAOAVariationalFourierParams,
                         'fourier_extended': QAOAVariationalFourierExtendedParams,
                         'fourier_w_bias': QAOAVariationalFourierWithBiasParams,
                         'annealing': QAOAVariationalAnnealingParams
                         }

SUPPORTED_INITIALIZATION_TYPES = ['rand', 'ramp', 'custom']


def _qaoa_variational_params_args(params_type: str,
                                  init_type: str,
                                  betas: Optional[Union[List,
                                                        np.ndarray]] = None,
                                  gammas: Optional[Union[List,
                                                         np.ndarray]] = None,
                                  betas_singles: Optional[Union[List,
                                                                np.ndarray]] = None,
                                  betas_pairs: Optional[Union[List,
                                                              np.ndarray]] = None,
                                  gammas_singles: Optional[Union[List,
                                                                 np.ndarray]] = None,
                                  gammas_pairs: Optional[Union[List,
                                                               np.ndarray]] = None,
                                  q: Optional[int] = None,
                                  v: Optional[Union[List, np.ndarray]] = None,
                                  u: Optional[Union[List, np.ndarray]] = None,
                                  v_singles: Optional[Union[List,
                                                            np.ndarray]] = None,
                                  v_pairs: Optional[Union[List,
                                                          np.ndarray]] = None,
                                  u_singles: Optional[Union[List,
                                                            np.ndarray]] = None,
                                  u_pairs: Optional[Union[List,
                                                          np.ndarray]] = None,
                                  total_annealing_time: Optional[float] = None,
                                  schedule: Optional[float] = None) -> Tuple[Union[List, np.ndarray]]:
    """
    Provided the given parameterisation type return the 
    variational parameters arguments as a tuple

    Parameters
    ----------
    params_type: ``str``
        Chosen QAOA parameterisation strategy 
    gammas: ``Union[List, np.ndarray]``
    betas: ``Union[List, np.ndarray]``
    gammas_singles: ``Union[List, np.ndarray]``
    gammas_pairs: ``Union[List, np.ndarray]``
    betas_singles: ``Union[List, np.ndarray]``
    betas_pairs: ``Union[List, np.ndarray]``
    q: ``int``
    v: ``Union[List, np.ndarray]``
    u: ``Union[List, np.ndarray]``
    v_singles: ``Union[List, np.ndarray]``
    v_pairs: ``Union[List, np.ndarray]``
    u_singles: ``Union[List, np.ndarray]``
    u_pairs: ``Union[List, np.ndarray]``
    total_annealing_time: ``float``
    schedule: ``Union[List, np.ndarray]``

    Returns
    -------
    variational_params_args: ``Tuple[Union[List,np.ndarray]]``
        Tuple object containing the correct angles for the specified parameterisation type
    """
    VARIATIONAL_PARAMS_MAPPER = {'standard': (betas, gammas),
                                 'standard_w_bias': (betas, gammas_singles, gammas_pairs),
                                 'extended': (betas_singles, betas_pairs, gammas_singles, gammas_pairs),
                                 'fourier': (q, v, u),
                                 'fourier_extended': (q, v_singles, v_pairs, u_singles, u_pairs),
                                 'fourier_w_bias': (q, v, u_singles, u_pairs),
                                 'annealing': (total_annealing_time, schedule)
                                 }
    final_variational_params = VARIATIONAL_PARAMS_MAPPER[params_type.lower()]
    if init_type != 'custom':
        final_variational_params = tuple(param for param in final_variational_params
                                         if param is not None)

    return final_variational_params


def create_qaoa_variational_params(qaoa_circuit_params: QAOACircuitParams,
                                   params_type: str,
                                   init_type: str,
                                   variational_params_dict: dict = {},
                                   linear_ramp_time: Optional[float] = None,
                                   q: Optional[int] = None,
                                   total_annealing_time: Optional[float] = None,
                                   seed: int = None) -> QAOAVariationalBaseParams:
    """
    Create QAOA Variational Parameters of the specified type.

    Parameters
    ----------
    qaoa_circuit_params: ``QAOACircuitParams``
        QAOA Circuit Parameters containing information on
        mixer and cost Hamiltonians.
    params_type: ``str``
        Type of variational parameters to be created.
    init_type: ``str``
        Type of initialisation to be used.
    variational_params_dict: ``dict``
        Dictionary containing the variational angles. It can only have the following
        keys (with corresponding angle values):
        For example, {'betas': [0.1, 0.2, 0.3], 'gammas': [0.1, 0.2, 0.3]}
    linear_ramp_time: ``float``
        Total time for the linear ramp initialisation method for parameters.
    Returns
    -------
        QAOAVariationalBaseParams object of the required type
    """

    params_type = params_type.lower()
    init_type = init_type.lower()
    variational_params_dict.update({'q': q})
    variational_params_dict.update(
        {'total_annealing_time': total_annealing_time})
    variational_params_args = _qaoa_variational_params_args(params_type,
                                                            init_type,
                                                            **variational_params_dict)

    try:
        params_class = PARAMS_CLASSES_MAPPER[params_type]
    except KeyError:
        raise ValueError(f"{params_type} Parameterisation is not supported."
                         f" Choose from {PARAMS_CLASSES_MAPPER.keys()}")

    if init_type == 'custom':
        try:
            qaoa_variational_params = params_class(
                qaoa_circuit_params, *variational_params_args)
        except:
            raise ValueError(f"For the selected {params_type} parameterisation, please specify a"
                             f" dictionary with correct {VARIATIONAL_PARAMS_DICT_KEYS[params_type]} keys")
    elif init_type == 'ramp':

        if isinstance(linear_ramp_time, float) or isinstance(linear_ramp_time, int):
            assert linear_ramp_time > 0, "Please specify the linear ramp time. Only positive values are allowed."
        elif linear_ramp_time is None:
            pass
        else:
            raise ValueError(f"Please specify a numeric value for linear_ramp_time.")
            
        qaoa_variational_params = params_class.linear_ramp_from_hamiltonian(qaoa_circuit_params,
                                                                            *variational_params_args,
                                                                            time = linear_ramp_time)
    elif init_type == 'rand':
        qaoa_variational_params = params_class.random(qaoa_circuit_params,
                                                      *variational_params_args,
                                                      seed = seed)
    else:
        raise ValueError(f"{init_type} Initialisation strategy is not yet supported."
                         f" Please choose from {SUPPORTED_INITIALIZATION_TYPES}")

    return qaoa_variational_params

def qaoa_variational_params_converter(target_params_type: str,
                                      current_params_obj: QAOAVariationalBaseParams) -> QAOAVariationalBaseParams:
    """
    Convert the current variational parameters object to the target variational parameters object.

    Parameters
    ----------
    target_params_type: ``str``
        Type of variational parameters to be created.
    current_params_obj: ``QAOAVariationalBaseParams``
        Current variational parameters object.

    Returns
    -------
    converted_params_obj: ``QAOAVariationalBaseParams``
        Converted variational parameters object.
    """

    try:
        new_params_class = PARAMS_CLASSES_MAPPER[target_params_type]
    except KeyError:
        raise ValueError(f"{target_params_type} Parameterisation is not supported."
                         f" Choose from {PARAMS_CLASSES_MAPPER.keys()}")

    converted_params_obj = new_params_class.from_other_parameters(current_params_obj)

    return converted_params_obj