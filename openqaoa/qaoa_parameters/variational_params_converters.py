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

"""
Conversion functions for the different QAOA Parametrizations. So far we only
only support going from less to more specialiced parametrizations. The type
tree looks as follows:
   Extended   <--------- FourierExtended
       ^                      ^
       |                      |
StandardWithBias <------ FourierWithBias
       ^                      ^
       |                      |
    Standard  <----------- Fourier
       ^
       |
    Annealing
"""
from __future__ import annotations
from typing import Type

from copy import deepcopy
import numpy as np
from scipy.fftpack import dct, dst

from . import (annealingparams,
               standardparams,
               fourierparams,
               extendedparams)

QAOAVariationalAnnealingParams = annealingparams.QAOAVariationalAnnealingParams
QAOAVariationalStandardParams = standardparams.QAOAVariationalStandardParams
QAOAVariationalExtendedParams = extendedparams.QAOAVariationalExtendedParams
QAOAVariationalStandardWithBiasParams = standardparams.QAOAVariationalStandardWithBiasParams
QAOAVariationalFourierParams = fourierparams.QAOAVariationalFourierParams
QAOAVariationalFourierWithBiasParams = fourierparams.QAOAVariationalFourierWithBiasParams
QAOAVariationalFourierExtendedParams = fourierparams.QAOAVariationalFourierExtendedParams


def annealing_to_standard(
		params: QAOAVariationalAnnealingParams) -> QAOAVariationalStandardParams:
    out = deepcopy(params)
    out.__class__ = QAOAVariationalStandardParams
    out.betas = params._annealing_time * (1 - params.schedule) / params.p
    out.gammas = params._annealing_time * params.schedule / params.p

    # and clean up after us
    del out.__schedule
    del out._annealing_time

    return out


def standard_to_standard_w_bias(
        params: QAOAVariationalStandardParams) -> QAOAVariationalStandardWithBiasParams:
    out = deepcopy(params)
    out.__class__ = QAOAVariationalStandardWithBiasParams
    out.gammas_singles = params.gammas
    out.gammas_pairs = params.gammas

    # and clean up after us
    del out.__gammas

    return out


def standard_w_bias_to_extended(
		params: QAOAVariationalStandardWithBiasParams) -> QAOAVariationalExtendedParams:
	out = deepcopy(params)
	out.__class__ = QAOAVariationalExtendedParams
	out.betas_singles = np.outer(params.betas, np.ones(len(params.mixer_1q_coeffs)))
	out.betas_pairs = np.outer(params.betas, np.ones(len(params.mixer_2q_coeffs)))

	out.gammas_singles = np.outer(params.gammas_singles,
									np.ones(len(params.cost_1q_coeffs)))
	out.gammas_pairs = np.outer(params.gammas_pairs,
								np.ones(len(params.cost_2q_coeffs)))
	return out


def fourier_to_standard(
	params: QAOAVariationalFourierParams) -> QAOAVariationalStandardParams:
    out = deepcopy(params)
    out.__class__ = QAOAVariationalStandardParams
    out.betas = dct(params.v, n=out.p)
    out.gammas = dst(params.u, n=out.p)

    # and clean up
    del out.__u
    del out.__v
    del out.q

    return out


def fourier_w_bias_to_standard_w_bias(
        params: QAOAVariationalFourierWithBiasParams) -> QAOAVariationalStandardWithBiasParams:
    out = deepcopy(params)
    out.__class__ = QAOAVariationalStandardWithBiasParams
    out.betas = dct(params.v, n=out.p)
    out.gammas_singles = dst(params.u_singles, n=out.p)
    out.gammas_pairs = dst(params.u_pairs, n=out.p)

    # and clean up
    del out.__u_singles
    del out.__u_pairs
    del out.__v
    del out.q

    return out


def fourier_to_fourier_w_bias(
		params: QAOAVariationalFourierParams) -> QAOAVariationalFourierWithBiasParams:
    out = deepcopy(params)
    out.__class__ = QAOAVariationalFourierWithBiasParams
    out.u_singles = params.u
    out.u_pairs = params.u

    # and clean up
    del out.__u

    return out


def fourier_w_bias_to_fourier_extended(
		params: QAOAVariationalFourierWithBiasParams) -> QAOAVariationalFourierExtendedParams:
	out = deepcopy(params)
	out.__class__ = QAOAVariationalFourierExtendedParams
	out.v_singles = np.outer(params.v, np.ones(len(params.mixer_1q_coeffs)))
	out.v_pairs = np.outer(params.v, np.ones(len(params.mixer_2q_coeffs)))
	out.u_singles = np.outer(params.u_singles,
								np.ones(len(params.cost_1q_coeffs)))
	out.u_pairs = np.outer(params.u_pairs,
							np.ones(len(params.cost_2q_coeffs)))

	del out.__v

	return out


def fourier_extended_to_extended(
        params: QAOAVariationalFourierExtendedParams) -> QAOAVariationalExtendedParams:
	out = deepcopy(params)
	out.__class__ = QAOAVariationalExtendedParams
	out.betas_singles = dct(params.v_singles, n=out.p, axis=0)
	out.betas_pairs = dct(params.v_pairs, n=out.p, axis=0)
	out.gammas_singles = dst(params.u_singles, n=out.p, axis=0)
	out.gammas_pairs = dst(params.u_pairs, n=out.p, axis=0)

    # and clean up
	del out.__u_singles
	del out.__u_pairs
	del out.__v_singles
	del out.__v_pairs
	del out.q

	return out


# #############################################################################
# And now all the possible compositions as well:
# Todo: Create this code automatically by traversing the tree?
# #############################################################################

def annealing_to_standard_w_bias(
        params: QAOAVariationalAnnealingParams) -> QAOAVariationalStandardWithBiasParams:
    return standard_to_standard_w_bias(annealing_to_standard(params))


def annealing_to_extended(
        params: QAOAVariationalAnnealingParams) -> QAOAVariationalExtendedParams:
    return standard_w_bias_to_extended(annealing_to_standard_w_bias(params))


def standard_to_extended(
        params: QAOAVariationalStandardParams) -> QAOAVariationalExtendedParams:
    return standard_w_bias_to_extended(standard_to_standard_w_bias(params))


def fourier_to_fourier_extended(
        params: QAOAVariationalFourierParams) -> QAOAVariationalFourierExtendedParams:
    return fourier_w_bias_to_fourier_extended(
                fourier_to_fourier_w_bias(params))


def fourier_to_standard_w_bias(
        params: QAOAVariationalFourierParams) -> QAOAVariationalStandardWithBiasParams:
    return standard_to_standard_w_bias(fourier_to_standard(params))


def fourier_to_extended(
        params: QAOAVariationalFourierParams) -> QAOAVariationalExtendedParams:
    return standard_w_bias_to_extended(fourier_to_standard_w_bias(params))


def fourier_w_bias_to_extended(
        params: QAOAVariationalFourierWithBiasParams) -> QAOAVariationalExtendedParams:
    return standard_w_bias_to_extended(
            fourier_w_bias_to_standard_w_bias(params))


# A dict with all the conversion functions accessible by
# (input_type, output_type)
conversion_functions =\
    {
     (QAOAVariationalFourierExtendedParams, QAOAVariationalExtendedParams): fourier_extended_to_extended,
     (QAOAVariationalFourierWithBiasParams, QAOAVariationalStandardWithBiasParams):\
     fourier_w_bias_to_standard_w_bias,
     (QAOAVariationalFourierParams, QAOAVariationalStandardParams): fourier_to_standard,
     (QAOAVariationalAnnealingParams, QAOAVariationalStandardParams): annealing_to_standard,
     (QAOAVariationalStandardParams, QAOAVariationalStandardWithBiasParams): standard_to_standard_w_bias,
     (QAOAVariationalStandardWithBiasParams, QAOAVariationalExtendedParams): standard_w_bias_to_extended,
     (QAOAVariationalFourierParams, QAOAVariationalFourierWithBiasParams): fourier_to_fourier_w_bias,
     (QAOAVariationalFourierWithBiasParams, QAOAVariationalFourierExtendedParams):\
     fourier_w_bias_to_fourier_extended,
     (QAOAVariationalAnnealingParams, QAOAVariationalStandardWithBiasParams): annealing_to_standard_w_bias,
     (QAOAVariationalAnnealingParams, QAOAVariationalExtendedParams): annealing_to_extended,
     (QAOAVariationalStandardParams, QAOAVariationalExtendedParams): standard_to_extended,
     (QAOAVariationalFourierParams, QAOAVariationalFourierExtendedParams): fourier_to_fourier_extended,
     (QAOAVariationalFourierParams, QAOAVariationalStandardWithBiasParams): fourier_to_standard_w_bias,
     (QAOAVariationalFourierParams, QAOAVariationalExtendedParams): fourier_to_extended,
     (QAOAVariationalFourierWithBiasParams, QAOAVariationalExtendedParams): fourier_w_bias_to_extended,
    }


def converter(params, target_type: type):
    """
    Convert ``params`` to type ``target_type``
    Parameters
    ----------
    params:
        The input parameters
    target_type:
        The target type
    Returns
    -------
    target_type:
        The converted parameters
    Raises
    ------
    TypeError:
        If conversion from type(params) to target_type is not supported.
    """
    try:
        out = conversion_functions[(type(params), target_type)](params)
        return out
    except KeyError:
        raise TypeError(f"Conversion from {type(params)} to {target_type} "
                        "not supported.")