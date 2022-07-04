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

from ..qaoa_parameters.baseparams import QAOAVariationalBaseParams
from .training_vqa import ScipyOptimizer, CustomScipyGradientOptimizer
from ..basebackend import VQABaseBackend


def available_optimizers():
    """
    Return a list of available optimizers.
    """

    optimizers = {
        'scipy': ScipyOptimizer.SCIPY_METHODS,
        'custom_scipy_gradient': CustomScipyGradientOptimizer.CUSTOM_GRADIENT_OPTIMIZERS
    }

    return optimizers


def get_optimizer(vqa_object: VQABaseBackend,
                  variational_params: QAOAVariationalBaseParams,
                  optimizer_dict: dict):
    """
    Initialise the specified optimizer class with provided method and optimizer-specific options

    Parameters
    ----------
    vqa_object:
        Backend object of class VQABaseBackend which contains information on the backend used to perform computations, and the VQA circuit.

    variational_params:
        Object of class QAOAVariationalBaseParams, which contains information on the circuit to be executed,  
        the type of parametrisation, and the angles of the VQA circuit.

    optimizer_dict:
        Optimizer information dictionary used to construct the optimizer with specified options

    Returns
    -------
    optimizer:
        Optimizer object of type specified by specified method
    """
    SUPPORTED_OPTIMIZERS = {
        'scipy': ScipyOptimizer,
        'custom_scipy_gradient': CustomScipyGradientOptimizer
    }

    method = optimizer_dict['method'].lower()
    optimizers = available_optimizers()

    for opt_class, methods in optimizers.items():
        if method in methods:
            selected_class = opt_class

    optimizer = SUPPORTED_OPTIMIZERS[selected_class](vqa_object, variational_params,
                                                         optimizer_dict)
   
    return optimizer
