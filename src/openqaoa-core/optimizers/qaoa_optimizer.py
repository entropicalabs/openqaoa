from .training_vqa import (
    ScipyOptimizer,
    CustomScipyGradientOptimizer,
    PennyLaneOptimizer,
)
from ..qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)
from ..backends.basebackend import VQABaseBackend


def available_optimizers():
    """
    Return a list of available optimizers.
    """

    optimizers = {
        "scipy": ScipyOptimizer.SCIPY_METHODS,
        "custom_scipy_gradient": CustomScipyGradientOptimizer.CUSTOM_GRADIENT_OPTIMIZERS,
        "custom_scipy_pennylane": PennyLaneOptimizer.PENNYLANE_OPTIMIZERS,
    }

    return optimizers


def get_optimizer(
    vqa_object: VQABaseBackend,
    variational_params: QAOAVariationalBaseParams,
    optimizer_dict: dict,
):
    """
    Initialise the specified optimizer class with provided method and
    optimizer-specific options

    Parameters
    ----------
    vqa_object:
        Backend object of class VQABaseBackend which contains information on the
        backend used to perform computations, and the VQA circuit.

    variational_params:
        Object of class QAOAVariationalBaseParams, which contains
        information on the circuit to be executed,
        the type of parametrisation, and the angles of the VQA circuit.

    optimizer_dict:
        Optimizer information dictionary used to construct
        the optimizer with specified options

    Returns
    -------
    optimizer:
        Optimizer object of type specified by specified method
    """
    SUPPORTED_OPTIMIZERS = {
        "scipy": ScipyOptimizer,
        "custom_scipy_gradient": CustomScipyGradientOptimizer,
        "custom_scipy_pennylane": PennyLaneOptimizer,
    }

    method = optimizer_dict["method"].lower()
    optimizers = available_optimizers()

    method_valid = False
    for opt_class, methods in optimizers.items():
        if method in methods:
            selected_class = opt_class
            method_valid = True

    assert method_valid, ValueError(
        f"Selected optimizer method '{method}' is not supported."
        f"Please choose from {available_optimizers()}"
    )

    optimizer = SUPPORTED_OPTIMIZERS[selected_class](
        vqa_object, variational_params, optimizer_dict
    )

    return optimizer
