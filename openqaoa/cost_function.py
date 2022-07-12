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

# Cost function to be used for QAOA training
from typing import Dict
from collections import OrderedDict
from .qaoa_parameters.operators import Hamiltonian
from .utilities import bitstring_energy


def expectation_value_classical(counts: Dict, hamiltonian: Hamiltonian):
    """
    Evaluate the cost function, i.e. expectation value ``$$\langle|H \rangle$$``
    w.r.t to the measurements results ``counts``.

    Parameters
    ----------
    counts: dict
            The counts of the measurements.
    hamiltonian: Hamiltonian	
            The Cost Hamiltonian defined for the optimization problem
    """
    shots = sum(counts.values())
    cost = 0
    for basis_state, count in counts.items():
        cost += count*(bitstring_energy(hamiltonian, basis_state) / shots)
    return cost


def cvar_expectation_value_classical(alpha: float, counts: Dict, hamiltonian: Hamiltonian):
    """
    CVaR computation of cost function. For the definition of the cost function, refer
    to https://arxiv.org/abs/1907.04769.

    Parameters
    ----------
    alpha: `float`
            The CVaR parameter.
    counts: `dict`
            The counts of the measurements.
    hamiltonian: `Hamiltonian`
            The Cost Hamiltonian defined for the optimization problem
    """
    assert alpha > 0 and alpha < 1, "Please specify a valid alpha value between 0 and 1"
    shots = sum(counts.values())

    # sort the dictionary by descending energy of measurement outcomes
    OrderedDict(sorted(counts.items(), key=lambda x: bitstring_energy(
        hamiltonian, x[0]), reverse=False))
    K = int((shots-1)*alpha+1)

    truncated_counts = {key: counts[key] for key in list(counts.keys())[:K]}
    cost = expectation_value_classical(truncated_counts, hamiltonian)
    return cost


def cost_function(counts: Dict, hamiltonian: Hamiltonian, alpha: float = 1):
    """
    The cost function to be used for QAOA training.

    Parameters
    ----------
    counts: `dict`
            The counts of the measurements.
    hamiltonian: `Hamiltonian`
            The Cost Hamiltonian defined for the optimization problem
    alpha: `float`
            The CVaR parameter.
    """
    if alpha == 1:
        return expectation_value_classical(counts, hamiltonian)
    else:
        return cvar_expectation_value_classical(alpha, counts, hamiltonian)
