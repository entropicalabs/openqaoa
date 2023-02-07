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

import numpy as np

from typing import Union
from docplex.mp.model import Model



from .problem import Problem
from .converters import FromDocplex2IsingModel


class BinPacking(Problem):
    """
    Creates an instance of the Bin Packing problem.
    https://en.wikipedia.org/wiki/Bin_packing_problem
    
    Parameters
    ----------
    weights: List[int]
        The weight of the items that must be placed in the bins.
    weight_capacity: int
        The maximum weight the bin can hold.
    penalty: float
        Penalty for the weight constraint.
        
    Returns
    -------
        An instance of the Bin PAcking problem.
    """

    __name__ = "binpacking"
    
    def __init__(self, weights:list=[], weight_capacity:int=0, penalty:Union[float, list]=None,
                 n_bins:int = None, simplifications=True, method="slack"):

        self.weights = weights
        self.weight_capacity = weight_capacity
        self.penalty = penalty
        self.n_items = len(weights)
        self.method = method
        self.simplifications = simplifications
        if n_bins is None:
            self.n_bins = self.n_items
        else:
            self.n_bins = n_bins

    def random_instance(self, n_items:int=3, min_weight:int=1, max_weight:int=5,
                        weight_capacity:int=10, seed:int=1):
        np.random.seed(seed)
        self.weight_capacity = weight_capacity
        self.n_items = n_items
        self.n_bins = n_items
        self.weights = list(np.random.randint(min_weight, max_weight, n_items))

        
    def docplex_model(self):
        mdl = Model("Bin Packing")

        y = mdl.binary_var_list(
            self.n_bins, name="y"
        )  # list of variables that represent the bins
        x = mdl.binary_var_matrix(
            self.n_items, self.n_bins, "x"
        )  # variables that represent the items on the specific bin
        
        if self.simplifications:
            # First simplification: we know the minimum number of bins
            min_bins = int(np.ceil(np.sum(self.weights) / self.weight_capacity))
            for bin_i in range(min_bins):
                y[bin_i] = 1
            # Assign the first item into the first bin
            x[0, 0] = 1
            for j in range(self.num_bins):
                x[0, j] = 0

        objective = mdl.sum(y)

        mdl.minimize(objective)

        for i in range(self.n_items):
            # First set of constraints: the items must be in any bin
            mdl.add_constraint(mdl.sum(x[i, j] for j in range(self.n_bins)) == 1)

        for j in range(self.n_bins):
            # Second set of constraints: weight constraints
            mdl.add_constraint(
                mdl.sum(self.weights[i] * x[i, j] for i in range(self.n_items)) <= self.weight_capacity * y[j]
            )
        return mdl

    def get_qubo_problem(self):
        """
        Returns the QUBO encoding of this problem.
        
        Returns
        -------
            The QUBO encoding of this problem.
        """
        self.cplex_model = self.docplex_model()
        if self.method == "slack":
            qubo = FromDocplex2IsingModel(self.cplex_model, multipliers=self.penalty).ising_model
        elif self.method == "unbalanced":
            qubo = FromDocplex2IsingModel(self.cplex_model, multipliers=self.penalty[0],
                                          unbalanced_const=True, strength_ineq=self.penalty[1:]).ising_model
        return qubo
