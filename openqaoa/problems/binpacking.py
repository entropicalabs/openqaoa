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
import matplotlib.pyplot as plt
from scipy.optimize import minimize


from .problem import Problem
from .converters import FromDocplex2IsingModel
from .qubo import QUBO
from openqaoa.workflows.optimizer import QAOA




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
                 n_bins:int = None, simplifications=True, method="slack", include_ineqs:bool=True):
        #include_ineqs: True if including the inequalities

        self.weights = weights
        self.weight_capacity = weight_capacity
        self.penalty = penalty
        self.n_items = len(weights)
        self.method = method
        self.simplifications = simplifications
        self.include_ineqs = include_ineqs
        self.eq_constraints = {}
        if n_bins is None:
            self.n_bins = self.n_items
        else:
            self.n_bins = n_bins
        if len(weights) > 0:
            self.solution = self.solution_dict()
            self.cplex_model = self.docplex_model()

            
    def solution_dict(self):
        solution = {f"y_{i}":None for i in range(self.n_bins)}
        for i in range(self.n_items):
            for j in range(self.n_bins):
                solution[f"x_{i}_{j}"] = None
        if self.simplifications:
            # First simplification: we know the minimum number of bins
            min_bins = int(np.ceil(np.sum(self.weights) / self.weight_capacity))
            for j in range(self.n_bins):
                if j < min_bins:
                    solution[f"y_{j}"] = 1
            solution["x_0_0"] = 1
            for j in range(1, self.n_bins):
                solution[f"x_0_{j}"] = 0
        return solution

    def random_instance(self, n_items:int=3, min_weight:int=1, max_weight:int=5,
                        weight_capacity:int=10, simplification=True, seed:int=1):
        np.random.seed(seed)
        self.weight_capacity = weight_capacity
        self.n_items = n_items
        self.n_bins = n_items
        self.weights = list(np.random.randint(min_weight, max_weight, n_items))
        self.simplifications = simplification
        self.solution = self.solution_dict()
        self.cplex_model = self.docplex_model()
        self.n_vars = self.cplex_model.number_of_binary_variables

        
    def docplex_model(self):
        mdl = Model("bin_packing")
        vars_ = {}
        for var in self.solution.keys():
            if self.solution[var] is None:
                vars_[var] = mdl.binary_var(var)
            else:
                vars_[var] = self.solution[var]
        objective = mdl.sum([vars_[y] for y in vars_.keys() if y[0] == "y"])
        self.vars_pos = {var.name:n for n, var in enumerate(mdl.iter_binary_vars())}

        mdl.minimize(objective)
        if self.simplifications:
            list_items = range(1, self.n_items)
        else:
            list_items = range(self.n_items)
        for i in list_items:
            # First set of constraints: the items must be in any bin
            self.eq_constraints[f"eq_{i}"] = [[self.vars_pos[f"x_{i}_{j}"] for j in range(self.n_bins)], [1]]
            mdl.add_constraint(mdl.sum(vars_[f"x_{i}_{j}"] for j in range(self.n_bins)) == 1)
        if self.include_ineqs:
            for j in range(self.n_bins):
                # Second set of constraints: weight constraints
                mdl.add_constraint(
                    mdl.sum(self.weights[i] * vars_[f"x_{i}_{j}"] for i in range(self.n_items)) <= self.weight_capacity * vars_[f"y_{j}"]
                )
        return mdl

    def get_qubo_problem(self):
        """
        Returns the QUBO encoding of this problem.
        
        Returns
        -------
            The QUBO encoding of this problem.
        """
        if self.method == "slack":
            qubo = FromDocplex2IsingModel(self.cplex_model, multipliers=self.penalty).ising_model
        elif self.method == "unbalanced":
            qubo = FromDocplex2IsingModel(self.cplex_model, multipliers=self.penalty[0],
                                          unbalanced_const=True, strength_ineq=self.penalty[1:]).ising_model
        return qubo
    
    def classical_solution(self, string=False):
        docplex_sol = self.cplex_model.solve()
        if string:
            solution = ""
        else:
            solution = self.solution.copy()
        for var in self.cplex_model.iter_binary_vars():
            if string:
                solution += str(int(np.round(docplex_sol.get_value(var), 1)))
            else:
                solution[var.name] = int(np.round(docplex_sol.get_value(var), 1))
        if not docplex_sol.is_valid_solution():
            raise ValueError("The Cplex solver does not find a solution.")
        return solution
    
    def plot_solution(self, solution:Union[dict, str], ax=None):
        if isinstance(solution, str):
            sol = self.solution.copy()
            for n, var in enumerate(self.cplex_model.iter_binary_vars()):
                sol[var.name] = int(solution[n])
            solution = sol
        colors = plt.cm.get_cmap("jet", len(self.weights))
        if ax is None:
            fig, ax = plt.subplots()
        for j in range(self.n_bins):
            sum_items = 0
            if solution[f"y_{j}"]:
                for i in range(self.n_items):
                    if solution[f"x_{i}_{j}"]:
                        ax.bar(j, self.weights[i], bottom=sum_items, label=f"item {i}", color=colors(i), alpha=0.7, edgecolor="black")
                        sum_items += self.weights[i]
        ax.hlines(self.weight_capacity, -0.5, self.n_bins - 0.5, linestyle="--", color="black", label="Max Weight")
        ax.set_xticks(np.arange(self.n_bins), fontsize=14)
        ax.set_xlabel("bin", fontsize=14)
        ax.set_ylabel("weight", fontsize=14)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2 + 0.011*self.n_items), ncol=5, fancybox=True, shadow=True)
        return fig 
    
    def find_multipliers(self, n_items=3):
        # n_items: the number of items to optimize from
        min_weight = min(self.weights)
        max_weight = max(self.weights)
        
        bin_packing = BinPacking(include_ineqs=False)
        bin_packing.random_instance(n_items, min_weight, max_weight, self.weight_capacity)
        cplex_model = bin_packing.cplex_model
        sol_str = bin_packing.classical_solution(string=True)
        def get_equality(lambda0, cplex_model, sol_str, callback=False):
            ising = FromDocplex2IsingModel(
                        cplex_model,
                        multipliers=lambda0[0],
                        unbalanced_const=False,
                        ).ising_model
            qaoa = QAOA()
            qaoa.set_circuit_properties(p=1, init_type="custom" , variational_params_dict={'betas': [-np.pi/8], 'gammas': [-np.pi/4]} )
            qaoa.set_classical_optimizer(maxiter=100)
            qaoa.compile(ising)
            qaoa.optimize()
            results = qaoa.results.lowest_cost_bitstrings(2**bin_packing.n_vars)
            pos = results["solutions_bitstrings"].index(sol_str)
            probability = results["probabilities"][pos]
            const_not = bin_packing.constraints_not_fulfilled(results["solutions_bitstrings"][0])
            if callback:
                return {"result":results, "pos":pos, "CoP":probability * 2**bin_packing.n_vars,
                        "probability":probability, "n_vars":bin_packing.n_vars, "x0":lambda0}
            
            print(f"lambda0: {lambda0[0]} | not fulfilled {const_not}| pos:{pos} | CoP:{probability*2**bin_packing.n_vars}")
            return const_not - probability
        
        sol = minimize(get_equality, x0=[0.1], args=(cplex_model, sol_str))
        # return sol
        x0 = sol.x
        return get_equality(x0, cplex_model, sol_str, callback=True)
            
    
    def normalizing(self, problem, normalized=-1):
        abs_weights = np.unique(np.abs(problem.weights))
        arg_sort = np.argsort(abs_weights)
        max_weight = abs_weights[arg_sort[normalized]]
        weights = [weight / max_weight for weight in problem.weights]
        terms = problem.terms
        terms.append([])
        weights.append(problem.constant/ max_weight)
        return QUBO(self.n_vars, terms, weights)
    
    def constraints_not_fulfilled(self, solution):
        check = 0
        for constraint in self.eq_constraints.values():
            vec = np.zeros(self.n_vars)
            vec[constraint[0]] = 1
            lh = np.sum(vec * np.array([int(i) for i in solution]))
            if lh != constraint[1]:
                check = False
        return check
