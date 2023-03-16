import numpy as np

from typing import Union
from docplex.mp.model import Model
import matplotlib.pyplot as plt


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

    __name__ = "bin_packing"

    def __init__(
        self,
        weights: list = [],
        weight_capacity: int = 0,
        penalty: list = [],
        n_bins: int = None,
        simplifications=True,
        method: str = "slack",
    ):
        # include_ineqs: True if including the inequalities
        if method not in ["slack", "unbalanced"]:
            raise ValueError(
                f"The method '{method}' is not a valid method. Choose between 'slack' and 'unbalanced'"
            )
        self.weights = weights
        self.weight_capacity = weight_capacity
        self.penalty = penalty
        self.n_items = len(weights)
        self.method = method
        self.simplifications = simplifications
        self.eq_constraints = {}
        self.ineq_constraints = {}
        if n_bins is None:
            self.n_bins = self.n_items
        else:
            self.n_bins = n_bins
        self.solution = self.solution_dict()
        self.cplex_model = self.docplex_model()
        self.n_vars = self.cplex_model.number_of_binary_variables

    def solution_dict(self):
        solution = {f"y_{i}": None for i in range(self.n_bins)}
        for i in range(self.n_items):
            for j in range(self.n_bins):
                solution[f"x_{i}_{j}"] = None
        if self.simplifications:
            # First simplification: we know the minimum number of bins
            self.min_bins = int(np.ceil(np.sum(self.weights) / self.weight_capacity))
            for j in range(self.n_bins):
                if j < self.min_bins:
                    solution[f"y_{j}"] = 1
            solution["x_0_0"] = 1
            for j in range(1, self.n_bins):
                solution[f"x_0_{j}"] = 0
        return solution

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the Bin Packing problem.

        Parameters
        ----------
        n_items: int
            The number of items that must be placed in the bins.
        seed: int
            Seed number for choosing the random weights
        min_weight: int
            Minimum weight to choose the weights from
        max_weight: int
            Maximum weight to choose the random weights from
        simplification: bool
            Use the simplification version of the problem
        method: ["slack", "unbalanced"]
            Use one of the two methods available to enconde the inequality constraints.
            The slack method requires extra-qubits.

        Returns
        -------
            A random instance of the bin packing problem.
        """
        seed = kwargs.get("seed", 1234)
        np.random.seed(seed)
        weight_capacity = kwargs.get("weight_capacity", 15)
        n_items = kwargs.get("n_items", 3)
        n_bins = n_items
        min_weight = kwargs.get("min_weight", 1)
        max_weight = kwargs.get("max_weight", 7)
        if min_weight >= max_weight:
            raise ValueError(
                f"min_weight: {min_weight} must be < max_weight:{max_weight}"
            )
        weights = list(np.random.randint(min_weight, max_weight, n_items))
        simplification = kwargs.get("simplification", True)
        penalty = kwargs.get("penalty", [])
        method = kwargs.get("method", "slack")
        return BinPacking(
            weights, weight_capacity, penalty, n_bins, simplification, method
        )

    def docplex_model(self):
        mdl = Model("bin_packing")
        vars_ = {}
        for var in self.solution.keys():
            if self.solution[var] is None:
                vars_[var] = mdl.binary_var(var)
            else:
                vars_[var] = self.solution[var]
        objective = mdl.sum([vars_[y] for y in vars_.keys() if y[0] == "y"])
        self.vars_pos = {var.name: n for n, var in enumerate(mdl.iter_binary_vars())}

        mdl.minimize(objective)
        if self.simplifications:
            list_items = range(1, self.n_items)
        else:
            list_items = range(self.n_items)

        for i in list_items:
            # First set of constraints: the items must be in any bin
            self.eq_constraints[f"eq_{i}"] = [
                [self.vars_pos[f"x_{i}_{j}"] for j in range(self.n_bins)],
                [1],
            ]
            mdl.add_constraint(
                mdl.sum(vars_[f"x_{i}_{j}"] for j in range(self.n_bins)) == 1
            )

        for j in range(self.n_bins):
            # Second set of constraints: weight constraints
            mdl.add_constraint(
                mdl.sum(
                    (self.weights[i] / self.weight_capacity) * vars_[f"x_{i}_{j}"]
                    for i in range(self.n_items)
                )
                <= vars_[f"y_{j}"]
            )
            if self.simplifications and j < self.min_bins:
                if j == 0:
                    self.ineq_constraints[f"ineq_{j}"] = [
                        [self.vars_pos[f"x_{i}_{j}"] for i in list_items],
                        [self.weight_capacity - self.weights[0]],
                    ]
                else:
                    self.ineq_constraints[f"ineq_{j}"] = [
                        [self.vars_pos[f"x_{i}_{j}"] for i in list_items],
                        [self.weight_capacity],
                    ]

            else:
                self.ineq_constraints[f"ineq_{j}"] = [
                    [self.vars_pos[f"x_{i}_{j}"] for i in list_items],
                    [self.weight_capacity * self.vars_pos[f"y_{j}"]],
                ]

        return mdl

    @property
    def qubo(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        if len(self.penalty) > 0:
            if self.method == "slack":
                qubo_docplex = FromDocplex2IsingModel(
                    self.cplex_model, multipliers=self.penalty[0]
                )
            elif self.method == "unbalanced":
                qubo_docplex = FromDocplex2IsingModel(
                    self.cplex_model,
                    multipliers=self.penalty[0],
                    unbalanced_const=True,
                    strength_ineq=self.penalty[1:],
                )
        else:
            if self.method == "slack":
                qubo_docplex = FromDocplex2IsingModel(self.cplex_model)
            elif self.method == "unbalanced":
                qubo_docplex = FromDocplex2IsingModel(
                    self.cplex_model, unbalanced_const=True
                )

        return qubo_docplex.ising_model

    def classical_solution(self, string: bool = False):
        """
        Return the classical solution of the bin packing problem

        Parameters
        ----------
        string : bool, optional
            If the solution is returned as a string. The default is False.

        Raises
        ------
        ValueError
            A flag if docplex does not find a valid solution.

        Returns
        -------
        solution : Union[str, dict]
            The classical solution of the specific problem as a string or a dict.

        """
        docplex_sol = self.cplex_model.solve()

        if docplex_sol is None:
            raise ValueError(
                f"solution not found: {self.cplex_model.solve_details.status}"
            )

        if string:
            solution = ""
        else:
            solution = self.solution.copy()
        for var in self.cplex_model.iter_binary_vars():
            if string:
                solution += str(int(np.round(docplex_sol.get_value(var), 1)))
            else:
                solution[var.name] = int(np.round(docplex_sol.get_value(var), 1))
        return solution

    def plot_solution(self, solution: Union[dict, str], ax=None):
        """
        A visualization method for the bin packing problem solution.

        Parameters
        ----------
        solution : Union[dict, str]
            The solution of the specific bin packing problem as a string or dictionary.
        ax : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        fig : matplotlib.pyplot.Figure()
            The plot visualization of the solution.

        """
        if isinstance(solution, str):
            sol = self.solution.copy()
            for n, var in enumerate(self.cplex_model.iter_binary_vars()):
                sol[var.name] = int(solution[n])
            solution = sol
        colors = plt.cm.get_cmap("jet", len(self.weights))
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        for j in range(self.n_bins):
            sum_items = 0
            if solution[f"y_{j}"]:
                for i in range(self.n_items):
                    if solution[f"x_{i}_{j}"]:
                        ax.bar(
                            j,
                            self.weights[i],
                            bottom=sum_items,
                            label=f"item {i}",
                            color=colors(i),
                            alpha=0.7,
                            edgecolor="black",
                        )
                        sum_items += self.weights[i]
        ax.hlines(
            self.weight_capacity,
            -0.5,
            self.n_bins - 0.5,
            linestyle="--",
            color="black",
            label="Max Weight",
        )
        ax.set_xticks(np.arange(self.n_bins), np.arange(self.n_bins), fontsize=14)
        ax.set_xlabel("bin", fontsize=14)
        ax.set_ylabel("weight", fontsize=14)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2 + 0.011 * self.n_items),
            ncol=5,
            fancybox=True,
            shadow=True,
        )
        return fig
