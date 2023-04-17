import numpy as np
from docplex.mp.model import Model
import networkx as nx
import matplotlib.pyplot as plt

from .problem import Problem
from .converters import FromDocplex2IsingModel
from .qubo import QUBO


class PortfolioOptimization(Problem):
    """
    Creates an instance of the portfolio optimization problem.
    https://en.wikipedia.org/wiki/Portfolio_optimization

    Parameters
    ----------
    mu: List[int]
        The expected return
    sigma: List[List[int]]
        The covariance matrix.
    risk_factor: float
        desired risk factor exposures
    budget: int
        maximum number of assets to buy
    penalty: float
        Penalty for the maximum budget.

    Returns
    -------
        An instance of the portfolio optimization problem.
    """

    __name__ = "portfolio_optimization"

    def __init__(self, mu, sigma, risk_factor, budget, penalty):
        # Check whether the input is valid. Number of values should match the number of weights.
        for s_i in sigma:
            if len(mu) != len(s_i):
                raise ValueError("Number of assets does not match sigma")
        self.mu = mu
        self.sigma = sigma
        self.risk_factor = risk_factor
        self.penalty = penalty
        self.num_assets = len(mu)
        self.budget = budget

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the portfolio optimization problem.

        Parameters
        ----------
        num_assets: int
            The number of assets.
        mu_bounds: tuple
            bounds of the expected return
        sigma_bounds: tuple
            bound of the covariance matrix
        risk_factor: float
            desired risk factor exposures
        seed: int
            Seed for the random number selection.

        Returns
        -------
            A random instance of the portfolio optimization problem.
        """
        num_assets = kwargs.get("num_assets", 8)
        seed = kwargs.get("seed", 1234)
        mu_bounds = kwargs.get("mu_bounds", (-1, 1))
        sigma_bounds = kwargs.get("sigma_bounds", (-0.1, 0.1))
        budget = kwargs.get("budget", 5)
        risk_factor = kwargs.get("risk_factor", 0.1)
        penalty = kwargs.get("penalty", 2)

        np.random.seed(seed)

        mu = [
            (mu_bounds[1] - mu_bounds[0]) * np.random.rand() + mu_bounds[0]
            for _ in range(num_assets)
        ]
        sigma = [[0 for i in range(num_assets)] for j in range(num_assets)]
        for i in range(num_assets):
            for j in range(num_assets):
                sigma[i][j] = (
                    sigma_bounds[1] - sigma_bounds[0]
                ) * np.random.rand() + sigma_bounds[0]

        return PortfolioOptimization(mu, sigma, risk_factor, budget, penalty)

    @property
    def docplex_model(self):
        num_assets = self.num_assets

        # Start the docplex model with Model("name of the model")
        mdl = Model("Portfolio_Optimization")

        # Consider the number of variables as num_assets,
        # and binary set of variables that represent the stocks
        # x vector in numpy array for matrix multiplication
        x = np.array(mdl.binary_var_list(num_assets, name="asset"))

        # Specific the objective of the
        # portfolio optimization function
        objective_function = -np.array(self.mu) @ x + x.T @ np.array(self.sigma) @ x

        # For this problem it aims to maximize the profit
        # of those assets minimizing the risk of the investment
        mdl.minimize(objective_function)

        # Budget constraint
        mdl.add_constraint(mdl.sum(x) == self.budget, ctname="budget")

        return mdl

    @property
    def qubo(self):
        """
        Returns the QUBO encoding of this problem.
        Returns
        -------
            The QUBO encoding of this problem.
        """
        cplex_model = self.docplex_model
        n_vars = cplex_model.number_of_binary_variables
        qubo_docplex = FromDocplex2IsingModel(cplex_model, multipliers=self.penalty)
        ising_model = qubo_docplex.ising_model
        return QUBO(
            n_vars,
            ising_model.terms + [[]],
            ising_model.weights + [ising_model.constant],
            self.problem_instance,
        )

    def classical_solution(self, string: bool = False):
        """
        Return the classical solution of the portfolio optimization problem
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
        cplex_model = self.docplex_model
        docplex_sol = cplex_model.solve()

        if docplex_sol is None:
            raise ValueError(f"solution not found: {cplex_model.solve_details.status}")

        if string:
            solution = ""
        else:
            solution = {}
        for var in cplex_model.iter_binary_vars():
            if string:
                # round is used because sometimes docplex gives the solution close to 1 but not 1,
                #if we want to store this as an integer and use int(value) and the value is 0.999999
                #it will store a zero.
                solution += str(round(docplex_sol.get_value(var), 1))
            else:
                solution[var.name] = round(docplex_sol.get_value(var), 1)
        return solution

    def plot_solution(self, solution, ax=None):
        G = nx.Graph()
        G.add_nodes_from(range(self.num_assets))
        edge_list = []
        width = []
        for i in range(self.num_assets - 1):
            for j in range(i + 1, self.num_assets):
                G.add_edge(i, j, weight=self.sigma[i][j])
                edge_list.append((i, j))
                width.append(self.sigma[i][j])
        pos = nx.circular_layout(G)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        graph_colors_sol = [
            "#F29D55" if int(i) else "#28C9C9" for i in solution.values()
        ]

        nx.draw(
            G,
            pos=pos,
            labels={i: k for i, k in enumerate(solution.keys())},
            node_size=900,
            node_color=graph_colors_sol,
            ax=ax,
            edgecolors="grey",
        )
        nx.draw_networkx_edges(G, pos, edgelist=edge_list, width=width, ax=ax)
        ax.plot(
            [],
            [],
            marker="o",
            linewidth=0,
            markeredgecolor="grey",
            color="#F29D55",
            label="selected",
            markersize=10,
        )
        ax.plot(
            [],
            [],
            marker="o",
            linewidth=0,
            markeredgecolor="grey",
            color="#28C9C9",
            label="non-selected",
            markersize=10,
        )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)
        return fig
