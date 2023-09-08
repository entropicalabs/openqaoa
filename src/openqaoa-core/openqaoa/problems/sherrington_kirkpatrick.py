import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union
from docplex.mp.model import Model

from .problem import Problem
from .converters import FromDocplex2IsingModel
from .qubo import QUBO


class SK(Problem):
    """
    Creates an instance of the Sherrington Kirkpatrick.
    https://en.wikipedia.org/wiki/Spin_glass

    Parameters
    ----------
    G: networkx.Graph
        Networkx graph of the problem


    Returns
    -------
        An instance of the Sherrington Kirkpatrick Problem.
    """

    __name__ = "sherrington_kirkpatrick"

    def __init__(self, G):
        self.G = G

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, input_networkx_graph):
        if not isinstance(input_networkx_graph, nx.Graph):
            raise TypeError("Input problem graph must be a networkx Graph.")

        # Relabel nodes to integers starting from 0
        mapping = dict(
            zip(input_networkx_graph, range(input_networkx_graph.number_of_nodes()))
        )
        self._G = nx.relabel_nodes(input_networkx_graph, mapping)

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the SK problem, whose graph is random following the Erdos-Renyi model.

        Parameters
        ----------
        **kwargs:

            n_nodes: int
                The number of nodes (spins) in the graph.
            mu: float
                Mean value of the distribution for the J_{ij} values.
            sigma: float
                Standard deviation of the distribution for the J_{ij} values.
            seed: int
                Seed to be used for the random case.

        Returns
        -------
            A random instance of the SK problem.
        """
        n_nodes = kwargs.get("n_nodes")
        mu = kwargs.get("mu", 0)
        sigma = kwargs.get("sigma", 1)
        seed = kwargs.get("seed", None)
        rng = np.random.default_rng(seed)

        G = nx.Graph()
        G.add_weighted_edges_from(
            [
                [i, j, round(rng.normal(mu, sigma), 3)]
                for i in range(n_nodes)
                for j in range(i + 1, n_nodes)
            ]
        )

        return SK(G)

    @property
    def docplex_model(self):
        mdl = Model("SK_Problem")
        num_spins = self.G.number_of_nodes()

        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(num_spins)}

        energy = -1 * mdl.sum(
            (2 * x[i] - 1) * (2 * x[j] - 1) * self.G[i][j]["weight"]
            for i, j in self.G.edges()
        )

        mdl.minimize(energy)

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
        qubo_docplex = FromDocplex2IsingModel(cplex_model).ising_model

        return QUBO(
            self.G.number_of_nodes(),
            qubo_docplex.terms + [[]],
            qubo_docplex.weights + [qubo_docplex.constant],
            self.problem_instance,
        )

    def classical_solution(self, string: bool = False):
        """
        Solves the SK problem and returns the CPLEX solution.

        Returns
        -------
        solution : dict
            A dictionary containing the solution found with spin values.
        """
        model = self.docplex_model
        model.solve()
        status = model.solve_details.status
        if status != "integer optimal solution":
            print(status)
        if string:
            return "".join(
                str(round(model.solution.get_value(var)))
                for var in model.iter_binary_vars()
            )
        else:
            return {
                var.name: model.solution.get_value(var)
                for var in model.iter_binary_vars()
            }

    def plot_solution(self, solution: Union[str, dict], ax=None):
        """
        Plots the solution of the SK problem.

        Parameters
        ----------
        solution : dict
            A dictionary containing the solution with spin values.

        Returns
        -------
        fig : matplotlib.pyplot.Figure()
            The graph visualization of the solution.
        """
        colors = ["#5EB1EB", "#F29D55"]
        if isinstance(solution, str):
            sol = {}
            for n, var in enumerate(self.docplex_model.iter_binary_vars()):
                sol[var.name] = int(solution[n])
            solution = sol
        if ax is None:
            _, ax = plt.subplots()

        pos = nx.circular_layout(self.G)
        nx.draw(
            self.G,
            pos=pos,
            edgecolors="black",
            node_color=[colors[round(solution[spin])] for spin in solution],
            labels={n: n for n in range(len(solution))},
            ax=ax,
            edge_color="#0B2340",
        )
        edge_labels = nx.get_edge_attributes(self.G, "weight")

        nx.draw_networkx_edge_labels(self.G, pos=pos, edge_labels=edge_labels, ax=ax)
        ax.set_title("SK Problem Solution")
        ax.plot([], [], color=colors[0], label=r"$S_{-1}$", marker="o", linewidth=0)
        ax.plot([], [], color=colors[1], label=r"$S_{+1}$", marker="o", linewidth=0)
        ax.legend()
