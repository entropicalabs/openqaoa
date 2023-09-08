import networkx as nx
from docplex.mp.model import Model
from typing import Union
import matplotlib.pyplot as plt
from matplotlib import colormaps

from .problem import Problem
from .converters import FromDocplex2IsingModel
from .qubo import QUBO


class KColor(Problem):
    """
    Creates an instance of the k-color problem. It determines if given a set of
    colors, a graph can be colored such that no nodes connected by an edge share
    the same color.
    https://en.wikipedia.org/wiki/Graph_coloring
    Parameters
    ----------
    G: networkx.Graph
        Networkx graph of the problem
    k: int
        Maximum number of colors.
    penalty: float
        Penalty for the edge constraints.

    Returns
    -------
        An instance of the k-color problem.
    """

    __name__ = "k_color"

    def __init__(self, G, k, penalty: Union[int, float] = 2):
        self.G = G
        self.k = k
        self.penalty = penalty

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
        Creates a random instance of the k-color problem, whose graph is
        random following the Erdos-Renyi model.

        Parameters
        ----------
        n_nodes: int
            The number of nodes (vertices) in the graph.
        edge_probability: float
            The probability with which an edge is added to the graph.
        k: int
            Maximum number of colors.
        seed: int, optional
            Random seed for reproducibility.
        penalty: Union(int, float) Optional
            Penalty for the edge constraints.


        Returns
        -------
            A random instance of the k-color problem.
        """
        n_nodes = kwargs.get("n_nodes", 5)
        edge_probability = kwargs.get("edge_probability", 0.3)
        k = kwargs.get("k", 2)
        seed = kwargs.get("seed", None)
        G = nx.generators.random_graphs.fast_gnp_random_graph(
            n=n_nodes, p=edge_probability, seed=seed
        )
        return KColor(G, k)

    @property
    def docplex_model(self):
        mdl = Model("k_color")

        x = {
            (vertex, color): mdl.binary_var(name=f"x_{vertex}_{color}")
            for vertex in self.G
            for color in range(self.k)
        }

        # Constraint: Each vertex must be assigned to exactly one color
        for vertex in self.G:
            mdl.add_constraint(
                mdl.sum(x[(vertex, color)] for color in range(self.k)) == 1
            )
        cost = 0
        # Constraint: Adjacent vertices cannot have the same color
        for vertex1 in self.G:
            for vertex2 in self.G:
                if vertex2 in self.G[vertex1]:
                    for color in range(self.k):
                        cost += self.penalty * x[(vertex1, color)] * x[(vertex2, color)]
        # Objective function: Minimize the cost, finding a valid solution.
        mdl.minimize(cost)
        return mdl

    def classical_solution(self, string: bool = False):
        """
        Return the classical solution of the k-color problem

        Parameters
        ----------
        string : bool, optional
            If the solution is returned as a string. The default is False.

        Returns
        -------
        solution : Union[str, dict]
            The classical solution of the specific problem as a string or a dict.
        """
        mdl = self.docplex_model
        solution = mdl.solve()

        if solution is None:
            raise ValueError("No valid coloring found.")

        if string:
            coloring = ""
        else:
            coloring = {}
        for var in mdl.iter_binary_vars():
            if string:
                coloring += str(int(round(solution.get_value(var), 1)))
            else:
                coloring[var.name] = int(round(solution.get_value(var), 1))
        return coloring

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

    def plot_solution(self, solution: Union[dict, str], ax=None, colors=None):
        """
        A visualization method for the k-color solution.

        Parameters
        ----------
        solution : Union[dict, str]
            The solution of the specific k-color problem as a string or dictionary.
        ax : matplotlib axes, optional
            The default is None.

        Returns
        -------
        fig : matplotlib.pyplot.Figure()
            The graph visualization of the solution.
        """
        if isinstance(solution, str):
            sol = {}
            for n, var in enumerate(self.docplex_model.iter_binary_vars()):
                sol[var.name] = int(solution[n])
            solution = sol

        if ax is None:
            _, ax = plt.subplots()

        colors = colormaps["jet"] if colors is None else colors
        pos = nx.circular_layout(self.G)
        nx.draw(
            self.G,
            pos=pos,
            node_color=[
                colors((col + 1) / (self.k + 1))
                for vertex in self.G
                for col in range(self.k)
                if solution[f"x_{vertex}_{col}"] == 1
            ],
            edge_color="black",
            with_labels=True,
            ax=ax,
            alpha=0.85,
            **{"edgecolors": "black"},
        )
