import networkx as nx
from docplex.mp.model import Model
from typing import Union
import matplotlib.pyplot as plt
import numpy as np

from ..utilities import check_kwargs
from .problem import Problem
from .converters import FromDocplex2IsingModel
from .qubo import QUBO


class MIS(Problem):
    """
    Creates an instance of the maximal independent set (MIS) problem.
    https://en.wikipedia.org/wiki/Maximal_independent_set

    Parameters
    ----------
    G: networkx.Graph
        Networkx graph of the problem 
    penalty: float
        Penalty for the edge constraints.

    Returns
    -------
        An instance of the MIS problem.
    """

    __name__ = "maximal_independent_set"

    def __init__(self, G, penalty:Union[int, float]=2):
        self.G = G
        self.penalty = penalty
        # self.cplex_model = self.docplex_model()

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
        Creates a random instance of the maximal independent set problem, whose graph is
        random following the Erdos-Renyi model.

        Parameters
        ----------
        **kwargs:
            Required keyword arguments are:

            n_nodes: int
                The number of nodes (vertices) in the graph.
            edge_probability: float
                The probability with which an edge is added to the graph.

        Returns
        -------
            A random instance of the maximal independent set problem.
        """
        n_nodes, edge_probability = check_kwargs(
            ["n_nodes", "edge_probability"], [None, None], **kwargs
        )
        seed = kwargs.get("seed", None)

        G = nx.generators.random_graphs.fast_gnp_random_graph(
            n=n_nodes, p=edge_probability, seed=seed
        )
        return MIS(G)

    @property
    def docplex_model(self):
        mdl = Model('MIS')
        num_vertices = self.G.number_of_nodes()

        x = {i: mdl.binary_var(name=f"x_{i}") for i in range(num_vertices)}

        mdl.minimize(-mdl.sum(x) + self.penalty * mdl.sum(
            x[i] * x[j] for (i, j) in self.G.edges
        ))
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
        
        return QUBO(self.G.number_of_nodes(), qubo_docplex.terms + [[]], qubo_docplex.weights + [qubo_docplex.constant], self.problem_instance)

    def classical_solution(self, string: bool = False):
        """
        Return the classical solution of the maximal independent set problem
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
            raise ValueError(
                f"solution not found: {self.cplex_model.solve_details.status}"
            )

        if string:
            solution = ""
        else:
            solution = {}
        for var in cplex_model.iter_binary_vars():
            if string:
                solution += str(int(round(docplex_sol.get_value(var), 1)))
            else:
                solution[var.name] = int(round(docplex_sol.get_value(var), 1))
        return solution

    def plot_solution(self, solution: Union[dict, str], ax=None):
        """
        A visualization method for the maximal independet set solution.
        Parameters
        ----------
        solution : Union[dict, str]
            The solution of the specific bin packing problem as a string or dictionary.
        ax : matplotlib axes, optional
            The default is None.
        Returns
        -------
        fig : matplotlib.pyplot.Figure()
            The graph visualization of the solution.
        """
        if isinstance(solution, str):
            sol = self.solution.copy()
            for n, var in enumerate(self.docplex_model().iter_binary_vars()):
                sol[var.name] = int(solution[n])
            solution = sol
        colors = ["#5EB1EB", "#F29D55"]
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        pos = nx.circular_layout(self.G)
        num_vertices = self.G.number_of_nodes()
        nx.draw(self.G, pos=pos, edgecolors="black", labels={i:str(i) for i in range(num_vertices)}, node_color=[colors[solution[f"x_{i}"]] for i in range(num_vertices)], ax=ax, edge_color="#0B2340")
        ax.plot([],[], marker="o", markersize=10, label="MIS", linewidth=0,color=colors[1], markeredgecolor="black")
        ax.legend(loc="upper center", bbox_to_anchor=[0.5, 1.1])
        return fig
