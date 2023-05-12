import networkx as nx

from ..utilities import check_kwargs
from .problem import Problem
from .qubo import QUBO


class MaximumCut(Problem):
    """
    Creates an instance of the Maximum Cut problem.

    Parameters
    ----------
    G: nx.Graph
        The input graph as NetworkX graph instance.

    Returns
    -------
        An instance of the Maximum Cut problem.
    """

    __name__ = "maximum_cut"

    DEFAULT_EDGE_WEIGHT = 1.0

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
        Creates a random instance of the Maximum Cut problem, whose graph is
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
            A random instance of the Maximum Cut problem.
        """
        n_nodes, edge_probability = check_kwargs(
            ["n_nodes", "edge_probability"], [None, None], **kwargs
        )
        seed = kwargs.get("seed", None)

        G = nx.generators.random_graphs.fast_gnp_random_graph(
            n=n_nodes, p=edge_probability, seed=seed
        )
        return MaximumCut(G)

    @property
    def qubo(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        # Iterate over edges (with weight) and store accordingly
        terms = []
        weights = []

        for u, v, edge_weight in self.G.edges(data="weight"):
            terms.append([u, v])

            # We expect the edge weight to be given in the attribute called
            # "weight". If it is None, assume a weight of 1.0
            weights.append(
                edge_weight if edge_weight else MaximumCut.DEFAULT_EDGE_WEIGHT
            )

        return QUBO(self.G.number_of_nodes(), terms, weights, self.problem_instance)
