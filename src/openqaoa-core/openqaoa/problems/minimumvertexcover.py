from collections import Counter
import networkx as nx

from ..utilities import check_kwargs
from .problem import Problem
from .qubo import QUBO


class MinimumVertexCover(Problem):
    """
    Creates an instance of the Minimum Vertex Cover problem.

    Parameters
    ----------
    G: nx.Graph
        The input graph as NetworkX graph instance.
    field: float
        The strength of the artificial field minimizing the size of the cover.
    penalty: float
        The strength of the penalty enforcing the cover constraint.

    Returns
    -------
    An instance of the Minimum Vertex Cover problem.
    """

    __name__ = "minimum_vertex_cover"

    def __init__(self, G, field, penalty):

        self.G = G
        self.field = field
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

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, input_field):

        if not isinstance(input_field, int) and not isinstance(input_field, float):
            raise TypeError(
                "The input parameter, field, has to be of type float or int"
            )

        self._field = input_field

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, input_penalty):

        if not isinstance(input_penalty, int) and not isinstance(input_penalty, float):
            raise TypeError(
                "The input parameter, penalty, has to be of type float or int"
            )

        self._penalty = input_penalty

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the Minimum Vertex Cover problem, whose graph is
        random following the Erdos-Renyi model. By default the artificial field is
        set to 1.0 and the default penalty os taken to be 10 times larger.

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
        A random instance of the Minimum Vertex Cover problem.
        """

        n_nodes, edge_probability = check_kwargs(
            ["n_nodes", "edge_probability"], [None, None], **kwargs
        )
        seed = kwargs.get("seed", None)
        G = nx.generators.random_graphs.fast_gnp_random_graph(
            n=n_nodes, p=edge_probability, seed=seed
        )

        DEFAULT_FIELD = 1.0
        DEFAULT_PENALTY = 10

        return MinimumVertexCover(G, DEFAULT_FIELD, DEFAULT_PENALTY)

    def terms_and_weights(self):
        """
        Creates the terms and weights for the Minimum Vertex Cover problem

        Returns
        -------
        terms_weights: tuple(list[list],list[float])
            Tuple containing list of terms and list of weights.
        """

        # Number of nodes
        n_nodes = self.G.number_of_nodes()

        # Number of edges
        edges = list(self.G.edges())

        # Connectivity of each node
        node_repetition = [e for edge in edges for e in edge]
        connectivity = dict(Counter(node_repetition))

        # Quadratic interation from penalty term
        quadratic_interaction = [(list(e), self.penalty / 4) for e in edges]

        # Linear terms from the artificial field
        linear_interaction = [
            ([i], -self.field / 2 + connectivity[i] * self.penalty / 4)
            if connectivity.get(i) is not None
            else ([i], -self.field / 2)
            for i in range(n_nodes)
        ]

        # Constant term
        constant_term = [([], n_nodes * self.field / 2 + len(edges) * self.penalty / 4)]

        # Generate tuple containing a list with the terms and a list with the weights
        terms_weights = tuple(
            zip(*(quadratic_interaction + linear_interaction + constant_term))
        )

        # Unzip to retrieve terms and weights in separate sequences
        return terms_weights

    @property
    def qubo(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
        The QUBO encoding of this problem.
        """

        # Extract terms and weights from the problem definition
        terms, weights = self.terms_and_weights()

        return QUBO(
            self.G.number_of_nodes(), list(terms), list(weights), self.problem_instance
        )
