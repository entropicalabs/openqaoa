import networkx as nx

from ..utilities import check_kwargs
from .problem import Problem
from .qubo import QUBO


class ShortestPath(Problem):
    """
    Creates an instance of the Shortest Path problem.

    Parameters
    ----------
    G: nx.Graph
        The input graph as NetworkX graph instance.
    source: int
        The index of the source node.
    dest: int
        The index of the destination node.

    Returns
    -------
        An instance of the Shortest Path problem.
    """

    __name__ = "shortest_path"

    def __init__(self, G, source, dest):

        # Relabel nodes to integers starting from 0
        mapping = dict(zip(G, range(G.number_of_nodes())))
        self.G = nx.relabel_nodes(G, mapping)

        self.source = source
        self.dest = dest

        assert source in list(G.nodes), f"Source node not within nodes of input graph"
        assert dest in list(
            G.nodes
        ), f"Destination node not within nodes of input graph"
        assert source != dest, "Source and destination nodes cannot be the same"

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the Shortest problem, whose graph is
        random following the Erdos-Renyi model. By default the node and edge
        weights are set to 1.0 and the default constraint is taken to be as large.

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
        A random instance of the Shortest Path problem.
        """

        n_nodes, edge_probability, seed, source, dest = check_kwargs(
            ["n_nodes", "edge_probability", "seed", "source", "dest"],
            [None, None, 1234, 0, 1],
            **kwargs,
        )
        G = nx.generators.random_graphs.fast_gnp_random_graph(
            n=n_nodes, p=edge_probability, seed=seed
        )

        DEFAULT_WEIGHTS = 1.0

        for (u, v) in G.edges():
            G.edges[u, v]["weight"] = DEFAULT_WEIGHTS
        for w in G.nodes():
            G.nodes[w]["weight"] = DEFAULT_WEIGHTS

        return ShortestPath(G, source, dest)

    def terms_and_weights(self):
        """
        Creates the terms and weights for the Shortest Path problem

        Returns
        -------
        terms_weights: tuple(list[list],list[float])
            Tuple containing list of terms and list of weights
        """
        s = self.source
        d = self.dest
        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()

        # # Linear terms due to node weights
        #     # For loop version
        #     node_terms_weights = []
        #     for i in range(n_nodes):
        #         if i not in [s, d]:
        #             shift = int(i>s)+int(i>d)
        #             node_terms_weights.append(([i-shift], self.G.nodes[i]['weight']))
        node_terms_weights = [
            ([i - (int(i > s) + int(i > d))], self.G.nodes[i]["weight"])
            for i in range(n_nodes)
            if i not in [s, d]
        ]

        # Linear terms due to edge weights (shift of n_nodes-2 since we removed 2 nodes)
        #     # For loop version
        #     edge_terms_weights = []
        #     for i, (u,v) in enumerate(self.G.edges()):
        #         edge_terms_weights.append(([i+n_nodes-2], self.G.edges[u,v]['weights']))
        edge_terms_weights = [
            ([i + n_nodes - 2], self.G.edges[u, v]["weight"])
            for i, (u, v) in enumerate(self.G.edges())
        ]

        # Source flow constraint
        #     # For loop version
        #     start_flow_terms_weights = []
        #     for i, x in enumerate(self.G.edges()):
        #         for j, y in enumerate(self.G.edges()):
        #             if s in x and s in y:
        #                 if i == j:
        #                     start_flow_terms_weights.append(([i+n_nodes-2], -1))
        #                 else:
        #                     start_flow_terms_weights.append(([i+n_nodes-2,j+n_nodes-2], 1))
        start_flow_terms_weights = [
            ([i + n_nodes - 2], -1)
            if i == j
            else ([i + n_nodes - 2, j + n_nodes - 2], 1)
            for i, x in enumerate(self.G.edges())
            for j, y in enumerate(self.G.edges())
            if (s in x and s in y)
        ]

        # Destination flow constraint
        #     # For loop version
        #     dest_flow_terms_weights = []
        #     for i, x in enumerate(self.G.edges()):
        #         for j, y in enumerate(self.G.edges()):
        #             if d in x and d in y:
        #                 if i == j:
        #                     dest_flow_terms_weights.append(([i+n_nodes-2], -1))
        #                 else:
        #                     dest_flow_terms_weights.append(([i+n_nodes-2,j+n_nodes-2], 1))
        dest_flow_terms_weights = [
            ([i + n_nodes - 2], -1)
            if i == j
            else ([i + n_nodes - 2, j + n_nodes - 2], 1)
            for i, x in enumerate(self.G.edges())
            for j, y in enumerate(self.G.edges())
            if (d in x and d in y)
        ]

        # Path flow constraint
        path_flow_terms_weights = []
        for i in range(n_nodes):
            if i != d and i != s:
                shift = int(i > s) + int(i > d)
                path_flow_terms_weights.append(([i - shift], 4))
                for j, x in enumerate(self.G.edges()):
                    if i in x:
                        path_flow_terms_weights.append(
                            ([i - shift, j + n_nodes - 2], -4)
                        )
                    for k, y in enumerate(self.G.edges()):
                        if i in x and i in y:
                            if j == k:
                                path_flow_terms_weights.append(([j + n_nodes - 2], 1))
                            else:
                                path_flow_terms_weights.append(
                                    ([j + n_nodes - 2, k + n_nodes - 2], 1)
                                )

        return tuple(
            zip(
                *(
                    node_terms_weights
                    + edge_terms_weights
                    + start_flow_terms_weights
                    + dest_flow_terms_weights
                    + path_flow_terms_weights
                )
            )
        )

    @property
    def qubo(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
        The QUBO encoding of this problem.
        """
        n = self.G.number_of_nodes() + self.G.number_of_edges() - 2
        # Extract terms and weights from the problem definition
        terms, weights = self.terms_and_weights()

        # Convert to Ising equivalent since variables are in {0, 1} rather than {-1, 1}
        ising_terms, ising_weights = QUBO.convert_qubo_to_ising(n, terms, weights)
        return QUBO(n, ising_terms, ising_weights, self.problem_instance)
