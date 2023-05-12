import networkx as nx
import numpy as np
import scipy

from ..utilities import check_kwargs
from .problem import Problem
from .qubo import QUBO


class TSP(Problem):
    """
    The Traveling Salesman Problem (TSP) requires to find, given a list of cities and
    the distances between each pair of cities (or the cities coordinates),
    the shortest possible path that visits each city exactly once and returns to the origin city.
    Additionally, one can also specify how cities are connected together.
    Our implementation accepts three different kind of inputs:

    #. A list of the cities' coordinates and, optionally, a (directed) graph specifiying the connectivity between cities
    #. A distance matrix encoding distances between each pair of cities and, optionally, a (directed) graph specifiying the connectivity between cities
    #. A weighted (directed) graph specifiying the connectivity and the distance between cities

    Initializes a TSP object via three different methods:

    #. Give a list of coordinates for the cities and optionally the connectivity between them via a (directed) graph.
    #. Give a distance matrix and optionally the connectivity between cities via a (directed) graph.
    #. Directly give a (directed) weighted graph, where edge weights are interpreted as distances between cities

    Whenever no graph connectivity is specified, it is assumed that all cities are connected.

    Parameters
    ----------
    city_coordinates : Optional[List[Tuple[float, float]]]
        List containing the coordinates of each city.
    distance_matrix : Optional[List[List[float]]]
        Distance between cities given as list of list representing a matrix
    G: Optional[nx.Graph]
        Graph encoding the connectivity between cities (can be directed)
    A: Optional[float]
        Quadratic penalty coefficient to enforce that a path is a Hamiltonian cycle.
    B: Optional[float]
        Penalty coefficient which accounts for the path cost.

    Returns
    -------
    None
    """

    __name__ = "tsp"

    def __init__(
        self,
        city_coordinates=None,
        distance_matrix=None,
        G=None,
        A=None,
        B=1,
    ):
        # Initialization when a weighted graph is given
        if G is not None and nx.is_weighted(G):
            TSP.validate_graph(G)
            n_cities = len(G)
        else:
            # Initialization when cities coordinates are given
            if city_coordinates is not None:
                TSP.validate_coordinates(city_coordinates)
                n_cities = len(city_coordinates)
                distance_matrix = scipy.spatial.distance_matrix(
                    city_coordinates, city_coordinates
                )
            # Initialization when a distance matrix is given
            elif distance_matrix is not None:
                TSP.validate_distance_matrix(distance_matrix)
                n_cities = len(distance_matrix)
                distance_matrix = np.array(distance_matrix)
            # Raise error if no input is given
            else:
                raise ValueError(
                    "Input missing: city coordinates, distance matrix or (weighted graph) required"
                )

            # Take into account graph connectivity if unweighted graph is given
            G = G if G else nx.complete_graph(n_cities)
            if n_cities != len(G):
                raise ValueError(
                    "Number of cities does not match the number of nodes in graph"
                )

            # Set edge weights to be the distances between corresponding cities
            for u, v in G.edges():
                G[u][v]["weight"] = distance_matrix[u, v]

        # Set number of cities
        self.n_cities = n_cities

        # Set the graph, making sure it is directed (useful when looping over edges during QUBO creation)
        self._G = nx.DiGraph(G)

        # Set penalty coefficients if given, otherwise give default value
        self.A = A if A else 2 * distance_matrix.max()
        self.B = B

    @property
    def graph(self):
        return self._G

    @staticmethod
    def validate_coordinates(city_coordinates):
        """
        Makes the necessary check given some city coordinates.

        Parameters
        ----------
        input_coordinates : List[Tuple[float, float]]
            List containing the coordinates of each city.

        Returns
        -------
            None
        """
        if not isinstance(city_coordinates, list):
            raise TypeError("The coordinates should be a list")

        for each_entry in city_coordinates:
            if not isinstance(each_entry, tuple):
                raise TypeError("The coordinates should be contained in a tuple")

            for each_value in each_entry:
                if not isinstance(each_value, float) and not isinstance(
                    each_value, int
                ):
                    raise TypeError("The coordinates must be of type float or int")

    @staticmethod
    def validate_distance_matrix(distance_matrix):
        """
        Makes the necessary check given some distance matrix.

        Parameters
        ----------
        distance_matrix : List[List[float]]
            Distance between cities given as list of list representing a matrix

        Returns
        -------
            None
        """
        if not isinstance(distance_matrix, list):
            raise TypeError("The distance matrix should be a list")

        for each_entry in distance_matrix:
            if not isinstance(each_entry, list):
                raise TypeError("Each row in the distance matrix should be a list")

            for each_value in each_entry:
                if not isinstance(each_value, float) and not isinstance(
                    each_value, int
                ):
                    raise TypeError(
                        "The distance matrix entries must be of type float or int"
                    )

                if each_value < 0:
                    raise ValueError("Distances should be positive")

    @staticmethod
    def validate_graph(G):
        """
        Makes the necessary check given some (weighted) graph.

        Parameters
        ----------
        G: nx.Graph
            Graph encoding the connectivity between cities (can be directed)

        Returns
        -------
            None
        """
        # Set edge weights to be the distances between corresponding cities
        for u, v, weight in G.edges(data="weight"):
            print(weight)
            if not isinstance(weight, float) and not isinstance(weight, int):
                raise TypeError("The edge weights must be of type float or int")

            if weight < 0:
                raise ValueError("Edge weights should be positive")

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the Traveling Salesman problem with
        fully connected cities.

        Parameters
        ----------
        n_cities: int
            The number of cities in the TSP instance. This is a required keyword argument.

        Returns
        -------
            A random instance of the Traveling Salesman problem.
        """
        n_cities = check_kwargs(["n_cities"], [None], **kwargs)[0]

        # Set a random number generator
        seed = kwargs.get("seed", None)
        seed = seed if isinstance(seed, int) else None
        rng = np.random.default_rng(seed)

        # Generate random coordinates in a box of size sqrt(n_cities) x sqrt(n_cities)
        box_size = np.sqrt(n_cities)
        city_coordinates = list(map(tuple, box_size * rng.random(size=(n_cities, 2))))
        return TSP(city_coordinates=city_coordinates)

    def terms_and_weights(self):
        """
        Returns the terms and weights used in the QUBO formulation of this TSP instance.
        The QUBO formulation used is the one presented in Section 7.2 in
        https://arxiv.org/pdf/1302.5843.pdf, and sets the first city to be visited to be
        the first city in order to reduce the number of variables.

        Returns
        -------
        Tuple[List[List[int]], List[float]]
            Tuple containing a list with the terms and a list with the corresponding weights.
        """

        # Constants (flags) useful for the helper function below
        ZERO_VALUED_VARIABLE = -2
        ONE_VALUED_VARIABLE = -1

        def get_variable_index(v, j):
            """
            Returns the actual configuration index given the two indices v (city) and j (step),
            to mirror the formulation given in https://arxiv.org/pdf/1302.5843.pdf. Whenever the
            city or step probed is the first one, it can also return a flag saying whether the
            variable is 0 (flag=-2) or 1 (flag=-1), since the first city is fixed to reduce the
            number of variables).
            """
            if j > self.n_cities + 1 or v > self.n_cities:
                raise ValueError("Index out of bounds")

            # Whenever the step is the first one (or n+1 equivalently)
            if j == 1 or j == self.n_cities + 1:
                # If the city is the first one, we have x_{1, 1} = 1
                if v == 1:
                    variable_index = ONE_VALUED_VARIABLE
                # Else we have x_{v, 1} = 0
                else:
                    variable_index = ZERO_VALUED_VARIABLE

            # When step j>1 is given
            else:
                # If first node, then x_{1, j} = 0
                if v == 1:
                    variable_index = ZERO_VALUED_VARIABLE
                # Else return the index corresponding to variable x_{v, j}
                else:
                    variable_index = (j - 2) * (self.n_cities - 1) + (v - 2)

            return variable_index

        # Init the various terms
        constant_term = 0
        single_terms = []
        interaction_terms = []

        # Constraints ensuring that a city only appears once in the cycle, and that there is only one city per step
        # (note that it was simplified to account that the first city is always city 1)
        constant_term += 2 * self.A * (self.n_cities - 1)

        for v in range(2, self.n_cities + 1):
            for j in range(2, self.n_cities + 1):
                single_terms.append(([get_variable_index(v, j)], -4 * self.A))

        for k in range(2, self.n_cities + 1):
            for l in range(2, self.n_cities + 1):
                for v in range(2, self.n_cities + 1):
                    interaction_terms.append(
                        ([get_variable_index(v, k), get_variable_index(v, l)], self.A)
                    )

        for j in range(2, self.n_cities + 1):
            for u in range(2, self.n_cities + 1):
                for v in range(2, self.n_cities + 1):
                    interaction_terms.append(
                        ([get_variable_index(u, j), get_variable_index(v, j)], self.A)
                    )

        # Constraint which penalizes going through edges which are not part of the graph
        for u, v in nx.complement(self.graph).edges():
            for j in range(1, self.n_cities + 1):
                interaction_terms.append(
                    (
                        [
                            get_variable_index(u + 1, j),
                            get_variable_index(v + 1, j + 1),
                        ],
                        self.A,
                    )
                )

        # Terms to account for the path cost
        for u, v in self.graph.edges():
            for j in range(1, self.n_cities + 1):
                interaction_terms.append(
                    (
                        [
                            get_variable_index(u + 1, j),
                            get_variable_index(v + 1, j + 1),
                        ],
                        self.B * self.graph[u][v]["weight"],
                    )
                )

        # Filtering linear and quadratic terms such that variables which are fixed (and have been flagged)
        # can be processed accordingly
        filtered_interaction_terms = []
        for interaction, weight in single_terms + interaction_terms:
            # If the term is non-zero (so no flag=-2 is present), we should consider it
            if ZERO_VALUED_VARIABLE not in interaction:
                # If the same variable appears in a quadratic term, it becomes a linear term
                if len(interaction) == 2 and interaction[0] == interaction[1]:
                    interaction.pop()

                # Update interaction to reduce the degree of a term if some variables are set to 1
                # (that is remove all flag=-1)
                interaction = list(
                    filter(lambda a: a != ONE_VALUED_VARIABLE, interaction)
                )

                # Add the updated term
                filtered_interaction_terms.append((interaction, weight))

        # Unzip to retrieve terms and weights in separate sequences
        return tuple(zip(*(filtered_interaction_terms + [([], constant_term)])))

    @property
    def qubo(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        n = (self.n_cities - 1) ** 2
        terms, weights = self.terms_and_weights()

        # Convert to Ising equivalent since variables are in {0, 1} rather than {-1, 1}
        ising_terms, ising_weights = QUBO.convert_qubo_to_ising(n, terms, weights)
        return QUBO(n, ising_terms, ising_weights, self.problem_instance)
