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

from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from random import seed

import networkx as nx
import numpy as np
import scipy
import scipy.spatial
import itertools

from .helper_functions import convert2serialize, check_kwargs
from openqaoa.qaoa_parameters.operators import Hamiltonian


class Problem(ABC):

    @staticmethod
    @abstractmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the problem.

        Parameters
        ----------
        **kwargs:
            Required keyword arguments

        Returns
        -------
            A random instance of the problem.
        """
        pass


class QUBO:
    """
    Creates an instance of Quadratic Unconstrained Binary Optimization (QUBO)
    class, which offers a way to encode optimization problems.

    Parameters
    ----------
    n: `int`
        The number of variables in the representation.
    terms: `List[Tuple[int, ...],List]`
        The different terms in the QUBO encoding, indicating the different
        interactions between variables.
    weights: `List[float]`
        The list of weights (or coefficients) corresponding to each
        interaction defined in `terms`.
    clean_terms_and_weights: `bool`
        Boolean indicating whether terms and weights can be cleaned by
        combining similar terms.

    Returns
    -------
        An instance of the Polynomial Unconstrained Binary Optimization 
        (QUBO) class.
    """
    # Maximum number of terms allowed to enable the cleaning procedure
    TERMS_CLEANING_LIMIT = 5000

    def __init__(self, n, terms, weights, clean_terms_and_weights=False):
        terms = list(terms)
        weights = list(weights)

        # Check that terms and weights have matching lengths
        if len(terms) != len(weights):
            raise ValueError('The number of terms and number of weights do not match')

        constant = 0
        try:
            constant_index = [i for i, term in enumerate(terms) if len(term) == 0][0]
            constant = weights.pop(constant_index)
            terms.pop(constant_index)
        except:
            pass
        
        # If the user wants to clean the terms and weights or if the number of
        # terms is not too big, we go through the cleaning process
        if clean_terms_and_weights or len(terms) <= QUBO.TERMS_CLEANING_LIMIT:
            self.terms, self.weights = QUBO.clean_terms_and_weights(terms, weights)
        else:
            self.terms, self.weights = terms, weights

        self.constant = constant
        self.n = n

    @property
    def hamiltonian(self):
        """
        Returns the Hamiltonian of the problem.
        """
        return Hamiltonian.classical_hamiltonian(self.terms,self.weights,self.constant)

    def asdict(self):
        return convert2serialize(self)

    @staticmethod
    def clean_terms_and_weights(terms, weights):
        """ Goes through the terms and weights and group them when possible"""
        # List to record the terms as sets
        unique_terms = []

        # Will record the weight for the unique terms (note that since Sets are
        # unhashable in Python, we use a dict with integers for the keys, that
        # are mapped with the corresponding indices of terms from unique_terms)
        new_weights_for_terms = defaultdict(float)

        # We do one pass over terms and weights
        for term, weight in zip(terms, weights):

            # Convert the term to a set
            term_set = set(term)

            # If this term is not yet recorded, we add it to the list of unique
            # terms and we use that it is the last element to find its index
            if term_set not in unique_terms:
                unique_terms.append(term_set)
                term_index = len(unique_terms) - 1

            # Else if the term is alreaddy recorded, we just need to retrieve
            # its index in the unique_terms list
            else:
                term_index = unique_terms.index(term_set)

            # Update the weight in the dictionary using the retrieved index
            new_weights_for_terms[term_index] += weight

        # Return terms and weights, making sure to convert the terms back to lists
        return [list(term) for term in unique_terms], list(new_weights_for_terms.values())

    @staticmethod
    def random_qubo(n, density=0.5, format_m='coo', max_abs_value=100):
        # Generate a random matrix (elements in [0, 1]) of type sparse
        random_matrix = scipy.sparse.rand(n,
                                          n,
                                          density=density,
                                          format=format_m)

        # Retrieve the indices of non-zero elements of the matrix as list of tuples
        terms = np.transpose(random_matrix.nonzero())

        # Get the matrix entries in a list, but scale the elements and
        # make them centered at 0 by subtracting 0.5
        weights = max_abs_value * (random_matrix.data - 0.5)

        # Return the terms and weights, taking care of converting to the correct types
        return QUBO(n, [list(map(int, i)) for i in terms],
                    [float(i) for i in weights])
    

    @property
    def hamiltonian(self):
        """
        Returns the Hamiltonian of the problem.
        """
        return Hamiltonian.classical_hamiltonian(self.terms,self.weights,self.constant)

class TSP(Problem):
    """
    Creates an instance of the Traveling Salesman problem.

    Parameters
    ----------
    coordinates: List[Tuple[float, float]]
        The list of coordinates of the different cities.

    Returns
    -------
        An instance of the Traveling Salesman problem.
    """
    def __init__(self, coordinates=None):
        
        self.coordinates = np.array(coordinates)
        self.n_cities = self.coordinates.shape[0]

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the Traveling Salesman problem.

        Parameters
        ----------
        n_cities: int
            The number of cities in the TSP instance. This is a required 
            keyword argument.

        Returns
        -------
            A random instance of the Traveling Salesman problem.
        """
        n_cities = check_kwargs(['n_cities'], [None], **kwargs)[0]
        seed = kwargs.get('seed')
        
        if isinstance(seed, int):
            np.random.seed(seed)
        
        box_size = np.sqrt(n_cities)
        coordinates = box_size * np.random.rand(n_cities, 2)
        return TSP(coordinates)

    def get_distance_matrix(self):
        # Return distance matrix: it uses Euclidean distance
        return scipy.spatial.distance_matrix(self.coordinates,
                                             self.coordinates)

    def all_pairs_all_steps(self):
        all_pairs_dict = {}
        for i in range(self.n_cities):
            for j in range(self.n_cities):

                if i != j:
                    all_pairs_dict[(i, j)] = TSP.city_pair_all_steps(
                        i, j, self.n_cities)

        return all_pairs_dict

    @staticmethod
    def city_pair_all_steps(c1, c2, n_cities):
        var_pairs = [(c1 + n_cities * j, c2 + n_cities * (j + 1))
                     for j in range(n_cities)]

        return var_pairs

    def TSP_instance_dict(self, pairs_dict, dists):
        problem_dict = defaultdict(float)

        for city_pair, var_pairs in pairs_dict.items():

            city_dist = dists[city_pair]

            for pair in var_pairs:
                problem_dict[pair] += city_dist

        return problem_dict

    def get_qubo_problem(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        distance_matrix = self.get_distance_matrix()

        # Basic problem
        polys_dict = TSP.all_pairs_all_steps(self)
        problem_dict = TSP.TSP_instance_dict(self, polys_dict, distance_matrix)
        pairs = [list(pair) for pair in problem_dict.keys()]
        coeffs = list(problem_dict.values())
        n = self.n_cities * (self.n_cities + 1)

        ising_terms, ising_coeffs = [], []
        
        constant_term = 0
        linear_terms = np.zeros(n)
        
        # Process the given terms and weights
        for weight, term in zip(coeffs, pairs):

            if len(term) == 2:
                u, v = term

                if u != v:
                    ising_terms.append([u, v])
                    ising_coeffs.append(weight / 4)
                else:
                    constant_term += weight / 4
                
                linear_terms[term[0]] -= weight / 4
                linear_terms[term[1]] -= weight / 4
                constant_term += weight / 4
            elif len(term) == 1:
                linear_terms[term[0]] -= weight / 2
                constant_term += weight / 2
            else:
                constant_term += weight
        
        for variable, linear_term in enumerate(linear_terms):
            ising_terms.append([variable])
            ising_coeffs.append(linear_term)
        
        ising_terms.append([])
        ising_coeffs.append(constant_term)

        return QUBO(n,ising_terms,ising_coeffs)

class NumberPartition(Problem):
    """
    Creates an instance of the Number Partitioning problem.

    Parameters
    ----------
    numbers: `List[int]`
        The list of numbers to be partitioned.

    Returns
    -------
        An instance of the Number Partitioning problem.
    """
    def __init__(self, numbers=None):
        # Set the numbers to be partitioned. If not given, generate a random list with integers
        self.numbers = numbers
        self.n_numbers = None if numbers==None else len(self.numbers)

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the Number Partitioning problem.

        Parameters
        ----------
        n_numbers: `int`
            The number of numbers to be partitioned. This is a required 
            keyword argument.

        Returns
        -------
            A random instance of the Number Partitioning problem.
        """
        n_numbers = check_kwargs(['n_numbers'], [None], **kwargs)
        seed = kwargs.get('seed')
        
        if isinstance(seed, int):
            np.random.seed(seed)
        
        numbers = list(map(int, np.random.randint(1, 10, size=n_numbers)))
        return NumberPartition(numbers)

    def get_qubo_problem(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        terms = []
        weights = []
        constant_term = 0

        # Consider every pair of numbers (ordered)
        for i in range(self.n_numbers):
            for j in range(i, self.n_numbers):

                # If i equals j, then whatever random sign we choose, if we square
                # it we can back 1. So we have a constant term.
                if i == j:
                    constant_term += self.numbers[i] * self.numbers[j]

                # Otherwise the weight is computed as being the product of the
                # numbers in the pair, multiplied by 2 (since we account for
                # both pair (i, j) and (j, i)
                else:
                    term = [i, j]
                    weight = 2 * self.numbers[i] * self.numbers[j]

                    terms.append(term)
                    weights.append(weight)

        # If the constant term is non-zero, we may add it to terms and weights
        if constant_term > 0:
            terms.append([])
            weights.append(constant_term)

        return QUBO(self.n_numbers, terms, weights)

class MaximumCut(Problem):
    """
    Creates an instance of the Maximum Cut problem.

    Parameters
    ----------
    G: `nx.Graph`
        The input graph as NetworkX graph instance.

    Returns
    -------
        An instance of the Maximum Cut problem.
    """

    DEFAULT_EDGE_WEIGHT = 1.0

    def __init__(self, G):

        # Relabel nodes to integers starting from 0
        mapping = dict(zip(G, range(G.number_of_nodes())))
        self.G = nx.relabel_nodes(G, mapping)

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
        n_nodes, edge_probability = check_kwargs(['n_nodes', 'edge_probability'],
                                                        [None, None], **kwargs)
        seed = kwargs.get('seed', None)

        G = nx.generators.random_graphs.fast_gnp_random_graph(n=n_nodes, p=edge_probability, seed=seed)
        return MaximumCut(G)

    def get_qubo_problem(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        # Iterate over edges (with weight) and store accordingly
        terms = []
        weights = []

        for u, v, edge_weight in self.G.edges(data='weight'):
            terms.append([u, v])

            # We expect the edge weight to be given in the attribute called
            # "weight". If it is None, assume a weight of 1.0
            weights.append(edge_weight if edge_weight else MaximumCut.DEFAULT_EDGE_WEIGHT)

        return QUBO(self.G.number_of_nodes(), terms, weights)

class Knapsack(Problem):
    """
    Creates an instance of the Kanpsack problem.

    Parameters
    ----------
    values: `List[int]`
        The values of the items that can be placed in the kanpsack.
    weights: `List[int]`
        The weight of the items that can be placed in the knapsack.
    weight_capacity: `int`
        The maximum weight the knapsack can hold.
    penalty: `float`
        Penalty for the weight constraint.

    Returns
    -------
        An instance of the Knapsack problem.
    """
    def __init__(self, values, weights, weight_capacity, penalty):
        # Check whether the input is valid. Number of values should match the number of weights.
        if len(values) != len(weights):
            raise ValueError('Number of items does not match given value and weights')

        self.n_items = len(weights)
        self.values = values
        self.weights = weights
        self.weight_capacity = weight_capacity
        self.penalty = penalty

    @classmethod
    def random_instance(cls, **kwargs):
        """
        Creates a random instance of the Knapsack problem.

        Parameters
        ----------
        n_items: int
            The number of items that can be placed in the knapsack.
        
        Returns
        -------
            A random instance of the Knapsack problem.
        """
        n_items = check_kwargs(['n_items'], [None], **kwargs)[0]
        seed = kwargs.get('seed')

        if isinstance(seed, int):
            np.random.seed(seed)

        values = list(map(int, np.random.randint(1, n_items, size=n_items)))
        weights = list(map(int, np.random.randint(1, n_items, size=n_items)))
        weight_capacity = np.random.randint(np.min(weights) * n_items, np.max(weights) * n_items)
        penalty = 2 * np.max(values)

        return cls(values, weights, weight_capacity, penalty)

    def terms_and_weights(self):
        n_variables_slack = int(np.ceil(np.log2(self.weight_capacity)))
        n_variables = self.n_items + n_variables_slack

        # Edges between variables to represent slack value (the s_j's)
        edges_slacks = itertools.combinations(range(n_variables_slack), 2)
        edges_slacks_with_weights = [(list(e), 2 * self.penalty * (2 ** e[0]) * (2 ** e[1])) for e in edges_slacks]

        # Edges between decision variables for weights (the x_i's)
        edges_decision_vars = itertools.combinations(range(n_variables_slack, n_variables), 2)
        edges_decision_vars_with_weights = [(list(e), 2 * self.penalty * self.weights[e[0] - n_variables_slack] * self.weights[e[1] - n_variables_slack]) for e in edges_decision_vars]

        # Edges between decisions and variables to represent slack value (the x_i's and s_j's)
        edges_slacks_decision_vars = itertools.product(range(n_variables_slack), range(n_variables_slack, n_variables))
        edges_slacks_decision_vars_with_weights = [(list(e), 2 * self.penalty * (2 ** e[0]) * self.weights[e[1] - n_variables_slack]) for e in edges_slacks_decision_vars]

        # Linear terms for the variables to represent slack value (s_j's)
        single_interaction_slacks = [([i], self.penalty * (2 ** (2*i) - 2 * self.weight_capacity * 2 ** i)) for i in range(n_variables_slack)]

        # Linear terms for the decision variables (the x_i's)
        single_interaction_decisions_vars = [([i], self.penalty * self.weights[i - n_variables_slack] ** 2 - 2 * self.penalty * self.weight_capacity * self.weights[i - n_variables_slack] - self.values[i - n_variables_slack]) for i in range(n_variables_slack, n_variables)]

        # The constant term
        constant_term = [([], self.penalty * self.weight_capacity ** 2)]

        # Unzip to retrieve terms and weights in separate sequences
        return tuple(zip(*(edges_slacks_with_weights +
                           edges_decision_vars_with_weights +
                           edges_slacks_decision_vars_with_weights +
                           single_interaction_slacks +
                           single_interaction_decisions_vars +
                           constant_term
                           )))

    def get_qubo_problem(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        n_variables_slack = int(np.ceil(np.log2(self.weight_capacity)))
        n_variables = self.n_items + n_variables_slack
        terms, weights = self.terms_and_weights()

#         ising_terms, ising_coeffs = terms,weights
        ising_terms, ising_coeffs = [], []
        
        constant_term = 0
        linear_terms = np.zeros(n_variables)
        
        # Process the given terms and weights
        for weight, term in zip(weights, terms):

            if len(term) == 2:
                u, v = term

                if u != v:
                    ising_terms.append([u, v])
                    ising_coeffs.append(weight / 4)
                else:
                    constant_term += weight / 4
                
                linear_terms[term[0]] -= weight / 4
                linear_terms[term[1]] -= weight / 4
                constant_term += weight / 4
            elif len(term) == 1:
                linear_terms[term[0]] -= weight / 2
                constant_term += weight / 2
            else:
                constant_term += weight
        
        for variable, linear_term in enumerate(linear_terms):
            ising_terms.append([variable])
            ising_coeffs.append(linear_term)
        
        ising_terms.append([])
        ising_coeffs.append(constant_term)

        return QUBO(n_variables,ising_terms,ising_coeffs)


class SlackFreeKnapsack(Knapsack):
    """
    A slack variable free approach to the Knapsack problem Hamiltonian. 
    The Hamiltonian consists of decision qubits with a quadratic penalty term centred
    on `W`, i.e. the maximum Knapsack Capacity.
    
    Creates an instance of the SlackFreeKanpsack problem.

    Parameters
    ----------
    values: List[int]
        The values of the items that can be placed in the kanpsack.
    weights: List[int]
        The weight of the items that can be placed in the knapsack.
    weight_capacity: int
        The maximum weight the knapsack can hold.
    penalty: float
        Penalty for the weight constraint.

    Returns
    -------
        An instance of the SlackFreeKnapsack problem.
    """

    def __init__(self, values, weights, weight_capacity, penalty):

        super().__init__(values, weights, weight_capacity, penalty)
    
    def terms_and_weights(self):
        """
        Implementation of single and two-qubit terms in the slack-free Hamiltonian 
        for the Knapsack problem. 
        """

        n_variables = self.n_items
                
        # Edges between decision variables for weights (the x_i's)
        edges_decision_vars = itertools.combinations(range(n_variables), 2)
        edges_decision_vars_with_weights = [(list(e), 2 * self.penalty * self.weights[e[0]] * self.weights[e[1]]) for e in edges_decision_vars]
                
        # Linear terms for the decision variables (the x_i's)
        single_interaction_decisions_vars = [([i], self.penalty * self.weights[i] ** 2 - 2 * self.penalty * self.weight_capacity * self.weights[i] - self.values[i]) for i in range(n_variables)]
        
        # The constant term
        constant_term = [([], self.penalty * self.weight_capacity ** 2)]
        
        # Unzip to retrieve terms and weights in separate sequences
        return tuple(zip(*(edges_decision_vars_with_weights + 
                           single_interaction_decisions_vars + 
                           constant_term
                           )))

    def get_qubo_problem(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        n_variables = self.n_items
        terms, weights = self.terms_and_weights()

        ising_terms, ising_coeffs = [], []
        
        constant_term = 0
        linear_terms = np.zeros(n_variables)
        
        # Process the given terms and weights
        for weight, term in zip(weights, terms):

            if len(term) == 2:
                u, v = term

                if u != v:
                    ising_terms.append([u, v])
                    ising_coeffs.append(weight / 4)
                else:
                    constant_term += weight / 4
                
                linear_terms[term[0]] -= weight / 4
                linear_terms[term[1]] -= weight / 4
                constant_term += weight / 4
            elif len(term) == 1:
                linear_terms[term[0]] -= weight / 2
                constant_term += weight / 2
            else:
                constant_term += weight
        
        for variable, linear_term in enumerate(linear_terms):
            ising_terms.append([variable])
            ising_coeffs.append(linear_term)
        
        ising_terms.append([])
        ising_coeffs.append(constant_term)

        return QUBO(n_variables,ising_terms,ising_coeffs)


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
    def __init__(self,G,field,penalty):

        # Relabel nodes to integers starting from 0
        mapping = dict(zip(G, range(G.number_of_nodes())))
        self.G = nx.relabel_nodes(G, mapping)

        self.field = field
        self.penalty = penalty

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

        n_nodes, edge_probability = check_kwargs(['n_nodes', 'edge_probability'],
                                                        [None, None], **kwargs)
        seed = kwargs.get('seed', None)
        G = nx.generators.random_graphs.fast_gnp_random_graph(n=n_nodes, p=edge_probability, seed=seed)

        DEFAULT_FIELD = 1.0
        DEFAULT_PENALTY = 10

        return MinimumVertexCover(G,DEFAULT_FIELD,DEFAULT_PENALTY)

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
        quadratic_interaction = [(list(e),self.penalty/4) for e in edges]

        # Linear terms from the artificial field
        linear_interaction = [([i],-self.field/2+connectivity[i]*self.penalty/4) if connectivity.get(i) is not None else ([i],-self.field/2) for i in range(n_nodes)]

        # Constant term
        constant_term = [([],n_nodes*self.field/2 + len(edges)*self.penalty/4)]

        # Generate tuple containing a list with the terms and a list with the weights
        terms_weights = tuple(zip(*(quadratic_interaction + linear_interaction + constant_term)))

        # Unzip to retrieve terms and weights in separate sequences
        return terms_weights

    def get_qubo_problem(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
        The QUBO encoding of this problem.
        """

        # Extract terms and weights from the problem definition
        terms, weights = self.terms_and_weights()

        return QUBO(self.G.number_of_nodes(), list(terms), list(weights))