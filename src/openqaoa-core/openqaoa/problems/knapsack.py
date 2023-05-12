import itertools
import numpy as np

from ..utilities import check_kwargs
from .problem import Problem
from .qubo import QUBO


class Knapsack(Problem):
    """
    Creates an instance of the Kanpsack problem.

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
        An instance of the Knapsack problem.
    """

    __name__ = "knapsack"

    def __init__(self, values, weights, weight_capacity, penalty):
        # Check whether the input is valid. Number of values should match the number of weights.
        if len(values) != len(weights):
            raise ValueError("Number of items does not match given value and weights")

        self.values = values
        self.weights = weights
        self.weight_capacity = weight_capacity
        self.penalty = penalty
        self.n_items = len(weights)

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, input_values):
        if not isinstance(input_values, list):
            raise TypeError("The input parameter, values, has to be a list")

        for each_entry in input_values:
            if not isinstance(each_entry, int):
                raise TypeError("The elements in values list must be of type int.")

        self._values = input_values

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, input_weights):
        if not isinstance(input_weights, list):
            raise TypeError("The input parameter, weights, has to be a list")

        for each_entry in input_weights:
            if not isinstance(each_entry, int):
                raise TypeError("The elements in weights list must be of type int.")

        self._weights = input_weights

    @property
    def weight_capacity(self):
        return self._weight_capacity

    @weight_capacity.setter
    def weight_capacity(self, input_weight_capacity):
        if not isinstance(input_weight_capacity, int):
            raise TypeError(
                "The input parameter, weight_capacity, has to be of type int"
            )

        if input_weight_capacity <= 0:
            raise TypeError(
                "The input parameter, weight_capacity, must be a positive integer greater than 0"
            )

        self._weight_capacity = input_weight_capacity

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
        Creates a random instance of the Knapsack problem.

        Parameters
        ----------
        n_items: int
            The number of items that can be placed in the knapsack.

        Returns
        -------
            A random instance of the Knapsack problem.
        """
        n_items = check_kwargs(["n_items"], [None], **kwargs)[0]

        # Set a random number generator
        seed = kwargs.get("seed", None)
        seed = seed if isinstance(seed, int) else None
        rng = np.random.default_rng(seed)

        values = list(map(int, rng.integers(1, n_items, size=n_items)))
        weights = list(map(int, rng.integers(1, n_items, size=n_items)))

        min_weights = np.min(weights)
        max_weights = np.max(weights)

        if min_weights != max_weights:
            weight_capacity = int(
                rng.integers(min_weights * n_items, max_weights * n_items)
            )
        else:
            weight_capacity = int(rng.integers(max_weights, max_weights * n_items))

        penalty = 2 * np.max(values)
        return Knapsack(values, weights, weight_capacity, int(penalty))

    def terms_and_weights(self):
        n_variables_slack = int(np.ceil(np.log2(self.weight_capacity)))
        n_variables = self.n_items + n_variables_slack

        # Edges between variables to represent slack value (the s_j's)
        edges_slacks = itertools.combinations(range(n_variables_slack), 2)
        edges_slacks_with_weights = [
            (list(e), 2 * self.penalty * (2 ** e[0]) * (2 ** e[1]))
            for e in edges_slacks
        ]

        # Edges between decision variables for weights (the x_i's)
        edges_decision_vars = itertools.combinations(
            range(n_variables_slack, n_variables), 2
        )
        edges_decision_vars_with_weights = [
            (
                list(e),
                2
                * self.penalty
                * self.weights[e[0] - n_variables_slack]
                * self.weights[e[1] - n_variables_slack],
            )
            for e in edges_decision_vars
        ]

        # Edges between decisions and variables to represent slack value (the x_i's and s_j's)
        edges_slacks_decision_vars = itertools.product(
            range(n_variables_slack), range(n_variables_slack, n_variables)
        )
        edges_slacks_decision_vars_with_weights = [
            (
                list(e),
                2 * self.penalty * (2 ** e[0]) * self.weights[e[1] - n_variables_slack],
            )
            for e in edges_slacks_decision_vars
        ]

        # Linear terms for the variables to represent slack value (s_j's)
        single_interaction_slacks = [
            ([i], self.penalty * (2 ** (2 * i) - 2 * self.weight_capacity * 2**i))
            for i in range(n_variables_slack)
        ]

        # Linear terms for the decision variables (the x_i's)
        single_interaction_decisions_vars = [
            (
                [i],
                self.penalty * self.weights[i - n_variables_slack] ** 2
                - 2
                * self.penalty
                * self.weight_capacity
                * self.weights[i - n_variables_slack]
                - self.values[i - n_variables_slack],
            )
            for i in range(n_variables_slack, n_variables)
        ]

        # The constant term
        constant_term = [([], self.penalty * self.weight_capacity**2)]

        # Unzip to retrieve terms and weights in separate sequences
        return tuple(
            zip(
                *(
                    edges_slacks_with_weights
                    + edges_decision_vars_with_weights
                    + edges_slacks_decision_vars_with_weights
                    + single_interaction_slacks
                    + single_interaction_decisions_vars
                    + constant_term
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
        n_variables_slack = int(np.ceil(np.log2(self.weight_capacity)))
        n = self.n_items + n_variables_slack
        terms, weights = self.terms_and_weights()

        # Convert to Ising equivalent since variables are in {0, 1} rather than {-1, 1}
        ising_terms, ising_weights = QUBO.convert_qubo_to_ising(n, terms, weights)
        return QUBO(n, ising_terms, ising_weights, self.problem_instance)


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

    __name__ = "slack_free_knapsack"

    def __init__(self, values, weights, weight_capacity, penalty):
        super().__init__(values, weights, weight_capacity, penalty)

    @staticmethod
    def random_instance(**kwargs):
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
        n_items = check_kwargs(["n_items"], [None], **kwargs)[0]

        # Set a random number generator
        seed = kwargs.get("seed", None)
        seed = seed if isinstance(seed, int) else None
        rng = np.random.default_rng(seed)

        values = list(map(int, rng.integers(1, n_items, size=n_items)))
        weights = list(map(int, rng.integers(1, n_items, size=n_items)))

        min_weights = np.min(weights)
        max_weights = np.max(weights)
        if min_weights != max_weights:
            weight_capacity = int(
                rng.integers(min_weights * n_items, max_weights * n_items)
            )
        else:
            weight_capacity = int(rng.integers(max_weights, max_weights * n_items))

        penalty = 2 * np.max(values)
        return SlackFreeKnapsack(values, weights, weight_capacity, int(penalty))

    def terms_and_weights(self):
        """
        Implementation of single and two-qubit terms in the slack-free Hamiltonian
        for the Knapsack problem.
        """

        n_variables = self.n_items

        # Edges between decision variables for weights (the x_i's)
        edges_decision_vars = itertools.combinations(range(n_variables), 2)
        edges_decision_vars_with_weights = [
            (list(e), 2 * self.penalty * self.weights[e[0]] * self.weights[e[1]])
            for e in edges_decision_vars
        ]

        # Linear terms for the decision variables (the x_i's)
        single_interaction_decisions_vars = [
            (
                [i],
                self.penalty * self.weights[i] ** 2
                - 2 * self.penalty * self.weight_capacity * self.weights[i]
                - self.values[i],
            )
            for i in range(n_variables)
        ]

        # The constant term
        constant_term = [([], self.penalty * self.weight_capacity**2)]

        # Unzip to retrieve terms and weights in separate sequences
        return tuple(
            zip(
                *(
                    edges_decision_vars_with_weights
                    + single_interaction_decisions_vars
                    + constant_term
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
        n = self.n_items
        terms, weights = self.terms_and_weights()

        # Convert to Ising equivalent since variables are in {0, 1} rather than {-1, 1}
        ising_terms, ising_weights = QUBO.convert_qubo_to_ising(n, terms, weights)
        return QUBO(n, ising_terms, ising_weights, self.problem_instance)
