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

import numpy as np

from .helper_functions import check_kwargs
from .problem import Problem
from .qubo import QUBO


class NumberPartition(Problem):
    """
    Creates an instance of the Number Partitioning problem.
    Parameters
    ----------
    numbers: List[int]
        The list of numbers to be partitioned.
    Returns
    -------
        An instance of the Number Partitioning problem.
    """

    __name__ = "number_partition"

    def __init__(self, numbers=None):
        # Set the numbers to be partitioned. If not given, generate a random list with integers
        self.numbers = numbers
        self.n_numbers = None if numbers == None else len(self.numbers)

    @property
    def numbers(self):
        return self._numbers

    @numbers.setter
    def numbers(self, input_numbers):

        if not isinstance(input_numbers, list):
            raise TypeError("The input parameter, numbers, has to be a list")

        for each_entry in input_numbers:
            if not isinstance(each_entry, int):
                raise TypeError(
                    "The elements in numbers list must be of type int.")

        self._numbers = input_numbers

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the Number Partitioning problem.
        Parameters
        ----------
        n_numbers: int
            The number of numbers to be partitioned. This is a required keyword argument.
        Returns
        -------
            A random instance of the Number Partitioning problem.
        """
        n_numbers = check_kwargs(["n_numbers"], [None], **kwargs)

        # Set a random number generator
        seed = kwargs.get("seed", None)
        seed = seed if isinstance(seed, int) else None
        rng = np.random.default_rng(seed)

        numbers = list(map(int, rng.integers(1, 10, size=n_numbers)))
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

        return QUBO(self.n_numbers, terms, weights, self.problem_instance)