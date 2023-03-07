from collections import defaultdict
from typing import List
import json
import numpy as np
import scipy

from ..qaoa_components import Hamiltonian
from ..utilities import delete_keys_from_dict, convert2serialize


class QUBO(object):
    """
    Creates an instance of Quadratic Unconstrained Binary Optimization (QUBO)
    class, which offers a way to encode optimization problems.

    Parameters
    ----------
    n: int
        The number of variables in the representation.
    terms: List[Tuple[int, ...],List]
        The different terms in the QUBO encoding, indicating the
        different interactions between variables.
    weights: List[float]
        The list of weights (or coefficients) corresponding to each
        interaction defined in `terms`.
    clean_terms_and_weights: bool
        Boolean indicating whether terms and weights can be cleaned


    Returns
    -------
        An instance of the Quadratic Unconstrained Binary Optimization (QUBO) class.
    """

    # Maximum number of terms allowed to enable the cleaning procedure
    TERMS_CLEANING_LIMIT = 5000

    def __init__(
        self,
        n,
        terms,
        weights,
        problem_instance: dict = {"problem_type": "generic_qubo"},
        clean_terms_and_weights=False,
    ):

        # check-type for terms and weights
        if not isinstance(terms, list) and not isinstance(terms, tuple):
            raise TypeError(
                "The input parameter terms must be of type of list or tuple"
            )

        if not isinstance(weights, list) and not isinstance(weights, tuple):
            raise TypeError(
                "The input parameter weights must be of type of list or tuple"
            )

        for each_entry in weights:
            if not isinstance(each_entry, float) and not isinstance(each_entry, int):
                raise TypeError(
                    "The elements in weights list must be of type float or int."
                )

        terms = list(terms)
        weights = list(weights)

        # Check that terms and weights have matching lengths
        if len(terms) != len(weights):
            raise ValueError("The number of terms and number of weights do not match")

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

        # attribute to store the problem instance, it will be checked
        # if it is json serializable in the __setattr__ method
        self.problem_instance = problem_instance

        # Initialize the metadata dictionary
        self.metadata = {}

    def __iter__(self):
        for key, value in self.__dict__.items():
            # remove "_" from the beginning of the key if it exists
            yield (key[1:] if key.startswith("_") else key, value)

    def __setattr__(self, __name, __value):
        # check if problem_instance is json serializable, also check if
        # metadata is json serializable
        if __name == "problem_instance" or __name == "metadata":
            try:
                _ = json.dumps(__value)
            except Exception as e:
                raise e

        super().__setattr__(__name, __value)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, input_n):

        if not isinstance(input_n, int):
            raise TypeError("The input parameter, n, has to be of type int")

        if input_n <= 0:
            raise TypeError(
                "The input parameter, n, must be a positive integer greater than 0"
            )

        self._n = input_n

    def set_metadata(self, metadata: dict = {}):
        """


        Parameters
        ----------
        metadata: dict
            The metadata of the problem. All keys and values will
            be stored in the metadata dictionary.
        """

        # update the metadata (it will be checked if it is json
        # serializable in the __setattr__ method)
        self.metadata = {**self.metadata, **metadata}

    def asdict(self, exclude_keys: List[str] = []):
        """
        Returns a dictionary containing the serialization of the class.

        Parameters
        ----------
        exclude_keys: List[str]


        Returns
        -------
            A dictionary containing the serialization of the class.
        """

        if exclude_keys == []:
            return convert2serialize(dict(self))
        else:
            return delete_keys_from_dict(
                obj=convert2serialize(dict(self)), keys_to_delete=exclude_keys
            )

    @staticmethod
    def from_dict(dict: dict, clean_terms_and_weights=False):
        """


        Parameters
        ----------
        dict: dict
            The dictionary containing the serialization of the QUBO object.
        clean_terms_and_weights: bool


        Returns
        -------
            A QUBO object.
        """

        # make a copy of the dictionary to avoid modifying the input
        dict = dict.copy()

        # extract the metadata
        metadata = dict.pop("metadata", {})

        # make a copy of the terms and weights to avoid modifying the input
        dict["terms"] = dict["terms"].copy()
        dict["weights"] = dict["weights"].copy()

        # add the constant term
        dict["terms"].append([])
        dict["weights"].append(dict.pop("constant", 0))

        # create the QUBO object
        qubo = QUBO(**dict, clean_terms_and_weights=clean_terms_and_weights)

        # add the metadata
        qubo.metadata = metadata.copy()

        # return the QUBO object
        return qubo

    @staticmethod
    def clean_terms_and_weights(terms, weights):
        """Goes through the terms and weights and group them when possible"""
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
        return (
            [list(term) for term in unique_terms],
            list(new_weights_for_terms.values()),
        )

    @staticmethod
    def random_instance(n, density=0.5, format_m="coo", max_abs_value=100):
        # Generate a random matrix (elements in [0, 1]) of type sparse
        random_matrix = scipy.sparse.rand(n, n, density=density, format=format_m)

        # Retrieve the indices of non-zero elements of the matrix as list of tuples
        terms = np.transpose(random_matrix.nonzero())

        # Get the matrix entries in a list, but scale the elements and
        # make them centered at 0 by subtracting 0.5
        weights = max_abs_value * (random_matrix.data - 0.5)

        # Return the terms and weights, taking care of converting to the correct types
        return QUBO(n, [list(map(int, i)) for i in terms], [float(i) for i in weights])

    @staticmethod
    def convert_qubo_to_ising(n, qubo_terms, qubo_weights):
        """Convert QUBO terms and weights to their Ising equivalent"""
        ising_terms, ising_weights = [], []
        constant_term = 0
        linear_terms = np.zeros(n)

        # Process the given terms and weights
        for term, weight in zip(qubo_terms, qubo_weights):

            if len(term) == 2:
                u, v = term

                if u != v:
                    ising_terms.append([u, v])
                    ising_weights.append(weight / 4)
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
            ising_weights.append(linear_term)

        ising_terms.append([])
        ising_weights.append(constant_term)
        return ising_terms, ising_weights

    @property
    def hamiltonian(self):
        """
        Returns the Hamiltonian of the problem.
        """
        return Hamiltonian.classical_hamiltonian(
            self.terms, self.weights, self.constant
        )
