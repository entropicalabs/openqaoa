from abc import ABC, abstractmethod
import networkx as nx


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

    def __iter__(self):
        for key, value in self.__dict__.items():
            # remove "_" from the beginning of the key if it exists
            new_key = key[1:] if key.startswith("_") else key
            # convert networkx graphs to dictionaries for serialization
            # (to get back to a graph, use nx.node_link_graph)
            new_value = (
                nx.node_link_data(value) if isinstance(value, nx.Graph) else value
            )
            yield (new_key, new_value)

    @property
    def problem_instance(self):
        """
        Returns a dictionary containing the serialization of the class and
        the problem type name, which will be passed as metadata to the QUBO class.

        Returns
        -------
            A dictionary containing the serialization of the class
            and the problem type name.
        """
        return {**{"problem_type": self.__name__}, **dict(self)}
