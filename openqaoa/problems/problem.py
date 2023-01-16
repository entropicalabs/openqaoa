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
            # convert networkx graphs to dictionaries for serialization (to get back to a graph, use nx.node_link_graph)
            new_value = (
                nx.node_link_data(value) if isinstance(value, nx.Graph) else value
            )
            yield (new_key, new_value)

    @property
    def problem_instance(self):
        """
        Returns a dictionary containing the serialization of the class and the problem type name, which will be passed as metadata to the QUBO class.

        Returns
        -------
            A dictionary containing the serialization of the class and the problem type name.
        """
        return {**{"problem_type": self.__name__}, **dict(self)}
