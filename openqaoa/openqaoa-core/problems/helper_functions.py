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

import networkx as nx
import inspect

from .problem import Problem
from .knapsack import Knapsack, SlackFreeKnapsack
from .maximumcut import MaximumCut
from .minimumvertexcover import MinimumVertexCover
from .numberpartition import NumberPartition
from .shortestpath import ShortestPath
from .tsp import TSP
from .converters import FromDocplex2IsingModel
from .qubo import QUBO


def create_problem_from_dict(problem_instance: dict) -> Problem:
    """
    Creates an object of the class corresponding to the problem type in the input instance, with the same attributes as the input instance.
    
    Parameters
    ----------
    problem_instance: dict
        The input instance.
        
    Returns
    -------
        An object of the class corresponding to the problem type in the input instance.
    """

    # copy the instance to avoid modifying the original instance
    problem_instance = problem_instance.copy()

    # mapper from problem type to class
    problem_mapper = {
        "generic_qubo": QUBO,
        "tsp": TSP,
        "number_partition": NumberPartition,
        "maximum_cut": MaximumCut,
        "knapsack": Knapsack,
        "slack_free_knapsack": SlackFreeKnapsack,
        "minimum_vertex_cover": MinimumVertexCover,
        "shortest_path": ShortestPath,
    }

    # check if the problem type is in the mapper
    assert (
        problem_instance["problem_type"] in problem_mapper
    ), f"Problem type {problem_instance['problem_type']} not supported."

    # get the class corresponding to the problem type
    problem_class = problem_mapper[problem_instance.pop("problem_type", "generic_qubo")]

    # check if the problem type is QUBO, if so, raise an exception
    if problem_class is QUBO:
        raise Exception(
            "This method does not work for generic QUBO. The input instance has type `generic_qubo`. You can use the `from_dict` method of the `QUBO` class instead."
        )

    # if the instance has a graph, convert it to a networkx graph
    if "G" in problem_instance:
        problem_instance["G"] = nx.node_link_graph(problem_instance["G"])

    # erase the keys that are not arguments of the class
    arguments = inspect.getfullargspec(problem_class).args[1:]
    for key in problem_instance.copy():
        if key not in arguments:
            del problem_instance[key]

    # create the problem instance and return it
    return problem_class(**problem_instance)
