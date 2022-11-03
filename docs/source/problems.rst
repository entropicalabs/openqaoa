Problems
========
.. autoclass:: openqaoa.problems.problem.Problem
    :members:
    :undoc-members:
    :inherited-members:


.. autoclass:: openqaoa.problems.problem.QUBO
    :members:
    :undoc-members:
    :inherited-members:


Built-in problems
=================

Traveling Salesman Problem
----------------------------------

The Traveling Salesman Problem (TSP) requires to find, given a list of cities and the distances between each pair of cities (or the cities coordinates), the shortest possible path that visits each city exactly once and returns to the origin city. Additionally, one can also specify how cities are connected together.
Our implementation accepts three different kind of inputs:
 * A list of the cities' coordinates and, optionally, a (directed) graph specifiying the connectivity between cities
 * A distance matrix encoding distances between each pair of cities and, optionally, a (directed) graph specifiying the connectivity between cities
 * A weighted (directed) graph specifiying the connectivity and the distance between cities

.. autoclass:: openqaoa.problems.problem.TSP
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: openqaoa.problems.problem.NumberPartition
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: openqaoa.problems.problem.MaximumCut
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: openqaoa.problems.problem.Knapsack
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: openqaoa.problems.problem.SlackFreeKnapsack
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: openqaoa.problems.problem.MinimumVertexCover
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Helper functions
----------------

.. automodule:: openqaoa.problems.helper_functions
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

.. autoclass:: openqaoa.problems.problem.ShortestPath
    :members:
    :show-inheritance:
    :inherited-members:
