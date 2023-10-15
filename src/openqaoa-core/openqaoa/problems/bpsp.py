import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union
from docplex.mp.model import Model

from .problem import Problem
from .converters import FromDocplex2IsingModel
from .qubo import QUBO


class BPSP(Problem):


    __name__ = "Binary_Paint_Shop_Problem"

    def __init__(self, car_sequence):
        
        self.car_sequence = car_sequence
        self._car_sequence = None
        self.car_positions = self.get_pos()
        self.bpsp_graph = self.construct_bpsp_graph()

    @property
    def car_sequence(self):
        return self._car_sequence

    @car_sequence.setter
    def car_sequence(self, sequence):
        # Check if each car ID appears exactly twice
        unique_ids, counts = np.unique(sequence, return_counts=True)
        if not all(count == 2 for count in counts):
            raise ValueError("Each car ID must appear exactly twice in the sequence.")

        # Check if range of car IDs is continuous
        if len(unique_ids) != (max(unique_ids) + 1):
            raise ValueError("The range of car IDs must be continuous.")

        # If all checks pass, assign the sequence to the private attribute
        self._car_sequence = sequence
    

    @staticmethod
    def random_instance(**kwargs):
        """
        Generates a random instance of the BPSP problem, based on the specified number of cars.
        
        The function creates a list with two occurrences of each car ID, then applies the Fisher-Yates shuffle
        to randomize the order of the cars. The shuffled sequence of cars is then used to create a BPSP instance.
        
        Parameters
        ----------
        **kwargs:
            num_cars: int
                The number of distinct cars to be considered in the sequence. Each car will appear twice in the 
                final sequence.
        
        Returns
        -------
            BPSP
                A random instance of the BPSP problem, based on the shuffled sequence of cars.
        """

        # Extract the number of cars from the provided keyword arguments.
        num_cars = kwargs.get("num_cars")

        # Generate a list with two occurrences of each car ID, i.e., [0, 1, ..., n, 0, 1, ..., n].
        car_sequence = np.array(list(range(num_cars)) + list(range(num_cars)))
        
        # Apply the Fisher-Yates shuffle to the car_sequence.
        # Start from the end of the list and swap the current element with a randomly chosen earlier element.
        for i in range(len(car_sequence)-1, 0, -1):
            # Select a random index between 0 and i (inclusive).
            j = np.random.randint(0, i+1)
            # Swap the elements at indices i and j.
            car_sequence[i], car_sequence[j] = car_sequence[j], car_sequence[i]
            
        # Return a BPSP instance using the shuffled car_sequence.
        return BPSP(car_sequence)
    

    def get_pos(self):
        """
        Retrieve the positions of each car ID in the car sequence.
        
        This method tracks the occurrences of each car ID in the car sequence.
        Since each car ID is expected to appear twice in the sequence, the method 
        returns a dictionary where keys are car IDs and values are tuples with the 
        positions of the two occurrences.
        
        Returns
        -------
        dict
            A dictionary where keys are car IDs and values are tuples with the positions
            of the two occurrences. For example, for a sequence [0, 1, 0, 1], the 
            output would be {0: (0, 2), 1: (1, 3)}.
        """
        
        # Initialize an empty dictionary to store car positions.
        car_pos = {}
        
        # Enumerate through each car ID in the sequence to track its positions.
        for idx, car in enumerate(self.car_sequence):
            # If the car ID is already in the dictionary, append the new position.
            if car in car_pos:
                car_pos[car].append(idx)
            # If this is the first occurrence of the car ID, initialize a list with the position.
            else:
                car_pos[car] = [idx]
        
        # Convert the lists of positions to tuples for a consistent output format.
        for car, positions in car_pos.items():
            car_pos[car] = tuple(positions)

        return car_pos
    

    def construct_bpsp_graph(self):
        """
        Construct a graph to represent the Binary Paint Shop Problem (BPSP) using the Ising model.

        This function takes a car sequence from the instance and translates it to an Ising graph.
        In the graph, nodes represent cars, and edges represent the interaction between two consecutive 
        cars in the sequence. The weight on each edge indicates the interaction strength and sign.
        
        Returns
        -------
        ising_graph : nx.Graph
            A graph representing the Ising model for the BPSP based on the car sequence of the instance.
        
        Notes
        -----
        The interaction strength is determined based on how many times the cars appeared 
        in the sequence before the current pair.
        """

        # Number of distinct cars in the sequence.
        num_cars = int(len(self.car_sequence) / 2)

        # A list to count occurrences of each car as the sequence is processed.
        car_occurrences = [0] * num_cars

        # Dictionary to hold edges and their weights.
        graph = {}

        # Helper function to add or update an edge in the graph.
        def add_edge(u, v, weight):
            # Sort the vertices to maintain a consistent edge representation.
            edge = (u, v) if u < v else (v, u)
            # Add the weight or update the existing weight of the edge.
            graph[edge] = graph.get(edge, 0) + weight

        # Process each pair of cars in the sequence.
        for i in range(len(self.car_sequence) - 1):
            # Get the current car pair.
            car1, car2 = self.car_sequence[i], self.car_sequence[i+1]

            # Get the occurrences of the cars in the sequence so far.
            occ_car1, occ_car2 = car_occurrences[car1], car_occurrences[car2]

            # Calculate the interaction weight based on car occurrences.
            weight = (-1)**(occ_car1 + occ_car2 + 1)
            
            # Add or update the graph with the edge and its weight.
            add_edge(car1, car2, weight)

            # Update the occurrence count for the first car of the current pair.
            car_occurrences[car1] += 1

        # Construct the final Ising graph with non-zero weights.
        ising_graph = nx.Graph()

        # Add edges without self-loops to the graph.
        for (u, v), weight in graph.items():
            if u != v:
                ising_graph.add_edge(u, v, weight=weight)

        # Ensure all car nodes are in the graph, even if they don't have any edges.
        ising_graph.add_nodes_from(range(num_cars))

        return ising_graph



