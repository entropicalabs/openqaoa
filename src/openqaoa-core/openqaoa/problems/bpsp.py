"""
bpsp.py

This module provides an implementation for the Binary Paint Shop Problem (BPSP), a combinatorial optimization problem arising from automotive paint shops. 
Given a fixed sequence of cars, each requiring two distinct colors, the objective is to determine the order of painting such that color changes between 
adjacent cars are minimized. The problem is NP-hard, and approximating it is also challenging. The module offers various solution strategies, including 
greedy and quantum approximate optimization approaches.

This implementation is based on the following works:
    - "Beating classical heuristics for the binary paint shop problem with the 
       quantum approximate optimization algorithm" 
       (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.104.012403)
    - "Some heuristics for the binary paint shop problem and their expected number 
       of colour changes"
       (https://www.sciencedirect.com/science/article/pii/S1570866710000559)
    - Upcoming/unpublished work by V Vijendran et al.

Author: V Vijendran (Vijey)
GitHub: https://github.com/vijeycreative
Email: v.vijendran@anu.edu.au
"""


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from docplex.mp.model import Model
from collections import defaultdict

from .problem import Problem
from .qubo import QUBO


class BPSP(Problem):
    """
    Binary Paint Shop Problem (BPSP)

    This class is dedicated to solving the Binary Paint Shop Problem (BPSP), a combinatorial 
    optimization challenge drawn from an automotive paint shop context. In this problem, there 
    exists a specific sequence of cars, with each car making two appearances in the sequence. 
    The primary objective is to find the best coloring sequence such that consecutive cars in the 
    sequence necessitate the fewest color switches.

    When initializing, the solver accepts a list of car numbers indicative of a valid BPSP instance 
    and subsequently establishes a Binary Paint Shop Problem instance based on this.

    To elaborate on the problem:
    
    Imagine we have 'n' distinct cars, labeled as c_1, c_2, ..., c_n. These cars are presented 
    in a sequence of length '2n', such that each car c_i makes two appearances. This sequence 
    is represented as w_1, w_2, ..., w_2n, with every w_i being one of the cars from the set.

    The challenge further presents two paint colors, represented as 1 and 2. For every car 
    in the sequence, a decision must be made on which color to paint it first, resulting in a 
    color assignment sequence like f_1, f_2, ..., f_2n. It's crucial to ensure that if two 
    occurrences in the sequence pertain to the same car, their color designations differ.

    The main goal is to choose a sequence of colors such that we minimize the instances where 
    consecutive cars require different paint colors. This is quantified by computing the difference 
    between color assignments for consecutive cars and aggregating this difference for the entire sequence.
    """

    __name__ = "binary_paint_shop_problem"

    def __init__(self, car_sequence):
        """
        Initialize the Binary Paint Shop Problem (BPSP) instance.

        Parameters:
        - car_sequence : list[int]
            A list of integers representing car numbers, denoting the sequence 
            in which cars appear in the BPSP. Each car number appears exactly 
            twice, symbolizing the two distinct color coatings each car requires.

        Attributes Set:
        - self.car_sequence : list[int]
            Stores the sequence of car numbers.

        - self.car_positions : dict[int, tuple]
            Maps each car number to a tuple of its two positions in the car_sequence. 
            Determined by the `car_pos` property.

        - self.bpsp_graph : networkx.Graph
            Represents the BPSP as a graph where nodes are car positions and edges 
            indicate adjacent car positions in the sequence. Constructed by the 
            `bpsp_graph` method.

        Returns:
        None
        """
        
        self.car_sequence = car_sequence
        self.car_positions = self.car_pos
        self.bpsp_graph = self.graph

    @property
    def problem_instance(self):
        instance = super().problem_instance  # Get the original problem_instance
        return self.convert_ndarrays(instance)

    @staticmethod
    def convert_ndarrays(obj):
        """
        Recursively convert numpy objects in the given object to their Python counterparts.
        Converts numpy.ndarrays to Python lists and numpy.int64 to Python int.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: BPSP.convert_ndarrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [BPSP.convert_ndarrays(item) for item in obj]
        else:
            return obj


    @property
    def car_sequence(self):
        """
        Getter for the 'car_sequence' property.

        This method retrieves the current value of the `_car_sequence` attribute, which represents 
        the sequence of cars to be painted. Each car is identified by a unique integer ID, 
        and each ID must appear exactly twice in the sequence, indicating the two times the car 
        is painted.

        Returns:
        -------
        list[int]
            The current sequence of car IDs. Each ID appears exactly twice in the sequence.
        """
        return self._car_sequence

    @car_sequence.setter
    def car_sequence(self, sequence):
        """
        Setter for the 'car_sequence' property.

        This method validates and sets a new value for the `_car_sequence` attribute. 
        The validation ensures:
        1. Each car ID appears exactly twice in the sequence.
        2. The range of car IDs is continuous (e.g., for IDs 0, 1, 2, both 0 and 2 cannot appear without 1).

        Parameters:
        ----------
        sequence : list[int]
            A new sequence of car IDs to be assigned to `_car_sequence`. 

        Raises:
        ------
        ValueError:
            If any of the validation checks fail, a ValueError is raised with an appropriate error message.

        """
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
            seed: int, optional
                The seed for the random number generator. If provided, the output will be deterministic based on this seed.
        
        Returns
        -------
            BPSP
                A random instance of the BPSP problem, based on the shuffled sequence of cars.
        """

        # Extract the number of cars and the seed from the provided keyword arguments.
        num_cars = kwargs.get("num_cars")
        seed = kwargs.get("seed", None)  # Default value is None if not provided.

        # Assert that the number of cars is greater than 0.
        assert num_cars > 0, "The number of cars must be greater than 0."

        # Generate a list with two occurrences of each car ID, i.e., [0, 1, ..., n, 0, 1, ..., n].
        car_sequence = np.array(list(range(num_cars)) + list(range(num_cars)))

        # Set the seed for numpy's random module.
        np.random.seed(seed)

        # Apply the Fisher-Yates shuffle to the car_sequence.
        # Start from the end of the list and swap the current element with a randomly chosen earlier element.
        for i in range(len(car_sequence)-1, 0, -1):
            # Select a random index between 0 and i (inclusive).
            j = np.random.randint(0, i+1)
            # Swap the elements at indices i and j.
            car_sequence[i], car_sequence[j] = car_sequence[j], car_sequence[i]

        # Return a BPSP instance using the shuffled car_sequence.
        return BPSP(car_sequence)
    
    @property
    def car_pos(self):
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
            # Convert the car to int type (assuming it's of type np.int64 or similar)
            car_key = int(car)

            # If the car ID is already in the dictionary, append the new position.
            if car_key in car_pos:
                car_pos[car_key].append(idx)
            # If this is the first occurrence of the car ID, initialize a list with the position.
            else:
                car_pos[car_key] = [idx]
        
        # Convert the lists of positions to tuples for a consistent output format.
        for car_key, positions in car_pos.items():
            car_pos[car_key] = tuple(positions)

        return car_pos

    
    @property
    def graph(self):
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
    
    @property
    def docplex_bpsp_model(self):
        """
        Construct a CPLEX model for the Binary Paint Shop Problem (BPSP).

        This method encodes the BPSP into a linear optimization problem.
        The BPSP seeks to determine a painting sequence for cars such that adjacent cars
        in the sequence don't receive the same color if they are the same car type.
        The sequence length is twice the number of cars as each car passes the painting station twice.

        The decision variables represent the paint color (binary: 0 or 1) for each car at each position in the sequence.
        The objective function aims to minimize the absolute difference between adjacent paint values in the sequence.

        Attributes
        ----------
        car_sequence : list
            A list representing the sequence of cars as they pass the paint shop.
            Each car is represented by its identifier (e.g., integer), and appears twice in the sequence.
        
        car_positions : dict
            A dictionary mapping each car to its two positions in the car_sequence.
            For example, if car 'a' appears in positions 2 and 5 in car_sequence, then
            car_positions['a'] = [2, 5].

        Returns
        -------
        Model
            A CPLEX model instance representing the BPSP, ready to be solved.
        """
        
        mdl = Model("BPSP_Problem")
        sequence = self.car_sequence
        car_pos = self.car_positions

        # Dictionary for storing the paint value for car 'x' at position 'j'.
        w_vars = {f"{w}_{i}": mdl.binary_var(name=f"w_{w}_{i}") for i, w in enumerate(sequence)}

        # This loop adds the constraint that a particular car cannot have the same paint
        # at both occurrences in the paint sequence. If the first occurrence is 0, the other has to 
        # be 1 and vice-versa.
        for car, positions in car_pos.items():
            w_key1, w_key2 = f"{car}_{positions[0]}", f"{car}_{positions[1]}"
            mdl.add_constraint(w_vars[w_key1] + w_vars[w_key2] == 1)

        # Encode the objective function: sum_i^2n-1 |paint[i] - paint[i+1]|. Since docplex accepts abs operator, 
        # you can directly utilize it for the objective. This makes the model simpler than in Gurobi.
        w_keys = list(w_vars.keys())
        objective_function = mdl.sum(mdl.abs(w_vars[w_keys[i]] - w_vars[w_keys[i + 1]]) for i in range(len(w_keys) - 1))
        
        # Set the objective to minimize.
        mdl.minimize(objective_function)

        return mdl

    @property
    def qubo(self):
        """
        Returns the QUBO encoding of the Binary Paint Shop Problem.

        This function provides a QUBO representation of the BPSP, where the objective is to minimize the
        total weight of violations. The violations here refer to two adjacent cars in the sequence receiving 
        the same color.

        Returns
        -------
        QUBO
            The QUBO encoding of the BPSP. This is a representation that describes 
            the quadratic objective function over binary variables.
        """
        
        # Iterate over edges (with weight) and store accordingly
        terms = []
        weights = []

        # Here, we use self.bpsp_graph to gather terms and weights.
        for u, v, edge_weight in self.bpsp_graph.edges(data="weight"):
            terms.append([u, v])

            weights.append(edge_weight)  

        return QUBO(self.bpsp_graph.number_of_nodes(), terms, weights, self.problem_instance,)

    def cplex_solution(self):
        """
        Solves the BPSP using the CPLEX solver and returns the solution and its objective value.

        The Binary Paintshop Problem (BPSP) solution represents a sequence of paint choices 
        where a value of 0 represents blue paint and a value of 1 represents red paint.

        Returns
        -------
        tuple
            A tuple where the first element is a list containing the paint choices 
            (0 for blue, 1 for red) and the second element is the objective value.
        """
        
        # Retrieve the model for BPSP
        model = self.docplex_bpsp_model
        
        # Solve the BPSP using CPLEX
        model.solve()
        
        # Check the status of the solution
        status = model.solve_details.status
        if status != "integer optimal solution":
            print(status)
        
        # Extract the solution values representing paint choices
        solution = [int(np.round(model.solution.get_value(var))) for var in model.iter_binary_vars()]
        
        # Get the objective value of the solution
        objective_value = model.objective_value
        # Return the paint choices and their corresponding objective value
        return solution, objective_value

    def qaoa_solution(self, bitstring):
        """
        Transforms a sequence of initial car colors to a paint sequence and computes the number of paint swaps.
        
        Given a bitstring sequence of car colors ('0' or '1'), this function transforms it into 
        a paint sequence. Each car's colors are determined by two consecutive positions in 
        the sequence. The function also computes the number of times the paint changes (swaps) 
        between consecutive positions in the paint sequence.
        
        Parameters
        ----------
        colors : str
            A string of car colors where each character is either '0' or '1'.
        
        Returns
        -------
        tuple
            A tuple where the first element is a list representing the paint sequence 
            (0 for blue, 1 for red) and the second element is the number of paint swaps.
        """
        
        # Convert the input string colors to a list of integers
        colors = [int(color) for color in bitstring]
        
        # Initialize the paint sequence with zeros
        paint_sequence = [0 for _ in range(2 * len(colors))]
        
        # Fill the paint sequence based on the input colors and car positions
        for car, color in enumerate(colors):
            pos1, pos2 = self.car_positions[car]
            
            paint_sequence[pos1] = color
            paint_sequence[pos2] = 1 - color  # The opposite color
        
        # Compute the number of paint swaps in the sequence
        color_swaps = sum(abs(paint_sequence[i] - paint_sequence[i + 1]) for i in range(len(paint_sequence) - 1))
        
        return paint_sequence, color_swaps

    def redfirst_solution(self):
        """
        The `red_first_solution` method applies a heuristic to generate a paint sequence for cars. 
        Specifically, it colors the first occurrence of each car as Red (1) and the second 
        occurrence as Blue (0). On average, this heuristic may not be as efficient as the greedy 
        algorithm.

        Attributes:
        ----------
        self.car_sequence : list[int]
            A list containing the sequence of cars that need to be painted.

        Returns:
        -------
        tuple
            A tuple containing two elements:
            1. A list representing the paint sequence with '1' indicating Red and '0' indicating Blue.
            2. An integer representing the total number of paint swaps in the sequence.
        """
        
        # Dictionary to keep track of whether a car has been painted or not
        cars_painted = defaultdict(bool)
        
        # Create the paint sequence
        paint_sequence = []
        for car in self.car_sequence:
            if not cars_painted[car]:
                paint_sequence.append(1)
                cars_painted[car] = True
            else:
                paint_sequence.append(0)

        # Compute the number of color swaps in the sequence
        color_swaps = sum(abs(paint_sequence[i] - paint_sequence[i + 1]) for i in range(len(self.car_sequence) - 1))
            
        return paint_sequence, color_swaps

    def greedy_solution(self):
        """
        The `greedy_solution` method determines a feasible paint sequence for cars using a 
        greedy approach. It processes the car sequence from left to right, coloring the 
        first occurrence of each car based on its predecessor and the second occurrence 
        with the opposite color.

        Attributes:
        ----------
        self.car_sequence : list[int]
            A list containing the sequence of cars that need to be painted.
        self.car_positions : dict[int, tuple]
            A dictionary mapping each car to a tuple containing the
            first and second position of the car's occurrence in `car_sequence`.

        Returns:
        -------
        tuple
            A tuple containing two elements:
            1. A list representing the paint sequence with '1' indicating Red and '0' indicating Blue.
            2. An integer representing the total number of paint swaps in the sequence.
        """
        
        # Dictionary to keep track of whether a car has been painted or not
        cars_painted = defaultdict(bool)
        
        # List to store the paint sequence
        paint_sequence = []
        
        # Variable to keep track of the last used color
        last_color = 0
        
        # Create the paint sequence
        for car in self.car_sequence:
            if not cars_painted[car]:
                if paint_sequence:  # if paint_sequence is not empty
                    last_color = paint_sequence[-1]
                paint_sequence.append(last_color)
                cars_painted[car] = True
            else:
                paint_sequence.append(1 - paint_sequence[self.car_positions[car][0]])

        # Compute the number of color swaps in the sequence
        color_swaps = sum(abs(paint_sequence[i] - paint_sequence[i + 1]) for i in range(len(self.car_sequence) - 1))
            
        return paint_sequence, color_swaps

    def plot_paint_sequence(self, paint_sequence, ax=None):
        """
        Plot a bar chart showing the colors assigned to cars based on the given paint_sequence.

        Parameters:
        ----------
        self.car_sequence : numpy.ndarray[int]
            Numpy array containing the order of cars to be painted.
        
        paint_sequence : list[int] or numpy.ndarray[int]
            List or numpy array containing 0 or 1 for each car in `self.car_sequence`. 
            A 0 indicates the car is painted Blue, while a 1 indicates it's painted Red.

        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If not provided, a new figure is created.
        
        Returns:
        -------
        None

        Note:
        -----
        This function uses a blue-red color mapping and plots bars without any gaps to visually 
        represent the paint sequence of cars. The plot's size is dynamically adjusted based on 
        the number of cars.
        """
        
        # Define color mapping for 0s and 1s in the paint_sequence to 'blue' and 'red' respectively
        color_map = {0: '#48cae4', 1: '#f25c54'}  # shades of blue and red
        plot_colors = [color_map[color] for color in paint_sequence]

        # Dynamically determine the width of the figure based on the number of cars in self.car_sequence
        fig_width = self.car_sequence.size * 0.8  # This provides each bar approximately 0.8 inch width

        # If no ax (subplot) is provided, create a new figure with the determined width
        if ax is None:
            fig, ax = plt.subplots(figsize=(fig_width, 2))
        
        # Plot bars for each car, colored based on the paint_sequence
        ax.bar(range(len(self.car_sequence)), np.ones_like(self.car_sequence), color=plot_colors, width=1, align='center')
        ax.set_xlim(-0.5, len(self.car_sequence) - 0.5)  # Set x limits to tightly fit bars
        ax.set_ylim(0, 1)  # Set y limits from 0 to 1 as the bars have a fixed height of 1
        ax.yaxis.set_visible(False)  # Hide the y-axis as it's not relevant
        
        # Set x-ticks to indicate car numbers and label them as "Car 3", "Car 2", "Car 2", etc.
        ax.set_xticks(range(len(self.car_sequence)))
        ax.set_xticklabels([f"Car {int(car)}" for car in self.car_sequence])

        # Hide the top, right, and left spines for cleaner visuals
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # If no ax is provided, show the plot directly
        if ax is None:
            plt.tight_layout()
            plt.show()