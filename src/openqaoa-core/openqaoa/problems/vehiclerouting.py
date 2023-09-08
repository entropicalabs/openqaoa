import networkx as nx
from docplex.mp.model import Model
from typing import Union
import numpy as np
import itertools
import copy
import matplotlib.pyplot as plt
from matplotlib import colormaps
import warnings

from .problem import Problem
from .converters import FromDocplex2IsingModel
from .qubo import QUBO


class VRP(Problem):
    """
    Creates an instance of the vehicle routing problem (VRP).
    https://en.wikipedia.org/wiki/Vehicle_routing_problem

    Parameters
    ----------
    G: networkx.Graph
        Networkx graph of the problem
    n_vehicles: int
        the number of vehicles used in the solution
    pos: list[list]
            position x, y of each node
    depot: int
        the node where all the vehicles leave for and return after
    subtours: list[list]
        if -1 (Default value): All the possible subtours are added to the constraints. Avoid it for large instances.
        if there are subtours that want be avoided in the solution, e.g, a 8 nodes
        VRP with an optimal solution showing subtour between nodes 4, 5, and 8 can be
        avoided introducing the constraint subtours=[[4,5,8]]. To additional information
        about subtours refer to https://de.wikipedia.org/wiki/Datei:TSP_short_cycles.png
    all_subtours:
    penalty: float
        Penalty for the constraints. If the method is 'unbalanced' three values are needed,
        one for the equality constraints and two for the inequality constraints.
    method: str
        Two available methods for the inequality constraints ["slack", "unbalanced"]
        For 'unblanced' see https://arxiv.org/abs/2211.13914
    Returns
    -------
        An instance of the VRP problem.
    """

    __name__ = "vehicle_routing"

    def __init__(
        self,
        G,
        n_vehicles,
        pos: list = None,
        depot: int = 0,
        subtours: list = -1,
        method: str = "slack",
        penalty: Union[int, float, list] = 4,
    ):
        self.G = G
        if (len(pos) != 0) and (len(G.nodes) != len(pos)):
            raise ValueError(
                f"The number of nodes in G is {len(G.nodes)} while the x, y coordinates in pos is {len(pos)}"
            )
        self.n_vehicles = n_vehicles
        self.pos = [] if pos is None else pos
        self.depot = depot
        self.subtours = subtours
        self.method = method
        if method == "unbalanced" and len(penalty) != 3:
            raise ValueError(
                "The penalty must have 3 parameters [lambda_0, lambda_1, lambda_2]"
            )
        else:
            self.penalty = penalty

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, input_networkx_graph):
        if not isinstance(input_networkx_graph, nx.Graph):
            raise TypeError("Input problem graph must be a networkx Graph.")

        # Relabel nodes to integers starting from 0
        mapping = dict(
            zip(input_networkx_graph, range(input_networkx_graph.number_of_nodes()))
        )
        self._G = nx.relabel_nodes(input_networkx_graph, mapping)

    @staticmethod
    def from_distance_matrix(**kwargs):
        """
        Creates a instance of the vehicle routing problem, from the distance
        matrix.

        Parameters
        ----------
        **kwargs:
            Required keyword arguments are:
            matrix: list, array
                The number of nodes (vertices) in the graph.
            n_vehicles: int
                Number of vehicles used in the problem
            method: str
                method for the inequality constraints ['slack', 'unbalanced'].
                For the unbalaced method refer to https://arxiv.org/abs/2211.13914
                For the slack method: https://en.wikipedia.org/wiki/Slack_variable
            subtours: list[list]
                Manually add the subtours to be avoided
            seed: int
                Seed for the random problems.

        Returns
        -------
            A instance of the vehicle routing problem.
        """
        matrix = kwargs.get("matrix")
        n_nodes = len(matrix)
        n_vehicles = kwargs.get("n_vehicles", 2)
        pos = kwargs.get("pos", [])
        method = kwargs.get("method", "slack")
        if method == "slack":
            penalty = kwargs.get("penalty", 4)
        elif method == "unbalanced":
            penalty = kwargs.get("penalty", [4, 1, 1])
        else:
            raise ValueError(f"The method '{method}' is not valid.")
        subtours = kwargs.get("subtours", -1)
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                G.add_weighted_edges_from([(i, j, matrix[i][j])])
        return VRP(
            G, n_vehicles, pos=pos, subtours=subtours, method=method, penalty=penalty
        )

    @staticmethod
    def from_coordinates(**kwargs):
        """
        Creates a instance of the vehicle routing problem, from the cartesian
        coordinates.

        Parameters
        ----------
        **kwargs:
            Required keyword arguments are:
            pos: list, array
                The nodes x, y (vertices) in the graph.
            n_vehicles: int
                Number of vehicles used in the problem
            method: str
                method for the inequality constraints ['slack', 'unbalanced'].
                For the unbalaced method refer to https://arxiv.org/abs/2211.13914
                For the slack method: https://en.wikipedia.org/wiki/Slack_variable
            subtours: list[list]
                Manually add the subtours to be avoided
            seed: int
                Seed for the random problems.

        Returns
        -------
            A instance of the vehicle routing problem.
        """
        pos = kwargs.get("pos")
        n_nodes = len(pos)
        n_vehicles = kwargs.get("n_vehicles", 2)
        method = kwargs.get("method", "slack")
        if method == "slack":
            penalty = kwargs.get("penalty", 4)
        elif method == "unbalanced":
            penalty = kwargs.get("penalty", [4, 1, 1])
        else:
            raise ValueError(f"The method '{method}' is not valid.")
        subtours = kwargs.get("subtours", -1)
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                r = np.sqrt((pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2)
                G.add_weighted_edges_from([(i, j, r)])
        return VRP(
            G, n_vehicles, pos=pos, subtours=subtours, method=method, penalty=penalty
        )

    @staticmethod
    def random_instance(**kwargs):
        """
        Creates a random instance of the vehicle routing problem, whose graph is
        random.

        Parameters
        ----------
        **kwargs:
            Required keyword arguments are:
            n_nodes: int
                The number of nodes (vertices) in the graph.
            n_vehicles: int
                Number of vehicles used in the problem
            method: str
                method for the inequality constraints ['slack', 'unbalanced'].
                For the unbalaced method refer to https://arxiv.org/abs/2211.13914
                For the slack method: https://en.wikipedia.org/wiki/Slack_variable
            subtours: list[list]
                Manually add the subtours to be avoided
            seed: int
                Seed for the random problems.

        Returns
        -------
            A random instance of the vehicle routing problem.
        """
        n_nodes = kwargs.get("n_nodes", 6)
        n_vehicles = kwargs.get("n_vehicles", 2)
        seed = kwargs.get("seed", None)
        method = kwargs.get("method", "slack")
        if method == "slack":
            penalty = kwargs.get("penalty", 4)
        elif method == "unbalanced":
            penalty = kwargs.get("penalty", [4, 1, 1])
        else:
            raise ValueError(f"The method '{method}' is not valid.")
        subtours = kwargs.get("subtours", -1)
        rng = np.random.default_rng(seed)
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        pos = [[0, 0]]
        pos += [list(2 * rng.random(2) - 1) for _ in range(n_nodes - 1)]
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                r = np.sqrt((pos[i][0] - pos[j][0]) ** 2 + (pos[i][1] - pos[j][1]) ** 2)
                G.add_weighted_edges_from([(i, j, r)])
        return VRP(
            G, n_vehicles, pos, subtours=subtours, method=method, penalty=penalty
        )

    @property
    def docplex_model(self):
        """
        Return a docplex model of the vehicle routing problem using the linear
        programming approach. In this approach, the edges between different nodes
        of the graph are the variables of the problem. Therefore, x_i_j will represent
        if the path between nodes i and j is selected. The idea is to minimize the distance
        traveled

        sum_{i, j > i} r_{ij} x_{ij}.

        where r_{ij} is the distance between nodes i and j (self.G.edges[i, j]["weight"]).

        Returns
        -------
        mdl : Model
            Docplex model of the Vehicle routing problem.

        """
        mdl = Model("VRP")
        num_nodes = self.G.number_of_nodes()
        # Variables: the edges between nodes for a symmetric problem. This means x_i_j == x_j_i and only x_i_j is used.
        x = {
            (i, j): mdl.binary_var(name=f"x_{i}_{j}")
            for i in range(num_nodes - 1)
            for j in range(i + 1, num_nodes)
        }
        # Distance traveled
        mdl.minimize(
            mdl.sum(
                self.G.edges[i, j]["weight"] * x[(i, j)]
                for i in range(num_nodes - 1)
                for j in range(i + 1, num_nodes)
            )
        )
        # Only 2 edges for node constraint
        for i in range(num_nodes):
            if i != self.depot:
                mdl.add_constraint(
                    mdl.sum(
                        [x[tuple(sorted([i, j]))] for j in range(num_nodes) if i != j]
                    )
                    == 2
                )
            else:
                mdl.add_constraint(
                    mdl.sum(
                        [x[tuple(sorted([i, j]))] for j in range(num_nodes) if i != j]
                    )
                    == 2 * self.n_vehicles
                )

        if self.subtours == -1:
            eq_const = mdl.number_of_constraints
            list_subtours = [[i for i in range(num_nodes) if i != self.depot]]
            for nodes in list_subtours:  # costly for large instances
                for i in range(3, num_nodes - 2 * self.n_vehicles):
                    for subtour in itertools.combinations(nodes, i):
                        tour = sorted(subtour)
                        n_subtour = len(subtour)
                        mdl.add_constraint(
                            mdl.sum(
                                [
                                    x[(tour[i], tour[j])]
                                    for i in range(n_subtour - 1)
                                    for j in range(i + 1, n_subtour)
                                ]
                            )
                            <= n_subtour - 1
                        )

            if num_nodes >= 10:
                warnings.warn(
                    f"All the possible subtour constraints are added. The number of inequality constraints is {mdl.number_of_constraints - eq_const}. Consider reducing the number of subtour constraints."
                )
        # Subtour constraints if any
        elif isinstance(self.subtours, list):
            list_subtours = self.subtours
            for subtour in list_subtours:
                tour = sorted(subtour)
                n_subtour = len(subtour)
                if n_subtour != 0:
                    mdl.add_constraint(
                        mdl.sum(
                            [
                                x[(tour[i], tour[j])]
                                for i in range(n_subtour)
                                for j in range(i + 1, n_subtour)
                            ]
                        )
                        <= n_subtour - 1
                    )
        else:
            raise ValueError(
                f"{type(self.subtour)} is not a valid format for the subtours."
            )
        return mdl

    @property
    def qubo(self):
        """
        Returns the QUBO encoding of this problem.

        Returns
        -------
            The QUBO encoding of this problem.
        """
        cplex_model = self.docplex_model
        if self.method == "slack":
            qubo_docplex = FromDocplex2IsingModel(
                cplex_model, multipliers=self.penalty
            ).ising_model
        elif self.method == "unbalanced":
            qubo_docplex = FromDocplex2IsingModel(
                cplex_model,
                multipliers=self.penalty[0],
                unbalanced_const=True,
                strength_ineq=self.penalty[1:],
            ).ising_model
        return QUBO(
            qubo_docplex.n,
            qubo_docplex.terms + [[]],
            qubo_docplex.weights + [qubo_docplex.constant],
            self.problem_instance,
        )

    def classical_solution(self, string: bool = False):
        """
        Return the classical solution of the vehicle routing problem
        Parameters
        ----------
        string : bool, optional
            If the solution is returned as a string. The default is False.
        Raises
        ------
        ValueError
            A flag if docplex does not find a valid solution.
        Returns
        -------
        solution : Union[str, dict]
            The classical solution of the specific problem as a string or a dict.
        """
        cplex_model = self.docplex_model
        docplex_sol = cplex_model.solve()

        if docplex_sol is None:
            raise ValueError(f"Solution not found: {cplex_model.solve_details.status}.")

        if string:
            solution = ""
        else:
            solution = {}
        for var in cplex_model.iter_binary_vars():
            if string:
                solution += str(round(docplex_sol.get_value(var)))
            else:
                solution[var.name] = round(docplex_sol.get_value(var))
        return solution

    def paths_subtours(self, sol):
        """
        Return the routes the different vehicles take and the subtours the solution
        has if any
        Parameters
        ----------
        sol : dict
            Solution to the vrp with x_i_j the edge between nodes i and j.

        Returns
        -------
        solution: dict
        Dictionary with two keys:
        - 'paths': solutions of the VRP,
        - 'subtours': subtours found in the solution
        """
        n_nodes = self.G.number_of_nodes()
        vars_list = []
        for i in range(n_nodes - 1):
            for j in range(i + 1, n_nodes):
                if round(sol[f"x_{i}_{j}"]):
                    vars_list.append([i, j])
        # ----------------  vehicle routing problem solutions -----------------
        paths = {}
        for i in range(self.n_vehicles):
            paths[i] = []
            nodes = [self.depot]
            count_depot = 0
            max_edges = n_nodes * (n_nodes - 1) / 2
            while count_depot < 2:
                for edge in vars_list:
                    if nodes[-1] in edge:
                        if self.depot in edge:
                            count_depot += 1
                        paths[i].append(edge)
                        vars_list.remove(edge)
                        nodes.append(edge[0] if edge[0] != nodes[-1] else edge[1])
                        break
                max_edges -= 1
                if max_edges < 0:
                    raise ValueError(
                        "Solution provided does not fulfill all the path conditions."
                    )
        # ----------------            subtours                -----------------
        subtours = {}
        i = 0
        max_edges = n_nodes * (n_nodes - 1) / 2
        while len(vars_list) > 0:
            subtours[i] = [vars_list.pop(0)]
            nodes = copy.copy(subtours[i][0])
            count = 1
            while count < 2:
                for edge in vars_list:
                    if nodes[-1] in edge:
                        if nodes[0] in edge:
                            count += 1
                        subtours[i].append(edge)
                        vars_list.remove(edge)
                        nodes.append(edge[0] if edge[0] != nodes[-1] else edge[1])
                        break
                max_edges -= 1
                if max_edges < 0:
                    raise ValueError("The subtours in the solution are broken.")
            i += 1
        return {"paths": paths, "subtours": subtours}

    def plot_solution(
        self,
        solution: Union[dict, str],
        ax=None,
        edge_width: float = 4,
        colors: list = None,
    ):
        """
        A visualization method for the vehicle routing problem solution.
        Parameters
        ----------
        solution : Union[dict, str]
            The solution of the specific vehicle routing problem as a string or dictionary.
        ax : matplotlib axes, optional
            The default is None.
        edge_width: float, optional
            Edges width, default is 4
        colors: list
            List of colors of the different vehicles' route. The default is None
        Returns
        -------
        fig : matplotlib.pyplot.Figure()
            The graph visualization of the solution.
        """
        pos = self.pos
        if pos is []:
            raise ValueError(
                "There are no x, y coordinates for any of the nodes in the problem. You need to provide the coordinates of the nodes through the initialization parameter, pos."
            )
        colors = colormaps["jet"] if colors is None else colors
        if type(colors) is list and len(colors) != self.n_vehicles:
            raise ValueError(
                f"The length of colors {len(colors)} and the number of vehicles {self.n_vehicles} do not match"
            )

        if isinstance(solution, str):
            sol = {}
            for n, var in enumerate(self.docplex_model.iter_binary_vars()):
                sol[var.name] = int(solution[n])
            solution = sol
        paths_and_subtours = self.paths_subtours(solution)
        paths = paths_and_subtours["paths"]
        subtours = paths_and_subtours["subtours"]
        tours_color = {}
        for vehicle in range(self.n_vehicles):
            for i, j in paths[vehicle]:
                if type(colors) is list:
                    tours_color[f"x_{i}_{j}"] = colors[vehicle]
                else:
                    tours_color[f"x_{i}_{j}"] = colors((vehicle + 1) / self.n_vehicles)
        for subtour in subtours.keys():
            for i, j in subtours[subtour]:
                tours_color[f"x_{i}_{j}"] = "black"
        color_node = "#5EB1EB"
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        num_vertices = self.G.number_of_nodes()
        G = nx.Graph()
        G.add_nodes_from(range(num_vertices))
        edge_color = []
        for i in range(num_vertices - 1):
            for j in range(i + 1, num_vertices):
                if int(solution[f"x_{i}_{j}"]):
                    edge_color.append(tours_color[f"x_{i}_{j}"])
                    G.add_edge(i, j)
        nx.draw(
            G,
            pos=pos,
            width=edge_width,
            edge_color=edge_color,
            node_color=color_node,
            alpha=0.8,
            labels={i: str(i) for i in range(num_vertices)},
            ax=ax,
            edgecolors="black",
        )
        return fig
