import unittest
from docplex.mp.model import Model
import networkx as nx

from openqaoa.problems import QUBO, MaximumCut, FromDocplex2IsingModel


class TestDocplex2IsingClass(unittest.TestCase):

    """
    Test the converter from docplex models

    """

    def test_qubo(self):
        """
        Test the QUBO class is generated from the function FromDocplex2IsingModel

        """
        # Creating a basic docplex model
        mdl = Model("Test")  # Docplex model
        num_z = 5  # Number of variables
        z = mdl.binary_var_list(num_z, name="Z")  # docplex variables
        objective = mdl.sum(z) - 2 * z[0] + z[3] * z[4] + 5  # objective function

        mdl.minimize(objective)  # Optimization

        ising_problem = FromDocplex2IsingModel(
            mdl
        ).ising_model  # Ising model of the Docplex model

        self.assertIsInstance(ising_problem, QUBO)

    def test_slack_penalization(self):
        """
        Test the equality and inequality constraints are encoded in the QUBO
        model using slack variables for the inequality constraints approach.

        """
        weights = [
            18.25,
            -4.0,
            -8.0,
            -4.0,
            -6.0,
            -12.0,
            -6.0,
            4.0,
            2.0,
            4.0,
            2.0,
            2.0,
            4.0,
            4.0,
            2.0,
            2.25,
            1.25,
            -2.0,
            -4.0,
            -2.0,
            -2.0,
            -2.0,
        ]
        # Creating a basic docplex model
        mdl = Model("Test inequal")  # Docplex model
        num_z = 2  # Number of variables
        z = mdl.binary_var_list(num_z, name="Z")  # docplex variables
        objective = mdl.sum(z) - 2 * z[0] + z[1] * z[0] + 5  # objective function
        # Adding constraints
        mdl.add_constraint(mdl.sum(z[i] for i in range(num_z)) == 1)
        mdl.add_constraint(2 * z[0] + 3 * z[1] >= 1)
        mdl.add_constraint(2 * z[1] + z[0] <= 2)
        mdl.minimize(objective)  # Optimization

        ising_problem = FromDocplex2IsingModel(
            mdl
        ).ising_model  # Ising model of the Docplex model

        self.assertIsInstance(ising_problem, QUBO)
        self.assertEqual(ising_problem.weights, weights)

    def test_unbalanced_penalizations(self):
        """
        Test the equality and inequality constraints are encoded in the QUBO
        model using the unblanaced penalization method.

        """
        weights = [4.25, -0.95, -2.45]
        # Creating a basic docplex model
        mdl = Model("Test inequal")  # Docplex model
        num_z = 2  # Number of variables
        z = mdl.binary_var_list(num_z, name="Z")  # docplex variables
        objective = mdl.sum(z) - 2 * z[0] + z[1] * z[0] + 5  # objective function
        # Adding constraints
        mdl.add_constraint(mdl.sum(z[i] for i in range(num_z)) == 1)
        mdl.add_constraint(2 * z[0] + 3 * z[1] >= 1)
        mdl.add_constraint(2 * z[1] + z[0] <= 2)
        mdl.minimize(objective)  # Optimization

        ising_problem = FromDocplex2IsingModel(
            mdl, unbalanced_const=True
        ).ising_model  # Ising model of the Docplex model

        self.assertIsInstance(ising_problem, QUBO)

        for weight_1, weight_2 in zip(ising_problem.weights, weights):
            self.assertAlmostEqual(weight_1, weight_2)

    def test_model_maxcut(self):
        """
        Test the Maxcut application of OpenQAOA gives the same result as the
        Docplex translation model.
        """

        # Graph representing the maxcut problem
        n_nodes = 5  # number of nodes
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from([[0, 1], [0, 2], [1, 2], [4, 3], [3, 2], [4, 0], [2, 4]])

        # Docplex model
        mdl = Model(name="Max-cut")
        x = mdl.binary_var_list(n_nodes, name="x")
        for w, v in G.edges:
            G.edges[w, v].setdefault("weight", 1)
        objective = mdl.sum(
            G.edges[i, j]["weight"] * x[i] * (1 - x[j])
            + G.edges[i, j]["weight"] * x[j] * (1 - x[i])
            for i, j in G.edges
        )
        mdl.maximize(objective)

        # Translating the problem to OQ ising Model
        ModelOQ = FromDocplex2IsingModel(mdl)
        Ising_model_OQ = ModelOQ.ising_model.asdict()
        # Using the predefine function of this problem
        IsingModelDirect = MaximumCut(G).qubo.asdict()
        # Comparing both results in this specific case MaxCut from OpenQAOa gives
        # the coefficients omultiplied by two of the DocplexToIsingModel
        for nn, term in enumerate(IsingModelDirect["terms"]):
            idx_OQ = Ising_model_OQ["terms"].index(term)
            self.assertAlmostEqual(
                2 * Ising_model_OQ["weights"][idx_OQ], IsingModelDirect["weights"][nn]
            )


if __name__ == "__main__":
    unittest.main()
