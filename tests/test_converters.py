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

import unittest

from openqaoa.problems.problem import QUBO, MaximumCut
from openqaoa.problems.converters import FromDocplex2IsingModel
from docplex.mp.model import Model
import networkx as nx


class TestDocplex2IsingClass(unittest.TestCase):
    
    """
    Test the converter from docplex models
    
    """

    def test_qubo(self): 
        """
        Test the QUBO class is generated from the function FromDocplex2IsingModel

        """
        # Creating a basic docplex model
        mdl = Model("Test") # Docplex model
        num_z = 5 # Number of variables
        z = mdl.binary_var_list(num_z, name="Z") # docplex variables
        objective = mdl.sum(z) - 2 * z[0]  + z[3] * z[4] + 5 # objective function

        mdl.minimize(objective) # Optimization

        ising_problem = FromDocplex2IsingModel(mdl).ising_model # Ising model of the Docplex model
        
        self.assertIsInstance(ising_problem, QUBO)
        
    def test_model_maxcut(self):
        """
        Test the Maxcut application of OpenQAOA gives the same result as the 
        Docplex translation model.
        """
        
        # Graph representing the maxcut problem
        n_nodes = 5 # number of nodes
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from([[0,1], [0,2], [1,2], [4,3], [3,2], [4,0], [2,4]])
        
        # Docplex model 
        mdl = Model(name="Max-cut")
        x = mdl.binary_var_list(n_nodes, name="x")
        for w, v in G.edges:
            G.edges[w, v].setdefault("weight", 1)
        objective = mdl.sum(G.edges[i, j]["weight"] * x[i] * (1 - x[j]) + G.edges[i, j]["weight"] * x[j] * (1 - x[i]) for i, j in G.edges)
        mdl.maximize(objective)
        
        # Translating the problem to OQ ising Model
        ModelOQ = FromDocplex2IsingModel(mdl)
        Ising_model_OQ = ModelOQ.ising_model.asdict()
        # Using the predefine function of this problem
        IsingModelDirect = MaximumCut(G).get_qubo_problem().asdict()
        # Comparing both results in this specific case MaxCut from OpenQAOa gives 
        # the coefficients omultiplied by two of the DocplexToIsingModel
        for nn, term in enumerate(IsingModelDirect["terms"]):
            idx_OQ = Ising_model_OQ["terms"].index(term)
            self.assertAlmostEqual(2 * Ising_model_OQ["weights"][idx_OQ], IsingModelDirect["weights"][nn])


if __name__ == '__main__':
    unittest.main()
