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
from openqaoa.workflows.optimizer import QAOA
from openqaoa.problems.helper_functions import docplex2qubo
from docplex.mp.model import Model
import networkx as nx




"""
Unittest based testing of docplex model conversion.
"""


class TestDocplex2IsingClass(unittest.TestCase):

    def test_qubo(self):        
        mdl = Model("Test") # Docplex model
        num_z = 5 # Number of variables
        z = mdl.binary_var_list(num_z, name="Z")


        objective = mdl.sum(z) - 2 * z[0]  + z[3] * z[4] + 5

        mdl.minimize(objective)

        ising_problem = docplex2qubo(mdl)
        
        assert isinstance(QUBO, ising_problem)
        
    def test_model_maxcut(self):
        n_nodes = 5 # number of nodes
        # Graph representing the maxcut problem
        G = nx.Graph()
        G.add_nodes_from(n_nodes)
        G.add_edges_from([[0,1], [0,2], [1,2], [4,3], [3,2], [4,0], [2,4]])
        # Docplex model 
        mdl = Model(name="Max-cut")
        x = mdl.binary_var_list(n_nodes, "x")
        for w, v in G.edges:
            G.edges[w, v].setdefault("weight", 1)
        objective = mdl.sum(G.edges[i, j]["weight"] * x[i] * (1 - x[j]) + G.edges[i, j]["weight"] * x[j] * (1 - x[i]) for i, j in G.edges)
        mdl.maximize(objective)
        
        mdl = Model("MaxCut") # Docplex model
        num_z = 5 # Number of variables
        z = mdl.binary_var_list(num_z, name="Z")


        objective = mdl.sum(z) - 2 * z[0]  + z[3] * z[4] + 5

        mdl.minimize(objective)

        qubo_problem = docplex2qubo(mdl)
        
        q = QAOA()
        q.compile(qubo_problem)
        q.optimize() 
        probs = q.results_information["best probability"][0]
        
        assert max(probs, key=probs.get) == '10000'

if __name__ == '__main__':
    unittest.main()
