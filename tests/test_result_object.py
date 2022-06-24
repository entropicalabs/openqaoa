import unittest
import networkx as nx

from openqaoa.problems.problem import MinimumVertexCover
from openqaoa.workflows.optimizer import QAOA  

class TestingLoggerClass(unittest.TestCase):

    def test_attribute_existence(self):
        """
        This test checks that 'solution', 'evals', 'intermediate', 'optimized' are correctly
        created by the default workflow
        """

        # Create the problem
        g = nx.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_pubo_problem()
        
        q = QAOA()
        q.compile(vc, verbose=False)
        q.optimize()

        for k in ['most_probable_states', 'evals', 'intermediate', 'optimized']:
            if k in [a for a in dir(q.results) if not a.startswith('__')]:
                pass
            else:
                raise ValueError()

if __name__ == '__main__':
    unittest.main()