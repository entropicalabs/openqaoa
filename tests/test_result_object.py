import unittest
import networkx as nx
import numpy as np

from openqaoa.problems.problem import MinimumVertexCover
from openqaoa.qaoa_parameters import PauliOp, Hamiltonian
from openqaoa.optimizers.result import Result
from openqaoa.optimizers.result import most_probable_bitstring
from openqaoa.utilities import qaoa_probabilities
from openqaoa.workflows.optimizer import QAOA  

class TestingLoggerClass(unittest.TestCase):

    def test_attribute_existence(self):
        """
        This test checks that 'most_probable_states', 'evals', 'intermediate', 'optimized' are correctly
        created by the default workflow
        """

        # Create the problem
        g = nx.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()
        
        q = QAOA()
        q.compile(vc, verbose=False)
        q.optimize()

        assert set(['most_probable_states', 'evals', 'intermediate', 'optimized']) <= set([a for a in dir(q.results) if not a.startswith('__')])


    def test_most_probable_bitstring(self):

        cost_hamil = Hamiltonian([PauliOp('ZZ',(0, 3)),
                         PauliOp('ZZ',(0, 1)),
                         PauliOp('ZZ',(1, 2)),
                         PauliOp('ZZ',(2, 3)),
                         PauliOp('Z',(0,)),
                         PauliOp('Z',(1,)),
                         PauliOp('Z',(2,)),
                         PauliOp('Z',(3,))], [2.5, 2.5, 2.5, 2.5, 4.5, 4.5, 4.5, 4.5], 12)

        # From a statevector backend
        optimized_measurement_outcomes_sv = np.array([ 0.18256422-0.1296918j , -0.03786405-0.13158363j,
                                                -0.03786405-0.13158363j, -0.01910622+0.3151214j ,
                                                -0.03786405-0.13158363j,  0.42313283+0.15529604j,
                                                -0.01910622+0.3151214j ,  0.016895  +0.05117122j,
                                                -0.03786405-0.13158363j, -0.01910622+0.3151214j ,
                                                 0.42313283+0.15529604j,  0.016895  +0.05117122j,
                                                -0.01910622+0.3151214j ,  0.016895  +0.05117122j,
                                                 0.016895  +0.05117122j,  0.23799586+0.04022615j])

        # from a shot simulator
        optimized_measurement_outcomes_shot = {'0010': 1,
                                            '0110': 2,
                                            '1000': 1,
                                            '1001': 13,
                                            '0000': 1,
                                            '1100': 10,
                                            '1110': 201,
                                            '1111': 37,
                                            '0101': 225,
                                            '1101': 57,
                                            '0011': 8,
                                            '1010': 228,
                                            '0111': 74,
                                            '1011': 115}
 
        mps_sv =  most_probable_bitstring(cost_hamil, qaoa_probabilities(optimized_measurement_outcomes_sv))
        mps_shot =  most_probable_bitstring(cost_hamil, optimized_measurement_outcomes_shot)

        assert set(['0101', '1010']) == set(mps_sv['solutions_bitstrings']) # Equality for sv
        assert set(['0101', '1010']) >= set(mps_shot['solutions_bitstrings']) # Subset for shot/qpu


    def test_plot_cost(self):

        # Create the problem
        g = nx.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()
        
        q = QAOA()
        q.compile(vc, verbose=False)
        q.optimize()

        q.results.plot_cost()

    def test_get_counts(self):

        # measurement outcome from a statevector_simulator backend
        optimized_measurement_outcomes_sv = np.array([ 0.18256422-0.1296918j , -0.03786405-0.13158363j,
                                                -0.03786405-0.13158363j, -0.01910622+0.3151214j ,
                                                -0.03786405-0.13158363j,  0.42313283+0.15529604j,
                                                -0.01910622+0.3151214j ,  0.016895  +0.05117122j,
                                                -0.03786405-0.13158363j, -0.01910622+0.3151214j ,
                                                 0.42313283+0.15529604j,  0.016895  +0.05117122j,
                                                -0.01910622+0.3151214j ,  0.016895  +0.05117122j,
                                                 0.016895  +0.05117122j,  0.23799586+0.04022615j])

        counts_from_sv = qaoa_probabilities(optimized_measurement_outcomes_sv)

        # measurement outcome from a shot simulator or QPU
        optimized_measurement_outcomes_shot = {'0010': 1,
                                            '0110': 2,
                                            '1000': 1,
                                            '1001': 13,
                                            '0000': 1,
                                            '1100': 10,
                                            '1110': 201,
                                            '1111': 37,
                                            '0101': 225,
                                            '1101': 57,
                                            '0011': 8,
                                            '1010': 228,
                                            '0111': 74,
                                            '1011': 115}


        assert(optimized_measurement_outcomes_shot == Result.get_counts(optimized_measurement_outcomes_shot))
        assert(counts_from_sv == Result.get_counts(optimized_measurement_outcomes_sv))
 

if __name__ == '__main__':
    unittest.main()