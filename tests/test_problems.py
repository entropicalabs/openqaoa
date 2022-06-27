import unittest
import networkx as nx
import numpy as np
from openqaoa.problems.problem import (
	NumberPartition, QUBO, TSP, Knapsack,
	SlackFreeKnapsack, MaximumCut, MinimumVertexCover
)

def terms_list_equality(terms_list1, terms_list2):
	"""
	Check the terms equality between two terms list
	where the order of edges do not matter.
	"""
	if len(terms_list1) != len(terms_list2):
		bool = False
	else:
		for term1,term2 in zip(terms_list1, terms_list2):
			bool = True if (term1 == term2  or term1 == term2[::-1]) else False
	
	return bool


class TestProblem(unittest.TestCase):

	#TESTING QUBO CLASS METHODS
	def test_qubo_terms_and_weight_same_size(self):
		"""
		Test that creating a QUBO problem with invalid terms and weights
		sizes raises an exception and check the constant is detected correctly
		"""
		n = 2
		terms_wrong = [[0], [1]]
		terms_correct = [[0], [1], []]
		weights = [1, 2, 3]

		with self.assertRaises(ValueError):
			qubo_problem = QUBO(n, terms_wrong, weights)
		
		qubo_problem = QUBO(n, terms_correct, weights)
		self.assertEqual(qubo_problem.constant, 3)

	def test_qubo_cleaning_terms(self):
		"""Test that cleaning terms works for a QUBO problem"""
		terms = [[1, 2], [0], [2, 3], [2, 1], [0]]
		weights = [3, 4, -3, -2, -1]

		cleaned_terms = [[1, 2], [0], [2, 3]]
		self.assertEqual(QUBO.clean_terms_and_weights(terms, weights)[0], cleaned_terms)

	def test_qubo_cleaning_weights(self):
		"""Test that cleaning weights works for a QUBO problem"""
		terms = [[1, 2], [0], [2, 3], [2, 1], [0]]
		weights = [3, 4, -3, -2, -1]

		cleaned_weights = [1, 3, -3]
		self.assertEqual(QUBO.clean_terms_and_weights(terms, weights)[1], cleaned_weights)
	

	## TESTING NUMBER PARITION CLASS
	def test_number_partitioning_terms_weights_constant(self):
		"""Test that Number Partitioning creates the correct terms, weights, constant"""
		list_numbers = [1, 2, 3]
		expected_terms = [[0, 1], [0, 2], [1, 2]]
		expected_weights = [4, 6, 12]
		expected_constant = 14

		np_problem = NumberPartition(list_numbers)
		qubo_problem = np_problem.get_qubo_problem()

		self.assertTrue(terms_list_equality(qubo_problem.terms, expected_terms))
		self.assertEqual(qubo_problem.weights, expected_weights)
		self.assertEqual(qubo_problem.constant, expected_constant)

	def test_number_partitioning_random_problem(self):
		"""Test randomly generated NumberPartition problem"""
		np.random.seed(1234)
		np_prob_random = NumberPartition.random_instance(n_numbers=5).get_qubo_problem()

		#regenerate the same numbers randomly
		np.random.seed(1234)
		random_numbers_list = list(map(int, np.random.randint(1, 10, size=5)))
		manual_np_prob = NumberPartition(random_numbers_list).get_qubo_problem()

		self.assertTrue(terms_list_equality(np_prob_random.terms,manual_np_prob.terms))
		self.assertEqual(np_prob_random.weights,manual_np_prob.weights)
		self.assertEqual(np_prob_random.constant,manual_np_prob.constant)

	
	## TESTING MAXIMUMCUT CLASS
	def test_maximumcut_terms_weights_constant(self):
		"""Test that MaximumCut creates a correct QUBO from the provided graph"""

		gr = nx.generators.random_graphs.fast_gnp_random_graph(n=10, p=0.8)
		gr_edges = [list(edge) for edge in gr.edges()]
		gr_weights = [1]*len(gr_edges)

		maxcut_prob_qubo = MaximumCut(gr).get_qubo_problem()


		self.assertTrue(terms_list_equality(gr_edges,maxcut_prob_qubo.terms))
		self.assertEqual(gr_weights,maxcut_prob_qubo.weights)
		self.assertEqual(0, maxcut_prob_qubo.constant)

	def test_maximumcut_random_problem(self):
		"""Test MaximumCut random instance method"""

		seed = 1234
		gr = nx.generators.random_graphs.fast_gnp_random_graph(n=10, p=0.8, seed=seed)
		maxcut_manual_prob = MaximumCut(gr).get_qubo_problem()

		np.random.seed(1234)
		maxcut_random_prob = MaximumCut.random_instance(n_nodes=10, edge_probability=0.8, seed=seed).get_qubo_problem()

		self.assertTrue(terms_list_equality(maxcut_manual_prob.terms,maxcut_random_prob.terms))
		self.assertEqual(maxcut_manual_prob.weights,maxcut_random_prob.weights)
		self.assertEqual(maxcut_manual_prob.constant, maxcut_random_prob.constant)
		

	## TESTING KNAPSACK CLASS
	def test_knapsack_terms_weights_constant(self):
		"""Test that Knapsack creates the correct QUBO problem"""

		values = [2,4,3,5]
		weights = [3,6,9,1]
		weight_capacity = 15
		n_qubits = len(values) + int(np.ceil(np.log2(weight_capacity)))
		penalty = 2*max(values)
		knap_terms = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [4, 5], [4, 6], [4, 7], [5, 6],
					  [5, 7], [6, 7], [0, 4], [0, 5], [0, 6], [0, 7], [1, 4], [1, 5], [1, 6], [1, 7],
					  [2, 4], [2, 5], [2, 6], [2, 7], [3, 4], [3, 5], [3, 6], [3, 7], [0], [1], [2],
					  [3], [4], [5], [6], [7]]
		knap_weights = [10.0, 20.0, 40.0, 40.0, 80.0, 160.0, 90.0, 135.0, 15.0, 270.0, 30.0, 45.0,
						 15.0, 30.0, 45.0, 5.0, 30.0, 60.0, 90.0, 10.0, 60.0, 120.0, 180.0, 20.0, 120.0,
						 240.0, 360.0, 40.0, -20.0, -40.0, -80.0, -160.0, -59.0, -118.0, -178.5, -17.5]
		knap_constant = 563.0

		knapsack_prob_qubo = Knapsack(values,weights,weight_capacity,penalty).get_qubo_problem()

		self.assertTrue(terms_list_equality(knap_terms,knapsack_prob_qubo.terms))
		self.assertEqual(knap_weights,knapsack_prob_qubo.weights)
		self.assertEqual(knap_constant, knapsack_prob_qubo.constant)
		self.assertEqual(n_qubits, knapsack_prob_qubo.n)

	def test_knapsack_random_problem(self):
		"""Test random instance method of Knapsack problem class"""

		np.random.seed(1234)
		n_items = 5
		values = list(map(int, np.random.randint(1, n_items, size=n_items)))
		weights = list(map(int, np.random.randint(1, n_items, size=n_items)))
		weight_capacity = np.random.randint(np.min(weights) * n_items, np.max(weights) * n_items)
		penalty = 2*np.max(values)

		knap_manual = Knapsack(values,weights,weight_capacity,penalty).get_qubo_problem()

		np.random.seed(1234)
		knap_random_instance = Knapsack.random_instance(n_items=n_items).get_qubo_problem()

		self.assertTrue(terms_list_equality(knap_manual.terms,knap_random_instance.terms))
		self.assertEqual(knap_manual.weights,knap_random_instance.weights)
		self.assertEqual(knap_manual.constant,knap_random_instance.constant)
		self.assertEqual(knap_manual.n,knap_random_instance.n)


	##TESTING SLACKFREEKNAPSACK CLASS
	def test_slackfreeknapsack_terms_weights_constant(self):
		"""Test that SlackFree Knapsack creates the correct QUBO problem"""

		values = [2,4,3,5]
		weights = [3,6,9,1]
		weight_capacity = 15
		n_qubits = len(values)
		penalty = 2*max(values)
		slknap_terms = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0], [1], [2], [3]]
		slknap_weights = [90.0, 135.0, 15.0, 270.0, 30.0, 45.0, 166.0, 332.0, 496.5, 57.5]
		slknap_constant = 613.0

		slknapsack_prob_qubo = SlackFreeKnapsack(values,weights,weight_capacity,penalty).get_qubo_problem()

		self.assertTrue(terms_list_equality(slknap_terms,slknapsack_prob_qubo.terms))
		self.assertEqual(slknap_weights,slknapsack_prob_qubo.weights)
		self.assertEqual(slknap_constant,slknapsack_prob_qubo.constant)
		self.assertEqual(n_qubits,slknapsack_prob_qubo.n)

	def test_slackfreeknapsack_random_problem(self):
		"""Test random instance method of SlackFree Knapsack problem class"""

		np.random.seed(1234)
		n_items = 5
		values = list(map(int, np.random.randint(1, n_items, size=n_items)))
		weights = list(map(int, np.random.randint(1, n_items, size=n_items)))
		weight_capacity = np.random.randint(np.min(weights) * n_items, np.max(weights) * n_items)
		penalty = 2*np.max(values)

		slknap_manual = SlackFreeKnapsack(values,weights,weight_capacity,penalty).get_qubo_problem()

		np.random.seed(1234)
		slknap_random_instance = SlackFreeKnapsack.random_instance(n_items=n_items).get_qubo_problem()

		self.assertTrue(terms_list_equality(slknap_manual.terms,slknap_random_instance.terms))
		self.assertEqual(slknap_manual.weights,slknap_random_instance.weights)
		self.assertEqual(slknap_manual.constant,slknap_random_instance.constant)
		self.assertEqual(slknap_manual.n,slknap_random_instance.n)

	
	# TESTING MINIMUMVERTEXCOVER CLASS
	def test_mvc_terms_weights_constant(self):
		"""Test terms,weights,constant of QUBO generated by MVC class"""

		mvc_terms = [[0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4],
					 [0], [1], [2], [3], [4]]
		mvc_weights = [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 2.0, 2.0, 2.0, 3.25, 3.25]
		mvc_constant = 10.0
		
		gr = nx.generators.fast_gnp_random_graph(5,0.8,seed=1234)
		mvc_prob = MinimumVertexCover(gr,field=1.0,penalty=5).get_qubo_problem()

		self.assertTrue(terms_list_equality(mvc_terms,mvc_prob.terms))
		self.assertEqual(mvc_weights,mvc_prob.weights)
		self.assertEqual(mvc_constant,mvc_prob.constant)

	def test_mvc_random_problem(self):
		"""Test the random_instance method of MVC class"""
		mvc_terms = [[0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4], [0], [1], [2], [3], [4]]
		mvc_weights = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 4.5, 4.5, 4.5, 7.0, 7.0]
		mvc_constant = 17.5

		mvc_prob_random = MinimumVertexCover.random_instance(n_nodes=5, edge_probability=0.8,seed=1234).get_qubo_problem()

		self.assertTrue(terms_list_equality(mvc_terms,mvc_prob_random.terms))
		self.assertEqual(mvc_weights,mvc_prob_random.weights)
		self.assertEqual(mvc_constant,mvc_prob_random.constant)
	

	# TESTING TSP PROBLEM CLASS
	def test_tsp_terms_weights_constant(self):
		"""Testing TSP problem creation"""

		tsp_terms = [[0, 4], [3, 7], [10, 6], [0, 5], [8, 3], [11, 6], [1, 3], [4, 6], [9, 7],
					 [1, 5], [8, 4], [11, 7], [2, 3], [5, 6], [8, 9], [2, 4], [5, 7], [8, 10],
					 [0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]
		tsp_weights = [1.076557410627608, 1.076557410627608, 1.076557410627608, 1.4220308321764457,
					   1.4220308321764457, 1.4220308321764457, 1.076557410627608, 1.076557410627608,
					   1.076557410627608, 1.3496158701037206, 1.3496158701037206, 1.3496158701037206,
					   1.4220308321764457, 1.4220308321764457, 1.4220308321764457, 1.3496158701037206,
					   1.3496158701037206, 1.3496158701037206, -2.4985882428040536, -2.4261732807313283,
					   -2.771646702280166, -4.997176485608107, -4.852346561462657, -5.543293404560332,
					   -4.997176485608107, -4.852346561462657, -5.543293404560332, -2.4985882428040536,
					   -2.4261732807313283, -2.771646702280166]
		tsp_constant = 23.089224677446644

		np.random.seed(1234)
		x_coords = np.random.uniform(0,10,3)
		y_coords = np.random.uniform(0,10,3)
		coords = list(zip(x_coords,y_coords))
		tsp_prob = TSP(coords).get_qubo_problem()

		self.assertTrue(terms_list_equality(tsp_terms,tsp_prob.terms))
		self.assertEqual(tsp_weights,tsp_prob.weights)
		self.assertEqual(tsp_constant,tsp_prob.constant)

	def test_tsp_random_instance(self):
		"""Testing the random_instance method of the TSP problem class"""

		np.random.seed(1234)
		n_cities=4
		box_size = np.sqrt(n_cities)
		coordinates = box_size * np.random.rand(n_cities, 2)

		tsp_prob = TSP(coordinates).get_qubo_problem()

		np.random.seed(1234)
		tsp_prob_random = TSP.random_instance(n_cities=n_cities).get_qubo_problem()

		self.assertTrue(terms_list_equality(tsp_prob_random.terms,tsp_prob.terms))
		self.assertEqual(tsp_prob_random.weights,tsp_prob.weights)
		self.assertEqual(tsp_prob_random.constant,tsp_prob.constant)


if __name__ == '__main__':
	unittest.main()