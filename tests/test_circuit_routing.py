# # unit testing for circuit routing functionality in OQ
# import unittest
# import networkx as nx
# from openqaoa.algorithms.qaoa.qaoa_workflow import circuit_routing

# # some device topologies to use for testing
# def aspen_m2():
#     device = nx.from_edgelist(
#         [
#             (0, 1),
#             (0, 7),
#             (0, 103),
#             (1, 2),
#             (1, 16),
#             (100, 101),
#             (100, 107),
#             (101, 102),
#             (101, 116),
#             (10, 11),
#             (10, 17),
#             (10, 113),
#             (11, 12),
#             (11, 26),
#             (110, 111),
#             (110, 117),
#             (111, 112),
#             (111, 126),
#             (20, 21),
#             (20, 27),
#             (20, 123),
#             (21, 22),
#             (21, 36),
#             (120, 121),
#             (120, 127),
#             (121, 122),
#             (121, 136),
#             (30, 31),
#             (30, 37),
#             (30, 133),
#             (31, 32),
#             (31, 46),
#             (130, 131),
#             (130, 137),
#             (131, 132),
#             (131, 146),
#             (40, 41),
#             (40, 47),
#             (40, 143),
#             (41, 42),
#             (140, 141),
#             (140, 147),
#             (141, 142),
#             (2, 3),
#             (2, 15),
#             (102, 103),
#             (102, 115),
#             (12, 13),
#             (12, 25),
#             (112, 113),
#             (112, 125),
#             (22, 23),
#             (22, 35),
#             (122, 123),
#             (122, 135),
#             (32, 33),
#             (32, 45),
#             (132, 133),
#             (132, 145),
#             (42, 43),
#             (142, 143),
#             (3, 4),
#             (103, 104),
#             (13, 14),
#             (113, 114),
#             (23, 24),
#             (123, 124),
#             (33, 34),
#             (133, 134),
#             (43, 44),
#             (143, 144),
#             (4, 5),
#             (104, 105),
#             (104, 7),
#             (14, 15),
#             (114, 115),
#             (114, 17),
#             (24, 25),
#             (124, 125),
#             (124, 27),
#             (34, 35),
#             (134, 135),
#             (134, 37),
#             (44, 45),
#             (144, 145),
#             (144, 47),
#             (5, 6),
#             (105, 106),
#             (15, 16),
#             (115, 116),
#             (25, 26),
#             (125, 126),
#             (35, 36),
#             (135, 136),
#             (45, 46),
#             (145, 146),
#             (6, 7),
#             (106, 107),
#             (16, 17),
#             (116, 117),
#             (26, 27),
#             (126, 127),
#             (36, 37),
#             (136, 137),
#             (46, 47),
#             (146, 147),
#         ]
#     )

#     mapping = dict(zip(device, range(len(device))))
#     device = nx.relabel_nodes(device, mapping)

#     return device


# def double_t7():
#     return [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]]


# def single_t5():
#     return [[0, 1], [1, 2], [2, 3], [3, 4]]


# def hanoi():
#     device = [
#         [0, 1],
#         [1, 2],
#         [2, 3],
#         [3, 5],
#         [5, 8],
#         [8, 9],
#         [8, 11],
#         [11, 14],
#         [14, 13],
#         [13, 12],
#         [12, 15],
#         [10, 12],
#         [10, 7],
#         [7, 6],
#         [7, 4],
#         [4, 1],
#         [15, 18],
#         [18, 17],
#         [18, 21],
#         [21, 23],
#         [23, 24],
#         [24, 25],
#         [25, 26],
#         [25, 22],
#         [22, 19],
#         [19, 20],
#         [16, 19],
#         [14, 16],
#     ]
#     return device


# def get_single_cycle(n_qubits):
#     device_edges = [[i, i + 1] for i in range(n_qubits - 1)]
#     device_edges.append([0, n_qubits - 1])
#     return device_edges


# def get_simple_heavyhex(nb_hexagons):
#     if nb_hexagons == 1:
#         return get_single_cycle(12)
#     elif nb_hexagons == 2:
#         one_hexagon = get_simple_heavyhex(1)
#         hexa2 = [[i, i + 1] for i in range(12, 19)]
#         device = one_hexagon + hexa2 + [[4, 12], [6, 20], [19, 20]]
#         return device
#     elif nb_hexagons == 3:
#         two_hexagons = get_simple_heavyhex(2)
#         hexa3 = [[i, i + 1] for i in range(21, 27)]
#         device = two_hexagons + hexa3 + [[8, 27], [19, 21]]
#         return device
#     elif nb_hexagons == 4:
#         three_hexagons = get_simple_heavyhex(3)
#         hexa4 = [[i, i + 1] for i in range(28, 34)]
#         device = three_hexagons + hexa4 + [[26, 28], [10, 34]]
#         return device


# device = nx.from_edgelist(get_simple_heavyhex(2))
# n_nodes = len(device)
# # Create fully-connected problem with as many variables as the number of nodes in the device
# problem = nx.complete_graph(n_nodes)

# # Initial mapping just maps each variable to the node with same label on the device
# initial_mapping = list(range(n_nodes))

# print(f"Number of Qubits: {n_nodes}")
# print(f"Number of Variables: {problem.number_of_nodes()}")
# print("Swaps Upper Bound:", (n_nodes - 1) * (n_nodes - 2) / 2)


# class TestingQAOACircuitRouting(unittest.TestCase):
#     """
#     simple unit-tests to ensure the proper
#     functioning of the pluggable routing algorithm
#     within OQ workflow
#     """
#     device = nx.from_edgelist(get_simple_heavyhex(2))
#     n_nodes = len(device)
#     problem = nx.complete_graph(n_nodes)

#     # First test with initial_mapping = None
#     def test_simple_routing(self):
#         initial_mapping_input = None
#         (gates, swap_mask, initial_mapping, final_mapping) = circuit_routing(
#             device, problem, initial_mapping_input
#         )
        
#         #ensure the swap_mask is a list of booleans with same size of gates
#         for mask in swap_mask:
#             assert isinstance(mask, bool)
#         assert len(gates) == len(swap_mask)
#         assert len(final_mapping) == len(initial_mapping)
        
#     def test_simple_routing_with_initial_mapping(self):
#         initial_mapping_input = list(range(n_nodes))
#         (gates, swap_mask, initial_mapping, final_mapping) = circuit_routing(
# 			device, problem, initial_mapping_input
# 		)
        
#         assert initial_mapping_input == initial_mapping, \
#             "The output initial mapping should match the input from user"
