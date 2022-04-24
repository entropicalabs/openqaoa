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

import numpy as np
from itertools import permutations
from typing import List


def dicke_basis(excitations: int, n_qubits: int) -> np.ndarray:
    """
    Generates the Dicke basis state |ek> with k excitations

    Parameters
    ----------
    excitations: 
        Number of excitations in the basis vector
    n_qubits:
        Total number of qubits in the system

    Returns
    -------
    total_basis_comp:
        Total basis states present in the expected Dicke vector in the computational basis
        np.ndarray[str] 

    """
    assert n_qubits >= excitations, "Excitations cannot be larger than total qubits in system"
    sub_sys_excitations = np.ones(excitations, dtype=int)
    sub_sys_ground = np.zeros(n_qubits-excitations, dtype=int)

    total_state = np.concatenate((sub_sys_ground, sub_sys_excitations))
    total_basis_comp = set(permutations(total_state))
    total_basis_comp = np.array(
        [''.join(str(i) for i in basis_comp) for basis_comp in total_basis_comp])

    return total_basis_comp


def dicke_wavefunction(excitations, n_qubits):
    """
    Generate the k-excitations dicke statevector

    Parameters
    ----------
    excitations: str
        The number of excitations in the basis

    n_qubits: int
        The number of qubits in the register
    """

    k_dicke = dicke_basis(excitations, n_qubits)
    k_dicke_ints = [int(state, 2) for state in k_dicke]
    wavefunction = np.array([1.+0.j if num in k_dicke_ints else 0.+0.j for num in range(
        2**n_qubits)], dtype=complex)/np.sqrt(len(k_dicke_ints))
    return wavefunction


def k_cumulative_excitations(k: int, n_qubits: int):
    """
    Generates the Upper bound excitations basis vector |Ek>, which a superposition of all
    Dicke basis vectors upto excitation number "k"

    Parameters
    ----------
    k: 
        Upper bound on number of excitations in the basis vector
    n_qubits:
        Total number of qubits in the system

    Returns
    -------
    wavefunction:
        The wavefunction vector for a given cumulative Dicke states with <=k excitations

    """
    cumulative_dicke_bases = np.array(['0'*n_qubits])
    for exc in range(1, k+1):
        cumulative_dicke_bases = np.concatenate(
            (cumulative_dicke_bases, dicke_basis(exc, n_qubits)))

    wavefn_locs = [int(basis, 2) for basis in cumulative_dicke_bases]
    wavefunction = np.array([1 if loc in wavefn_locs else 0 for loc in range(
        2**n_qubits)], dtype=complex)/np.sqrt(len(wavefn_locs))

    return wavefunction


def knapsack_balanced_basis(weight_capacity: int, weights_list: List, decision_register: List, slack_register: List):
    """
    Generates the basis where the system register is balanced for Knapsack Hamiltonian, i.e. Slack register
    compensating the decision register state to cancel the penalty term in Hamiltonian. 
    NOTE: Cannot cancel the penalty term if decision register weight exceeds the weight capacity of the sack

    Parameters
    ----------
    weight_capacity:
        The knapsack capacity, i.e. the upper limit on the weights that can be added in the knapsack
    weights_list:
        List of item weights in the problem
    decision_register:
        The qubit regsiter of the decision bits
    slack_register:
        The qubit regsiter of the slack bits 

    Returns
    -------
    wavefunction:
    """
    n_decision_qubits = len(decision_register)
    n_slack_qubits = len(slack_register)
    n_total_qubits = n_slack_qubits + n_decision_qubits

    def to_bin(number, n_qubits):
        # not using np.binary_repr because it is deprecated!
        binary_form = bin(number)[2:].zfill(n_qubits)
        return binary_form

    decision_config_weights = {to_bin(dec_i, n_decision_qubits): sum([weight*int(to_bin(dec_i, n_decision_qubits)[i])
                                                                      for i, weight in enumerate(weights_list)]) for dec_i in range(2**n_decision_qubits)}

    decision_slack_configs = {to_bin(dec_i, n_decision_qubits): (to_bin(weight_capacity-decision_config_weights[to_bin(dec_i, n_decision_qubits)], n_slack_qubits)
                                                                 if decision_config_weights[to_bin(dec_i, n_decision_qubits)] < weight_capacity else to_bin(0, n_slack_qubits)) for dec_i in range(2**n_decision_qubits)}

    all_configs = []
    for dec_config, slack_config in decision_slack_configs.items():
        config = np.empty(n_total_qubits, dtype=str)
        for i, loc in enumerate(decision_register):
            config[loc] = dec_config[i]
        for i, loc in enumerate(slack_register[::-1]):
            config[loc] = slack_config[i]

        config_str = ''.join(i for i in config)
        all_configs.append(config_str[::-1])

    wavefn_locs = [int(basis, 2) for basis in all_configs]
    wavefunction = np.array([1 if loc in wavefn_locs else 0 for loc in range(
        2**n_total_qubits)], dtype=complex)/np.sqrt(len(wavefn_locs))

    return wavefunction

# ############################################################################
# Lightcone QAOA building blocks
# ############################################################################

# def _lightcone_registers(graph: nx.Graph,
#                          n_steps: int,
#                          edge: Tuple) -> Tuple[List, List, List]:
#     """
#     Determine the qubits in the lightcone for each step. Note this is slightly modified
#     from the corresponding method in EntropicaQAOA, to return lists instead of sets at
#     the end.

#     Parameters
#     ----------
#     graph:
#         The graph whose lightcones we want to determine. Nodes must be Ints.
#     n_steps:
#         Depth of the QAOA circuit (i.e. ``p``)
#     edge:
#         The edge whose lightcone you want. Should be a tuple ``(node1, node2)``


#     Returns
#     -------
#     Tuple[List, List, List]:
#         Three lists.
#         The first contains the sets of qubits to apply the RX rotations for
#         all steps.
#         The second contains the sets of qubits to apply the RZ rotations
#         for all steps.
#         The third contains the qubit pairs to apply the RZZ rotations for
#         all steps

#     """
#     # will be a list of lists of qubits that need a mixer rotation at each step
#     regs = []
#     # will be a list of lists of qubits that need a bias rotation at each step
#     qubits_singles = []
#     # will be a list of lists of pairs of qubits that need a cost rotation at each step
#     qubits_pairs = [{edge}]
#     # get a list of the qubits with a bias on them
#     bias_qubits = [n[0] for n in graph.nodes(data=True) if 'weight' in n[1]]

#     # fill the lists
#     for _ in range(n_steps):
#         new_pairs = set()
#         old_pairs = qubits_pairs[-1]
#         new_reg = {q for pair in old_pairs for q in pair}
#         regs.append(new_reg)
#         qubits_singles.append(new_reg.intersection(bias_qubits))
#         for pair in old_pairs:
#             new_pairs.update(graph.edges(pair[0]))
#             new_pairs.update(graph.edges(pair[1]))
#             new_pairs = {tuple(sorted(p)) for p in new_pairs}
#         qubits_pairs.append(new_pairs)
#     qubits_pairs.pop(0)

#     # Now reverse the lists: this is because we are essentially working 'outwards' from the operator corresponding
#     # to the edge of interest, back to the beginning of the circuit - we have the 'time-reversed' order of operations.
#     # Thus, in order to get the operations in the correct time order, we need to reverse them.
#     qubits_pairs.reverse()
#     qubits_singles.reverse()
#     regs.reverse()

#     regs = [list(i) for i in regs]
#     qubits_singles = [list(i) for i in qubits_singles]
#     qubits_pairs = [list(i) for i in qubits_pairs]

#     return regs, qubits_singles, qubits_pairs


# def _lightcone_param_indices(params: Type[AbstractParams],
#                              regs: List,
#                              qubits_singles: List,
#                              qubits_pairs: List) -> Tuple[List, List, List]:
#     """
#     Return index arrays for ``params.x_rotation_angles``,
#     ``params.z_rotation_angles``, to act on the qubits specified by ``regs``,
#     ``qubits_singles``,....

#     Parameter
#     ---------
#     params:
#         QAOAParameter object
#     regs:
#         The first list produced by ``_lightcone_registers``
#     qubits_singles:
#         The second list produced by ``_lightcone_registers``
#     qubits_pairs:
#         The third list produced by ``_lightcone_registers``

#     Returns
#     -------
#     Tuple[List, List, List]:
#         Index arrays for ``params.x_rotation_angles``,
#         ``params.z_rotation_angles``, ... to get exactly the angles that
#         act on the qubits in ``regs``, ``qubits_singles``,...
#     """
#     reg_index_dict = {q: i for i, q in enumerate(params.reg)}
#     singles_index_dict = {q: i for i, q in enumerate(params.qubits_singles)}
#     pairs_index_dict = {tuple(sorted(pair)): i
#                         for i, pair in enumerate(params.qubits_pairs)}

#     x_indices_list = []
#     z_indices_list = []
#     zz_indices_list = []

#     for i in range(params.n_steps):
#         x_indices = np.array([reg_index_dict[q] for q in regs[i]], dtype=int)
#         z_indices = np.array([singles_index_dict[q]
#                               for q in qubits_singles[i]], dtype=int)
#         zz_indices = np.array([pairs_index_dict[p] for p in qubits_pairs[i]],
#                               dtype=int)

#         x_indices_list.append(x_indices)
#         z_indices_list.append(z_indices)
#         zz_indices_list.append(zz_indices)

#     return x_indices_list, z_indices_list, zz_indices_list

# def _prepare_lightcone_params(lc_qubits: List,
#                              lc_regs: List,
#                              lc_singles: List,
#                              lc_pairs: List,
#                              params: Type[AbstractParams],
#                              x_indices,
#                              z_indices,
#                              zz_indices) -> tuple:
#     """
#     Prepare the QAOA lightcone program on the qubits specified in
#     ``regs``, ``qubits_singles``,...

#     Parameters
#     ----------
#     lc_regs:
#         The first list produced by ``_lightcone_registers`` for the lc in question
#     lc_qubits_singles:
#         The second list produced by ``_lightcone_registers`` for the lc in question
#     lc_qubits_pairs:
#         The third list produced by ``_lightcone_registers`` for the lc in question
#     params:
#         QAOAParameter object, corresponding to the whole problem

#     Returns
#     -------
#     tuple:
#         A tuple with the QAOA circuit parameters, and the relevant biases and couplings, for the lightcone program
#     """

#     # Map qubits_singles and qubits_pairs to this register
#     pairs = lc_pairs[0]
#     #pairs_reg = [(lc_qubits.index(pair[0]), lc_qubits.index(pair[1])) for pair in pairs]
#     singles = lc_singles[0]
#     #singles_reg = [lc_qubits.index(qb) for qb in singles]

#     # Get the biases and couplings
#     all_system_pairs = [tuple(sorted(pair)) for pair in params.qubits_pairs]

#     lc_couplings = []
#     for pair in pairs:
#         ind = all_system_pairs.index(pair)
#         lc_couplings.append(params.pair_qubit_coeffs[ind])

#     all_system_singles = params.qubits_singles
#     lc_biases = []
#     if singles:
#         for qb in singles:
#             ind = all_system_singles.index(qb)
#             lc_biases.append(params.single_qubit_coeffs[ind])

#     # Angles needed for the lightcone
#     x_angles_lc = []
#     z_angles_lc = []
#     zz_angles_lc = []
#     for i in range(params.n_steps):

#         # X rotation angles needed
#         angles_x = [0]*len(lc_qubits)
#         for j in lc_regs[i]:
#             ind = lc_qubits.index(j)
#             angles_x[ind] = params.x_rotation_angles[i, j]
#         x_angles_lc.append(angles_x)

#         if singles:
#             # Z rotation angles needed
#             angles_z = [0]*len(singles)
#             for j,qb in enumerate(lc_singles[i]):
#                 ind = singles.index(qb)
#                 angles_z[ind] = params.z_rotation_angles[i, z_indices[i][j]]
#             z_angles_lc.append(angles_z)

#         # ZZ rotation angles needed
#         angles_zz = [0]*len(lc_pairs[0])
#         for k, pair in enumerate(lc_pairs[i]):
#             ind = pairs.index(pair)
#             angles_zz[ind] = params.zz_rotation_angles[i,zz_indices[i][k]]
#         zz_angles_lc.append(angles_zz)

#     #lc_params = (params.n_steps, x_angles_lc, lc_biases, z_angles_lc, lc_couplings, zz_angles_lc)
#     lc_params = (x_angles_lc, z_angles_lc, zz_angles_lc)

#     return lc_params

# def lightcone_hams(graph: nx.Graph) -> List[np.array]:
#     """
#     Extracts the local QAOA terms from a graph and returns them as a list.

#     Parameters
#     ----------
#     graph:
#         The graph with edge and possibly node weights.

#     Returns
#     -------
#     List[np.array]:
#         A list of 4x4 numpy arrays corresponding to the cost terms in the total
#         hamiltonian. Sorted in the way graph.edges() returns the edges.
#     """
#     terms = []
#     all_qubits = {n[0]: n[1] for n in graph.nodes(data=True)}
#     for edge in graph.edges(data=True):
#         mat = np.zeros(shape=(4, 4))
#         mat += edge[2]['weight'] * np.diag([1, -1, -1, 1])
#         edge = tuple(sorted([edge[0], edge[1]]))
#         if edge[0] in all_qubits:
#             attrs = all_qubits.pop(edge[0])
#             try:
#                 mat += attrs['weight'] * np.diag([1, -1, 1, -1])
#             except KeyError:
#                 pass
#         if edge[1] in all_qubits:
#             attrs = all_qubits.pop(edge[1])
#             try:
#                 mat += attrs['weight'] * np.diag([1, 1, -1, -1])
#             except KeyError:
#                 pass
#         terms.append(mat)
#     return terms


# class Lightcone(object):
#     """A container class for QAOA lightcones.

#     Parameters
#     ----------
#     graph:
#         The graph whose lightcones we want to build.
#     params:
#         QAOA parameter object to use for this QAOA run.
#     edge:
#         The edge whose lightcone you want to build.

#     Attributes
#     ----------
#     regs: List[Set]
#         The i-th set in the list contains the qubits that need a RX rotation
#         in the i-th QAOA layer
#     qubits_singles: List[Set]
#         The i-th set in the list contains the qubits that need a RZ rotation
#         in the i-th QAOA layer
#     qubits_pairs: List[Set[Tuple]]
#         The i-th set in the list contains the qubits that need a RZ rotation
#         in the i-th QAOA layer
#     prepare_ansatz: Program
#         The lightcone QAOA pyquil program
#     self.n_qubits: int
#         The actual size of the lightcone in qubits
#     self.i:
#         The index of the first qubit in ``edge``
#     self.j:
#         The index of the second qubit in ``edge``
#     """

#     ## TODO:
#     #  - Fix the docstring above
#     #  - make __str__ or __repr___ method for this class, which would output something
#     #    similar to what is done with the ordinary QAOA param objects

#     def __init__(self,
#                  graph: nx.Graph,
#                  params: Type[AbstractParams],
#                  edge: Tuple):

#         # get the lists of relevant qubits and pairs in each step
#         self.lc_regs, self.lc_singles, self.lc_pairs\
#             = _lightcone_registers(graph, params.n_steps, edge)

#         # get the correct indices for the rotation angles
#         self.x_indices, self.z_indices, self.zz_indices\
#             = _lightcone_param_indices(params, self.lc_regs,
#                                        self.lc_singles, self.lc_pairs)

#         # all qubits in this lightcone
#         self.lc_qubits = sorted({q for pair in self.lc_pairs[0] for q in pair})
#         self.lc_n_qubits = len(self.lc_qubits)

#         # get the angles needed for the circuit, and the biases and couplings
#         self.lc_params =\
#             _prepare_lightcone_params(self.lc_qubits, self.lc_regs, self.lc_singles,
#                                      self.lc_pairs, params, self.x_indices,
#                                      self.z_indices, self.zz_indices)

#         # Some helper parameters needed to compute the density matrix corresponding to the output
#         # of this lightcone circuit
#         final_qubits = sorted(self.lc_regs[-1])
#         self.i = self.lc_n_qubits - 1 - self.lc_qubits.index(final_qubits[0])
#         self.j = self.lc_n_qubits - 1 - self.lc_qubits.index(final_qubits[1])

#     def density_matrix(self, wf):
#         """
#         compute the reduced density matrix on the two qubits in `edge`
#         from a wavefunction on all the qubits in that anything is happening on
#         """
#         einstring = ascii_letters[0:self.j] + "W" + ascii_letters[self.j:self.i-1]
#         einstring += "X" + ascii_letters[self.i:self.lc_n_qubits - 1] + ","
#         einstring += ascii_letters[0:self.j] + "Y" + ascii_letters[self.j:self.i-1]
#         einstring += "Z" + ascii_letters[self.i:self.lc_n_qubits - 1]
#         einstring += "->WXYZ"
#         wf = wf.reshape([2]*self.lc_n_qubits)
#         rho = np.einsum(einstring, wf, wf.conj())
#         return rho.reshape((4, 4))
