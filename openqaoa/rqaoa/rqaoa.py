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

from openqaoa.qaoa_parameters import Hamiltonian
from openqaoa.utilities import bitstring_energy
from openqaoa.problems.problem import QUBO




def max_terms(exp_vals_z: np.ndarray, corr_matrix: np.ndarray, n_elim: int):
    """
    Extracts the n_elim expectation values (single spin and correlation) with 
    highest magnitude, and uses them to impose the elimination constraint on 
    the spins.

    Parameters
    ----------
    exp_vals_z: `np.array`
        Single spin expectation values.
    corr_matrix: `np.array`
        Correlation matrix.
    n_elim: `int`
        Number of spins to eliminate.

    Returns
    -------
    max_terms_and_stats: `dict` 
        Dictionary containing terms to be eliminated and their expectation values.
    """
    # Copy list of single spin expectation values
    Z = exp_vals_z.copy()

    # Copy correlation matrix
    M = corr_matrix.copy()

    # Initialize dict of terms and respective costs
    max_terms_and_stats = {}

    # Find the n_max+1 largest correlation pairs
    for _ in range(n_elim):

        # Obtain max_spin, its Z expectation value, and store it - round to suppress errors from numbers approximating 0
        max_spin = np.unravel_index(abs(Z).argmax(), Z.shape)
        max_val = np.round(Z[max_spin], 10)

        # Obtain max_term, its correlation, and store it
        max_term = np.unravel_index(abs(M).argmax(), M.shape)
        max_corr = np.round(M[max_term], 10)

        # Eliminate the expectation value with highest absolute value - round to suppress errors from numbers approximating 0
        if np.abs(max_corr) > np.abs(max_val):
            max_terms_and_stats.update({max_term: max_corr})

            # Set the term to 0 to check for the next one
            M[max_term] = 0

        else:

            # If maximum value is 0, all expectation values are vanishing
            if max_val == 0:
                break
            else:
                max_terms_and_stats.update({max_spin: max_val})

            # Set the spin value to 0 to check for the next one
            Z[max_spin] = 0

    # Flag if we have have not been able to extract any relation for the terms
    if max_terms_and_stats == {}:
        print(f'All expectation values are 0: Breaking degeneracy by fixing a qubit\n')
        max_terms_and_stats = {(0,):-1.0}
        
    return max_terms_and_stats


def ada_max_terms(exp_vals_z: np.ndarray, corr_matrix: np.ndarray, n_max: int):
    """
    Extracts the n_max+1 expectation values (single spin and correlation) with 
    highest magnitude, computes the average among them and selects the ones
    above average for elimination. The maximum number of potential candidates is n_max.

    Parameters
    ----------
    exp_vals_z: `np.array`
        Single spin expectation values.
    corr_matrix: `np.array`
        Correlation matrix.
    n_max: `np.array`
        Maximum number of potential candidates for elimination.

    Returns
    -------
    max_terms_and_stats: `dict`
        Dictionary containing terms to be eliminated and their expectation values.
    """

    # Copy list of single spin expectation values
    Z = exp_vals_z.copy()

    # Copy correlation matrix
    M = corr_matrix.copy()

    # Initialize dict of terms and respective costs
    max_terms_and_stats = {}

    # Find the n_max+1 largest correlation pairs
    for _ in range(n_max + 1):

        # Obtain max_spin, its Z expectation value, and store it - round to suppress errors from numbers approximating 0
        max_spin = np.unravel_index(abs(Z).argmax(), Z.shape)
        max_val = np.round(Z[max_spin], 10)

        # Obtain max_term, its correlation, and store it
        max_term = np.unravel_index(abs(M).argmax(), M.shape)
        max_corr = np.round(M[max_term], 10)

        # Eliminate the expectation value with highest absolute value - round to suppress errors from numbers approximating 0
        if np.abs(max_corr) > np.abs(max_val):
            max_terms_and_stats.update({max_term: max_corr})

            # Set the term to 0 to check for the next one
            M[max_term] = 0

        else:

            # If maximum value is 0, all expectation values are vanishing
            if max_val == 0:
                break
            else:
                max_terms_and_stats.update({max_spin: max_val})

            # Set the spin value to 0 to check for the next one
            Z[max_spin] = 0

    # Flag if we have have not been able to extract any relation for the terms
    if max_terms_and_stats == {}:
        print(f'All expectation values are 0: Breaking degeneracy by fixing a qubit\n')
        max_terms_and_stats = {(0,):-1.0}
        
    # Correlation average magnitude
    avg_mag_stats = np.round(
        np.mean(np.abs(list(max_terms_and_stats.values()))), 10)

    # Select only the ones above average
    max_terms_and_stats = {key: value for key, value in max_terms_and_stats.items(
    ) if np.abs(value) >= avg_mag_stats}

    # Cut down the number of eliminations if, due to symmetry, they exceed the number allowed - relevant for unweighted graphs
    if len(max_terms_and_stats) > n_max:

        max_keys = list(max_terms_and_stats.keys())[0:n_max]
        max_terms_and_stats = {
            key: max_terms_and_stats[key] for key in max_keys}

    return max_terms_and_stats


def find_parent(spin_map: dict, spin: int, factor: int = 1):
    """
    Finds parent spin recursively following the chain of dependencies in the spin map.

    Parameters
    ----------
    spin_map: `dict`
        Mapping containing all dependencies for eliminated spins.
    spin: `int`
        Spin whose parent we want to find.
    factor: `int`, optional
        Defaults to 1 to initialize the multiplicative factor.

    Returns
    -------
    parent_spin: `int`
        Parent spin.
    factor: `int`
        Cumulative factor connecting the input spin with its parent spin.
    """

    # Extract parent spin
    parent_spin = spin_map[spin][1]
    factor *= spin_map[spin][0]

    # If spin maps to itself or it is None, we have found parent spin
    if parent_spin == spin or parent_spin is None:
        return parent_spin, factor

    # Recursively follow dependencies
    else:
        spin = parent_spin

    return find_parent(spin_map, spin, factor)


def spin_mapping(problem: QUBO, max_terms_and_stats: dict):
    """
    Generates a map between spins in the original problem graph and in the reduced graph.
    Elimination constraints from correlations define a constrained spin, to be removed, and a 
    parent spin, to be kept. Parent spins determine the state of multiple spins by a 
    chain of dependencies between spins due to the different constraints. Note that there 
    is always a parent spin and only one. If cycles are present, less edges will 
    be eliminated to satisfy this requirement. Constraints following from biases result in 
    fixing spins to a specific value. In this case, the parent spin is set 
    to None. Spins in the map that are not eliminated are mapped to themselves.

    Parameters
    ----------
    problem: `QUBO`
        QUBO problem object containing the problem statement.
    max_terms_and_stats: `dict`
        Dictionary containing the terms to be eliminated and their expectation values.

    Returns
    -------
    spin_map: `dict`
        Dictionary containing all the mapping dependencies. The keys
        correspond to the original spins. The values are tuple pairs,
        containing the parent spin and the factor relating both.
        The structure is: { old_spin : (factor, new_spin) }.
    """

    register = list(range(problem.n))
    max_terms = list(max_terms_and_stats.keys())

    # Initialize with every spin mapping to itself
    spin_map = dict((spin, (1, spin)) for spin in register)

    # Retrieve all spins involved in terms with highest correlation
    spin_candidates = set([spin for term in max_terms for spin in term])

    # Order term entries in descending magnitude order for correct insertion in solution
    sorted_max_ts = sorted(max_terms_and_stats.items(),
                           key=lambda x: np.abs(x[1]), reverse=True)
    
    # Build spin map from all expectation values
    for term, stat in sorted_max_ts:

        # Single spin expectation value
        if len(term) == 1:

            # Select spin
            spin = term[0]

            # Check if it has been fixed already
            parent, _ = find_parent(spin_map, spin)

            # If not, fix it
            if parent is not None:
                spin_map.update({spin: (np.sign(stat), None)})
                
        # Correlation terms
        else:

            # Select spins
            spin_keep, spin_remove = term

            # Retrieve parents and fact
            parent_keep, factor_keep = find_parent(spin_map, spin_keep)
            parent_remove, factor_remove = find_parent(spin_map, spin_remove)

            # If neither spin has been fixed, store relation
            if parent_keep is not None and parent_remove is not None:

                # Ensure we keep the spin with lowest label for future state reconstruction
                if parent_remove < parent_keep:
                    parent_keep, parent_remove = parent_remove, parent_keep
                    factor_keep, factor_remove = factor_remove, factor_keep

                # If a cycle is encountered one edge is discarded
                if parent_remove == parent_keep:
                    continue

                # Update the spin map
                else:
                    spin_map.update({parent_remove: (
                        factor_remove**(-1) * factor_keep * np.sign(stat), parent_keep)})

            # If both spins have been fixed, ignore correlation
            elif parent_keep is None and parent_remove is None:
                continue

            # If one spin has been fixed, fix the second one according to correlation value
            else:
            
                # Extract fixed and unfixed spins
                spin_fixed, factor_fixed = (parent_keep, factor_keep) if parent_keep is None else (
                    parent_remove, factor_remove)
                spin_unfixed, factor_unfixed = (
                    parent_remove, factor_remove) if spin_fixed == parent_keep else (parent_keep, factor_fixed)

                # Fix spin
                spin_map.update({spin_unfixed: (
                    factor_unfixed**(-1) * factor_fixed * np.sign(stat), spin_fixed)})

    # Correct all dependencies
    for spin in spin_candidates:
        parent_spin, cumulative_factor = find_parent(spin_map, spin)
        spin_map.update({spin: (cumulative_factor, parent_spin)})
    
    return spin_map


def problem_from_dict(problem_dict: dict):
    """
    Transforms a QUBO problem, input as a dictionary, into a QUBO problem, output as
    a QUBO object, ensuring proper labelling of the nodes. For example, for a set
    of nodes [0,1,4,6] with edges [(0,1),(1,4),(4,6),(0,6)], after the relabelling, the
    Hamiltonian object will be constructed with node labels [0,1,2,3] and edges 
    [(0,1),(1,2),(2,3),(1,3)]. 

    Parameters
    ----------
    problem_dict: `dict`
        QUBO problem as a dictionary containing the edges as keys and weights as values.

    Returns
    -------
    problem: `QUBO`
        A QUBO problem object constructed using the classical_hamiltonian() method.    
    """

    edges = list(problem_dict.keys())
    weights = list(problem_dict.values())

    # Current indices after elimination
    register = list(set([i for edge in edges for i in edge]))

    # Index mapping
    label_mapping = {unique: i for i, unique in enumerate(register)}

    # Initialize qubit terms mapper
    label_edges_mapping = {}

    # Edges mapping
    for edge in edges:

        # Map linear term to linear term
        if len(edge) == 1:
            label_edges_mapping.update({edge: (label_mapping.get(edge[0]),)})

        # Map quadratic term to quadratic term
        elif len(edge) == 2:
            label_edges_mapping.update(
                {edge: (label_mapping.get(edge[0]), label_mapping.get(edge[1]))})

        # If constant term, just map to itself
        else:
            label_edges_mapping.update({edge: edge})

    # New edges
    new_edges = list(label_edges_mapping.values())
    
    # New hamiltonian
    problem = QUBO(n= len(register), terms=new_edges, weights=weights)

    return problem


def redefine_problem(problem: QUBO, spin_map: dict):
    """
    Returns the QUBO object of the reduced problem. Using the spin map, we construct a  dictionary
    containing the new set of edges and weights defining the new problem.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object containing the problem statement.
    spin_map: `dict`
        Dictionary containing the spin dependencies.

    Returns
    -------
    new_problem: `QUBO`
        QUBO object containing the reduced problem statement after spin eliminations.
    spin_map: `dict`
        Updated spin_map with sponatenous eliminations from cancellations during spin removal process.
    """

    # Define new QUBO problem as a dictionary
    new_problem_dict = {}

    # Scan all terms and weights
    for term, weight in zip(problem.terms, problem.weights):

        # Bias terms
        if len(term) == 1:

            # Extract spin from the map
            spin = term[0]

            # Extract parent spin and associated factor
            parent_spin, factor_spin = spin_map[spin][1], spin_map[spin][0]

            # If spin is fixed it only contributes with a constant term
            if parent_spin is None:
                continue

            # If unfixed, define new edge and weight
            else:
                new_edge = (parent_spin,)
                new_weight = factor_spin*weight

                # Add new edge if not already present in the dictionary
                if new_problem_dict.get(new_edge) is None:
                    new_problem_dict.update({new_edge: new_weight})

                # If new edge already present, add weight
                else:
                    new_problem_dict[new_edge] += new_weight

        # Quadratic terms
        else:

            # Extract spins from term
            spin1, spin2 = term

            # Extract parent spins and associated factors
            factor_spin1, factor_spin2 = spin_map[spin1][0], spin_map[spin2][0]
            parent_spin1, parent_spin2 = spin_map[spin1][1], spin_map[spin2][1]

            # If both spins are fixed or have same parent, they only contribute with a constant term
            if parent_spin1 == parent_spin2:
                continue

            # If neither parent spin is fixed and they are different, define new quadratic term
            elif parent_spin1 is not None and parent_spin2 is not None:

                # Define new edge through the spin map and order them
                new_edge = [parent_spin1, parent_spin2]
                new_edge = tuple([min(new_edge), max(new_edge)])

                # Define new weight from factors in the spin map
                new_weight = factor_spin1*factor_spin2*weight

                # Add new edge if not already present in the dictionary
                if new_problem_dict.get(new_edge) is None:
                    new_problem_dict.update({new_edge: new_weight})

                # If new edge already present, add weight
                else:
                    new_problem_dict[new_edge] += new_weight

            # If only one spin is fixed, quadratic term becomes a bias term
            else:

                # Define new bias term keeping the unfixed spin
                new_edge = (parent_spin1,) if parent_spin2 is None else (
                    parent_spin2,)
                new_weight = factor_spin1*factor_spin2*weight

                # Add new edge if not already present in the dictionary
                if new_problem_dict.get(new_edge) is None:
                    new_problem_dict.update({new_edge: new_weight})

                # If new term already present, add weight
                else:
                    new_problem_dict[new_edge] += new_weight

    # New qubit register
    new_register = set([spin for term in new_problem_dict.keys() for spin in term])
    
    # Remove vanishing edges
    new_problem_dict = {edge:weight for edge,weight in new_problem_dict.items() if round(weight,10) != 0}

    # Define quadratic register after removing vanishing terms
    new_quadratic_register = set([spin for edge in new_problem_dict.keys() if len(edge) == 2 for spin in edge])

    # If lengths do not match, there are isolated nodes
    if len(new_register) != len(new_quadratic_register):
        isolated_nodes = new_register.difference(new_quadratic_register)
        
        # Fix isolated nodes
        for node in isolated_nodes:
            singlet = (node,)

            # If no linear term acting on the node, fix arbitrarily
            if new_problem_dict.get(singlet) is None:
                spin_map.update({node:(1,None)})

            # If linear term present, fix accordingly by anti-aligning
            else:
                factor = -np.sign(new_problem_dict.get(singlet))
                spin_map.update({node:(factor,None)})

                # Delete isolated node from new problem
                new_problem_dict.pop((node,))

    # Redefine new QUBO problem from the dictionary
    new_problem = problem_from_dict(new_problem_dict)

    return new_problem, spin_map


def final_solution(elimination_tracker: list, cl_states: list, hamiltonian: Hamiltonian):
    """
    Constructs the final solution to the problem by obtaining the final states from adding the removed 
    spins into the classical results and computing the corresponding energy.

    Parameters
    ----------
    elimination_tracker: `list`
        List of dictionaries, where each dictionary contains the elimination rules
        applied at each step of the process. Dictionary keys correspond to spin
        pairs (i,j), always with i<j to ensure proper reconstruction of the state,
        where spin j was eliminated in favor of i. The values of each pair correspond 
        to the correlation sign between spins i and j. For fixed spins, in the pair (i,j),
        i is None, and the dictionary value corresponds to the fixed state of j.
    cl_states: `list`
        Set of states as solutoins in 
    hamiltonian: `Hamiltonian`
        Hamiltonian object containing the original problem statement.

    Returns
    -------
    full_solution: `dict`
        Dictionary containing the solution states of the problem and their
        respective energies.
    """

    # Reverse max_cost_list, reverse max_pair_list
    elimination_tracker = elimination_tracker[::-1]

    # Initialize list containg full solutions
    full_solution = {}

    # Re-construct each solution
    for cl_state in cl_states:

        # Transform every string solution into binary integer
        state = [int(bit) for bit in cl_state]

        # Extract terms and costs from each elimination step
        for terms_and_stats in elimination_tracker:

            # Back track elimination from the specific step
            for term, val in terms_and_stats.items():

                # Extract qubits, by definition i<j
                i, j = term

                # If i (parent spin) is None, j is fixed to cost
                if i is None:

                    # Basis change
                    binary_val = int((1 - val)/2)
                    
                    # Insert fixed value of j
                    state.insert(j, binary_val)

                # If j unfixed, it is enslaved to i
                else:

                    # Insert new value in position j according to correlation with i
                    prev_corr = state[i]

                    state.insert(
                        j, prev_corr) if val > 0 else state.insert(j, prev_corr ^ 1)

        # Store solution states and their energy
        full_solution.update({"".join(str(i) for i in state):bitstring_energy(hamiltonian, state)})

    return full_solution
