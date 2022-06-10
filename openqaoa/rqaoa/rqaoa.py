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

from typing import Union
import numpy as np
from operator import itemgetter

from openqaoa.backends.qaoa_backend import get_qaoa_backend
from openqaoa.basebackend import QAOABaseBackend
from openqaoa.backends.qpus.qpu_auth import AccessObjectQiskit, AccessObjectPyQuil
from openqaoa.qaoa_parameters import QAOACircuitParams, QAOAVariationalBaseParams, Hamiltonian, create_qaoa_variational_params
from openqaoa.optimizers.qaoa_optimizer import get_optimizer
from openqaoa.utilities import bitstring_energy, ground_state_hamiltonian, get_mixer_hamiltonian, expectation_values

def create_access_params(qpu_params: dict):
    """
    Generates the AccessObject containing the credentials for the QPU backend.

    Parameters
    ----------
    qpu_params: `dict`
        Dictionary containing all the respective QPU attributes for backend retrieval.

    Returns
    -------
    access_params: `AccessObject`
        AccessObject containing the credentials to retrieve the QPU backend.
    """

    if 'ibmq' in qpu_params['device']:

        access_params = AccessObjectQiskit(api_token=qpu_params['token'], hub=qpu_params['hub'], group=qpu_params['group'],
                                           project=qpu_params['project'], selected_qpu=qpu_params['device'])

    # elif 'pyquil' in qpu_params['device']:

    #     access_params =  NAccessObjectPyQuil(selected_qpu=qpu_params['device'])

    else:
        raise NotImplementedError()

    return access_params


def qaoa_cost_backend(params: QAOACircuitParams, device: str, shots: int, qpu_params: dict = None):
    """
    Returns a QAOA backend initialised by the specified initialisation strategy.

    Parameters
    ----------
    params: `QAOACircuitParams`
        Parameters that define the circuit.
    device: `str`
        Specific device we want to use. Defaults to 'vectorized'.
    shots: `int`
        The number of shots if one chooses shot-based simulations.
        Defaults to None.
    qpu_params: `dict`, optional
        Dictionary containing all information regarding the QPU system
        used to run the circuit on. Defaults to None.

    Returns
    -------
    qaoa_backend: `QAOABaseBackend`
        The specified QAOA backend.
    """

    # Construct backend
    if qpu_params is None:

        qaoa_backend = get_qaoa_backend(params, device=device, n_shots=shots)

    # QPU hardware backend
    else:

        # Define access object for QPU
        access_params = create_access_params(qpu_params)

        # Obtain backend
        qaoa_backend = get_qaoa_backend(
            params, device=access_params, qubit_layout=qpu_params['qubit_layout'], n_shots=shots)

    return qaoa_backend


def optimize_qaoa(qaoa_backend: QAOABaseBackend, variational_params: QAOAVariationalBaseParams, optimizer_dict: dict):
    """
    Creates an optimizer object with the specified parameters and run the optimization process. 
    Post which obtain the results and return the ''OptimizeVQA.results_information()''

    Parameters
    ----------
    qaoa_backend: `QAOABaseBackend`
        The backend on which to run the optimization.

    variational_params: `QAOAVariationalBaseParams`
        Set of variational parameters in the QAOA ansatz.

    optimizer_dict: `dict`
        Dictionary containing the classical optimizer method and the
        number of iterations allowed.

    Returns
    -------
    opt_results:
        Results from optimization process.
    """

    optimizer = get_optimizer(qaoa_backend, variational_params, optimizer_dict)

    # Run optimization
    optimizer()

    # Obtain optimization results
    opt_results = optimizer.results_information()

    return opt_results


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

            # If maximum value is 0, all expectation values are vanishing
            if max_corr == 0:
                break

            else:
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
        print(f'All expectation values are 0 ---> Nodes are disconnected\n')

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

            # If maximum value is 0, all expectation values are vanishing
            if max_corr == 0:
                break

            else:
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

    # Flag if we have have not been able to extract any relation for the terms
    if max_terms_and_stats == {}:
        print(f'All expectation values are 0 ---> Nodes are disconnected\n')

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
        # factor *= spin_map[spin][0]
        spin = parent_spin

    return find_parent(spin_map, spin, factor)


def spin_mapping(hamiltonian: Hamiltonian, max_terms_and_stats: dict):
    """
    Generates a spin map between the original problem graph and the reduced graph.
    Elimination constraints from correlations define a constrained spin, to be removed, and a 
    parent spin,to be kept. Ultimately, parent spins determine the state of multiple spins by a 
    chain of dependencies between spins due to the different constraints. Note that there 
    is always a parent spin and only one. If cycles are present, less edges will 
    be eliminated to satisfy this requirement. Constraints following from biases in the
    Hamiltonian result in fixing spins to a specific value. In this case, the parent spin is set 
    to None.

    Additionally, it generates a different version of the max_terms_and_costs object, where
    the terms corresponds the actual elimination rules as defined in the spin map, and
    the costs correspond to the sign of the factors connecting the spin and its parent.
    The pairs in the dictionary keys are ordered as (i,j) enforcing i<j, for easy retrieval
    of the full solution at the end of the process.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object containing the problem statement.
    max_terms_and_stats: `dict`
        Dictionary containing the terms to be eliminated and their expectation values.

    Returns
    -------
    spin_map: `dict`
        Dictionary containing all the mapping dependencies. The keys
        correspond to the original spins. The values are tuple pairs,
        containing the new (parent) spin and the factor relating both.
        The structure is: { old_spin : (factor, new_spin) }.

        Spins in the map that are not eliminated (parent spins and spins 
        that are not candidates for elimination) are mapped to themselves 
        with factor 1. The rest are mapped to a parent spin with a multiplicative 
        factor resulting from the constraints, with the parent spin being None if
        the spin is fixed.

    new_max_terms_and_stats: `dict`
        Dictionary containing the terms and associated expectation values, obtained from
        the elimination relations resulting from the pair and costs in the
        original max_terms_and_costs. This dictionary is ordered in increasing
        order with respect to j in the terms (i,j) defining the dictionary 
        keys. This facilitates the backtracking of the elimination when 
        constructing the final solution at the end of the process.

    """

    register = hamiltonian.qureg
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

                # Check for cyclic dependenencies
                if parent_remove == parent_keep:

                    # If a cycle is encountered one edge is discarded
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

    # Initialize new dictionary containing max_terms_and_costs associated with elimination rules
    new_max_terms_and_stats = {}

    # Correct all dependencies and populate new_max_terms_and_costs
    for spin in spin_candidates:

        parent_spin, cumulative_factor = find_parent(spin_map, spin)
        spin_map.update({spin: (cumulative_factor, parent_spin)})

        # Ensure we do not store a (i,i) term, note that parent_spin < spin
        if spin != parent_spin:
            new_max_terms_and_stats.update(
                {(parent_spin, spin): np.sign(cumulative_factor)})

    # Order the terms for easier reconstruction of the final solution
    keys = sorted(new_max_terms_and_stats.keys(), key=itemgetter(1))
    new_max_terms_and_stats = {
        key: new_max_terms_and_stats[key] for key in keys}

    return spin_map, new_max_terms_and_stats


def hamiltonian_from_dict(hamiltonian_dict: dict):
    """
    Transforms a hamiltonian, input as a dictionary, into a classical Hamiltonian, output as
    a Hamiltonian object, ensuring proper labelling of the nodes. For example, for a set
    of nodes [0,1,4,6] with edges [(0,1),(1,4),(4,6),(0,6)], after the relabelling, the
    Hamiltonian object will be constructed with node labels [0,1,2,3] and edges 
    [(0,1),(1,2),(2,3),(1,3)]. 

    Parameters
    ----------
    hamiltonian_dict: `dict`
        Hamiltonian as a dictionary containing the edges as keys and weights as values.

    Returns
    -------
    hamiltonian: `Hamiltonian`
        A Hamiltonian object constructed using the classical_hamiltonian() method.    
    """

    edges = list(hamiltonian_dict.keys())
    weights = list(hamiltonian_dict.values())

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
    hamiltonian = Hamiltonian.classical_hamiltonian(
        terms=new_edges, coeffs=weights, constant=0)

    return hamiltonian


def redefine_hamiltonian(hamiltonian: Hamiltonian, spin_map: dict):
    """
    Returns the hamiltonian of the reduced problem. Using the spin map, we construct a  dictionary
    containing the new set of edges and weights defining the new classical Hamiltonian.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object containing the problem statement.
    spin_map: `dict`
        Dictionary containing the spin dependencies.

    Returns
    -------
    new_hamiltonian: `Hamiltonian`
        Hamiltonian object containing the reduced problem statement after spin eliminatinos.

    """

    # Define new Hamiltonian as a dictionary
    new_hamiltonian_dict = {}

    # Scan all terms and weights
    for term, weight in zip(hamiltonian.terms, hamiltonian.coeffs):

        # Bias terms
        if len(term) == 1:

            # Extract spin from the map
            spin = term.qubit_indices[0]

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
                if new_hamiltonian_dict.get(new_edge) is None:
                    new_hamiltonian_dict.update({new_edge: new_weight})

                # If new edge already present, add weight
                else:
                    new_hamiltonian_dict[new_edge] += new_weight

        # Quadratic terms
        else:

            # Extract spins from term
            spin1, spin2 = term.qubit_indices

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
                if new_hamiltonian_dict.get(new_edge) is None:
                    new_hamiltonian_dict.update({new_edge: new_weight})

                # If new edge already present, add weight
                else:
                    new_hamiltonian_dict[new_edge] += new_weight

            # If only one spin is fixed, quadratic term becomes a bias term
            else:

                # Define new bias term keeping the unfixed spin
                new_edge = (parent_spin1,) if parent_spin2 is None else (
                    parent_spin2,)
                new_weight = factor_spin1*factor_spin2*weight

                # Add new edge if not already present in the dictionary
                if new_hamiltonian_dict.get(new_edge) is None:
                    new_hamiltonian_dict.update({new_edge: new_weight})

                # If new term already present, add weight
                else:
                    new_hamiltonian_dict[new_edge] += new_weight

    # Redefine new Hamiltonian from the dictionary
    new_hamiltonian = hamiltonian_from_dict(new_hamiltonian_dict)

    return new_hamiltonian


def final_solution(max_terms_and_stats_list: list, cl_states: list, hamiltonian: Hamiltonian):
    """
    Constructs the final solution to the problem by obtaining the final states from adding the removed 
    spins into the classical results and computing the corresponding energy.

    Parameters
    ----------
    max__terms_and_stats_list: `dict`
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
    max_terms_and_stats_list = max_terms_and_stats_list[::-1]

    # Initialize list containg full solutions
    full_solution = {}

    # Re-construct each solution
    for cl_state in cl_states:

        # Transform every string solution into binary integer
        state = [int(bit) for bit in cl_state]

        # Extract terms and costs from each elimination step
        for terms_and_stats in max_terms_and_stats_list:

            # Back track elimination from the specific step
            for term, val in terms_and_stats.items():

                # Extract qubits, by definition i<j
                i, j = term

                # If i (parent spin) is None, j is fixed to cost
                if i is None:

                    # Basis change
                    state = int((1 - val)/2)

                    # Insert fixed value of j
                    state.insert(j, state)

                # If j unfixed, it is enslaved to i
                else:

                    # Insert new value in position j according to correlation with i
                    prev_corr = state[i]

                    state.insert(
                        j, prev_corr) if val > 0 else state.insert(j, prev_corr ^ 1)

        # Store solution states and their energy
        full_solution.update({"".join(str(i) for i in state):bitstring_energy(hamiltonian, state)})

    return full_solution


def adaptive_rqaoa(hamiltonian: Hamiltonian,
                   mixer: dict = {'type':'x','connectivity':None,'coeffs':None},
                   p: int = 1,
                   n_max: int = 1,
                   n_cutoff: int = 5,
                   backend: str = 'vectorized',
                   params_type: str = 'standard',
                   init_type: str = 'ramp',
                   shots: int = None,
                   optimizer_dict: dict = {'method': 'cobyla', 'maxiter': 200},
                   qpu_params: dict = None,
                   max_terms_and_stats_list: list = None,
                   original_hamiltonian: Hamiltonian = None
                   ):
    """
    Performs the Recursive QAOA algorithm, with the number of spins to be 
    eliminated at each step given by a statistical criterion. The top 
    n_max+1 expectation values correlations are considered and an average is performed. The
    algorithm eliminates all the terms ranking above average. Thus, n_max
    is the maximum number of possible eliminated spins.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object containing the problem statement.
    mixer: `dict`, optional
        Dictionary containing mixer properties: type, connectivity and coeffs.
    p: `int`, optional
        Number of QAOA layers in each run. Defaults to 1.
    n_max: `int`, optional
        Maximum number of spins that can be eliminated at each step. Defaults to 1.
    n_cutoff: `int`, optional
        Cut-off value of spins at which the system is solved classically. 
        Defaults to 5.
    backend: `str`, optional
        Backend to be used during QAOA run. Defaults to 'vectorized'.     
    params_type: `str`, optional
        Parametrization to be used during QAOA run. Defaults to 'standard'.   
    init_type: `str`, optional
        Parametrization to be used during QAOA run. Defaults to 'ramp'.
    shots: `int`, optional
        The number of shots if one chooses shot-based simulations.
        Defaults to None.  
    optimizer_dict: `dict`, optional
        Specifications on the optimizer to be used during QAOA run.
        Method defaults to 'cobyla'. Maximum iterations default to 200.
    qpu_params: `dict`, optional
        Dictionary containing all information regarding the QPU system
        used to run the circuit on. Defaults to None.             
    max_terms_and_stats_list: `list`, optional
        List tracking the terms with with highest expectation value and the
        associated expectation value. Defaults to None.
    original_hamiltonian: `Hamiltonian`, optional
        Hamiltonian associated with the original problem. Defaults to None
        and it is stored in the first step of the process.   
    Returns
    -------
    rqaoa_info: `dict`
        Dictionary containing all the information about the RQAOA run: the
        solution states and energies (key: 'solution'), the output of the classical 
        solver (key: 'classical output'), the elimination rules for each step
        (key: 'elimination rules'), the number of eliminations at each step (key: 'schedule') 
        and total number of steps (key: 'total steps').
    """

    # Define number of spins
    n_qubits = hamiltonian.n_qubits

    # Define mixer Hamiltonian based on input type
    if mixer.get('connectivity') is None or type(mixer.get('connectivity')) == str:
        mixer_hamiltonian = get_mixer_hamiltonian(n_qubits,mixer.get('type'),mixer.get('connectivity'),mixer.get('coeffs'))
    else:
        raise NotImplementedError(f'Custom mixer connectivities are not currently supported')


    # Store original hamiltonian
    if original_hamiltonian is None:
        original_hamiltonian = hamiltonian

    # Initialize tracker
    if max_terms_and_stats_list is None:
        max_terms_and_stats_list = []

    # Ensure we do not eliminate beyond the cutoff
    if (n_qubits - n_cutoff) < n_max:
        n_max = n_qubits - n_cutoff

    # If below cutoff, proceed classically
    if n_qubits <= n_cutoff:

        # Solve the problem classically
        cl_energy, cl_ground_states = ground_state_hamiltonian(hamiltonian)

        # Extract optimal solutions and their costs
        classical_sol_dict = {'minimum energy': cl_energy,
                              'optimal states': cl_ground_states}

        # Retrieve full solutions including eliminated spins and their energies
        full_solutions = final_solution(
            max_terms_and_stats_list, cl_ground_states, original_hamiltonian)

        # Compute description dictionary containing all the information
        rqaoa_info = {}
        rqaoa_info['solution'] = full_solutions
        rqaoa_info['classical output'] = classical_sol_dict
        rqaoa_info['elimination rules'] = max_terms_and_stats_list
        rqaoa_info['schedule'] = [len(max_tc) for max_tc in max_terms_and_stats_list]
        rqaoa_info['total steps'] = len(max_terms_and_stats_list)

        # Return classical solution
        return rqaoa_info

    # If above cutoff, proceed quantumly
    else:

        # Define circuit parameters
        circuit_params = QAOACircuitParams(hamiltonian, mixer_hamiltonian, p=p)

        # Define variational parameters
        variational_params = create_qaoa_variational_params(
            circuit_params, params_type, init_type)

        # Retrieve backend
        qaoa_backend = qaoa_cost_backend(
            circuit_params, backend, shots, qpu_params)

        # Run QAOA
        qaoa_results = optimize_qaoa(
            qaoa_backend, variational_params, optimizer_dict)

        # Obtain statistical results
        exp_vals_z, corr_matrix = expectation_values(
            variational_params, qaoa_results, qaoa_backend, hamiltonian, mixer_type = mixer['type'], p = p)

        # Retrieve highest expectation values according to adaptive method
        max_terms_and_stats = ada_max_terms(exp_vals_z, corr_matrix, n_max)

        # If no expectation values were obtained, singlets are correlations are equal to 0 and leftover spins are disconnected
        if max_terms_and_stats == {}:

            # Using leftover freedom we fix all spins up to cutoff value
            for j in range(n_qubits-n_cutoff):

                # We arbitrarily choose to fix the last spins with value 1
                max_terms_and_stats.update({(j + n_cutoff,): 1})

        # Generate spin map and redefine list of max_terms_and_costs according to elimination rules
        spin_map, final_max_terms_and_stats = spin_mapping(
            hamiltonian, max_terms_and_stats)
        max_terms_and_stats_list.append(final_max_terms_and_stats)

        # Eliminate spins and redefine cost hamiltonian
        new_hamiltonian = redefine_hamiltonian(hamiltonian, spin_map)

        # Restart process with new parameters
        return adaptive_rqaoa(new_hamiltonian, mixer, p, n_max, n_cutoff, backend, params_type, init_type, shots,
                              optimizer_dict, qpu_params, max_terms_and_stats_list, original_hamiltonian)


def custom_rqaoa(hamiltonian: Hamiltonian,
                 mixer: dict = {'type':'x','connectivity':None,'coeffs':None},
                 p: int = 1,
                 n_cutoff: int = 5,
                 steps: Union[list,int] = 1,
                 backend: str = 'vectorized',
                 params_type: str = 'standard',
                 init_type: str = 'ramp',
                 shots: int = None,
                 optimizer_dict: dict = {'method': 'cobyla', 'maxiter': 200},
                 qpu_params: dict = None,
                 max_terms_and_stats_list: list = None,
                 original_hamiltonian: Hamiltonian = None,
                 counter=None
                 ):
    """
    Performs the Recursive QAOA algorithm, with the number of spins to be 
    eliminated at each step given as an input. Is a single value is passed, this
    value is used as the number of spins to be eliminated at each step.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object containing the problem statement.
    mixer: `dict`, optional
        Dictionary containing mixer properties: type, connectivity and coeffs.
    p: `int`, optional
        Number of QAOA layers in each run. Defaults to 1.
    steps: `list` or `int`, optional
        The custom list of spin eliminations at each step. If a single
        value is passed, at every step the same amount of spins will
        be eliminated.
    n_cutoff: `int`, optional
        Cut-off value of spins at which the system is solved classically. 
        Defaults to 5.
    backend: `str`, optional
        Backend to be used during QAOA run. Defaults to 'vectorized'.
    params_type: `str`, optional
        Parametrization to be used during QAOA run. Defaults to 'standard'.
    init_type: `str`, optional
        Parametrization to be used during QAOA run. Defaults to 'ramp'.
    shots: `int`, optional
        The number of shots if one chooses shot-based simulations.
        Defaults to None.
    optimizer_dict: `dict`, optional
        Specifications on the optimizer to be used during QAOA run.
        Method defaults to 'cobyla'. Maximum iterations default to 200.
    qpu_params: `dict`, optional
        Dictionary containing all information regarding the QPU system
        used to run the circuit on. Defaults to None.
    max_terms_and_stats_list: 
        List tracking the terms with with highest expectation value and the
        associated expectation value. Defaults to None.
    original_hamiltonian: `Hamiltonian`, optional
        Hamiltonian associated with the original problem. Defaults to None
        and it is stored in the first step of the process.

    Returns
    -------
    rqaoa_info: `dict`
        Dictionary containing all the information about the RQAOA run: the
        solution states and energies (key: 'solution'), the output of the classical 
        solver (key: 'classical output'), the elimination rules for each step
        (key: 'elimination rules'), the number of eliminations at each step (key: 'schedule') 
        and total number of steps (key: 'total steps').
    """

    # Define number of spins
    n_qubits = hamiltonian.n_qubits

    # Define mixer Hamiltonian based on input type
    if mixer.get('connectivity') is None or type(mixer['connectivity']) == str:
        mixer_hamiltonian =  get_mixer_hamiltonian(n_qubits,mixer.get('type'),mixer.get('connectivity'),mixer.get('coeffs'))
    else:
        raise NotImplementedError(f'Custom mixer connectivities are not currently supported')

    # Store original hamiltonian
    if original_hamiltonian is None:
        original_hamiltonian = hamiltonian

    # Initialize tracker
    if max_terms_and_stats_list is None:
        max_terms_and_stats_list = []

    # If schedule is not given one spin is eliminated at a time
    if type(steps) is int:
        steps = [steps]*(n_qubits-n_cutoff)

    # Initialize counter check schedule
    if counter is None:
        counter = 0

        # Ensure there are enough steps in the schedule
        assert np.abs(n_qubits - n_cutoff) <= sum(steps),\
            f"Schedule is incomplete, add {np.abs(n_qubits - n_cutoff) - sum(steps)} more eliminations"

    # If below cutoff, proceed classically
    if n_qubits <= n_cutoff:

        # Solve the problem classically
        cl_energy, cl_ground_states = ground_state_hamiltonian(hamiltonian)

        # Extract optimal solutions and their costs
        classical_sol_dict = {'minimum energy': cl_energy,
                              'optimal states': cl_ground_states}

        # Retrieve full solutions including eliminated spins and their energies
        full_solutions = final_solution(
            max_terms_and_stats_list, cl_ground_states, original_hamiltonian)

        # Compute description dictionary containing all the information
        rqaoa_info = {}
        rqaoa_info['solution'] = full_solutions
        rqaoa_info['classical output'] = classical_sol_dict
        rqaoa_info['elimination rules'] = max_terms_and_stats_list
        rqaoa_info['schedule'] = [len(max_tc)
                                  for max_tc in max_terms_and_stats_list]
        rqaoa_info['total steps'] = len(max_terms_and_stats_list)

        # Return classical solution
        return rqaoa_info

    # If above cutoff, proceed quantumly
    else:

        # Number of spins to eliminate according the schedule
        n_elim = steps[counter]

        # If the step eliminates more spins than available, reduce step to match cutoff
        if n_qubits - n_elim <= 0:
            n_elim = n_qubits - n_cutoff

        # Define circuit parameters
        circuit_params = QAOACircuitParams(hamiltonian, mixer_hamiltonian, p=p)

        # Define variational parameters
        variational_params = create_qaoa_variational_params(
            circuit_params, params_type, init_type)

        # Retrieve backend
        qaoa_backend = qaoa_cost_backend(
            circuit_params, backend, shots, qpu_params)

        # Run QAOA
        qaoa_results = optimize_qaoa(
            qaoa_backend, variational_params, optimizer_dict)

        # Obtain statistical results
        exp_vals_z, corr_matrix = expectation_values(
            variational_params, qaoa_results, qaoa_backend, hamiltonian, mixer_type = mixer['type'], p = p)

        # Retrieve highest expectation values
        max_terms_and_stats = max_terms(exp_vals_z, corr_matrix, n_elim)

        # If no expectation values were obtained, singlets are correlations are equal to 0 and leftover spins are disconnected
        if max_terms_and_stats == {}:

            # Using leftover freedom we fix all spins up to cutoff value
            for j in range(n_qubits-n_cutoff):

                # We arbitrarily choose to fix the last spins with value 1
                max_terms_and_stats.update({(j + n_cutoff,): 1})

        # Generate spin map and redefine list of max_terms_and_costs according to elimination rules
        spin_map, final_max_terms_and_stats = spin_mapping(
            hamiltonian, max_terms_and_stats)
        max_terms_and_stats_list.append(final_max_terms_and_stats)

        # Eliminate spins and redefine hamiltonian
        new_hamiltonian = redefine_hamiltonian(hamiltonian, spin_map)

        # Add one step to the counter
        counter += 1

        # Restart process with new parameters
        return custom_rqaoa(new_hamiltonian, mixer, p, n_cutoff, steps, backend, params_type, init_type, shots,
                            optimizer_dict, qpu_params, max_terms_and_stats_list, original_hamiltonian, counter)

# END
