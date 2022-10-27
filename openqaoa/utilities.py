#   Copyright 2021 Entropica Labs
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

"""
Utility and convenience functions for a number of QAOA applications.
"""

from typing import Optional, Union, List, Tuple
import itertools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from .qaoa_parameters import Hamiltonian, PauliOp, QAOAVariationalBaseParams


def X_mixer_hamiltonian(n_qubits: int,
                        coeffs: List[float] = None) -> Hamiltonian:
    """Construct a Hamiltonian object to implement the X mixer.

    Parameters
    ----------
    n_qubits: `int`
        The number of qubits in the mixer Hamiltonian.
    coeffs: `list`, optional
        The coefficients of the X terms in the Hamiltonian.

    Returns
    -------
    hamiltonian: `Hamiltonian`
        The Hamiltonian object corresponding to the X mixer.
    """
    # If no coefficients provided, set all to -1
    coeffs = [-1]*n_qubits if coeffs is None else coeffs

    # Initialize list of terms
    terms = []

    # Generate terms in the X mixer
    for i in range(n_qubits):
        terms.append(PauliOp.X(i))
    
    # Define mixer Hamiltonian
    hamiltonian = Hamiltonian(pauli_terms=terms, coeffs=coeffs, constant=0)

    return hamiltonian


def XY_mixer_hamiltonian(n_qubits: int,
                         qubit_connectivity: Union[List[list],
                                                   List[tuple], str] = 'full',
                         coeffs: List[float] = None) -> Hamiltonian:
    """
    Construct a Hamiltonian object to implement the XY mixer.

    Parameters
    ----------
    n_qubits: `int`
        The number of qubits in the system.
    qubit_connectivity: `Union[List[list],List[tuple], str]`, optional
        The connectivity of the qubits in the mixer Hamiltonian.
    coeffs: `list`, optional
        The coefficients of the XY terms in the Hamiltonian.

    Returns
    -------
    hamiltonian: Hamiltonian
        The Hamiltonian object corresponding to the XY mixer.

    Notes
    -----
        The XY mixer is not implemented with $$RXY$$ Gates, but with $$H_{XY} = \\frac{1}{2}(\\sum_{i,j} X_iX_j+Y_iY_j)$$
    """
    # Set of topologies supported by default
    connectivity_topology_dict = {'full': list(itertools.combinations(range(n_qubits), 2)),
                                  'chain': [(i, i+1) for i in range(n_qubits-1)],
                                  'star': [(0, i+1) for i in range(n_qubits-1)]}

    # Check if input connectivity is a default value
    if isinstance(qubit_connectivity, str):
        try:
            # Define qubit connectivity from default value
            qubit_connectivity = connectivity_topology_dict[qubit_connectivity]
        except KeyError:
            raise ValueError(
                f'Please choose connection topology from {list(connectivity_topology_dict.keys())}')

    # Define connectivty according to user input
    else:
        # Extract indices from connectivity
        indices = set([qubit for term in qubit_connectivity for qubit in term])

        # Ensure all indices are defined within the range of number of qubits
        assert max(indices) <= n_qubits - \
            1, 'Qubit index in connectivity list is out of range'
        assert min(indices) >= 0, 'Qubit index should be a positive integer'

    # If no coefficients provided, set all to the number of terms
    coeffs = [0.5]*2*len(qubit_connectivity) if coeffs is None else coeffs

    # Initialize list of terms
    terms = []

    # Generate terms in the XY mixer
    for pair in qubit_connectivity:
        i, j = pair
        terms.append(PauliOp.X(i)@PauliOp.X(j))
        terms.append(PauliOp.Y(i)@PauliOp.Y(j))

    # Define mixer Hamiltonian
    hamiltonian = Hamiltonian(pauli_terms=terms, coeffs=coeffs, constant=0)

    return hamiltonian


def get_mixer_hamiltonian(n_qubits: int, mixer_type: str = 'x', qubit_connectivity: Union[List[list],List[tuple], str] = None, coeffs: List[float] = None):
    """
    Parameters
    ----------
    n_qubits: `int`
        Number of qubits in the Hamiltonian.
    mixer_type: `str`, optional
        Name of the mixer Hamiltonian. Choose from `x` or `xy`.
    qubit_connectivity: `list` or `str`, optional
        The connectivity of the qubits in the mixer Hamiltonian.
    coeffs: `list`, optional
        The coefficients of the terms in the Hamiltonian.

    Returns:
    --------
    mixer: `Hamiltonian`
        Hamiltonian object containing the specificied mixer.
    """

    # Return mixer Hamiltonian according to specified type
    if mixer_type == 'x':
        mixer = X_mixer_hamiltonian(n_qubits, coeffs)
    else:
        mixer = XY_mixer_hamiltonian(n_qubits, qubit_connectivity, coeffs)

    return mixer


################################################################################
# METHODS FOR PRINTING HAMILTONIANS AND GRAPHS, AND PRINTING ONE FROM EACH OTHER
################################################################################


def graph_from_hamiltonian(hamiltonian: Hamiltonian) -> nx.Graph:
    """
    Creates a networkx graph corresponding to a specified problem Hamiltonian.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        The Hamiltonian of interest. Must be specified a Hamiltonian object.

    Returns
    -------
    G: `Networkx Graph`
        The corresponding networkx graph with the edge weights being the
        two-qubit coupling coefficients,
        and the node weights being the single-qubit bias terms.

    Notes
    -----
    This function cannot handle non-QUBO terms.
    Linear terms are stored as nodes with weights.
    """
    # Define graph
    G = nx.Graph()

    # Add nodes for each qubit in the register
    for qubit in hamiltonian.qureg:
        G.add_node(qubit, weight=0)

    # Add each term to the graph as an attribute
    for term, weight in zip(hamiltonian.terms, hamiltonian.coeffs):

        # Extract indices from Pauli term
        term_tuple = term.qubit_indices

        # If term is linear add as a node with a weight attribute
        if(len(term) == 1):
            G.add_node(term_tuple[0], weight=weight)

        # If term is quadratic add as an edge with a weight attribute
        elif(len(term) == 2):
            G.add_edge(term_tuple[0], term_tuple[1], weight=weight)

    return G


def hamiltonian_from_graph(G: nx.Graph) -> Hamiltonian:
    """
    Builds a cost Hamiltonian as a collection of PauliOp objects
    from a specified networkx graph, extracting any node and edge weights.

    Parameters
    ----------
    G: `Networkx Graph`
        The specified networkx graph.

    Returns
    -------
    hamiltonian: `Hamiltonian`
        The Hamiltonian object constructed from the specified graph.
    """
    # Node bias terms
    nodes_info = nx.get_node_attributes(G, 'weight')
    singlet_terms = [(node,) for node in nodes_info.keys()]
    singlet_coeffs = list(nodes_info.values())

    # Edge terms
    edges_info = nx.get_edge_attributes(G, 'weight')
    pair_terms = list(edges_info.keys())
    pair_coeffs = list(edges_info.values())

    # Collect all terms and coefficients
    terms = singlet_terms + pair_terms
    coeffs = singlet_coeffs + pair_coeffs
    
    # Define Hamiltonian
    hamiltonian = Hamiltonian.classical_hamiltonian(
        terms=terms, coeffs=coeffs, constant=0)

    return hamiltonian


def random_k_regular_graph(degree: int,
                           nodes: List[int],
                           seed: int = None,
                           weighted: bool = False,
                           biases: bool = False) -> nx.Graph:
    """
    Produces a random graph with specified number of nodes, each having degree k.

    Parameters
    ----------
    degree: `int`
        Desired degree for the nodes.
    nodes: `list`
        The node set of the graph.
    seed: `int`, optional
        A seed for the random number generator.
    weighted: `bool`, optional
        Whether the edge weights should be uniform or different. If false, all weights are set to 1.
        If true, the weight is set to a random number drawn from the uniform distribution in the
        interval 0 to 1.
    biases: `bool`, optional
        Whether or not the graph nodes should be assigned a weight.
        If true, the weight is set to a random number drawn from the uniform
        distribution in the interval 0 to 1.

    Returns
    -------
    nx.Graph: `Networkx Graph`
        A graph with the properties as specified.
    """
    # Set numpy seed
    np.random.seed(seed=seed)

    # Create a random regular graph on the nodes
    G = nx.random_regular_graph(degree, len(nodes), seed)

    # Relabel nodes
    nx.relabel_nodes(G, {i: n for i, n in enumerate(nodes)})

    # Add edges between nodes
    for edge in G.edges():

        # If weighted attribute is False, all weights are set to 1
        if not weighted:
            G[edge[0]][edge[1]]['weight'] = 1

        # If weighted attribute is True, weights are assigned as random integers
        else:
            G[edge[0]][edge[1]]['weight'] = np.random.rand()

    # If biases attribute is True, add node weights as random integers
    if biases:
        for node in G.nodes():
            G.nodes[node]['weight'] = np.random.rand()

    return G

def plot_graph(G: nx.Graph, ax=None, colormap='seismic') -> None:
    """
    Plots a networkx graph.

    Parameters
    ----------
    G: `Networkx Graph`
        The networkx graph of interest.
    ax: `Matplotlib axes object`, optional
        Matplotlib axes to plot on. Defaults to None.
    colormap: `str`, optional
        Colormap to use for plotting. Defaults to 'seismic'.
    """
    
    # Create plot figure
    fig = plt.figure(figsize=(10, 6))
        
    # Extract all graph attributes
    biases_and_nodes = nx.get_node_attributes(G, 'weight')
    biases = list(biases_and_nodes.values())
    edges_and_weights = nx.get_edge_attributes(G, 'weight')
    pos = nx.shell_layout(G)

    # extract minimum and maximum weights for side bar limits
    weights = list(edges_and_weights.values())
    # Define color map
    cmap = plt.cm.get_cmap(colormap)
    
    if len(set(weights)) > 1:
        edge_vmin = min(weights)
        edge_vmax = max(weights)


        # Define normalized color map
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax))
        # Add colormap to plot
        cbar = plt.colorbar(sm,pad=0.08)
        cbar.ax.set_ylabel('Edge Weights', rotation=270,labelpad=15)
    else:
        weights = [1] * len(G.edges())
        edge_vmin = None
        edge_vmax = None
        cmap = None    
    
    # If biases are present define reference values and color map for side bar
    if len(set(biases)) > 1:
        cmap = plt.cm.get_cmap(colormap)
        vmin = min(biases)
        vmax = max(biases)
        sm2 = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar2 = plt.colorbar(sm2, location='left')
        cbar2.ax.set_ylabel('Single Qubit Biases', rotation=90)

        # Draw graph
        nx.draw(G, pos, node_size=500, node_color=biases, edge_color=weights, width=2.5, cmap=cmap,
                edge_cmap=cmap, vmin=vmin, vmax=vmax, edge_vmin=edge_vmin,
                edge_vmax=edge_vmax, with_labels=True)

    else:
        # Draw graph
        nx.draw(G, pos, node_size=500, edge_color=weights, width=2.5,
                edge_cmap=cmap, edge_vmin=edge_vmin,
                edge_vmax=edge_vmax, with_labels=True)
    
    # Show plot
    
    plt.show
    return None

def random_classical_hamiltonian(reg: List[int],
                                 seed: int = None,
                                 weighted: bool = True,
                                 biases: bool = True,
                                 constant: int = 0) -> Hamiltonian:
    """
    Creates a random classical cost hamiltonian.

    Parameters
    ----------
    reg: `list`
        Register to build the hamiltonian on.
    seed: `int`, optional
        A seed for the random number generator. Defaults to None.
    weighted: `bool`, optional
        Whether the edge weights should be uniform or different. If false, all
        weights are set to 1. If true, the weight is set to a random number 
        drawn from the uniform distribution in the interval 0 to 1. Defaults to
        True.
    biases: `bool`, optional
        Whether or not the graph nodes should be assigned a weight.
        If true, the weight is set to a random number drawn from the uniform
        distribution in the interval 0 to 1. Defaults to True.
    constant: `int`, optional
        The constant term in the Hamiltonian. Defaults to 0.

    Returns
    -------
    random_hamil: `Hamiltonian`
        A random hamiltonian with randomly selected terms and coefficients and 
        with the specified constant term.

    Notes
    -----
    Randomly selects which qubits that will have a bias term, then assigns them a bias coefficient.
    Randomly selects which qubit pairs will have a coupling term, then assigns them a coupling coefficient.
    In both cases, the random coefficient is drawn from the uniform distribution on the interval [0,1).
    """
    # Set the random seed
    np.random.seed(seed=seed)

    # Initialize terms and weights
    terms = []
    weights = []

    # If biases attribute is True, add lineat terms
    if biases:
        # Choose a random set of qubits to add linear terms
        n_biases = np.random.randint(len(reg))
        bias_qubits = np.random.choice(reg, n_biases)

        # Generate coefficients for linear terms
        bias_coeffs = np.random.rand(
            n_biases) if weighted else np.ones(n_biases)

        # Store linear terms and coefficients
        for qubit, coeff in zip(bias_qubits, bias_coeffs):
            terms.append([qubit])
            weights.append(coeff)
    
    # Generate quiadratic terms, scanning all possible combinations
    for q1, q2 in itertools.combinations(reg, 2):

        # Choose at random to couple terms
        are_coupled = np.random.randint(2)

        # If coupled, generate coefficients and store along with term
        if are_coupled:
            couple_coeff = np.random.rand() if weighted else 1
            terms.append([q1, q2])
            weights.append(couple_coeff)

    # Ensure each term has an associated weight
    assert len(terms) == len(
        weights), "Each term should have an associated weight"

    # Define classical Hamiltonian
    hamiltonian = Hamiltonian.classical_hamiltonian(
        terms, weights, constant=constant)

    return hamiltonian


################################################################################
# HAMILTONIANS AND DATA
################################################################################


def ground_state_hamiltonian(hamiltonian: Hamiltonian, bounded = True) -> Tuple[float, list]:
    """
    Computes the exact ground state and ground state energy of a classical Hamiltonian. Uses standard numpy module.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object whose ground state properties are computed.
    bounded: `bool`, optional
        If set to True, the function will not perform computations for qubit
        numbers above 25. If False, the user can specify any number. Defaults
        to True.

    Returns
    -------
    min_energy: `float`
        The minimum eigenvalue of the cost Hamiltonian.

    config: `np.array`
        The minimum energy eigenvector as a binary array
        configuration: qubit-0 as the first element in the sequence.
    """
    # Extract number of qubits 
    n_qubits = hamiltonian.n_qubits
    
    # If number of qubits is too high warn the user
    if bounded and n_qubits > 25:
        raise ValueError("The number of qubits is too high, computation could take a long time. If still want to proceed set argument `bounded` to False")

    # Generate qubit register
    register = range(n_qubits)

    # Intialize energies
    energies = np.zeros(2**len(register))

    # Obtain spectrum, scanning term by term
    for i, term in enumerate(hamiltonian.terms):

        # Extract coefficient
        out = np.real(hamiltonian.coeffs[i])

        # Compute tensor product
        for qubit in register:
            if qubit in term.qubit_indices:
                out = np.kron([1, -1], out)
            else:
                out = np.kron([1, 1], out)

        # Add energy contribution of the term
        energies += out
    
    # Add constant term to the spectrum
    energies += hamiltonian.constant

    # Extract minimum energy
    min_energy = np.min(energies)

    # Extract indices of minimum energies
    indices = np.where(energies == min_energy)[0]

    # Generate ground states
    config_strings = [np.binary_repr(index, len(register))[::-1]
                      for index in indices]

    return min_energy, config_strings


def bitstring_energy(hamiltonian: Hamiltonian,
                     bitstring: Union[List[int], str]) -> float:
    """
    Computes the energy of a given bitstring with respect to a classical cost Hamiltonian.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object determining the energy levels.
    bitstring : `list` or `str`
        A list of integers 0 and 1, or a string, representing a configuration.

    Returns
    -------
    energy: `float`
        The energy of the given bitstring with respect to the cost Hamiltonian.
    """
    # Initialize energy value
    energy = 0
    
    # Compute energy contribution term by term
    for i, term in enumerate(hamiltonian.terms):

        # Compute sign of spin interaction term
        variables_product = np.prod([(-1)**int(bitstring[k]) for k in term.qubit_indices])

        # Add energy contribution
        energy += hamiltonian.coeffs[i]*variables_product
    
    # Add constant contribution
    energy += hamiltonian.constant

    return energy


def energy_expectation(hamiltonian: Hamiltonian,
                       measurement_counts: dict) -> float:
    """
    Computes the energy expectation value from a set of measurement counts, with respect to a classical cost Hamiltonian.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object determining the energy levels.
    measurement_counts : `dict`
        Measurement counts dictionary for which to calculate energy expectation.

    Returns
    -------
    energy: `float`
        The energy expectation value for the set of measurement outcomes
    """

    # Starting value for the energy
    energy = 0

    # Number of measurement shots
    shots = sum(measurement_counts.values())

    # Compute average energy adding one by one the contribution from each state
    for state, prob in measurement_counts.items():

        # Number of ones (spins pointing down) from the specific configuration for each Hamiltonian term
        num_ones_list = [sum([int(state[i]) for i in term.qubit_indices])
                         for term in hamiltonian.terms]

        # Compute configuration energy
        config_energy = sum([hamiltonian.coeffs[i] if num_ones % 2 == 0 else -1*hamiltonian.coeffs[i]
                            for i, num_ones in enumerate(num_ones_list)])

        # Add contribution to total energy
        energy += prob*config_energy

    # Normalize with respect to the number of shots
    energy *= (1/shots)

    return energy


def energy_spectrum_hamiltonian(hamiltonian: Hamiltonian) -> np.ndarray:
    """
    Computes exactly the energy spectrum of the hamiltonian defined by terms
    and weights and its corresponding configuration of variables. Uses 
    standard numpy module.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object whose spectrum is computed.

    Returns
    -------
    energies: `np.ndarray`
        The energy spectra of the given hamiltonian
    """
    # Extract number of qubits
    n_qubits = hamiltonian.n_qubits

    # Define qubit register
    register = range(n_qubits)

    # Intialize energies
    energies = np.zeros((2**len(register)))

    # Obtain spectrum, scanning term by term
    for i, term in enumerate(hamiltonian.terms):

        # Extract coefficients
        out = np.real(hamiltonian.coeffs[i])

        # Compute tensor product 
        for qubit in register:
            if qubit in term.qubit_indices:
                out = np.kron([1, -1], out)
            else:
                out = np.kron([1, 1], out)

        # Add energy contribution of the term
        energies += out

    # Add constant term to the spectrum
    energies = energies + hamiltonian.constant

    return energies


def plot_energy_spectrum(hamiltonian: Hamiltonian,
                         high_k_states: Optional[int] = None,
                         low_k_states: Optional[int] = None,
                         ax=None,
                         cmap='winter') -> None:
    """
    Compute and plot the energy spectrum of a given hamiltonian on
    a matplotlib figure.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object whose spectrum is computed.
    high_k_states: `int`, optional
        Optionally plot the highest k energy levels. Defaults to None.
    low_k_states: `int`, optional
        Optionally, plot the lowest k energy levels. Defaults to None.
    ax: Matplotlib axes object, optional
        Axes to plot on. Defaults to None.
    cmap: `str`, optional
        Specify the matplotlib colormap to use for plotting. Defaults to 'winter'.
    """
    # Compute energy spectrum
    energies = energy_spectrum_hamiltonian(hamiltonian)

    # Extract energy levels and their degeneracy
    unique_energies, degeneracy = np.unique(energies, return_counts=True)

    # If required extract highest or lowest k energy levels
    if high_k_states is not None:
        unique_energies, degeneracy = unique_energies[-high_k_states:], degeneracy[-high_k_states:]
    elif low_k_states is not None:
        unique_energies, degeneracy = unique_energies[:low_k_states], degeneracy[:low_k_states]

    # Define colormap
    cmap = plt.cm.get_cmap(cmap, len(unique_energies))

    # If no axis provided, define figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 5))

    # Plot energy levels
    for i, energy in enumerate(unique_energies):
        ax.axhline(energy, label=f'Degeneracy={degeneracy[i]}', color=cmap(i))

    # Set axis attributes and legend
    ax.set(xticks=[], yticks=unique_energies, ylabel='Energy[a.u.]',
           title='Hamiltonian Energy spectrum')
    ax.legend(loc='center left', fontsize=8)

    return None


def low_energy_states(hamiltonian: Hamiltonian,
                      threshold_per: float) -> Tuple[float, list]:
    """
    Return threshold energy and the low energy states of the
    specified hamiltonian which are below this threshold. 

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Compute the low energy states of this Hamiltonian
    threshold_per: `float`
        Threshold percentage away from the ground state, defining the energy we window we search
        in for low energy states.

    Returns
    -------
    low_energy_threshold: `float`
        The energy threshold below which we retrieve states. 
    states: `list`
        The list of low energy states that lie below the low
        energy threshold.

    Notes
    -----
    The threshold is calculated as `threshols_per` factors away from the
    ground state of the Hamiltonian.
    """
    # Asserr threshold is bounded between 0 and 1
    assert threshold_per >= 0.0, "Threshold percentage should be above 0"
    assert threshold_per <= 1.0, "Threshold percentage should be below 1"

    # Compute energy spectrum of the Hamiltonian
    energies = energy_spectrum_hamiltonian(hamiltonian)

    # Extract ground state and highest excited state
    ground_state_energy = np.min(energies)
    highest_state_energy = np.max(energies)

    # Compute the low energy threshols
    low_energy_threshold = ground_state_energy + threshold_per * \
        np.abs(highest_state_energy-ground_state_energy)

    # Initilize inndices for low energy states
    low_energy_indices = []

    # Obtain indices for energies below the threshold
    for idx in range(len(energies)):
        if energies[idx] <= low_energy_threshold:
            low_energy_indices.append(idx)
    
    # Extract states from the Hamiltonian spectrum
    states = [np.binary_repr(index, hamiltonian.n_qubits)[::-1]
              for index in low_energy_indices]

    return low_energy_threshold, states


def low_energy_states_overlap(hamiltonian: Hamiltonian,
                              threshold_per: float,
                              prob_dict: dict) -> float:
    """
    Calculates the overlap between the low energy states of a Hamiltonian,
    below a specific threshold away from the ground state energy, and an input
    state, expressed in terms of a probability dictionary. 

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Compute overlap with respect to energies of this Hamiltonian
    threshold_per: `float`
        Threshold percentage away from the ground state, defining the energy we window we search
        in for low energy states.
    prob_dict: `Dict`
        The measurement outcome dictionary generated from the 
        circuit execution.

    Returns
    -------
    total_overlap: `float`
        The total overlap with the low-energy states.

    Notes
    -----
    The threshold is calculated as `threshold_per` factors away from the ground state of the Hamiltonain.
    For `threshold_per=0` the function returns the ground state overlap of the QAOA output.
    """
    # Extract number of qubits from probability dictionary
    n_qubits = len(list(prob_dict.keys())[0])

    # Ensure number of qubits matches the number of qubits registered in the Hamiltonian
    assert n_qubits == hamiltonian.n_qubits, "Number of qubits in the Hamiltonian does not match the probabilities specified"

    # Extract low energy states
    _, states = low_energy_states(hamiltonian, threshold_per)

    # Compute overlap
    total_overlap = sum([prob_dict[state]
                        for state in states])/sum(list(prob_dict.values()))

    return total_overlap


def exp_val_single(spin: int, prob_dict: dict):
    """
    Computes expectation value <Z> of a given spin.

    Parameters
    ----------
    spin: `int`
        Spin whose expectation value we compute.
    prob_dict: `dict`
        Dictionary containing the configuration probabilities of each spin.

    Returns
    -------
    exp_val: `float`
        Expectation value of the spin
    """

    # Initialize expectation value
    exp_val = 0
    norm = sum(prob_dict.values())
    
    # Compute correlation
    for bitstring, prob in prob_dict.items():

        # If 0, spin is pointing up, else it is pointing down
        Z = int(bitstring[spin])

        # Add contribution if spin points up or subtract if points down
        exp_val += -prob/norm if Z > 0 else prob/norm

    return exp_val


def exp_val_pair(spins: tuple, prob_dict: dict):
    """
    Computes the correlation Mij = <Z_{i}Z_{j}> between qubits i,j using the QAOA optimized 
    wavefunction.

    NOTE: In the presence of linear terms the <Z_{i}><Z_{j}> contribution needs to be
    subtracted later. This is done in the exp_val_hamiltonian_termwise() function used as a 
    wrapper for this function. 

    Parameters
    ----------
    spins: `tuple`
        Tuple containing the spins whose correlation is computed.
    prob_dict: `dict`
        The dictionary containing the configuration probabilities of each spin.

    Returns
    -------
    corr: `float`
        Correlation between the two spins in the term.

    """

    # Initialize correlation
    corr = 0
    
    norm = sum(prob_dict.values())
    # Compute correlation
    for bitstring, prob in prob_dict.items():

        # If 0 or 2, spins are aligned, else they are anti-aligned
        num_ones = sum([int(bitstring[i]) for i in spins])

        # Add contribution if spins aligned or subtract if anti-aligned
        corr += prob/norm if num_ones % 2 == 0 else -prob/norm

    return corr


def exp_val_hamiltonian_termwise(variational_params: QAOAVariationalBaseParams,
                                qaoa_backend,
                                hamiltonian: Hamiltonian, 
                                mixer_type:str, 
                                p: int,
                                qaoa_optimized_angles: Optional[list] = None,
                                qaoa_optimized_counts: Optional[dict] = None,
                                analytical: bool = True):
    """
    Computes the single spin expectation values <Z_{i}> and the correlation matrix Mij = <Z_{i}Z_{j}>,
    using the optimization results obtained from QAOA tranining the specified QAOA cost backend.

    Parameters
    ----------
    variational_params: `QAOAVariationalBaseParams`
        Set of variational parameters in the QAOA ansatz.
    qaoa_backend: `QAOABaseBackend`
        Chosen backend on which QAOA is performed.
    hamiltonian: `Hamiltonian`
        Hamiltonian object containing the problem statement.
    p: `int`
        Number of layers in QAOA ansatz.
    qaoa_optimized_angles: `list`
        Optimized angles of the underlying QAOA.
    qaoa_optimized_counts: `dict`
        Dictionary containing the measurement counts of optimized QAOA circuit.
    analytical: `bool`
        Boolean that indicates whether to use analytical or numerical expectation
        calculation methods.

    Returns
    -------
    exp_vals_z: `np.array`
        Single spin expectation values as a numpy array.
    corr_matrix: `np.array`
        Correlation matrix as a numpy Matrix object.
    """

    # Define number of qubits, problem hamiltonian and QAOA parameters
    n_qubits = hamiltonian.n_qubits

    # Extract Hamiltonian terms
    terms = list(hamiltonian.terms)
    
    # Initialize the z expectation values and correlation matrix with 0s
    exp_vals_z = np.zeros(n_qubits)
    corr_matrix = np.zeros((n_qubits, n_qubits))

    # If single layer ansatz use analytical results
    if (analytical == True and p == 1 and mixer_type == 'x' and
        isinstance(qaoa_optimized_angles, list)):

        # Compute expectation values and correlations of terms present in the Hamiltonian
        for term in terms:

            # If bias term compute expectation value
            if len(term) == 1:
                i = term.qubit_indices[0]
                exp_vals_z[i] = exp_val_single_analytical(
                    i, hamiltonian, qaoa_optimized_angles)

            # If two-body term compute correlation
            elif len(term) == 2:
                i, j = term.qubit_indices
                corr_matrix[i][j] = exp_val_pair_analytical(
                    (i, j), hamiltonian, qaoa_optimized_angles)

            # If constant term, ignore
            else:
                continue

    # If multilayer ansatz, perform numerical computation
    else:

        if isinstance(qaoa_optimized_counts, dict):
            counts_dict = qaoa_optimized_counts
        else:
            raise ValueError("Please specify optimized counts to compute expectation values.")

        # Compute expectation values and correlations of terms present in the Hamiltonian
        for term in terms:

            # If bias term compute expectation value
            if len(term) == 1:
                i = term.qubit_indices[0]
                exp_vals_z[i] = exp_val_single(i, counts_dict)

            # If two-body term compute correlation
            elif len(term) == 2:
                i, j = term.qubit_indices
                corr_matrix[i][j] = exp_val_pair((i, j), counts_dict)

            # If constant term, ignore
            if len(term) == 0:
                continue

    # Remove expectation value contribution from the correlations
    corr_matrix -= np.outer(exp_vals_z, exp_vals_z)

    return exp_vals_z, corr_matrix


################################################################################
# ANALYTIC & KNOWN FORMULAE
################################################################################


def exp_val_single_analytical(spin: int, hamiltonian: Hamiltonian, qaoa_angles: tuple):
    """
    Computes the single spin expectation value <Z> from an analytically
    derived expression for a single layer QAOA Ansatz. 
    
    NOTE: Only valid for single layer QAOA Ansatz with X mixer Hamiltonian.

    Parameters
    ----------
    spin: `int`
        The spin whose expectation value we compute.

    hamiltonian: `Hamiltonian`
        Hamiltonian object containing the problem statement.

    qaoa_angles: `tuple`
        Pair of (gamma,beta) angles defined from QAOA ansatz and
        obtained in the QAOA process.

    Returns:
    -------
    exp_val: `float`
        Spin expectation value <Z>.
    """

    # Number of qubits in the system
    n_qubits = hamiltonian.n_qubits

    # Extract graph properties of the Hamiltonian
    terms = list(hamiltonian.terms)
    edges = [terms[j].qubit_indices for j in range(len(terms))]
    weights = hamiltonian.coeffs

    # Hamiltonian from graph definitions
    hamil_graph = dict(zip(edges, weights))

    # Spin biases
    h_u = hamil_graph[(spin,)] if hamil_graph.get((spin,)) is not None else 0

    # QAOA angles
    beta, gamma = qaoa_angles

    # Spin register as a list without the spin we focus on
    iter_qubits = [j for j in range(0, spin)] + \
        [j for j in range(spin+1, n_qubits)]

    # Initialize products
    exp_val = -np.sin(2*beta) * np.sin(2*gamma*h_u)

    # Loop over edges connecting u and v to other spins
    for n in iter_qubits:

        # Edges between the spin and others in the register
        edge = tuple([min(spin, n), max(spin, n)])

        # If edge not present in the graph the associated weight is set to 0
        J_un = 0 if hamil_graph.get(edge) is None else hamil_graph[edge]

        # Add factor to the products
        exp_val *= np.cos(2*gamma*J_un)

    return exp_val


def exp_val_pair_analytical(spins: tuple, hamiltonian: Hamiltonian, qaoa_angles: tuple):
    """
    Computes <Z_{i}Z_{j}> correlation between apair of spins analytically. It is an extension from the 
    expression derived by Bravyi et al. in arXiv:1910.08980 which includes the effect of biases. 

    NOTE: Only valid for single layer QAOA Ansatz with X mixer Hamiltonian.

    NOTE: In the presence of linear terms the <Z_{i}><Z_{j}> contribution needs to be
    subtracted later. This is done in the exp_val_hamiltonian_termwise() function used as a 
    wrapper for this function. 

    NOTE: OpenQAOA uses a different sign convention for the QAOA Ansatz than Bravy et al. - there is 
    a relative minus sign between the cost function and the mixer in OpenQAOA, which 
    is accounted for in this implementation. Additionally, the result below is valid 
    for a Hadamard state initialization and in the absence of bias terms in the Hamiltonian.

    Parameters
    ----------
    spins: `tuple`
        Pair of spins whose correlation we compute.

    hamiltonian: `Hamiltonian`
        Hamiltonian object containing the problem statement.

    qaoa_angles: `tuple`
        Pair of (gamma,beta) angles defined from QAOA ansatz and
        obtained in the QAOA process.

    Returns
    -------
    corr:
        Correlation <ZZ> between the specified spin pair.
    """

    # Number of qubits in the system
    n_qubits = hamiltonian.n_qubits

    # Extract graph properties of the Hamiltonian
    terms = list(hamiltonian.terms)
    edges = [terms[j].qubit_indices for j in range(len(terms))]
    weights = hamiltonian.coeffs

    # Hamiltonian from graph definitions
    hamil_graph = dict(zip(edges, weights))

    # Spins whose correlation we compute
    u, v = spins

    # Coupling between the spins
    J_uv = hamil_graph[spins] if hamil_graph.get(spins) is not None else 0

    # Spin biases
    h_u = hamil_graph[(u,)] if hamil_graph.get((u,)) is not None else 0
    h_v = hamil_graph[(v,)] if hamil_graph.get((v,)) is not None else 0

    # QAOA angles
    beta, gamma = qaoa_angles

    # Factors in the expression
    s = np.sin(2*beta)
    c = np.cos(2*beta)

    # Spin register as a list without u,v spins
    iter_qubits = [j for j in range(0, min(u, v))] + [j for j in range(
        min(u, v)+1, max(u, v))] + [j for j in range(max(u, v)+1, n_qubits)]

    # Initialize products
    prod1 = s**2/2 * np.cos(2*gamma*(h_u - h_v))
    prod2 = -s**2/2 * np.cos(2*gamma*(h_u + h_v))
    prod3 = -c*s*np.sin(2*gamma*J_uv) * np.cos(2*gamma*h_u)
    prod4 = -c*s*np.sin(2*gamma*J_uv) * np.cos(2*gamma*h_v)

    # Loop over edges connecting u and v to other spins
    for n in iter_qubits:

        # Edges between u,v and another spin in the register
        edge1 = tuple([min(u, n), max(u, n)])
        edge2 = tuple([min(v, n), max(v, n)])

        # If edge not present in the graph the associated weight is set to 0
        J_un = 0 if hamil_graph.get(edge1) is None else hamil_graph[edge1]
        J_vn = 0 if hamil_graph.get(edge2) is None else hamil_graph[edge2]

        # Add factor to the products
        prod1 *= np.cos(2*gamma*(J_un - J_vn))
        prod2 *= np.cos(2*gamma*(J_un + J_vn))
        prod3 *= np.cos(2*gamma*J_un)
        prod4 *= np.cos(2*gamma*J_vn)

    # Add the contribution from each product term
    corr = prod1 + prod2 + prod3 + prod4

    return corr


def energy_expectation_analytical(angles:Union[list,tuple],hamiltonian:Hamiltonian):
    """
    Computes the expectation value of the Hamiltonian for an analytical expression.

    NOTE: Only valid for single layer QAOA Ansatz with X mixer Hamiltonian and classical
    Hamiltonians with up to quadratic terms.

    Parameters
    ----------
    angles: `list` or `tuple`
        QAOA angles at which the Hamiltonian expectation value is computed
    hamiltonian: `Hamiltonian`
        Classical Hamiltonian from which the expectation value is computed.
    """
    
    # Extract terms and coefficients from the Hamiltonian
    terms = [pauli_term.qubit_indices for pauli_term in hamiltonian.terms]
    coeffs = hamiltonian.coeffs
    
    energy = 0
    
    # Compute the expectation value of each term and add its local energy contribution
    for coeff,term in zip(coeffs,terms):
        
        if len(term) == 2:
        
            local_energy = exp_val_pair_analytical(term,hamiltonian,angles)
            
        else:
            
            local_energy = exp_val_single_analytical(term[0],hamiltonian,angles)
        
        energy += coeff * local_energy
    
    # Add constant shift contribution
    energy += hamiltonian.constant
    
    return energy


def ring_of_disagrees(reg: List[int]) -> Hamiltonian:
    """
    Builds the cost Hamiltonian for the "Ring of Disagrees".

    Parameters
    ----------
    reg: `list`
        register of qubits in the system.

    Returns
    -------
    ring_hamil: `Hamiltonian`
        Hamiltonian object containing Ring of Disagrees model.

    Notes
    -----
    This model is introduced in https://arxiv.org/abs/1411.4028
    """

    # Number of qubits from input register
    n_qubits = len(reg)

    # Define terms for the ring structure
    terms = [(reg[i], reg[(i+1) % n_qubits]) for i in range(n_qubits)]

    # Define coefficients as in original formulation of the model
    coeffs = [0.5]*len(terms)

    # Constant term as in original formulation of the model
    constant = -len(terms)*0.5

    # Define Hamiltonian
    ring_hamil = Hamiltonian.classical_hamiltonian(terms, coeffs,
                                                   constant=constant)
    return ring_hamil


################################################################################
# OTHER MISCELLANEOUS
################################################################################


def flip_counts(counts_dictionary: dict) -> dict:
    """
    Returns a counts/probability dictionary that have their keys flipped. This
    formats the bit-strings from a right-most bit representing being the first 
    qubit to the left-most bit representing the first qubit.

    Parameters
    ----------
    counts_dictionary: `dict`
        Count dictionary whose keys are flipped.
        
    Returns
    -------
    output_counts_dictionary: `dict`
        Count dictionary with flipped keys.
    """

    output_counts_dictionary = dict()

    for key, value in counts_dictionary.items():
        output_counts_dictionary[key[::-1]] = value

    return output_counts_dictionary

def qaoa_probabilities(statevector) -> dict:
    """
    Return a qiskit-style probability dictionary from a statevector.

    Parameters
    ----------
    statevector: `np.ndarray[complex]`
        The wavefunction whose probability distribution needs to be calculated.

    Returns
    -------
    prob_dict: `dict`
        Probabilities represented as a python dictionary with basis states stored
        as keys and their probabilities as their corresponding values.
    """
    # Define list of probabilities from wavefunction amplitudes
    prob_vec = np.real(np.conjugate(statevector)*statevector)

    # Extract number of qubits from size of probability
    n_qubits = int(np.log2(len(prob_vec)))

    # Initialize probability dictionary
    prob_dict = {}

    for x in range(len(prob_vec)):

        # Define binary representation of each state, with qubit-0 most significant bit
        key = np.binary_repr(x, n_qubits)[::-1]

        # Update probability dictionary
        prob_dict.update({key: prob_vec[x]})

    return prob_dict
