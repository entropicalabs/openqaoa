"""
Utility and convenience functions for a number of QAOA applications.
"""

from __future__ import annotations
from typing import Optional, Union, List, Tuple
import itertools
import numpy as np
import uuid
import matplotlib.pyplot as plt
import networkx as nx
import datetime

from .qaoa_components import Hamiltonian, PauliOp

from .qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)  # ff
from .qaoa_components.ansatz_constructor.gatemap import TwoQubitRotationGateMap  # ff


def X_mixer_hamiltonian(n_qubits: int, coeffs: List[float] = None) -> Hamiltonian:
    """Construct a Hamiltonian object to implement the X mixer.

    Parameters
    ----------
    n_qubits: `int`
        The number of qubits in the mixer Hamiltonian.
    coeffs: `List[float]`
        The coefficients of the X terms in the Hamiltonian.

    Returns
    -------
    `Hamiltonian`
        The Hamiltonian object corresponding to the X mixer.
    """
    # If no coefficients provided, set all to -1
    coeffs = [-1] * n_qubits if coeffs is None else coeffs

    # Initialize list of terms
    terms = []

    # Generate terms in the X mixer
    for i in range(n_qubits):
        terms.append(PauliOp.X(i))

    # Define mixer Hamiltonian
    hamiltonian = Hamiltonian(pauli_terms=terms, coeffs=coeffs, constant=0)

    return hamiltonian


def XY_mixer_hamiltonian(
    n_qubits: int,
    qubit_connectivity: Union[List[list], List[tuple], str] = "full",
    coeffs: List[float] = None,
) -> Hamiltonian:
    r"""
    Construct a Hamiltonian object to implement the XY mixer.

    .. Important::
            The XY mixer is not implemented with :math:`RXY` Gates, but with :math:`H_{XY}=\frac{1}{2}(\sum_{i,j} X_iX_j+Y_iY_j)`

    Parameters
    ----------
    n_qubits: `int`
        The number of qubits in the system.
    qubit_connectivity: `Union[List[list],List[tuple], str]`
        The connectivity of the qubits in the mixer Hamiltonian.
    coeffs: `List[float]`
        The coefficients of the XY terms in the Hamiltonian.

    Returns
    -------
    `Hamiltonian`
        The Hamiltonian object corresponding to the XY mixer.
    """
    # Set of topologies supported by default
    connectivity_topology_dict = {
        "full": list(itertools.combinations(range(n_qubits), 2)),
        "chain": [(i, i + 1) for i in range(n_qubits - 1)],
        "star": [(0, i + 1) for i in range(n_qubits - 1)],
    }

    # Check if input connectivity is a default value
    if isinstance(qubit_connectivity, str):
        try:
            # Define qubit connectivity from default value
            qubit_connectivity = connectivity_topology_dict[qubit_connectivity]
        except KeyError:
            raise ValueError(
                f"Please choose connection topology from {list(connectivity_topology_dict.keys())}"
            )

    # Define connectivty according to user input
    else:
        # Extract indices from connectivity
        indices = set([qubit for term in qubit_connectivity for qubit in term])

        # Ensure all indices are defined within the range of number of qubits
        assert (
            max(indices) <= n_qubits - 1
        ), "Qubit index in connectivity list is out of range"
        assert min(indices) >= 0, "Qubit index should be a positive integer"

    # If no coefficients provided, set all to the number of terms
    coeffs = [0.5] * 2 * len(qubit_connectivity) if coeffs is None else coeffs

    # Initialize list of terms
    terms = []

    # Generate terms in the XY mixer
    for pair in qubit_connectivity:
        i, j = pair
        terms.append(PauliOp.X(i) @ PauliOp.X(j))
        terms.append(PauliOp.Y(i) @ PauliOp.Y(j))

    # Define mixer Hamiltonian
    hamiltonian = Hamiltonian(pauli_terms=terms, coeffs=coeffs, constant=0)

    return hamiltonian


def quick_create_mixer_for_topology(
    input_gatemap: TwoQubitRotationGateMap,
    n_qubits: int,
    qubit_connectivity: Union[List[list], List[tuple], str] = "full",
    coeffs: List[float] = None,
) -> Tuple[List[TwoQubitRotationGateMap], List[float]]:
    """
    Quickly generates a gatemap list and coeffs for a specific topology.
    Can only be used with 2-Qubit Gates.

    Parameters
    ----------
    input_gatemap: `TwoQubitRotationGateMap`
        The GateMap whose connectivity we are trying to create.
    n_qubits: `int`
        The number of qubits in the system.
    qubit_connectivity: `Union[List[list],List[tuple], str]`
        The connectivity of the qubits in the mixer.
    coeffs: `List[float]`, optional
        The coefficients of the GateMap in the Mixer Blocks.

    Returns
    -------
    `Tuple[List[TwoQubitRotationGateMap], List[float]]`
        Returns tuple containing the list of gatemaps and their associated coefficients.
        If no coefficients were on initialisation provided,
        a default of 1.0 is used for all gatemap objects.
    """

    # Set of topologies supported by default
    connectivity_topology_dict = {
        "full": list(itertools.combinations(range(n_qubits), 2)),
        "chain": [(i, i + 1) for i in range(n_qubits - 1)],
        "star": [(0, i + 1) for i in range(n_qubits - 1)],
    }

    # Check if input connectivity is a default value
    if isinstance(qubit_connectivity, str):
        try:
            # Define qubit connectivity from default value
            qubit_connectivity = connectivity_topology_dict[qubit_connectivity]
        except KeyError:
            raise ValueError(
                f"Please choose connection topology from {list(connectivity_topology_dict.keys())}"
            )

    # Define connectivty according to user input
    else:
        # Extract indices from connectivity
        indices = set([qubit for term in qubit_connectivity for qubit in term])

        # Ensure all indices are defined within the range of number of qubits
        assert (
            max(indices) <= n_qubits - 1
        ), "Qubit index in connectivity list is out of range"
        assert min(indices) >= 0, "Qubit index should be a positive integer"

    # If no coefficients provided, set all to the number of terms
    coeffs = [1.0] * len(qubit_connectivity) if coeffs is None else coeffs

    # Initialize list of terms
    gatemaps = []

    # Generate terms in the 2Q mixer
    for pair in qubit_connectivity:
        i, j = pair
        gatemaps.append(input_gatemap(i, j))

    return gatemaps, coeffs


def get_mixer_hamiltonian(
    n_qubits: int,
    mixer_type: str = "x",
    qubit_connectivity: Union[List[list], List[tuple], str] = None,
    coeffs: List[float] = None,
) -> Hamiltonian:
    """
    Parameters
    ----------
    n_qubits: `int`
        Number of qubits in the Hamiltonian.
    mixer_type: `str`
        Name of the mixer Hamiltonian. Choose from `x` or `xy`.
    qubit_connectivity: `Union[List[list],List[tuple], str]`, optional
        The connectivity of the qubits in the mixer Hamiltonian.
    coeffs: `List[float]`, optional
        The coefficients of the terms in the Hamiltonian.

    Returns:
    --------
    `Hamiltonian`
        Hamiltonian object containing the specificied mixer.
    """

    # Return mixer Hamiltonian according to specified type
    if mixer_type == "x":
        mixer = X_mixer_hamiltonian(n_qubits, coeffs)
    else:
        mixer = XY_mixer_hamiltonian(n_qubits, qubit_connectivity, coeffs)

    return mixer


################################################################################
# decorators
################################################################################
def round_value(function):
    """
    Round a value to a given precision.
    This function will be used as a decorator to round the values given by the
    ``expectation`` and ``expectation_w_uncertainty`` methods.

    Parameters
    ----------
    function: `Callable`
        The function to be decorated

    Returns
    -------
        The rounded value(s)

    """

    PRECISION = 12

    def wrapper(*args, **kwargs):
        values = function(*args, **kwargs)
        if isinstance(values, dict):
            return {k: round(v, PRECISION) for k, v in values.items()}
        else:
            return np.round(values, PRECISION)

    return wrapper


################################################################################
# METHODS FOR PRINTING HAMILTONIANS AND GRAPHS, AND PRINTING ONE FROM EACH OTHER
################################################################################


def graph_from_hamiltonian(hamiltonian: Hamiltonian) -> nx.Graph:
    """
    Creates a networkx graph corresponding to a specified problem Hamiltonian.

    .. Important::
        This function cannot handle non-QUBO terms.
        Linear terms are stored as nodes with weights.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        The Hamiltonian of interest. Must be specified a Hamiltonian object.

    Returns
    -------
    G: `networkx.Graph`
        The corresponding networkx graph with the edge weights being the
        two-qubit coupling coefficients,
        and the node weights being the single-qubit bias terms.



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
        if len(term) == 1:
            G.add_node(term_tuple[0], weight=weight)

        # If term is quadratic add as an edge with a weight attribute
        elif len(term) == 2:
            G.add_edge(term_tuple[0], term_tuple[1], weight=weight)

    return G


def hamiltonian_from_graph(G: nx.Graph) -> Hamiltonian:
    """
    Builds a cost Hamiltonian as a collection of PauliOp objects
    from a specified networkx graph, extracting any node and edge weights.

    Parameters
    ----------
    G: `networkx.Graph`
        The specified networkx graph.

    Returns
    -------
    hamiltonian: `Hamiltonian`
        The Hamiltonian object constructed from the specified graph.
    """
    # Node bias terms
    nodes_info = dict(G.nodes(data="weight"))
    singlet_terms = [
        (node,) for node, weight in nodes_info.items() if weight is not None
    ]
    singlet_coeffs = [coeff for coeff in nodes_info.values() if coeff is not None]

    # Edge terms
    pair_terms, pair_coeffs = [], []
    for u, v, edge_weight in G.edges(data="weight"):
        pair_terms.append((u, v))
        # We expect the edge weight to be given in the attribute called
        # "weight". If it is None, assume a weight of 1.0
        pair_coeffs.append(edge_weight if edge_weight else 1)

    # Collect all terms and coefficients
    terms = singlet_terms + pair_terms
    coeffs = singlet_coeffs + pair_coeffs

    # Define Hamiltonian
    hamiltonian = Hamiltonian.classical_hamiltonian(
        terms=terms, coeffs=coeffs, constant=0
    )

    return hamiltonian


def random_k_regular_graph(
    degree: int,
    nodes: List[int],
    seed: int = None,
    weighted: bool = False,
    biases: bool = False,
) -> nx.Graph:
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
            G[edge[0]][edge[1]]["weight"] = 1

        # If weighted attribute is True, weights are assigned as random integers
        else:
            G[edge[0]][edge[1]]["weight"] = np.random.rand()

    # If biases attribute is True, add node weights as random integers
    if biases:
        for node in G.nodes():
            G.nodes[node]["weight"] = np.random.rand()

    return G


def plot_graph(G: nx.Graph, ax=None, colormap="seismic") -> None:
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
    biases_and_nodes = nx.get_node_attributes(G, "weight")
    biases = list(biases_and_nodes.values())
    edges_and_weights = nx.get_edge_attributes(G, "weight")
    pos = nx.shell_layout(G)

    # extract minimum and maximum weights for side bar limits
    weights = list(edges_and_weights.values())
    # Define color map
    cmap = plt.cm.get_cmap(colormap)

    if len(set(weights)) > 1:
        edge_vmin = min(weights)
        edge_vmax = max(weights)

        # Define normalized color map
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        )
        # Add colormap to plot
        cbar = plt.colorbar(sm, pad=0.08)
        cbar.ax.set_ylabel("Edge Weights", rotation=270, labelpad=15)
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
        sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cbar2 = plt.colorbar(sm2, location="left")
        cbar2.ax.set_ylabel("Single Qubit Biases", rotation=90)

        # Draw graph
        nx.draw(
            G,
            pos,
            node_size=500,
            node_color=biases,
            edge_color=weights,
            width=2.5,
            cmap=cmap,
            edge_cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            with_labels=True,
        )

    else:
        # Draw graph
        nx.draw(
            G,
            pos,
            node_size=500,
            edge_color=weights,
            width=2.5,
            edge_cmap=cmap,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            with_labels=True,
        )

    # Show plot

    plt.show
    return None


def random_classical_hamiltonian(
    reg: List[int],
    seed: int = None,
    weighted: bool = True,
    biases: bool = True,
    constant: int = 0,
) -> Hamiltonian:
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

    .. Important::
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
        bias_coeffs = np.random.rand(n_biases) if weighted else np.ones(n_biases)

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
    assert len(terms) == len(weights), "Each term should have an associated weight"

    # Define classical Hamiltonian
    hamiltonian = Hamiltonian.classical_hamiltonian(terms, weights, constant=constant)

    return hamiltonian


################################################################################
# HAMILTONIANS AND DATA
################################################################################


def ground_state_hamiltonian(
    hamiltonian: Hamiltonian, bounded=True
) -> Tuple[float, list]:
    """
    Computes the exact ground state and ground state energy of a classical Hamiltonian.
    Uses standard numpy module.

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
        raise ValueError(
            "The number of qubits is too high, computation could take a long time. If still want to proceed set argument `bounded` to False"
        )

    # Generate qubit register
    register = range(n_qubits)

    # Intialize energies
    energies = np.zeros(2 ** len(register))

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
    config_strings = [np.binary_repr(index, len(register))[::-1] for index in indices]

    return min_energy, config_strings


def bitstring_energy(
    hamiltonian: Hamiltonian, bitstring: Union[List[int], str]
) -> float:
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
        variables_product = np.prod(
            [(-1) ** int(bitstring[k]) for k in term.qubit_indices]
        )

        # Add energy contribution
        energy += hamiltonian.coeffs[i] * variables_product

    # Add constant contribution
    energy += hamiltonian.constant

    return energy


def energy_expectation(hamiltonian: Hamiltonian, measurement_counts: dict) -> float:
    """
    Computes the energy expectation value from a set of measurement counts,
    with respect to a classical cost Hamiltonian.

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
        # Number of ones (spins pointing down) from the specific
        # configuration for each Hamiltonian term
        num_ones_list = [
            sum([int(state[i]) for i in term.qubit_indices])
            for term in hamiltonian.terms
        ]

        # Compute configuration energy
        config_energy = sum(
            [
                hamiltonian.coeffs[i]
                if num_ones % 2 == 0
                else -1 * hamiltonian.coeffs[i]
                for i, num_ones in enumerate(num_ones_list)
            ]
        )

        # Add contribution to total energy
        energy += prob * config_energy

    # Normalize with respect to the number of shots
    energy *= 1 / shots

    # Add constant term in Hamiltonian
    energy += hamiltonian.constant

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
    energies = np.zeros((2 ** len(register)))

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


def plot_energy_spectrum(
    hamiltonian: Hamiltonian,
    high_k_states: Optional[int] = None,
    low_k_states: Optional[int] = None,
    ax=None,
    cmap="winter",
) -> None:
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
        unique_energies, degeneracy = (
            unique_energies[-high_k_states:],
            degeneracy[-high_k_states:],
        )
    elif low_k_states is not None:
        unique_energies, degeneracy = (
            unique_energies[:low_k_states],
            degeneracy[:low_k_states],
        )

    # Define colormap
    cmap = plt.cm.get_cmap(cmap, len(unique_energies))

    # If no axis provided, define figure
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 5))

    # Plot energy levels
    for i, energy in enumerate(unique_energies):
        ax.axhline(energy, label=f"Degeneracy={degeneracy[i]}", color=cmap(i))

    # Set axis attributes and legend
    ax.set(
        xticks=[],
        yticks=unique_energies,
        ylabel="Energy[a.u.]",
        title="Hamiltonian Energy spectrum",
    )
    ax.legend(loc="center left", fontsize=8)

    return None


def low_energy_states(
    hamiltonian: Hamiltonian, threshold_per: float
) -> Tuple[float, list]:
    """
    Return threshold energy and the low energy states of the
    specified hamiltonian which are below this threshold.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Compute the low energy states of this Hamiltonian
    threshold_per: `float`
        Threshold percentage away from the ground state,
        defining the energy we window we search in for low energy states.

    Returns
    -------
    low_energy_threshold: `float`
        The energy threshold below which we retrieve states.
    states: `list`
        The list of low energy states that lie below the low
        energy threshold.

    .. Important::
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
    low_energy_threshold = ground_state_energy + threshold_per * np.abs(
        highest_state_energy - ground_state_energy
    )

    # Initilize inndices for low energy states
    low_energy_indices = []

    # Obtain indices for energies below the threshold
    for idx in range(len(energies)):
        if energies[idx] <= low_energy_threshold:
            low_energy_indices.append(idx)

    # Extract states from the Hamiltonian spectrum
    states = [
        np.binary_repr(index, hamiltonian.n_qubits)[::-1]
        for index in low_energy_indices
    ]

    return low_energy_threshold, states


def low_energy_states_overlap(
    hamiltonian: Hamiltonian, threshold_per: float, prob_dict: dict
) -> float:
    """
    Calculates the overlap between the low energy states of a Hamiltonian,
    below a specific threshold away from the ground state energy, and an input
    state, expressed in terms of a probability dictionary.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Compute overlap with respect to energies of this Hamiltonian
    threshold_per: `float`
        Threshold percentage away from the ground state, defining the
        energy we window we search in for low energy states.
    prob_dict: `dict`
        The measurement outcome dictionary generated from the
        circuit execution.

    Returns
    -------
    total_overlap: `float`
        The total overlap with the low-energy states.

    .. Important::
        The threshold is calculated as `threshold_per` factors away
        from the ground state of the Hamiltonain.
        For `threshold_per=0` the function returns the ground state overlap of the QAOA output.
    """
    # Extract number of qubits from probability dictionary
    n_qubits = len(list(prob_dict.keys())[0])

    # Ensure number of qubits matches the number of qubits registered in the Hamiltonian
    assert (
        n_qubits == hamiltonian.n_qubits
    ), "Number of qubits in the Hamiltonian does not match the probabilities specified"

    # Extract low energy states
    _, states = low_energy_states(hamiltonian, threshold_per)

    # Compute overlap
    total_overlap = sum([prob_dict[state] for state in states]) / sum(
        list(prob_dict.values())
    )

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
        exp_val += -prob / norm if Z > 0 else prob / norm

    return exp_val


def exp_val_pair(spins: tuple, prob_dict: dict):
    r"""
    Computes the correlation :math:`Mij = <Z_{i}Z_{j}>` between qubits i,j using the QAOA optimized
    wavefunction.

    .. Important::
        In the presence of linear terms the :math:`<Z_{i}><Z_{j}>` contribution needs to be
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
        corr += prob / norm if num_ones % 2 == 0 else -prob / norm

    return corr


def exp_val_hamiltonian_termwise(
    hamiltonian: Hamiltonian,
    mixer_type: str,
    p: int,
    qaoa_optimized_angles: Optional[list] = None,
    qaoa_optimized_counts: Optional[dict] = None,
    analytical: bool = True,
):
    """
    Computes the single spin expectation values <Z_{i}> and the correlation matrix Mij = <Z_{i}Z_{j}>,
    using the optimization results obtained from QAOA tranining the specified QAOA cost backend.

    Parameters
    ----------
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
    if (
        analytical == True
        and p == 1
        and mixer_type == "x"
        and isinstance(qaoa_optimized_angles, list)
    ):
        # Compute expectation values and correlations of terms present in the Hamiltonian
        for term in terms:
            # If bias term compute expectation value
            if len(term) == 1:
                i = term.qubit_indices[0]
                exp_vals_z[i] = exp_val_single_analytical(
                    i, hamiltonian, qaoa_optimized_angles
                )

            # If two-body term compute correlation
            elif len(term) == 2:
                i, j = term.qubit_indices
                corr_matrix[i][j] = exp_val_pair_analytical(
                    (i, j), hamiltonian, qaoa_optimized_angles
                )
            else:
                continue

    # If multilayer ansatz, perform numerical computation
    else:
        if isinstance(qaoa_optimized_counts, dict):
            counts_dict = qaoa_optimized_counts
        else:
            raise ValueError(
                "Please specify optimized counts to compute expectation values."
            )

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

            else:
                continue

    # Remove expectation value contribution from the correlations
    corr_matrix -= np.outer(exp_vals_z, exp_vals_z)

    return exp_vals_z, corr_matrix


@round_value
def calculate_calibration_factors(
    hamiltonian: Hamiltonian,
    calibration_measurements: dict,
    calibration_registers: list,
    qubit_mapping: list,
) -> Dict:
    """
    Computes the single spin and pairs of spins calibration factors, which are the expectation value of the observables found in the particular Hamiltonian, <Z_{i}> and <Z_{i}Z_{j}>, from the calibration data provided. The calibration data is obtained under BFA on an empty (initial state = |000..0>) QAOA circuit.
    See arXiv:2012.09738 and arXiv:2106.05800.

    Parameters
    ----------
    hamiltonian: `Hamiltonian`
        Hamiltonian object containing the problem statement.
    calibration_measurements: `dict`
        Dictionary containing the measurement counts of an empty QAOA circuit.
    calibration_registers: `list`
        List specifying the physical (device) qubits on which the calibration data has been obtained.
        This is required because the calibration is usually performed over the whole device and hence the measurement outcomes (the calibration data) are strings with the size of the whole device while usually only a particular section is used.
    qubit_mapping: `list`
        List specifying the physical (device) qubits on which the QAOA circuit is executed. Related to qubit selection and qubit routing.

    Returns
    -------
    calibration_factors: `dict`
        Calibration factors as a dict.
    """
    calibration_registers_dict = {v: k for k, v in enumerate(calibration_registers)}

    # Define number of qubits, problem hamiltonian and QAOA parameters
    n_qubits = hamiltonian.n_qubits

    # Extract Hamiltonian terms
    terms = list(hamiltonian.terms)

    if qubit_mapping == None:
        qubit_mapping = np.arange(0, n_qubits)

    # Initialize an empty dict
    calibration_factors = {}

    # Compute single spin and pairs of spins expectation values of terms present in the Hamiltonian
    for term in terms:
        # If bias term compute single spins expectation value
        if len(term) == 1:
            i = term.qubit_indices[0]
            i_phys = qubit_mapping[i]
            i_cal = calibration_registers_dict[i_phys]
            exp_val_z = exp_val_single(i_cal, calibration_measurements)
            calibration_factors.update({(i,): exp_val_z})

        # If two-body term compute pairs of spins expectation values
        elif len(term) == 2:
            i, j = term.qubit_indices  # problem indices, ex: (0,1)
            i_phys, j_phys = (
                qubit_mapping[i],
                qubit_mapping[j],
            )  # physical indices, ex: (133, 131) after routing
            i_cal, j_cal = (
                calibration_registers_dict[i_phys],
                calibration_registers_dict[j_phys],
            )  # calibration indices, i.e. to which location on the measurement string each physical qubit corresponds to, ex: (63, 61)
            exp_val_zz = exp_val_pair((i_cal, j_cal), calibration_measurements)

            calibration_factors.update(
                {(i, j): exp_val_zz}
            )  # calibration factors are calculated for the terms in the hamiltonian/problem

        # If constant term, ignore
        if len(term) == 0:
            continue

    assert all(
        value != 0 for value in calibration_factors.values()
    ), "One (or more) of the calibration factors is 0 which means that the measurement is faulty. Please check the data."

    assert all(
        value <= 1 for value in calibration_factors.values()
    ), "One (or more) of the calibration factors is larger than 1 which is not physical. Please check the data."

    return calibration_factors


################################################################################
# ANALYTIC & KNOWN FORMULAE
################################################################################


def exp_val_single_analytical(spin: int, hamiltonian: Hamiltonian, qaoa_angles: tuple):
    """
    Computes the single spin expectation value :math:`<Z>` from an analytically
    derived expression for a single layer QAOA Ansatz.

    .. Important::
        Only valid for single layer QAOA Ansatz with X mixer Hamiltonian.

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
    iter_qubits = [j for j in range(0, spin)] + [j for j in range(spin + 1, n_qubits)]

    # Initialize products
    exp_val = -np.sin(2 * beta) * np.sin(2 * gamma * h_u)

    # Loop over edges connecting u and v to other spins
    for n in iter_qubits:
        # Edges between the spin and others in the register
        edge = tuple([min(spin, n), max(spin, n)])

        # If edge not present in the graph the associated weight is set to 0
        J_un = 0 if hamil_graph.get(edge) is None else hamil_graph[edge]

        # Add factor to the products
        exp_val *= np.cos(2 * gamma * J_un)

    return exp_val


def exp_val_pair_analytical(spins: tuple, hamiltonian: Hamiltonian, qaoa_angles: tuple):
    """
    Computes :math:`<Z_{i}Z_{j}>` correlation between apair of spins analytically.
    It is an extension from the expression derived by
    Bravyi et al. in https://arxiv.org/abs/1910.08980 which includes the effect of biases.

    .. Important::
        * Only valid for single layer QAOA Ansatz with X mixer Hamiltonian.
        * In the presence of linear terms the <Z_{i}><Z_{j}> contribution needs to be subtracted later. This is done in the exp_val_hamiltonian_termwise() function used as a wrapper for this function.
        * OpenQAOA uses a different sign convention for the QAOA Ansatz than Bravy et al. - there is a relative minus sign between the cost function and the mixer in OpenQAOA, which is accounted for in this implementation. Additionally, the result below is valid for a Hadamard state initialization and in the absence of bias terms in the Hamiltonian.

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
    corr: `float`
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
    s = np.sin(2 * beta)
    c = np.cos(2 * beta)

    # Spin register as a list without u,v spins
    iter_qubits = (
        [j for j in range(0, min(u, v))]
        + [j for j in range(min(u, v) + 1, max(u, v))]
        + [j for j in range(max(u, v) + 1, n_qubits)]
    )

    # Initialize products
    prod1 = s**2 / 2 * np.cos(2 * gamma * (h_u - h_v))
    prod2 = -(s**2) / 2 * np.cos(2 * gamma * (h_u + h_v))
    prod3 = -c * s * np.sin(2 * gamma * J_uv) * np.cos(2 * gamma * h_u)
    prod4 = -c * s * np.sin(2 * gamma * J_uv) * np.cos(2 * gamma * h_v)

    # Loop over edges connecting u and v to other spins
    for n in iter_qubits:
        # Edges between u,v and another spin in the register
        edge1 = tuple([min(u, n), max(u, n)])
        edge2 = tuple([min(v, n), max(v, n)])

        # If edge not present in the graph the associated weight is set to 0
        J_un = 0 if hamil_graph.get(edge1) is None else hamil_graph[edge1]
        J_vn = 0 if hamil_graph.get(edge2) is None else hamil_graph[edge2]

        # Add factor to the products
        prod1 *= np.cos(2 * gamma * (J_un - J_vn))
        prod2 *= np.cos(2 * gamma * (J_un + J_vn))
        prod3 *= np.cos(2 * gamma * J_un)
        prod4 *= np.cos(2 * gamma * J_vn)

    # Add the contribution from each product term
    corr = prod1 + prod2 + prod3 + prod4

    return corr


def energy_expectation_analytical(angles: Union[list, tuple], hamiltonian: Hamiltonian):
    """
    Computes the expectation value of the Hamiltonian for an analytical expression.

    .. Important::
        Only valid for single layer QAOA Ansatz with X mixer Hamiltonian and classical
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
    for coeff, term in zip(coeffs, terms):
        if len(term) == 2:
            local_energy = exp_val_pair_analytical(term, hamiltonian, angles)

        else:
            local_energy = exp_val_single_analytical(term[0], hamiltonian, angles)

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

    .. Important::
        This model is introduced in https://arxiv.org/abs/1411.4028
    """

    # Number of qubits from input register
    n_qubits = len(reg)

    # Define terms for the ring structure
    terms = [(reg[i], reg[(i + 1) % n_qubits]) for i in range(n_qubits)]

    # Define coefficients as in original formulation of the model
    coeffs = [0.5] * len(terms)

    # Constant term as in original formulation of the model
    constant = -len(terms) * 0.5

    # Define Hamiltonian
    ring_hamil = Hamiltonian.classical_hamiltonian(terms, coeffs, constant=constant)
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


def permute_counts_dictionary(
    counts_dictionary: dict, final_qubit_layout: List[int]
) -> dict:
    """Permutes the order of the qubits in the counts dictionary to the
    original order if SWAP gates were used leading to modified qubit layout.
    Parameters
    ----------
    counts_dictionary : `dict`
        The measurement outcomes obtained from the Simulator/QPU
    original_qubit_layout: List[int]
        The qubit layout in which the qubits were initially
    final_qubit_layout: List[int]
        The final qubit layout after application of SWAPs

    Returns
    -------
    `dict`
        The permuted counts dictionary with qubits in the original place
    """

    # Create a mapping of original positions to final positions
    original_qubit_layout = list(range(len(final_qubit_layout)))
    mapping = {
        original_qubit_layout[i]: final_qubit_layout[i]
        for i in range(len(original_qubit_layout))
    }
    permuted_counts = {}

    for basis, counts in counts_dictionary.items():

        def permute_string(basis_state: str = basis, mapping: dict = mapping):
            # Use the mapping to permute the string
            permuted_string = "".join(
                [basis_state[mapping[i]] for i in range(len(basis_state))]
            )
            return permuted_string

        permuted_counts.update({permuted_string: counts})

    return permuted_counts


def negate_counts_dictionary(counts_dictionary: dict, s: int) -> dict:
    """Negates every bitstring of the counts dictionary according to
    the position of the X gates before the measurement.
    Used in SPAM Twirling.
    Parameters
    ----------
    counts_dictionary : `dict`
        The measurement outcomes obtained from the Simulator/QPU
    s: int
        Syndrome whose binary representation denotes the negated qubits. For example, 4 = 100, signifies that the first qubit had an X gate just before the measurement, which requires the first digit of the every key to be classically negated inside this function.

    Returns
    -------
    `dict`
        The negated counts dictionary
    """
    negated_counts = {}
    for key in counts_dictionary.keys():
        n_qubits = len(key)
        negated_key = s ^ int(
            key, 2
        )  # bitwise XOR to classically negate randomly chosen qubits, specified by s
        negated_counts.update(
            [(format(negated_key, "b").zfill(n_qubits), counts_dictionary[key])]
        )
    return negated_counts


@round_value
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
    prob_vec = np.real(np.conjugate(statevector) * statevector)

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


################################################################################
# DICTIONARY MANIPULATION and SERIALIZATION
################################################################################
def delete_keys_from_dict(obj: Union[list, dict], keys_to_delete: List[str]):
    """
    Recursively delete all the keys keys_to_delete from a object (or list of dictionaries)
    Parameters
    ----------
    obj: dict or list[dict]
        dictionary or list of dictionaries from which we want to delete keys
    keys_to_delete: list
        list of keys to delete from the dictionaries

    Returns
    -------
    obj: dict or list[dict]
        dictionary or list of dictionaries from which we have deleted keys
    """
    if isinstance(obj, dict):
        for key in keys_to_delete:
            if key in obj:
                del obj[key]
        for key in obj:
            if isinstance(obj[key], dict):
                delete_keys_from_dict(obj[key], keys_to_delete)
            elif isinstance(obj[key], list):
                for item in obj[key]:
                    delete_keys_from_dict(item, keys_to_delete)
    elif isinstance(obj, list):
        for item in obj:
            delete_keys_from_dict(item, keys_to_delete)

    return obj


def convert2serialize(obj, complex_to_string: bool = False):
    """
    Recursively converts object to dictionary.

    Parameters
    ----------
    obj: object
        Object to convert to dictionary.
    complex_to_string: bool
        If True, convert complex numbers to string, so the result can be serialized to JSON.

    Returns
    -------
    dict: dict
        Dictionary representation of the object.
    """
    if isinstance(obj, dict):
        return {
            k: convert2serialize(v, complex_to_string)
            for k, v in obj.items()
            if v is not None
        }
    elif hasattr(obj, "_ast"):
        return convert2serialize(obj._ast(), complex_to_string)
    elif isinstance(obj, tuple):
        return tuple(
            convert2serialize(v, complex_to_string) for v in obj if v is not None
        )
    elif not isinstance(obj, str) and hasattr(obj, "__iter__"):
        return [convert2serialize(v, complex_to_string) for v in obj if v is not None]
    elif hasattr(obj, "__dict__"):
        return {
            k: convert2serialize(v, complex_to_string)
            for k, v in obj.__dict__.items()
            if not callable(v) and v is not None
        }
    elif complex_to_string and isinstance(obj, complex):
        return str(obj)
    else:
        return obj


################################################################################
# UUID and Timestamp
################################################################################


def generate_timestamp() -> str:
    """
    Generate a timestamp string in UTC+0. Format: YYYY-MM-DDTHH:MM:SS.

    Returns
    -------
    timestamp: `str`
        String representation of a timestamp.
    """
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


def generate_uuid() -> str:
    """
    Generate a UUID string.

    Returns
    -------
    uuid: `str`
        String representation of a UUID.
    """
    return str(uuid.uuid4())


def is_valid_uuid(uuid_to_test: str) -> bool:
    """
    Check if a string is a valid UUID.

    Parameters
    ----------
    uuid_to_test: `str`
        String to check if it is a valid UUID.

    Returns
    -------
    is_valid: `bool`
        Boolean indicating if the string is a valid UUID.
    """

    try:
        # generate a UUID object from the string, if it is a valid UUID it won't throw an error
        _ = uuid.UUID(uuid_to_test, version=4)
        return True
    except ValueError:
        # If it's a value error, then the string is not a valid string for a UUID.
        return False


def permute_counts_dictionary(
    counts_dictionary: dict, permutation_order: List[int]
) -> dict:
    """Permutes the order of the qubits in the counts dictionary to the
    original order if SWAP gates were used leading to modified qubit layout.
    Parameters
    ----------
    counts_dictionary : `dict`
        The measurement outcomes obtained from the Simulator/QPU
    permutation_order: List[int]
        The qubit order to permute the dictionary with

    Returns
    -------
    `dict`
        The permuted counts dictionary with qubits in the original place
    """

    # Create a mapping of original positions to final positions
    # original order always goes from 0 -> n-1
    original_order = list(range(len(permutation_order)))
    mapping = {
        original_order[i]: permutation_order[i] for i in range(len(original_order))
    }
    permuted_counts = {}

    for basis_state, counts in counts_dictionary.items():
        # Use the mapping to permute the string
        permuted_string = "".join(
            [basis_state[mapping[i]] for i in range(len(basis_state))]
        )
        permuted_counts.update({permuted_string: counts})

    return permuted_counts


################################################################################
# CHECKING FUNCTION
################################################################################


def check_kwargs(list_expected_params, list_default_values, **kwargs):
    """
    Checks that the given list of expected parameters can be found in the
    kwargs given as input. If so, it returns the parameters from kwargs, else
    it raises an exception.

    Args:
        list_expected_params: List[str]
            List of string containing the name of the expected parameters in
            kwargs
        list_default_values: List
            List containing the deafult values of the expected parameters in
            kwargs
        **kwargs:
            Keyword arguments where keys are supposed to be the expected params

    Returns:
        A tuple with the actual expected parameters if they are found in kwargs.

    Raises:
        ValueError:
            If one of the expected arguments is not found in kwargs and its
            default value is not specified.
    """

    def check_kwarg(expected_param, default_value, **kwargs):
        param = kwargs.pop(expected_param, default_value)

        if param is None:
            raise ValueError(f"Parameter '{expected_param}' should be specified")

        return param

    params = []
    for expected_param, default_value in zip(list_expected_params, list_default_values):
        params.append(check_kwarg(expected_param, default_value, **kwargs))

    return tuple(params)


####################################################################################
# QAOALIB
####################################################################################


def dicke_basis(excitations: int, n_qubits: int) -> np.ndarray:
    """
    Generates the Dicke basis state $|ek>$ with $k$ excitations

    Parameters
    ----------
    excitations: `int`
        Number of excitations in the basis vector
    n_qubits: `int`
        Total number of qubits in the system

    Returns
    -------
    `np.ndarray`
        Total basis states present in the expected Dicke vector in the computational basis
    """
    assert (
        n_qubits >= excitations
    ), "Excitations cannot be larger than total qubits in system"
    sub_sys_excitations = np.ones(excitations, dtype=int)
    sub_sys_ground = np.zeros(n_qubits - excitations, dtype=int)

    total_state = np.concatenate((sub_sys_ground, sub_sys_excitations))
    total_basis_comp = set(itertools.permutations(total_state))
    total_basis_comp = np.array(
        ["".join(str(i) for i in basis_comp) for basis_comp in total_basis_comp]
    )

    return total_basis_comp


def dicke_wavefunction(excitations, n_qubits):
    """
    Generate the k-excitations dicke statevector

    Parameters
    ----------
    excitations: int
        The number of excitations in the basis

    n_qubits: int
        The number of qubits in the register

    Returns
    -------
    `np.ndarray`
        The wavefunction vector for a given cumulative Dicke states with $<=k$ excitations
    """

    k_dicke = dicke_basis(excitations, n_qubits)
    k_dicke_ints = [int(state, 2) for state in k_dicke]
    wavefunction = np.array(
        [
            1.0 + 0.0j if num in k_dicke_ints else 0.0 + 0.0j
            for num in range(2**n_qubits)
        ],
        dtype=complex,
    ) / np.sqrt(len(k_dicke_ints))
    return wavefunction


def k_cumulative_excitations(k: int, n_qubits: int) -> np.ndarray:
    """
    Generates the Upper bound excitations basis vector $|Ek>$, which a superposition of all
    Dicke basis vectors upto excitation number $k$

    Parameters
    ----------
    k: `int`
        Upper bound on number of excitations in the basis vector
    n_qubits: `int`
        Total number of qubits in the system

    Returns
    -------
    wavefunction: `np.ndarray`
        The wavefunction vector for a given cumulative Dicke states with $<=k$ excitations
    """
    cumulative_dicke_bases = np.array(["0" * n_qubits])
    for exc in range(1, k + 1):
        cumulative_dicke_bases = np.concatenate(
            (cumulative_dicke_bases, dicke_basis(exc, n_qubits))
        )

    wavefn_locs = [int(basis, 2) for basis in cumulative_dicke_bases]
    wavefunction = np.array(
        [1 if loc in wavefn_locs else 0 for loc in range(2**n_qubits)], dtype=complex
    ) / np.sqrt(len(wavefn_locs))

    return wavefunction


def knapsack_balanced_basis(
    weight_capacity: int,
    weights_list: List,
    decision_register: List,
    slack_register: List,
):
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
    np.ndarray
    """
    n_decision_qubits = len(decision_register)
    n_slack_qubits = len(slack_register)
    n_total_qubits = n_slack_qubits + n_decision_qubits

    def to_bin(number, n_qubits):
        # not using np.binary_repr because it is deprecated!
        binary_form = bin(number)[2:].zfill(n_qubits)
        return binary_form

    decision_config_weights = {
        to_bin(dec_i, n_decision_qubits): sum(
            [
                weight * int(to_bin(dec_i, n_decision_qubits)[i])
                for i, weight in enumerate(weights_list)
            ]
        )
        for dec_i in range(2**n_decision_qubits)
    }

    decision_slack_configs = {
        to_bin(dec_i, n_decision_qubits): (
            to_bin(
                weight_capacity
                - decision_config_weights[to_bin(dec_i, n_decision_qubits)],
                n_slack_qubits,
            )
            if decision_config_weights[to_bin(dec_i, n_decision_qubits)]
            < weight_capacity
            else to_bin(0, n_slack_qubits)
        )
        for dec_i in range(2**n_decision_qubits)
    }

    all_configs = []
    for dec_config, slack_config in decision_slack_configs.items():
        config = np.empty(n_total_qubits, dtype=str)
        for i, loc in enumerate(decision_register):
            config[loc] = dec_config[i]
        for i, loc in enumerate(slack_register[::-1]):
            config[loc] = slack_config[i]

        config_str = "".join(i for i in config)
        all_configs.append(config_str[::-1])

    wavefn_locs = [int(basis, 2) for basis in all_configs]
    wavefunction = np.array(
        [1 if loc in wavefn_locs else 0 for loc in range(2**n_total_qubits)],
        dtype=complex,
    ) / np.sqrt(len(wavefn_locs))

    return wavefunction
