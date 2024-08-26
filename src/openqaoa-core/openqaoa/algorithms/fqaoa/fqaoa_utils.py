from typing import List, Optional
from itertools import combinations
from scipy import linalg
import numpy as np

ALLOWED_LATTICE = ["cyclic", "chain"]

def get_givens_rotation_angle(orbitals: np.array) -> List[float]:
    """
    Compute the Givens rotation angles for transforming the orbital matrix `orbitals`.

    The function calculates the Givens rotation angles required to convert
    the input orbitals matrix `orbitals` to a left-aligned diagonal matrix.
    The angles are calculated based on the matrix entries and are returned
    as a list of angles in radians.

    Parameters
    ----------
    orbitals : np.array
        A 2D NumPy array representing the matrix of orbitals. The matrix should have a shape of 
        (n_fermions, n_qubits), where `n_fermions` is the number of rows and `n_qubits` is 
        the number of columns.

    Returns
    -------
    list of float
        A list of Givens rotation angles in radians.

    Raises
    ------
    ValueError
        If the number of rows (n_fermions) is greater than the number of columns (n_qubits).

    Notes
    -----
    The function modifies the `orbitals` matrix in-place during the calculation of the 
    Givens rotation angles.
    """

    n_fermions, n_qubits = orbitals.shape

    if n_fermions > n_qubits:
        raise ValueError(f"n_fermions ({n_fermions}) cannot be greater than n_qubits ({n_qubits}).")

    # perform unitary transformations to reduce the number of Givens rotation operations
    orbitals = _unitary_sparsification(orbitals)

    # Calculate the Givens rotation angles
    gtheta = []
    for irow in range(n_fermions):
        for icol in range(n_qubits - n_fermions + irow, irow, -1):
            if orbitals[irow][icol - 1] != 0.0:
                rate = orbitals[irow][icol] / orbitals[irow][icol - 1]
                uk = 1.0 / np.sqrt(1 + rate ** 2)
                vk = -rate / np.sqrt(1 + rate ** 2)
                angle = np.arccos(uk) if vk >= 0 else -np.arccos(uk)
            else:
                angle = np.pi/2.0
            gtheta.append(angle)

            # Apply the computed rotation angle to the matrix
            for ik in range(n_fermions):
                temp = orbitals[ik][icol - 1]
                orbitals[ik][icol - 1] = temp * np.cos(angle) - orbitals[ik][icol] * np.sin(angle)
                orbitals[ik][icol] = temp * np.sin(angle) + orbitals[ik][icol] * np.cos(angle)

    return gtheta

def get_statevector(orbitals: np.array) -> np.ndarray:
    """
    Compute the statevector from fermionic orbitals.

    Parameters
    ----------
    orbitals : np.array
        A 2D NumPy array representing the matrix of orbitals. The matrix should have a shape of 
        (n_fermions, n_qubits), where `n_fermions` is the number of rows and `n_qubits` is 
        the number of columns.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array of complex numbers representing the statevector of the quantum system. 
        The length of this array is `2**n_qubits`, corresponding to all possible basis states.

    Raises
    ------
    ValueError
        If the number of rows (n_fermions) is greater than the number of columns (n_qubits).

    Notes
    -----
    The statevector is calculated by iterating over all possible combinations of `n_fermions` 
    occupied orbitals and computing the determinant of the corresponding submatrix of `orbitals`.
    """

    n_fermions, n_qubits = orbitals.shape

    if n_fermions > n_qubits:
        raise ValueError(f"n_fermions ({n_fermions}) cannot be greater than n_qubits ({n_qubits}).")

    statevector = np.zeros(2**n_qubits, dtype = np.complex128)
    cof = np.zeros((n_fermions, n_fermions), dtype = np.complex128)
    # Generate all combinations of bit positions with the number of fermions being 1
    for bits in combinations(range(n_qubits), n_fermions):
        indices = list(bits)
        inum = sum(1 << i for i in indices)
        # Update the cof matrix based on the current combination of bit positions
        for i, j in enumerate(indices):
            cof[:, i] = orbitals[:, j]
        # Calculate the determinant and store it in the statevector
        statevector[inum] = np.linalg.det(cof)

    return statevector

def get_analytical_fermi_orbitals(
        n_qubits: int,
        n_fermions: int,
        lattice: str,
        hopping: float,
) -> np.ndarray:
    """
    Compute fermionic orbitals from the analytical solution.

    This function computes a matrix representing fermionic orbitals based on an analytical 
    solution for a free fermions on a cyclic lattice.
    The specific orbitals are found in Eq. (11) of T. Yoshoka et al. arXiv:2312.04710v1 [quant-ph].

    Parameters
    ----------
    n_qubits : int
        The number of qubits, which determines the size of the system.
    n_fermions : int
        The number of fermions, which determines how many rows of the 
        orbital matrix are considered in the computation.
    lattice : str
        The type of lattice configuration. Currently, only 'cyclic' lattice configurations 
        are supported.
    hopping : float
        The hopping parameter, which must be greater than 0. This value represents 
        the amplitude of hopping between lattice sites.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (n_fermions, n_qubits) representing the fermionic orbitals.

    Raises
    ------
    ValueError
        If `n_fermions` is greater than `n_qubits`.
    ValueError
        If the `lattice` is not 'cyclic'.
    ValueError
        If `hopping` is less than or equal to 0.

    Notes
    -----
    The orbitals computed by this function are based on the analytical solutions described in:
    - Eq. (11) in arXiv:2312.04710v1 [quant-ph], https://arxiv.org/pdf/2312.04710

    The function currently only supports systems with a cyclic lattice configuration.
    """

    if n_fermions > n_qubits:
        raise ValueError(f"n_fermions ({n_fermions}) cannot be greater than n_qubits ({n_qubits}).")

    if lattice not in 'cyclic':
        raise ValueError("analytical solutions support only 'cyclic'")

    if hopping <= 0.0: raise ValueError("analytical solutions support hopping > 0")

    orbitals = np.zeros((n_fermions, n_qubits), dtype = np.float64)
    if n_fermions % 2 == 0:
        for jw in range(n_qubits):
            for k in range(int(n_fermions/2.0)):
                k2 = k + int((n_fermions)/2.0)
                angle = jw * 2.0 * np.pi * ((k+0.5) / n_qubits)
                orbitals[ k][jw] = np.sin( angle) * np.sqrt(2.0/n_qubits)
                orbitals[k2][jw] = np.cos(-angle) * np.sqrt(2.0/n_qubits)
    else:
        for jw in range(n_qubits):
            orbitals[0][jw] = np.sqrt(1.0/n_qubits)
            for k in range(int((n_fermions-1)/2.0)):
                k2 = k + int((n_fermions-1)/2.0)
                angle = jw * 2.0 * np.pi * ((k+1) / n_qubits)
                orbitals[ k+1][jw] = np.sin( angle) * np.sqrt(2.0/n_qubits)
                orbitals[k2+1][jw] = np.cos(-angle) * np.sqrt(2.0/n_qubits)

    return orbitals

def get_fermi_orbitals(
        n_qubits: int,
        n_fermions: int,
        lattice: str,
        hopping: float
) -> np.ndarray:
    """
    Compute fermionic orbitals from the Hamiltonian eigenvectors.

    This function generates a matrix representing fermionic orbitals by computing the eigenvectors 
    of a given fermionic mixer Hamiltonian. 

    Parameters
    ----------
    n_qubits : int
        The number of qubits, which corresponds to the number of sites or modes in the system.
    n_fermions : int
        The number of fermions, which determines the number of occupied states in the system.
    lattice : str
        The type of lattice configuration. Only specific lattice types defined in `cyclic` and `chain`
        are supported.
    hopping : float
        The hopping parameter, which defines the amplitude of hopping between adjacent lattice sites. 
        It must be non-zero.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (n_fermions, n_qubits) representing the fermionic orbitals. 
        Each row corresponds to an orbital and each column corresponds to a qubit (site).

    Raises
    ------
    ValueError
        If `n_fermions` is greater than `n_qubits`.
    ValueError
        If the `lattice` type is not recognized (i.e., not in [`cyclic`, `chain`]).
    ValueError
        If `hopping` is zero.

    Returns
    -------
    numpy.ndarray
        matrix representation of Fermionic orbitals.
    
    Notes
    -----
    The orbitals are derived from the eigenvectors of the Hamiltonian corresponding to the given 
    system parameters. The specific eigenvector calculation is handled by the `_get_free_eigen` 
    function, which depends on the system's configuration.
    """

    if n_fermions > n_qubits:
        raise ValueError(f"n_fermions ({n_fermions}) cannot be greater than n_qubits ({n_qubits}).")

    if lattice not in ALLOWED_LATTICE:
        raise ValueError(f"In FQAOA, lattice {lattice} is not recognised. Please use {ALLOWED_LATTICE}")

    if hopping == 0.0: raise ValueError("In FQAOA, hopping = 0 is not recgnized. Please use hopping != 0")

    orbitals = np.zeros((n_fermions, n_qubits), dtype = np.float64)
    eig = _get_free_eigen(n_qubits, n_fermions, lattice, hopping)
    for jw in range(n_qubits):
        for k in range(n_fermions):
            orbitals[k][jw] = eig[jw][k]

    return orbitals

def generate_random_portfolio_data(
        num_assets: int,
        num_days: int,
        seed: Optional[int] = None,
) -> tuple[list[float], list[list[float]], np.ndarray]:
    """
    Generates random portfolio data including mean returns, covariance matrix,
    and historical price movements for a given number of assets and days.

    Parameters
    ----------
    num_assets : int
        The number of assets in the portfolio.
    num_days : int
        The number of days over which the historical data is generated.
    seed : Optional[int], optional
        An optional random seed for reproducibility, by default None.

    Returns
    -------
    mu : List[float]
        The mean returns for each asset.
    sigma : List[List[float]]
        The covariance matrix of the asset returns.
    hist_exp : np.ndarray
        The generated historical price movements for the assets.

    Notes
    -----
    The function simulates historical price movements by generating random data
    influenced by a time trend and random fluctuations, suitable for use in
    portfolio optimization and risk analysis.
    """

    # If a seed is provided, set the random seed
    if seed is not None:
        np.random.seed(seed)

    # Generate historical-like data for multiple assets over a number of days
    random_asset_factors = (1 - 2 * np.random.rand(num_assets)).reshape(-1, 1)
    day_indices = np.array([np.arange(num_days) for i in range(num_assets)]) + np.random.randint(10)
    random_fluctuations = 1 - 2 * np.random.rand(num_assets, num_days)

    # The resulting matrix hist_exp represents the daily returns or price levels of the assets
    hist_exp = random_asset_factors * day_indices + random_fluctuations

    # Calculate the mean returns (mu) for each asset
    # and the covariance matrix (sigma) of the asset returns
    mu = hist_exp.mean(axis=1).tolist()
    sigma = np.cov(hist_exp).tolist()

    return mu, sigma, hist_exp

def _get_free_eigen(
        n_qubits: int,
        n_fermions: int,
        lattice: str,
        hopping: float,
) -> np.ndarray:
    """
    Compute the eigenvectors of the fermionic mixer Hamiltonian for a given lattice and hopping parameter.

    This function constructs the Hamiltonian for a system with a specified number of qubits 
    and fermions, based on the lattice configuration and hopping amplitude. It then computes the 
    eigenvectors of this Hamiltonian matrix.

    Parameters
    ----------
    n_qubits : int
        The number of qubits, corresponding to the size of the Hamiltonian matrix (n_qubits x n_qubits).
    n_fermions : int
        The number of fermions in the system, which affects the phase of the cyclic boundary condition 
        if the lattice is cyclic.
    lattice : str
        The type of lattice configuration. Currently, only 'cyclic' is supported.
    hopping : float
        The hopping parameter that scales the Hamiltonian matrix elements. It defines the amplitude 
        of hopping between adjacent lattice sites.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (n_qubits, n_qubits) containing the eigenvectors of the Hamiltonian matrix.
        Each column of the array represents an eigenvector.

    Notes
    -----
    The Hamiltonian matrix is constructed with nearest-neighbor interactions and optional cyclic boundary 
    conditions, depending on the lattice type. The eigenvectors are computed using `scipy.linalg.eigh`, 
    which returns them in columns.
    """

    fermi_hamiltonian = np.zeros((n_qubits, n_qubits), dtype = np.float64)
    for jw in range(1, n_qubits):
        fermi_hamiltonian[jw, jw-1] = -1.0
    if lattice == 'cyclic':
        fermi_hamiltonian[n_qubits-1, 0] = (-1.0)**n_fermions
    fermi_hamiltonian = fermi_hamiltonian*hopping
    eig = linalg.eigh(fermi_hamiltonian)

    return eig[1]

def _unitary_sparsification(orbitals: np.array) -> np.ndarray:
    """
    Perform a unitary transformation to sparsify a matrix `orbitals`
    by setting the elements in the upper triangular region to zero.

    This method applies a series of Givens rotations to eliminate the upper
    triangular elements of the input matrix `orbitals`. The transformation is
    carried out in-place, modifying `orbitals` directly.

    Parameters
    ----------
    orbitals : numpy.ndarray
        A 2D NumPy array representing the non-square matrix to be transformed.
        The matrix shape is expected to be `(n_fermions, n_qubits)` where 
        `n_fermions <= n_qubits`.

    Returns
    -------
    numpy.ndarray
        The modified matrix `orbitals` with its upper triangular elements set to zero.
    """

    n_fermions, n_qubits = orbitals.shape

    for it in range(n_fermions - 1):
        icol = n_qubits - 1 - it
        for irot in range(n_fermions - 1 - it):
            if orbitals[irot + 1][icol] == 0.0:
                # Swap rows if necessary
                orbitals[irot], orbitals[irot + 1] = orbitals[irot + 1], orbitals[irot]
            else:
                # Apply Givens rotation
                rate = orbitals[irot][icol] / orbitals[irot + 1][icol]
                factor = np.sqrt(1 + rate ** 2)
                for jw in range(n_qubits):
                    orbitals[irot][jw], orbitals[irot + 1][jw] = (
                        (orbitals[irot][jw] - rate * orbitals[irot + 1][jw]) / factor,
                        (orbitals[irot + 1][jw] + rate * orbitals[irot][jw]) / factor,
                    )

    return orbitals
