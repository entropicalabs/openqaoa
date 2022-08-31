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
from scipy.sparse import diags, kron
import unittest

# RX and CHPHASE are never used
from openqaoa.backends.simulators.qaoa_vectorized import QAOAvectorizedBackendSimulator, _permute_qubits, _get_perm, RX, RZ, CPHASE
from openqaoa.utilities import X_mixer_hamiltonian, ring_of_disagrees
from openqaoa.qaoa_parameters import QAOAVariationalExtendedParams, QAOAVariationalStandardParams, Hamiltonian, PauliOp
from openqaoa.qaoa_parameters.baseparams import QAOACircuitParams
from openqaoa.qaoa_parameters.qaoa_params import create_qaoa_variational_params

"""
A set of tests for FastQAOA: see the notebook Test_Examples.ipynb for explanations of how the
expected answers are derived.
"""

######################################################
# TESTS OF SIMPLE PERMUTATION AND RESHAPING OPERATIONS
######################################################

#no changes except for angle gamma

def Disagrees_SetUp(n_qubits):
    """
    Helper function for the tests below
    """

    register = range(n_qubits)
    p = 1
    cost_hamil = ring_of_disagrees(register)
    mixer_hamil = X_mixer_hamiltonian(n_qubits)

    betas = [np.pi/8]
    gammas = [np.pi/8] #adjusted angle

    qaoa_circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p)
    variational_params_std = QAOAVariationalStandardParams(qaoa_circuit_params,
                                                           betas,
                                                           gammas)
    # Get the part of the Hamiltonian proportional to the identity

    return register, cost_hamil, qaoa_circuit_params, variational_params_std


class TestingQAOAMEBackend(unittest.TestCase):
    """
    Unittest based testing of QAOACostVector 
    """

    def test_permute(self):

        nqubits = 3

        arr = np.arange(2**nqubits)
        arr.shape = [2]*nqubits
        #reshaped_arr = backend._reshape_qubits(nqubits, arr)
        perm = [2, 0, 1]
        permuted_arr = _permute_qubits(arr, perm)

        expected_arr = np.array([[[0, 2], [4, 6]], [[1, 3], [5, 7]]])

        assert np.array_equal(permuted_arr, expected_arr)

    def test_get_perm(self):

        nqubits = 4

        perm1, perminv1 = _get_perm(nqubits, [1, 3])
        perm2, perminv2 = _get_perm(nqubits, [0, 2])

        perm1_expected = np.array([2, 0, 1, 3])
        perminv1_expected = np.array([1, 2, 0, 3])

        perm2_expected = np.array([3, 1, 0, 2])
        perminv2_expected = np.array([2, 1, 3, 0])

        assert np.array_equal(perm1, perm1_expected)
        assert np.array_equal(perminv1, perminv1_expected)
        assert np.array_equal(perm2, perm2_expected)
        assert np.array_equal(perminv2, perminv2_expected)

    ##########################################################
    # TESTS OF BASIC CIRCUIT OPERATIONS (SAME AS FOR PROJECTQ)
    ##########################################################

    #new tests for ME backend

    def test_rod(self):
        classical_terms = [(0, 1),(1, 2),(2, 3), (3, 0)] # 4 node ring of disagrees
        coeffs = [1, 1, 1, 1]
        constant = 0
        cost_hamil = Hamiltonian.classical_hamiltonian(terms=classical_terms,coeffs=coeffs,constant=constant)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=4)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        variate_params = create_qaoa_variational_params(circuit_params,'standard','rand')
        variate_params = QAOAVariationalStandardParams(circuit_params, betas=[np.pi/8], gammas=[np.pi/8])
        mc_sim = create_device('local','mcsolver')

        mc_backend = get_qaoa_backend(circuit_params=circuit_params, device=mc_sim, n_shots=10)
        result = me_backend.expectation(variate_params)

        mc_backend = get_qaoa_backend(circuit_params=circuit_params, device=mc_sim, n_shots=100, noise_model={'decay': False, 'dephasing': False, 'overrot': False, 'spam': False, 'readout01': False, 'readout10': False, 'depol1': gate_error_list[0], 'depol2': gate_error_list[1]}, times=t_gate_list + [t_readout], allowed_jump_qubits=None, target_basis = ['id', 'x', 'sx', 'rz', 'cx'])
        depol_result = mc_backend.expectation(variate_params)
        mc_backend = get_qaoa_backend(circuit_params=circuit_params, device=mc_sim, n_shots=100, noise_model={'decay': T1, 'dephasing': False, 'overrot': False, 'spam': False, 'readout01': False, 'readout10': False, 'depol1': False, 'depol2': False}, times=t_gate_list + [t_readout], allowed_jump_qubits=None, target_basis = ['id', 'x', 'sx', 'rz', 'cx'])
        decay_result = mc_backend.expectation(variate_params)
        mc_backend = get_qaoa_backend(circuit_params=circuit_params, device=mc_sim, n_shots=100, noise_model={'decay': False, 'dephasing': T2, 'overrot': False, 'spam': False, 'readout01': False, 'readout10': False, 'depol1': False, 'depol2': False}, times=t_gate_list + [t_readout], allowed_jump_qubits=None, target_basis = ['id', 'x', 'sx', 'rz', 'cx'])
        dephasing_result = mc_backend.expectation(variate_params)
        mc_backend = get_qaoa_backend(circuit_params=circuit_params, device=mc_sim, n_shots=100, noise_model={'decay': False, 'dephasing': False, 'overrot': False, 'spam': spam_error_prob, 'readout01': False, 'readout10': False, 'depol1': False, 'depol2': False}, times=t_gate_list + [t_readout], allowed_jump_qubits=None, target_basis = ['id', 'x', 'sx', 'rz', 'cx'])
        spam_result = mc_backend.expectation(variate_params)
        mc_backend = get_qaoa_backend(circuit_params=circuit_params, device=mc_sim, n_shots=100, noise_model={'decay': False, 'dephasing': False, 'overrot': False, 'spam': False, 'readout01': meas0_prep1_prob, 'readout10': meas1_prep0_prob, 'depol1': False, 'depol2': False}, times=t_gate_list + [t_readout], allowed_jump_qubits=None, target_basis = ['id', 'x', 'sx', 'rz', 'cx'])
        readout_result = mc_backend.expectation(variate_params)
        mc_backend = get_qaoa_backend(circuit_params=circuit_params, device=mc_sim, n_shots=100, noise_model={'decay': False, 'dephasing': False, 'overrot': overrot_st_dev, 'spam': False, 'readout01': False, 'readout10': False, 'depol1': False, 'depol2': False}, times=t_gate_list + [t_readout], allowed_jump_qubits=None, target_basis = ['id', 'x', 'sx', 'rz', 'cx'])
        overrot_results.append(me_backend.expectation(variate_params))

        assert np.allclose(result, -2, atol=0.03)
        assert np.allclose(depol_result, -1.4927636376002065, atol=0.03)
        assert np.allclose(decay_result, -1.9164556120225553, atol=0.03)
        assert np.allclose(dephasing_result, -1.959752905192587, atol=0.03)
        assert np.allclose(spam_result, -1.6374608581741066, atol=0.03)
        assert np.allclose(readout_result, -1.8062489855953998, atol=0.03)
        assert np.allclose(overrot_result, -1.6738008048133115, atol=0.03) 

if __name__ == "__main__":
    unittest.main()

