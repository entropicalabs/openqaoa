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


def Disagrees_SetUp(n_qubits):
    """
    Helper function for the tests below
    """

    register = range(n_qubits)
    p = 1
    cost_hamil = ring_of_disagrees(register)
    mixer_hamil = X_mixer_hamiltonian(n_qubits)

    betas = [np.pi/8]
    gammas = [np.pi/4]

    qaoa_circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p)
    variational_params_std = QAOAVariationalStandardParams(qaoa_circuit_params,
                                                           betas,
                                                           gammas)
    # Get the part of the Hamiltonian proportional to the identity

    return register, cost_hamil, qaoa_circuit_params, variational_params_std


class TestingQAOAvectorizedBackend(unittest.TestCase):
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

    def test_qaoa_circuit(self):

        # Test circuit with p = 1 on 3 qubits
        # Performs a round of ZZ rotations through pi, and a round of X mixer rotations through pi
        nqubits = 3
        register = [i for i in range(nqubits)]
        p = 1
        bias_qubits = []
        bias_angles = []
        pairs = [[0, 1], [0, 2], [1, 2]]
        weights = [1, 1, 1]
        pairs_angles = [np.pi]  # [[np.pi]*len(pairs)]
        mixer_angles = [np.pi]  # [[np.pi]*nqubits]

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            pairs, weights, constant=0)
        mixer_hamiltonian = X_mixer_hamiltonian(nqubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalStandardParams(qaoa_circuit_params,
                                                               mixer_angles,
                                                               pairs_angles)

        backend_vectorized = QAOAvectorizedBackendSimulator(
            qaoa_circuit_params, prepend_state=None, append_state=None, init_hadamard=True)

        backend_vectorized.qaoa_circuit(variational_params_std)
        wf = backend_vectorized.wavefn
        wf.shape = 2**nqubits
        wf = wf/wf[0]

        expected_wf = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        assert np.allclose(wf, expected_wf)

    def test_wavefunction_single_terms(self):

        # Test wavefunction and expectation values with hamiltonian object, without 2-qubit terms
        cost_hamil = Hamiltonian(
            [PauliOp('Z', (0,)), PauliOp('Z', (1,))], [1, 1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        circuit_params = QAOACircuitParams(cost_hamil, mixer_hamil, p=1)
        variate_params = create_qaoa_variational_params(
            circuit_params, 'standard', 'ramp')
        backend_obj = QAOAvectorizedBackendSimulator(
            circuit_params, None, None, True)

        args = [np.pi/4, np.pi/4]  # beta, gamma
        variate_params.update_from_raw(args)

        assert np.allclose(backend_obj.wavefunction(
            variate_params), [0, 0, 0, 1j])
        assert np.isclose(backend_obj.expectation(variate_params), -1)

    def test_wavefunction(self):

        nqubits = 3

        # Test circuit with p = 1 on 3 qubits
        terms = [[0, 1], [0, 2], [0]]
        weights = [1, 1, -0.5]
        register = [0, 1, 2]
        p = 1

        betas_singles = [np.pi, 0, 0]
        betas_pairs = []
        gammas_singles = [np.pi]
        gammas_pairs = [[np.pi/2]*2]

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            terms, weights, constant=0)
        mixer_hamiltonian = X_mixer_hamiltonian(nqubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_ext = QAOAVariationalExtendedParams(qaoa_circuit_params,
                                                               betas_singles,
                                                               betas_pairs,
                                                               gammas_singles,
                                                               gammas_pairs)

        backend_vectorized = QAOAvectorizedBackendSimulator(qaoa_circuit_params, prepend_state=None,
                                                            append_state=None, init_hadamard=True)

        wf = backend_vectorized.wavefunction(variational_params_ext)

        cost_op1 = diags([-1, 1, 1, -1, -1, 1, 1, -1], 0, format='csc')
        cost_op2 = diags([-1, 1, -1, 1, 1, -1, 1, -1], 0, format='csc')
        cost_op3 = -1j*diags([-1, 1, -1, 1, -1, 1, -1, 1], 0, format='csc')

        # Factors of 2 needed to produce a rotation for total
        mixer = kron(RX(0), kron(RX(0), RX(-2*np.pi)))
        # time of pi (pi-pulse) on the Bloch sphere

        input_wf = np.ones(2**nqubits)/np.sqrt(2**nqubits)
        direct_wf = -mixer @ cost_op3 @ cost_op2 @ cost_op1 @ input_wf
        
        expected_wf = -1j*np.array([-1, 1, 1, -1, 1, -1, -1, 1])/(2*np.sqrt(2))
        
        assert np.allclose(wf, direct_wf)
        assert np.allclose(wf, expected_wf)
            
    def test_execute_exp_val(self):

        n_qubits = 8
        register, cost_hamil, qaoa_circuit_params, variate_params = Disagrees_SetUp(
            n_qubits)

        backend_vectorized = QAOAvectorizedBackendSimulator(qaoa_circuit_params, prepend_state=None,
                                                            append_state=None, init_hadamard=True)
        exp_val, std_dev1 = backend_vectorized.expectation_w_uncertainty(
            variate_params)

        # Check correct expecation value
        assert np.isclose(exp_val, -6)

        # Check standard deviation
        # Get the matrix form of the Hamiltonian (note we just keep the diagonal part) and square it
        ham_matrix = np.zeros((2**len(register)))
        for i, term in enumerate(cost_hamil.terms):
            out = np.real(cost_hamil.coeffs[i])
            for qubit in register:
                if qubit in term.qubit_indices:
                    out = np.kron([1, -1], out)
                else:
                    out = np.kron([1, 1], out)
            ham_matrix += out
        ham_matrix += cost_hamil.constant

        ham_matrix_sq = np.square(ham_matrix)

        # Get the wavefunction
        wf = backend_vectorized.wavefunction(variate_params)

        # Get the probabilities
        probs = [np.abs(el)**2 for el in wf]

        # Standard deviation
        exp_2 = np.dot(probs, ham_matrix)
        std_dev2 = np.sqrt(np.dot(probs, ham_matrix_sq) - exp_2**2)

        assert np.isclose(std_dev1, std_dev2)

    def test_cost_call(self):
        """
        testing the __call__ method of the base class.
        Only for vectorized and Qiskit Local Statevector Backends.
        """

        n_qubits = 8
        register = range(n_qubits)
        p = 1

        betas = [np.pi/8]
        gammas = [np.pi/4]
        cost_hamiltonian = ring_of_disagrees(register)
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalStandardParams(
            qaoa_circuit_params, betas, gammas)

        backend_vectorized = QAOAvectorizedBackendSimulator(qaoa_circuit_params, prepend_state=None,
                                                            append_state=None, init_hadamard=True)

        exp_vec = backend_vectorized.expectation(variational_params_std)

        assert np.isclose(exp_vec, -6)
  
    def test_get_wavefunction(self):
        
        n_qubits = 3
        terms = [[0, 1], [0, 2], [0]]
        weights = [1, 1, -0.5]
        p = 1
        
        betas_singles = [np.pi,0,0]
        betas_pairs = []
        gammas_singles = [np.pi]
        gammas_pairs = [[1/2*np.pi]*2]
        
        cost_hamiltonian = Hamiltonian.classical_hamiltonian(terms=terms, coeffs=weights, constant=0)
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)
        qaoa_circuit_params = QAOACircuitParams(cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalExtendedParams(qaoa_circuit_params,
                                                               betas_singles=betas_singles,
                                                               betas_pairs=betas_pairs,
                                                               gammas_singles=gammas_singles,
                                                               gammas_pairs=gammas_pairs) 

        backend_vectorised_statevec = QAOAvectorizedBackendSimulator(qaoa_circuit_params,prepend_state=None,
                                                            append_state=None,init_hadamard=True)

        wf_vectorised_statevec = backend_vectorised_statevec.wavefunction((variational_params_std))
        expected_wf = 1j*np.array([-1,1,1,-1,1,-1,-1,1])/(2*np.sqrt(2))
        
        try:
            assert np.allclose(wf_vectorised_statevec, expected_wf)
        except AssertionError:
            assert np.allclose(np.real(np.conjugate(wf_vectorised_statevec)*wf_vectorised_statevec),
                                   np.conjugate(expected_wf)*expected_wf)
            
    def test_exact_solution(self):
        """
        NOTE:Since the implementation of exact solution is backend agnostic
            Checking it once should be okay. 

        Nevertheless, for the sake of completeness it will be tested for all backend
        instances.

        """

        n_qubits = 8
        register = range(n_qubits)
        p = 1

        correct_energy = -8
        correct_config = [0, 1, 0, 1, 0, 1, 0, 1]

        # The tests pass regardless of the value of betas and gammas is this correct?
        betas = [np.pi/8]
        gammas = [np.pi/4]

        cost_hamiltonian = ring_of_disagrees(register)
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalStandardParams(qaoa_circuit_params,
                                                               betas, gammas)

        backend_vectorized = QAOAvectorizedBackendSimulator(qaoa_circuit_params, prepend_state=None,
                                                            append_state=None, init_hadamard=True)
        # exact solution is defined as the property of the cost function
        energy_vec, config_vec = backend_vectorized.exact_solution

        assert np.isclose(energy_vec, correct_energy)

        config_vec = [config.tolist() for config in config_vec]

        assert correct_config in config_vec

    # ADD TESTS FOR PREPEND AND APPEND STATES BELOW
    # def test_with_init_prog_A(self):

    #     """
    #     Checks the trivial case that when we use a given set of params (params_std) as the
    #     init_prog followed by a circuit whose angles are zero (new_params), we should get the same as
    #     if we run the program with no init_prog, but where the angles are given by params_std
    #     """

    #     n_qubits = 8
    #     _,_,_, params_std = Disagrees_SetUp(n_qubits)

    #     new_params = copy(params_std)
    #     new_params.betas = [0]
    #     new_params.gammas = [0]

    #     vector_cost = QAOACostVector(params_std, init_prog=params_std)
    #     exp_val = vector_cost.execute_exp_val(new_params)

    #     # Check correct expecation value
    #     assert np.isclose(exp_val, -6)

    # def test_with_init_prog_B(self):

    #     """
    #     Set up a problem with p = 2 and solve directly
    #     Set up the same problem with an init_prog corresponding to the p=1 params,
    #     and the program to execute being the p=2 params.
    #     """

    #     n = 4
    #     p = 2
    #     register = range(n)
    #     terms = [[0, 1], [0, 2], [0, 3], [3, 0], [1, 3]]
    #     weights = [0.1, 1, -0.5, 0.3, 3]

    #     # Direct way for p=2
    #     hyperparams = HyperParams(terms,weights,p=p)
    #     stand_params = StandardParams.linear_ramp_from_hamiltonian(hyperparams)
    #     vector_cost = QAOACostVector(stand_params)
    #     wf_direct = vector_cost.wavefunction(stand_params)

    #     # Indirect way
    #     init_hyperparams = HyperParams(terms,weights,p=1)
    #     init_betas = stand_params.betas[0]
    #     init_gammas = stand_params.gammas[0]
    #     init_prog = StandardParams(init_hyperparams, (init_betas, init_gammas))

    #     betas_step2 = stand_params.betas[1]
    #     gammas_step2 = stand_params.gammas[1]
    #     params_step2 = StandardParams(init_hyperparams,(betas_step2, gammas_step2))
    #     vector_cost = QAOACostVector(params_step2, init_prog = init_prog)
    #     wf_indirect = vector_cost.wavefunction(params_step2)

    #     assert np.allclose(wf_direct, wf_indirect)

    # def test_with_init_prog_C(self):

    #     """
    #     Similar to above, but uses a random example with ExtendedParams
    #     """

    #     n = 8
    #     p = 3
    #     reg = range(n)
    #     terms, weights = zip(*random_hamiltonian(reg).items())
    #     terms, weights = list(terms),list(weights)
    #     hyperparams = HyperParams(terms,weights,p=p)
    #     abs_params = AbstractParams(hyperparams)
    #     n_gammas_singles = len(abs_params.qubits_singles)
    #     n_gammas_pairs = len(abs_params.qubits_pairs)

    #     # Direct way
    #     betas = np.random.rand(p,n)
    #     gammas_singles = np.random.rand(p,n_gammas_singles)
    #     gammas_pairs = np.random.rand(p,n_gammas_pairs)
    #     params = (betas,gammas_singles, gammas_pairs)
    #     ext_params = ExtendedParams.from_AbstractParameters(abs_params,params)
    #     vector_cost = QAOACostVector(ext_params)
    #     wf_direct = vector_cost.wavefunction(ext_params)

    #     # Indirect way: pass in the p=2 state as the init_prog to the final step
    #     init_hyperparams = HyperParams(terms,weights,p=2)
    #     init_betas = [ext_params.betas[0], ext_params.betas[1]]
    #     init_gammas_singles = [ext_params.gammas_singles[0], ext_params.gammas_singles[1]]
    #     init_gammas_pairs = [ext_params.gammas_pairs[0], ext_params.gammas_pairs[1]]
    #     init_prog = ExtendedParams(init_hyperparams, (init_betas, init_gammas_singles, init_gammas_pairs))

    #     hyperparams3 = HyperParams(terms,weights,p=1)
    #     betas_step3 = ext_params.betas[2]
    #     gammas_singles_step3 = ext_params.gammas_singles[2]
    #     gammas_pairs_step3 = ext_params.gammas_pairs[2]
    #     params_step3 = ExtendedParams(hyperparams3,(betas_step3, gammas_singles_step3, gammas_pairs_step3))
    #     vector_cost = QAOACostVector(params_step3, init_prog = init_prog)
    #     wf_indirect = vector_cost.wavefunction(params=params_step3)

    #     assert np.allclose(wf_direct, wf_indirect)


if __name__ == "__main__":
    unittest.main()

######################################
# TEST CONSISTENCY WITH ENTROPICA_QAOA
######################################

# def test_consistent_entropica_qaoa():

#     """
#     Compare the outcome of VectorSim from FastQAOA with that of EntropicaQAOA
#     Tests the outcome with two sets of initial parameters: linear ramp (for the problem at hand), and random params
#     We check that the expectation values and standard deviations are all consistent
#     """

#     n = 4
#     p = 3
#     ham = PauliSum.from_compact_str('(0.1)*Z0*Z1 + (1)*Z0*Z2 + (-0.5)*Z0*Z3 + (0.2)*Z3*Z0 + (3)*Z1*Z3')

#     # Define the random parameters
#     np.random.seed(123)
#     rand_params = np.random.uniform(0, 2 * np.pi, size=2 * p)

#     # VectorSim
#     stand_params_vec = StandardParams.linear_ramp_from_hamiltonian(hamiltonian=ham, n_steps=p)
#     sim_vec = QAOASim_Vector(n)
#     vec_cf = CostFunction(hamiltonian=ham, params=stand_params_vec, backend=sim_vec)

#     # Expectations and standard deviations for vector sim for both sets of parameters
#     #vec_exp_ramp, vec_dev_ramp = vec_cf(stand_params_vec.raw())
#     #vec_exp_rand, vec_dev_rand = vec_cf(rand_params)
#     vec_exp_ramp = vec_cf(stand_params_vec.raw())
#     vec_exp_rand = vec_cf(rand_params)

#     # Similarly for EntropicaQAOA
#     stand_params_ent = StandardParams.linear_ramp_from_hamiltonian(hamiltonian=ham, n_steps=p)
#     sim_ent = WavefunctionSimulator()
#     ent_cf = QAOACostFunctionOnWFSim(hamiltonian=ham, params=stand_params_ent, sim=sim_ent)

#     ent_exp_ramp = ent_cf(stand_params_ent.raw())
#     ent_exp_rand = ent_cf(rand_params)

#     # Check consistency
#     assert np.isclose(ent_exp_ramp, vec_exp_ramp)
#     assert np.isclose(vec_exp_rand, ent_exp_rand)
    #assert np.isclose(ent_dev_ramp, vec_dev_ramp)
    #assert np.isclose(vec_dev_rand, ent_dev_rand)
