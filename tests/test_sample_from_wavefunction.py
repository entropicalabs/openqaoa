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

import unittest
import numpy as np

from openqaoa.backends.qaoa_backend import get_qaoa_backend
from openqaoa.qaoa_parameters import QAOAVariationalStandardParams, Hamiltonian
from openqaoa.qaoa_parameters.baseparams import QAOACircuitParams
from openqaoa.utilities import random_classical_hamiltonian, X_mixer_hamiltonian
from openqaoa.devices import DeviceLocal


class TestGetSamplesMethod(unittest.TestCase):
    """
    Test the `get_samples` method in `cost_function.py` 
    -> Test whether it produces the right results while sampling the wavefunction
    """

    def test_samples_with_one_string(self):
        """
        This function generates samples for a wavefunction that has 
        unit probability for one a basis state and zero for others.
        In other words, the final wavefunction is a product state
        in the computational basis.
        """
        # 6-qubit hamiltonian with only linear terms and no quadratic terms
        reg = [0, 1, 2, 3, 4, 5]
        solution = bin(0)[2:].zfill(len(reg))
        terms = [(i,) for i in reg]
        weights = [1]*len(terms)
        p = 1
        betas, gammas = [np.pi/4], [-np.pi/4]

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            terms, weights, constant=0)
        mixer_hamiltonian = X_mixer_hamiltonian(len(reg))
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalStandardParams(
            qaoa_circuit_params, betas, gammas)

        backend_vectorized = get_qaoa_backend(
            qaoa_circuit_params, DeviceLocal('vectorized'))
        backend_qiskit_statevec = get_qaoa_backend(
            qaoa_circuit_params, DeviceLocal('qiskit.statevector_simulator'))

        shot_results_vec = backend_vectorized.sample_from_wavefunction(
            variational_params_std, n_samples=15)
        shot_results_qiskit = backend_qiskit_statevec.sample_from_wavefunction(
            variational_params_std, n_samples=15)

        bool_list_vec = [x == solution for x in shot_results_vec]
        bool_list_qiskit = [x == solution for x in shot_results_qiskit]

        assert all(bool_list_vec)
        assert all(bool_list_qiskit)

    def test_samples_limiting_case(self):
        """
        Check whether sample distribution approximates the probability vector in the limit of 
        large number of shots

        NOTE: if the assertion error still keeps failing for no apparent reason, try changing
              ``decimal`` argument in `np.testing.assert_array_almost_equal` to something lower!
        """
        # testing a large number of shots!
        nshots = 500000

        # random 6-qubit Hamiltonian with betas, gammas = 0,0 => only hadamards
        reg = [0, 1, 2, 3, 4, 5]
        p = 1

        betas, gammas = ([0], [0])

        cost_hamiltonian = random_classical_hamiltonian(reg)
        n_qubits = cost_hamiltonian.n_qubits
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalStandardParams(
            qaoa_circuit_params, betas, gammas)

        backend_vectorized = get_qaoa_backend(
            qaoa_circuit_params, DeviceLocal('vectorized'))
        backend_qiskit_statevec = get_qaoa_backend(
            qaoa_circuit_params, DeviceLocal('qiskit.statevector_simulator'))

        # wf_vec = backend_vectorized.wavefunction(variational_params_std)
        prob_wf_vec = np.array(list(backend_vectorized.probability_dict(
            variational_params_std).values()), dtype=float)

        # wf_qiskit = backend_qiskit_statevec.wavefunction(variational_params_std)
        prob_wf_qiskit = np.array(list(backend_qiskit_statevec.probability_dict(
            variational_params_std).values()), dtype=float)

        samples_vec = backend_vectorized.sample_from_wavefunction(
            variational_params_std, n_samples=nshots)
        samples_qiskit = backend_qiskit_statevec.sample_from_wavefunction(
            variational_params_std, n_samples=nshots)

        samples_dict_vec = {bin(x)[2:].zfill(
            n_qubits): 0 for x in range(2**n_qubits)}
        samples_dict_qiskit = {bin(x)[2:].zfill(
            n_qubits): 0 for x in range(2**n_qubits)}

        for shot_result in samples_vec:
            samples_dict_vec[shot_result] += 1/nshots

        for shot_result in samples_qiskit:
            samples_dict_qiskit[shot_result] += 1/nshots

        samples_prob_vec = np.array(
            list(samples_dict_vec.values()), dtype=float)
        samples_prob_qiskit = np.array(
            list(samples_dict_qiskit.values()), dtype=float)

        np.testing.assert_array_almost_equal(
            prob_wf_vec, samples_prob_vec, decimal=3)
        np.testing.assert_array_almost_equal(
            prob_wf_qiskit, samples_prob_qiskit, decimal=3)

    # def testing_w_init_prog(self):
    # 	"""
    # 	This function creates an init_prog and then implements a Hamiltonian
    # 	with zero betas and gammas.
    # 	"""
    # 	reg = [0,1,2,3,4,5]
    # 	init_terms = [(i,) for i in reg]
    # 	init_weights = [1]*len(init_terms)
    # 	init_p=2

    # 	init_params = ([[np.pi/4]*len(reg),
    # 					[np.pi/2,0,0,0,0,np.pi/2]],
    # 					[[-np.pi/4]*len(init_terms),
    # 					[0]*len(reg)],
    # 					[[],[]])
    # 	hyperparams_init = HyperParams(init_terms,init_weights,reg,init_p)
    # 	init_prog = ExtendedParams(hyperparams_init,init_params)

    # 	terms,weights = zip(*random_hamiltonian(reg).items())
    # 	terms, weights = list(terms),list(weights)
    # 	p=1
    # 	trainable_params = ([0],[0])
    # 	hyperparams = HyperParams(terms,weights,reg,p)

    # 	qaoa_params = StandardParams(hyperparams,trainable_params)
    # 	cost_fn = QAOACostVector(qaoa_params,init_prog=init_prog)

    # 	solution = '100001'
    # 	shot_results = cost_fn.get_samples(qaoa_params.raw(), nshots=15)
    # 	bool_list = [x == solution for x in shot_results]

    # 	assert all(bool_list)


if __name__ == "__main__":
    unittest.main()
