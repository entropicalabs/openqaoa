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


"""
Test the core functionality in parameters.py 
"""
import numpy as np
import unittest
from scipy.fft import dst, dct

from openqaoa.qaoa_parameters import *
from openqaoa.utilities import X_mixer_hamiltonian

from openqaoa.qaoa_parameters.qaoa_params import create_qaoa_variational_params, PARAMS_CLASSES_MAPPER

register = [0, 1, 2]
terms = [[0, 1], [2], [0, 2]]
weights = [1, 0.5, -2.0]

terms_wo_bias = [[0, 1], [1, 2]]
weights_wo_bias = [1, 0.5]

cost_hamiltonian = Hamiltonian.classical_hamiltonian(
    terms, weights, constant=0)
cost_hamiltonian_wo_bias = Hamiltonian.classical_hamiltonian(terms_wo_bias,
                                                             weights_wo_bias,
                                                             constant=0.7)
mixer_hamiltonian = X_mixer_hamiltonian(len(register))


class TestingQAOAParameters(unittest.TestCase):

    def test_QAOAVariationalExtendedParams(self):
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)
        params = QAOAVariationalExtendedParams.linear_ramp_from_hamiltonian(
            qaoa_circuit_params, time=2)

        assert np.allclose(qaoa_circuit_params.qureg, [0, 1, 2])
        assert np.allclose(params.betas_singles, [[0.75] * 3, [0.25] * 3])
        assert np.allclose(params.betas_pairs, [])
        assert np.allclose(params.gammas_singles, [[0.25], [0.75]])
        assert np.allclose(params.gammas_pairs, [[0.25, 0.25], [0.75, 0.75]])
        # Test updating and raw output
        raw = np.random.rand(len(params))
        params.update_from_raw(raw)
        assert np.allclose(raw, params.raw())

    # TODO check that the values also make sense
    def test_QAOAVariationalExtendedParamsCustomInitialisation(self):
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)

        betas_singles = [[0.0, 0.1, 0.3], [0.5, 0.2, 1.2]]
        betas_pairs = []
        gammas_singles = [[0.0], [0.5]]
        gammas_pairs = [[0.1, 0.3], [0.2, 1.2]]
        params = QAOAVariationalExtendedParams(qaoa_circuit_params,
                                               betas_singles,
                                               betas_pairs,
                                               gammas_singles,
                                               gammas_pairs)

        cost_2q_angles = 2 * np.array([1, -2.0]) * gammas_pairs
        cost_1q_angles = 2 * np.array([0.5]) * gammas_singles
        mixer_1q_angles = 2 * np.array([-1, -1, -1]) * betas_singles
        mixer_2q_angles = []

        assert np.allclose(params.betas_singles, betas_singles)
        assert np.allclose(params.betas_pairs, betas_pairs)
        assert np.allclose(params.gammas_singles, gammas_singles)
        assert np.allclose(params.gammas_pairs, gammas_pairs)

        assert np.allclose(params.mixer_1q_angles, mixer_1q_angles)
        assert np.allclose(params.mixer_2q_angles, mixer_2q_angles)
        assert np.allclose(params.cost_1q_angles, cost_1q_angles)
        assert np.allclose(params.cost_2q_angles, cost_2q_angles)

    def test_QAOAVariationalStandardWithBiasParamsCustomInitialisation(self):
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)

        betas = [np.pi, 0.4]
        gammas_singles = [10, 24]
        gammas_pairs = [8.8, 2.3]
        params = QAOAVariationalStandardWithBiasParams(qaoa_circuit_params,
                                                       betas,
                                                       gammas_singles,
                                                       gammas_pairs)

        cost_2q_angles = 2 * np.outer(gammas_pairs, np.array([1, -2.0]))
        cost_1q_angles = 2 * np.outer(gammas_singles, np.array([0.5]))
        mixer_1q_angles = 2 * np.outer(betas, np.array([-1, -1, -1]))
        mixer_2q_angles = []

        assert np.allclose(params.betas, betas)
        assert np.allclose(params.gammas_singles, gammas_singles)
        assert np.allclose(params.gammas_pairs, gammas_pairs)

        assert np.allclose(params.mixer_1q_angles, mixer_1q_angles)
        assert np.allclose(params.mixer_2q_angles, mixer_2q_angles)
        assert np.allclose(params.cost_1q_angles, cost_1q_angles)
        assert np.allclose(params.cost_2q_angles, cost_2q_angles)
        assert type(params) == QAOAVariationalStandardWithBiasParams

    def test_QAOAVariationalAnnealingParamsCustomInitialisation(self):

        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)
        schedule = [0.4, 1.0]
        annealing_params = QAOAVariationalAnnealingParams(qaoa_circuit_params, total_annealing_time=5,
                                                          schedule=schedule)

        assert type(annealing_params) == QAOAVariationalAnnealingParams

    def test_QAOAVariationalFourierParamsCustomInitialisation(self):

        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)
        v = [0.4, 1.0]
        u = [0.5, 1.2]
        fourier_params = QAOAVariationalFourierParams(
            qaoa_circuit_params, q=2, v=v, u=u)

        assert np.allclose(fourier_params.betas, dct(
            v, n=qaoa_circuit_params.p, axis=0))
        assert np.allclose(fourier_params.gammas, dst(
            u, n=qaoa_circuit_params.p, axis=0))
        assert type(fourier_params) == QAOAVariationalFourierParams

    def test_QAOAVariationalFourierExtendedParamsCustomInitialisation(self):

        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)
        v_singles = [[0.4]*3, [1.0]*3]
        v_pairs = []
        u_singles = [[0.5], [1.2]]
        u_pairs = [[4.5]*2, [123]*2]
        fourier_params = QAOAVariationalFourierExtendedParams(qaoa_circuit_params,
                                                              2,
                                                              v_singles,
                                                              v_pairs,
                                                              u_singles,
                                                              u_pairs)

        assert np.allclose(fourier_params.betas_singles, dct(
            v_singles, n=qaoa_circuit_params.p, axis=0))
        assert np.allclose(fourier_params.gammas_singles, dst(
            u_singles, n=qaoa_circuit_params.p, axis=0))
        assert np.allclose(fourier_params.gammas_pairs, dst(
            u_pairs, n=qaoa_circuit_params.p, axis=0))

        assert type(fourier_params) == QAOAVariationalFourierExtendedParams

    def test_QAOAVariationalAnnealingParams(self):

        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)
        params = QAOAVariationalAnnealingParams.linear_ramp_from_hamiltonian(qaoa_circuit_params,
                                                                             total_annealing_time=2)

        assert np.allclose(qaoa_circuit_params.qureg, [0, 1, 2])
        assert np.allclose(params.mixer_1q_angles, [
                           [-2*0.75] * 3, [-2*0.25] * 3])
        assert np.allclose(params.cost_1q_angles, [[2*0.125], [2*0.375]])
        assert np.allclose(params.cost_2q_angles, [
                           [2*0.25, -0.5*2], [2*0.75, -1.5*2]])

        # Test updating and raw output
        raw = np.random.rand(len(params))
        params.update_from_raw(raw)
        assert np.allclose(raw, params.raw())

    def test_QAOAVariationalStandardWithBiasParams(self):

        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)
        params = QAOAVariationalStandardWithBiasParams.linear_ramp_from_hamiltonian(qaoa_circuit_params,
                                                                                    time=2)
        assert np.allclose(qaoa_circuit_params.qureg, [0, 1, 2])
        assert np.allclose(params.mixer_1q_angles, [
                           [-2*0.75] * 3, [-2*0.25] * 3])
        assert np.allclose(params.cost_1q_angles, [[2*0.125], [2*0.375]])
        assert np.allclose(params.cost_2q_angles, [
                           [2*0.25, -0.5*2], [2*0.75, -1.5*2]])

        # Test updating and raw output
        raw = np.random.rand(len(params))
        params.update_from_raw(raw)
        assert np.allclose(raw, params.raw())

    def test_QAOAVariationalStandardParams(self):

        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)
        params = QAOAVariationalStandardParams.linear_ramp_from_hamiltonian(qaoa_circuit_params,
                                                                            time=2)

        assert np.allclose(qaoa_circuit_params.qureg, [0, 1, 2])
        assert np.allclose(params.mixer_1q_angles, [
                           [-2*0.75] * 3, [-2*0.25] * 3])
        assert np.allclose(params.cost_1q_angles, [[2*0.125], [2*0.375]])
        assert np.allclose(params.cost_2q_angles, [
                           [2*0.25, -0.5*2], [2*0.75, -1.5*2]])

        # Test updating and raw output
        raw = np.random.rand(len(params))
        params.update_from_raw(raw)
        assert np.allclose(raw, params.raw())

    def test_non_fourier_params_are_consistent(self):
        """
        Check that StandardParams, StandardWithBiasParams and
        ExtendedParams give the same rotation angles, given the same data"""
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)
        p1 = QAOAVariationalStandardParams.linear_ramp_from_hamiltonian(
            qaoa_circuit_params, time=2)
        p2 = QAOAVariationalExtendedParams.linear_ramp_from_hamiltonian(
            qaoa_circuit_params, time=2)
        p3 = QAOAVariationalStandardWithBiasParams.linear_ramp_from_hamiltonian(
            qaoa_circuit_params, time=2)

        assert np.allclose(p1.mixer_1q_angles, p2.mixer_1q_angles)
        assert np.allclose(p2.mixer_1q_angles, p3.mixer_1q_angles)
        assert np.allclose(p1.cost_1q_angles, p2.cost_1q_angles)
        assert np.allclose(p2.cost_1q_angles, p3.cost_1q_angles)
        assert np.allclose(p1.cost_2q_angles, p2.cost_2q_angles)
        assert np.allclose(p2.cost_2q_angles, p3.cost_2q_angles)

    def test_QAOAVariationalFourierParams(self):

        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=3)
        params = QAOAVariationalFourierParams.linear_ramp_from_hamiltonian(
            qaoa_circuit_params, q=2, time=2)

        # just access the angles, to check that it actually creates them
        assert len(params.cost_1q_angles) == len(params.cost_2q_angles)
        assert np.allclose(params.v, [1/3, 0])
        assert np.allclose(params.u, [1/3, 0])
        # Test updating and raw output
        raw = np.random.rand(len(params))
        params.update_from_raw(raw)
        assert np.allclose(raw, params.raw())

    def test_QAOAVariationalFourierWithBiasParams(self):

        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=3)
        params = QAOAVariationalFourierWithBiasParams.linear_ramp_from_hamiltonian(qaoa_circuit_params,
                                                                                   q=2,
                                                                                   time=2)

        # just access the angles, to check that it actually creates them
        assert len(params.cost_1q_angles) == len(params.cost_2q_angles)
        assert np.allclose(params.v, [1/3, 0])
        assert np.allclose(params.u_singles, [1/3, 0])
        assert np.allclose(params.u_pairs, [1/3, 0])
        # Test updating and raw output
        raw = np.random.rand(len(params))
        params.update_from_raw(raw)
        assert np.allclose(raw, params.raw())

    def test_QAOAVariationalFourierExtendedParams(self):

        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=3)
        params = QAOAVariationalFourierExtendedParams.linear_ramp_from_hamiltonian(qaoa_circuit_params,
                                                                                   q=2,
                                                                                   time=2)

        # just access the angles, to check that it actually creates them
        assert len(params.cost_1q_angles) == len(params.cost_2q_angles)
        # Test updating and raw output

        assert np.allclose(params.v_singles, [[1/3] * 3, [0] * 3])
        assert np.allclose(params.u_singles, [[1/3] * 1, [0] * 1])
        assert np.allclose(params.u_pairs, [[1/3] * 2, [0] * 2])
        # Test updating and raw output
        raw = np.random.rand(len(params))
        params.update_from_raw(raw)
        assert np.allclose(raw, params.raw())

    def test_FourierParams_are_consistent(self):
        """
        Check, that both Fourier Parametrizations give the same rotation angles,
        given the same data"""
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=3)

        params1 = QAOAVariationalFourierParams.linear_ramp_from_hamiltonian(
            qaoa_circuit_params, q=2, time=2)
        params2 = QAOAVariationalFourierWithBiasParams.linear_ramp_from_hamiltonian(
            qaoa_circuit_params, q=2, time=2)
        params3 = QAOAVariationalFourierExtendedParams.linear_ramp_from_hamiltonian(
            qaoa_circuit_params, q=2, time=2)

        assert np.allclose(params1.mixer_1q_angles, params2.mixer_1q_angles)
        assert np.allclose(params1.cost_1q_angles, params2.cost_1q_angles)
        assert np.allclose(params1.cost_2q_angles, params2.cost_2q_angles)
        assert np.allclose(params1.mixer_1q_angles, params3.mixer_1q_angles)
        assert np.allclose(params1.cost_1q_angles, params3.cost_1q_angles)
        assert np.allclose(params1.cost_2q_angles, params3.cost_2q_angles)

    def test_inputChecking(self):

        # Check that an error is raised if we pass in an extra angle in `betas` (depth is 3, here we give 4 beta values)
        reg = [0, 1]
        terms = [[0, 1]]
        weights = [0.7]
        n_layers = 3
        betas_singles = [1, 2, 3, 4]
        betas_pairs = []
        gammas_singles = []
        gammas_pairs = [1, 2, 3]
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=n_layers)
        self.assertRaises(ValueError,
                          QAOAVariationalExtendedParams,
                          qaoa_circuit_params,
                          betas_singles,
                          betas_pairs,
                          gammas_singles,
                          gammas_pairs)
        
    def test_str_and_repr(self):
        
        qaoa_circuit_params = QAOACircuitParams(
            cost_hamiltonian, mixer_hamiltonian, p=2)
        
        for each_key_value in PARAMS_CLASSES_MAPPER.keys():
            variate_params = create_qaoa_variational_params(qaoa_circuit_params, 
                                                            params_type = each_key_value, 
                                                            init_type = 'rand', 
                                                            q = 1, 
                                                            total_annealing_time = 1)
            
            self.assertEqual(variate_params.__str__(), variate_params.__repr__())
        
        
    # # Plot Tests

    # def test_StandardParams_plot(self):

    #     register_wo_bias = [0, 1, 2]
    #     terms_wo_bias = [[0, 1], [0, 2]]
    #     weights_wo_bias = [1, -2.0]

    #     p = 5
    #     hyperparams = HyperParams(terms,weights,register,p)
    #     params = StandardParams.linear_ramp_from_hamiltonian(hyperparams)
    #     fig, ax = plt.subplots()
    #     params.plot(ax=ax)
    #     # plt.show()

    #     p = 8
    #     hyperparams = HyperParams(terms,weights,register,p)
    #     params = StandardParams(hyperparams,([0.1]*p, [0.2]*p))
    #     fig, ax = plt.subplots()
    #     params.plot(ax=ax)
    #     # plt.show()

    #     p = 2
    #     hyperparams = HyperParams(terms_wo_bias,weights_wo_bias,register_wo_bias,p)
    #     params = StandardParams(hyperparams,([5]*p, [10]*p))
    #     fig, ax = plt.subplots()
    #     params.plot(ax=ax)
    #     # plt.show()

    # def test_ExtendedParams_plot(self):

    #     register_wo_bias = [0, 1, 2]
    #     terms_wo_bias = [[0, 1], [0, 2]]
    #     weights_wo_bias = [1, -2.0]

    #     p = 5
    #     hyperparams = HyperParams(terms_wo_bias,weights_wo_bias,register_wo_bias,p)
    #     params = ExtendedParams.linear_ramp_from_hamiltonian(hyperparams, p)
    #     fig, ax = plt.subplots()
    #     params.plot(ax=ax)
    #     # plt.show()

    #     p = 8
    #     hyperparams = HyperParams(terms_wo_bias,weights_wo_bias,register_wo_bias,p)
    #     params = ExtendedParams(hyperparams,
    #                             ([0.1] * p*len(register),
    #                             [],
    #                             [0.2] * p*len(weights_wo_bias)))
    #     fig, ax = plt.subplots()
    #     params.plot(ax=ax)
    #     # plt.show()

    # def test_extended_get_constraints(self):
    #     register = [0,1,2,3]
    #     terms = [[0, 1], [1, 2], [2, 3], [3, 0]]
    #     weights = [0.1, 0.3, 0.5, -0.7]
    #     p = 2

    #     hyperparams = HyperParams(terms,weights,register,p)
    #     params = ExtendedParams.linear_ramp_from_hamiltonian(hyperparams)

    #     actual_constraints = params.get_constraints()

    #     expected_constraints = [(0, 2 * np.pi), (0, 2 * np.pi),
    #                             (0, 2 * np.pi), (0, 2 * np.pi),
    #                             (0, 2 * np.pi), (0, 2 * np.pi),
    #                             (0, 2 * np.pi), (0, 2 * np.pi),
    #                             (0, 2 * np.pi / weights[0]),
    #                             (0, 2 * np.pi / weights[1]),
    #                             (0, 2 * np.pi / weights[2]),
    #                             (0, 2 * np.pi / weights[3]),
    #                             (0, 2 * np.pi / weights[0]),
    #                             (0, 2 * np.pi / weights[1]),
    #                             (0, 2 * np.pi / weights[2]),
    #                             (0, 2 * np.pi / weights[3])]

    #     assert(np.allclose(expected_constraints, actual_constraints))

    #     cost_function = QAOACostVector(params)
    #     np.random.seed(0)
    #     random_angles = np.random.uniform(-100, 100, size=len(params.raw()))
    #     value = cost_function(random_angles)

    #     normalised_angles = [random_angles[i] % actual_constraints[i][1]
    #                         for i in range(len(params.raw()))]
    #     normalised_value = cost_function(normalised_angles)

    #     assert(np.allclose(value, normalised_value))


if __name__ == "__main__":
    unittest.main()
