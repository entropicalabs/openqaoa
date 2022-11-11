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

from openqaoa.workflows.optimizer import QAOA, RQAOA
from openqaoa.backends.qaoa_backend import (DEVICE_NAME_TO_OBJECT_MAPPER,
                                            DEVICE_ACCESS_OBJECT_MAPPER)
from openqaoa.devices import create_device,SUPPORTED_LOCAL_SIMULATORS
import unittest
import networkx as nw
import numpy as np
import itertools
import os

from openqaoa.problems.problem import MinimumVertexCover, QUBO
from openqaoa.qaoa_parameters.operators import Hamiltonian

ALLOWED_LOCAL_SIMUALTORS = SUPPORTED_LOCAL_SIMULATORS


class TestingResultOutputs(unittest.TestCase):

    """
    Test the Results Output after an optimization loop
    """

    def test_flags_result_outputs_workflow(self):
        
        """
        Run an optimization problem for 5 iterations.
        Should expect certain fields of the results output to be filled based 
        on some of the users inputs. (Default settings)
        Can be checked for cobyla.
        
        Check for all available supported local backends.
        """
        
        g = nw.circulant_graph(3, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()
        
        choice_combination = list(itertools.product([True, False], [True, False], [True, False]))
        recorded_evals = [0, 5]
        
        for device_name in ALLOWED_LOCAL_SIMUALTORS:
            
            for each_choice in choice_combination:
            
                q = QAOA()
                q.set_classical_optimizer(method = 'cobyla', 
                                          parameter_log = each_choice[0],
                                          cost_progress = each_choice[1],
                                          optimization_progress = each_choice[2], 
                                          maxiter = 5)
                device = create_device('local', device_name)
                q.set_device(device)
                q.compile(vc)
                q.optimize()
                self.assertEqual(recorded_evals[each_choice[0]], len(q.results.intermediate['angles log']))
                self.assertEqual(recorded_evals[each_choice[1]], len(q.results.intermediate['intermediate cost']))
                self.assertEqual(recorded_evals[each_choice[2]], len(q.results.intermediate['intermediate measurement outcomes']))


class TestingRQAOAResultOutputs(unittest.TestCase):
    """
    Test the  Results Output after a full RQAOA loop
    """        

    def _run_rqaoa(self, type='custom', eliminations=1, p=1, param_type='standard', mixer='x', method='cobyla', maxiter=15, name_device='qiskit.statevector_simulator'):
        """
        private function to run the RQAOA
        """

        n_qubits = 6
        n_cutoff = 3
        g = nw.circulant_graph(n_qubits, [1])
        problem = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()

        r = RQAOA()
        qiskit_device = create_device(location='local', name=name_device)
        r.set_device(qiskit_device)
        if type == 'adaptive':
            r.set_rqaoa_parameters(n_cutoff = n_cutoff, n_max=eliminations, rqaoa_type=type)
        else:
            r.set_rqaoa_parameters(n_cutoff = n_cutoff, steps=eliminations, rqaoa_type=type)
        r.set_circuit_properties(p=p, param_type=param_type, mixer_hamiltonian=mixer)
        r.set_backend_properties(prepend_state=None, append_state=None)
        r.set_classical_optimizer(method=method, maxiter=maxiter, optimization_progress=True, cost_progress=True, parameter_log=True)
        r.compile(problem)
        r.optimize()

        return r.results
    
    def test_rqaoa_result_outputs(self):
        """
        Test the result outputs for the RQAOA class
        """

        n_qubits = 6
        n_cutoff = 3

        # Test for the standard RQAOA
        results = self._run_rqaoa()
        for key in results['solution'].keys():
            assert len(key) == n_qubits, 'Number of qubits solution is not correct'
        assert isinstance(results['classical_output']['minimum_energy'], float)
        assert isinstance(results['classical_output']['optimal_states'], list)
        for rule in results['elimination_rules']:
            assert isinstance(rule, dict), 'Elimination rule item is not a dictionary'
        assert isinstance(results['schedule'], list), 'Schedule is not a list'
        assert sum(results['schedule']) + n_cutoff == n_qubits, 'Schedule is not correct'
        for step in results['intermediate_steps']:
            assert isinstance(step['QUBO'], QUBO), 'QUBO is not of type QUBO'
            assert isinstance(step['QAOA'], QAOA), 'QAOA is not of type QAOA'
        assert isinstance(results['number_steps'], int), 'Number of steps is not an integer'

        # Test for the adaptive RQAOA
        results = self._run_rqaoa(type='adaptive')
        for key in results['solution'].keys():
            assert len(key) == n_qubits, 'Number of qubits solution is not correct'
        assert isinstance(results['classical_output']['minimum_energy'], float)
        assert isinstance(results['classical_output']['optimal_states'], list)
        for rule in results['elimination_rules']:
            assert isinstance(rule, dict), 'Elimination rule item is not a dictionary'
        assert isinstance(results['schedule'], list), 'Schedule is not a list'
        assert sum(results['schedule']) + n_cutoff == n_qubits, 'Schedule is not correct'
        for step in results['intermediate_steps']:
            assert isinstance(step['QUBO'], QUBO), 'QUBO is not of type QUBO'
            assert isinstance(step['QAOA'], QAOA), 'QAOA is not of type QAOA'
        assert isinstance(results['number_steps'], int), 'Number of steps is not an integer'


    def test_rqaoa_result_methods_steps(self):
        """
        Test the methods for the RQAOAResult class for the steps
        """

        # run the RQAOA
        results = self._run_rqaoa()

        # angles that we should get
        optimized_angles_to_find_list = [[0.34048594327263326, 0.3805304635645852], [0.4066391532372541, 0.3764245401202528], [0.8574965024416041, -0.5645176360484713]]

        # test the methods
        for i in range(results['number_steps']):
            step = results.get_step(i)
            assert isinstance(step, dict), 'Step is not a dictionary'
            assert isinstance(step['QAOA'], QAOA), 'QAOA is not of type QAOA'
            assert isinstance(step['QUBO'], QUBO), 'QUBO is not of type QUBO'

            qaoa = results.get_qaoa_step(i)
            assert isinstance(qaoa, QAOA), 'QAOA is not of type QAOA'

            optimized_angles_to_find = optimized_angles_to_find_list[i]
            optimized_angles = results.get_qaoa_step_optimized_angles(i)
            assert optimized_angles == optimized_angles_to_find, 'Optimized angles are not correct'

            problem = results.get_problem_step(i)
            assert isinstance(problem, QUBO), 'QUBO is not of type QUBO'

            hamiltonian = results.get_hamiltonian_step(i)
            assert isinstance(hamiltonian, Hamiltonian), 'Hamiltonian is not of type Hamiltonian'


    #test dumps
    def test_rqaoa_result_dumps(self):
        """
        Test the dumps for the RQAOAResult class
        """

        # string that should be equal to the dump
        string_to_find = '{"solution": {"101010": 3.0}, "classical_output": {"minimum_energy": -5.5, "optimal_states": ["110"]}, "elimination_rules": [{"(0, 1)": -1.0}, {"(1, 2)": -1.0}, {"(None, 2)": -1.0}], "schedule": [1, 1, 1], "intermediate_steps": [{"QUBO": {"terms": [[0, 4], [0, 1], [1, 2], [2, 3], [3, 4], [1], [2], [3], [4]], "weights": [2.5, -2.5, 2.5, 2.5, 2.5, 4.5, 4.5, 4.5, 4.5], "constant": 0, "_n": 5}, "QAOA": {"method": "cobyla", "cost_hamiltonian": {"n_qubits": 6, "terms": [{"qubit_indices": [0, 5], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [0, 1], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [1, 2], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [2, 3], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [3, 4], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [4, 5], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [0], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [1], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [2], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [3], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [4], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [5], "pauli_str": "Z", "phase": 1}], "coeffs": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5], "constant": 18.0, "qubits_pairs": [{"qubit_indices": [0, 5], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [0, 1], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [1, 2], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [2, 3], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [3, 4], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [4, 5], "pauli_str": "ZZ", "phase": 1}], "qubits_singles": [{"qubit_indices": [0], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [1], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [2], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [3], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [4], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [5], "pauli_str": "Z", "phase": 1}], "single_qubit_coeffs": [4.5, 4.5, 4.5, 4.5, 4.5, 4.5], "pair_qubit_coeffs": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]}, "evals": {"number of evals": 15, "jac evals": 0, "qfim evals": 0}, "intermediate": {"angles log": [], "intermediate cost": [], "intermediate measurement outcomes": [], "intermediate runs job id": []}, "optimized": {"optimized angles": [0.34048594327263326, 0.3805304635645852], "optimized cost": 14.213473974133734, "optimized measurement outcomes": ["(0.01712704841629681-0.06173377785851483j)", "(0.01663584347301325-0.06203815749119674j)", "(0.01663584347301319-0.06203815749119676j)", "(0.06455547646428364+0.029535817890403975j)", "(0.016635843473013237-0.062038157491196746j)", "(-0.017280564941595987-0.10095863271214792j)", "(0.06455547646428367+0.02953581789040397j)", "(-0.0910981854337323-0.0262313516491015j)", "(0.016635843473013223-0.062038157491196726j)", "(-0.1502862194536621-0.14679237208900925j)", "(-0.017280564941595973-0.10095863271214789j)", "(0.05177527035835482+0.15741106889341416j)", "(0.06455547646428365+0.029535817890403996j)", "(0.05177527035835486+0.1574110688934142j)", "(-0.09109818543373235-0.026231351649101474j)", "(0.0032267101328793558+0.11866118807395662j)", "(0.016635843473013258-0.062038157491196774j)", "(-0.01728056494159598-0.10095863271214792j)", "(-0.15028621945366205-0.14679237208900925j)", "(0.05177527035835485+0.15741106889341422j)", "(-0.017280564941595973-0.10095863271214789j)", "(0.3077807318054451+0.012753929701494072j)", "(0.05177527035835482+0.15741106889341425j)", "(0.05347414467880818-0.011833262528595291j)", "(0.06455547646428367+0.029535817890403986j)", "(0.051775270358354816+0.15741106889341416j)", "(0.05177527035835484+0.1574110688934142j)", "(-0.07953150983325792-0.057667001905456636j)", "(-0.09109818543373231-0.0262313516491015j)", "(0.05347414467880818-0.011833262528595283j)", "(0.003226710132879353+0.11866118807395658j)", "(0.05362855755106562-0.029968128351769402j)", "(0.01663584347301324-0.062038157491196726j)", "(0.06455547646428365+0.02953581789040398j)", "(-0.017280564941595963-0.10095863271214794j)", "(-0.09109818543373233-0.02623135164910149j)", "(-0.15028621945366205-0.14679237208900925j)", "(0.05177527035835485+0.15741106889341416j)", "(0.05177527035835483+0.1574110688934142j)", "(0.0032267101328793523+0.11866118807395658j)", "(-0.017280564941595983-0.10095863271214789j)", "(0.05177527035835485+0.15741106889341416j)", "(0.3077807318054451+0.0127539297014941j)", "(0.053474144678808226-0.011833262528595295j)", "(0.051775270358354844+0.15741106889341416j)", "(-0.07953150983325791-0.057667001905456636j)", "(0.05347414467880822-0.01183326252859531j)", "(0.05362855755106567-0.02996812835176942j)", "(0.0645554764642837+0.029535817890404006j)", "(-0.09109818543373228-0.026231351649101495j)", "(0.05177527035835486+0.15741106889341416j)", "(0.003226710132879375+0.11866118807395663j)", "(0.05177527035835486+0.1574110688934142j)", "(0.05347414467880822-0.011833262528595304j)", "(-0.0795315098332579-0.05766700190545663j)", "(0.053628557551065624-0.029968128351769402j)", "(-0.0910981854337323-0.026231351649101512j)", "(0.0032267101328793887+0.11866118807395659j)", "(0.05347414467880822-0.011833262528595293j)", "(0.053628557551065645-0.02996812835176943j)", "(0.0032267101328793766+0.11866118807395658j)", "(0.05362855755106564-0.029968128351769402j)", "(0.05362855755106565-0.02996812835176945j)", "(0.13340416638626754-0.020162387430041936j)"], "optimized run job id": []}, "most_probable_states": {"solutions_bitstrings": ["101010", "010101"], "bitstring_energy": 3.0}}}, {"QUBO": {"terms": [[0, 3], [0, 1], [1, 2], [2, 3], [2], [3]], "weights": [2.5, -2.5, -2.5, 2.5, 4.5, 4.5], "constant": 0, "_n": 4}, "QAOA": {"method": "cobyla", "cost_hamiltonian": {"n_qubits": 5, "terms": [{"qubit_indices": [0, 4], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [0, 1], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [1, 2], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [2, 3], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [3, 4], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [1], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [2], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [3], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [4], "pauli_str": "Z", "phase": 1}], "coeffs": [2.5, -2.5, 2.5, 2.5, 2.5, 4.5, 4.5, 4.5, 4.5], "constant": 0, "qubits_pairs": [{"qubit_indices": [0, 4], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [0, 1], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [1, 2], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [2, 3], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [3, 4], "pauli_str": "ZZ", "phase": 1}], "qubits_singles": [{"qubit_indices": [1], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [2], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [3], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [4], "pauli_str": "Z", "phase": 1}], "single_qubit_coeffs": [4.5, 4.5, 4.5, 4.5], "pair_qubit_coeffs": [2.5, -2.5, 2.5, 2.5, 2.5]}, "evals": {"number of evals": 15, "jac evals": 0, "qfim evals": 0}, "intermediate": {"angles log": [], "intermediate cost": [], "intermediate measurement outcomes": [], "intermediate runs job id": []}, "optimized": {"optimized angles": [0.4066391532372541, 0.3764245401202528], "optimized cost": -1.7745788740483766, "optimized measurement outcomes": ["(-0.038014829992670626-0.11888336318999058j)", "(-0.03801482999267061-0.11888336318999058j)", "(0.07490161926921533-0.00408490931809663j)", "(-0.09751214852874845+0.06444642510944228j)", "(-0.05940619518438736-0.19606672453866691j)", "(-0.17796563271469296+0.004245630654183513j)", "(-0.07026294872164032+0.09448494426422854j)", "(0.20773036935288222-0.08562113315175908j)", "(-0.177965632714693+0.004245630654183444j)", "(-0.05940619518438742-0.19606672453866697j)", "(0.21971994930516753+0.18380146908850506j)", "(0.11882706551436312-0.22055705115807978j)", "(0.139266465119223+0.12360067463324634j)", "(0.13926646511922305+0.12360067463324628j)", "(0.11143977458793546+0.13179666318124356j)", "(-0.02467350432027174+0.04032394964477435j)", "(-0.09751214852874843+0.06444642510944229j)", "(0.07490161926921532-0.0040849093180966006j)", "(0.16962441600852118+0.17489201649635017j)", "(0.16962441600852116+0.1748920164963502j)", "(0.11882706551436312-0.22055705115807986j)", "(0.2197199493051676+0.18380146908850506j)", "(0.0437903999133875-0.168897858140231j)", "(-0.07476903761691814+0.03141449705261942j)", "(0.20773036935288222-0.08562113315175907j)", "(-0.07026294872164032+0.09448494426422853j)", "(-0.07476903761691811+0.031414497052619464j)", "(0.04379039991338747-0.16889785814023095j)", "(-0.024673504320271718+0.04032394964477434j)", "(0.11143977458793543+0.13179666318124356j)", "(0.0355028813050779-0.13402103274955557j)", "(0.03550288130507792-0.13402103274955562j)"], "optimized run job id": []}, "most_probable_states": {"solutions_bitstrings": ["10101"], "bitstring_energy": -2.5}}}, {"QUBO": {"terms": [[0, 2], [0, 1], [1], [2]], "weights": [2.5, -2.5, 2.5, 2.0], "constant": 0, "_n": 3}, "QAOA": {"method": "cobyla", "cost_hamiltonian": {"n_qubits": 4, "terms": [{"qubit_indices": [0, 3], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [0, 1], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [1, 2], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [2, 3], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [2], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [3], "pauli_str": "Z", "phase": 1}], "coeffs": [2.5, -2.5, -2.5, 2.5, 4.5, 4.5], "constant": 0, "qubits_pairs": [{"qubit_indices": [0, 3], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [0, 1], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [1, 2], "pauli_str": "ZZ", "phase": 1}, {"qubit_indices": [2, 3], "pauli_str": "ZZ", "phase": 1}], "qubits_singles": [{"qubit_indices": [2], "pauli_str": "Z", "phase": 1}, {"qubit_indices": [3], "pauli_str": "Z", "phase": 1}], "single_qubit_coeffs": [4.5, 4.5], "pair_qubit_coeffs": [2.5, -2.5, -2.5, 2.5]}, "evals": {"number of evals": 15, "jac evals": 0, "qfim evals": 0}, "intermediate": {"angles log": [], "intermediate cost": [], "intermediate measurement outcomes": [], "intermediate runs job id": []}, "optimized": {"optimized angles": [0.8574965024416041, -0.5645176360484713], "optimized cost": -4.994719869189089, "optimized measurement outcomes": ["(-0.028089082875898397-0.0788158766087439j)", "(0.02058930062122948-0.06149822599533241j)", "(0.02284656506787168-0.12169533619833914j)", "(-0.028089082875898397-0.07881587660874392j)", "(-0.050228418048355955-0.09501392845169296j)", "(-0.053501166960092834+0.02514256079466251j)", "(-0.09890680154548381-0.11233157906510446j)", "(-0.09940761425207562+0.05388538236812204j)", "(-0.0994076142520756+0.05388538236812204j)", "(-0.053501166960092834+0.02514256079466249j)", "(-0.09890680154548383-0.11233157906510448j)", "(-0.050228418048355955-0.09501392845169299j)", "(-0.4443180223622645-0.008373141661698044j)", "(-0.47871172898303793-0.17554053865500546j)", "(-0.4410452734505276-0.1285296309080535j)", "(-0.4443180223622645-0.008373141661698037j)"], "optimized run job id": []}, "most_probable_states": {"solutions_bitstrings": ["1011"], "bitstring_energy": 1.0}}}], "number_steps": 3, "device": {"device_name": "qiskit.statevector_simulator", "device_location": "local"}, "circuit_properties": {"_param_type": "standard", "_init_type": "ramp", "qubit_register": [], "_p": 1, "variational_params_dict": {"total_annealing_time": 0.7}, "_annealing_time": 0.7, "linear_ramp_time": 0.7, "_mixer_hamiltonian": "x"}, "backend_properties": {"init_hadamard": true, "n_shots": 100, "cvar_alpha": 1}, "classical_optimizer": {"optimize": true, "method": "cobyla", "maxiter": 15, "parameter_log": false, "optimization_progress": false, "cost_progress": false, "save_intermediate": false}, "rqaoa_parameters": {"rqaoa_type": "custom", "n_max": 1, "steps": [1, 1, 1], "n_cutoff": 3, "counter": 0}}'
        string_to_find_human = '{"solution": {"101010": 3.0}, "classical_output": {"minimum_energy": -5.5, "optimal_states": ["110"]}, "elimination_rules": [{"(0, 1)": -1.0}, {"(1, 2)": -1.0}, {"(None, 2)": -1.0}], "schedule": [1, 1, 1], "intermediate_steps": [{"QUBO": {"terms": [[0, 4], [0, 1], [1, 2], [2, 3], [3, 4], [1], [2], [3], [4]], "weights": [2.5, -2.5, 2.5, 2.5, 2.5, 4.5, 4.5, 4.5, 4.5], "constant": 0, "_n": 5}, "QAOA": {"solutions_bitstrings": ["101010", "010101"], "bitstring_energy": 3.0}}, {"QUBO": {"terms": [[0, 3], [0, 1], [1, 2], [2, 3], [2], [3]], "weights": [2.5, -2.5, -2.5, 2.5, 4.5, 4.5], "constant": 0, "_n": 4}, "QAOA": {"solutions_bitstrings": ["10101"], "bitstring_energy": -2.5}}, {"QUBO": {"terms": [[0, 2], [0, 1], [1], [2]], "weights": [2.5, -2.5, 2.5, 2.0], "constant": 0, "_n": 3}, "QAOA": {"solutions_bitstrings": ["1011"], "bitstring_energy": 1.0}}], "number_steps": 3, "device": {"device_name": "qiskit.statevector_simulator", "device_location": "local"}, "circuit_properties": {"_param_type": "standard", "_init_type": "ramp", "qubit_register": [], "_p": 1, "variational_params_dict": {"total_annealing_time": 0.7}, "_annealing_time": 0.7, "linear_ramp_time": 0.7, "_mixer_hamiltonian": "x"}, "backend_properties": {"init_hadamard": true, "n_shots": 100, "cvar_alpha": 1}, "classical_optimizer": {"optimize": true, "method": "cobyla", "maxiter": 15, "parameter_log": false, "optimization_progress": false, "cost_progress": false, "save_intermediate": false}, "rqaoa_parameters": {"rqaoa_type": "custom", "n_max": 1, "steps": [1, 1, 1], "n_cutoff": 3, "counter": 0}}'    

        # Test for .dumps returning a string
        results = self._run_rqaoa()
        string_dumps = results.dumps(string=True)
        string_dumps_human = results.dumps(string=True, human=True)
        dictionay_dumps = results.dumps(string=False)
        dictionay_dumps_human = results.dumps(string=False, human=True)

        assert string_dumps == string_to_find, 'String dump is not correct'
        assert string_dumps_human == string_to_find_human, 'String dump for humans is not correct'
        assert isinstance(dictionay_dumps, dict), 'Dictionary dump is not a dictionary'
        assert isinstance(dictionay_dumps_human, dict), 'Dictionary dump for humans is not a dictionary'


    #test dump 
    def test_rqaoa_result_dump(self):
        """
        Test the dump method for the RQAOAResult class
        """

        # name for the file that will be created and deleted
        name_file = 'results.json'

        #run the algorithm
        results = self._run_rqaoa()

        # Test for .dump creating a file and containing the correct information
        for human in [True, False]:
            results.dump(name_file, human=human)
            assert os.path.isfile(name_file), 'Dump file does not exist'
            assert open(name_file, "r").read() == results.dumps(string=True, human=human), 'Dump file does not contain the correct data'
            os.remove(name_file)