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
from openqaoa.optimizers.result import Result
from openqaoa.backends.qaoa_backend import (DEVICE_NAME_TO_OBJECT_MAPPER,
                                            DEVICE_ACCESS_OBJECT_MAPPER)
from openqaoa.devices import create_device,SUPPORTED_LOCAL_SIMULATORS
import unittest
import networkx as nw
import numpy as np
import itertools

from openqaoa.problems import MinimumVertexCover, QUBO
from openqaoa.qaoa_parameters.operators import Hamiltonian
from openqaoa.rqaoa.rqaoa_results import RQAOAResults

ALLOWED_LOCAL_SIMUALTORS = SUPPORTED_LOCAL_SIMULATORS


def _test_keys_in_dict(obj, expected_keys):
    """
    private function to test the keys. It recursively tests the keys of the nested dictionaries, or lists of dictionaries
    """

    if isinstance(obj, dict):
        for key in obj:
            if key in expected_keys.keys(): expected_keys[key] = True

            if isinstance(obj[key], dict):
                _test_keys_in_dict(obj[key], expected_keys)
            elif isinstance(obj[key], list):
                for item in obj[key]:
                    _test_keys_in_dict(item, expected_keys)
    elif isinstance(obj, list):
        for item in obj:
            _test_keys_in_dict(item, expected_keys)


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
                self.assertEqual(recorded_evals[each_choice[0]], len(q.results.intermediate['angles']))
                self.assertEqual(recorded_evals[each_choice[1]], len(q.results.intermediate['cost']))
                self.assertEqual(recorded_evals[each_choice[2]], len(q.results.intermediate['measurement_outcomes']))

    def test_qaoa_result_asdict(self):
        """
        Test the qaoa result.asdict method
        """

        # run the QAOA
        qaoa = QAOA()
        qaoa.compile(problem = QUBO.random_instance(n=8))
        qaoa.optimize()
        
        # get dict
        results_dict = qaoa.results.asdict()

        # list of expected keys
        expected_keys = ['method', 'cost_hamiltonian', 'n_qubits', 'terms', 'qubit_indices', 'pauli_str', 'phase', 'coeffs', 'constant', 'qubits_pairs', 'qubits_singles', 'single_qubit_coeffs', 'pair_qubit_coeffs', 'evals', 'number_of_evals', 'jac_evals', 'qfim_evals', 'most_probable_states', 'solutions_bitstrings', 'bitstring_energy', 'intermediate', 'angles', 'cost', 'measurement_outcomes', 'job_id', 'optimized', 'angles', 'cost', 'measurement_outcomes', 'job_id']
        
        #we append all the keys that we find in rqaoa.results, so if we introduce a new key, we will know that we need to update the result.asdict method
        for key in vars(qaoa.results).keys():
            if not key in expected_keys and not '_Result__' in key: expected_keys.append(key)

        #create a dictionary with all the expected keys and set them to False
        expected_keys_dict = {item: False for item in expected_keys}

        #test the keys, it will set the keys to True if they are found
        _test_keys_in_dict(results_dict, expected_keys_dict)

        # Check if the dictionary has all the expected keys 
        for key, value in expected_keys_dict.items():
            assert value==True, f'Key {key} was not found in the dictionary of the RQAOAResult class.'


        ## now we repeat the same test but we do not include the cost hamiltonian

        #get dict without cost hamiltonian
        results_dict = qaoa.results.asdict(keep_cost_hamiltonian = False)

        #expected keys 
        expected_keys_dict = {item: False for item in expected_keys}    
        expected_keys_not_in_dict = ['cost_hamiltonian', 'n_qubits', 'terms', 'qubit_indices', 'pauli_str', 'phase', 'coeffs', 'constant', 'qubits_pairs', 'qubits_singles', 'single_qubit_coeffs', 'pair_qubit_coeffs']        

        #test the keys, it will set the keys to True if they are found, except the ones that were not included which should be those in expected_keys_not_in_dict
        _test_keys_in_dict(results_dict, expected_keys_dict) 

        # Check if the dictionary has all the expected keys except the ones that were not included
        for key, value in expected_keys_dict.items():
            if not key in expected_keys_not_in_dict:
                assert value==True, f'Key {key} was not found in the dictionary of the RQAOAResult class.'
            else:
                assert value==False, f'Key {key} was found in the dictionary of the RQAOAResult class, but it should not have been.'


        ## now we repeat the same test but we do not include some keys

        #get dict without some values
        results_dict = qaoa.results.asdict(exclude_keys = ['solutions_bitstrings', 'method'])

        #expected keys
        expected_keys_dict = {item: False for item in expected_keys}
        expected_keys_not_in_dict = ['solutions_bitstrings', 'method']

        #test the keys, it will set the keys to True if they are found, except the ones that were not included which should be those in expected_keys_not_in_dict
        _test_keys_in_dict(results_dict, expected_keys_dict)

        # Check if the dictionary has all the expected keys except the ones that were not included
        for key, value in expected_keys_dict.items():
            if not key in expected_keys_not_in_dict:
                assert value==True, f'Key {key} was not found in the dictionary of the RQAOAResult class.'
            else:
                assert value==False, f'Key {key} was found in the dictionary of the RQAOAResult class, but it should not have been.'

        """
        to get the list of expected keys, run the following code:

            def get_keys(obj, list_keys):
                if isinstance(obj, dict):
                    for key in obj:
                        if not key in list_keys: list_keys.append(key)

                        if isinstance(obj[key], dict):
                            get_keys(obj[key], list_keys)
                        elif isinstance(obj[key], list):
                            for item in obj[key]:
                                get_keys(item, list_keys)
                elif isinstance(obj, list):
                    for item in obj:
                        get_keys(item, list_keys)

            expected_keys = []
            get_keys(rqaoa.results.asdict(), expected_keys)
            print(expected_keys)
        """

    #test eval_number 
    def test_qaoa_result_eval_number(self):
        """
        Test the eval_number method for the QAOA result class
        """

        for method in ['cobyla', 'spsa', 'vgd', 'newton', 'natural_grad_descent']:
            # run the QAOA and get the results
            q = QAOA()
            q.set_classical_optimizer(maxiter=15, method=method, jac='finite_difference', hess='finite_difference')
            q.compile(problem = QUBO.random_instance(n=8))
            q.optimize()

            # test the eval_number method
            assert q.results.intermediate['cost'].index(min(q.results.intermediate['cost'])) + 1 == q.results.optimized['eval_number'], 'optimized eval_number does not return the correct number of the optimized evaluation, when using {} method'.format(method)




class TestingRQAOAResultOutputs(unittest.TestCase):
    """
    Test the  Results Output after a full RQAOA loop
    """        

    def __run_rqaoa(self, type='custom', eliminations=1, p=1, param_type='standard', mixer='x', method='cobyla', maxiter=15, name_device='qiskit.statevector_simulator'):
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
        results = self.__run_rqaoa()
        assert isinstance(results, RQAOAResults), 'Results of RQAOA are not of type RQAOAResults'
        for key in results['solution'].keys():
            assert len(key) == n_qubits, 'Number of qubits solution is not correct'
        assert isinstance(results['classical_output']['minimum_energy'], float)
        assert isinstance(results['classical_output']['optimal_states'], list)
        for rule_list in results['elimination_rules']:
            for rule in rule_list:
                assert isinstance(rule, dict), 'Elimination rule item is not a dictionary'
        assert isinstance(results['schedule'], list), 'Schedule is not a list'
        assert sum(results['schedule']) + n_cutoff == n_qubits, 'Schedule is not correct'
        for step in results['intermediate_steps']:
            assert isinstance(step['problem'], QUBO), 'problem is not of type QUBO'
            assert isinstance(step['qaoa_results'], Result), 'QAOA_results is not of type QAOA Results'
            assert isinstance(step['exp_vals_z'], np.ndarray), 'exp_vals_z is not of type numpy array'
            assert isinstance(step['corr_matrix'], np.ndarray), 'corr_matrix is not of type numpy array'
        assert isinstance(results['number_steps'], int), 'Number of steps is not an integer'
       
        # Test for the adaptive RQAOA
        results = self.__run_rqaoa(type='adaptive')
        assert isinstance(results, RQAOAResults), 'Results of RQAOA are not of type RQAOAResults'
        for key in results['solution'].keys():
            assert len(key) == n_qubits, 'Number of qubits solution is not correct'
        assert isinstance(results['classical_output']['minimum_energy'], float)
        assert isinstance(results['classical_output']['optimal_states'], list)
        for rule_list in results['elimination_rules']:
            for rule in rule_list:
                assert isinstance(rule, dict), 'Elimination rule item is not a dictionary'
        assert isinstance(results['schedule'], list), 'Schedule is not a list'
        assert sum(results['schedule']) + n_cutoff == n_qubits, 'Schedule is not correct'
        for step in results['intermediate_steps']:
            assert isinstance(step['problem'], QUBO), 'QUBO is not of type QUBO'
            assert isinstance(step['qaoa_results'], Result), 'QAOA_results is not of type QAOA Results'
            assert isinstance(step['exp_vals_z'], np.ndarray), 'exp_vals_z is not of type numpy array'
            assert isinstance(step['corr_matrix'], np.ndarray), 'corr_matrix is not of type numpy array'
        assert isinstance(results['number_steps'], int), 'Number of steps is not an integer'
       

    def test_rqaoa_result_methods_steps(self):
        """
        Test the methods for the RQAOAResult class for the steps
        """

        # run the RQAOA
        results = self.__run_rqaoa()

        # test the solution method
        assert results.get_solution() == results['solution'], 'get_solution method is not correct'

        # test the methods for the intermediate steps 
        for i in range(results['number_steps']):

            #methods for intermediate qaao results
            assert results.get_qaoa_results(i) == results['intermediate_steps'][i]['qaoa_results'], 'get_qaoa_results method is not correct'
            assert results.get_qaoa_optimized_angles(i) == results.get_qaoa_results(i).optimized['angles'], 'get_qaoa_optimized_angles method is not correct'

            #methods for intermediate qubo
            assert results.get_problem(i) == results['intermediate_steps'][i]['problem'], 'get_qubo method is not correct'
            assert isinstance(results.get_hamiltonian(i), Hamiltonian), 'get_hamiltonian method is not correct'

            #methods for intermediate exp_vals_z and corr_matrix
            assert results.get_exp_vals_z(i) is results['intermediate_steps'][i]['exp_vals_z'], 'get_exp_vals_z method is not correct'
            assert results.get_corr_matrix(i) is results['intermediate_steps'][i]['corr_matrix'], 'get_corr_matrix method is not correct'

    def test_rqaoa_result_plot_corr_matrix(self):
        """
        Test the plot_corr_matrix method for the RQAOAResult class
        """

        # run the RQAOA
        results = self.__run_rqaoa()

        # test the plot_corr_matrix method
        for i in range(results['number_steps']):
            results.plot_corr_matrix(step=i)

    def test_rqaoa_result_asdict(self):
        """
        Test the plot_exp_vals_z method for the RQAOAResult class
        """

        # run the RQAOA
        results = self.__run_rqaoa()
        
        # get dict
        results_dict = results.asdict()

        #create a list of expected keys
        expected_keys = ['solution', 'classical_output', 'minimum_energy', 'optimal_states', 'elimination_rules', 'pair', 'correlation', 'schedule', 'intermediate_steps', 'problem', 'terms', 'weights', 'constant', 'n', 'qaoa_results', 'method', 'cost_hamiltonian', 'n_qubits', 'qubit_indices', 'pauli_str', 'phase', 'coeffs', 'qubits_pairs', 'qubits_singles', 'single_qubit_coeffs', 'pair_qubit_coeffs', 'evals', 'number_of_evals', 'jac_evals', 'qfim_evals', 'most_probable_states', 'solutions_bitstrings', 'bitstring_energy', 'intermediate', 'angles', 'cost', 'measurement_outcomes', 'job_id', 'optimized', 'angles', 'cost', 'measurement_outcomes', 'job_id', 'exp_vals_z', 'corr_matrix', 'number_steps']

        #we append all the keys that we find in rqaoa.results, so if we introduce a new key, we will know that we need to update the result.asdict method
        for key in results.keys():
            if not key in expected_keys: expected_keys.append(key)
        for key in results['intermediate_steps'][0].keys():
            if not key in expected_keys: expected_keys.append(key)

        # dictionary with all the expected keys and set them to False
        expected_keys = {item: False for item in expected_keys}

        #test the keys, it will set the keys to True if they are found
        _test_keys_in_dict(results_dict, expected_keys)

        # Check if the dictionary has all the expected keys except the ones that were not included
        for key, value in expected_keys.items():
            assert value==True, f'Key {key} was not found in the dictionary of the RQAOAResult class.'


        ## now we repeat the same test but we do not include some keys

        #get dict without some values
        results_dict = results.asdict(exclude_keys = ['solutions_bitstrings', 'method'])

        #expected keys
        expected_keys_dict = {item: False for item in expected_keys}
        expected_keys_not_in_dict = ['solutions_bitstrings', 'method']

        #test the keys, it will set the keys to True if they are found, except the ones that were not included which should be those in expected_keys_not_in_dict
        _test_keys_in_dict(results_dict, expected_keys_dict)

        # Check if the dictionary has all the expected keys except the ones that were not included
        for key, value in expected_keys_dict.items():
            if not key in expected_keys_not_in_dict:
                assert value==True, f'Key {key} was not found in the dictionary of the RQAOAResult class.'
            else:
                assert value==False, f'Key {key} was found in the dictionary of the RQAOAResult class, but it should not have been.'

        """
        to get the list of expected keys, run the following code:

            def get_keys(obj, list_keys):
                if isinstance(obj, dict):
                    for key in obj:
                        if not key in list_keys: list_keys.append(key)

                        if isinstance(obj[key], dict):
                            get_keys(obj[key], list_keys)
                        elif isinstance(obj[key], list):
                            for item in obj[key]:
                                get_keys(item, list_keys)
                elif isinstance(obj, list):
                    for item in obj:
                        get_keys(item, list_keys)

            expected_keys = []
            get_keys(rqaoa.results.asdict(), expected_keys)
            print(expected_keys)
        """

if __name__ == "__main__":
	unittest.main()
 