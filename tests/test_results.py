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
import json

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


    def _run_qaoa(self):
        """
        Run a QAOA algorithm and return the results
        """

        # Create the problem
        g = nw.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field=1.0, penalty=10).get_qubo_problem()

        q = QAOA()
        q.compile(vc, verbose=False)
        q.optimize()

        return q.results

    def _test_results_dict(self, results_dict):
        """
        private function to test the results dictionary
        """
        # test is a dictionary
        assert isinstance(results_dict, dict), 'Results are not a dictionary'
        
        # test the keys
        expected_keys = ['method', 'cost_hamiltonian', 'evals', 'intermediate', 'optimized', 'most_probable_states']
        for key in expected_keys:
            assert key in results_dict.keys(), '{} key not in results dictionary'.format(key)

    #test as_dict method
    def test_qaoa_result_as_dict(self):
        """
        Test as_dict method for the QAOA result class
        """

        # run the QAOA and get the results as a dictionary with the as_dict method
        results = self._run_qaoa()
        results_dict = results.as_dict()
        
        # test the dictionary
        self._test_results_dict(results_dict)

    #test dumps
    def test_qaoa_result_dumps(self):
        """
        Test the dumps for the QAOA result class
        """

        # Test for .dumps returning a string
        results = self._run_qaoa()
        json_string = results.dumps()
        assert isinstance(json_string, str), 'json_string is not a string'

        # read the json string and test the dictionary that is returned
        results_dict = json.loads(json_string)
        self._test_results_dict(results_dict)

    #test dump 
    def test_qaoa_result_dump(self):
        """
        Test the dump method for the QAOA result class 
        """

        # name for the file that will be created and deleted
        name_file = 'results.json'

        #run the algorithm
        results = self._run_qaoa()

        # Test for .dump creating a file and containing the correct information
        results.dump(name_file, indent=None)
        assert os.path.isfile(name_file), 'Dump file does not exist'
        with open(name_file, 'r') as file:
            assert file.read() == results.dumps(indent=None), 'Dump file does not contain the correct data'

        # read the json string 
        with open(name_file, 'r') as file:
            results_dict = json.load(file)

        # test the dictionary that is returned
        self._test_results_dict(results_dict)

        # delete the file
        os.remove(name_file)


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

    def _test_results_dict(self, results_dict):
        """
        private function to test the results dictionary
        """
        # test is a dictionary
        assert isinstance(results_dict, dict), 'Results are not a dictionary'
        
        # test the keys
        expected_keys = ['solution', 'classical_output', 'elimination_rules', 'schedule', 'intermediate_steps', 
                         'number_steps', 'device', 'circuit_properties', 'backend_properties', 'classical_optimizer',
                         'rqaoa_parameters']
        for key in expected_keys:
            assert key in results_dict.keys(), '{} key not in results dictionary'.format(key)

    #test as_dict method
    def test_rqaoa_result_as_dict(self):
        """
        Test as_dict method for the RQAOAResult class
        """

        # run the RQAOA and get the results as a dictionary with the as_dict method
        results = self._run_rqaoa()
        results_dict = results.as_dict()
        
        # test the dictionary
        self._test_results_dict(results_dict)

    #test dumps
    def test_rqaoa_result_dumps(self):
        """
        Test the dumps for the RQAOAResult class
        """

        # Test for .dumps returning a string
        results = self._run_rqaoa()
        json_string = results.dumps()
        assert isinstance(json_string, str), 'json_string is not a string'

        # read the json string and test the dictionary that is returned
        results_dict = json.loads(json_string)
        self._test_results_dict(results_dict)

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
        results.dump(name_file, indent=None)
        assert os.path.isfile(name_file), 'Dump file does not exist'
        with open(name_file, 'r') as file:
            assert file.read() == results.dumps(indent=None), 'Dump file does not contain the correct data'

        # read the json string 
        with open(name_file, 'r') as file:
            results_dict = json.load(file)

        # test the dictionary that is returned
        self._test_results_dict(results_dict)

        # delete the file
        os.remove(name_file)

if __name__ == "__main__":
	unittest.main()
 