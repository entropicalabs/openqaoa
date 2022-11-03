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

from openqaoa import rqaoa
from openqaoa.workflows.optimizer import QAOA, RQAOA
from openqaoa.backends.qaoa_backend import (DEVICE_NAME_TO_OBJECT_MAPPER,
                                            DEVICE_ACCESS_OBJECT_MAPPER)
from openqaoa.devices import create_device,SUPPORTED_LOCAL_SIMULATORS
import unittest
import networkx as nw
import numpy as np
import itertools

from openqaoa.problems.problem import MinimumVertexCover, QUBO

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

    def _run_rqaoa(self, problem, type='custom', n_cutoff=3, eliminations=1, p=1, param_type='standard', mixer='x', method='cobyla', maxiter=15, name_device='qiskit.statevector_simulator'):

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
        g = nw.circulant_graph(n_qubits, [1])
        vc = MinimumVertexCover(g, field =1.0, penalty=10).get_qubo_problem()

        # Test for the standard RQAOA
        results = self._run_rqaoa(problem=vc, n_cutoff=n_cutoff)
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
        results = self._run_rqaoa(type='adaptive', problem=vc, n_cutoff=n_cutoff)
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

