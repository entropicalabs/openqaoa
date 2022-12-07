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

from openqaoa.backends.simulators.qaoa_analytical_sim import QAOABackendAnalyticalSimulator 
from openqaoa.utilities import random_k_regular_graph
from openqaoa.workflows.optimizer import QAOA, RQAOA
from openqaoa.problems.problem import QUBO, MaximumCut
from openqaoa.devices import create_device

"""
A set of tests for FastQAOA: see the notebook Test_Examples.ipynb for explanations of how the
expected answers are derived.
"""
def run_Rqaoa_experiment():  
    
    g = random_k_regular_graph(degree=3, nodes=range(8), weighted=True, biases=False) 

    # Define te problem and translate it into a binary Qubo.
    maxcut_prob = MaximumCut(g)
    maxcut_qubo = maxcut_prob.get_qubo_problem()

    
    # Define the RQAOA object (default rqaoa_type = 'adaptive')
    r = RQAOA()

    # Set parameters for RQAOA, in this case we fix the n_max to 1 (default), the final cutoff value to 3
    r.set_rqaoa_parameters(steps=1, n_cutoff=3)

    ## Setting up the QAOA properties
    init_beta = 0.42
    init_gamma = 0.42

    r.set_circuit_properties(p=1, init_type='custom', variational_params_dict={"betas":[init_beta], "gammas":[init_gamma]}, mixer_hamiltonian='x')

    # Define the device you want to run your problem on using the create_device() function - Here we choose the local wavefunction simulator supported by OpenQAOA
    device = create_device(location='local', name='analytical_simulator')
    r.set_device(device)
    
    # Set the classical method used to optimiza over QAOA angles and its properties
    r.set_classical_optimizer(method="COBYLA", optimizer_options = {"stepsize" : 10**(-10)}, tol=10**(-1), maxfev=1, maxiter=1, optimization_progress=False, cost_progress=False, parameter_log=False)

   
    # Compile problem instance on RQAOA, just like with QAOA
    r.compile(maxcut_qubo)
    
    r.optimize()
    
    print(r.results)
    
    
run_Rqaoa_experiment()    


'''
if __name__ == "__main__":
    unittest.main()
'''