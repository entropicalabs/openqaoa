from openqaoa.problems.maximumcut import MaximumCut
from openqaoa.backends import create_device

import networkx as nx
from openqaoa import QAOA  
from openqaoa.problems import MaximumCut
from openqaoa.utilities import ground_state_hamiltonian
import matplotlib.pyplot as plt
from openqaoa.utilities import plot_graph
from qiskit_aer.noise import (NoiseModel, depolarizing_error)

one_qubit_gates = ['h','rx']
two_qubits_gates = ['rzz']

#create depol. noise
def add_depolarizing_error(noise_model,prob1, prob2):
    noise_model = add_one_qubit_depolarizing_error(noise_model,prob1)
    noise_model = add_two_qubits_depolarizing_error(noise_model,prob2)
    return noise_model

#create 1 qubit depol. noise
def add_one_qubit_depolarizing_error(noise_model,prob):
    error = depolarizing_error(prob, 1)
    noise_model.add_all_qubit_quantum_error(error,one_qubit_gates)
    return noise_model

#create 2 qubits depol.noise
def add_two_qubits_depolarizing_error(noise_model,prob):
    error = depolarizing_error(prob, 2)
    noise_model.add_all_qubit_quantum_error(error, two_qubits_gates)
    return noise_model

noise_model = add_depolarizing_error(NoiseModel(),0.0001989, 0.007905) #ibm_quebec, 19/01/2024



""" maxcut_prob = MaximumCut.random_instance(n_nodes = 10, edge_probability = 0.5)
plot_graph(maxcut_prob.G)
maxcut_qubo = maxcut_prob.qubo
maxcut_hamiltonian = maxcut_qubo.hamiltonian """

graph1 = nx.Graph()
graph1.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
graph1.add_edges_from([(0, 6), (0, 8), (1, 5), (2, 7), (2, 8), (3, 5), (3, 7), (4, 8), (6, 7), (7, 9), (8, 9)])

graph2 = nx.Graph()
graph2.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
graph2.add_edges_from([(0, 2), (0, 3), (0, 4), (0, 5), (0, 7), (0, 8), (1, 4), (1, 6), (1, 8), (1, 9), (2, 4), (2, 6), (3, 4), (3, 6), (3, 8), (3, 9), (4, 5), (4, 7), (4, 9), (5, 8), (6, 7), (7, 8), (7, 9), (8, 9)])

graph3 = nx.Graph()
graph3.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
graph3.add_edges_from([(0, 1), (0, 2), (0, 4), (0, 9), (1, 5), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (2, 6), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 5), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (5, 9), (6, 8), (6, 9), (7, 8), (7, 9)])

graph4 = nx.Graph()
graph4.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
graph4.add_edges_from([(0, 3), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (2, 3), (3, 4), (3, 6), (3, 7), (4, 5), (4, 6), (4, 9), (5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (6, 9), (7, 9), (8, 9)])

graph5 = nx.Graph()
graph5.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
graph5.add_edges_from([(0, 5), (0, 7), (0, 8), (1, 2), (1, 3), (1, 4), (1, 6), (1, 9), (2, 3), (2, 4), (2, 5), (2, 8), (2, 9), (3, 5), (3, 6), (3, 7), (3, 9), (4, 6), (4, 7), (4, 9), (5, 6), (6, 7), (6, 8), (7, 9)])

mc1 = MaximumCut(graph1)
mc2 = MaximumCut(graph2)
mc3 = MaximumCut(graph3)
mc4 = MaximumCut(graph4)
mc5 = MaximumCut(graph5)

mcs = [mc1, mc2, mc3, mc4, mc5]
ps = [1, 2, 3, 4, 5, 6]
param_types = [ "standard", "extended"]
init_types = ["rand", "ramp"]
mixer_hams = ["x", "xy"]
optimizers = ["Nelder-Mead", "Powell", "COBYLA"]

for index, maxcut_prob in enumerate(mcs):
    maxcut_qubo = maxcut_prob.qubo
    maxcut_hamiltonian = maxcut_qubo.hamiltonian

    for p in ps:
        for param_type in param_types:
            for init_type in init_types:
                for mixer_hamiltonian in mixer_hams:
                    for optimizer in optimizers:
                        print("params = %s-%s-%s-%s-%s" % (str(p), str(param_type), str(init_type), str(mixer_hamiltonian), str(optimizer)))
                        q1 = QAOA()
                        qiskit_device = create_device(location='local', name='qiskit.shot_simulator')
                        q1.set_device(qiskit_device)
                        q1.set_circuit_properties(p=p, param_type=param_type, init_type=init_type, mixer_hamiltonian=mixer_hamiltonian)
                        """  variational_params_dict={
                            'gammas_pairs':[0.42], 
                            'gammas_singles':[0.97], 
                            'betas':[0.13]}
                         ) """
                        q1.set_backend_properties(n_shots=10000)
                        q1.set_classical_optimizer(method=optimizer, maxiter=200, tol=0.001,
                        optimization_progress=True, cost_progress=True, parameter_log=True)
                        q1.compile(maxcut_qubo)
                        q1.optimize()
                        correct_solution1 = ground_state_hamiltonian(q1.cost_hamil)
                        print(correct_solution1)
                        opt_results1 = q1.result
                        q2 = QAOA()
                        qiskit_device = create_device(location='local', name='qiskit.shot_simulator')
                        """ q2.set_device(qiskit_device)
                        q2.set_circuit_properties(p=p, param_type=param_type, init_type=init_type, mixer_hamiltonian=mixer_hamiltonian)
                        q2.set_classical_optimizer(method=optimizer, maxiter=200, tol=0.001,
                                                optimization_progress=True, cost_progress=True, parameter_log=True)
                        q2.set_backend_properties(n_shots=5000, seed_simulator=1,
                                                noise_model=noise_model)
                        q2.compile(maxcut_qubo)
                        q2.optimize()
                        correct_solution2 = ground_state_hamiltonian(q2.cost_hamil)
                        opt_results2 = q2.result
                        print(correct_solution2)
                        qiskit_device = create_device(location='local', name='qiskit.shot_simulator')
                        q3 = QAOA()
                        q3.set_device(qiskit_device)
                        q3.set_circuit_properties(p=p, param_type=param_type, init_type=init_type, mixer_hamiltonian=mixer_hamiltonian)
                        q3.set_classical_optimizer(method=optimizer, maxiter=200, tol=0.001,
                                                optimization_progress=True, cost_progress=True, parameter_log=True)
                        q3.set_backend_properties(n_shots=5000, seed_simulator=1, noise_model=noise_model)
                        q3.set_error_mitigation_properties(error_mitigation_technique='mitiq_zne',n_batches=64,calibration_data_location="caldata.json")
                        q3.compile(maxcut_qubo)
                        q3.optimize()
                        correct_solution3 = ground_state_hamiltonian(q3.cost_hamil)
                        print(correct_solution3)
                        opt_results3 = q3.result """

                        #plot_cost(opt_results2,)

                        fig, ax = plt.subplots(figsize=(7,4))
                        opt_results1.plot_cost(figsize=(7,4),color='blue',label='qaoa',ax=ax)
                        """ opt_results2.plot_cost(figsize=(7,4),color='red',label='qaoa+noise',ax=ax)
                        opt_results3.plot_cost(figsize=(7,4),color='green',label='qaoa+noise+zne',ax=ax) """
                        plt.savefig("../../results/Marco_Results/MaxCut/no_noise/inst_%d/%s-%s-%s-%s-%s-1.png" % (index, str(p), str(param_type), str(init_type), str(mixer_hamiltonian), str(optimizer)))
                        """ fig2, ax2 = plt.subplots(figsize=(7,4))
                        opt_results2.plot_cost(figsize=(7,4),color='red',label='qaoa+noise',ax=ax2)
                        opt_results3.plot_cost(figsize=(7,4),color='green',label='qaoa+noise+zne',ax=ax2)
                        plt.savefig("%s-%s-%s-%s-%s-2.png" % (str(p), str(param_type), str(init_type), str(mixer_hamiltonian), str(optimizer))) """

""" import json
#factories = [ "Exp", "FakeNodes", "Linear", "Richardson"]
factories = ["AdaExp"]
scales = ["fold_gates_at_random", "fold_gates_from_left", "fold_gates_from_right"]

for factory in factories:
    for scaling in scales:
        print("%s %s" % (factory, scaling))

        cal_data = {
            "factory": factory,
            "scaling": scaling,
            "seed": 1,
            "scale_factor": [2,5,8]
        }
        json_object = json.dumps(cal_data, indent=4)
 
        # Writing to sample.json
        with open("caldata.json", "w") as outfile:
            outfile.write(json_object)
        q2 = QAOA()

        qiskit_device = create_device(location='local', name='qiskit.shot_simulator')
        q2.set_device(qiskit_device)
        q2.set_circuit_properties(p=3,param_type='standard', init_type='rand', mixer_hamiltonian='x')
        q2.set_classical_optimizer(method='nelder-mead', maxiter=150, tol=0.001,
                                optimization_progress=True, cost_progress=True, parameter_log=True)
        q2.set_backend_properties(n_shots=5000, seed_simulator=1,
                                noise_model=noise_model)
        q2.set_error_mitigation_properties(error_mitigation_technique='mitiq_zne',n_batches=64,calibration_data_location="caldata.json")
        q2.compile(maxcut_qubo)
        q2.optimize()
        correct_solution3 = ground_state_hamiltonian(q2.cost_hamil)
        print(correct_solution3)
        opt_results2 = q2.result

        #plot_cost(opt_results2,)

        fig, ax = plt.subplots(figsize=(7,4))
        opt_results2.plot_cost(figsize=(7,4),color='red',label='qaoa+noise+zne',ax=ax)
        plt.savefig("%s-%s.png" % (str(factory), str(scaling))) """
        