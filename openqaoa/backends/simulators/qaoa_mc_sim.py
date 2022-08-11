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

# General Imports
from ...basebackend import QAOABaseBackendParametric, QAOABaseBackendShotBased, QAOABaseBackendStatevector, QAOABaseBackend
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from ...utilities import flip_counts
from ...cost_function import cost_function
from ...qaoa_parameters.pauligate import (RXPauliGate, RYPauliGate, RZPauliGate, RXXPauliGate,
                                          RYYPauliGate, RZZPauliGate, RZXPauliGate)

import numpy as np
import math
from typing import Union, List, Tuple, Optional
from scipy.integrate import solve_ivp
import sys

# IBM Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qutip import *

"""
QASM Simluator can be used for different simulation purposes with different Simulation methods
    - supports different error models
    - supports incluing real IBM backend error models
"""

class QAOAMCBackendSimulator(QAOABaseBackend, QAOABaseBackendParametric):
    """
    Monte Carlo solver

    Parameters
    ----------
    circuit_params: `QAOACircuitParams`
        An object of the class ``QAOACircuitParams`` which contains information on 
        circuit construction and depth of the circuit.

    prepend_state: `QuantumCircuit`
        The state prepended to the circuit.

    append_state: `QuantumCircuit`
        The state appended to the circuit.

    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the 
        QAOA part of the circuit.

    cvar_alpha: `float`
        The value of alpha for the CVaR cost function.

    noise_model: `dict`
        The noise parameters to be used for the simulation with format 
        {'decay': 5e-5, 'dephasing': 1e-4, 'overrot': 2, 'spam': 5e-2, 'readout01': 4e-2, 'readout10': 1e-2, 'depol1': 12e-4, 'depol2': 3e-2}.
        To deactivate individual error source, set entry to False.

    times: `list`
        The times for single qubit gates, two qubit gates and readout, defaults to [20e-9, 200e-9, 5800e-9]

    allowed_jump_qubits: `list` 
        The indices of the qubits on which jumps are allowed to occur, None means there are no restrictions
        """   

    QISKIT_PAULIGATE_LIBRARY = [RXPauliGate, RYPauliGate, RZPauliGate, RXXPauliGate,
                            RYYPauliGate, RZZPauliGate, RZXPauliGate]

    def __init__(self,
                 circuit_params: QAOACircuitParams,
                 n_shots: int,
                 prepend_state: Optional[QuantumCircuit],
                 append_state: Optional[QuantumCircuit],
                 init_hadamard: bool,
                 cvar_alpha: float,
                 noise_model: Optional[dict] = {'decay': 5e-5, 'dephasing': 1e-4, 'overrot': 2, 'spam': 5e-2, 'readout01': 4e-2, 'readout10': 1e-2, 'depol1': 12e-4, 'depol2': 3e-2},
                 times: Optional[list] = [20e-9, 200e-9, 5800e-9],
                 allowed_jump_qubits: Optional[list] = None):
        
        QAOABaseBackend.__init__(self,
                 circuit_params,
                 prepend_state,
                 append_state,
                 init_hadamard,
                 cvar_alpha)

        self.noise_model = noise_model
        self.times = times
        self.allowed_jump_qubits = allowed_jump_qubits
        self.n_shots = n_shots
        self.qureg = QuantumRegister(self.n_qubits)
        self.qubit_layout = self.circuit_params.qureg
        
        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.qubits), "Cannot attach a bigger circuit " \
                                                                "to the QAOA routine"
        # For parametric circuits
        self.parametric_circuit = self.parametric_qaoa_circuit

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> QuantumCircuit:
        """
        The final QAOA circuit.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
        
        Returns
        -------
        qaoa_circuit: `QuantumCircuit`
            The final QAOA circuit after binding angles from variational parameters.
        """
        angles_list = self.obtain_angles_for_pauli_list(self.pseudo_circuit, params)
        memory_map = dict(zip(self.qiskit_parameter_list, angles_list))
        new_parametric_circuit = self.parametric_circuit.bind_parameters(memory_map)
        return new_parametric_circuit
    
    @property
    def parametric_qaoa_circuit(self) -> QuantumCircuit:
        """
        Creates a parametric QAOA circuit, given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit.
        """
        # self.reset_circuit()
        parametric_circuit = QuantumCircuit(self.qureg)

        if self.prepend_state:
            parametric_circuit = parametric_circuit.compose(self.prepend_state)
        # Initial state is all |+>
        if self.init_hadamard:
            parametric_circuit.h(self.qureg)
        
        self.qiskit_parameter_list=[]
        for each_gate in self.pseudo_circuit:
            angle_param = Parameter(str(each_gate.pauli_label))
            self.qiskit_parameter_list.append(angle_param)
            each_gate.rotation_angle = angle_param
            if type(each_gate) in self.QISKIT_PAULIGATE_LIBRARY:
                decomposition = each_gate.decomposition('trivial')
            else: 
                decomposition = each_gate.decomposition('standard')
            # Create Circuit
            for each_tuple in decomposition:
                low_gate = each_tuple[0]()
                parametric_circuit = low_gate.apply_ibm_gate(*each_tuple[1],parametric_circuit)
        
        if self.append_state:
            parametric_circuit = parametric_circuit.compose(self.append_state)
        
        #parametric_circuit.measure_all()

        return parametric_circuit
    
    def build_mastereq(self, H, c_ops=None):
        """
        Parameters
        ----------
        H: `np.array`
            Hamiltonian (in matrix form) to be used in the master equation
        
        c_ops: `list`
            List of collapse operators (in matrix form) to be used in the master equation

        
        Returns
        -------
        get_rhs: `np.array`
            The (parametrized) right hand side of the master equation describing the temporal change of the density matrix
        """
    
        def get_rhs(t, rho):
            rho = rho.reshape([len(H),len(H)])
            me_rhs = -1j*((H@rho)-(rho@H))

            if c_ops is not None:
                for c_op in c_ops:
                    c_op_dag = np.conjugate(np.transpose(c_op))
                    me_rhs += c_op@rho@c_op_dag - 0.5*(c_op_dag@c_op@rho + rho@c_op_dag@c_op)
                
            return np.ndarray.flatten(me_rhs)
        return get_rhs

    def get_circuit_list(self, params: QAOAVariationalBaseParams, target_basis=['id', 'x', 'sx', 'rz', 'cx']):
        """
        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
        
        target_basis: `list`
            List of gates (encoded in strings) representing the native basis of the hardware platform. Available gates can be extended
            in the future to allow for the simulation of different hardware platforms (currently IBM superconducting circuits).

        Returns
        -------
        circuit_list: `list`
            List containing qiskit circuits, whereby each circuit corresponds to a layer of the original circuit (in the correct order). 
            We choose the layers such that there are as few layers as possible and that in any given layer, each qubit is involved in at 
            most one (single qubit or two qubit) gate.
        """
        circuit_list = list()
        circuit = self.qaoa_circuit(params)
        qc_qaoa = transpile(circuit, basis_gates=target_basis, optimization_level=0)
        dag = circuit_to_dag(qc_qaoa)
        layers = list(dag.multigraph_layers())

        for k in range(len(layers)):
            if k==0 or k==len(layers)-1:
                continue
            qc_qaoa_res = transpile(qc_qaoa, basis_gates=target_basis, optimization_level=0)
            dag = circuit_to_dag(qc_qaoa_res)
            layers = list(dag.multigraph_layers())
            for layer in layers:
                for node in layer:
                    if layers.index(layer) != k and node._type == 'op':
                        dag.remove_op_node(node)

            layer_circ = transpile(dag_to_circuit(dag), basis_gates=target_basis, optimization_level=0)
            circuit_list.append(layer_circ)
        return circuit_list

    def hamiltonian_from_list(self, params: QAOAVariationalBaseParams):
        """
        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
    
        Returns
        -------
        hamiltonian_list: `list`
            List of scaled Hamiltonians (np.array), each entry corresponds to the Hamiltonian of a layer
        
        time_list: `list`
            List of times, each entry corresponds to the time of a layer
        
        gate_list: `list`
            List of dictionaries (corresponding to the layers) containing the names of gates acting on specific qubits             
        """
        rz_type = "<class 'qiskit.circuit.library.standard_gates.rz.RZGate'>"
        sx_type = "<class 'qiskit.circuit.library.standard_gates.sx.SXGate'>"
        cx_type = "<class 'qiskit.circuit.library.standard_gates.x.CXGate'>"
        x_type = "<class 'qiskit.circuit.library.standard_gates.x.XGate'>"
                
        circuit_list = self.get_circuit_list(params)
        t_gate_list = self.times[:2]
        width = circuit_list[0].width()

        rz_h = Qobj(0.5*np.array([[1, 0], [0, -1]]))
        sx_h = Qobj(np.pi*np.array([[0, 0.25], [0.25, 0]]))
        cx_t_h = Qobj(np.pi*0.5*np.array([[1, -1], [-1, 1]]))
        cx_c_h = Qobj(np.array([[0, 0], [0, 1]]))
        x_h = Qobj(np.pi*0.5*np.array([[1, -1], [-1, 1]]))
        id_h = qeye(2)
        
        def get_gate(gate_type):
            if gate_type==rz_type:
                return rz_h*float(gate[0].params[0])
            elif gate_type==sx_type:
                return sx_h
            elif gate_type==x_type:
                return x_h
            elif gate_type==cx_type:
                return [cx_t_h, cx_c_h]
            else:
                raise ValueError(f'Gate type is {gate_type}')
        
        def get_tensorprod(gate_type, position):
            matrices = [id_h]*width
            if gate_type==cx_type:
                matrices[position[0]] = get_gate(gate_type)[0]  
                matrices[position[1]] = get_gate(gate_type)[1] 
            else:
                matrices[position] = get_gate(gate_type)
            
            if width == 1:
                tensorprod=matrices[0]
            else:
                tensorprod = tensor(matrices[0], matrices[1])
                for k in range(width-2):
                    l = k+2
                    tensorprod = tensor(tensorprod, matrices[l])
            return tensorprod
                            
        hamiltonian_list = list()
        time_list = list()
        gate_list = list()
        
        for layer in circuit_list:
            layer_dict = dict()
            current_hamiltonian = 0
            two_qubit_gate = 0
            for g,gate in enumerate(layer):
                gate_type=str(type(gate[0]))
                if gate_type==cx_type:
                    two_qubit_gate = 1
                    control_qubit=gate[1][0].index
                    target_qubit=gate[1][1].index
                    layer_dict[control_qubit] = 'cx_c'
                    layer_dict[target_qubit] = 'cx_t'
                    current_hamiltonian += get_tensorprod(gate_type, [target_qubit, control_qubit])
                elif gate_type==rz_type or gate_type==sx_type or gate_type==x_type:
                    target_qubit=gate[1][0].index
                    if gate_type==x_type:
                        layer_dict[target_qubit] = 'x'
                    elif gate_type==sx_type:
                        layer_dict[target_qubit] = 'sx'
                    elif gate_type==rz_type:
                        layer_dict[target_qubit] = 'rz'
                    current_hamiltonian += get_tensorprod(gate_type, target_qubit)
                else:
                    raise ValueError(f'Gate type is {gate_type}')
            gate_list.append(layer_dict)
            if two_qubit_gate==0:
                time_list.append(t_gate_list[0])
                hamiltonian_list.append(np.array(current_hamiltonian)/t_gate_list[0])
            else:
                time_list.append(t_gate_list[1])
                hamiltonian_list.append(np.array(current_hamiltonian)/t_gate_list[1])
                    
        return hamiltonian_list, time_list, gate_list  

    def insert_op(self, op, k, n):
        """
        Parameters
        ----------
        op: `qutip.qobj.Qobj`
            Qutip quantum operator (single qubit quantum operator) to be positioned at a position k in a list of length n 
        
        k: `int`
            Position at which the operator op should be placed
    
        Returns
        -------
        current_op: `qutip.qobj.Qobj`
            Qutip object that is constructed by tensor products of n Qutip objects, namely of the operator op at position k and the single qubit
            identity operator at all other positions
        """
        if k==0:
            current_op = op
        else:
            current_op = identity(2)
            for j in list(range(n))[1:k]:
                current_op = tensor(current_op, identity(2))
            current_op = tensor(current_op, op)
        for l in list(range(n))[k+1:]:
            current_op = tensor(current_op, identity(2))
        return current_op

    def run_circuit_mastereqn(self, params: QAOAVariationalBaseParams): 
        """
        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
    
        Returns
        -------
        states: `list`
            List containing the states of the circuit at times indicated in next list
        
        times: `list`
            List containing the times of the states indicated in the above list

        jumps: `list`
            List of dicts specifying which jumps happened at what times and on which qubits
        """
        hamiltonian_list, time_list, gate_list = self.hamiltonian_from_list(params)
        width = len(hamiltonian_list[0])
        n = int(np.log2(width))
        psi0 = basis(width,0)
        jumps = dict() # time: [qubit, jump operator]
        states = list()
        times = list()

        decay = bool(self.noise_model['decay'])
        dephasing = bool(self.noise_model['dephasing'])
        overrot = bool(self.noise_model['overrot'])
        depol = bool(self.noise_model['depol1'] or self.noise_model['depol2'])
        spam = bool(self.noise_model['spam'])
        readout = bool(self.noise_model['readout01'] or self.noise_model['readout10'])
        
        T1 = float(self.noise_model['decay'])
        T2 = float(self.noise_model['dephasing'])
        overrot_st_dev = float(self.noise_model['overrot'])
        gate_error_list = [float(self.noise_model['depol1']), float(self.noise_model['depol2'])]
        spam_error_prob = float(self.noise_model['spam'])
        meas1_prep0_prob = float(self.noise_model['readout10'])
        meas0_prep1_prob = float(self.noise_model['readout01'])

        t_readout = self.times[-1]
        
        if self.allowed_jump_qubits is None:
            self.allowed_jump_qubits = list(range(n))
        
        ops_ind = list()
        
        decay_ops = list()
        if decay:
            for k in range(n):
                if k not in self.allowed_jump_qubits:
                    continue
                current_op = Qobj(np.array(self.insert_op(destroy(2), k, n))) #*float(1/(np.sqrt(T1))) 
                decay_ops.append(current_op)
                ops_ind.append([k, 'decay'])
                
        dephasing_ops = list()
        if dephasing:
            for k in range(n):
                if k not in self.allowed_jump_qubits:
                    continue
                current_op = Qobj(np.array(self.insert_op(sigmaz(), k, n))) #*float(1/(np.sqrt(T2))), Qobj([[0,0],[0,1]])
                dephasing_ops.append(current_op) 
                ops_ind.append([k, 'dephasing'])

        t_start = 0

        for k in range(len(hamiltonian_list)):
            c_ops = list([x*np.sqrt((1-np.exp(-time_list[k]/T1))/time_list[k]) for x in decay_ops] + [x*np.sqrt((1-np.exp(-time_list[k]/(2*T2)))/time_list[k]) for x in dephasing_ops])
            current_ops_ind = ops_ind.copy()
            for key in gate_list[k].keys():
                if key not in self.allowed_jump_qubits:
                    continue
                if overrot:
                    val = np.random.normal(0,overrot_st_dev)
                    x = val/100*2*np.pi
                    if gate_list[k][key]=='x' or gate_list[k][key]=='sx' or gate_list[k][key]=='cx_t':
                        c_ops.append(Qobj(np.array(self.insert_op(Qobj([[np.cos(x/2), -1j*np.sin(x/2)],[-1j*np.sin(x/2), np.cos(x/2)]])*np.sqrt(1/time_list[k]),key,n))))
                        
                    elif gate_list[k][key]=='rz':
                        c_ops.append(Qobj(np.array(self.insert_op(Qobj([[np.exp(-1j*0.5*x),0],[0,np.exp(1j*0.5*x)]])*np.sqrt(1/time_list[k]),key,n))))
                    current_ops_ind.append([key, 'overrot'])
                if depol:
                    if gate_list[k][key]=='cx_c' or gate_list[k][key]=='cx_t':
                        c_ops.append(Qobj(np.array(self.insert_op(sigmax()*np.sqrt(gate_error_list[1]/(time_list[k]*3)),key,n))))
                        c_ops.append(Qobj(np.array(self.insert_op(sigmay()*np.sqrt(gate_error_list[1]/(time_list[k]*3)),key,n))))
                        c_ops.append(Qobj(np.array(self.insert_op(sigmaz()*np.sqrt(gate_error_list[1]/(time_list[k]*3)),key,n))))
                        
                    elif gate_list[k][key]=='x' or gate_list[k][key]=='sx' or gate_list[k][key]=='rz':
                        c_ops.append(Qobj(np.array(self.insert_op(sigmax()*np.sqrt(gate_error_list[0]/(time_list[k]*3)),key,n))))
                        c_ops.append(Qobj(np.array(self.insert_op(sigmay()*np.sqrt(gate_error_list[0]/(time_list[k]*3)),key,n))))
                        c_ops.append(Qobj(np.array(self.insert_op(sigmaz()*np.sqrt(gate_error_list[0]/(time_list[k]*3)),key,n))))
                    current_ops_ind.append([key, 'depolx'])
                    current_ops_ind.append([key, 'depoly'])
                    current_ops_ind.append([key, 'depolz'])
                    
            if spam and k==len(hamiltonian_list)-1:
                for m in range(n):
                    if m not in self.allowed_jump_qubits:
                        continue
                    c_ops.append(Qobj(np.array(self.insert_op(sigmax()*np.sqrt(spam_error_prob/time_list[k]), m, n))))
                    current_ops_ind.append([m, 'spam'])

            tlist = np.linspace(0,time_list[k],100)
            times += [x+t_start for x in tlist]
            stdout_backup = sys.stdout
            sys.stdout = sys.__stdout__
            mc = mcsolve(H=Qobj(hamiltonian_list[k]), psi0=psi0, tlist=tlist, c_ops=c_ops, e_ops=[], ntraj=[1], progress_bar=None)
            sys.stdout = stdout_backup
            if c_ops==[]:
                psi0 = mc.states[-1] # final result of previous step
                states.extend(mc.states)
            else:
                psi0 = mc.states[0][-1]
                states.extend(mc.states[0])
            jump_times = mc.col_times
            jump_ind = mc.col_which
            if jump_ind is not None:
                for count,ind in enumerate(jump_ind[0]):
                    jumps[t_start+jump_times[0][count]] = current_ops_ind[ind] # time: [qubit, type]
            t_start += time_list[k]
            
        if readout:
            readout_ops = list()
            current_ops_ind = list()
            for k in range(n):
                if k not in self.allowed_jump_qubits:
                    continue
                current_op = Qobj(np.array(self.insert_op(destroy(2), k, n)))
                readout_ops.append(current_op*np.sqrt(meas0_prep1_prob/t_readout)) 
                current_op = Qobj(np.array(self.insert_op(destroy(2).dag(), k, n)))
                readout_ops.append(current_op*np.sqrt(meas1_prep0_prob/t_readout))
                current_ops_ind.append([k, 'readout01'])
                current_ops_ind.append([k, 'readout10'])
            
            tlist = np.linspace(0,t_readout,100)
            times += [x+t_start for x in tlist]
            stdout_backup = sys.stdout
            sys.stdout = sys.__stdout__
            mc = mcsolve(H=Qobj(np.array(self.insert_op(identity(2), 1, n))), psi0=psi0, tlist=tlist, c_ops=readout_ops, e_ops=[], ntraj=[1], progress_bar=None)
            sys.stdout = stdout_backup
            psi0 = mc.states[0][-1] # final result of previous step
            jump_times = mc.col_times
            jump_ind = mc.col_which
            if jump_ind is not None:
                for count,ind in enumerate(jump_ind[0]):
                    jumps[t_start+jump_times[0][count]] = current_ops_ind[ind] 
            states.extend(mc.states[0])
                                    
        return states, times, jumps

    def get_results(self, params: QAOAVariationalBaseParams) -> np.array:
        """
        Returns the density matrix of the final QAOA circuit after binding angles from variational parameters.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
        
        Returns
        -------
        results: `list`
            List containing the final density matrix of the QAOA circuit, a list of states sampled during the execution of the circuit at times indicated 
            in a list called times and a dictionary jumps, with keys being times at which jumps occurred and values being lists [qubit, type] where qubit 
            indicates on which qubit the respective jump occurred and type what kind of jump occurred. All for n_shots trajectories.
        """
        results = list()
        for k in range(self.n_shots):
            states, times, jumps = self.run_circuit_mastereqn(params)
            results.append([states, times, jumps])

        return results

    def get_counts(self, params: QAOAVariationalBaseParams) -> dict:
        """
        Returns n_shots trajectories of the final QAOA circuit after binding angles from variational parameters.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
        
        Returns
        -------
        counts: `dict`
            The n_shots trajectories of the final QAOA circuit after binding angles from variational parameters produced by the probability distribution
            obtrained when averaging over n_shots trajectories
        """
        results = self.get_results(params)

        n = self.n_qubits
        shots = self.n_shots
        counts = dict()
        logn = self.n_qubits
        outcomes = list()

        for k in range(len(results)):
            rho = np.array(results[k][0][-1]*results[k][0][-1].dag())
            sample = np.random.choice(a=[f'{m:0{logn}b}' for m in range(len(rho))], p=[np.real(rho[m][m]) for m in range(len(rho))], size=1)
            outcomes.append(sample[0])

        for k in outcomes:
            if str(k) in counts.keys():
                continue
            counts[str(k)] = outcomes.count(k)

        return counts

    #expected cost given probability distribution from density matrix, there is already a function that essentially does that somewhere else
    def expectation(self, params: QAOAVariationalBaseParams):
        """
        expected cost given probability distribution from density matrix
        """
        exp = cost_function(self.probability_dict(params), self.cost_hamiltonian)
        return np.real(exp)

    #dictionary {string: probability} with probabilities of outcomes
    def probability_dict(self, params: QAOAVariationalBaseParams): 
        """
        Dictionary {string: probability} with probabilities of outcomes
        """
        probs = dict()
        results = self.get_results(params)
        rho_sum = np.array(results[0][0][-1]*results[0][0][-1].dag())
        
        if len(results)>1:
            for k in list(range(len(results)))[1:]:
                rho_sum += np.array(results[k][0][-1]*results[k][0][-1].dag())          

        rho_sum /= self.n_shots

        n = len(rho_sum)
        logn = int(np.log2(n))

        for k in range(n):
            probs[f'{k:0{logn}b}'] = np.real(rho_sum[k][k])

        return probs

    def circuit_to_qasm(self, params: QAOAVariationalBaseParams):
        """
        A method to convert the QAOA circuit to QASM.
        """
        raise NotImplementedError()
#         qasm_circuit = self.parametric_circuit.qasm()
#         return qasm_circuit

    def reset_circuit(self):
        raise NotImplementedError()

    def qfim(self, params: QAOAVariationalBaseParams,eta: float = 0.00000001):

        raise NotImplementedError()




    #are the functions below needed? If so, reconstruct G from circuit_params and vice versa for optimisation procedure?
    #def get_exp(mat, G, params: QAOAVariationalBaseParams):
    #    expectation = 0
    #    for k in range(len(mat)):
    #        prob = np.real(mat[k][k])
    #        n = len(G.nodes())
    #        bitstring = f'{k:0{n}b}'
    #        obj = maxcut_obj(bitstring, G)
    #        expectation += prob * obj
    #    return expectation

    #change this so that G is not needed, update p, change how circuit is created
    #G only needed for creating circuit (not needed here) and for calculating expectation
    #def get_expectation_me(self, G, p, sign=1, allowed_jump_qubits=None, params: QAOAVariationalBaseParams): #why is this red?
    #    n = len(G.nodes())
        
    #    def execute_circ(theta):        
    #        circuit = self.qaoa_circuit(self, params)
    #        expectation = self.get_exp(self.run_circuit_mastereqn(circuit, params, allowed_jump_qubits=allowed_jump_qubits), G)
            
    #        return expectation*sign
        
    #    return execute_circ
    
    #change this so that G is not needed, update params
    #create new circuit by passing params?

    #def run_qaoa_me(self, G, p, params: QAOAVariationalBaseParams, method='COBYLA'):
    #    bounds = None #[(0,2*np.pi) for x in range(2*p_qt)]
    #    tol=1e-5

    #    expectation = self.get_expectation_me(G=G, p=p, params=params)
    #    min_res = minimize(expectation,[0.1 for x in range(2*p)], method=method)
    #    circuit = create_qaoa_circ(G, min_res.x, meas=False)
    #    min_expectation_me = get_exp(run_circuit_mastereqn(circuit, params), G)
    #    return min_expectation_me 

    #for k in range(n):
    #string = f'{k:0{n}b}'
    #count = shots*rho[k][k]
    #counts[string] = count
    