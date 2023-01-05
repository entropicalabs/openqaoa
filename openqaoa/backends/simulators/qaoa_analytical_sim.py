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
Energy expectation as a function of angles computed accordingly to the analytical expression for p=1.
"""
import numpy as np

from ...basebackend import QAOABaseBackend, QuantumCircuitBase
from ...qaoa_parameters.baseparams import QAOAVariationalBaseParams
from ...qaoa_parameters.operators import Hamiltonian
from openqaoa.utilities import energy_expectation_analytical

# not sure about those ones:
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams 
from typing import Union, List, Tuple, Type, Optional


class QAOABackendAnalyticalSimulator(QAOABaseBackend):
    r"""
    A simulator class, specific for QAOA with a single layer, p=1, starting with a layer of Hadamards and using the X mixer. 
    Works by calculating the expectation value of the given quantum circuit (specificied with beta and gamma angles) from the analytical formula derived in arXiv:2011.13420v2.
     
    Parameters
    ----------
    circuit_params: QAOACircuitParams
        An object of the class ``QAOACircuitParams`` which contains information on 
        circuit construction and depth of the circuit. Note that it only works for p=1 and the X Mixer.
    """
    
    def __init__(self, circuit_params: QAOACircuitParams, 
                                         prepend_state = None,
                                         append_state = None,
                                         init_hadamard = True,
                                         cvar_alpha = 1):
        
        # checking if not supported parameters are passed
        for k, val in {"Prepend_state": (prepend_state, None), "append_state": (append_state, None), "init_hadamard": (init_hadamard, True), "cvar_alpha": (cvar_alpha, 1)}.items():
            if val[0] != val[1]:
                print(f"{k} is not supported for the analytical backend. {k} is set to None.")
        
                                       
        QAOABaseBackend.__init__(self, circuit_params,
                                         prepend_state = None,
                                         append_state = None,
                                         init_hadamard = True,
                                         cvar_alpha = 1)
        
        self.measurement_outcomes = {}  # passing empty dict for the logger since measurements are irrelevant for this backend.
        
        # check if conditions for the analytical formula are met
        assert self.circuit_params.p == 1, "Analytical formula only holds for p=1."
        
        for gatemap in self.circuit_params.mixer_qubits_singles:
            assert gatemap == 'RXGateMap', "Analytical formula only holds for X mixer."
        
    def assign_angles(self):
        raise NotImplementedError("This method is irrelevant for this backend")
        
    def obtain_angles_for_pauli_list(self):
        raise NotImplementedError("This method is irrelevant for this backend")
        
    def qaoa_circuit(self):
        raise NotImplementedError("This method is irrelevant for this backend")
        
    def get_counts(self):
        raise NotImplementedError("This method is irrelevant for this backend")
        
    def expectation(self, params: QAOAVariationalBaseParams) -> float:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian analytically.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing 
            variable parameters.

        Returns
        -------
        float:
            Expectation value of cost operator wrt to the QAOA parameters according to the analytical expression for p=1.
        """
        betas = params.betas
        gammas = params.gammas
        
        cost = energy_expectation_analytical([betas, gammas], self.cost_hamiltonian)
        return cost
    
    def expectation_w_uncertainty(self, params):
        raise NotImplementedError("Not implemented yet. In progress.")
        
    def reset_circuit(self):
        raise NotImplementedError("This method is irrelevant for this backend")
    
    def circuit_to_qasm(self):
        raise NotImplementedError("This method is irrelevant for this backend")
    
    
        
    
        
    
    
    
