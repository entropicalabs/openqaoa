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
from typing import Tuple
import numpy as np
from pyquil import Program, gates
from pyquil.api import WavefunctionSimulator

from ...basebackend import QAOABaseBackendParametric, QAOABaseBackendStatevector
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from ...qaoa_parameters.pauligate import (
    RXPauliGate, RYPauliGate, RZPauliGate)
from ...cost_function import cost_function


class QAOAPyQuilWavefunctionSimulatorBackend(QAOABaseBackendStatevector, QAOABaseBackendParametric):
    """
    A local Wavefunction simulator backend for the PyQuil service provider
    """
    PYQUIL_PAULIGATE_LIBRARY = [RXPauliGate, RYPauliGate, RZPauliGate]

    def __init__(self,
                 circuit_params: QAOACircuitParams,
                 prepend_state: Program = None,
                 append_state: Program = None,
                 init_hadamard: bool = True,
                 cvar_alpha: float = 1):

        QAOABaseBackendStatevector.__init__(self, circuit_params, prepend_state,
                                            append_state, init_hadamard,
                                            cvar_alpha)

        self.parametric_circuit = self.parametric_qaoa_circuit

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> Program:
        """
        Creates a QAOA circuit (pyquil.Program object), given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit. To do this, you will need to subsequently execute the command self.eng.flush().

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        `pyquil.Program`
                A pyquil.Program object.
        """
        angles_list = self.obtain_angles_for_pauli_list(
            self.pseudo_circuit, params)
        angle_declarations = list(self.parametric_circuit.declarations.keys())
        for i, param_name in enumerate(angle_declarations):
            self.parametric_circuit.write_memory(
                region_name=param_name, value=angles_list[i])

        return self.parametric_circuit

    @property
    def parametric_qaoa_circuit(self) -> Program:
        """
        Creates a QAOA circuit (pyquil.Program object), given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit. To do this, you will need to subsequently execute the command self.eng.flush().

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        `pyquil.Program`
                A pyquil.Program object.
        """
        parametric_circuit = Program()

        if self.prepend_state:
            parametric_circuit += self.prepend_state
        # Initial state is all |+>
        if self.init_hadamard:
            for i in range(self.n_qubits):
                parametric_circuit += gates.H(i)

        # create a list of gates in order of application on quantum circuit
        low_level_gate_list = []
        for each_gate in self.pseudo_circuit:
            gate_label = ''.join(str(label) for label in each_gate.pauli_label)
            angle_param = parametric_circuit.declare(
                f'pauli{gate_label}', 'REAL', 1)
            each_gate.rotation_angle = angle_param
            if type(each_gate) in self.PYQUIL_PAULIGATE_LIBRARY:
                decomposition = each_gate.decomposition('trivial')
            else:
                decomposition = each_gate.decomposition('standard')
            # using the list above, construct the circuit
            for each_tuple in decomposition:
                # qbitplaceholder = something
                gate = each_tuple[0](*each_tuple[1])
                parametric_circuit = gate.apply_gate(parametric_circuit, 'pyquil',
                                                     angle_param=angle_param)

        if self.append_state:
            parametric_circuit += self.append_state

        return parametric_circuit

    def wavefunction(self,
                     params: QAOAVariationalBaseParams):
        """
        Get the wavefunction of the state produced by the parametric circuit.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
                The QAOA parameters - an object of one of the parameter classes, containing 
                the variational parameters (angles).

        Returns
        -------
        wf: `List[complex]`
                pyquil Wavefunction object.
        """
        program = self.qaoa_circuit(params)

        wf_sim = WavefunctionSimulator()
        wf = wf_sim.wavefunction(program)
        return wf

    def expectation(self, params: QAOAVariationalBaseParams) -> float:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
                The QAOA parameters - an object of one of the parameter classes, containing 
                variable parameters.

        Returns
        -------
        `float`
                expectation value of cost operator wrt to quantum state produced by QAOA circuit
        """
        prob_dict = self.probability_dict(params)
        cost = cost_function(
            prob_dict, self.circuit_params.cost_hamiltonian, self.cvar_alpha)
        return cost

    def expectation_w_uncertainty(self,
                                  params: QAOAVariationalBaseParams) -> Tuple[float, float]:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian and its uncertainty

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
                The QAOA parameters - an object of one of the parameter classes, containing 
                variable parameters.

        Returns
        -------
        `Tuple[float]`
                expectation value and its uncertainty of cost operator wrt 
                to quantum state produced by QAOA circuit.
        """
        prob_dict = self.probability_dict(params)
        cost = cost_function(
            prob_dict, self.circuit_params.cost_hamiltonian, self.cvar_alpha)
        cost_sq = cost_function(prob_dict,
                                self.circuit_params.cost_hamiltonian.hamiltonian_squared,
                                self.cvar_alpha)

        uncertainty = np.sqrt(cost_sq - cost**2)

        return (cost, uncertainty)

    def circuit_to_qasm(self):
        raise NotImplementedError()

    def reset_circuit(self):
        raise NotImplementedError()
