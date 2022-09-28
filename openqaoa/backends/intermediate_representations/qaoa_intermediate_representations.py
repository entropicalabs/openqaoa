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
#   limitations under the License

from abc import abstractmethod
from typing import Optional, List
from pyqir.generator import BasicQisBuilder, SimpleModule, ir_to_bitcode, types
from pyqir.generator import SimpleModule
from openqaoa.qaoa_parameters.baseparams import QAOACircuitParams
from openqaoa.qaoa_parameters.baseparams import QAOAVariationalBaseParams
from openqaoa.qaoa_parameters import gates
from openqaoa.qaoa_parameters import (
	RotationGateMap, TwoQubitRotationGateMap, OneQubitGate,
	TwoQubitGate, TwoQubitGateWithAngle
)


class QAOAIntermediateBaseRepresentation:
	"""
	This class provides a skeleton for classes implementing
	Intermediate Representations (IRs) for a given QAOA circuit
	provided information about the circuit to be constructed.
	Internally, OpenQAOA uses a custom set of objects as instructions
	required to build the circuit. This class provides a mechanism to
	map this internal representation to the industry accepted Intermediate
	Representations, for instance, `OpenQASM3`, `QIR` ... 

    Parameters
    ----------
    circuit_params: `QAOACircuitParams`
        An object of the class ``QAOACircuitParams`` which contains information on 
        circuit construction and depth of the circuit.
    prepend_state: `QIR`
        Instructions prepended to the circuit.
    append_state: `QIR`
        Instructions appended to the circuit.
    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the 
        QAOA part of the circuit.
    """

	def __init__(self, 
				circuit_params: QAOACircuitParams,
				prepend_state: Optional[str],
				append_state: Optional[str],
				init_hadamard: bool,
				qubit_layout: List[int] = []
				):

		self.n_qubits = circuit_params.cost_hamiltonian.n_qubits
		self.qureg = circuit_params.qureg
		self.abstract_circuit = circuit_params.abstract_circuit
		self.prepend_state = prepend_state
		self.append_state = append_state
		self.init_hadamard = init_hadamard
		self.qubit_layout = qubit_layout

	@abstractmethod
	def gate_mapper(self):
		"""
		Construct a mapping from OpenQAOA gates to instructions of
		the Intermediate Representation

		Returns
		-------
		mapping: `dict`
			A dictionary representing a mapping between Intermediate
			Representation instructions and OpenQAOA gates
		"""
		raise NotImplementedError()
	
	@abstractmethod
	def qaoa_circuit(self, params: QAOAVariationalBaseParams):
		"""
		Generate circuit instructions in Intermediate Representation using
		the qaoa_circuit_params object, assigned with variational_params

		Parameters
		----------
		params: `QAOAVariationalBaseParams`
			The QAOA variational parameters
		
		Returns
		-------
		qaoa_circuit: Intermediate Representation
			The QAOA instructions with designated angles based on `params`
		"""
		raise NotImplementedError()

	def assign_angles(self, params: QAOAVariationalBaseParams):
		"""
        Assigns the angle values of the variational parameters to the circuit gates
        specified as a list of gates in the ``abstract_circuit``.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The variational parameters(angles) to be assigned to the circuit gates
        """
        # if circuit is non-parameterised, then assign the angle values to the circuit
		abstract_pauli_circuit = self.abstract_circuit

		for each_pauli in abstract_pauli_circuit:
			pauli_label_index = each_pauli.pauli_label[2:]
			if isinstance(each_pauli, TwoQubitRotationGateMap):
				if each_pauli.pauli_label[1] == 'mixer':
					angle = params.mixer_2q_angles[pauli_label_index[0],
													pauli_label_index[1]]
				elif each_pauli.pauli_label[1] == 'cost':
					angle = params.cost_2q_angles[pauli_label_index[0],
													pauli_label_index[1]]
			elif isinstance(each_pauli, RotationGateMap):
				if each_pauli.pauli_label[1] == 'mixer':
					angle = params.mixer_1q_angles[pauli_label_index[0],
													pauli_label_index[1]]
				elif each_pauli.pauli_label[1] == 'cost':
					angle = params.cost_1q_angles[pauli_label_index[0],
													pauli_label_index[1]]
			each_pauli.rotation_angle = angle
		self.abstract_circuit = abstract_pauli_circuit


class QAOAQIR(QAOAIntermediateBaseRepresentation):

	def __init__(self, 
				circuit_params: QAOACircuitParams,
				prepend_state: Optional[str] = None,
				append_state: Optional[str] = None,
				init_hadamard: bool = True,
				qubit_layout: List[int] = []):

		super().__init__(circuit_params, prepend_state,
						append_state, init_hadamard,
						qubit_layout)
			
		self.qaoa_module = SimpleModule("QAOA",
										num_qubits=self.n_qubits,
										num_results=self.n_qubits)
		self.qureg = self.qaoa_module.qubits
		self.qis_builder = BasicQisBuilder(self.qaoa_module.builder)

	@property
	def gate_mapper(self):
		"""
		Construct a mapping between gates of BasicQisBuilder
		and OpenQAOA

		Parameters
		----------
		qis_builder: `BasicQisBuilder`
			The object to hold the circuit instructions

		Returns
		-------
		mapping: `dict`
			The mapping between two gate sets
		"""
		mapper = {
			gates.CX: self.qis_builder.cx,
			gates.CZ: self.qis_builder.cz,
			gates.RX: self.qis_builder.rx,
			gates.RY: self.qis_builder.ry,
			gates.RZ: self.qis_builder.rz,
		}
		
		return mapper

	def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> SimpleModule:

		self.assign_angles(params)

		#TODO: Prepend QIS instructions provided by the user
		# if isinstance(self.prepend_state, SimpleModule)
			#join the circuits

		if self.init_hadamard is True:
			for i in self.qureg:
				self.qis_builder.h(i)

		for i,each_gate in enumerate(self.abstract_circuit):
			decomposition = each_gate.decomposition('standard')
			for each_tuple in decomposition:
				gate = each_tuple[0]
				callable_gate = self.gate_mapper[gate]
				if issubclass(gate,OneQubitGate):
					qubit_idx, rotation_angle_obj = each_tuple[1]
					qubit = self.qureg[qubit_idx]
					callable_gate(rotation_angle_obj.rotation_angle,qubit)
				elif issubclass(gate, TwoQubitGate):
					qubit_idx1,qubit_idx2 = each_tuple[1][0]
					qubit1, qubit2 = self.qureg[qubit_idx1],self.qureg[qubit_idx2]
					callable_gate(qubit1, qubit2)
				elif issubclass(gate, TwoQubitGateWithAngle):
					[qubit_idx1,qubit_idx2], rotation_angle_obj = each_tuple[1][0]
					qubit1, qubit2 = self.qureg[qubit_idx1],self.qureg[qubit_idx2]	
					callable_gate(rotation_angle_obj.rotation_angle,qubit1,qubit2)

		#TODO: Apply append state if provided by user
		# if isinstance(self.append_state, SimpleModule):
			#append the circuit
	
		return None

	def convert_to_ir(self, params: QAOAVariationalBaseParams):
		"""
		Convert the QAOA circuit designated with `params` using the 
		BasicQisBuilder into Intermediate Representation using
		built-in function

		Parameters
		----------
		params: `QAOAVariationalBaseParams`
			The QAOA variational parameters

		Returns
		-------
		ir: `str`
			The intermediate representation corresponding to the 
			QAOA circuit
		"""

		#construct the circuit using BasicQisBuilder
		self.qaoa_circuit(params)

		#return the IR using the QAOA SimpleModule
		ir = self.qaoa_module.ir()

		return ir

	
		

									
