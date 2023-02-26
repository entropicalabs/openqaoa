from typing import Callable
from ..qaoa_components.ansatz_constructor import gates as gates_core
from ..qaoa_components.ansatz_constructor.rotationangle import RotationAngle
from .qaoa_vectorized import QAOAvectorizedBackendSimulator as vec_backend

class VectorizedGateApplicator(gates_core.GateApplicator):

	VECTORIZED_OQ_GATE_MAPPER = lambda x: {
        gates_core.RZ: x.apply_rz,
		gates_core.RX: x.apply_rx,
		gates_core.RY: x.apply_ry,
		gates_core.RXX: x.apply_rxx,
		gates_core.RZX: x.apply_rzx,
		gates_core.RZZ: x.apply_rzz,
		gates_core.RYY: x.apply_ryy,
		gates_core.RYZ: x.apply_ryz,
		gates_core.RiSWAP: x.apply_rxy
	}

	def __init__(self):
		self.library = 'vectorized'

	def gate_selector(self, gate:gates_core.Gate, vectorized_backend: vec_backend):
		try:
			selected_gate = VectorizedGateApplicator.VECTORIZED_OQ_GATE_MAPPER(vectorized_backend)[gate]
		except KeyError:
			raise ValueError("Specified gate is not supported by the backend")
		return selected_gate

	@staticmethod
	def apply_1q_rotation_gate(
        vectorized_gate: Callable,
		qubit_1: int,
		rotation_object: RotationAngle
	):
		vectorized_gate(
			qubit_1,
			rotation_object.rotation_angle
		)

	@staticmethod
	def apply_2q_rotation_gate(
		vectorized_gate: Callable,
		qubit_1: int,
		qubit_2: int,
		rotation_object: RotationAngle
	):
		vectorized_gate(
			qubit_1,
			qubit_2,
			rotation_object.rotation_angle
		)

	@staticmethod
	def apply_1q_fixed_gate(
		vectorized_gate: Callable,
		qubit_1: int
	):
		vectorized_gate(
			qubit_1
		)

	@staticmethod
	def apply_2q_fixed_gate(
		vectorized_gate: Callable,
		qubit_1: int,
		qubit_2: int
	):
		vectorized_gate(
			qubit_1,
			qubit_2
		)

	def apply_gate(self, gate: gates_core.Gate, *args):
		selected_vector_gate = self.gate_selector(gate, args.pop[:-1])
		if gate.n_qubits == 1:
			if hasattr(gate, 'rotation_object'):
				# *args must be of the following format -- (qubit_1,rotation_object)
				self.apply_1q_rotation_gate(selected_vector_gate, *args)
			else:
				# *args must be of the following format -- (qubit_1)
				self.apply_1q_fixed_gate(selected_vector_gate, *args)
		elif gate.n_qubits == 2:
			if hasattr(gate, 'rotation_object'):
				# *args must be of the following format -- (qubit_1,qubit_2,rotation_object)
				self.apply_2q_rotation_gate(selected_vector_gate, *args)
			else:
				# *args must be of the following format -- (qubit_1,qubit_2)
				self.apply_2q_fixed_gate(selected_vector_gate, *args)
		else:
			raise ValueError("Error applying the requested gate. Please check in the input")