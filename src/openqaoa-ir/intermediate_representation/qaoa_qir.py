from abc import abstractmethod
from typing import Optional, List, Callable
from pyqir.generator import BasicQisBuilder, SimpleModule, ir_to_bitcode

# from pyqir.qis import cx, cz, rx, ry, rz
from openqaoa.qaoa_components.ansatz_constructor import QAOADescriptor
from openqaoa.qaoa_components.variational_parameters import (
    QAOAVariationalBaseParams,
)
from openqaoa.qaoa_components.ansatz_constructor import gates, RotationAngle
from openqaoa.backends.basebackend import (
    QAOABaseBackendShotBased,
)
from openqaoa_azure.backends import DeviceAzure


class QAOAIntermediateBaseRepresentation(QAOABaseBackendShotBased):
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
    qaoa_descriptor: `QAOADescriptor`
        An object of the class ``QAOADescriptor`` which contains information on
        circuit construction and depth of the circuit.
    prepend_state: `QIR`
        Instructions prepended to the circuit.
    append_state: `QIR`
        Instructions appended to the circuit.
    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the
        QAOA part of the circuit.
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        n_shots: int,
        prepend_state: Optional[str],
        append_state: Optional[str],
        init_hadamard: bool,
        cvar_alpha: float,
        qubit_layout: List[int] = [],
    ):
        super().__init__(
            qaoa_descriptor,
            n_shots,
            prepend_state,
            append_state,
            init_hadamard,
            cvar_alpha,
        )
        self.n_qubits = qaoa_descriptor.cost_hamiltonian.n_qubits
        self.qureg = qaoa_descriptor.qureg
        self.abstract_circuit = qaoa_descriptor.abstract_circuit
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

    # @abstractmethod
    # def qaoa_circuit(self, params: QAOAVariationalBaseParams):
    #     """
    #     Generate circuit instructions in Intermediate Representation using
    #     the qaoa_descriptor object, assigned with variational_params
    #     Parameters
    #     ----------
    #     params: `QAOAVariationalBaseParams`
    #         The QAOA variational parameters

    #     Returns
    #     -------
    #     qaoa_circuit: Intermediate Representation
    #         The QAOA instructions with designated angles based on `params`
    #     """
    #     raise NotImplementedError()

    # def assign_angles(self, params: QAOAVariationalBaseParams):
    #     """
    #     Assigns the angle values of the variational parameters to the circuit gates
    #     specified as a list of gates in the ``abstract_circuit``.
    #     Parameters
    #     ----------
    #     params: `QAOAVariationalBaseParams`
    #         The variational parameters(angles) to be assigned to the circuit gates
    #     """
    #     # if circuit is non-parameterised, then assign the angle values to the circuit
    #     abstract_pauli_circuit = self.abstract_circuit

    #     for each_pauli in abstract_pauli_circuit:
    #         pauli_label_index = each_pauli.pauli_label[2:]
    #         if isinstance(each_pauli, TwoQubitRotationGateMap):
    #             if each_pauli.pauli_label[1] == "mixer":
    #                 angle = params.mixer_2q_angles[
    #                     pauli_label_index[0], pauli_label_index[1]
    #                 ]
    #             elif each_pauli.pauli_label[1] == "cost":
    #                 angle = params.cost_2q_angles[
    #                     pauli_label_index[0], pauli_label_index[1]
    #                 ]
    #         elif isinstance(each_pauli, RotationGateMap):
    #             if each_pauli.pauli_label[1] == "mixer":
    #                 angle = params.mixer_1q_angles[
    #                     pauli_label_index[0], pauli_label_index[1]
    #                 ]
    #             elif each_pauli.pauli_label[1] == "cost":
    #                 angle = params.cost_1q_angles[
    #                     pauli_label_index[0], pauli_label_index[1]
    #                 ]
    #         each_pauli.rotation_angle = angle
    #     self.abstract_circuit = abstract_pauli_circuit


class QISGateApplicator(gates.GateApplicator):
    library = "PYQIR"

    def __init__(self, qaoa_module: SimpleModule, qis_builder: BasicQisBuilder):
        self.qaoa_module = qaoa_module
        self.qis_builder = qis_builder

        self.QIS_OQ_MAPPER = {
            gates.CX.__name__: self.qis_builder.cx,
            gates.CZ.__name__: self.qis_builder.cz,
            gates.RX.__name__: self.qis_builder.rx,
            gates.RY.__name__: self.qis_builder.ry,
            gates.RZ.__name__: self.qis_builder.rz,
        }

    def gate_selector(self, gate: gates.Gate, qis_builder: BasicQisBuilder) -> Callable:
        selected_qiskit_gate = self.QIS_OQ_MAPPER[gate.__name__]
        return selected_qiskit_gate

    def apply_1q_rotation_gate(
        self, qis_gate, qubit_1, rotation_object: RotationAngle, ckt=None
    ) -> None:
        qubit_1 = self.qaoa_module.qubits[qubit_1]
        qis_gate(rotation_object.rotation_angle, qubit_1)

    def apply_2q_rotation_gate(
        self, qis_gate, qubit_1, qubit_2, rotation_object: RotationAngle, ckt=None
    ) -> None:
        qubit_1 = self.qaoa_module.qubits[qubit_1]
        qubit_2 = self.qaoa_module.qubits[qubit_2]
        qis_gate(rotation_object.rotation_angle, qubit_1, qubit_2)

    def apply_1q_fixed_gate(self, qis_gate, qubit_1: int, ckt=None) -> None:
        qubit_1 = self.qaoa_module.qubits[qubit_1]
        qis_gate(qubit_1)

    def apply_2q_fixed_gate(
        self, qis_gate, qubit_1: int, qubit_2: int, ckt=None
    ) -> None:
        qubit_1 = self.qaoa_module.qubits[qubit_1]
        qubit_2 = self.qaoa_module.qubits[qubit_2]
        qis_gate(qubit_1, qubit_2)

    def apply_gate(self, gate: gates.Gate, *args):
        selected_qiskit_gate = self.gate_selector(gate, self.qis_builder)
        if gate.n_qubits == 1:
            if hasattr(gate, "rotation_object"):
                # *args must be of the following format -- (qubit_1,rotation_object,circuit)
                return self.apply_1q_rotation_gate(selected_qiskit_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,circuit)
                return self.apply_1q_fixed_gate(selected_qiskit_gate, *args)
        elif gate.n_qubits == 2:
            if hasattr(gate, "rotation_object"):
                # *args must be of the following format -- (qubit_1,qubit_2,rotation_object,circuit)
                return self.apply_2q_rotation_gate(selected_qiskit_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,qubit_2,circuit)
                return self.apply_2q_fixed_gate(selected_qiskit_gate, *args)
        else:
            raise ValueError("Only 1 and 2-qubit gates are supported.")


class QAOAQIR(QAOAIntermediateBaseRepresentation):
    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        device: DeviceAzure,
        n_shots: int,
        prepend_state: Optional[str] = None,
        append_state: Optional[str] = None,
        init_hadamard: bool = True,
        cvar_alpha: float = 1,
        qubit_layout: List[int] = [],
    ):
        super().__init__(
            qaoa_descriptor,
            n_shots,
            prepend_state,
            append_state,
            init_hadamard,
            cvar_alpha,
            qubit_layout,
        )

        self.qaoa_module = SimpleModule(
            "QAOA", num_qubits=self.n_qubits, num_results=self.n_qubits
        )
        self.qureg = self.qaoa_module.qubits
        self.qis_builder = BasicQisBuilder(self.qaoa_module.builder)
        self.gate_applicator = QISGateApplicator(self.qaoa_module, self.qis_builder)

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
        # mapper = {
        #     gates.CX: self.qis_builder.cx,
        #     gates.CZ: self.qis_builder.cz,
        #     gates.RX: self.qis_builder.rx,
        #     gates.RY: self.qis_builder.ry,
        #     gates.RZ: self.qis_builder.rz,
        # }
        mapper = None
        return mapper

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> SimpleModule:
        self.assign_angles(params)

        # TODO: Replace all logic with `apply_gate` method
        # if isinstance(self.prepend_state, SimpleModule)
        # join the circuits

        if self.init_hadamard is True:
            for i in self.qureg:
                self.qis_builder.h(i)

        for i, each_gate in enumerate(self.abstract_circuit):
            decomposition = each_gate.decomposition("standard")
            for each_tuple in decomposition:
                oq_gate = each_tuple[0](self.gate_applicator, *each_tuple[1])
                oq_gate.apply_gate(None)

        # TODO: Apply append state if provided by user
        # if isinstance(self.append_state, SimpleModule):
        # append the circuit
        for i in range(len(self.qureg)):
            self.qis_builder.mz(self.qureg[i], self.qaoa_module.results[i])

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

        # construct the circuit using BasicQisBuilder
        self.qaoa_circuit(params)

        # return the IR using the QAOA SimpleModule
        ir = self.qaoa_module.ir()

        return ir

    @staticmethod
    def append_record_output_to_ir(ir: str):
        instructions = "declare void @__quantum__rt__array_start_record_output()\n\n"
        instructions += "declare void @__quantum__rt__array_end_record_output()\n\n"
        instructions += (
            "declare void @__quantum__rt__result_record_output(%Result*)\n\n"
        )
        return instructions

    def convert_to_bitcode(self, params: QAOAVariationalBaseParams):
        """
        Convert the IR into bitcode using the QAOA SimpleModule
        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA variational parameters
        Returns
        -------
        bitcode:
                The bitcode corresponding to the QAOA circuit
        """
        ir = self.convert_to_ir(params)

        bitcode = ir_to_bitcode(ir)
        return bitcode

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        pass

    def circuit_to_qasm(self):
        pass

    def reset_circuit(self):
        pass
