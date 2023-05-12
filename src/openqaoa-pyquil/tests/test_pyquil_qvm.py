import unittest
import numpy as np
import pytest

from pyquil import Program, quilbase
from pyquil.gates import RX, RY, RZ

from openqaoa import QAOA
from openqaoa.qaoa_components import (
    create_qaoa_variational_params,
    QAOADescriptor,
    PauliOp,
    Hamiltonian,
)
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa.backends import QAOAvectorizedBackendSimulator, create_device
from openqaoa.problems import NumberPartition, QUBO
from openqaoa_pyquil.backends import DevicePyquil, QAOAPyQuilQPUBackend


class TestingQAOACostPyquilQVM(unittest.TestCase):

    """This Object tests the QAOA Cost PyQuil QPU object, which is tasked with the
    creation and execution of a QAOA circuit for the selected QPU provider and
    backend. `as_qvm` is set to be True throughout.

    For all of these tests, qvm and quilc must be running.
    """

    @pytest.mark.qvm
    def test_connection(self):
        """
        Checks if connection to qvm and quilc is successful.
        TODO : improve test
        """

        # Check connection to qvm
        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=3, compiler_timeout=3
        )

        # Check connection to quilc compiler
        program = Program().inst(RX(np.pi, 0))
        device_pyquil.quantum_computer.compiler.quil_to_native_quil(program)

        pass

    @pytest.mark.qvm
    def test_active_reset(self):
        """
        Test if active_reset works fine.
        Check for RESET instruction in parametric circuit when active_reset = True / False
        """

        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=3, compiler_timeout=3
        )
        cost_hamil = Hamiltonian([PauliOp("Z", (0,)), PauliOp("Z", (1,))], [1, 2], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)

        backend_obj_pyquil = QAOAPyQuilQPUBackend(
            qaoa_descriptor=qaoa_descriptor,
            device=device_pyquil,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1,
            n_shots=1,
            active_reset=True,
        )
        assert "RESET" in [
            str(instr) for instr in backend_obj_pyquil.parametric_circuit
        ]

        backend_obj_pyquil = QAOAPyQuilQPUBackend(
            qaoa_descriptor=qaoa_descriptor,
            device=device_pyquil,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1,
            n_shots=1,
            active_reset=False,
        )
        assert "RESET" not in [
            str(instr) for instr in backend_obj_pyquil.parametric_circuit
        ]

    @pytest.mark.qvm
    def test_rewiring(self):
        """
        Test if rewiring works fine.

        """

        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=3, compiler_timeout=3
        )
        cost_hamil = Hamiltonian([PauliOp("Z", (0,)), PauliOp("Z", (1,))], [1, 2], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)

        # Test if error is raised correctly
        self.assertRaises(
            ValueError,
            lambda: QAOAPyQuilQPUBackend(
                qaoa_descriptor=qaoa_descriptor,
                device=device_pyquil,
                prepend_state=None,
                append_state=None,
                init_hadamard=True,
                cvar_alpha=1,
                n_shots=1,
                rewiring="illegal string",
            ),
        )

        # Test when rewiring = 'PRAGMA INITIAL_REWIRING "NAIVE"'
        backend_obj_pyquil = QAOAPyQuilQPUBackend(
            qaoa_descriptor=qaoa_descriptor,
            device=device_pyquil,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1,
            n_shots=1,
            rewiring='PRAGMA INITIAL_REWIRING "NAIVE"',
        )
        assert 'PRAGMA INITIAL_REWIRING "NAIVE"' in [
            str(instr) for instr in backend_obj_pyquil.parametric_circuit
        ]

        # Test when rewiring = 'PRAGMA INITIAL_REWIRING "PARTIAL"'
        backend_obj_pyquil = QAOAPyQuilQPUBackend(
            qaoa_descriptor=qaoa_descriptor,
            device=device_pyquil,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1,
            n_shots=1,
            rewiring='PRAGMA INITIAL_REWIRING "PARTIAL"',
        )
        assert 'PRAGMA INITIAL_REWIRING "PARTIAL"' in [
            str(instr) for instr in backend_obj_pyquil.parametric_circuit
        ]

    @pytest.mark.qvm
    def test_qaoa_pyquil_expectation(self):
        """
        Checks if expectation value agrees with known values. Since angles are selected such that the final state is one of the computational basis states, shots do not matter (there is no statistical variance).
        """

        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=3, compiler_timeout=3
        )

        # Without interaction terms
        cost_hamil = Hamiltonian([PauliOp("Z", (0,)), PauliOp("Z", (1,))], [1, 1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)
        variate_params = create_qaoa_variational_params(
            qaoa_descriptor, "standard", "ramp"
        )

        args = [np.pi / 4, np.pi / 4]  # beta, gamma
        variate_params.update_from_raw(args)

        backend_obj_pyquil = QAOAPyQuilQPUBackend(
            qaoa_descriptor=qaoa_descriptor,
            device=device_pyquil,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1,
            n_shots=1,
        )
        backend_obj_pyquil.expectation(variate_params)

        assert np.isclose(backend_obj_pyquil.expectation(variate_params), -1)

    @pytest.mark.qvm
    def test_qaoa_pyquil_gate_names(self):
        """
        Checks if names of gates are correct, and no. of measurement gates match the no. of qubits.
        """

        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=3, compiler_timeout=3
        )

        # Without interaction terms
        cost_hamil = Hamiltonian([PauliOp("Z", (0,)), PauliOp("Z", (1,))], [1, 1], 1)
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)
        params = create_qaoa_variational_params(qaoa_descriptor, "standard", "ramp")
        backend_obj_pyquil = QAOAPyQuilQPUBackend(
            qaoa_descriptor=qaoa_descriptor,
            device=device_pyquil,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1,
            n_shots=1,
        )

        gate_names = [
            instr.name
            for instr in backend_obj_pyquil.parametric_circuit
            if type(instr) == quilbase.Gate
        ]
        assert gate_names == [
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RZ",
            "RX",
            "RX",
        ]

        measurement_gate_no = len(
            [
                instr
                for instr in backend_obj_pyquil.qaoa_circuit(params)
                if type(instr) == quilbase.Measurement
            ]
        )
        assert measurement_gate_no == 2

        # With interaction terms
        cost_hamil = Hamiltonian(
            [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
            [1, 1, 1],
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)
        params = create_qaoa_variational_params(qaoa_descriptor, "standard", "ramp")
        backend_obj_pyquil = QAOAPyQuilQPUBackend(
            qaoa_descriptor=qaoa_descriptor,
            device=device_pyquil,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1,
            n_shots=1,
        )

        gate_names = [
            instr.name
            for instr in backend_obj_pyquil.parametric_circuit
            if type(instr) == quilbase.Gate
        ]
        assert gate_names == [
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RZ",
            "RZ",
            "RZ",
            "CPHASE",
            "RX",
            "RX",
        ]

        measurement_gate_no = len(
            [
                instr
                for instr in backend_obj_pyquil.qaoa_circuit(params)
                if type(instr) == quilbase.Measurement
            ]
        )
        assert measurement_gate_no == 2

    @pytest.mark.qvm
    def test_circuit_init_hadamard(self):
        """
        Checks correctness of circuit for the argument `init_hadamard`.
        """

        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=3, compiler_timeout=3
        )

        # With hadamard
        cost_hamil = Hamiltonian(
            [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
            [1, 1, 1],
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)
        pyquil_backend = QAOAPyQuilQPUBackend(
            device_pyquil,
            qaoa_descriptor,
            n_shots=10,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1,
        )

        assert [
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RZ",
            "RZ",
            "RZ",
            "CPHASE",
            "RX",
            "RX",
        ] == [
            instr.name
            for instr in pyquil_backend.parametric_circuit
            if type(instr) == quilbase.Gate
        ]

        # Without hadamard
        cost_hamil = Hamiltonian(
            [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
            [1, 1, 1],
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)
        pyquil_backend = QAOAPyQuilQPUBackend(
            device_pyquil,
            qaoa_descriptor,
            n_shots=10,
            prepend_state=None,
            append_state=None,
            init_hadamard=False,
            cvar_alpha=1,
        )

        assert ["RZ", "RZ", "RZ", "RZ", "CPHASE", "RX", "RX"] == [
            instr.name
            for instr in pyquil_backend.parametric_circuit
            if type(instr) == quilbase.Gate
        ]
        
    @pytest.mark.qvm
    def test_circuit_append_state(self):
        """
        Checks correctness of circuit for the argument `append_state`.
        """

        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=3, compiler_timeout=3
        )

        # With append_state
        cost_hamil = Hamiltonian(
            [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
            [1, 1, 1],
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)

        params = create_qaoa_variational_params(qaoa_descriptor, "standard", "ramp")

        append_circuit = Program().inst(RX(np.pi, 0), RY(np.pi / 2, 1), RZ(np.pi, 0))

        pyquil_backend = QAOAPyQuilQPUBackend(
            device_pyquil,
            qaoa_descriptor,
            n_shots=10,
            prepend_state=None,
            append_state=append_circuit,  
            init_hadamard=False,
            cvar_alpha=1,
        )

        assert set([
            "RZ",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "CZ",
            "RZ",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "CZ",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
            "RZ",
            "RX",
            "RZ",
            "RZ",
            "RX",
            "RZ",
            "RX",
            "RZ",
        ]) == set([
            instr.name
            for instr in pyquil_backend.qaoa_circuit(params)
            if type(instr) == quilbase.Gate
        ])

    @pytest.mark.qvm
    def test_circuit_prepend_state(self):
        """
        Checks correctness of circuit for the argument `prepend_state`.
        """

        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=3, compiler_timeout=3
        )

        # With prepend_state
        cost_hamil = Hamiltonian(
            [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
            [1, 1, 1],
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)

        prepend_circuit = Program().inst(RX(np.pi, 0), RY(np.pi / 2, 1), RZ(np.pi, 0))

        pyquil_backend = QAOAPyQuilQPUBackend(
            device_pyquil,
            qaoa_descriptor,
            n_shots=10,
            prepend_state=prepend_circuit,
            append_state=None,
            init_hadamard=False,
            cvar_alpha=1,
        )

        assert ["RX", "RY", "RZ", "RZ", "RZ", "RZ", "RZ", "CPHASE", "RX", "RX"] == [
            instr.name
            for instr in pyquil_backend.parametric_circuit
            if type(instr) == quilbase.Gate
        ]

        # Test if error is raised correctly
        prepend_circuit = Program().inst(RX(np.pi, 0), RY(np.pi / 2, 1), RZ(np.pi, 2))
        self.assertRaises(
            AssertionError,
            lambda: QAOAPyQuilQPUBackend(
                qaoa_descriptor=qaoa_descriptor,
                device=device_pyquil,
                prepend_state=prepend_circuit,
                append_state=None,
                init_hadamard=True,
                cvar_alpha=1,
                n_shots=1,
            ),
        )

    @pytest.mark.qvm
    def test_pyquil_vectorized_agreement(self):
        """
        Checks correctness of expectation values with vectorized backend, up to a tolerance of delta = std.dev.
        """

        # Without interaction terms
        device_pyquil = DevicePyquil(
            device_name="2q-qvm", as_qvm=True, execution_timeout=3, compiler_timeout=3
        )
        device_pyquil.quantum_computer.qam.random_seed = 1

        cost_hamil = Hamiltonian(
            [PauliOp("Z", (0,)), PauliOp("Z", (1,)), PauliOp("ZZ", (0, 1))],
            [1, 1, 1],
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=2)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=1)
        variate_params = create_qaoa_variational_params(
            qaoa_descriptor, "standard", "ramp"
        )
        args = [np.pi / 8, np.pi / 4]  # beta, gamma

        variate_params.update_from_raw(args)
        backend_obj_pyquil = QAOAPyQuilQPUBackend(
            qaoa_descriptor=qaoa_descriptor,
            device=device_pyquil,
            prepend_state=None,
            append_state=None,
            init_hadamard=True,
            cvar_alpha=1,
            n_shots=10,
        )
        expt_pyquil = backend_obj_pyquil.expectation(variate_params)

        variate_params.update_from_raw(args)
        backend_obj_vectorized = QAOAvectorizedBackendSimulator(
            qaoa_descriptor, prepend_state=None, append_state=None, init_hadamard=True
        )
        expt_vec, std_dev_vec = backend_obj_vectorized.expectation_w_uncertainty(
            variate_params
        )

        self.assertAlmostEqual(expt_vec, expt_pyquil, delta=std_dev_vec)

    @pytest.mark.qvm
    def test_remote_qubit_overflow(self):
        """
        If the user creates a circuit that is larger than the maximum circuit size
        that is supported by the QPU. An Exception should be raised with the
        appropriate error message alerting the user to the error.
        """

        shots = 100
        set_of_numbers = np.random.randint(1, 10, 6).tolist()
        qubo = NumberPartition(set_of_numbers).qubo
        mixer_hamil = X_mixer_hamiltonian(n_qubits=6)
        qaoa_descriptor = QAOADescriptor(qubo.hamiltonian, mixer_hamil, p=1)
        variate_params = create_qaoa_variational_params(
            qaoa_descriptor, "standard", "rand"
        )

        device_qvm = DevicePyquil(
            device_name="5q-qvm", as_qvm=True, execution_timeout=3, compiler_timeout=3
        )
        try:
            qvm_backend = QAOAPyQuilQPUBackend(
                device_qvm, qaoa_descriptor, shots, None, None, True, 1.0
            )
            qvm_backend.expectation(variate_params)
        except Exception as e:
            self.assertEqual(
                str(e),
                "There are lesser qubits on the device than the number of qubits required for the circuit.",
            )

    @pytest.mark.qvm
    def test_job_ids(self):
        """
        Test if correct job ids are generated and returned when running on qvm
        """

        # define problem
        problem = QUBO.random_instance(3)

        # initialize
        q = QAOA()

        # device
        device = create_device(location="qcs", name="3q-qvm")
        q.set_device(device)

        # classical optimizer only 3 iterations
        q.set_classical_optimizer(maxiter=3)

        # compile
        q.compile(problem)

        # run
        q.optimize()

        # check if we have job ids
        opt_id = q.result.optimized["job_id"]
        assert len(opt_id) == 36 and isinstance(
            opt_id, str
        ), f"QCS QVM: job id is not a string of length 36, but {opt_id}"

        inter_id = q.result.intermediate["job_id"]
        for id in inter_id:
            assert len(id) == 36 and isinstance(
                id, str
            ), f"QCS QVM: on intermediate job id is not a string of length 36, but {id}"


if __name__ == "__main__":
    unittest.main()
