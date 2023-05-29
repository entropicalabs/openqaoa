from openqaoa_qiskit.backends import QAOAQiskitQPUBackend

import qb.core
from openqaoa.backends.basebackend import (
    QAOABaseBackendShotBased,
    QAOABaseBackendCloud,
    QAOABaseBackendParametric,
)
from openqaoa.qaoa_components import QAOADescriptor
from openqaoa.qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)
from .devices import DeviceQristal

class QAOAQristalQPUBackend(QAOAQiskitQPUBackend):
    """A QAOA backend for running quantum circuits on Qristal backends

    Parameters
    ----------
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        device: DeviceQristal,
        n_shots: int,
        prepend_state,
        append_state,
        init_hadamard: bool,
        initial_qubit_mapping=None,
        qiskit_optimization_level: int = 1,
        cvar_alpha: float = 1,
    ):
        super().__init__(
            qaoa_descriptor,
            device,
            n_shots,
            prepend_state,
            append_state,
            init_hadamard,
            initial_qubit_mapping,
            qiskit_optimization_level,
            cvar_alpha,
        )
        # Choose how many qubits to simulate
        self.device.sim.qn = self.qaoa_descriptor.n_qubits

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None):
        n_shots = self.n_shots if n_shots is None else n_shots

        circuit = self.circuit_to_qasm(params)
        self.device.sim.sn = n_shots

        self.device.sim.instring = f"""
        __qpu__ VOID MY QUANTUM CIRCUIT (qreg q)
        {
            {circuit}
        }
        """

        self.device.sim.run()
        return self.device.sim.out_raw[0][0]
