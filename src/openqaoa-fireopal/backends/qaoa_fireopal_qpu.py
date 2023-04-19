import fireopal

from openqaoa_qiskit.backends import QAOAQiskitQPUBackend
from openqaoa.utilities import flip_counts
from openqaoa.qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)


class QAOAFireOpalQPUBackend(QAOAQiskitQPUBackend):
    """

    Parameters
    ----------
    device: `DeviceQiskit`
            An object of the class ``DeviceQiskit`` which contains the credentials
            for accessing the QPU via cloud and the name of the device.
    qaoa_descriptor: `QAOADescriptor`
            An object of the class ``QAOADescriptor`` which contains information on
            circuit construction and depth of the circuit.
    n_shots: `int`
            The number of shots to be taken for each circuit.
    prepend_state: `QuantumCircuit`
            The state prepended to the circuit.
    append_state: `QuantumCircuit`
            The state appended to the circuit.
    init_hadamard: `bool`
            Whether to apply a Hadamard gate to the beginning of the
            QAOA part of the circuit.
    cvar_alpha: `float`
            The value of alpha for the CVaR method.
    """

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        """
        Execute the QAOA ciruit with FireOpal execute function
        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
                An object of the class ``QAOAVariationalBaseParams`` which
                contains the parameters of the QAOA circuit.
        n_shots: `int`
                The number of shots to be taken for each circuit.

        Returns
        -------
        counts: `dict`
                The counts of the circuit.
        """
        if n_shots is None:
            n_shots = self.n_shots
        # circuit = self.qaoa_circuit(params)
        qasm_str = self.circuit_to_qasm(params)
        fireopal_results = fireopal.execute(
            circuits=[qasm_str],
            shot_count=n_shots,
            backend_name=self.device.device_name,
            credentials=self.device.credentials,
        )
        bitstring_results = fireopal_results["results"]
        counts = flip_counts(bitstring_results[0])
        self.measurement_outcomes = counts
        return counts
