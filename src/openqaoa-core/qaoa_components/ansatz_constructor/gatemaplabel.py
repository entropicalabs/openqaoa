from enum import Enum
from typing import Union


class GateMapType(Enum):

    MIXER = "MIXER"
    COST = "COST"
    FIXED = "FIXED"

    @classmethod
    def supported_types(cls):
        return list(map(lambda c: c.name, cls))


class GateMapLabel:
    """
    This object helps keeps track of labels associated with
    gates for their identification in the circuit.
    """

    def __init__(
        self,
        n_qubits: int = None,
        layer_number: int = None,
        application_sequence: int = None,
        gatemap_type: GateMapType = None,
    ):
        """
        Parameters
        ----------
        layer_number: `int`
            The label for the algorthmic layer
        application_sequence: `int`
            The label for the sequence of application
            for the gate
        gate_type: `GateMapType`
            Gate type for distinguishing gate between different QAOA blocks,
            and between parameterized or non-parameterized
        n_qubits: `int`
            Number of qubits in the gate
        """
        if (
            isinstance(layer_number, (int, type(None)))
            and isinstance(application_sequence, (int, type(None)))
            and isinstance(gatemap_type, (GateMapType, type(None)))
        ):
            self.layer = layer_number
            self.sequence = application_sequence
            self.type = gatemap_type
            self.n_qubits = n_qubits
        else:
            raise ValueError("Some or all of input types are incorrect")

    def __repr__(self):
        """
        String representation of the Gatemap label
        """
        representation = f"{self.n_qubits}Q_" if self.n_qubits is not None else ""
        representation += f"{self.type.value}" if self.type.value is not None else ""
        representation += f"_seq{self.sequence}" if self.sequence is not None else ""
        representation += f"_layer{self.layer}" if self.layer is not None else ""

        return representation

    def update_gatelabel(
        self,
        new_layer_number: int = None,
        new_application_sequence: int = None,
        new_gatemap_type: GateMapType = None,
    ) -> None:
        """
        Change the properties of the gatemap label to update
        the gate identity
        """
        if (
            new_layer_number is None
            and new_gatemap_type is None
            and new_application_sequence is None
        ):
            raise ValueError(
                "Pass atleast one updated attribute to update the gatemap label"
            )
        else:
            if isinstance(new_layer_number, int):
                self.layer = new_layer_number
            if isinstance(new_application_sequence, int):
                self.sequence = new_application_sequence
            if isinstance(new_gatemap_type, GateMapType):
                self.type = new_gatemap_type
