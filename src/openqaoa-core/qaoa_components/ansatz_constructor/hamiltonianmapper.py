from typing import List
from copy import deepcopy

from .gatemap import RotationGateMapFactory, GateMap
from .gatemaplabel import GateMapType
from .operators import Hamiltonian


class HamiltonianMapper(object):
    def generate_gate_maps(
        hamil_obj: Hamiltonian, gatemap_type: GateMapType
    ) -> List[GateMap]:
        """
        This method gets the rotation gates based on the input Hamiltonian into the Mapper

        Parameters
        ----------
        hamil_obj : `Hamiltonian`
            The Hamiltonian object to construct the gates from
        input_label : `GateMapType`
            Input label defining the type of gate

        Returns
        -------
        `list[GateMap]`
            List of RotationGateMap objects defining part of the circuit
        """
        assert isinstance(
            gatemap_type, GateMapType
        ), f"gatemap_type must be of supported types: {GateMapType.supported_types}"
        return RotationGateMapFactory.rotationgatemap_list_from_hamiltonian(
            hamil_obj, gatemap_type
        )

    def repeat_gate_maps(
        gatemap_list: List[GateMap], n_layers: int
    ) -> List[List[GateMap]]:
        """
        Repeat the gates for n_layers based on the input gatelist

        Parameters
        ----------
        gatemap_list : `List[GateMap]`
            Repeat the gates from the gatemap_list
        n_layers: `int`
            The number of times the layer of gates have to be repeated.
        """
        output_gate_list = []
        for each_layer in range(n_layers):
            output_gate_list.append(
                RotationGateMapFactory.gatemaps_layer_relabel(gatemap_list, each_layer)
            )

        return output_gate_list
