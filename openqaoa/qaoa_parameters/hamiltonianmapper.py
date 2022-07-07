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

from .gatemap import RotationGateMap, RotationGateMapFactory
from .operators import Hamiltonian
from typing import List


class HamiltonianMapper(object):

    def get_gate_maps(hamil_obj: Hamiltonian, input_label: List = []) -> List[RotationGateMap]:
        """
        This method gets the pauli gates based on the input Hamiltonian into the Mapper

        Parameters
        ----------
        hamil_obj : `Hamiltonian`
            The Hamiltonian object to construct the gates from
        input_label : List[str]
            Input label defining the type of gate

        Return
        ------
        List[RotationGateMap]
            List of RotationGateMap objects defining part of the circuit
        """

        assert type(input_label) is list, 'input_label must be of type list'

        return RotationGateMapFactory.convert_hamiltonian_to_gate_maps(hamil_obj, input_label)

    def repeat_gate_maps(hamil_obj: Hamiltonian, tag: str, n_repetitions: int) -> List[List[RotationGateMap]]:
        """
        Repeat the gates for n_repetitions layers based on the input Hamiltonian into the Mapper.

        Parameters
        ----------
        hamil_obj : `Hamiltonian`
            The Hamiltonian object to construct the gates from
        tag : str
            The tag to be used for the repeated gates
        """
        output_gate_list = []

        for each_repetition in range(n_repetitions):
            output_gate_list.append(HamiltonianMapper.get_gate_maps(
                hamil_obj, [tag.lower(), each_repetition]))

        return output_gate_list
