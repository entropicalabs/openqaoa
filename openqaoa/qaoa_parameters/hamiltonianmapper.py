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

from .pauligate import PauliGate, PauliGateFactory
from .operators import Hamiltonian
from typing import List


class HamiltonianMapper(object):

    def get_pauli_gates(hamil_obj: Hamiltonian, input_label: List = []) -> List[PauliGate]:
        """
        This method gets the pauli gates based on the input Hamiltonian into the 
        Mapper
        """

        assert type(input_label) is list, 'input_label must be of type list'

        return PauliGateFactory.convert_hamiltonian_to_pauli_gates(hamil_obj, input_label)

    def repeat_paulis(hamil_obj: Hamiltonian, tag: str, n_repetitions: int) -> List[List[PauliGate]]:

        output_gate_list = []

        for each_repetition in range(n_repetitions):
            output_gate_list.append(HamiltonianMapper.get_pauli_gates(
                hamil_obj, [tag.lower(), each_repetition]))

        return output_gate_list
