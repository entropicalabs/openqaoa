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

"""
construct Hamiltonian operators here in the standard PauliFormulation
"""
from collections import Counter
import numpy as np
from typing import List, Union, Tuple

Identity = np.array(([1, 0], [0, 1]), dtype=complex)
PauliX = np.array(([0, 1], [1, 0]), dtype=complex)
PauliY = np.array(([0, -1j], [1j, 0]), dtype=complex)
PauliZ = np.array(([1, 0], [0, -1]), dtype=complex)

PAULIS_SET = set('XYZI')

PAULI_MULT_RULES = {'XX': 'I', 'YY': 'I', 'ZZ': 'I',
                    'XY': 'Z', 'YX': 'Z', 'XZ': 'Y',
                    'ZX': 'Y', 'YZ': 'X', 'ZY': 'X'}
PAULI_MULT_RULES.update({f'I{op}': op for op in PAULIS_SET})
PAULI_MULT_RULES.update({f'{op}I': op for op in PAULIS_SET})

PAULI_MAPPERS = {'X': PauliX, 'Y': PauliY, 'Z': PauliZ, 'I': Identity,
                 'iX': 1j*PauliX, 'iY': 1j*PauliY, 'iZ': 1j*PauliZ,
                 '-iX': -1j*PauliX, '-iY': -1j*PauliY, '-iZ': -1j*PauliZ}

PAULI_PHASE_MAPPERS = {'XX': 1, 'YY': 1, 'ZZ': 1,
                       'XY': 1j, 'ZX': 1j, 'YZ': 1j,
                       'ZY': -1j, 'XZ': -1j, 'YX': -1j,
                       'X': 1, 'Y': 1, 'Z': 1, 'I': 1}
PAULI_PHASE_MAPPERS.update({f'I{op}': 1 for op in PAULIS_SET})
PAULI_PHASE_MAPPERS.update({f'{op}I': 1 for op in PAULIS_SET})


class PauliOp:
    """
    Pauli operator class to initiliase and handle Pauli operators
    """

    def __init__(self,
                 pauli_str: str,
                 qubit_indices: Tuple[int]):
        """
        Initialise the Pauli Operator

        Parameters
        ----------
        pauli_str: `str`
                The Pauli operator basis string
        qubit_indices: `int`
                The qubits on which the Pauli operates

        Attributes
        ----------
        pauli_str: `str`
                The Pauli operator basis string
        qubit_indices: `Tuple[int]`
                The qubits on which the Pauli operates
        phase: `complex`
                The phase of the Pauli operator
        """
        assert len(pauli_str) == len(
            qubit_indices), "Each Pauli operator must have a unique qubit index"
        # simplify if needed
        pauli_str, qubit_indices, phase = self._simplify(
            pauli_str, qubit_indices)
        # sort if needed
        self.qubit_indices, self.pauli_str = self._sort_pauli_op(
            qubit_indices, pauli_str)
        self.phase = phase

    @staticmethod
    def _sort_pauli_op(qubit_indices: Tuple[int], pauli_str: str):
        """
        Sort the Pauli Operator in the 
        increasing order of qubit indices.

        Example:
                Example:
                PauliOp('YZX',(3,1,2)) -> PauliOp('ZXY',(1,2,3)) with appropriate phase

        Parameters
        ----------
        qubit_indices: `Tuple[int]`
                The qubit indices of the Pauli Operator
        pauli_str: `str`
                The Pauli Operator basis string

        Returns
        -------
        sorted_qubit_indices: `Tuple[int]`
                The sorted qubit indices in increasing order
        sorted_pauli_str: `str`
                The sorted Pauli Operator basis string
        """
        sorted_pauli_str = ''
        sorted_qubit_indices = []
        for index, string in sorted(zip(qubit_indices, pauli_str)):
            if string not in PAULIS_SET:
                raise ValueError(
                    f"{string} is not a valid Pauli. Please choose from the set {PAULIS_SET}")
            sorted_qubit_indices.append(index)
            sorted_pauli_str += string

        sorted_qubit_indices = tuple(sorted_qubit_indices)
        return sorted_qubit_indices, sorted_pauli_str

    @staticmethod
    def _simplify(pauli_str: str, qubit_indices: Tuple[int]):
        """
        Simplify the definition of Pauli Operator

        Example:
                PauliOp('XZX',(3,2,2)) -> PauliOp('XY',(3,2)) with appropriate phase

        Parameters
        ----------
        qubit_indices: `Tuple[int]`
                The qubit indices of the Pauli Operator
        pauli_str: `str`
                The Pauli Operator basis string

        Returns
        -------
        new_pauli_str: `str`
                The updated Pauli Operator basis string
        new_qubit_indices: `Tuple[int]`
                The updated qubit indices in increasing order
        """
        new_phase = 1
        qubit_reps = Counter(qubit_indices)
        if len(qubit_indices) == len(qubit_reps.values()):
            # no repetitions, do nothing
            new_pauli_str = pauli_str
            new_qubit_indices = qubit_indices
        else:
            # simplify the operator
            repeating_indices = [index for index,
                                 rep in qubit_reps.items() if rep > 1]
            paulis_list_to_contract = []
            for index in repeating_indices:
                paulis_list_to_contract.append([pauli_str[i] for i in range(
                    len(qubit_indices)) if qubit_indices[i] == index])
            for paulis in paulis_list_to_contract:
                i = 0
                while len(paulis) > 1:
                    pauli_mult = paulis[i] + paulis[i+1]
                    paulis[0] = PAULI_MULT_RULES[pauli_mult]
                    new_phase *= PAULI_PHASE_MAPPERS[pauli_mult]
                    paulis.pop(i+1)

            repeating_pauli_str = ''.join(
                pauli[0] for pauli in paulis_list_to_contract)
            non_repeating_indices = [index for index,
                                     rep in qubit_reps.items() if rep == 1]
            non_repeating_paulis = ''.join(
                pauli_str[list(qubit_indices).index(idx)] for idx in non_repeating_indices)

            new_pauli_str = non_repeating_paulis + repeating_pauli_str
            new_qubit_indices = tuple(
                non_repeating_indices + repeating_indices)

        return new_pauli_str, new_qubit_indices, new_phase

    @property
    def _is_trivial(self) -> bool:
        """
        Return ``True`` if the PauliOp only contains
        Identity ``I`` terms
        """
        return self.pauli_str == 'I'*len(self.qubit_indices)

    @property
    def matrix(self):
        """
        Matrix representation of the Pauli Operator
        """
        mat = PAULI_MAPPERS[self.pauli_str[0]]
        for pauli in self.pauli_str[1:]:
            mat = np.kron(mat, PAULI_MAPPERS[pauli])
        return mat

    def __len__(self):
        """
        Length of the Pauli term
        """
        return len(self.qubit_indices)

    def __eq__(self, other_pauli_op):
        """
        Check whether two pauli_operators are equivalent
        """
        condition1 = True if self.qubit_indices == other_pauli_op.qubit_indices else False
        condition2 = True if self.pauli_str == other_pauli_op.pauli_str else False

        return condition1 and condition2

    def __copy__(self):
        """
        Create a new `PauliOp` by copying the current one
        """
        copied_pauli_op = self.__class__.__new__(self.__class__)
        for attribute, value in vars(self).items():
            setattr(copied_pauli_op, attribute, value)
        return copied_pauli_op

    def __str__(self):
        """
        String representation of the Pauli Operator
        """
        term_str = ''.join(pauli_base + '_' + str({index})
                           for pauli_base, index in zip(self.pauli_str, self.qubit_indices))
        return term_str

    def __repr__(self):
        """
        Repr of the Pauli Operator
        """
        term_repr = f'PauliOp({self.pauli_str},{self.qubit_indices})'
        return term_repr

    def __mul__(self, other_pauli_op):
        """
        Multiply two Pauli Operators

        Parameters
        ----------
        other_pauli_op: `PauliOp`
                The other Pauli Operator to be multiplied

        Return
        ------
        new_pauli_op: `PauliOp`
                The resulting Pauli Operator after the multiplication
        """
        assert isinstance(
            other_pauli_op, PauliOp), "Please specify a Pauli Operator"

        copied_current_pauli_op = self.__copy__()
        copied_current_pauli_op.__matmul__(other_pauli_op)

        return copied_current_pauli_op

    def __matmul__(self, other_pauli_op):
        """
        In-place Multiplication of Pauli Operators
        Contract `other_pauli_op` into `self`

        Parameters
        ----------
        other_pauli_op: `PauliOp`
                The Pauli Operator to be multiplied
        """
        assert isinstance(
            other_pauli_op, PauliOp), "Please specify a Pauli Operator"

        n_qubits = max(max(self.qubit_indices), max(
            other_pauli_op.qubit_indices))+1
        self_pauli_str_list = list(self.pauli_str)
        other_pauli_str_list = list(other_pauli_op.pauli_str)
        for i in range(n_qubits):
            if i not in self.qubit_indices:
                self_pauli_str_list.insert(i, 'I')
            if i not in other_pauli_op.qubit_indices:
                other_pauli_str_list.insert(i, 'I')

        new_full_operator = ''
        new_phase = 1
        for idx in range(n_qubits):
            pauli_composition = self_pauli_str_list[idx] + \
                other_pauli_str_list[idx]
            mult_phase = PAULI_PHASE_MAPPERS[pauli_composition]
            mult_pauli = PAULI_MULT_RULES[pauli_composition]

            new_full_operator += mult_pauli
            new_phase *= mult_phase

        self.qubit_indices = tuple([idx for idx in range(
            n_qubits) if new_full_operator[idx] != 'I'])
        self.pauli_str = new_full_operator.replace('I', '')
        self.phase = new_phase

        return self

    @classmethod
    def X(cls, qubit_idx):
        return cls('X', (qubit_idx,))

    @classmethod
    def Y(cls, qubit_idx):
        return cls('Y', (qubit_idx,))

    @classmethod
    def Z(cls, qubit_idx):
        return cls('Z', (qubit_idx,))

    @classmethod
    def I(cls, qubit_idx):
        return cls('I', (qubit_idx,))


class Hamiltonian:
    """
    General Quantum Hamiltonian class.
    """

    def __init__(self,
                 pauli_terms: List[PauliOp],
                 coeffs: List[Union[complex, int, float]],
                 constant: float,
                 divide_into_singles_and_pairs: bool = True):
        """
        Parameters
        ----------
        terms: `List[PauliOp]
        coeffs: `List[Union[complex,float,int]]`
        constant: `float`
        divide_into_singles_and_pairs: ``bool``
                Whether to divide the Hamiltonian into singles and pairs
        """
        assert len(pauli_terms) == len(coeffs), \
            "Number of Pauli terms in Hamiltonian should be same as number of coefficients"
        
        physical_qureg = []
        for pauli_term in pauli_terms:
            if isinstance(pauli_term, PauliOp):
                physical_qureg.extend(pauli_term.qubit_indices)
            else:
                raise TypeError(f"Pauli terms should be of type PauliOp and not {type(term)}")
        physical_qureg = list(set(physical_qureg))
        self.n_qubits = len(physical_qureg)

        need_remapping = False
        if physical_qureg != self.qureg:
            print(f'Qubits in the specified Hamiltonian are remapped to {self.qureg}.'
                   'Please specify the physical quantum register as an arugment in the Backend')
            need_remapping = True
            qubit_mapper = dict(zip(physical_qureg, self.qureg))

        self.terms = []
        self.coeffs = []
        self.constant = constant
        for term,coeff in zip(pauli_terms,coeffs):
            if term._is_trivial:
                self.constant += coeff
            else:
                if need_remapping:
                    new_indices = tuple(qubit_mapper[i] for i in term.qubit_indices)
                    pauli_str = term.pauli_str
                    self.terms.append(PauliOp(pauli_str, new_indices))
                else:
                    self.terms.append(term)
                # update the coefficients with phase from Pauli Operators
                self.coeffs.append(coeff*term.phase)

        if divide_into_singles_and_pairs:
            self._divide_into_singles_pairs()

    @property
    def qureg(self):
        """
        List of qubits from 0 to n-1 in Hamiltonian
        """
        return list(range(self.n_qubits))

    def _divide_into_singles_pairs(self):

        self.qubits_pairs = []
        self.qubits_singles = []
        self.single_qubit_coeffs = []
        self.pair_qubit_coeffs = []

        for term, coeff in zip(self.terms, self.coeffs):
            if len(term) == 1:
                self.qubits_singles.append(term)
                self.single_qubit_coeffs.append(coeff)
            elif len(term) == 2:
                self.qubits_pairs.append(term)
                self.pair_qubit_coeffs.append(coeff)
            elif set(term.pauli_str) == {'I'}:
                self.constant += coeff
            else:
                raise NotImplementedError(
                    "Hamiltonian only supports Linear and Quadratic terms")

    def __str__(self):
        """
        Return a string representation of the Hamiltonian
        """
        hamil_str = ''
        for coeff, term in zip(self.coeffs, self.terms):
            hamil_str += str(np.round(coeff, 3)) + '*' + term.__str__() + ' + '
        hamil_str += str(self.constant)
        return hamil_str

    def __len__(self):
        """
        Return the number of terms in the Hamiltonian
        """
        return len(self.terms)

    @property
    def expression(self):
        try:
            from sympy import Symbol
        except ImportError:
            raise ImportError(
                "Sympy is not installed. Pip install sympy to use this method")
        hamiltonian_expression = Symbol(str(self.constant))
        for term, coeff in zip(self.terms, self.coeffs):
            hamiltonian_expression += Symbol(str(coeff)+term.__str__())
        return hamiltonian_expression

    def __add__(self, other_hamiltonian):
        """
        Add two Hamiltonians in place to update `self`

        Parameters
        ----------
        other_hamiltonian: `Hamiltonian`
                The other Hamiltonian to be added to `self`
        """
        assert isinstance(other_hamiltonian, Hamiltonian)
        for other_term, other_coeff in zip(other_hamiltonian.terms, other_hamiltonian.coeffs):
            if other_term in self.terms:
                self.coeffs[self.terms.index(other_term)] += other_coeff
            else:
                self.terms.append(other_term)
                self.coeffs.append(other_coeff)

    @property
    def hamiltonian_squared(self):
        """
        @classmethod
        Compute the squared of the Hamiltonian. Necessary in computing 
        the error in expectation values
        """
        hamil_sq_terms = []
        hamil_sq_coeffs = []
        hamil_sq_constant = self.constant**2

        for i, term1 in enumerate(self.terms):
            for j, term2 in enumerate(self.terms):
                new_term = term1*term2

                # If multiplication yields a constant add it to hamiltonian constant
                if new_term.pauli_str == '':
                    hamil_sq_constant += self.coeffs[i]*self.coeffs[j]

                # If it yields non-trivial term add it to list of terms
                else:
                    new_phase = new_term.phase
                    new_coeff = self.coeffs[i]*self.coeffs[j]
                    hamil_sq_terms.append(new_term)
                    hamil_sq_coeffs.append(new_coeff)

        hamil_sq_terms.extend(self.terms)
        hamil_sq_coeffs.extend(
            [2*self.constant*coeff for coeff in self.coeffs])

        hamil_squared = Hamiltonian(hamil_sq_terms, hamil_sq_coeffs, hamil_sq_constant,
                                    divide_into_singles_and_pairs=False)
        return hamil_squared

    @classmethod
    def classical_hamiltonian(cls,
                              terms: List[Union[Tuple, List]],
                              coeffs: List[Union[float, int]],
                              constant: float):

        for coeff in coeffs:
            
            if not isinstance(coeff, int) and not isinstance(coeff, float):
                raise ValueError(
                    "Classical Hamiltonians only support Integer or Float coefficients")

        pauli_ops = []
        for term in terms:
            if len(term) == 2:
                pauli_ops.append(PauliOp('ZZ', term))
            elif len(term) == 1:
                pauli_ops.append(PauliOp('Z', term))
            else:
                raise ValueError(
                    "Hamiltonian only supports Linear and Quadratic terms")

        return cls(pauli_ops, coeffs, constant)
