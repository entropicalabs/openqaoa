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

from collections import Counter
import numpy as np
from copy import deepcopy
from pyquil import Program, gates, quilbase
from pyquil.quil import Pragma

from ...basebackend import QAOABaseBackendShotBased, QAOABaseBackendCloud, QAOABaseBackendParametric
from ...qaoa_parameters.baseparams import QAOACircuitParams, QAOAVariationalBaseParams
from ...devices import DevicePyquil
from ...qaoa_parameters.gatemap import RZZGateMap, SWAPGateMap

def check_edge_connectivity(executable: Program, device: DevicePyquil):

    '''
    Check that the program does not contain 2-qubit terms that is not present in the QPU's topology (to prevent quilc from crashing).
    
    Parameters
    ----------
    executable: `Program`
        pyQuil executable program.
    device: `DevicePyquil`
        An object of the class ``DevicePyquil`` which contains information on pyQuil's `QuantumComputer` object, used to extract the selected QPU's topology.
    
    Returns
    -------
        None
    """
    '''

    qpu_graph = device.quantum_computer.qubit_topology()
    
    instrs = [instr for instr in executable if type(instr) == quilbase.Gate]
    pair_instrs = [list(instr.get_qubits()) for instr in instrs if len(instr.get_qubits()) == 2]

    for term in pair_instrs:
        if len(term) == 2:

            assert term in qpu_graph.edges(), f"Term {term} is not an edge on the QPU graph of {device.device_name}."
    

class QAOAPyQuilQPUBackend(QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased):
    """
    A QAOA backend object for real Rigetti QPUs

    Parameters
    ----------
    device: `DevicePyquil`
        The device object to access pyquil devices with credentials.
    circuit_params: `QAOACircuitParams`
        An object of the class ``QAOACircuitParams`` which contains information on 
        circuit construction and depth of the circuit.
    n_shots: `int`
        The number of shots to be taken for each circuit.
    prepend_state: `pyquil.Program`
        The state prepended to the circuit.
    append_state: `pyquil.Program`
        The state appended to the circuit.
    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the QAOA part of the circuit.
    cvar_alpha: `float`
        Conditional Value-at-Risk (CVaR) – a measure that takes into account only the tail of the
        probability distribution arising from the circut's count dictionary. Must be between 0 and 1. Check
        https://arxiv.org/abs/1907.04769 for further details.
    active_reset: 
        Whether to use the pyQuil's active reset scheme to reset qubits between shots. 
    rewiring:
        Rewiring scheme to be used for Pyquil. 
        Either PRAGMA INITIAL_REWIRING "NAIVE" or PRAGMA INITIAL_REWIRING "PARTIAL". 
        If None, pyquil defaults according to:
        NAIVE: The qubits used in all instructions in the program satisfy the topological constraints of the device.
        PARTIAL: Otherwise.
    unfence:
        Whether to allow 2Q gates to be executed in parallel by removing global fence statements (which are by 
        default enabled). Only works when `device` has `as_qvm = False`.
    trivial_parallelization:
        Whether to re-order pyQuil program according to 'trivial' commutation rules, i.e. re-arranging order of
        gates that act on independent sets of qubits.
    
    """

    def __init__(self,
                 device: DevicePyquil,
                 circuit_params: QAOACircuitParams,
                 n_shots: int,
                 prepend_state: Program,
                 append_state: Program,
                 init_hadamard: bool,
                 cvar_alpha: float,
                 active_reset: bool = False,
                 rewiring: str = '',
                 qubit_layout: list = [],
                 unfence: bool = False,
                 trivial_parallelization: bool = True,
                 ):

        QAOABaseBackendShotBased.__init__(self,
                                          circuit_params,
                                          n_shots,
                                          prepend_state,
                                          append_state,
                                          init_hadamard,
                                          cvar_alpha)
        QAOABaseBackendCloud.__init__(self, device)

        self.active_reset = active_reset
        self.rewiring = rewiring
        self.unfence = unfence
        self.trivial_parallelization = trivial_parallelization
        self.qureg = self.circuit_params.qureg

        # self.qureg_placeholders = QubitPlaceholder.register(self.n_qubits)
        self.qubit_layout = self.qureg if qubit_layout == [] else qubit_layout
        self.qubit_mapping = dict(zip(self.qureg, self.qubit_layout))
        
        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.get_qubits()), "Cannot attach a bigger circuit " \
                                                                "to the QAOA routine"
        self.parametric_circuit = self.parametric_qaoa_circuit
        
        # 'trivial' parallelization pass
        if trivial_parallelization:
            self.parametric_circuit = parallelize_pyquil_prog(self.parametric_circuit)
            
        native_prog = self.device.quantum_computer.compiler.quil_to_native_quil(
            self.parametric_circuit)
        
        # Unfencing pass
        if self.unfence == True:
            if self.device.as_qvm == False:
                native_prog += unfence_2q_gates(self.device.quantum_computer.compiler.get_calibration_program())
            else:
                print('Unfencing pass ignored : self.device.as_qvm was True')
        
        self.prog_exe = self.device.quantum_computer.compiler.native_quil_to_executable(
            native_prog)
        
        # Check program connectivity against QPU connectivity
        # TODO: reconcile with PRAGMA PRESERVE

        # check_edge_connectivity(self.prog_exe, device)


    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> Program:
        """
        Injects angles into created executable parametric circuit.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        `pyquil.Program`
            A pyquil.Program (executable) object.
        """
        angles_list = np.array(self.obtain_angles_for_pauli_list(
            self.abstract_circuit, params), dtype=float)
        angle_declarations = list(self.prog_exe.declarations.keys())
        angle_declarations.remove('ro')
        for i, param_name in enumerate(angle_declarations):
            self.prog_exe.write_memory(
                region_name=param_name, value=angles_list[i])

        return self.prog_exe

    @property
    def parametric_qaoa_circuit(self) -> Program:
        """
        Creates a parametric QAOA circuit (pyquil.Program object), given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit. 

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
        
        Returns
        -------
        `pyquil.Program`
            A pyquil.Program object.
        """
        if self.active_reset:
            parametric_circuit = Program(gates.RESET())
        else:
            parametric_circuit = Program()
        
        if self.rewiring != None:
            if self.rewiring in ['PRAGMA INITIAL_REWIRING "NAIVE"', 'PRAGMA INITIAL_REWIRING "PARTIAL"', '']:
                parametric_circuit += Program(self.rewiring)
            else:
                raise ValueError('Rewiring command not recognized. Please use ''PRAGMA INITIAL_REWIRING "NAIVE"'' or ''PRAGMA INITIAL_REWIRING "PARTIAL"''')
        
        # declare the read-out register
        ro = parametric_circuit.declare('ro', 'BIT', self.n_qubits)

        if self.prepend_state:
            parametric_circuit += self.prepend_state
            
        # Initial state is all |+>
        if self.init_hadamard:
            for i in self.qureg:
                parametric_circuit += gates.RZ(np.pi, self.qubit_mapping[i]) 
                parametric_circuit += gates.RX(np.pi/2, self.qubit_mapping[i]) 
                parametric_circuit += gates.RZ(np.pi/2, self.qubit_mapping[i]) 
                parametric_circuit += gates.RX(-np.pi/2, self.qubit_mapping[i]) 

        # create a list of gates in order of application on quantum circuit
        for each_gate in self.abstract_circuit:
            gate_label = ''.join(str(label) for label in each_gate.pauli_label)
            angle_param = parametric_circuit.declare(
                f'pauli{gate_label}', 'REAL', 1)
            each_gate.rotation_angle = angle_param
            if isinstance(each_gate, RZZGateMap) or isinstance(each_gate, SWAPGateMap):
                decomposition = each_gate.decomposition('standard2')
            else:
                decomposition = each_gate.decomposition('standard')
                
            # using the list above, construct the circuit
            for each_tuple in decomposition:
                gate = each_tuple[0]()
                qubits, rotation_angle = each_tuple[1]
                if isinstance(qubits,list):
                    new_qubits = [self.qubit_mapping[qubit] for qubit in qubits]
                else:
                    new_qubits = self.qubit_mapping[qubits]
                parametric_circuit = gate.apply_pyquil_gate(new_qubits,rotation_angle,parametric_circuit)

        if self.append_state:
            parametric_circuit += self.append_state
            
        # Measurement instructions
        for i, qubit in enumerate(self.qureg):
            parametric_circuit += gates.MEASURE(self.qubit_mapping[qubit], ro[i])

        parametric_circuit.wrap_in_numshots_loop(self.n_shots)

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams) -> dict:
        """
        Execute the circuit and obtain the counts.

        Parameters
        ----------
        params: QAOAVariationalBaseParams
            The QAOA parameters - an object of one of the parameter classes, containing 
            variable parameters.

        Returns
        -------
        counts : dictionary
            A dictionary with the bitstring as the key and the number of counts as its value.
        """
        executable_program = self.qaoa_circuit(params)

        result = self.device.quantum_computer.run(executable_program)

        # TODO: check the endian (big or little) ordering of measurement outcomes
        meas_list = [''.join(str(bit) for bit in bitstring)
                     for bitstring in result.readout_data['ro']]
        
        # Expose counts
        counts = Counter(list(meas_list))
        self.measurement_outcomes = counts
        return counts

    def circuit_to_qasm(self, params: QAOAVariationalBaseParams) -> str:
        """
        A method to convert the pyQuil program to a OpenQASM string.
        """
        raise NotImplementedError()
        # qasm_program = self.device.quantum_computer.compiler.quil_to_qasm(self.qaoa_circuit(params))
        # return qasm_program

    def reset_circuit(self):
        """
        Reset self.program after performing a computation. Also handle active reset and rewirings.
        """
        pass

    
##### PyQuil specific parallelization functions ######################### 
    
def unfence_2q_gates(calibration_program: Program) -> Program:
    
    """
    Remove the global fence statements from all 2Q gates, allowing them to be executed in parallel.

    >>> calibration_program = device.get_calibration_program()
    >>> unfenced_cal_program = unfence_2q_gates(calibration_program)
    >>> program = Program()
    >>> program += unfenced_cal_program

    :param calibration_program: The quilT calibration program.
    :returns: A modified copy of the quilT calibration program.
    """
    
    modified_calibration_program = calibration_program.copy_everything_except_instructions()
    unfenced_calibrations = []
    for calibration in calibration_program.calibrations:
        if isinstance(calibration, DefCalibration):
            if calibration.name in {"CPHASE", "CZ", "XY"}:
                updated_instrs = []
                for instr in calibration.instrs:
                    if isinstance(instr, FenceAll):  # replace FenceAll
                        updated_instrs.append(Fence(calibration.qubits))
                    else:
                        updated_instrs.append(instr)
                unfenced_calibrations.append(
                    DefCalibration(calibration.name, calibration.parameters, calibration.qubits, updated_instrs)
                )
            else:
                unfenced_calibrations.append(calibration)
        else:
            unfenced_calibrations.append(calibration)

    modified_calibration_program._calibrations = unfenced_calibrations
    return modified_calibration_program

def parallelize_sequence(gate_sequence):
    
    '''
    Separates and reorders `gate_sequence` into blocks where gates in each block can be simultaneously 
    applied, without changing the circuit structure. That is, re-orders the pyQuil program according 
    to 'trivial' commutation rules, i.e. re-arranging order of gates that act on independent sets of qubits.
    Also returns the original position of the gates in the parallelized blocks as `ordering_block_lst`.
    
    Warning : This pass incurs some computation time.
    
    Parameters
    ----------
    gate_sequence: list
        a list of singles/tuples representing physical indices on which 1Q/2Q gates are acting on.
    
    Returns
    -------
    remapped_block_lst : 
        Blocks where gates in each block can be simultaneously applied, without changing the circuit structure.
    
    ordering_block_lst : 
        Original position of the gates in the parallelized blocks.
    '''
    
    for gate in gate_sequence:
        assert len(gate) <= 2, f"Error in specified gate_sequence: length of term {gate} is {len(gate)} > 2"
    
    # Re-map gate sequence so that indices range from 0 to n (so that binary_marker is indexed correctly)
    zeroed_terms = list(range(len(set([item for sublist in gate_sequence for item in sublist]))))
    original_terms = list(set([item for sublist in gate_sequence for item in sublist]))
    zeroed_mapping = dict((original_terms[i], i) for i in range(len(zeroed_terms)))
    rezeroed_gate_sequence = [[zeroed_mapping[gate_ind] for gate_ind in gate] for gate in gate_sequence]

    # Original ordering of the gates; Will be modified according to this later.
    original_order = [[i] for i in list(range(len(gate_sequence) + 1))]

    n_qubits = len(original_terms)
    parallelizable_block_lst, ordering_block_lst = [], []

    while len(rezeroed_gate_sequence) != 0:
        
        binary_marker = np.zeros(n_qubits) # used to mark the qubit index whenever a non-parallel gate is encountered

        commuting_block = [rezeroed_gate_sequence[0]]
        ordering_block = [original_order[0]]

        rezeroed_gate_sequence.pop(0)
        original_order.pop(0)

        n_popped = 0
        ref_original_order = deepcopy(original_order)
        for i, gate in enumerate(deepcopy(rezeroed_gate_sequence)):

            if np.array_equal(binary_marker, np.ones(n_qubits)) == True:
                break

            # If gate commutes with everything in commuting_block, add into commuting_block.
            # else, if it does not commute with one of the gates, update binary marker for indices at the gate's qubits.
            encountered_noncommuting = False

            for commuting_gate in commuting_block:
                if len(set(gate).intersection(set(commuting_gate))) != 0 :
                    encountered_noncommuting = True
                    break

            # If this gate is not blocked by gates before it, and does not share qubits with `commuting_gate`, 
            # add this to the block, and pop() it from `rezeroed_gate_sequence`. (same for original_order)
            if encountered_noncommuting == False and all([binary_marker[gate_ind] == 0 for gate_ind in gate]):

                commuting_block.append(gate)
                rezeroed_gate_sequence.pop(i-n_popped)

                ordering_block.append(ref_original_order[i])
                original_order.pop(i-n_popped)

                n_popped += 1
                
            for commuting_gate in commuting_block:
                for index in gate:
                    binary_marker[index] = 1

        parallelizable_block_lst.append(commuting_block)
        ordering_block_lst.append(ordering_block)
        
    # Map the re-zeroed gate sequence back to the original sequence
    remapped_block_lst = []
    for block in parallelizable_block_lst:
        remapped_block_lst.append([[original_terms[term_ind] for term_ind in term] for term in block])
        
    return remapped_block_lst, ordering_block_lst

def parallelize_pyquil_prog(prog):
    
    '''
    Uses `ordering_block_lst` output of `parallelize_sequence` to re-order a pyQuil program.
    
    Parameters
    ----------
    prog: Program
        pyQuil Program to be operated on.
    
    Returns
    -------
    final_prog : 
        pyQuil Program with order of gates modified for parallelization.
        
    '''
    
    gate_list, gate_indices_list = [], []
    measure_list = []
    initial_decl_list = []

    for instruction in prog:
        if str(instruction)[:2] in ['RX', 'RY', 'RZ', 'CP', 'CZ', 'XY']:
            gate_list.append(deepcopy(instruction))
            gate_indices_list.append([qubits.index for qubits in instruction.qubits])

        elif str(instruction)[:2] in ['ME']:
            measure_list.append(deepcopy(instruction))
        else:
            initial_decl_list.append(deepcopy(instruction))

    # re-order gate_list based on parallelized blocks
    remapped_block_lst, ordering_block_lst = parallelize_sequence(gate_indices_list)
    
    '''
    # Calculate single/two-qubit layer depths
    depth_1q, depth_2q = 0, 0
    for blk in remapped_block_lst:
        if max([len(term) for term in blk]) == 2:
            depth_2q += 1
        else:
            depth_1q += 1
    print(f"Parallelized... final 1q-depth = {depth_1q}, 2q-depth = {depth_2q}, total = {len(remapped_block_lst)}")
    '''
    
    re_ordered_gate_list = []
    for blocks in ordering_block_lst:
        for prog_ind in blocks:
            re_ordered_gate_list.append(gate_list[prog_ind[0]])

    # Re-construct pyQuil program
    final_prog_list = initial_decl_list + [Pragma('PRESERVE_BLOCK')] + re_ordered_gate_list + [Pragma('END_PRESERVE_BLOCK')] + measure_list
    final_prog = Program()

    for instruction in final_prog_list:
        final_prog += instruction
        
    return final_prog