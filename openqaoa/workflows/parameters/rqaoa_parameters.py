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

from typing import Union
from .parameters import Parameters
from openqaoa.qaoa_parameters import Hamiltonian


class QaoaParameters(Parameters):
    """
    Parameter class to initialise parameters to run a desired QAOA program.


    Attributes
    ----------
    p: `int`
        Number of QAOA layers.
    backend: `str`
        String containing the specific backend in which QAOA is run.
    params_type: `str`
        String specificying the parametrization of the QAOA ansatz.
    init_type: `str`
        String specifiying the initialization of the QAOA variational parameters.
    shots: `int`
        Number of shots considered for statistics.
    qpu_params: `dict`
        Dictionary containing further specifications of a hardware backend.
    """
    def __init__(self,
                 p: int = 1,
                 backend: str = 'vectorized',
                 params_type: str = 'standard',
                 init_type: str = 'ramp',
                 shots: int = None,
                 optimizer_dict: dict = {'method': 'cobyla', 'maxiter': 200},
                 qpu_params: dict = None):
        """
        Initialises QAOA program parameters.

        Parameters
        ----------
        p: `int`
            Number of QAOA layers. Defaults to 1.
        backend: `str`
            String containing the specific backend in which QAOA is run. Defaults to vectorized.
        params_type: `str`
            String specificying the parametrization of the QAOA ansatz. Defaults to 'standard'.
        init_type: `str`
            String specifiying the initialization of the QAOA variational parameters. Defaults to 'ramp'.
        shots: `int`
            Number of shots considered for statistics. Defaults to None.
        qpu_params: `dict`
            Dictionary containing further specifications of a hardware backend. Defaults to None.
        """
        self.p = p
        self.backend = backend
        self.params_type = params_type
        self.init_type = init_type
        self.shots = shots
        self.optimizer_dict = optimizer_dict
        self.qpu_params = qpu_params


class RqaoaParameters(Parameters):
    """
    Parameter class to initialise parameters to run a desired RQAOA program.

    Attributes
    ----------
    rqaoa_type: `int`
        String specifying the RQAOA scheme under which eliminations are computed. The two methods are 'custom' and
        'adaptive'.
    n_max: `int`
        Maximum number of eliminations allowed at each step when using the adaptive method.
    steps: `Union[list,int]`
        Elimination schedule for the RQAOA algorithm. If an integer is passed, it sets the number of spins eliminated
        at each step. If a list is passed, the algorithm will follow the list to select how many spins to eliminate 
        at each step. Note that the list needs enough elements to specify eliminations from the initial number of qubits
        up to the cutoff value. If the list contains more, the algorithm will follow instructions until the cutoff value 
        is reached.
    n_cutoff: `int`
        Cutoff value at which the RQAOA algorithm obtains the solution classically.
    shots: `int`
        Number of shots considered for statistics.
    qpu_params: `dict`
        Dictionary containing further specifications of a hardware backend.
    original_hamiltonian: `Hamiltonian`
        Hamiltonian encoding the original problem fed into the RQAOA algorithm.
    """
    
    def __init__(self,
                 rqaoa_type: str = 'adaptive',
                 n_max: int = 1,
                 steps: Union[list,int] = 1,
                 n_cutoff: int = 5,
                 original_hamiltonian: Hamiltonian = None):
        """
        Initialises RQAOA program parameters.

        Parameters
        ----------
        rqaoa_type: `int`
            String specifying the RQAOA scheme under which eliminations are computed. The two methods are 'custom' and
            'adaptive'. Defaults to 'adaptive'.
        n_max: `int`
            Maximum number of eliminations allowed at each step when using the adaptive method. Defaults to 1.
        steps: `Union[list,int]`
            Elimination schedule for the RQAOA algorithm. If an integer is passed, it sets the number of spins eliminated
            at each step. If a list is passed, the algorithm will follow the list to select how many spins to eliminate 
            at each step. Note that the list needs enough elements to specify eliminations from the initial number of qubits
            up to the cutoff value. If the list contains more, the algorithm will follow instructions until the cutoff value 
            is reached. Defaults to 1.
        n_cutoff: `int`
            Cutoff value at which the RQAOA algorithm obtains the solution classically. Defaults to 5.
        original_hamiltonian: `Hamiltonian`
            Hamiltonian encoding the original problem fed into the RQAOA algorithm. Defaults to None.
        """
        self.rqaoa_type = rqaoa_type
        self.n_max = n_max
        self.steps = steps
        self.n_cutoff = n_cutoff
        self.original_hamiltonian = original_hamiltonian
