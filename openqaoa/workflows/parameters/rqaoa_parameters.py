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

ALLOWED_RQAOA_TYPES = ['adaptive', 'custom']

class RqaoaParameters(Parameters):
    """
    Parameter class to initialise parameters to run a desired RQAOA program.

    Attributes
    ----------
    rqaoa_type: `int`
        String specifying the RQAOA scheme under which eliminations are computed. The two methods are 'custom' and
        'adaptive'. Defaults to 'adaptive'.
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
    original_hamiltonian: `Hamiltonian`
        Hamiltonian encoding the original problem fed into the RQAOA algorithm.
    counter: `int`
        Variable to count the step in the schedule. If counter = 3 the next step is schedule[3]. 
        Default is 0, but can be changed to start in the position of the schedule that one wants.
    """
    
    def __init__(self,
                 rqaoa_type: str = 'custom',
                 n_max: int = 1,
                 steps: Union[list,int] = 1,
                 n_cutoff: int = 5,
                 original_hamiltonian: Hamiltonian = None,
                 counter: int = 0):
        """
        Initialises RQAOA program parameters.
        
        Parameters
        ----------
        rqaoa_type: `int`
            String specifying the RQAOA scheme under which eliminations are computed. The two methods are 'custom' and
            'adaptive'. Defaults to "custom".
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
        counter: `int`
            Variable to count the step in the schedule. If counter = 3 the next step is schedule[3]. 
            Default is 0, but can be changed to start in the position of the schedule that one wants.
        """
        self.rqaoa_type = rqaoa_type.lower()
        self.n_max = n_max
        self.steps = steps
        self.n_cutoff = n_cutoff
        self.original_hamiltonian = original_hamiltonian
        self.counter = counter
        
        #check if the rqaoa type is correct
        if not self.rqaoa_type in ALLOWED_RQAOA_TYPES:
            self.compiled = False        
            raise Exception(f'rqaoa_type {self.rqaoa_type} is not supported. Please select "adaptive" or "custom".')

        # check if the parameters given are in accordance with the rqaoa_type
        if self.rqaoa_type == 'adaptive':
            if self.steps != 1:
                raise ValueError(
                    f'When using the adaptive method, the `steps` parameter is not required.  \
                    The parameter that specifies the maximum number of eliminations per step is `n_max`.')
            if self.counter != 0:
                raise ValueError(
                    f'When using the adaptive method, the `counter` parameter is not required.')
        else:
            if self.n_max != 1:
                raise ValueError(
                    f'When using the custom method, the `n_max` parameter is not required.  \
                    The parameter that specifies the number of eliminations is `steps`, which can be a string or a list.')

        


