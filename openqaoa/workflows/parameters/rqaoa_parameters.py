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

from .parameters import Parameters
from openqaoa.qaoa_parameters import Hamiltonian


class QaoaParameters(Parameters):
    def __init__(self,
                 p: int = 1,
                 backend: str = 'vectorized',
                 params_type: str = 'standard',
                 init_type: str = 'ramp',
                 shots: int = None,
                 optimizer_dict: dict = {'method': 'cobyla', 'maxiter': 200},
                 qpu_params: dict = None):
        self.p = p
        self.backend = backend
        self.params_type = params_type
        self.init_type = init_type
        self.shots = shots
        self.optimizer_dict = optimizer_dict
        self.qpu_params = qpu_params


class RqaoaParameters(Parameters):
    def __init__(self,
                 rqaoa_type: str = 'adaptive',
                 n_max: int = 5,
                 n_cutoff: int = 5,
                 max_terms_and_stats_list: list = None,
                 original_hamiltonian: Hamiltonian = None):
        self.rqaoa_type = rqaoa_type
        self.n_max = n_max
        self.n_cutoff = n_cutoff
        self.max_terms_and_stats_list = max_terms_and_stats_list
        self.original_hamiltonian = original_hamiltonian
