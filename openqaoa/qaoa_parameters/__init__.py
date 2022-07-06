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
Different hyper-parameterisations and trainable-parameterisations for QAOA

Hyper-Parameters help with problem hamiltonian initialisation and fixing `p`
for the QAOA problem

AbstractParams and child classes let one choose the parameterisation for
solving the QAOA problem in hand.  
"""

from .standardparams import *
from .extendedparams import *
from .fourierparams import *
from .annealingparams import *
from .qaoa_params import *
from .operators import *
from .baseparams import *
from .gatemap import *
from .gates import *
from .variational_params_converters import converter