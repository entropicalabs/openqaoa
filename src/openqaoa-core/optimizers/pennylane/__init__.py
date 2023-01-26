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
PennyLane optimizers directory for the classical optimization loop for QAOA 
"""

from .pennylane_optimizers import *
from . import numpy
from . import math
from . import fourier

#empty class to be used as a placeholder for the QNode class from PennyLane
class QNode:
	def __init__(self):
		pass
