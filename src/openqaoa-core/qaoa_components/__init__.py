"""
Different hyper-parameterisations and trainable-parameterisations for QAOA

Hyper-Parameters help with problem hamiltonian initialisation and fixing `p`
for the QAOA problem

AbstractParams and child classes let one choose the parameterisation for
solving the QAOA problem in hand.  
"""

from .ansatz_constructor import *
from .variational_parameters import *
