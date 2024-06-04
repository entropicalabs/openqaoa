import importlib.metadata

from .algorithms import QAOA, RQAOA, QAOABenchmark
from .problems import QUBO
from .backends import create_device

__version__ = importlib.metadata.version("openqaoa")
