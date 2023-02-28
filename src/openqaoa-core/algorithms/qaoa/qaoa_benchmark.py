import numpy as np

from . import QAOA
from ...backends import create_device

class Benchmark():
    """
    Benchmark is a class that implements benchmarking for QAOA.
    """
    def __init__(self, qaoa: QAOA, range: tuple = None):
        """
        Constructor for the Benchmark class.
        """
        #check if the QAOA object inputted is valid and save it
        assert isinstance(qaoa, QAOA), "`qaoa` must be an instance of QAOA"
        assert qaoa.compiled, "`qaoa` must be compiled before benchmarking"
        self.qaoa = qaoa

        #create a reference QAOA object, which will be used to compare the results of the benchmarked QAOA object
        self.reference = QAOA()
        self.reference.circuit_properties = self.qaoa.circuit_properties
        self.reference.set_device(create_device(location='local', name='analytical_simulator'))
        self.reference.compile(self.qaoa.problem)

        # TODO: be able to use it for p>1, and all the other parameters

        if range is None: self.run()
        else: self.run(range)

    def run(self, range: tuple = (0, 2*np.pi)):
        
        pass


        