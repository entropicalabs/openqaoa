import numpy as np
from math import log2
from random import shuffle
from typing import List
from matplotlib import pyplot as plt
from copy import deepcopy

from . import QAOA
from ...backends import create_device

class QAOABenchmark:
    """
    Benchmark is a class that implements benchmarking for QAOA.
    """
    def __init__(self, qaoa: QAOA):
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

    def run(    
            self, 
            n_points_axis,
            ranges: list = None, 
            ):

        if ranges is None: 
            n_params = len(self.qaoa.variate_params)
            ranges = [ (0, 2*np.pi) for _ in range(n_params) ]
        else:
            n_params = sum([1 for r in ranges if len(r)==2])

        
        params_obj = deepcopy(self.qaoa.variate_params)

        values = np.zeros((n_points_axis, n_points_axis))
        for i, point in enumerate(self.__ordered_points(n_params, n_points_axis)):
            print("Point", i+1, "out of", n_points_axis**n_params)

            params = [ point[i]*(ranges[i][1]-ranges[i][0])/(n_points_axis-1) + ranges[i][0] for i in range(n_params) ] 
            params_obj.update_from_raw(params)
            values[tuple(point)] = self.qaoa.backend.expectation(params_obj)

        # save the results
        self.ranges = ranges
        self.values = values

    def plot(self):

        axis = [ np.linspace(r[0], r[1], self.values.shape[i]) for i, r in enumerate(self.ranges) ]

        fig, ax = plt.subplots()
        ax.pcolorfast(*axis, self.values)
        plt.show()

    @staticmethod
    def __ordered_points(n_params, n_points_axis):

        assert isinstance(n_params, int) and n_params > 1, \
            "The number of parameters must be an integer, and greater than 1"
        assert isinstance(n_points_axis, int) and n_points_axis > 3, \
            "The number of points per axis must be an integer, and greater than 3"

        ## we create a grid of points for each axis
        axis_points = list(range(n_points_axis))

        ## first we create all the points to evaluate and we order them such that every 2**n_params one point is used at each round, and the points are shuffled

        # we separate the points in two lists, [0, 2, 4, ...] and [1, 3, 5, ...]
        axis_points_separated = ( [[i] for i in axis_points[::2]], [[i] for i in axis_points[1::2]] ) 

        # we create a list of lists, where each list will tell how to combine the two lists of points to create the points for each round 
        zero_one = [[0], [1]]
        order_list = zero_one
        for i in range(n_params-1):
            order_list = [order + y for order in order_list for y in zero_one]

        # the variable points will be a list of lists, where each list is a round of points to evaluate
        points = [axis_points_separated[order[0]] for order in order_list] # we start with the first axis

        # we create the points for each round by combining the points of the previous round with the points of the next axis
        for k in range(1, n_params):
            for i in range(len(points)):
                points[i] = [ point_i + point_j for point_i in points[i] for point_j in axis_points_separated[order_list[i][k]] ]
                shuffle(points[i]) # we shuffle the points at each round

        # the final list of points to evaluate is a concatenation of all the rounds
        ordered_points = []
        for round in range(len(points)):
            ordered_points += points[round]

        ## then we reorder the points such that the first round we have a grid of points with very low resolution, and then at each round we increase the resolution
        values = np.zeros([n_points_axis for _ in range(n_params)])

        used_axis_points = []
        reordered_points = []
        for round in range(1, int(log2(len(axis_points)-1))+1):
            new = [k for k in axis_points[::(len(axis_points)-1)//2**round] if k not in used_axis_points]
            used_axis_points += new

            points_ = ordered_points.copy()
            for point in ordered_points:
                if all([i in used_axis_points for i in point]) and values[tuple(point)]==0:
                    values[tuple(point)] = round            
                    reordered_points.append(points_.pop(points_.index(point)))

            ordered_points = points_ # there are less points now

        return reordered_points