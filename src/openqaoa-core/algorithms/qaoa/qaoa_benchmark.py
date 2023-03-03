import numpy as np
from math import log2
from random import shuffle
from typing import List
from matplotlib import pyplot as plt
from copy import deepcopy
from IPython.display import clear_output

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
        try:
            self.reference.set_device(create_device(location='local', name='analytical_simulator'))
            self.reference.compile(self.qaoa.problem)
        except:
            self.reference.set_device(create_device(location='local', name='vectorized'))
            self.reference.compile(self.qaoa.problem)
            print("Warning: vectorized simulator will be used for the reference, since the analytical simulator is not available for the circuit properties of the benchmarked QAOA object")        

    @property
    def difference(self):
        if not hasattr(self, "values"):
            raise Exception("You must run the benchmark before calculating the difference")
        if not hasattr(self, "reference_values"):
            raise Exception("You must run the reference before calculating the difference")
        if self.values.shape != self.reference_values.shape:
            raise Exception("The ranges and number of points of the values and reference values must be the same")
        return self.values - self.reference_values
    
    @property
    def difference_mean(self):
        return np.mean(self.difference)

    def run_reference(
            self, 
            n_points_axis: int = None, 
            ranges: List[tuple] = None, 
            plot: bool = False,
            plot_options:dict = {}
            ):

        # check the input -> n_points_axis
        if n_points_axis == None and hasattr(self, "values"):
            n_points_axis = self.values.shape[0]
        elif n_points_axis == None:
            raise Exception("You must specify the number of points per axis")

        # check the input -> ranges
        if ranges == None and hasattr(self, "ranges"):
            ranges = self.ranges
        elif ranges == None:
            raise Exception("You must specify the ranges")
        self.ranges_reference = ranges

        # n_params = sum([1 for r in ranges if len(r)==2])
        
        ranges_to_use = [ r for r in ranges if len(r)==2 ]
        axes = [ np.linspace(r[0], r[1], n_points_axis) for r in ranges_to_use ]      
        params = lambda new_params: [ r[0] if len(r) == 1 else new_params.pop(0) for r in ranges ]        
        params_obj = deepcopy(self.qaoa.variate_params)
        self.reference_values = np.zeros([n_points_axis for _ in range(len(ranges_to_use))])

        # evaluate all the points in the grid
        for k, i_point in enumerate(np.ndindex(*[len(axis) for axis in axes])):
            print("\r", "Reference point", k+1, "out of", n_points_axis**len(ranges_to_use), end="")

            new_params = [ axis[i] for axis, i in zip(axes, i_point) ]
            params_obj.update_from_raw(params(new_params))
            self.reference_values[i_point[::-1]] = self.reference.backend.expectation(params_obj)

        if plot:
            self.plot(main=False, reference=True, difference=False, **plot_options)
            


    def run(    
            self, 
            n_points_axis,
            ranges: List[tuple], 
            run_reference: bool = True,
            plot: bool = False,
            plot_reference: bool = False,
            plot_difference: bool = False,
            plot_options:dict = {}
            ):

        assert isinstance(n_points_axis, int), "The number of points per axis must be an integer"
        assert isinstance(ranges, list) or ranges is None, "The ranges argument must be a list of tuples: (min, max) or (value,)"
        assert len(ranges) == len(self.qaoa.variate_params), "The number of ranges must be equal to the number of variate parameters"
        assert all([isinstance(r, tuple) or isinstance(r, list) for r in ranges]), "Each range must be a tuple: (min, max) or (value,)"
        assert all([len(r)==1 or len(r)==2 for r in ranges]), "Each range must be a tuple of length 1 or 2: (min, max) or (value,)"
        assert isinstance(plot, bool), "The plot argument must be a boolean"

        # save the ranges
        self.ranges = ranges
  
        ranges_to_use = [ r for r in ranges if len(r)==2 ]
        n_params = len(ranges_to_use)     
        params = lambda new_params: [ r[0] if len(r) == 1 else new_params.pop(0) for r in ranges ]
        params_obj = deepcopy(self.qaoa.variate_params)
        self.values = np.zeros([n_points_axis for _ in range(n_params)])        

        for k, point in enumerate(self.__ordered_points(n_params, n_points_axis)):
            print("\r", "Point", k+1, "out of", n_points_axis**n_params, end="")

            new_params = [ point[i]*(r[1]-r[0])/(n_points_axis-1) + r[0] for i, r in enumerate(ranges_to_use) ] 
            params_obj.update_from_raw(params(new_params))
            self.values[tuple(point[::-1])] = self.qaoa.backend.expectation(params_obj)

            #plot every 1000 points and at the end
            if ((k%1000==0 and k>0) or k+1==n_points_axis**n_params) and plot:
                clear_output(wait=True)
                if (not k+1==n_points_axis**n_params) and n_params==1:
                    plot_opt={'marker':'o', 'linestyle':'', 'markersize':3}
                else:
                    plot_opt={}
                self.plot(plot_options=plot_opt, **plot_options)

        # run the reference if requested
        if run_reference:
            print("\nRunning the reference")
            self.run_reference(plot=plot_reference, plot_options=plot_options)
        # plot the reference if requested and the reference has been run
        elif plot_reference:
            self.plot(main=False, reference=True, difference=False, **plot_options)

        # plot the difference if requested and the reference has been run
        if plot_difference:
            self.plot(main=False, reference=False, difference=True, **plot_options)


    def plot(
            self, 
            ax = None, 
            main:bool = True, 
            reference:bool = False, 
            difference:bool = False, 
            plot_options:dict = {}
            ):

        ax_input = ax

        values = {
            "main": (
                main, 
                self.values if main and hasattr(self, "values") else None
            ),
            "reference": (
                reference,
                self.reference_values if reference and hasattr(self, "reference_values") else None
            ),
            "difference": (
                difference,
                self.difference if difference and hasattr(self, "values") and hasattr(self, "reference_values") else None
            ),
        }

        for key in values:
            if values[key][0]:

                if values[key][1] is None:
                    raise Exception("You must run the benchmark before plotting the results, there are no values for the " + key + " plot")
                
                ranges = self.ranges if key != "reference" else self.ranges_reference
                ranges_to_use = [ r for r in ranges if len(r)==2 ]
                n_params = len(ranges_to_use)
                axes = [ np.linspace(r[0], r[1], self.values.shape[i]) for i, r in enumerate(ranges_to_use) ]

                if ax_input is None:
                    fig, ax = plt.subplots()

                if n_params==1:
                    ax.plot(*axes, values[key][1], **plot_options)
                elif n_params==2:          
                    ax.pcolorfast(*axes, values[key][1], **plot_options)
                else:
                    raise Exception("Only 1 or 2 parameters can be plotted")

                if ax_input is None:
                    plt.show()


    @staticmethod
    def __ordered_points(n_params, n_points_axis):

        assert isinstance(n_params, int) and n_params > 0, \
            "The number of parameters must be an integer, and greater than 0"
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