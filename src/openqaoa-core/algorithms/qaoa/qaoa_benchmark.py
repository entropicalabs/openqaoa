import numpy as np
from math import log2
from random import shuffle
from typing import List, Union
from matplotlib import pyplot as plt
from copy import deepcopy
from IPython.display import clear_output
import time

from . import QAOA
from ...backends import create_device


class QAOABenchmark:
    """
    Benchmark is a class that implements benchmarking for QAOA.

    Attributes
    ----------
    qaoa : QAOA
        The QAOA object to be benchmarked.
    reference : QAOA
        The reference QAOA object, which will be used to compare the results of the benchmarked QAOA object.
    ranges : List[tuple]
        The ranges of the variate parameters of the benchmarked QAOA object.
    ranges_reference : List[tuple]
        The ranges of the variate parameters of the reference QAOA object.
    values : np.ndarray
        The values of the benchmarked QAOA object.
    values_reference : np.ndarray
        The values of the reference QAOA object.
    difference : np.ndarray
        The difference between the values of the benchmarked QAOA object and the values of the reference QAOA object.
    difference_mean : float
        The mean of the difference between the values of the benchmarked QAOA object and the values of the reference QAOA object.
    """

    def __init__(self, qaoa: QAOA):
        """
        Constructor for the Benchmark class.

        Parameters
        ----------
        qaoa : QAOA
            The QAOA object to be benchmarked.
        """
        # check if the QAOA object inputted is valid and save it
        assert isinstance(qaoa, QAOA), "`qaoa` must be an instance of QAOA"
        assert qaoa.compiled, "`qaoa` must be compiled before benchmarking"
        self.qaoa = qaoa

        # create a reference QAOA object, which will be used to compare the results of the benchmarked QAOA object
        self.reference = QAOA()
        self.reference.circuit_properties = (
            self.qaoa.circuit_properties
        )  # copy the circuit properties of the benchmarked QAOA object
        try:
            self.reference.set_device(
                create_device(location="local", name="analytical_simulator")
            )
            self.reference.compile(self.qaoa.problem)
        except:
            self.reference.set_device(
                create_device(location="local", name="vectorized")
            )
            self.reference.compile(self.qaoa.problem)
            print(
                "Warning: vectorized simulator will be used for the reference, since the analytical simulator is not available for the circuit properties of the benchmarked QAOA object"
            )

        # initialize values and ranges
        self.values = None
        self.values_reference = None
        self.ranges = None
        self.ranges_reference = None

    @property
    def difference(self):
        "The difference between the values of the benchmarked QAOA object and the values of the reference QAOA object."
        if self.values is None:
            raise Exception(
                "You must run the benchmark before calculating the difference"
            )
        if self.values_reference is None:
            raise Exception(
                "You must run the reference before calculating the difference"
            )
        if self.values.shape != self.values_reference.shape:
            raise Exception(
                "The ranges and number of points of the values and reference values must be the same"
            )
        return self.values - self.values_reference

    @property
    def difference_mean(self):
        "The mean of the difference between the values of the benchmarked QAOA object and the values of the reference QAOA object."
        return np.mean(self.difference)

    @property
    def difference_std(self):
        "The standard deviation of the difference between the values of the benchmarked QAOA object and the values of the reference QAOA object."
        return np.std(self.difference)

    def __assert_run_inputs(
        self,
        n_points_axis: int,
        ranges: List[tuple],
        run_main: bool,
        run_reference: bool,
        plot: bool,
        plot_difference: bool,
        plot_options: dict,
        verbose: bool,
    ):
        "Private method that checks the inputs of the run (and run_reference) method."
        assert isinstance(
            n_points_axis, int
        ), "The number of points per axis must be an integer"
        assert (
            isinstance(ranges, list) or ranges is None
        ), "The ranges argument must be a list of tuples: (min, max) or (value,)"
        assert len(ranges) == len(
            self.qaoa.variate_params
        ), "The number of ranges must be equal to the number of variate parameters, which is {}".format(
            len(self.qaoa.variate_params)
        )
        assert all(
            [isinstance(r, tuple) or isinstance(r, list) for r in ranges]
        ), "Each range must be a tuple: (min, max) or (value,)"
        assert all(
            [len(r) == 1 or len(r) == 2 for r in ranges]
        ), "Each range must be a tuple of length 1 or 2: (min, max) or (value,)"
        assert (
            len([r for r in ranges if len(r) == 2]) > 0
        ), "At least one range must be a tuple of length 2: (min, max)"
        for bools in [
            "run_main",
            "run_reference",
            "plot",
            "plot_difference",
            "verbose",
        ]:
            assert isinstance(
                eval(bools), bool
            ), "The {} argument must be a boolean".format(bools)
        assert (
            run_main or run_reference
        ), "You must run the main or the reference or both"
        assert isinstance(
            plot_options, dict
        ), "The plot_options argument must be a dictionary"

    def __print_expected_time(self, k, n_points):
        "Private method that prints the expected remaining time to complete the benchmark."
        print("\rPoint {} out of {}.".format(k + 1, n_points), end="")
        if k == 0:
            self.__start_time = time.time()
        elif k > 0:
            expected_time = (time.time() - self.__start_time) * (n_points - k) / k
            print(
                " Expected remaining time to complete: {}, it will be finished at {}.".format(
                    time.strftime("%H:%M:%S", time.localtime(expected_time)),
                    time.strftime(
                        "%H:%M:%S", time.localtime(time.time() + expected_time)
                    ),
                ),
                end="",
            )

    def run(
        self,
        n_points_axis,
        ranges: List[tuple],
        run_main: bool = True,
        run_reference: bool = True,
        plot: bool = False,
        plot_every: int = 1000,
        plot_difference: bool = False,
        plot_options: dict = {},
        verbose: bool = True,
    ):
        """
        Evaluates the QAOA circuit of the benchmarked (and the reference) QAOA object for a given number of points per axis and ranges.

        Parameters
        ----------
        n_points_axis : int
            The number of points per axis. It is recommended to use a number that is a power of 2 + 1: 2**k+1 (where, k is an integer).
            Using this number will ensure that the points evaluated are ordered such that there are k rounds, each round covers the whole range,
            and each round has the double number of points (per axis) of the previous round.
        ranges : List[tuple]
            The sweep ranges of the parameters. The expected format is a list of tuples: (min, max) or (value,). One tuple per parameter.
            If the length of the tuple is 1, the parameter will be fixed to the value of the tuple.
            If the length of the tuple is 2, the parameter will be swept from the first value of the tuple to the second value of the tuple.
        run_main : bool, optional
            If True, the benchmarked QAOA object will be evaluated for the requested points.
            The default is True.
        run_reference : bool, optional
            If True, the reference QAOA object will be evaluated for the same points.
            The default is True.
        plot : bool, optional
            If True, the values will be plotted.
            The default is False.
        plot_every : int, optional
            The number of evaluations after which the plot will be updated.
            The default is 1000.
        plot_difference : bool, optional
            If True, the difference between the values of the benchmarked QAOA object and the values of the reference QAOA object will be plotted.
            The default is False.
        plot_options : dict, optional
            The options for the plot. The expected format is a dictionary with the keys of the plot method of this class.
            The default is {}.
        verbose : bool, optional
            If True, the expected remaining time to complete the benchmark will be printed.
            The default is True.
        """

        # check the inputs
        self.__assert_run_inputs(
            n_points_axis,
            ranges,
            run_main,
            run_reference,
            plot,
            plot_difference,
            plot_options,
            verbose,
        )

        # plot options
        plot_options = {**{"verbose": verbose}, **plot_options}

        # save the ranges
        if run_main:
            self.ranges = ranges
        if run_reference:
            self.ranges_reference = ranges

        # get only the ranges that will be swept, and the number of parameters swept
        ranges_to_use = [r for r in ranges if len(r) == 2]
        n_params = len(ranges_to_use)

        # prepare the axes of the grid
        axes = [
            np.linspace(r[0], r[1], n_points_axis) for r in ranges_to_use
        ]  # the starting and ending points are given by the ranges and the number of points is given by n_points_axis

        # function that given the values used for the parameters swept, returns the values of all the parameters in order
        params = lambda new_params: [
            r[0] if len(r) == 1 else new_params.pop(0) for r in ranges
        ]

        # copy the parameters object, which will be updated before each evaluation
        params_obj = deepcopy(self.qaoa.variate_params)

        # initialize the array that will contain the values of the benchmarked QAOA object
        if run_main:
            self.values = np.zeros([n_points_axis for _ in range(n_params)])
        if run_reference:
            self.values_reference = np.zeros([n_points_axis for _ in range(n_params)])

        # initialize the arrays that will contain the difference between the values of the benchmarked QAOA object and the values of the reference QAOA object
        both_qaoa = [
            qaoa
            for qaoa, boolean in zip(
                [self.qaoa, self.reference], [run_main, run_reference]
            )
            if boolean
        ]
        both_values = [
            values
            for values, boolean in zip(
                [self.values, self.values_reference], [run_main, run_reference]
            )
            if boolean
        ]
        both_strings = [
            string
            for string, boolean in zip(
                ["benchmark", "reference"], [run_main, run_reference]
            )
            if boolean
        ]

        # loop over the benchmarked QAOA object and the reference QAOA object (if requested)
        for qaoa, values, string in zip(both_qaoa, both_values, both_strings):

            if verbose:
                print(f"Running {string}.")

            # evaluate all the points in the grid, in the order provided by the function __ordered_points. We loop over the indices of the grid.
            for k, i_point in enumerate(self.__ordered_points(n_params, n_points_axis)):

                if verbose:
                    self.__print_expected_time(
                        k, n_points_axis**n_params
                    )  # print the expected remaining time, info for the user

                new_params = [
                    axis[i] for axis, i in zip(axes, i_point)
                ]  # from the indices of the grid, get the values of the parameters that will be evaluated
                params_obj.update_from_raw(params(new_params))
                values[tuple(i_point[::-1])] = qaoa.backend.expectation(params_obj)

                # plot every 'plot_every' points and at the end
                if (
                    (
                        (k % plot_every == 0 and k > 0)
                        or k + 1 == n_points_axis**n_params
                    )
                    and plot
                    and string == "benchmark"
                ):
                    clear_output(wait=True)

                    if (not k + 1 == n_points_axis**n_params) and n_params == 1:
                        plot_opt = {"marker": "o", "linestyle": "", "markersize": 3}
                    else:
                        plot_opt = {}
                    self.plot(plot_options=plot_opt, **plot_options)

            if verbose:
                print(
                    " "
                )  # print a blank line, necessary because previous print: end=""

        # plot the reference if requested
        if plot and run_reference:
            self.plot(main=False, reference=True, difference=False, **plot_options)

        # plot the difference if requested and the reference has been run
        if plot_difference:
            try:
                self.plot(main=False, reference=False, difference=True, **plot_options)
            except:
                print(
                    "The difference cannot be plotted because the reference has not been run."
                )

    def plot(
        self,
        ax: plt.Axes = None,
        title: Union[str, List[str]] = None,
        labels: List[str] = None,
        labels_legend: Union[str, List[str]] = None,
        main: bool = True,
        reference: bool = False,
        difference: bool = False,
        one_plot: bool = False,
        # params_to_plot:List[int] = None, TODO: difficult because you need to specify the values of the other parameters
        plot_options: dict = {},
        verbose: bool = True,
    ):
        """
        Plots the values of the benchmarked QAOA object and/or the values of the reference QAOA
          object and/or the difference between the values of the benchmarked QAOA object and the values of the reference QAOA object.

        Parameters
        ----------
        ax : plt.Axes, optional
            The matplotlib Axes object to plot on. If None, a new figure will be created.
            The default is None.
        title : Union[str, List[str]], optional
            The title of the plot. The expected format is a string or a list of strings: one for each plot.
            If None, the title will be the default: `main plot`, `reference plot` or `difference plot`.
            The default is None.
        labels : List[str], optional
            The labels of the axes. The expected format is a list of two strings: one for each axis.
            If None, the labels will be the number of the parameters.
            The default is None.
        labels_legend : Union[str, List[str]], optional
            The labels of the legend. The expected format is a string or a list of strings: one for each line.
            It is only available if the sweep is over just one parameter, and if `one_plot` is True or `ax` is not None.
            If None, the labels will be the default: `main`, `reference` or `difference`.
            The default is None.
        main : bool, optional
            If True, the values of the benchmarked QAOA object will be plotted.
            The default is True.
        reference : bool, optional
            If True, the values of the reference QAOA object will be plotted.
            The default is False.
        difference : bool, optional
            If True, the difference between the values of the benchmarked QAOA object and the values of the reference QAOA object will be plotted.
            The default is False.
        one_plot : bool, optional
            If True, the values of the benchmarked QAOA object,
            the values of the reference QAOA object and the difference between the values of the benchmarked QAOA object
            and the values of the reference QAOA object will be plotted in the same plot.
            It is not available if the sweep is over more than one parameter.
            The default is False.
        plot_options : dict, optional
            The options for the plot (plt.plot for one parameter, plt.pcolorfast for two parameters).
        """

        # check the inputs
        assert (
            isinstance(ax, plt.Axes) or ax is None
        ), "ax must be a matplotlib Axes or None"
        assert (
            isinstance(labels, list) or labels is None
        ), "labels must be a list of two strings or None"
        assert (
            all([isinstance(label, str) for label in labels])
            if labels is not None
            else True
        ), "labels must be a list of two strings or None"
        assert (
            len(labels) == 2 if labels is not None else True
        ), "labels must be a list of two strings, one for each axis"
        for plot in ["main", "reference", "difference"]:
            assert isinstance(eval(plot), bool), plot + " must be a boolean"
        assert (
            main or reference or difference
        ), "You must specify at least one of the main, reference or difference plots"
        assert isinstance(plot_options, dict), "plot_options must be a dictionary"

        # if labels_legend is not a list or tuple, make it a list
        if not isinstance(labels_legend, (list, tuple)):
            labels_legend = [labels_legend for _ in range(3)]

        # create a dictionary where the keys are the three possible plots and the values are a tuple
        # with the boolean (which says if the plot is requested) and the values to plot
        values = {
            "main": (main, self.values if main else None),
            "reference": (reference, self.values_reference if reference else None),
            "difference": (difference, self.difference if difference else None),
        }

        # create the figure and axes if the axis are not provided
        ax_input = ax  # save the input axis
        nrows = 1
        if ax_input is None:
            nrows = (
                1 if one_plot else sum([1 if values[key][0] else 0 for key in values])
            )
            fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(6.5, 5 * nrows))

        # plot the requested plots
        count_sp = 0  # counter of subplots
        for key in values:
            if values[key][0]:  # skip the plot if it is not requested

                # raise an exception if the values for a plo requested are not available
                if values[key][1] is None:
                    raise Exception(
                        "You must run the benchmark before plotting the results, there are no values for the "
                        + key
                        + " plot"
                    )

                # get the ranges to use, use the reference ranges if the key is "reference", otherwise use the main ranges
                ranges = self.ranges if key != "reference" else self.ranges_reference
                ranges_to_use = [r for r in ranges if len(r) == 2]
                n_params = len(ranges_to_use)

                # create the axes, the range is provided by the ranges_to_use and the number of values is provided by the values[key][1].shape
                axes = [
                    np.linspace(r[0], r[1], values[key][1].shape[i])
                    for i, r in enumerate(ranges_to_use)
                ]

                # skip the plot if there are more than two parameters to plot, and print a warning
                if n_params > 2:
                    print(
                        "Only 1 or 2 parameters can be plotted using this method. You are trying to plot "
                        + str(n_params)
                        + " parameters. You can use the argument params_to_plot to specify which parameters to plot."
                    )
                    break
                # if verbose, print the parameters used for the plot
                elif verbose:
                    print(
                        "Plotting the " + key + " plot with the following parameters:"
                    )
                    for i, r in enumerate(ranges):
                        if len(r) == 2:
                            print(
                                "\tParameter "
                                + str(i)
                                + ": "
                                + str(r[0])
                                + " to "
                                + str(r[1])
                                + ", with "
                                + str(values[key][1].shape[i])
                                + " values"
                            )
                        else:
                            print("\tParameter " + str(i) + ": " + str(r[0]))

                ## plot the values

                # if we are plotting only one plot, use the same axes, otherwise different axes for each subplot
                axis = ax if nrows == 1 else ax[count_sp]

                # set the labels and plot the values
                if n_params == 1:
                    axis.set_xlabel(
                        "Parameter {}".format(
                            [i for i, r in enumerate(ranges) if len(r) == 2][0]
                        )
                    )
                    axis.set_ylabel("Expectation value")

                    label_legend = (
                        labels_legend[count_sp]
                        if labels_legend[count_sp] is not None
                        else key
                    )
                    axis.plot(*axes, values[key][1], label=label_legend, **plot_options)
                else:
                    if one_plot:
                        raise Exception(
                            "For two parameters, you must specify one_plot=False"
                        )

                    axis.set_xlabel(
                        "Parameter {}".format(
                            [i for i, r in enumerate(ranges) if len(r) == 2][0]
                        )
                    )
                    axis.set_ylabel(
                        "Parameter {}".format(
                            [i for i, r in enumerate(ranges) if len(r) == 2][1]
                        )
                    )
                    plot = axis.pcolorfast(*axes, values[key][1], **plot_options)
                    _ = plt.colorbar(plot)

                # set the title
                if not one_plot and title is None:
                    axis.set_title(key + " plot")
                elif not title is None:
                    if not isinstance(title, list) and not isinstance(title, tuple):
                        axis.set_title(title)
                    else:
                        if len(title) < nrows:
                            raise Exception(
                                "If title is a list or tuple, it must have the same length as the number of plots"
                            )
                        axis.set_title(title[count_sp])

                # replace the labels if specified
                if labels is not None:
                    axis.set_xlabel(labels[0])
                    axis.set_ylabel(labels[1])

                # increase the counter for the axes
                count_sp += 1

        # add legend if one_plot
        if one_plot:
            ax.legend()

        # show the figure if the axis were not provided
        if ax_input is None:
            if nrows > 1:
                fig.subplots_adjust(hspace=0.3)  # add some space between the subplots
            plt.show()
            plt.close(fig)

    @staticmethod
    def __ordered_points(n_params, n_points_axis):
        """
        This method creates a grid of points for each parameter, and then creates all the combinations of these points.
        The points are ordered such that we evaluate the grid by rounds, the first round the resolution is very low,
        next rounds the resolution is higher,
        and also, every n_points = 2**n_params one point is used at each round, and the points are shuffled.

        It works very well for n_points_axis = 2**k + 1 (where k is any integer), but it also works for other values.

        Parameters
        ----------
        n_params : int
            The number of parameters to sweep.
        n_points_axis : int
            The number of points per axis.

        Returns
        -------
        points : list
            The list of points to evaluate in order.
        """

        assert (
            isinstance(n_params, int) and n_params > 0
        ), "The number of parameters must be an integer, and greater than 0"
        assert (
            isinstance(n_points_axis, int) and n_points_axis > 3
        ), "The number of points per axis must be an integer, and greater than 3"

        ## we create a grid of points for each axis
        axis_points = list(range(n_points_axis))

        ## first we create all the points to evaluate and we order them such that every n_points = 2**n_params one point is used at each round, and the points are shuffled

        # we separate the points in two lists, [0, 2, 4, ...] and [1, 3, 5, ...]
        axis_points_separated = (
            [[i] for i in axis_points[::2]],
            [[i] for i in axis_points[1::2]],
        )

        # we create a list of lists, where each list will tell how to combine the two lists of points to create the points for each round
        zero_one = [[0], [1]]
        order_list = zero_one
        for _ in range(n_params - 1):
            order_list = [order + y for order in order_list for y in zero_one]

        # the variable points will be a list of lists, where each list is a round of points to evaluate
        points = [
            axis_points_separated[order[0]] for order in order_list
        ]  # we start with the first axis

        # we create the points for each round by combining the points of the previous round with the points of the next axis
        for k in range(1, n_params):
            for x in range(len(points)):
                points[x] = [
                    point_i + point_j
                    for point_i in points[x]
                    for point_j in axis_points_separated[order_list[x][k]]
                ]

        # we shuffle the points at each round
        for i in range(len(points)):
            shuffle(points[i])

        # the final list of points to evaluate is a concatenation of all the rounds
        ordered_points = []
        for round in range(len(points)):
            ordered_points += points[round]

        ## then we reorder the points such that the first round we have a grid of points with very low resolution, and then at each round we increase the resolution
        values = np.zeros([n_points_axis for _ in range(n_params)])

        used_axis_points = []
        reordered_points = []
        for round in range(1, int(log2(len(axis_points) - 1)) + 1):
            new = [
                k
                for k in axis_points[:: (len(axis_points) - 1) // 2**round]
                if k not in used_axis_points
            ]
            used_axis_points += new

            points_ = ordered_points.copy()
            for point in ordered_points:
                if (
                    all([i in used_axis_points for i in point])
                    and values[tuple(point)] == 0
                ):
                    values[tuple(point)] = round
                    reordered_points.append(points_.pop(points_.index(point)))

            ordered_points = points_  # there are less points now

        return reordered_points
