import unittest
import numpy as np

from openqaoa.algorithms.qaoa import QAOABenchmark
from openqaoa import QAOA, create_device
from openqaoa.problems import QUBO


class TestingBenchmark(unittest.TestCase):
    
        def test_simple_benchmark(self):

            qaoa = QAOA()
            qaoa.set_device(create_device(name='analytical_simulator', location='local'))
            qaoa.compile(QUBO.random_instance(5))

            benchmark = QAOABenchmark(qaoa)
            benchmark.run(n_points_axis=2**4, ranges=[(0,2), (-2,5)])

            assert benchmark.ranges == [(0, 2), (-2, 5)], "The ranges saved are not correct."
            assert benchmark.ranges_reference == [(0, 2), (-2, 5)], "The ranges saved for the reference are not correct."
            assert benchmark.values.shape == (16, 16), "The shape of the values matrix is not correct."
            assert benchmark.values_reference.shape == (16, 16), "The shape of the values matrix of the reference is not correct."
            assert benchmark.difference.shape == (16, 16), "The shape of the difference matrix is not correct."
            assert all(benchmark.difference.flatten() == 0), "The difference matrix should be all 0."
            assert benchmark.difference_mean == 0, "The difference mean should be 0."

        def __compare_values_benchmark(self, qaoa, n_points_axis, ranges, run_reference=False):

            benchmark = QAOABenchmark(qaoa)
            benchmark.run(n_points_axis=n_points_axis, ranges=ranges, run_reference=run_reference)

            ranges_to_use = [r for r in ranges if len(r) == 2]
            values = np.zeros([n_points_axis for _ in range(len(ranges_to_use))])
            axes = [np.linspace(r[0], r[1], n_points_axis) for r in ranges_to_use]

            params = lambda x: [r[0] if len(r) == 1 else x.pop(0) for r in ranges]

            if len(ranges_to_use) == 1:
                for i in range(n_points_axis):
                    values[i] = qaoa.evaluate_circuit(params([axes[0][i]]))['cost']
            elif len(ranges_to_use) == 2:                
                for i in range(n_points_axis):
                    for j in range(n_points_axis):
                        values[j,i] = qaoa.evaluate_circuit(params([axes[0][i], axes[1][j]]))['cost']
            elif len(ranges_to_use) == 3:
                for i in range(n_points_axis):
                    for j in range(n_points_axis):
                        for k in range(n_points_axis):
                            values[k,j,i] = qaoa.evaluate_circuit(params([axes[0][i], axes[1][j], axes[2][k]]))['cost']
            else:
                raise ValueError("Bug in the tests: The number of ranges must be 1, 2 or 3.")

            assert benchmark.ranges == ranges, "The ranges saved are not correct. Inputs: n_points_axis={}, ranges={}".format(n_points_axis, ranges)
            assert benchmark.values.shape == values.shape, "The shape of the values matrix is not correct. Inputs: n_points_axis={}, ranges={}".format(n_points_axis, ranges)
            assert all(benchmark.values.flatten() == values.flatten()), "The values matrix is not correct. Inputs: n_points_axis={}, ranges={}".format(n_points_axis, ranges)

            if run_reference:
                assert benchmark.ranges_reference == ranges, "The ranges saved for the reference are not correct. Inputs: n_points_axis={}, ranges={}, run_reference={}".format(n_points_axis, ranges, run_reference)
                assert benchmark.values_reference.shape == values.shape, "The shape of the values matrix of the reference is not correct. Inputs: n_points_axis={}, ranges={}, run_reference={}".format(n_points_axis, ranges, run_reference)
                assert benchmark.difference.shape == values.shape, "The difference matrix should have the same shape as the values matrix. Inputs: n_points_axis={}, ranges={}, run_reference={}".format(n_points_axis, ranges, run_reference)
                assert isinstance(benchmark.difference_mean, float), "The difference mean should be a float. Inputs: n_points_axis={}, ranges={}, run_reference={}".format(n_points_axis, ranges, run_reference)
            else:
                assert benchmark.values_reference is None, "There should be no reference values, since run_reference=False. Inputs: n_points_axis={}, ranges={}, run_reference={}".format(n_points_axis, ranges, run_reference)
                assert benchmark.ranges_reference is None, "There should be no reference ranges, since run_reference=False. Inputs: n_points_axis={}, ranges={}, run_reference={}".format(n_points_axis, ranges, run_reference)

                error = False
                try:
                    benchmark.difference
                except:
                    error = True
                assert error, "There should be no difference matrix, since run_reference=False. An error should be raised. Inputs: n_points_axis={}, ranges={}, run_reference={}".format(n_points_axis, ranges, run_reference)

        def test_values_1D(self):
             
            # standard case
            qaoa = QAOA()
            qaoa.compile(QUBO.random_instance(5))
            self.__compare_values_benchmark(qaoa, n_points_axis=10, ranges=[(-24.78, 65.97), (356.467,)])

            # extended parameters
            qaoa = QAOA()
            qaoa.set_circuit_properties(param_type='extended')
            qaoa.compile(QUBO(n=5, terms=[[0,1], [1,2], [2,3], [3,4], [4,2], [4,0]], weights=[1,2,1,2,1,2]))
            self.__compare_values_benchmark(qaoa, n_points_axis=10, ranges=[
                (-24.78, ), (65.97,) , (356.467, ), (-24.78, ), (65.97,), (0.03, 0.031), (356.467, ), (2,), (-24.78, ), (65.97,) , (356.467, ), ])

        def test_values_2D(self):
                
            # standard case
            qaoa = QAOA()
            qaoa.compile(QUBO.random_instance(5))
            self.__compare_values_benchmark(qaoa, n_points_axis=4, ranges=[(-1.43, 2.78), (12.3,25.68)])

            # p=2
            qaoa = QAOA()
            qaoa.set_circuit_properties(p=2)
            qaoa.compile(QUBO.random_instance(5))
            self.__compare_values_benchmark(qaoa, n_points_axis=4, ranges=[(-1.43, 2.78), (12.3,), (0.5, 1.5), (1,)], run_reference=True)

        def test_values_3D(self):
                    
            # standard case
            qaoa = QAOA()
            qaoa.set_circuit_properties(p=2)
            qaoa.compile(QUBO.random_instance(5))
            self.__compare_values_benchmark(qaoa, n_points_axis=4, ranges=[(-1.43, 2.78), (12.3,25.68), (0.5, 1.5), (1,)])

            # p=3
            qaoa = QAOA()
            qaoa.set_circuit_properties(p=3)
            qaoa.compile(QUBO.random_instance(5))
            self.__compare_values_benchmark(qaoa, n_points_axis=4, ranges=[(-1.43, 2.78), (245,), (0.5, 1.5), (1,), (0.5, 1.5), (19,)], run_reference=True)

        def test_difference(self):

            # device shot-based
            qaoa = QAOA()
            qaoa.set_device(create_device(location="local", name="qiskit.shot_simulator"))
            qaoa.compile(QUBO.random_instance(5))

            benchmark = QAOABenchmark(qaoa)
            benchmark.run(n_points_axis=4, ranges=[(0,np.pi), (-5,9)])

            assert np.abs(benchmark.difference_mean) > 0, "The difference mean should be non-zero."
            assert np.all(benchmark.difference == benchmark.values - benchmark.values_reference), "The difference matrix should be the difference between the values and the reference values."

            # test raise errors if not run main or run reference
            qaoa = QAOA()
            qaoa.compile(QUBO.random_instance(5))
            benchmark = QAOABenchmark(qaoa)

            error = False
            try:
                benchmark.difference
            except:
                error = True
            assert error, "An error should be raised if the difference is called before running the benchmark."

            benchmark.run(n_points_axis=4, ranges=[(-np.pi, 23), (3, 4)], run_reference=True, run_main=False)
            error = False
            try:
                benchmark.difference
            except:
                error = True
            assert error, "An error should be raised if the difference is called before running the benchmark."

            qaoa = QAOA()
            qaoa.compile(QUBO.random_instance(5))
            benchmark = QAOABenchmark(qaoa)

            benchmark.run(n_points_axis=4, ranges=[(-np.pi, 23), (3, 4)], run_reference=False, run_main=True)
            error = False
            try:
                benchmark.difference
            except:
                error = True
            assert error, "An error should be raised if the difference is called before running the benchmark."
            
        def test_input_checks(self):
            
            qaoa = QAOA()
            qaoa.compile(QUBO.random_instance(5))
            benchmark = QAOABenchmark(qaoa)

            def there_is_an_error(function, **kwargs):
                error = False
                try:
                    function(**kwargs)
                except:
                    error = True
                return error
            
            assert there_is_an_error(benchmark.run, ranges=[(1,2), (1,3)],), "An error should be raised when n_points_axis is not passed."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ), "An error should be raised when ranges is not passed."
            #todo: checks for run_reference=True and run_main=False
            assert there_is_an_error(benchmark.run, n_points_axis=4.5, ranges=[(1,2), (1,3)],), "An error should be raised when n_points_axis is not an integer."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=56,), "An error should be raised when ranges is not a list."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,2),]), "An error when len(ranges) != len(qaoa.circuit_parameters)."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,2),(3,), (34,)]), "An error when len(ranges) != len(qaoa.circuit_parameters)."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,2), 'yes'],), "An error should be raised when ranges is not a list of tuples."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,2), (1,3,4)],), "An error should be raised when ranges is not a list of tuples of length 2."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,), (1,)],), "An error should be raised when ranges all ranges are tuples of length 1."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,2), (1,3)], run_main='fhn'), "An error should be raised when run_main or run_reference are not booleans."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,2), (1,3)], run_reference='fhn'), "An error should be raised when run_main or run_reference are not booleans."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,2), (1,3)], plot='fhn', plot_reference=True), "An error should be raised when plot or plot_reference are not booleans."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,2), (1,3)], plot=True, plot_reference='fhn'), "An error should be raised when plot or plot_reference are not booleans."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,2), (1,2)], run_main=False, run_reference=False), "An error should be raised if nor run_main nor run_reference are True."
            assert there_is_an_error(benchmark.run, n_points_axis=4, ranges=[(1,2), (1,3)], plot_options='fhn'), "An error should be raised when plot_options is not a dictionary."

        def test_plot(self):
            
            # test standard plot
            qaoa = QAOA()
            qaoa.compile(QUBO.random_instance(5))
            benchmark = QAOABenchmark(qaoa)

            #2D
            benchmark.run(n_points_axis=4, ranges=[(0,np.pi), (-5,9)])
            benchmark.plot()
            benchmark.plot(main=False, reference=True)
            benchmark.plot(main=False, difference=True)
            benchmark.plot(main=True, reference=True, difference=True)
            benchmark.plot(main=False, reference=True, difference=True)
            benchmark.plot(main=True, reference=False, difference=True)
            benchmark.plot(main=True, reference=True, difference=False)

            #1D
            benchmark.run(n_points_axis=4, ranges=[(0,np.pi), (1,)])
            benchmark.plot()
            benchmark.plot(main=False, reference=True)
            benchmark.plot(main=False, difference=True)
            benchmark.plot(main=True, reference=True, difference=True)
            benchmark.plot(main=False, reference=True, difference=True)
            benchmark.plot(main=True, reference=False, difference=True)
            benchmark.plot(main=True, reference=True, difference=False)
            benchmark.plot(main=True, reference=True, difference=True, one_plot=True)
            benchmark.plot(main=False, reference=True, difference=True, one_plot=True)
            benchmark.plot(main=True, reference=False, difference=True, one_plot=True)
            benchmark.plot(main=True, reference=True, difference=False, one_plot=True)

        def test_plot_input_checks(self):

            qaoa = QAOA()
            qaoa.compile(QUBO.random_instance(5))
            benchmark = QAOABenchmark(qaoa)

            def there_is_an_error(function, **kwargs):
                error = False
                try:
                    function(**kwargs)
                except:
                    error = True
                return error

            benchmark.run(n_points_axis=4, ranges=[(0,np.pi), (-5,9)])
            assert there_is_an_error(benchmark.plot, ax=''), "An error should be raised when ax is not a matplotlib Axes"
            assert there_is_an_error(benchmark.plot, labels=''), "An error should be raised when labels is not a list of two strings"
            assert there_is_an_error(benchmark.plot, labels=['']), "An error should be raised when labels is not a list of two strings"
            assert there_is_an_error(benchmark.plot, labels=['','','']), "An error should be raised when labels is not a list of two strings"
            assert there_is_an_error(benchmark.plot, main=''), "An error should be raised when main is not a boolean"
            assert there_is_an_error(benchmark.plot, reference=''), "An error should be raised when reference is not a boolean"
            assert there_is_an_error(benchmark.plot, difference=''), "An error should be raised when difference is not a boolean"
            assert there_is_an_error(benchmark.plot, plot_options=''), "An error should be raised when plot_options is not a dict"
            assert there_is_an_error(benchmark.plot, main=True, reference=True, difference=True, one_plot=True), "An error should be raised when one_plot is True and more than one plot is requested when 2D"

            # plot without values
            # 3D