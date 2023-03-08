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
            

