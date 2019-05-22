import unittest

from simulator.simulation_distributions import DiscreteSimulationDistribution
from utils.utils_distribution import convertToRowMajor
import numpy as np


class TestDistributions(unittest.TestCase):
    def test_discrete_function(self):
        discrete_dist: np.ndarray = np.ndarray(np.arange(0,9)).reshape(3,3)
        dist: DiscreteSimulationDistribution = DiscreteSimulationDistribution()
        dist.distributionFunction = discrete_dist

        # convertToRowMajor takes observations row-wise
        # observations = [(0, 2), 
        #                 (1,2)]
        lookup = np.array([[0,2],[1,2]])
        #row-major algorithm
        indices: np.ndarray = convertToRowMajor(lookup , discrete_dist.shape)

        self.assertEqual(dist.getAction(indices), [2, 5])

    def test_distribution_arguments(self):
        distribution_function = None
        dist = DiscreteSimulationDistribution()

        with self.assertRaises(AssertionError):
            dist.distributionFunction = distribution_function
    
    def test_continuous_function(self):
        raise NotImplementedError