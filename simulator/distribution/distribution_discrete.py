from typing import Callable
from utils.utils_decorators import inputDecorators
from utils.utils_distribution import convertToRowMajor

import numpy as np

from simulator.distribution.distribution_base import SimulationDistributionBase

class DiscreteSimulationDistribution(SimulationDistributionBase):

    def __init__(self, config={}):
        super().__init__(config)

    @property
    def distributionFunction(self) -> np.array:
        '''Retrieves the underlying distribution function for the instance.
        
        Returns
        -------
        np.nparray
            for DiscreteSimulationDistribution objects.
        Callable[[np.ndarray], float]
            for ContinuousSimulationDistribution objects.
        '''
        return self._dist

    @distributionFunction.setter
    @inputDecorators.non_null
    def distributionFunction(self, distribution: np.array) -> None:
        super(DiscreteSimulationDistribution, self.__class__).distributionFunction.fset(self, distribution)

        self._dist_raveled = distribution.ravel()

    @inputDecorators.non_null
    def getAction(self, observation: np.ndarray) -> np.array:
        '''Retrieves action(s) for the supplied observation(s)
        Note: Multiple observations should be vertically stacked.

        Parameters
        ----------
        observation : np.ndarray
            The set of observations for which to compute an action.
        
        Returns
        -------
        np.ndarray
            An array of actions for each observation (order is preserved).
        '''
        indices = convertToRowMajor(observation, self._dist.shape)
        return self._dist_raveled[indices]