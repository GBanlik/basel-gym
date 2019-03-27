from typing import Callable
from utils.utils_decorators import inputDecorators

import numpy as np

class SimulationDistributionBase(object):
    '''Base class for a custom distribution to be used by MonteCarloSimulator instances.
    '''
    
    def __init__(self, config={}):
        self._dist: np.ndarray = config.get("distribution_function", None)

    def validate(self) -> bool:
        return not self._dist is None

    @property
    def distributionFunction(self) -> np.ndarray:
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
    def distributionFunction(self, distribution) -> None:
        raise NotImplementedError

    def getAction(self, observation: np.ndarray) -> float:
        raise NotImplementedError
    

class DiscreteSimulationDistribution(SimulationDistributionBase):

    def __init__(self, config={}):
        super().__init__(config)

    @property
    def distributionFunction(self) -> np.ndarray:
        return super().distributionFunction

    @distributionFunction.setter 
    @inputDecorators.non_null
    def distributionFunction(self, distribution: np.ndarray) -> None:
        self._dist = distribution.ravel()

    @inputDecorators.non_null
    def getAction(self, observation: np.ndarray) -> float:
        return self.distributionFunction[observation]

class ContinuousSimulationDistribution(SimulationDistributionBase):
    def __init__(self, config={}):
        super().__init__(config)

    @property
    def distributionFunction(self) -> np.ndarray:
        return super().distributionFunction

    @distributionFunction.setter 
    @inputDecorators.non_null
    def distributionFunction(self, distribution: Callable[[np.ndarray], float]) -> None:
        super(DiscreteSimulationDistribution, self.__class__).distributionFunction.fset(self, distribution)

    @inputDecorators.non_null
    def getAction(self, observation: np.ndarray) -> float:
        return self._dist(observation)