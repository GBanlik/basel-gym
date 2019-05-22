from typing import Callable

import numpy as np

class SimulationDistributionBase(object):
    '''Base class for a custom distribution to be used by MonteCarloSimulator instances.
    '''
    
    def __init__(self, config={}):
        self.distributionFunction = config.get("distribution_function", None)

    def validate(self) -> bool:
        return not self._dist is None

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
    def distributionFunction(self, distribution: np.array) -> None:
        self._dist = distribution

    def getAction(self, observation: np.array) -> float:
        raise NotImplementedError