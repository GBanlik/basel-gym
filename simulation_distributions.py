from typing import Callable, Tuple

import numpy as np

class SimulationDistribution(object):
    '''Base class for a custom distribution to be used by MonteCarloSimulator instances.
    '''

    def __init__(self, config={}):
        self.__dist = None

        
    def validate(self) -> bool:
        return not self.__dist is None
        
    @property
    def distributionFunction(self, dist):
        '''Retrieves the underlying distribution function for the instance.
        
        Returns
        -------
        np.nparray[]
            for DiscreteSimulationDistribution objects
        Callable[[np.ndarray[float]], float]
            for ContinuousSimulationDistribution objects
        '''

        return self.__dist

    @distributionFunction.setter 
    def distributionFunction(self, distribution, dims_vec: np.ndarray):
        raise NotImplementedError

    def getAction(self, observation: np.ndarray) -> float:
        raise NotImplementedError
    

class DiscreteSimulationDistribution(SimulationDistribution):

    def __init__(self, config={}):
        super().__init__(config)

    @SimulationDistribution.distributionFunction.setter 
    def distributionFunction(self, dist: np.ndarray):
        self.__dist = dist.tolist()

    def getAction(self, observation: np.ndarray) -> float:
        # time np.tolist()(index1, index2, ...) < np.item(index1, index2, ...) < np[index1, index2, ...] < np.[index1][index2][]...]
        return self.__dist.item(observation.ravel())

class ContinuousSimulationDistribution(SimulationDistribution):
    def __init__(self, config={}):
        super.__init__(config)

    @SimulationDistribution.distributionFunction.setter 
    def distributionFunction(self, dist: Callable[[np.ndarray], float]):
        self.__dist = dist

    def getAction(self, observation: np.ndarray) -> float:
        return self.__dist(observation)