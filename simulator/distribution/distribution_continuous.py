from utils.utils_decorators import inputDecorators

import numpy as np

from simulator.distribution.distribution_base import SimulationDistributionBase

class ContinuousSimulationDistribution(SimulationDistributionBase):
    def __init__(self, config={}):
        super().__init__(config)

    @inputDecorators.non_null
    def getAction(self, observation: np.array) -> float:
        return self._dist(observation)