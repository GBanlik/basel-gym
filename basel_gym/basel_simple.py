import numpy as np

from math import sqrt

from basel_base import BaselBase
from basel_base import EventTransition

from gym import spaces

from scipy.stats import norm
from scipy.special import ndtr

class BaselSimple(BaselBase):
    '''
    Class for creating a Seixas' Basel environment.
    Action space: 3000
    Observation space: [250, 12]
    '''
    def __init__(self, config):
        super().__init__(config)

        self.action_space = spaces.Discrete(3000)
        self.observation_space = spaces.MultiDiscrete([250, (self._EC_Max + 1), len(self._kMultipliersListing)])

    def _getActionValue(self, action: float) -> float:
        return action * 0.001

    def _computeReward(self) -> float:
        return -1 if self._isBankrupt() else 0.00001 * (
                self.action_value * self._kMultipliersListing[self.state[2]] * self._normalVaR10)
    
    def _isBankrupt(self) -> bool:
        return self.state[2] == (len(self._kMultipliersListing) - 1)


    def reset(self):
        current_kmul_index = self.defaultMultiplierIndex if self.defaultMultiplierIndex != None else self.np_random.randint(
            0, len(self._kMultipliersListing) - 1)

        initialECs = self.np_random.randint(0, self._EC_Max - 1) if self._useRandomEC else 0
        self.state = (250 - initialECs, initialECs , current_kmul_index)
        
        return self._get_obs()

    def _generateProbabilities(self) -> (float, float, float, float):
        reportedValue = self.action_value * self._normalVaR
        # 400, not having an exceedence; 100+i, not going bankrupt with one day left to backtesting
        probNoEC, probNB = ndtr([reportedValue,  reportedValue * self._kMultipliersListing[self.state[2]] * sqrt(10)])
        # 200+i, having an EC and going bankrupt
        probBC = 1 - probNB
        # 300+i, having an EC and not going BC
        probECNoBC = 1 - probNoEC - (1 - probNB)

        return probNoEC, probNB, probBC, probECNoBC

    def _getTransitionedEvent(self, isEndState: bool = False) -> EventTransition:
        if (self._isBankrupt()):
            return EventTransition.NONE

        probNoEC, probNB, probBC, probECNoBC = self._generateProbabilities()
        rndProb: np.ndarray[float] = self._rndGenerator.fetch()
        
        transition_event: EventTransition = None

        if (isEndState):
            ec_count: int = self.state[0]

            if (ec_count <= 4):
                transition_event = EventTransition.NONE if rndProb < probNB else EventTransition.BANKRUPTCY
                return transition_event

        if (rndProb < probNoEC):  # 400
            transition_event = EventTransition.NONE
        elif (rndProb >= probNoEC and rndProb < (probNoEC + probECNoBC)):  # 300
            transition_event = EventTransition.EXCEEDANCE
        else:  # 200
            transition_event = EventTransition.BANKRUPTCY

        return transition_event
