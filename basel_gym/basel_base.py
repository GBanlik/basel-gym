from __future__ import absolute_import
from __future__ import division

import gym
from gym.utils import seeding

import numpy as np
from scipy.stats import norm

from math import sqrt
from enum import Enum, unique

#PseudoRandomNumberQueue class
from collections import deque

@unique
class EventTransition(Enum):
    BANKRUPTCY = 1,
    EXCEEDANCE = 2,
    NONE = 0

class PseudoRandomNumberQueue(object):   
    
    '''
    Helper class to generate and retrieve multiple random numbers to and from a deque.
    @param initSize - The per-repopulation size. 
    '''
    def __init__(self, initSize):
        self.seed_size = initSize
        self.rnd_num = deque()

    def __del__(self):
        pass
    
    def repopulate(self) -> None:
        self.rnd_num.extend(np.random.rand(self.seed_size).tolist())
    
    def fetch(self) -> float:
        '''
        Pops a random number.
        If the deque is empty, initializes it based on @param initSize.
        '''
        if(not self.rnd_num):
            self.repopulate()    

        return self.rnd_num.pop()

class BaselBase(gym.Env):
    '''
    Base class for creating a Basel environment.
    Contains the basic functionality and behaviour associated with said object.
    Note however, that despite having the full behavior, it relies on overriden
    methods by its children to operate, hence, class-extension is required.

    All deriving classes must implement:
        * _getActionValue()

    '''

    def __init__(self, config):
        self._obs: np.ndarray = None
    
        # User Configuration Section
        self.defaultMultiplierIndex = config.get("initial_multiplier_index", None)
        self._useRandomEC: bool = config.get("use_random_ec", None)

        # Basel Configuration Section
        self._confidenceLevel: float = 0.99
        self._normalMean: float = 0
        self._normalStdDev: float = 1
        self._EC_Max: int = 11
        self._kMultipliersListing = np.array([3, 3.4, 3.5, 3.65, 3.75, 3.85, 4, np.inf])

        self._normalVaR: float = norm.ppf(self._confidenceLevel, self._normalMean, self._normalStdDev)
        self._normalVaR10: float = - self._normalVaR * sqrt(10)

        # Observation Variables
        self._action_value : float = 0.0
        
        # Gym Configuration Section
        self.seed()
        self.render = False
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        self._rndGenerator = PseudoRandomNumberQueue(10000000)

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.action_value = self._getActionValue(action)
        self._updateEnvironment()

        done = bool(self.state[0] == 0 or self._isBankrupt())
        reward = self._computeReward()

        return self._get_obs(), reward, done, {}

    def reset(self):
        raise NotImplementedError

    def _get_obs(self):
        return np.array(self.state)

    def _computeReward(self) -> float:
        return -1 if self._isBankrupt() else (
                self.action_value * self._kMultipliersListing[self.state[2]] * self._normalVaR10)

    def _updateEnvironment(self) -> None:
        state = self.state
        
        ttob = state[0]
        ec_number = state[1]
        k_mul = state[2]

        if (not self._isBankrupt()):
            transitioned_event = self._getTransitionedEvent(ttob == 1)

            if (transitioned_event == EventTransition.EXCEEDANCE):
                ec_number = min(ec_number + 1, self._EC_Max)
            elif (transitioned_event == EventTransition.BANKRUPTCY):
                ec_number = self._EC_Max
                k_mul = 7

        if (ec_number == self._EC_Max):
            k_mul = len(self._kMultipliersListing) - 1
        elif (ttob == 1):
            k_mul = 0 if ec_number <= 4 else ec_number - 4

        # Transition TtoB
        ttob -= 1

        self.state = (ttob, ec_number, k_mul)

    def _generateProbabilities(self) -> (float, float, float, float):
        raise NotImplementedError

    def _getTransitionedEvent(self, isEndState: bool = False) -> EventTransition:
        raise NotImplementedError

    def _isBankrupt(self) -> bool:
        return self.state[2] == (len(self._kMultipliersListing) - 1)

    def _getActionValue(self, action: float) -> float:
        raise NotImplementedError

    @property
    def defaultMultiplierIndex(self) -> int:
        return self._defaultMultiplierIndex

    @defaultMultiplierIndex.setter
    def defaultMultiplierIndex(self, mulValue: int) -> None:
        self._defaultMultiplierIndex = mulValue

class RewardScaler(gym.RewardWrapper):

    def reward(self, reward):
        raise NotImplementedError("Basel environments scale by default. "
          "To remove auto-scaling override _computeReward(self).")