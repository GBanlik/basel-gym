import numpy as np

from math import sqrt

from typing import Deque

#from scipy.special import ndtr
from scipy.stats.distributions import norm

from simulator.simulation_distributions import SimulationDistributionBase
from simulator.simulation_monitor import SimulationMonitorBase
from simulator.simulation_monitor import BaselSimulationMonitor
from utils.utils_decorators import inputDecorators

class SimulationProfileBase(object):
    '''Simulation profiles describe the underlying dynamics of a simulation.
    
    Parameters
    ----------
    dist : SimulationDistributionBase
        The underlying profile's distribution. Must inherit from SimulationDistributionBase.
    monitor : SimulationMonitorBase
        A SimulationMonitorBase instance for the profile. Should one not be passed.
        Defaults to DefaultSimulationMonitor. 

    Raises
    ------
    ValueError
        For invalid inputs' type or value.
    '''

    def __init__(self, dist: SimulationDistributionBase = SimulationDistributionBase(), monitor: SimulationMonitorBase = SimulationMonitorBase(), config: dict = {}):
        self._dist: SimulationDistributionBase = dist
        self._monitor: SimulationMonitorBase = monitor

    def performTransition(self, daily_return: np.ndarray, observation: np.ndarray) -> None:
        raise NotImplementedError

    @property
    def distribution(self):
        return self._dist

    @distribution.setter
    @inputDecorators.non_null
    def distribution(self, new_dist):
        self._dist = new_dist

    @property
    def monitor(self):
        return self._monitor

    @monitor.setter
    @inputDecorators.non_null
    def monitor(self, new_monitor: SimulationMonitorBase):
        self._monitor = new_monitor

    def validate(self):
        return not  self._dist is None and not  self._monitor is None and \
             issubclass(self._dist.__class__, SimulationDistributionBase) and issubclass( self._monitor.__class__, SimulationMonitorBase)

class BaselSimulationProfile(SimulationProfileBase):
    def __init__(self, dist: SimulationDistributionBase = SimulationDistributionBase(), monitor: SimulationMonitorBase = SimulationMonitorBase(), config: dict = {}):
        super().__init__(dist, monitor, config)

        self._confidence_level: float = config.get("confidence_level", 0.99)
        self._normal_mean: float = config.get("normal_mean", 0)
        self._normal_stddev: float = config.get("normal_stddev", 1)
        self._normal_var: float = config.get("normal_var", norm.ppf(self._confidence_level, self._normal_mean, self._normal_stddev))
        self._normal_var10: float = config.get("normal_var10", -self._normal_var * sqrt(10))

    def performTransition(self, daily_return: np.ndarray, observation: np.ndarray) -> None:
        pass
