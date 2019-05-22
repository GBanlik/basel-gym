from typing import Deque, Type

import numpy as np

from simulator.distribution.distribution_base import SimulationDistributionBase
from simulator.monitor.monitor_base import SimulationMonitorBase
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

    def __init__(self, dist: Type[SimulationDistributionBase], monitor: Type[SimulationMonitorBase], config: dict = {}):
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
