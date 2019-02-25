from typing import Callable, Dict
from collections import deque

import numpy as np

from simulation_distributions import SimulationDistribution

class SimulationMonitor(object):
    """ Monitors MonteCarloSimulator instances by keeping track of its statistics.
    """
    
    def __init__(self):
        self.__disclosure_history = deque()

    def addRecord(self, record: np.ndarray):
        if(record is None):
            raise ValueError

        if len(self.__disclosure_history) == 60:
            self.__disclosure_history.popleft()
            self.__disclosure_history.append(record[0])

    @property
    def disclosure_history(self):
        return self.__disclosure_history

class SimulationProfile(object):
    '''Simulation profiles describe the underlying dynamics of a simulation.
    
    Parameters
    ----------
    dist : SimulationDistribution
        The underlying profile's distribution. Must inherit from SimulationDistribution.
    monitor : SimulationMonitor
        A SimulationMonitor instance for the profile. Should one not be passed, it is automatically created. 

    Raises
    ------
    ValueError
        For invalid inputs' type or value.
    '''

    def __init__(self, dist: SimulationDistribution, monitor: SimulationMonitor = SimulationMonitor()):
        if(not self._validateArgs(dist, monitor)):
            raise ValueError("Invalid supplied arguments.")

        self.__dist = dist
        self.__monitor = monitor

    @property
    def distribution(self):
        return self.__dist

    @property
    def monitor(self):
        return self.__monitor

    def _validateArgs(self, dist: SimulationDistribution, monitor: SimulationMonitor):
        return not dist is None and not monitor is None and issubclass(dict, SimulationDistribution) and issubclass(monitor, SimulationMonitor)

class MonteCarloSimulator(object):
    '''Runs Monte Carlo simulations on SimulationProfile instances. 

    Parameters
    ----------
    config : dict
        Configuration dictionary to override default settings.
    '''

    def __init__(self, config = {}):
        self.__sim_num = config.get["simulation_number", 3000]
        self.__num_trading_days = config.get["trading_days", 250]
        self.__num_years_per_sim = config.get["simulation_years", 30]

        # Holds simulation profiles, {name, SimulationProfile}
        self.__simulation_profiles: Dict[str, SimulationProfile] = {}

    def startSimulation(self) -> bool:
        if not self._validate():
            return False

        if len(self.__simulation_profiles) == 0:
            return False

        for self.sim_num in range(self.simulations_number):

            for day in range(self.__num_trading_days, 0, -1):
                daily_return = np.random.rand(self.__num_years_per_sim)

                for profile_name, profile in self.__simulation_profiles.items():
                    pass

        return True

    def addSimulationProfile(self, name: str, sim_dist: SimulationDistribution, monitor: SimulationMonitor = SimulationMonitor()) -> bool:
        if not issubclass(sim_dist, SimulationDistribution):
            raise ValueError("Invalid distribution class.")

        if not self.__simulation_profiles.get(name):
            self.__simulation_profiles[name] = SimulationProfile(sim_dist, monitor)
        else:
            print("MonteCarloSimulator: Failed to add distribution simulation profile ", name, ".")
        
    def _validate(self) -> bool:
        if(self.__simulation_profiles is None):
            raise ValueError("Unable to locate the disclosure distribution.")

        if(not self.simulations_number > 0):
            raise ValueError("Missing simulation number.")

        return True

    @property
    def simulations_number(self) -> int:
        return self.__sim_num

    @simulations_number.setter
    def simulations_number(self, amount: int) -> None:
        if (amount <= 0):
            raise ValueError("The amount of simulations to perform has to be positive.")
        
        self.__sim_num = amount