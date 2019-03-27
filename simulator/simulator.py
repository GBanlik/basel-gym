from typing import Callable, Dict, Type

import logging

import numpy as np

from simulator.simulation_distributions import SimulationDistributionBase
from simulator.simulation_monitor import SimulationMonitorBase
from simulator.simulation_profile import SimulationProfileBase
from utils.utils_decorators import inputDecorators

logging.basicConfig(format='%(asctime)s-%(process)d-%(levelname)s-%(messages)s', level=logging.INFO)

class MonteCarloSimulator(object):
    '''Runs Monte Carlo simulations on SimulationProfileBase instances. 

    Parameters
    ----------
    config : dict
        Configuration dictionary to override default settings.
    '''

    # Note: The variables schema is shared by all instances of MonteCarloSimulator.
    # Given it is mutable, modifying on the class is the same as modifying on the instance.
    variables_schema = {
        "_trading_days_order" : ["A", "D"],
    }

    def __init__(self, config = {}):
        # Simulator Configuration
        self._simulations_number = config.get("simulation_number", 3000)
        self._num_trading_days = config.get("trading_days", 250)
        self._num_years_per_sim = config.get("simulation_years", 30)
        self._trading_days_order = config.get("day_order", "A")

        # Holds simulation profiles
        self._simulation_profiles: Dict[str, SimulationProfileBase] = {}

        # Simulator status
        self._is_running: bool = False

    def startSimulation(self) -> bool:
        if self.isRunning():
            logging.info(self.__class__.__name__, ":startSimulation already in progress.")
            return False

        if not self._validate():
            return False

        if len(self._simulation_profiles) == 0:
            return False

        for sim_num in range(self.simulations_number):
            logging.debug(self.__class__.__name__, 'Starting simulation {}'.format(sim_num))

            for day in range(self._num_trading_days, 0, -1):
                daily_return = np.random.rand(self._num_years_per_sim)

                for profile_name, profile in self._simulation_profiles.items():
                    logging.debug(self.__class__.__name__, ": Simulating profile {}".format(profile_name))

                    done = day == 1
                    profile.performTransition(daily_return, (sim_num, day, done))

        return True

    def addSimulationProfile(self, name: str, sim_profile: SimulationProfileBase) -> bool:
        if not issubclass(sim_profile.__class__, SimulationProfileBase):
            logging.error(self.__class__.__name__, ":addSimulationProfile Invalid profile class {} . ".format(sim_profile.__name__))

        if not self._simulation_profiles.get(name):
            self._simulation_profiles[name] = sim_profile
        else:
            logging.error(self.__class__.__name__, ":addSimulationProfile Duplicate simulation profile {} . ".format(name))
    
    def removeSimulationProfile(self, profile_name :str) -> SimulationProfileBase:
        return self._simulation_profiles.pop(profile_name, None)
    
    def createAndAddSimulationProfile(self, name:str, profile_class: Type[SimulationProfileBase], sim_dist: SimulationDistributionBase, monitor: SimulationMonitorBase = SimulationMonitorBase()) -> SimulationProfileBase:
        '''
        Creates a new profile and adds it to the MonteCarloSimulator instance stack.
        
        Parameters
        ----------
        name : str
            A unique profile name.
        profile_class : Type[SimulationProfileBase]
            The profile class to be instantiated.
        sim_dist : SimulationDistributionBase
            The underlying distribution function.
        monitor : SimulationMonitorBase, optional
            A monitor object derived from SimulatorMonitorBase.
        
        Returns
        -------
        SimulationProfileBase
            The created object if succesfully created, None otherwise.
        '''


        if not issubclass(profile_class, SimulationProfileBase):
            logging.error(self.__class__.__name__, ":addSimulationProfile Invalid profile class {} . ".format(sim_dist.__name__))
        
        if not issubclass(sim_dist.__class__, SimulationDistributionBase):
            logging.error(self.__class__.__name__, ":addSimulationProfile Invalid distribution class {} . ".format(sim_dist.__name__))

        if not self._simulation_profiles.get(name):
            self._simulation_profiles[name] = profile_class(sim_dist, monitor)
            return self._simulation_profiles[name]
        else:
            logging.error(self.__class__.__name__, ":addSimulationProfile Duplicate simulation profile {} . ".format(name))

    def _validate(self) -> bool:
        if(not MonteCarloSimulator.variables_schema is None):

            for config_schema_name, config_schema_list  in MonteCarloSimulator.variables_schema.items():
                configuration = getattr(self, config_schema_name, None)

                if(configuration is None or not configuration in config_schema_list):
                    raise ValueError(self.__class__.__name__, ":_vdalidate Invalid {} in {}.".format(configuration, config_schema_name))                    

        if(len(self._simulation_profiles) == 0):
            raise ValueError(self.__class__.__name__, ":_validate Unable to locate simulation profiles.")
        else:

            for profile_name, profile in self._simulation_profiles.items():
                if(not profile.validate()):
                    raise ValueError(self.__class__.__name__, ":_validate invalid profile {}.".format(profile_name))

        if(not self.simulations_number > 0):
            raise ValueError(self.__class__.__name__, ":_validate Missing simulation number.")

        return True

    @property
    def simulations_number(self) -> int:
        return self._simulations_number

    @simulations_number.setter
    @inputDecorators.non_null
    @inputDecorators.all_positive
    def simulations_number(self, amount: int) -> None:
        '''
        [Sets the number of simulations to create]
        
        Note that if a simulation is in progress, its behavior will be unchanged.
        In order to reflect the changes, the MonteCarloSimulator instance must be restarted.
        
        Parameters
        ----------
        amount : int
            The target number of simulations to be ran by the MonteCarloSimulator instance.
        
        Returns
        -------
        None
        '''

        self._simulations_number = amount

    def isRunning(self) -> bool:
        return self._is_running