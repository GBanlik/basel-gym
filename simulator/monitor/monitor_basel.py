from enum import auto, Enum, unique

import numpy as np

from simulator.monitor.monitor_base import SimulationMonitorBase

class BaselSimulationMonitor(SimulationMonitorBase):

    @unique
    class BaselRecordCategory(Enum):
        # Last 60-day reported VaRs
        DISCLOSURE = auto(),
        # Exceedence map per simulation
        EXCEEDENCES = auto(),
        # kMultiplier map per year per simulationID
        KMULTIPLIERS_VALUE = auto(),
        # kMultiplier's index map per year per simulationID
        KMULTIPLIERS_INDECES = auto(),
        # Bankruptcy map per year per simulationID
        BANKRUPTCY = auto(),
        # The yearly average disclosed value
        DISCLOSURE_YEAR_MEAN = auto(),
        # The daily return as provided by the simulator, per year
        RETURN_DAILY = auto(),
        # Return map per year per simulationID,
        RETURN_YEAR = auto(),
    
    def __init__(self, config: dict):
        super().__init__(config)


    def preConfigure(self, config={}) -> None:
        '''Pre-configure the monitor instance.
        
        The function allows for prematurely initializing variables on object construction.
        Useful for instance, to pre-allocate space based on the number of simulations.

        Parameters
        ----------
        config : dict, optional
            Configuration to be used by the Type[SimulationMonitorBase] to pre-configure itself.
        
        '''
        super().preConfigure(config)

        obs_config = config.get("basel_records", None)

        #TODO Add appending for when the initial size isn't specified or pass it directly
        #TODO from the simulation profile

        if obs_config:
            observation_dims = obs_config.get("record_shape", 0)
            # expand the dim's 0 dimension (simulations) to accomodate the additional revised multiplier
            obs_dims_extended = (observation_dims[0] +1, ) + observation_dims[1:]
            daily_disclosure_dims = obs_config.get("daily_disclosure_record_shape", 0)
    
            # pre-allocate space for yearly records
            self._generic_records[BaselSimulationMonitor.BaselRecordCategory.EXCEEDENCES] = \
                np.zeros(observation_dims)
            self._generic_records[BaselSimulationMonitor.BaselRecordCategory.BANKRUPTCY] = \
                np.zeros(observation_dims)
            self._generic_records[BaselSimulationMonitor.BaselRecordCategory.DISCLOSURE_YEAR_MEAN] = \
                np.zeros(observation_dims)
            # extended to accomodate reviewed following year
            self._generic_records[BaselSimulationMonitor.BaselRecordCategory.KMULTIPLIERS_VALUE] = \
                np.zeros(obs_dims_extended)
            self._generic_records[BaselSimulationMonitor.BaselRecordCategory.KMULTIPLIERS_INDECES] = \
                np.zeros(obs_dims_extended)
                
            # pre-allocate space for daily records for yearly averaging purposes
            self._generic_records[BaselSimulationMonitor.BaselRecordCategory.DISCLOSURE] = \
                np.zeros(daily_disclosure_dims)
            self._generic_records[BaselSimulationMonitor.BaselRecordCategory.RETURN_DAILY] = \
                np.zeros(daily_disclosure_dims)
            
            # yearly statistics
            self._generic_records[BaselSimulationMonitor.BaselRecordCategory.RETURN_YEAR] = \
                np.zeros(observation_dims)
        else:
            raise ValueError(self.__class__.__name__, ":__init__ Missing configuration for ", "basel_records")
    
    def addRecord(self, category_key: str, record: np.array, record_key: int, flush: bool = False) -> None:
        if(category_key == BaselSimulationMonitor.BaselRecordCategory.DISCLOSURE):

            disclosure_history = self.disclosure_history
            if(not disclosure_history is None):
                if(flush):
                    disclosure_history.clear()

                self.disclosure_history[record_key] = record

            return
        
        super().addRecord(category_key, record, record_key, flush)

    @property
    def disclosure_history(self) -> np.array:
        return self._generic_records[BaselSimulationMonitor.BaselRecordCategory.DISCLOSURE]