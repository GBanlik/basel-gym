from collections import deque
from collections import defaultdict
from enum import auto, Enum, unique
from typing import Deque, Dict

import numpy as np

class SimulationMonitorBase(object):
    """ Monitors MonteCarloSimulator instances by keeping track of its statistics.
    """

    @unique
    class RecordBaseCategory(Enum):
        OBSERVATIONS = auto(),

    def __init__(self, config: dict):
        self._generic_records: Dict = defaultdict(deque)
        self.preConfigure(config)

    def preConfigure(self, config={}) -> None:
        '''Pre-configure the monitor instance.
        
        The function allows for prematurely initializing variables on object construction.
        Useful for instance, to pre-allocate space based on the number of simulations.

        Parameters
        ----------
        config : dict, optional
            Configuration to be used by the Type[SimulationMonitorBase] to pre-configure itself.
        
        '''
        obs_config = config.get("default_records", None)

        if obs_config:
            initial_observation_dims = obs_config.get("record_shape", 0)
            self._generic_records[SimulationMonitorBase.RecordBaseCategory.OBSERVATIONS] = \
                np.zeros(initial_observation_dims)
        else:
            raise ValueError(self.__class__.__name__, ":__init__ Missing configuration for ", "default_records")
    
    def addRecord(self, category_key: str, record: np.array, record_key: int, flush: bool = False) -> None:
        '''Inserts a new record for the specified key.
        
        Parameters
        ----------
        category_key : str
            The category under which the record falls.
        record : np.ndarray
            An array contained the records.
        flush : bool
            Should the previous records for the specified category be flushed.
        
        '''
        if(category_key is None):
            raise ValueError("SimulationMonitorBase:addRecord Invalid category_key.")

        if(record is None):
            raise ValueError("SimulationMonitorBase:addRecord Invalid record.")

        record_book: np.array = self._generic_records[category_key]

        if(flush):
            record_book.fill(0)
 
        record_book[record_key] = record

        return record_book[-1]

    def record(self, record_category: str = None) -> np.array:
        '''Returns a record based on the provided category, if one is provided.

        Note: The method does not create a copy of the record hence, any modifications made to the object
        will be reflected in the monitored record.
        
        Parameters
        ----------
        record_category : str, optionals
            The record category to search for, defaults to None.
        
        Returns
        -------
        np.array
            The observation array for the specified details. If no detail is provided, all the records are returned.
        '''
        
        return self._generic_records if record_category is None else self._generic_records[record_category]

    def flush(self, category_key: str = None) -> None:
        '''Flushes the records container for the specified category key. If none is provided,
        the entire container is cleared.
        
        Parameters
        ----------
        category_key : str, optional
            The record category to be flushed.
        
        '''
        (self._generic_records if category_key is None else self._generic_records[category_key]).clear()

    
    def dump(self, out_name: str, category_key, delimiter:str = ',') -> None:
        """
        Dumps the record for the specified category into a file.
        Note: currently only supports individual category dumping.
        
        Parameters
        ----------
        out_name : str
            The output file name including extension, e.g. 'file.csv'
        category_key : str, optional
            The record's category to be saved
        delimiter : str, optional
            The per-item delimiter, by default ','
        
        Returns
        -------
        None
        """
        if not out_name:
            raise ValueError(self.__class__.__name__, ":dump invalid output file name.")
        
        if not category_key or category_key is None:
            raise ValueError(self.__class__.__name__, ":dump category_key (", category_key ,") not found.")

        records = self._generic_records[category_key]

        if not records is None:
            records.savetxt(out_name, records, delimiter)
        else:
            raise ValueError(self.__class__.__name__, ":dump category_key (", category_key ,") not found.")
