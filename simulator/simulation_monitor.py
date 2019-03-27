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

    def __init__(self):
        self._generic_records: Dict = defaultdict(deque)

    def addRecord(self, category_key: str, record: np.ndarray, flush: bool = False) -> np.ndarray:
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

        record_book: Deque = self._generic_records[category_key]

        if(flush):
            record_book.clear()
 
        record_book.append(record)

        return record_book[-1]

    @property
    def record(self, record_category: str = None) -> np.ndarray:
        '''Returns a record based on the provided category, if one is provided.
        
        Parameters
        ----------
        record_category : str, optionals
            The record category to search for, defaults to None.
        
        Returns
        -------
        np.ndarray
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

class BaselSimulationMonitor(SimulationMonitorBase):

    @unique
    class BaselRecordCategory(Enum):
        # Last 60-day reported VaRs
        DISCLOSURE = auto(),
        # Exceedence map per simulation
        EXCEEDENCES = auto(),
        # kMultiplier map per year per simulationID
        KMULTIPLIERS = auto(),
        # Bankruptcy map per year per simulationID
        BANKRUPTCY = auto(),
    
    def __init__(self):
        super().__init__()

    def addRecord(self, category_key: str, record: np.ndarray, flush: bool = False) -> np.ndarray:
        return super().addRecord(category_key, record, flush)

    @property
    def disclosure_history(self) -> Deque:
        return self._generic_records[BaselSimulationMonitor.BaselRecordCategory.DISCLOSURE]