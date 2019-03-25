import numpy as np

def convertToRowMajor(observations: np.ndarray, shape: np.ndarray):
    '''Converts a set of horizontally stacked observations into row-major ordering.
    
    Parameters
    ----------
    observations : np.darray
        [Horizontally stacked observations, e.g. [[0,2], [1,2]] for two observations (0,2) and (1,2)
    shape : np.ndarray
        [The shape of the distribution from which to retrieve actions]
    
    '''
    return np.ravel_multi_index(observations.T, shape, order='C')