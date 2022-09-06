from typing import Union
import numpy as np
from .cngarch import *

WINDOW_TYPE_ROLLING = 0
WINDOW_TYPE_GROWING = 0

TOTALREALIZEDVARIANCE = 0
PEAKDREALIZEDVARIANCE = 1



def backtesting(
    model:gmodel, Returns:np.ndarray, windowtype:int=WINDOW_TYPE_ROLLING, estimatewindowsize:int=2000, 
    forecasthorizon:Union[int, np.ndarray]=1, longerhorizontype:int=TOTALREALIZEDVARIANCE)->dict:
    """
        This function will backtest the gmodel forecast and return the detailed results in a big dict
        The model needs to be initialized with current parameters, thetas if 
    """



