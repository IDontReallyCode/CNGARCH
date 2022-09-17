from typing import Union
import numpy as np
from .cngarch import *
import multiprocessing as mp
from sklearn import metrics

WINDOW_TYPE_ROLLING = 0
WINDOW_TYPE_GROWING = 1

TOTALREALIZEDVARIANCE = 0
PEAKDREALIZEDVARIANCE = 1

ESTIMATE_METHOD_ONE_THETA = 0
ESTIMATE_METHOD_PARALLEL_ONCE = 1
ESTIMATE_METHOD_PARALLEL_ALL = 2

def backtesting(
    model:gmodel, Returns:np.ndarray, Real:np.ndarray, windowtype:int=WINDOW_TYPE_ROLLING, estimatewindowsize:int=2000, 
    estimatemethod:int=ESTIMATE_METHOD_PARALLEL_ONCE, Ncores:int=4, estpool=None, 
    forecasthorizon:Union[int, np.ndarray]=1, longerhorizontype:int=TOTALREALIZEDVARIANCE)->dict:
    """
        This function will backtest the gmodel forecast and return the detailed results in a big dict
        The model needs to be initialized with current parameters, thetas if 
    """

    if estimatemethod in [ESTIMATE_METHOD_PARALLEL_ONCE, ESTIMATE_METHOD_PARALLEL_ALL]:
        if estpool is None:
            estpool = mp.Pool(Ncores)

    totalnbdays = np.size(Returns,0)
    output = {}
    maxforecast = np.max(forecasthorizon)
    nforecast = len(forecasthorizon)

    if estimatemethod==ESTIMATE_METHOD_PARALLEL_ONCE:
        model.R = Returns[:(estimatewindowsize)]
        model.parallel(estpool=estpool)

    model_forecast = np.zeros((totalnbdays-estimatewindowsize-maxforecast+1,nforecast))
    bench_forecast = np.zeros((totalnbdays-estimatewindowsize-maxforecast+1,nforecast))

    for index in range(totalnbdays-estimatewindowsize-maxforecast+1):
        if windowtype==WINDOW_TYPE_ROLLING:
            model.R = Returns[index:(estimatewindowsize+index)]
        elif windowtype==WINDOW_TYPE_GROWING:
            model.R = Returns[:(estimatewindowsize+index)]
        else:
            raise Exception('This is an invalid window type, please use the constants.')
        
        if estimatemethod==ESTIMATE_METHOD_PARALLEL_ALL:
            model.parallel(estpool=estpool)
        else:
            model.estimate()

        model.forecast(maxforecast)

        for ih, ihor in enumerate(forecasthorizon):
            model_forecast[index,ih] = np.sum(model.vforecast[:ihor])

        pausehere=1

    # compute the R-square, RMSE, and MAE

    for index, ihor in enumerate(forecasthorizon):
        # look at HAR code to get the metrics.
        model_Rsquare = metrics.r2_score(Real[:,index], model_forecast[:,index])
        model_RMSE = np.sqrt(metrics.mean_squared_error(Real[:,index], model_forecast[:,index]))
        model_evs = metrics.explained_variance_score(Real[:,index], model_forecast[:,index])
        model_mae = metrics.mean_absolute_error(Real[:,index], model_forecast[:,index])
        model_mape = metrics.mean_absolute_percentage_error(Real[:,index], model_forecast[:,index])


        output[ihor] = {'model':{'Rsquare':model_Rsquare, 'RMSE':model_RMSE, 'explainedvariancescore':model_evs, 'forecast':model_forecast[:,index], 'mae':model_mae, 'mape':model_mape}}
    # package a nice dict 

    pausebeforereturn = 1

    return output





