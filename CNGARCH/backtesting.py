from typing import Union
import numpy as np
from .cngarch import *
import multiprocessing as mp

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
        corr_matrix = np.corrcoef(Real[:,index], model_forecast[:,index])
        corr = corr_matrix[0,1]
        model_Rsquare = corr**2
        # corr_matrix = np.corrcoef(x, bench_forecast)
        # corr = corr_matrix[0,1]
        # bench_Rsquare = corr**2

        beta = np.linalg.lstsq(np.reshape(Real[:,index],(-1,1)), np.reshape(model_forecast[:,index],(-1,1)),rcond=None)[0]
        yhat = np.matmul(np.reshape(Real[:,index],(-1,1)),beta)
        SS_Residual = np.sum((model_forecast[:,index]-yhat)**2)       
        SS_Total = np.sum((model_forecast[:,index]-np.mean(model_forecast[:,index]))**2)     
        r_squared = 1 - (float(SS_Residual))/SS_Total
        model_ad_r_squared = 1 - (1-r_squared)*(np.size(Real,0)-1)/(np.size(Real,0)-np.size(model.x,0)-1)

        # beta = np.linalg.lstsq(np.reshape(x,(-1,1)),bench_forecast,rcond=None)[0]
        # yhat = np.matmul(np.reshape(x,(-1,1)),beta)
        # SS_Residual = sum((bench_forecast-yhat)**2)       
        # SS_Total = sum((bench_forecast-np.mean(bench_forecast))**2)     
        # bench_ad_r_squared = 1 - (float(SS_Residual))/SS_Total
        # bench_ad_r_squared = 1 - (1-r_squared)*(len(x)-1)/(len(x)-len(aggregatesampling)-1)


        model_RMSE = np.sqrt(np.mean((Real[:,index]-model_forecast[:,index])**2))
        # bench_RMSE = np.sqrt(np.mean((x-bench_forecast)**2))

        model_AME = np.mean(np.abs(Real[:,index]-model_forecast[:,index]))
        # bench_AME = np.mean(np.abs(x-bench_forecast))

        output[ihor] = {'model':{'Rsquare':model_Rsquare, 'RMSE':model_RMSE, 'AME':model_AME, 'forecast':model_forecast[:,index], 'AdjRsquare':model_ad_r_squared}}
    # package a nice dict 

    pausebeforereturn = 1

    return output





