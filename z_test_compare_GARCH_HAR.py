import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import HAR
import cngarch as cg
import multiprocessing as mp

def main():
    # load intraday
    ticker = 'F'
    # data = pd.read_csv(f"./intradaysample{ticker}.csv", index_col=0)
    data = pd.read_pickle(f"D:/DATA/STOCK/INTRADAY/cleaner_pkl/{ticker}_min.pkl")
    data = data.between_time(datetime.time(hour=9,minute=30), datetime.time(hour=16), include_start=True, include_end=True)

    data1d = data.resample('1D', closed='right').agg({'symbol':'last', 'volume':'sum', 
                                                            'open': 'first', 
                                                            'high': 'max', 
                                                            'low': 'min', 
                                                            'close': 'last', 'date_eod':'last'}).dropna()

    data5m = data.resample('5min', closed='right').agg({'symbol':'last', 'volume':'sum', 
                                                            'open': 'first', 
                                                            'high': 'max', 
                                                            'low': 'min', 
                                                            'close': 'last', 'date_eod':'last'}).dropna()

    rv, rvdates = HAR.rv(data5m, datecolumnname='date_eod', closingpricecolumnname='close')

    # Calculate daily log-returns
    # Drop NAN or whatever
    data1d['lret'] = np.log1p(data1d.close.pct_change())
    data1d = data1d.dropna()

    # make things fair and drop the first RV as well
    rv = rv[1:]
    rvdates = rvdates[1:]

    # # Keep only the the last 3200 days of time series if longer
    # if len(TS)>3200:
    #     TS = TS.iloc[-4800:]


    # fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=False)
    # # ax1.plot(data1d['date_eod'], np.sqrt(np.array(data1d['lret']**2)*252), label='from squared daily returns')
    # ax1.plot(rvdates, np.sqrt(rv*252), label='from Realized Variance')
    # ax1.set_title('Raw squared returns')
    
    # ax2.set_title('GARCH filtering')

    # plt.show()

    Rall = np.array(data1d['lret'])
    
    CGARCH = cg.cgarch([0.1, 0.02, 0.75, 0.01, 0.95, 0.001], Rall)

    bounds_garch    = ((0,None), (0.0001,0.06), (0.1,1),   (0,0.5), (0.7,1),   (0,0.5))
    initrange_garch = ((0,0.5),  (0.001, 0.01), (0.3,0.7), (0.02,0.1), (0.7,0.9), (0.002,0.1))

    CGARCH.OptimizationBounds=bounds_garch

    theta = CGARCH.genrandomthetas(initrange_garch, n=500, seed=1)

    Ncores=25
    mypool = mp.Pool(Ncores)
    CGARCH.estimate()
    # CGARCH.parallel(thetas=theta, Ncores=25)
    # CGARCH.parallel(thetas=theta, estpool=mypool)
    CGARCH.filter()
    # ax1.plot(data1d['date_eod'], np.sqrt(CGARCH.vpath*252), label='ST-GARCH')
    # ax1.plot(data1d['date_eod'], np.sqrt(CGARCH.qpath*252), label='LT-GARCH')
    # ax1.plot(data1d['date_eod'], CGARCH.uncvol*np.sqrt(252)*np.ones_like(CGARCH.vpath), label='uncvol')
    # ax1.set_ylabel('Annual volatility (%)')
    # ax1.legend()
    # fig.tight_layout()
    # plt.show()

    aggregatesampling = [1,5,20]
    horizons = [1, 5, 10, 20]
    # horizons = [1]
    # model = HAR.MODEL_HARQ
    # model = HAR.MODEL_HAR
    # model = HAR.MODEL_HARM
    model = HAR.MODEL_HARMC
    # datatransform = HAR.TRANSFORM_TAKE_LOG
    datatransform = HAR.TRANSFORM_DO_NOTHN
    estimationmethod = HAR.METHOD_WOLS
    # estimationmethod = HAR.METHOD_RFR
    ndaystoestimate = 2520
    # ndaystoestimate = 4754-22
    # mywindowtype = HAR.WINDOW_TYPE_ROLLING
    mywindowtype = HAR.WINDOW_TYPE_GROWING
    target = HAR.TOTALREALIZEDVARIANCE
    # target = HAR.PEAKDREALIZEDVARIANCE

    results = HAR.backtesting(data=data, aggregatesampling=aggregatesampling, 
                            datecolumnname='date_eod', closingpricecolumnname='close', 
                            windowtype=mywindowtype, estimatewindowsize=ndaystoestimate, 
                            model=model, datatransformation=datatransform, estimationmethod=estimationmethod, 
                            forecasthorizon=horizons, longerhorizontype=target)

    # ax2.plot(data1d['date_eod'][ndaystoestimate:], np.sqrt(results[1]['realized']['target'][minT:maxT]*252), label='REAL')
    # ax2.plot(data1d['date_eod'][ndaystoestimate:], np.sqrt(results[1]['model']['forecast'][minT:maxT]*252), label='HAR_WOLS')
    # ax2.plot(data1d['date_eod'][ndaystoestimate:], np.sqrt(results[1]['bench']['forecast'][minT:maxT]*252), label='bench')
    # ax2.set_title(f"1-day forecast: HAR RMSE={results[1]['model']['RMSE']:0.2E}, Bench RMSE={results[1]['bench']['RMSE']:0.2E}")
    # ax2.legend()

    """
        Use the same number of days to estimate and the same window type
        first estimate using a parallel, then, inside the loop, only use estimate
        the loop updates the returns time series, then estimates, then forecasts over N days
        Gather those forecasts, and plot.
    """
    x = np.zeros((np.size(rv,0)-ndaystoestimate-max(horizons)+1,len(horizons)))
    for index, ihor in enumerate(horizons):
        if (ihor>1) and (target==HAR.TOTALREALIZEDVARIANCE):
            x[:,index] = HAR.HAR._running_sumba(rv[ndaystoestimate:np.size(rv,0)-max(horizons)+ihor,], ihor)
        elif (ihor>1) and (target==HAR.PEAKDREALIZEDVARIANCE):
            x[:,index] = HAR.HAR._running_maxba(rv[ndaystoestimate:np.size(rv,0)-max(horizons)+ihor,], ihor)
        else:
            x[:,index] = rv[ndaystoestimate:-max(horizons)+1,]

    cgarch_results = cg.backtesting(CGARCH, Rall, x, mywindowtype, ndaystoestimate, estimatemethod=cg.ESTIMATE_METHOD_PARALLEL_ONCE, estpool=mypool, 
                                    forecasthorizon= horizons, longerhorizontype= target)

    minT = 0
    maxT = -1

    fig, axes = plt.subplots(2,2)
    axes[0,0].plot( np.sqrt(cgarch_results[1]['model']['forecast'][minT:maxT]*252), label='GARCH')
    axes[0,0].plot( np.sqrt(results[1]['model']['forecast'][minT:maxT]*252), label='HAR_WOLS')
    axes[0,0].plot( np.sqrt(results[1]['realized']['target'][minT:maxT]*252), label='REAL')
    axes[0,0].set_title(f"1-day forecast: GARCH RMSE={cgarch_results[1]['model']['RMSE']:0.2E}, HAR RMSE={results[1]['model']['RMSE']:0.2E}\n"+
                            f" GARCH R^2={cgarch_results[1]['model']['Rsquare']:0.2}, HAR R^2={results[1]['model']['Rsquare']:0.2}")
    axes[0,0].legend()

    axes[0,1].plot( np.sqrt(cgarch_results[5]['model']['forecast'][minT:maxT]*252), label='GARCH')
    axes[0,1].plot( np.sqrt(results[5]['model']['forecast'][minT:maxT]*252), label='HAR_WOLS')
    axes[0,1].plot( np.sqrt(results[5]['realized']['target'][minT:maxT]*252), label='REAL')
    axes[0,1].set_title(f"5-day forecast: GARCH RMSE={cgarch_results[5]['model']['RMSE']:0.2E}, HAR RMSE={results[5]['model']['RMSE']:0.2E}\n"+
                            f" GARCH R^2={cgarch_results[5]['model']['Rsquare']:0.2}, HAR R^2={results[5]['model']['Rsquare']:0.2}")
    axes[0,1].legend()

    axes[1,0].plot( np.sqrt(cgarch_results[10]['model']['forecast'][minT:maxT]*252), label='GARCH')
    axes[1,0].plot( np.sqrt(results[10]['model']['forecast'][minT:maxT]*252), label='HAR_WOLS')
    axes[1,0].plot( np.sqrt(results[10]['realized']['target'][minT:maxT]*252), label='REAL')
    axes[1,0].set_title(f"10-day forecast: GARCH RMSE={cgarch_results[10]['model']['RMSE']:0.2E}, HAR RMSE={results[10]['model']['RMSE']:0.2E}\n"+
                            f" GARCH R^2={cgarch_results[10]['model']['Rsquare']:0.2}, HAR R^2={results[10]['model']['Rsquare']:0.2}")
    axes[1,0].legend()

    axes[1,1].plot( np.sqrt(cgarch_results[20]['model']['forecast'][minT:maxT]*252), label='GARCH')
    axes[1,1].plot( np.sqrt(results[20]['model']['forecast'][minT:maxT]*252), label='HAR_WOLS')
    axes[1,1].plot( np.sqrt(results[20]['realized']['target'][minT:maxT]*252), label='REAL')
    axes[1,1].set_title(f"20-day forecast: GARCH RMSE={cgarch_results[20]['model']['RMSE']:0.2E}, HAR RMSE={results[20]['model']['RMSE']:0.2E}\n"+
                            f" GARCH R^2={cgarch_results[20]['model']['Rsquare']:0.2}, HAR R^2={results[20]['model']['Rsquare']:0.2}")
    axes[1,1].legend()
    fig.tight_layout()
    if target==HAR.PEAKDREALIZEDVARIANCE:
        fig.suptitle('Forecasting PEAK realized variance\n Benchmark is a Martingale forecast')
    elif target==HAR.TOTALREALIZEDVARIANCE:
        fig.suptitle('Forecasting TOTAL realized variance\n Benchmark is a Martingale forecast')
    plt.show()


    wearedone = 1


    


#### __name__ MAIN()
if __name__ == '__main__':
    main()
