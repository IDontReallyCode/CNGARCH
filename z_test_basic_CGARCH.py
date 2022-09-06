import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import HAR
import cngarch as cg

def main():
    # load intraday
    ticker = 'SPY'
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

    # # Keep only the the last 3200 days of time series if longer
    # if len(TS)>3200:
    #     TS = TS.iloc[-4800:]


    fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=False)
    # ax1.plot(data1d['date_eod'], np.sqrt(np.array(data1d['lret']**2)*252), label='from squared daily returns')
    ax1.plot(rvdates, np.sqrt(rv*252), label='from Realized Variance')
    ax1.set_title('Raw squared returns')
    
    ax2.set_title('GARCH filtering')

    # plt.show()

    Rall = np.array(data1d['lret'])
    
    CGARCH = cg.cgarch([0.1, 0.02, 0.75, 0.01, 0.95, 0.001], Rall)

    bounds_garch    = ((0,None), (0.0001,0.06), (0.1,1),   (0,0.5), (0.7,1),   (0,0.5))
    initrange_garch = ((0,0.5),  (0.001, 0.01), (0.3,0.7), (0.02,0.1), (0.7,0.9), (0.002,0.1))

    CGARCH.OptimizationBounds=bounds_garch

    theta = CGARCH.genrandomthetas(initrange_garch, n=500, seed=1)

    CGARCH.estimate()
    CGARCH.parallel(thetas=theta, Ncores=25)
    CGARCH.filter()
    ax1.plot(data1d['date_eod'], np.sqrt(CGARCH.vpath*252), label='ST-GARCH')
    ax1.plot(data1d['date_eod'], np.sqrt(CGARCH.qpath*252), label='LT-GARCH')
    ax1.plot(data1d['date_eod'], CGARCH.uncvol*np.sqrt(252)*np.ones_like(CGARCH.vpath), label='uncvol')
    ax1.set_ylabel('Annual volatility (%)')
    ax1.legend()
    fig.tight_layout()
    plt.show()
    CGARCH.forecast(kdays=20)


    wearedone = 1


    


#### __name__ MAIN()
if __name__ == '__main__':
    main()
