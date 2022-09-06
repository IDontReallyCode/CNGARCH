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
    ax1.plot(data1d['date_eod'], np.sqrt(np.array(data1d['lret']**2)*252), label='from squared daily returns')
    ax1.plot(rvdates, np.sqrt(rv*252), label='from Realized Variance')
    ax1.set_title('Raw squared returns')
    
    ax2.set_title('GARCH filtering')

    # plt.show()

    Rall = np.array(data1d['lret'])
    
    GARCH = cg.garch([0.1, 0.02, 0.95, 0.01], Rall)

    bounds_garch    = ((0,None), (0.0001,0.06), (0.7,1),   (0,0.5))
    initrange_garch = ((0,0.5),  (0.001, 0.01), (0.3,0.9), (0.02,0.1))

    GARCH.OptimizationBounds=bounds_garch

    theta = GARCH.genrandomthetas(initrange_garch, n=500, seed=1)

    GARCH.estimate()
    GARCH.parallel(thetas=theta, Ncores=25)
    GARCH.filter()
    ax1.plot(data1d['date_eod'], np.sqrt(GARCH.vpath*252), label='GARCH')
    ax1.plot(data1d['date_eod'], GARCH.uncvol*np.sqrt(252)*np.ones_like(GARCH.vpath), label='uncvol')
    ax1.set_ylabel('Annual volatility (%)')
    ax1.legend()
    fig.tight_layout()
    plt.show()
    GARCH.forecast(kdays=20)


    
    # NGARCH.estimate()
    # NGARCH.filter()
    # # ax2.plot(np.sqrt(NGARCH.vpath*252)*100, label='NGARCH')
    # NGARCH.forecast(kdays=20)

    # CNGARCH.estimate()
    # CNGARCH.filter()
    # ax2.plot(np.sqrt(CNGARCH.vpath*252)*100, label='CNGARCH total')
    # ax2.plot(np.sqrt(CNGARCH.qpath*252)*100, label='CNGARCH LT')
    # CNGARCH.forecast(kdays=20)
    # print(CNGARCH)

    # CNGARCH.parallel(theta, Ncores=16)
    # CNGARCH.filter()
    # ax2.plot(np.sqrt(CNGARCH.vpath*252)*100, label='CNGARCH 2 total')
    # ax2.plot(np.sqrt(CNGARCH.qpath*252)*100, label='CNGARCH 2 LT')
    # print(CNGARCH)


    # ax2.legend()
    # plt.show()


    wearedone = 1


    


#### __name__ MAIN()
if __name__ == '__main__':
    main()
