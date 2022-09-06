# from cmath import inf
import numpy as np
import pandas as pd
import cngarch as cg
import matplotlib.pyplot as plt


def main():
    # Get a daily time series of prices
    TS = pd.read_csv(f"./hd_QQQ.csv")
    # Calculate daily log-returns
    TS['lret'] = np.log1p(TS.close.pct_change())
    # Drop NAN or whatever
    TS = TS.dropna()
    # Keep only the the last 3200 days of time series if longer
    if len(TS)>3020:
        TS = TS.iloc[-4800:]


    fig, (ax1,ax2) = plt.subplots(2,1, sharex=True, sharey=False)
    ax1.plot(np.array(TS['lret']**2))
    ax1.set_title('Raw squared returns')
    ax2.set_title('GARCH filtering')

    # plt.show()

    # Divide the time series in 75% to estimate, and 25% to test
    n = len(TS['lret'])
    # n_is1 = int(np.round(n*0.75,0))
    # Rin1 = np.array(TS['lret'].iloc[0:n_is1])
    # Rout = np.array(TS['lret'].iloc[n_is1+1:])
    Rall = np.array(TS['lret'])
    
    GARCH = cg.garch([0.1, 0.02, 0.95, 0.01], Rall)
    NGARCH = cg.ngarch([0.1, 0.02, 0.95, 0.01, 0.1], Rall)
    CNGARCH = cg.cngarch([0.1, 0.02, 0.65, 0.01, 0.1, 0.995, 0.01, 0.1], Rall)

    bounds_ngarch   = ((0,None), (0.001,0.06), (0.8,1), (0,0.5), (-5,+5))
    bounds_cngarch   = ((0,None), (0.001,0.06), (0.5,1), (0.01,0.1), (-5,+5), (0.9,0.99999), (0,0.2), (-5,+5))
    initrange_cngarch   = ((0,0.5), (0.001,0.01), (0.3,0.9), (0.02,0.1), (-5,+5), (0.9,0.99999), (0.001,0.05), (-5,+5))


    NGARCH.OptimizationBounds=bounds_ngarch
    CNGARCH.OptimizationBounds=bounds_cngarch
    theta = CNGARCH.genrandomthetas(initrange_cngarch, n=500, seed=1)

    GARCH.estimate()
    GARCH.filter()
    ax2.plot(np.sqrt(GARCH.vpath*252)*100, label='GARCH')
    ax2.set_ylabel('Annual volatility (%)')
    plt.show()
    GARCH.forecast(kdays=20)


    
    NGARCH.estimate()
    NGARCH.filter()
    # ax2.plot(np.sqrt(NGARCH.vpath*252)*100, label='NGARCH')
    NGARCH.forecast(kdays=20)

    CNGARCH.estimate()
    CNGARCH.filter()
    ax2.plot(np.sqrt(CNGARCH.vpath*252)*100, label='CNGARCH total')
    ax2.plot(np.sqrt(CNGARCH.qpath*252)*100, label='CNGARCH LT')
    CNGARCH.forecast(kdays=20)
    print(CNGARCH)

    CNGARCH.parallel(theta, Ncores=16)
    CNGARCH.filter()
    ax2.plot(np.sqrt(CNGARCH.vpath*252)*100, label='CNGARCH 2 total')
    ax2.plot(np.sqrt(CNGARCH.qpath*252)*100, label='CNGARCH 2 LT')
    print(CNGARCH)


    ax2.legend()
    plt.show()


    wearedone = 1


    


#### __name__ MAIN()
if __name__ == '__main__':
    main()
