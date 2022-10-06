# from cmath import inf
import numpy as np
import pandas as pd
import cngarch as cg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    # Get a daily time series of prices
    TS = pd.read_csv(f"./hd_QQQ.csv")
    # Convert the date to a nice format for the graphs
    TS['date_eod'] = pd.to_datetime(TS['date_eod'])
    # Calculate daily log-returns
    TS['lret'] = np.log1p(TS.close.pct_change())
    # Drop NAN or whatever
    TS = TS.dropna()
    # Keep only the the last 10 years (2520 days) of time series
    if len(TS)>2520:
        TS = TS.iloc[-2520:]
    # Get the returns into a numpy array
    Rall = np.array(TS['lret'])
    
    # Prepare 2 GARCH specifications with a set of initial values and the series of returns
    GARCH = cg.garch([0.1, 0.02, 0.90, 0.01], Rall)
    CNGARCH = cg.cngarch([0.1, 0.02, 0.65, 0.01, 0.1, 0.995, 0.01, 0.1], Rall)

    # Determine bounds on the coefficients to help the estimation
    GARCH.OptimizationBounds   = ((0,None), (0.001,0.06), (0.5,1), (0.01,0.1))
    CNGARCH.OptimizationBounds = ((0,None), (0.001,0.06), (0.5,1), (0.01,0.1), (-5,+5), (0.9,0.99999), (0,0.2), (-5,+5))

    # We can generate a series of random starting values to avoid local minima in the estimation
    # I use 5 as an example on how to use it, but 500 would be better
    initrange___garch  = ((0,0.5), (0.001,0.01), (0.3,0.9), (0.02,0.1))
    initrange_cngarch  = ((0,0.5), (0.001,0.01), (0.3,0.9), (0.02,0.1), (-5,+5), (0.9,0.99999), (0.001,0.05), (-5,+5))
    theta___garch =   GARCH.genrandomthetas(initrange___garch, n=50, seed=1)
    theta_cngarch = CNGARCH.genrandomthetas(initrange_cngarch, n=50, seed=1)

    forecastndays = 40

    # We can estimate using the set of coefficeints from the object creation
    GARCH.estimate()
    GARCH.filter()
    print(GARCH)
    # We can now forecast
    GARCH.forecast(kdays=forecastndays+1)
    print("Forecast, including the current day filtered volatility value")
    print(np.sqrt(GARCH.vforecast*252)*100)

    # We can estimate using the set of random starting values
    GARCH.parallel(Ncores=4)
    print(GARCH)
    GARCH.forecast(kdays=forecastndays)
    print("Forecast, including the current day filtered volatility value")
    print(np.sqrt(GARCH.vforecast*252)*100)
    # Observe that the regular estimation hit a local minima and the result is not that good.
    
    fig, axes = plt.subplots(3,1, sharex=False, sharey=False)
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    axes[0].plot(TS['date_eod'], np.sqrt(np.array(TS['lret']**2)*252)*100)
    axes[0].set_title('Raw squared returns with simple GARCH filtering overlay')
    axes[0].xaxis.set_major_locator(years)
    axes[0].xaxis.set_major_formatter(yearsFmt)
    axes[0].xaxis.set_minor_locator(months)
    axes[1].plot(TS['date_eod'], np.sqrt(GARCH.vpath*252)*100, label='GARCH')
    axes[1].set_title('simple GARCH filtering compared to CNGARCH filtering')
    axes[1].set_ylabel('Annual volatility (%)')
    fig.tight_layout()

    # Now let us use a more flexible model with 2 components
    # Multiple starting values is strongly recommended for the first estimation
    # Once you have the model well estimated, in backtesting setting, or live setting, 
    # every following day, you can simply estimate with the previously estimated coefficeints as starting values
    CNGARCH.parallel(Ncores=4)
    CNGARCH.filter()
    CNGARCH.forecast(kdays=forecastndays)
    print(CNGARCH)
    print(np.sqrt(CNGARCH.vforecast*252)*100)

    axes[1].plot(TS['date_eod'],np.sqrt(GARCH.vpath*252)*100,   label='GARCH')
    axes[1].plot(TS['date_eod'],np.sqrt(CNGARCH.vpath*252)*100, label='CNGARCH total')
    axes[1].plot(TS['date_eod'],np.sqrt(CNGARCH.qpath*252)*100, label='CNGARCH Long-Term')
    axes[1].legend()


    # Now, let us compare the forecasting by showing the last n days and next n days
    xpast = np.arange(-forecastndays+1,1,1)
    xfutu = np.arange(0,forecastndays+1,1)
    axes[2].plot(xpast, np.sqrt(GARCH.vpath[-forecastndays:]*252)*100,   label='past')
    axes[2].plot(xfutu, np.sqrt(GARCH.vforecast*252)*100,   label='forecast')
    axes[2].plot(xpast, np.sqrt(CNGARCH.vpath[-forecastndays:]*252)*100,   label='past')
    axes[2].plot(xfutu, np.sqrt(CNGARCH.vforecast*252)*100,   label='forecast')
    axes[2].set_title(f"GARCH and CNGARCH forecast comparison for {forecastndays} days")
    axes[2].legend()

    plt.show()



    


#### __name__ MAIN()
if __name__ == '__main__':
    main()
