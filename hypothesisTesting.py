import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    trades = generateDummyData()
    months = sorted(trades['month'].unique())
    for month in months:
        getSignificance(trades, month)

def generateDummyData():
    month = np.random.randint(low=1, high=13, size=(1000,1))
    returns = np.random.normal(loc=0.0, scale=1.0, size=(1000,1))

    data=np.hstack([month, returns])  
    trades = pd.DataFrame(columns=["month", "returns"],
                          data=data)
    return trades

def getSignificance(trades, month):
    monthTrades = trades[trades['month'] == month]
    hypothesisReturn = 0.5
    numberOfIterations = 100_00#0
    meanReturns = []
    pValue=0
    for _ in range(numberOfIterations):
        sampledTrades = monthTrades.sample(3)
        meanReturn = sampledTrades['returns'].mean()
        pValue = pValue+1 if meanReturn>=hypothesisReturn or meanReturn<=-1*hypothesisReturn else pValue 
        meanReturns.append(meanReturn)
    pValue = pValue/len(meanReturns)
    print(pValue)
    plt.figure()
    plt.hist(meanReturns, bins=25)
    plt.title(f'Month {int(month)} Returns Sampling Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Distribution')
    plt.annotate(s=f'p_value={pValue}', xy=(0.75,0.85), xycoords='axes fraction')
    plt.axvline(x=hypothesisReturn, c='r')
    plt.axvline(x=-1*hypothesisReturn, c='r')
    plt.grid()
    plt.savefig(f'./test_{int(month)}.png',format='png')
    pass


if __name__=='__main__':
    main()
