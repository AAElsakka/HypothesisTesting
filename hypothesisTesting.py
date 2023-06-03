import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    trades = generateDummyData()
    months = sorted(trades['month'].unique())
    for month in months:
        statisticalTest(trades, month)
        break

def generateDummyData():
    month = np.random.randint(low=1, high=13, size=(1000,1))
    returns = np.random.normal(loc=0.0, scale=1.0, size=(1000,1))

    data=np.hstack([month, returns])  
    trades = pd.DataFrame(columns=["month", "returns"],
                          data=data)
    return trades

def statisticalTest(trades, month):
    monthTrades = trades[trades['month'] == month]
    threshold = getHypothesisThreshold(month)
    numberOfIterations = 10_000
    numberOfSamples = 3
    meanReturns = simulateTrades(numberOfIterations, monthTrades, numberOfSamples)
    simulatedTradesWinningsPercentage(meanReturns, threshold)
    zScore = zScoreCalculator(meanReturns, threshold,numberOfIterations)
    pValue = pValueCalculator(zScore)
    alpha=0.05
    isSignificant=getSignificance(pValue, alpha)
    plotSamplingDistribution(meanReturns, month, pValue, threshold)
    print(isSignificant)
    pass

def getHypothesisThreshold(month):
    return 1.0

def simulateTrades(numberOfIterations, trades, numberOfSamples):
    meanReturns=np.array([])
    for _ in range(numberOfIterations):
        sampledTrades = trades.sample(numberOfSamples)
        meanReturn = sampledTrades['returns'].mean()
        meanReturns=np.append(meanReturns, meanReturn)
    return meanReturns

def simulatedTradesWinningsPercentage(meanReturns, threshold):
    winningTradesCount = np.where(meanReturns>=threshold)[0].shape[0]
    allTradesCount = meanReturns.shape[0]
    winningPercentage = np.round(winningTradesCount / allTradesCount * 100, 2)
    print(f'Winning Percentages {winningPercentage}')
    return winningPercentage

def zScoreCalculator(meanReturns, threshold,numberOfIterations):
    meanReturnsMean = meanReturns.mean()
    meanReturnsStdev = meanReturns.std()    
    zScore = (meanReturnsMean - threshold) / (meanReturnsStdev / np.sqrt(numberOfIterations))
    return zScore

def pValueCalculator(zScore):
    cdf=(1.0 + np.math.erf(zScore / np.sqrt(2.0))) / 2.0
    pValue = 1 - cdf
    return pValue

def getSignificance(pValue, alpha):
    if pValue < alpha:
        print("There is a significant indication that the strategy has positive returns above Threshold.")
    else:
        print("There is no significant indication that the strategy has positive returns above Threshold.")
    isSignificant = True if pValue < alpha else False
    return isSignificant

def plotSamplingDistribution(meanReturns, month, pValue, threshold):
    plt.figure()
    plt.hist(meanReturns, bins=25)
    plt.title(f'Month {int(month)} Returns Sampling Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Distribution')
    plt.annotate(s=f'p_value={pValue}', xy=(0.75,0.85), xycoords='axes fraction')
    plt.axvline(x=threshold, c='r')
    plt.grid()
    plt.savefig(f'./test_{int(month)}.png',format='png')



if __name__=='__main__':
    main()
