import pandas as pd
import numpy as np

def read_goog_sp500_dataframe():
    googFile = 'C:\\tensor_e\\finance_data\\GOOG.csv'
    spFile = 'C:\\tensor_e\\finance_data\\sp_500.csv'

    goog = pd.read_csv(googFile, sep=",", usecols=[0,5], names=['Date', 'Goog'], header=0)
    sp = pd.read_csv(spFile, sep=",", usecols=[0,5], names=['Date', 'SP500'], header=0)

    goog['SP500'] = sp['SP500']
    #date object
    goog['Date'] = pd.to_datetime(goog['Date'], format='%Y-%m-%d')
    goog = goog.sort_values(['Date'], ascending=[True])

    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]].pct_change()

    return returns

def read_goog_sp500_logistic_data():

    returns = read_goog_sp500_dataframe()
    returns['Intercept'] = 1

    xData = np.array(returns[["SP500", "Intercept"]][1:-1])
    yData = (returns["Goog"]>0)[1:-1]

    return (xData, yData)
