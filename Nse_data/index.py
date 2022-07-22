from lib2to3.refactor import get_all_fix_names
from re import X
import pandas as pd
import numpy as np
from math import floor, log
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# from nsedata import Nse
from pprint import pprint
import plotly.graph_objects as go

# nse = Nse()
def conv_ctime(date=(2020, 1, 1)):
    return str(floor(datetime(date[0], date[1], date[2], 5, 30).timestamp()))

def historical_data(code, date_from=(2020,1,1), date_to=(2021,1,1)):
    url = 'https://query1.finance.yahoo.com/v7/finance/download/{}.NS?period1={}&period2={}&interval=1d&events=history&includeAdjustedClose=true'.format(code.upper(), conv_ctime(date_from), conv_ctime(date_to))
            
    df = pd.read_csv(url)
    return df
# df = historical_data("^indiavix")
# pprint(data.columns)
nifty = pd.read_csv("ML/Nse_data/2017-2018 nifty.csv")
vix = pd.read_csv("ML/Nse_data/vix_01-Apr-2017_31-Mar-2018.csv")
hindunilvr = pd.read_csv("ML/Nse_data/2017-2018 hindunilvr.csv")

nifty["% Change"] = nifty["Close"].pct_change()

# pprint(nifty)
# pprint(vix)

"""
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

fig.show()
"""

"""
gold and inflation -> directly
dollar index and nifty and vix
"""

x_train = nifty["% Change"]
# y_train = hindunilvr["Close Price"]
y_train = vix["% Change"]




def gradient_descent(alpha1, alpha2, x, y):
    
    m = len(x)

    theta0 = 0
    theta1 = 0
  
    J = sum([(theta0 + theta1*x.iloc[i] - y.iloc[i])**2 for i in range(1, m)])

    for j in range(1, m):

        grad0 = sum([(theta0 + theta1*(x.iloc[i]) - y.iloc[i]) for i in range(1, m)])
        grad1 = sum([(theta0 + theta1*(x.iloc[k]) - y.iloc[k])*x.iloc[k] for k in range(1, m)])

        # print(sum([(theta0 + theta1*x.iloc[j] - y.iloc[j]) for j in range(1, m)]))
        # print(grad0, grad1)

        theta0 -= alpha1*grad0
        theta1 -= alpha2*grad1

        # e = sum([(theta0 + theta1*x.iloc[i] - y.iloc[i])**2 for i in range(1,m)])

        # print(abs(J - e))
        # J = e
    return theta0, theta1

alpha1, alpha2 = 0.001, 1
# x = [x_train.iloc[6], x_train.iloc[200]]
x= [-0.02, 0.02]
c, m = gradient_descent(alpha1, alpha2, x_train, y_train)
y = [m*i + c for i in x]
print(m,c)

#"""
from scipy import stats
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x_train[1:], y_train[1:])
print(intercept, slope) 
y0 = [slope*i + intercept for i in x]
#"""

# theta0 = 1
# theta1 = 1
# m = len(nifty)
# print(([(theta0 + theta1*nifty["% Change"].iloc[i] - vix["% Change"].iloc[i]) for i in range(m-1)]))

sns.scatterplot(x_train, y_train, alpha=0.7, s=30)
plt.plot(x, y)
# plt.plot(x, y0)
plt.show()