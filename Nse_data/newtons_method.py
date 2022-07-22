import numpy as np
import sympy as sp
import pandas as pd
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt

nifty = pd.read_csv("Nse_data/2017-2018 nifty.csv")
vix = pd.read_csv("Nse_data/vix_01-Apr-2017_31-Mar-2018.csv")

nifty["% Change"] = nifty["Close"].pct_change()

x = nifty["% Change"].drop([0]).to_numpy().T
x_train = np.vstack([x, np.ones(len(x))]).T
y = vix["% Change"].drop([0]).to_numpy()
y_train = y[:, np.newaxis]


def newton(x, y, eps=0.00000001):

    theta = np.zeros(x.shape[1])
    h_x = np.exp(1/(1 + np.exp(-x.dot(theta))))
    m = h_x.shape[0]
    gradient_J_theta = x.T.dot(h_x - y) / m

    def hessian(x):
        
        alpha = np.array([h_x * (1 - h_x)]) * (np.identity(m))
        return (((x.T.dot(alpha)).dot(x))/m)

    while True:
        theta_old = theta

        theta += np.linalg.inv(hessian(x)).dot(gradient_J_theta)

        if np.linalg.norm(theta_old-theta, ord=1) < eps:
            break


    return theta


beta = (newton(x_train, y))

sns.scatterplot(x, y)
plt.plot(x, beta[0]*x + beta[1], 'r')
plt.show()