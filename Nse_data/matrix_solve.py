import pandas as pd
from pprint import pprint
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

nifty = pd.read_csv("Nse_data/2017-2018 nifty.csv")
vix = pd.read_csv("Nse_data/vix_01-Apr-2017_31-Mar-2018.csv")

nifty["% Change"] = nifty["Close"].pct_change()

x = nifty["% Change"].drop([0]).to_numpy().T
x_train = np.vstack([x, np.ones(len(x))]).T
# print(x_train)
y = vix["% Change"].drop([0]).to_numpy()
y_train = y[:, np.newaxis]

theta = np.dot(np.dot(np.linalg.inv(np.dot(x_train.T, x_train)), x_train.T), y_train)

print(theta)
sns.scatterplot(x, y)
plt.plot(x, theta[0]*x + theta[1], 'r')
plt.show()

"""
----Newton's method----

def hessian(x):
    x_grad = np.gradien
    
    
    
    
    t(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)

    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian 

print(hessian(x_train))
"""