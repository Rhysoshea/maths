"""
Simulating Geometric Brownian Motion

Solution
S(t[i+1]) = S(t[i])exp([mu - 0.5sigma**2](t[i+1] - t[i]) + sigma*sqrt(t[i+1] - t[i])*Z[i+1] )
"""

import pandas as pd
import numpy as np
import math
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('MacOSX')


def gbm_sims(n_sim, T, dt, mu, sigma, S0):
    m = (mu - 0.5 * sigma*sigma) * dt
    S = S0 * np.ones((n_sim, T+1))
    BM = sigma * math.sqrt(dt) * np.random.randn(n_sim, T)
    # print (S)
    # exit()
    for r in range(len(BM)):
        BM[r] = np.cumsum(m + BM[r])

    print (BM)
    # exit()
    for r in range(len(S)):
        for t in range(len(S[r])-1):
            S[r][t+1] = S0 * math.exp(BM[r][t-1])
    # S[:, 1:] = S0 * math.exp(np.cumsum(m+BM, axis=1))
    return S

def plotter(data):
    f, ax = plt.subplots(figsize=(10,10))
    for row in data:
        # print (row)
        plt.plot(range(len(row)), row)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()

n_sim = 5
T = 100
dt = 0.1
mu = 0
sigma = 10
S0 = 100

result = gbm_sims(n_sim, T, dt, mu, sigma, S0)
print (result)
plotter(result)
