'''
G-Research trading game

Expected value of product is determined by probability density y=2/(1+x)^3

Rules:
- each time step a new price will be exposed according to the probability distribution
- this value is captured according to an algorithm you have to write
- if you decide to take a value as profit then it blocks you from trading for the next 5 turns
'''

import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# density function
def y(x):
    return (2/(1+x)**3)

# probability distribution. Explicit integral of density function over discrete steps to calculate a distribution approximately
def prob_y(x1,x2):
    return (-1/(1+x2)**2) + (1/(1+x1)**2)

# calculate a range of probabilities for each price level, to be inserted in the random choice function
def calc_prob():
    probabilities = [prob_y(x, x+0.1) for x in np.arange(0,100,0.1)]
    total =0
    for x in probabilities:
        total += x
    probabilities[0] += 1-total
    return probabilities

def price_generator(probabilities):
    x = np.random.choice(np.arange(0,100,0.1), p=probabilities)
    return x

def graph_price(data):
    for x in data:
        plt.plot(range(len(x)),x)
    names = [x for x in price_levels]
    for line, name in zip(data,names):
        y = line[len(line)-1]
        plt.annotate('{:.1f}'.format(name),
                     xy=(len(line)-1,y),
                     size=8,
                     va="center")
    plt.show()

def graph(data, price_levels):
    for x in data:
        plt.plot(range(len(x)),x)
    names = [x for x in price_levels]
    for line, name in zip(data,names):
        y = line[len(line)-1]
        plt.annotate('{:.1f}'.format(name),
                     xy=(len(line)-1,y),
                     size=8,
                     va="center")
    plt.show()

def graph_curve(data, price_levels):
    plt.plot(price_levels, data)
    plt.show()

def hist(x):
    plt.hist(x,bins=50)
    plt.show()

class Trade_bot:
    def __init__(self, limit):
        self.trades = []
        self.limit = limit
        self.locked = False
        self.counter = 0

    def trade(self, price):
        if self.locked == True:
            self.counter -= 1
            if self.counter == 0:
                self.locked = False

        else:
            if price >= self.limit:
                self.trades.append(price)
                self.locked = True
                self.counter = 5


if __name__ == "__main__":
    probs = calc_prob()
    prices = []
    n = 10000 #time_steps

    price_levels = np.arange(0,10,0.1)

    bots = [Trade_bot(x) for x in price_levels]

    for _ in range(n):

        price = price_generator(probs)
        prices.append(price)

        for bot in bots:
            bot.trade(price)


    graph_price([prices])
    hist(prices)

    p = [pd.Series(bot.trades).cumsum() for bot in bots]
    p_tot = [x[len(x)-1] for x in p]

    graph(p, price_levels)
    graph_curve(p_tot, price_levels)
