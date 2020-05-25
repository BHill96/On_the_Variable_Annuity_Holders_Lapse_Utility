#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 00:00:17 2020

@author: blakehillier
"""

from numpy import arange, linspace, exp, log
from numpy.linalg import norm
import pandas as pd

# Returns the probability of dying at any age from x to m based on Gompertz Law
def gompertz(k, deltaT, x, m, b):
    return 1-exp((1-exp(k*deltaT/b))*exp((x-m)/b))

def Gomp_b(m,tMax,b, P, ages):
    total = []
    for p, t, x in zip(P, range(1, len(P)+1), ages):
        total.append((x-m)/log(log(p)/(1-exp(t/b))))
    return total

"""
minB is a list of the b values with the lowest error.
Each item corresponds to an age from the data in increasing order.
"""
def Tune_B(bRange, m, mortData, ages, tMax):
    bGrid = arange(bRange[0], bRange[1]+bRange[2], step=bRange[2])
    results = [Gomp_b(m,tMax,b, mortData, ages) for b in bGrid]
    results = pd.DataFrame(results)
    minB = []
    for col in results:
        error = abs(col-bGrid)
        minE = error[0]
        bestB = bGrid[0]
        for e, b in zip(error[1:], bGrid[1:]):
            if e < minE:
                minE = e
                bestB = b
        minB.append(bestB)
    return minB

"""
Each row contains the optimal b value per age for a given m value
"""
def Tune_M(survRates, ages, tMax, maxB=10, step=1):
    bVals = []
    mList = linspace(65, 120, 22)
    for m in mList:
        bVals.append(Tune_B([0,maxB,step], m, survRates, ages, tMax))
    return bVals, mList

# Could definitly be optimized
def optimal_M_B(survRates, ages, params, minAge, dt=1, P=1):
    errors = []
    for m, b in zip(params.index, params):
        g = [1-gompertz(k, dt, minAge, m, b) for k in range(0, len(ages))]
        error = (norm(g-survRates, P))/len(g)
        errors.append([error, m, b])

    errors = pd.DataFrame(errors, columns=['error', 'm', 'b'])

    minE = errors.error[0]
    j = 0
    for i in range(1, len(errors)):
        e = errors.error[i]
        if e < minE:
            j = i
            minE = e
    return errors.iloc[j]