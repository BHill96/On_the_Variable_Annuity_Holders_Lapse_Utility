#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:57:38 2020

@author: blakehillier
"""

from numpy import exp, linspace, log, sqrt
import matplotlib.pyplot as plt
from scipy.stats import norm

# Returns L2 (refer to 3.7 in paper)
def L2(mortFunc, x, m, b, T, deltaT, alpha, S2, mu, sigma, rho, q, optimalPhi):
    total = 0
    for k in range(1,int((T[5]-T[2])/deltaT)+1):
        total += (1-mortFunc(k, deltaT, x, m, b))*exp(-rho*k*deltaT)*exp((alpha*(mu-0.5*(sigma**2))*
                  k*deltaT)+(alpha*(alpha-1)*(sigma**2)*k*deltaT)/2)*((1-optimalPhi)**((k-1)*alpha))
    return total*(optimalPhi**alpha)*(S2**alpha)

# Returns R2 (refer to 3.9 in paper)
def R2(mortFunc, x, m, b, T, deltaT, alpha, mu, sigma, rho, V0, q):
    total = 0
    for k in range(1,int((T[5]-T[2])/deltaT)+1):
        total += (1-mortFunc(k, deltaT, x, m, b))*exp(-rho*k*deltaT)*((q*V0)**alpha)
    return total

# Returns the withdrawal percentage that gives greatest withdrawal utility
def findBestPhi(mortFunc, x, m, b, T, deltaT, alpha, rho, S2, mu, sigma, q):
    bestPhi = 0
    maxU = L2(mortFunc, x, m, b, T, deltaT, alpha, S2, mu, sigma, rho, q, bestPhi)
    for phi in linspace(0,1,num=100):
        U = L2(mortFunc, x, m, b, T, deltaT, alpha, S2, mu, sigma, rho, q, phi)
        if U > maxU:
            maxU = U
            bestPhi = phi
    plt.show()
    return bestPhi

# Returns whether the policyholder will withdrawal or not at t2
# True = withdrawal; False = not withdrawal
def Ind(mortFunc, x, m, b, T, deltaT, alpha, S2, mu, sigma, rho, V0, q):
    optimalPhi = findBestPhi(mortFunc, x, m, b, T, deltaT, alpha, rho, S2, mu, sigma)
    print(optimalPhi)
    L = L2(mortFunc, x, m, b, T, deltaT, alpha, S2, mu, sigma, rho, q, optimalPhi)
    R = R2(mortFunc, x, m, b, T, deltaT, alpha, mu, sigma, rho, V0, q)
    return L > R

# startS = lower bound of S2
# endS = upper bound of S2
# tol = tolerance coefficient
# A method that returns the value of S2 when L2 = R2 (the lapse/withdrawal boundary)
def SecantMethod(startS, endS, tol, gompertz, x, m, b, T, deltaT, alpha, rho, S0, mu, sigma, q):
    def func(S):
        phi = findBestPhi(gompertz, x, m, b, T, deltaT, alpha, rho, S, mu, sigma, q)
        return L2(gompertz, x, m, b, T, deltaT, alpha, S, mu,
                  sigma, rho, q, phi)-R2(gompertz, x, m, b, T, deltaT, alpha, mu, sigma, rho, S0, q)

    f0 = func(startS)
    f1 = func(endS)
    adj = False
    while f1 < 0:
        adj = True
        endS = endS*2
        f1 = func(endS)
    if adj:
        print('Adjusted search range to ({0},{1})'.format(startS, endS))
    adjEndS = endS

    # Start search
    i = 0
    # Should this be tmpf>1
    while f1-f0>tol:
        i += 1
        newS = endS-f1*(endS-startS)/(f1-f0)
        tmpf = func(newS)
        if tmpf > 0:
            f1 = tmpf
            endS = newS
        elif tmpf < 0:
            f0 = tmpf
            startS = newS
        else:
            # tmpf contains the exact intersection point
            break

    return int(newS), startS, adjEndS

def LapseProb(sigma, t2, dt, S, S0, mu):
    t = t2/dt
    # s = (sigma**2)*t
    x = (log(S/S0)-(mu-(sigma**2)/2)*t)/(sigma*sqrt(t))
    return 1-norm.cdf(x)

# Returns a plot with two lines whose intersection is the lapse boundary as well as the value of S2
# at the intersection
def lapseBoundaryFinder(mortFunc, x, m, b, T, deltaT, alpha, rho, S0, mu, sigma, q, fileName):
    # Find intersection point
    tol = 1
    L = 100000
    U = 300000
    intersection, L, U = SecantMethod(L, U, tol, mortFunc, x, m, b, T, deltaT, alpha, rho, S0, mu, sigma, q)
    # Plot graph
    # We use the interval from the Secant Method to ensure the intersection is in the bounds of the graph
    gridSize = 10 # step between lower bound and upper bound
    S2val = []
    diff = []
    for S2 in linspace(L,U,gridSize):
      S2val.append(S2)
      phi = findBestPhi(mortFunc, x, m, b, T, deltaT, alpha, rho, S2, mu, sigma, q)
      diff.append(L2(mortFunc, x, m, b, T, deltaT, alpha, S2, mu, sigma, rho, q, phi)
                   - R2(mortFunc, x, m, b, T, deltaT, alpha, mu, sigma, rho, S0, q))
    # Create Graph
    plt.plot(S2val,diff)
    plt.plot(S2val, [0]*len(S2val),label='Zero Line')
    plt.xlabel('S2')
    plt.ylabel('Difference')
    plt.legend()
    plt.title('Lapse Boundary')
    plt.savefig(fileName)
    plt.show()

    prob = LapseProb(sigma, T[2], deltaT, intersection, S0, mu)
    print('q = {0}'.format(q/deltaT))
    print('intersection at: $' + str(intersection))
    print('Probability of Lapse at {0}'.format(prob))

def payoutRatio(V0, alpha, mu, sigma, t, t2, t5, dt, rho, mortFunc, m, x, b, maxq):
    # print(int(((t5-t2)/dt)))
    denomItems = [(1-mortFunc(k, dt, x, m, b))*exp(-rho*k*dt)*V0 for k in range(1, int(((t5-t2)/dt))+1)]
    # print(denomItems)
    denom = sum(denomItems)
    # print(denom)
    q = V0*exp((mu-(sigma**2)/2)*t+(sigma**2)*t/2)/denom
    if q > maxq:
        print("ERROR: payout ratio {0} is greater than {1}!".format(q, maxq))
        q = maxq
    return q

# Plots Utility Function for all phis in [0,1]
def plotUtilityWithdrawl(mortFunc, x, m, b, T, deltaT, alpha, S2, mu, sigma, rho, q, figName, numSteps=1001):
    phis = [phi for phi in linspace(0,1,num=numSteps)]
    U = [L2(mortFunc,x=x,m=m,b=b,T=T, deltaT=deltaT, alpha=alpha, S2=S2, mu=mu, sigma=sigma,
            rho=rho, q=q, optimalPhi=phi) for phi in phis]
    plt.plot(phis, U)
    plt.title('Phi v. Utility of Withdrawl Policy\nMax Utility when phi = {0}'.format(phis[U.index(max(U))]))
    plt.xlabel('Phi')
    plt.ylabel('Utility of Withdrawl Policy')
    plt.savefig(figName)
    plt.show()
