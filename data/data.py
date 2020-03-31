#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:32:30 2020

@author: cstraehl
"""

import numpy as np
import pandas as pd
import data.priors as priors
from deep_bayes import theta as thetaed

def load_country(S0,country="Germany"):
    Idf = pd.read_csv("data/time_series_covid19_confirmed_global_narrow.csv",skiprows=[1],parse_dates=["Date"],index_col="Date")
    Rdf = pd.read_csv("data/time_series_covid19_recovered_global_narrow.csv",skiprows=[1],parse_dates=["Date"],index_col="Date")
    Ddf = pd.read_csv("data/time_series_covid19_deaths_global_narrow.csv",skiprows=[1],parse_dates=["Date"],index_col="Date")
    I = Idf.loc[Idf['Country/Region'] == country].resample('D').interpolate('cubic')["Value"].to_numpy()
    R = Rdf.loc[Rdf['Country/Region'] == country].resample('D').interpolate('cubic')["Value"].to_numpy()
    D = Ddf.loc[Ddf['Country/Region'] == country].resample('D').interpolate('cubic')["Value"].to_numpy()
    
    # truncate dataseries to points where at least one infection happened
    idx = np.where(I > 0)[0][0]
    I = np.concatenate([np.array([0]),I[idx:]])
    R = np.concatenate([np.array([0]),R[idx:]])
    D = np.concatenate([np.array([0]),D[idx:]])
    
    assert(len(I)==len(R))
    assert(len(D)==len(R))
    
    S = S0-I-R-D
    print("S", S)
    print("I", I)
    print("D", D)
    print("R", R)
    N=S0
    return S/N,I/N,D/N,R/N


def enhance_contrast(x):
    assert((x>=0).all())
    assert((x<=1).all())
    a = np.log(x+1e-8)
    b = np.log(1-x+1e-8)
    t = np.repeat(np.arange(0,x.shape[1])[None,:,None],x.shape[0],axis=0)
    
    return np.concatenate([a,b,t],axis=2)

def add_counting_noise(x,N,p):
    # have binomial counting noise on delta infected rate
    delta_I = x[:,1:,1]-x[:,:-1,1]
    dIcounts = (delta_I*N).astype(np.int64)
    new_deltas = np.random.binomial(np.abs(dIcounts),p)
    new_deltas = np.where(dIcounts > 0, new_deltas, -new_deltas)/N
    Inew = np.cumsum(new_deltas,axis=1)

    # compute correction factor for measured R
    Rfactors = Inew / (x[:,1:,1]+1e-10)
    x[:,1:,1] = Inew
    x[:,1:,-1] *= Rfactors
    
    return x


#@jit(parallel=True,nopython=False)
def forward_model(aparams, t,psteps=0,init_params=None):
    """Forward model of the SIRD."""
    result = np.zeros((aparams.shape[0],t.shape[0]+psteps,4))
    offset = 0
    if init_params is None:
        offset = 5
        
    arh_params = thetaed.decode(aparams[:,offset+0:offset+7],priors.rh_low, priors.rh_high)
    ath_params = thetaed.decode(aparams[:,offset+7:],priors.th_low, priors.th_high)
    at = np.repeat(t[None,:],aparams.shape[0],axis=0)
    
    for pn in range(aparams.shape[0]):
        #print(pn)
        if init_params is None:
            N, S_0, I_0, D_0, R_0 = aparams[pn,:5]
        else:
            N, S_0, I_0, D_0, R_0 = init_params
            
        S, I, D, R = [S_0], [I_0], [D_0], [R_0]

        #extract time dependent rhos from params
        rh_params = arh_params[pn]
        t = at[pn]
        
        rhos = rh_params[0:3]
        rhos_t = rh_params[3:7]
        rhos_t = rhos_t / np.sum(rhos_t)
        rhos_t = np.cumsum(rhos_t)[0:3]*t.shape[0]
        rhos_t = np.clip(rhos_t.astype(np.int64),0,t.shape[0])

        #construct dense rhos array
        rho = np.ones((t.shape[0]))
        for i in range(rhos_t.shape[0]):
            rho[rhos_t[i]:] = rhos[i]   
        th_params = ath_params[pn]
        E_0, p, beta, alpha, gamma, d = th_params
        E = [E_0/N]

        dt = t[1] - t[0]
        #print(rho, th_params)
        
        if psteps != 0:
            pt = np.linspace(t[-1], t[-1]+dt*psteps, psteps)
            t = np.concatenate([t,pt])
            rhop = np.ones(pt.shape)*rho[-1]
            rho = np.concatenate([rho,rhop])

        for i,_ in enumerate(t[1:]):
            next_S = S[-1] + (-rho[i]*beta*S[-1]*I[-1])*dt
            next_E = E[-1] + (rho[i]*beta*S[-1]*I[-1] - alpha*E[-1])*dt
            next_I = I[-1] + (alpha*E[-1] - gamma*I[-1] - d*I[-1])*dt
            next_R = R[-1] + (gamma*I[-1])*dt
            next_D = D[-1] + (d*I[-1])*dt
            
            #print(I[-1],next_R, next_D, next_D - D[-1],d)

            S.append(next_S)
            E.append(next_E)
            I.append(next_I)
            D.append(next_D)
            R.append(next_R)

        result[pn,...] = np.stack([S, I, D, R]).T
        #print("timeseries", dt, th_params, result[pn])
    return result


if __name__ == "__main__":    
    S,I,R,D = load_country(S0=82000000, country = "Germany")
    print("S",S)
    print("I",I)
    print("R",R)
    print("D",D)
    