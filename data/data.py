#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:32:30 2020

@author: cstraehl
"""

import numpy as np
import pandas as pd


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
    # have binomial counting noise on infected rate
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

if __name__ == "__main__":    
    S,I,R,D = load_country(S0=82000000, country = "Germany")
    print("S",S)
    print("I",I)
    print("R",R)
    print("D",D)
    