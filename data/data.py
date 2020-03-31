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
import tensorflow as tf

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


def data_generator(batch_size, t_obs=None, t_min=30, t_max=100, dt=1, to_tensor=True, do_enhance_contrast = True, **args):
    """
    Runs the forward model 'batch_size' times by first sampling fromt the prior
    theta ~ p(theta) and running x ~ p(x|theta).
    ----------
    
    Arguments:
    batch_size : int -- the number of samples to draw from the prior
    to_tensor  : boolean -- converts theta and x to tensors if True
    ----------
    
    Output:
    theta : tf.Tensor or np.ndarray of shape (batch_size, theta_dim) - the data gen parameters 
    x     : tf.Tensor of np.ndarray of shape (batch_size, n_obs, x_dim)  - the generated data
    """
    S0 = np.random.randint(1*1e6,400*1e6,size=(batch_size,1))
    I0 = np.zeros((batch_size,1))
    D0 = np.zeros((batch_size,1))
    R0 = np.zeros((batch_size,1))
    
    init_vals = np.concatenate([S0, S0/S0, I0, D0, R0],axis=1)
    
    # Sample from prior
    # theta is a np.array of shape (batch_size, theta_dim)
    otheta = priors.prior_sample(batch_size)
    theta = thetaed.encode(otheta,priors.th_low, priors.th_high)
    
    if t_obs is None:
        t_obs = np.random.randint(t_min, t_max+1)
    
    t = np.linspace(0, t_obs, int(t_obs/dt) + 1)
    
    
    #generate 3 random rhos with decreasing values
    rhos = np.random.rand(batch_size,3)*0.6+0.4
    rhos = np.cumprod(rhos,axis=1)
    #print("rhos", rhos)

    rhos_t = np.random.randint(t_obs-1, size=(batch_size,4))+1
    rhos_t = (rhos_t / np.sum(rhos_t,axis=1,keepdims=True))[:,:]
    
    #print("atanh_rhos", rhos)
    # construct rhoparam matrix
    rhoparams = thetaed.encode(np.concatenate([rhos,rhos_t],axis=1), priors.rh_low, priors.rh_high)
    
    #construct new theta
    theta_all = np.concatenate([rhoparams, theta],axis=1)

    # construct thetap
    thetap = np.concatenate([init_vals, theta_all],axis=1)
    
    # Generate data
    # x is a np.ndarray of shape (batch_size, n_obs, x_dim)
    #x = np.apply_along_axis(forward_model, axis=1, arr=thetap, t=t, **args)
    x = forward_model(thetap, t)
    
    x = add_counting_noise(x,S0,otheta[:,1:2])
    
    x = np.clip(x,0,1) # FIXME: WHY ARE THERE SOMETIMES VALUES <0 ?????????
    if do_enhance_contrast:
        x = enhance_contrast(x)
    
    # Convert to tensor, if specified 
    if to_tensor:
        theta_all = tf.convert_to_tensor(theta_all, dtype=tf.float64)
        x = tf.convert_to_tensor(x, dtype=tf.float64)
    return {'theta': theta_all, 'x': x}


if __name__ == "__main__":    
    S,I,R,D = load_country(S0=82000000, country = "Germany")
    print("S",S)
    print("I",I)
    print("R",R)
    print("D",D)
    