#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:32:30 2020

@author: cstraehl
"""

import numpy as np
import pandas as pd


def load_country(S0,country="Germany"):
    Idf = pd.read_csv("time_series_covid19_confirmed_global_narrow.csv",skiprows=[1],parse_dates=["Date"],index_col="Date")
    print(Idf.index)
    Rdf = pd.read_csv("time_series_covid19_recovered_global_narrow.csv",skiprows=[1],parse_dates=["Date"],index_col="Date")
    Ddf = pd.read_csv("time_series_covid19_deaths_global_narrow.csv",skiprows=[1],parse_dates=["Date"],index_col="Date")
    I = Idf.loc[Idf['Country/Region'] == country].resample('D').interpolate('cubic')["Value"].to_numpy()
    R = Rdf.loc[Rdf['Country/Region'] == country].resample('D').interpolate('cubic')["Value"].to_numpy()
    D = Ddf.loc[Ddf['Country/Region'] == country].resample('D').interpolate('cubic')["Value"].to_numpy()
    
    assert(len(I)==len(R))
    assert(len(D)==len(R))
    
    S = S0-I-R-D
    
    return S,I,R,D

if __name__ == "__main__":    
    S,I,R,D = load_country(S0=82000000, country = "Germany")
    print("S",S)
    print("I",I)
    print("R",R)
    print("D",D)
    