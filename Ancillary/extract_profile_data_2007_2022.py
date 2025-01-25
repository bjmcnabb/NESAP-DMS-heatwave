# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 19:52:05 2025

@author: bcamc
"""

#%%
def extract_profile(directory, data_names=['Temperature:CTD [deg_C_(ITS90)]','Salinity:CTD [PSS-78]','Sigma-t:CTD [kg/m^3]'],):
    
    import pandas as pd
    import numpy as np
    
    metadata_names = ['LOC:LATITUDE','LOC:LONGITUDE','LOC:STATION','Pressure:CTD [dbar]']
    CTD = pd.read_csv(directory, header=None, encoding = "ISO-8859-1", low_memory=False)
    # find locs of hypen seperated comments
    locs = []
    i = 0
    for j in CTD.iloc[:,0]:
        if isinstance(j,str):
            if j[0] == '-':
                locs.append(i)
        i+=1
    # drop comment header
    CTD = CTD.iloc[locs[-1]+1:,:]
    CTD = CTD.dropna(subset=0).reset_index().drop('index', axis=1)
    # relabel columns
    CTD.columns = CTD.iloc[0].values
    CTD = CTD.iloc[1:,:]
    CTD = CTD.reset_index().drop('index', axis=1)
    # drop gaps between data
    CTD = CTD.dropna(subset='Zone')
    # cast data as floats
    for col in CTD.columns[4:]:
        if col != 'LOC:STATION' and col != 'Comments' and col != 'Comments by sample_number' and col != 'INS:LOCATION':
            CTD[col] = CTD[col].astype('float64')
    # replace missing data with nans
    CTD = CTD.replace(-99,np.nan, regex=True)
    
    cols = metadata_names+data_names
    data = CTD.loc[:,cols]
    data = data.set_index(metadata_names, append=True)
    
    return data

