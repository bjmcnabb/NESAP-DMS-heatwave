# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 22:57:31 2025

@author: bcamc
"""

#%% Import packages
from tqdm import tqdm
import numpy as np
import pandas as pd
import xarray as xr
import os
import dask.array as da

dir_ = 'H:/noaa_SST_raw_for_clim_1985-2012'
nc_output_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/projects/sulfur/LineP_2022/Scripts/1985-2012_aug_SST_clim.nc'

#%% load in data
files = os.listdir(dir_)

# set iteration counter
i = 0
for file in tqdm(files):
    
    # extract data
    data = xr.open_dataset(dir_+os.sep+file)
    data.close()
    
    # pull out NESAP region SST data
    sst_raw = pd.DataFrame(data.analysed_sst.values[0,:,:], index=data.lat, columns=data.lon)
    sst_raw = sst_raw.loc[48:53, -147:-123]
    
    # save grid coordinates on last iteration
    if i == len(files)-1:
        lats = sst_raw.index.values
        lons = sst_raw.columns.values
    
    # convert to numpy arrays, add dimension for concatenation
    sst_raw = da.from_array(sst_raw.values, chunks=(80,440))
    sst_raw = da.expand_dims(sst_raw, axis=2)
    
    # concatneate arrays on subsquent iterations
    if i == 0:
        sst = sst_raw
    else: 
        sst = da.append(sst, sst_raw, axis=2)
    
    # update iteration counter
    i+=1
    del data    

#%% compute the means and SD

sst_mean = sst.mean(axis=2).compute()
sst_sd = sst.std(axis=2).compute()

#%% save to netcdf file

ds = xr.Dataset(
    data_vars={"sst_mean": (("latitude", "longitude"), sst_mean),
    "sst_std": (("latitude", "longitude"), sst_sd)},
    coords={
        "longitude": lons,
        "latitude": lats,
        "julian days": pd.Series([pd.Timestamp(i.split('_')[2].split('.')[0]).strftime('%j') for i in files]).unique(),
    },
)
ds.to_netcdf(nc_output_dir)