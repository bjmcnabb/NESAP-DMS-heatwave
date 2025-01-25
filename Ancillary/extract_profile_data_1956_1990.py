# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:33:21 2024

@author: bcamc
"""

#%%
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import gsw

LineP_stn_coords = pd.read_csv('C:/Users/bcamc/OneDrive/Desktop/2022_cruise_data/LineP_stations_coords.csv')

#### CTD + Bottle data
dir_ = 'C:/Users/bcamc/OneDrive/Desktop/2022_cruise_data/DFO_aug_1956-1990/consolidated_dat'
#### CTD only
# dir_ = 'C:/Users/bcamc/OneDrive/Desktop/2022_cruise_data/DFO_aug_1956-1990/consolidated_ctd'

files = os.listdir(dir_)

var_names = {
    '!Depth':'depth',
    '!Press':'pres',
    '!Pressu':'pres',
    'Nitra':'NO3',
    'Nitrat':'NO3',
    'Nitrate':'NO3',
    'Oxyge':'O2',
    'Oxygen':'O2',
    'Phosph':'PO4',
    'Phospha':'PO4',
    'Salini':'sal',
    'Salinit':'sal',
    'Silic':'Si',
    'Silica':'Si',
    'Silicat':'Si',
    'Temper':'temp',
    'Tempera':'temp'}

ctd_his = {}

for file in tqdm(files):
    df = pd.read_fwf(dir_+os.sep+file)              
    df1 = pd.DataFrame([i[0].split('  ') for i in df.values])
    df2 = pd.DataFrame([i[0].split(' ') for i in df.values])
    header_ver = df2.iloc[0,3]
    
    # for files <1974: depth is first column values
    col_starts = ['!Depth', '!Press', '!Pressu']
    for strval in col_starts:
        try:
            cols = list(df2.loc[np.argwhere(df2.loc[:,0].values == strval)[0][0],:].dropna())
        # for files >1974: pressure is first column value
        except:
            pass
    
    # filter through to get close to datavalues
    for i,val in enumerate(df1.loc[:,0]):
        if val == '*END OF HEADER':
            h = i+1
    # re-read in file as csv, skipping rows found
    rawdata = pd.read_fwf(dir_+os.sep+file)
    for i,val in enumerate(rawdata.iloc[:,0]):
        if val == '*END OF HEADER':
            h = i+1
    # skip the header text
    rawdata = rawdata.iloc[h:,:]
    
    # drop other columns with None or NaN values (applies to some files, doesn't affect others)
    rawdata = rawdata.dropna(axis=1)
    
    # iterate through rows, seperating each string by removing space characters
    data = np.empty((rawdata.shape[0], len(cols)))
    for row in range(rawdata.shape[0]):
        # main case: parse assuming spaces are between all values
        try:
            data[row, :] = list(filter(None, [i.split(' ') for i in rawdata.iloc[row, :]][0]))
        # edge case: no space seperating -99 flags, use more complicated parsing
        except:
            # do a first pass to seperate the row values by space characters
            datastr = list(filter(None, [i.split(' ') for i in rawdata.iloc[row, :]][0]))
            # now seperate joined values by negative sign
            datastr = [i.split('-') for i in datastr]
            # concatenate the nested lists into a single list
            datastr = list(itertools.chain.from_iterable(datastr))
            # filter out None values from concatenation
            datastr = list(filter(None, datastr))
            # add back in the negative values to the now parsed '99' flags
            datastr = ['-'+i if i[:2]=='99' else i for i in datastr]
            # now add parsed values to array
            data[row, :] = datastr
            
    # turn parsed arary into dataframe
    data = pd.DataFrame(data, columns=cols)
    
    # rename column names using dict
    cols_re = []
    for col in data.columns:
        cols_re.append(var_names[col])
    data.columns = cols_re
    
    
    # now extract metadata
    try: # for files < 1975-10
        station = df2.loc[np.argwhere(df2.loc[:,0].values == "STATION")[0][0],:].dropna()[-1:].values[0]
    except: # for files > 1975-10: have to ignore colon (seperated by several spaces):
        station = df2.loc[np.argwhere(df2.loc[:,0].values == "STATION			:")[0][0],:].dropna()[-1:].values[0]
    
    # if station != ':':
    
    lon = df2.loc[np.argwhere(df1.loc[:,0].values == 'LONGITUDE')[0][0],:].dropna()
    lon = [s for s in list(lon) if s not in ['LONGITUDE', ':','W', '!', '(deg', 'min)','']]
    lon = (float(lon[0])+(float(lon[1])/60))*-1
    
    lat = df2.loc[np.argwhere(df1.loc[:,0].values == 'LATITUDE')[0][0],:].dropna()
    lat = [s for s in list(lat) if s not in ['LATITUDE', ':','N', '!', '(deg', 'min)','']]
    lat = (float(lat[0])+(float(lat[1])/60))
    
    # add metadata as columns to dataframe
    data.loc[:,'station'] = np.tile(station, len(data))
    data.loc[:,'lat'] = np.tile(lat, len(data))
    data.loc[:,'lon'] = np.tile(lon, len(data))
    
    # if pressure is used, convert to depth (make depths postive values, to match datafiles reporting depth directly)
    if 'pres' in data.columns:
        data.loc[:,'depth'] = abs(gsw.conversions.z_from_p(data['pres'].values, data['lat'].values))
    
    # save data
    ctd_his[file] = data

# concatenate to dataframe
ctd_his = pd.concat(ctd_his)
# reset index and drop filenames
ctd_his = ctd_his.reset_index().drop('level_1', axis=1).rename({'level_0':'filename'}, axis=1)
# replace -99 values with nans
ctd_his = ctd_his.replace(-99,np.nan)


# correct for files with unlabelled stations - use 2022 list of stations to find closest matching coords
for file in ctd_his['filename'].unique():
    if ctd_his[ctd_his['filename'] == file].loc[:,'station'].unique()[0] == ':':
        lon_diff = np.min(abs(ctd_his[ctd_his['filename'] == file].loc[:,'lon'].unique()[0] - LineP_stn_coords['LOC:LONGITUDE']))
        lat_diff = np.min(abs(ctd_his[ctd_his['filename'] == file].loc[:,'lat'].unique()[0] - LineP_stn_coords['LOC:LATITUDE']))
        
        lonind = abs(ctd_his[ctd_his['filename'] == file].loc[:,'lon'].unique()[0] - LineP_stn_coords['LOC:LONGITUDE'])
        latind = abs(ctd_his[ctd_his['filename'] == file].loc[:,'lon'].unique()[0] - LineP_stn_coords['LOC:LONGITUDE'])
        coord_ind = np.argmin(lonind + latind)
        
        stn = LineP_stn_coords.loc[coord_ind, 'LOC:STATION']
        ctd_his.loc[ctd_his[ctd_his['filename'] == file].index, 'station'] = np.tile(stn, len(ctd_his[ctd_his['filename'] == file]['station']))
        


