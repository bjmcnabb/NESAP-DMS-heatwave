# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 08:45:51 2025

@author: Brandon McNabb
"""
#%% Start timer
import timeit
analysis_start = timeit.default_timer()
#%% Import packages

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import xarray as xr
import scipy
from scipy.stats import pearsonr, spearmanr
import gsw

# load custom functions
from FRRF_data_extraction import FRRF_extract
from turnover_rates import get_rates
from extract_profile_data_2007_2022 import extract_profile
from VGPM_NPP_toolbox import LatToDayLength, calculate_NPP
from bin1d import bin1d

def barplot_annotate_brackets(ax, num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, horz=False, under_bar=None, bbox=None, maxasterix=None, lw=0.5):
    """ 
    Annotate barplot with p-values. Taken from: https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = 0.05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n.s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr is not None and yerr.any():
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    if under_bar is None:
        mid = ((lx+rx)/2, y+barh)
    else:
        mid = ((lx+rx)/2, y+barh+under_bar)

    kwargs = dict()
    if horz is False:
        kwargs['ha'] = 'center'
        kwargs['va'] = 'bottom'
    else:
        kwargs['ha'] = 'center'
        kwargs['va'] = 'center'
    if fs is not None:
        kwargs['fontsize'] = fs
    if bbox is not None:
        kwargs['bbox'] = bbox
    
    if horz is False:
        ax.plot(barx,
                bary,
                c='black',
                clip_on=False,
                zorder=2,
                lw=lw)
        ax.text(*mid,
                text,
                zorder=1,
                clip_on=False,
                **kwargs)
    else:
        ax.plot(bary,
                barx,
                c='black',
                zorder=10,
                clip_on=False,
                lw=lw)
        ax.text(*mid[::-1],
                text,
                rotation=90,
                clip_on=False,
                **kwargs)

# set constants
idx = pd.IndexSlice
min_lon, max_lon, min_lat, max_lat = -146, -123, 48, 52
bounds = [min_lat, max_lat, min_lon, max_lon]

# set main folder directories
local_dir_ = 'C:/Users/bcamc/OneDrive/Desktop/Python/projects/sulfur/LineP_2022' # location of scripts
fig_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/projects/sulfur/LineP_2022/Figures/Final_figures/' # output dir for figs
root_dir = 'C:/Users/bcamc/OneDrive/Desktop/2022_cruise_data/' # location of data

# set specific data file directories
chemtax_dir = root_dir+'2022_LineP_HPLC_CHEMTAX_results.xlsx'
hplc_dir = root_dir+'2022-008_HPLCdata.xlsx'
SST_dir = root_dir+'SST_anomalies_2022_V2/'
MHW_dir_ = root_dir+'LineP_MHW_categories/'
MHW_his_stats_dir = root_dir+'1985-2012_aug_SST_clim.nc'
linep_his_nitrate_dir_ = root_dir+'LineP_historic_che'
linep_his_ctd_dir_ = root_dir+'LineP_historic_ctd'
ctd_aug2022_dir = root_dir+'2022-008-ctd-cruise.csv'
linep_2022_nitrate_dir_ = root_dir+'2022-008-che-cruise.csv'
incub_dir = root_dir+'LineP_incubation_expts.xlsx'

frrf_dirs_ = [
    root_dir+'Brandon_data/2022_LineP_reprocessed/P16',
    root_dir+'Brandon_data/2022_LineP_reprocessed/P20',
    root_dir+'Brandon_data/2022_LineP_reprocessed/P26',
    ]

#%% Build ML NESAP models

# run models on climatological data - see McNabb & Tortell (2022)
from NESAP_build_models import *

models_combined = pd.DataFrame(pd.concat([np.sinh(y_pred_stack),np.sinh(ANN_y_pred)],axis=1).mean(axis=1).values, index=X_full.index)
models_combined = models_combined.loc[[8]]

# get the SD across all 2000 models (RFR & ANN), at each pixel
models_sd = np.std(np.sinh(y_pred_raw)+np.sinh(ANN_predict_raw), axis=0)
models_sd = pd.Series(models_sd, index=X_full.index)
models_sd = models_sd.loc[[8]]

PMEL.loc[:,'DateTime'] = pd.Series([pd.Timestamp(i) for i in PMEL.loc[:,'DateTime']])

#%% Switch to main local directory, if not already in it

if os.getcwd() != local_dir_:
    os.chdir(local_dir_)

#%% Load in underway Line P/La Perouse DMS/P data

from process_uw_DMS_2022 import *

#%% Line P major stations
LineP_stn_names = ['P1', 'P4', 'P12', 'P16', 'P20', 'P26']
LineP_stn_coords = np.array([[48.575, -125.5],
                             [48.65, -126.6666666666666666],
                             [48.9700000000000001, -130.6666666666666666],
                             [49.2833333333333333, -134.6666666666666666],
                             [49.5666666666666667, -138.6666666666666666],
                             [50, -145]])

LineP_stns = pd.DataFrame(LineP_stn_coords, index=LineP_stn_names, columns=['lat', 'lon'])

#%% Load HPLC and CHEMTAX data

stations = [
    'Haro59',
    'JF2',
    'P1',
    'P2',
    'P3',
    'P4',
    'P5',
    'P6',
    'P7',
    'P8',
    'P9',
    'P10',
    'P11',
    'P12',
    'P14',
    'P15',
    'P16',
    'P17',
    'P18',
    'P19',
    'P20',
    'P21',
    'P22',
    'P23',
    'P24',
    'P25',
    'P35',
    'P26',
]

taxa = pd.read_excel(chemtax_dir, sheet_name='average')
hplc = pd.read_excel(hplc_dir, sheet_name='HPLC-pigments')
tchl = hplc['T chl a'].copy()
taxa['tchl'] = tchl

taxa = taxa.where(taxa['Pressure']<10).dropna()
taxa = taxa.set_index(['Station','Pressure'])
taxa = taxa.drop(['Cruise_Number','Sample_Number','Sample','ClusterCode'], axis=1)
taxa = taxa.groupby('Station').mean()
taxa.columns = [
    'Chlorophytes',
    'Cryptophytes',
    'Cyanobacteria',
    'Diatoms',
    'Dinoflagellates',
    'Haptophytes',
    'Dictyophytes',
    'Prasinophytes',
    'tchl']

hplc = hplc.drop(['Cruise_Number','Sample_Number','ID', 'ClusterCode','Unnamed: 33'], axis=1)
hplc = hplc.where(hplc['Pressure']<10).dropna()
hplc = hplc.set_index(['Station','Pressure'])
hplc = hplc.groupby('Station').mean()

# calculate dd:chl and dt:chl
dd_dt_norm = pd.DataFrame(hplc.loc[:,['diadino','diato']].values / pd.concat([hplc.loc[:,'T chl a'],hplc.loc[:,'T chl a']], axis=1).values, columns=['Dd','Dt'], index = hplc.index)

# calculate de-epoxidation ratios = Dt / (Dt + Dd)
de_epox1 = {}
for stn in stations:
    de_epox1[stn] =  pd.Series(hplc.loc[stn,'diato'] / (hplc.loc[stn,'diato'] + hplc.loc[stn,'diadino']))
de_epox1_mean = pd.concat(de_epox1)
de_epox1_mean = de_epox1_mean.reset_index().drop('level_1', axis=1).rename({'level_0':'station', 0:'dd_dt_ratio'},axis=1).set_index('station')

taxa_per = {}
for group in taxa.columns:
    taxa_per[group] = (taxa.loc[:,group].copy() / taxa.loc[:,'tchl'].copy())*100
taxa_per = pd.concat(taxa_per, axis=1)
taxa_per = taxa_per.drop('tchl',axis=1)

#%% Load SST anomalies

files = os.listdir(SST_dir)
SST_anom = {}
SST_anom_full = {}
mask = {}
for file in tqdm(files):
    name = file.split('.')[1].split('_')[1][-2:]
    data = xr.open_dataset(SST_dir+file)
    SST_lat = data.coords['lat'].values
    SST_lon = data.coords['lon'].values
    SST = data.sea_surface_temperature_anomaly.values[0,:,:]
    
    SST = pd.DataFrame(SST, index=SST_lat, columns=SST_lon)
    SST.index.name = 'lat'
    SST.columns.name = 'lon'
    
    SST_anom_full[name] = SST
    SST_anom[name] = SST.loc[max_lat+1:min_lat, min_lon:max_lon]
    data.close()
    del data
SST_anom_full = pd.concat(SST_anom_full, names=['date'])
SST_anom_stack = pd.concat(SST_anom, names=['date'])

#------------------------------------------------------------------------------
day_thres = 5
uw_DMS_ind = uw_DMS.loc[uw_DMS.loc[:,['time','lon','lat']].dropna().index]
SSTA_unstack = SST_anom_stack.stack().unstack('date').copy()
dts = pd.Series([pd.Timestamp(i.strftime('%Y-%m-%d')) for i in uw_DMS_ind['time']])

SSTA_time = {'datetime':[],
             'lat':[],
             'lon':[],
             'SST_anom':[],
             }

for d in tqdm(dts.unique()):
    # find past SST values for each point sampled within a certain day range
    date = pd.Timestamp(d).strftime('%Y-%m-%d')
    ds = []
    for di in range(day_thres+1):
        ds.append(str(int(date[-2:])-di).zfill(2))
    # since there are hourly values, find all values within the day that match up
    inds = np.argwhere((d-dts).values == pd.Timedelta(0)).flatten()
    for ind in inds:
        latind = np.argmin(abs(uw_DMS_ind.iloc[ind].loc['lat'] - SSTA_unstack.index.get_level_values('lat')))
        lonind = np.argmin(abs(uw_DMS_ind.iloc[ind].loc['lon'] - SSTA_unstack.index.get_level_values('lon')))
        loc_ = np.argmin(abs(uw_DMS_ind.iloc[ind].loc['lat'] - SSTA_unstack.index.get_level_values('lat'))+abs(uw_DMS_ind.iloc[ind].loc['lon'] - SSTA_unstack.index.get_level_values('lon')))
        SSTA_time['datetime'].append(uw_DMS_ind.iloc[ind].loc['time'])
        SSTA_time['lat'].append(SSTA_unstack.index.get_level_values('lat')[latind])
        SSTA_time['lon'].append(SSTA_unstack.index.get_level_values('lon')[lonind])
        vals = SSTA_unstack.iloc[loc_].loc[ds]
        vals = vals.reset_index()
        vals = vals.drop('date', level=0, axis=1).squeeze()
        SSTA_time['SST_anom'].append(vals)

SSTA_mat = pd.DataFrame(SSTA_time['SST_anom']).reset_index().rename({'level_0':'lat', 'level_1':'lon'}, axis=1)
SSTA_mat.insert(0,'datetime',pd.Series(SSTA_time['datetime']))
SSTA_mat = SSTA_mat.set_index(['datetime','lon','lat'])

#%% Load MHW categories

min_lon, max_lon, min_lat, max_lat = -146, -122, 46, 54
extent = [min_lon, max_lon, min_lat, max_lat]

files = os.listdir(MHW_dir_)

MHW = {}
for file in tqdm(files):
    ds = xr.open_dataset(MHW_dir_+os.sep+file)
    data = pd.DataFrame(ds.heatwave_category.values[0,:,:], index=ds.lat.values, columns=ds.lon.values)
    MHW[file.split('_')[-1].split('.')[0][-2:]] = data.loc[min_lat:max_lat, min_lon:max_lon]
    ds.close()
MHW = pd.concat(MHW).stack()
MHW = MHW.reset_index().rename({'level_0':'date','level_1':'lat','level_2':'lon',0:'category'}, axis=1).set_index(['date','lon','lat']).squeeze()

#------------------------------------------------------------------------------

day_thres = 5
uw_DMS_ind = uw_DMS.loc[uw_DMS.loc[:,['time','lon','lat']].dropna().index]
dts = pd.Series([pd.Timestamp(i.strftime('%Y-%m-%d')) for i in uw_DMS_ind['time']])

MHW_matched = {'datetime':[],
             'lat':[],
             'lon':[],
             'MHW':[],
             'DMS':[],
             }
for d in tqdm(dts.unique()):
    # since there are hourly values, find all values within the day that match up
    date = pd.Timestamp(d).strftime('%d')
    inds = np.argwhere((d-dts).values == pd.Timedelta(0)).flatten()
    for ind in inds:
        latind = np.argmin(abs(uw_DMS_ind.iloc[ind].loc['lat'] - MHW.loc[date].index.get_level_values('lat')))
        lonind = np.argmin(abs(uw_DMS_ind.iloc[ind].loc['lon'] - MHW.loc[date].index.get_level_values('lon')))
        loc_ = np.argmin(abs(uw_DMS_ind.iloc[ind].loc['lat'] - MHW.loc[date].index.get_level_values('lat'))+abs(uw_DMS_ind.iloc[ind].loc['lon'] - MHW.loc[date].index.get_level_values('lon')))
        MHW_matched['datetime'].append(uw_DMS_ind.iloc[ind].loc['time'])
        MHW_matched['DMS'].append(uw_DMS_ind.iloc[ind].loc['conc'])
        MHW_matched['lat'].append(MHW.loc[date].index.get_level_values('lat')[latind])
        MHW_matched['lon'].append(MHW.loc[date].index.get_level_values('lon')[lonind])
        MHW_matched['MHW'].append(MHW.loc[date].iloc[loc_])

MHW_matched = pd.DataFrame(MHW_matched).set_index(['datetime','lon','lat'])

#%% Get averaged MHW categories

# load in statistics generated from "MHW_get_clim_from_raw.py"
MHW_his_stats = xr.open_dataset(MHW_his_stats_dir)
MHW_his_stats.close()

# extract mean and SD values from file as dataframes
sst_mean = pd.DataFrame(MHW_his_stats['sst_mean'].values, index=MHW_his_stats.coords['latitude'].values, columns=MHW_his_stats.coords['longitude'].values)
sst_std = pd.DataFrame(MHW_his_stats['sst_std'].values, index=MHW_his_stats.coords['latitude'].values, columns=MHW_his_stats.coords['longitude'].values)

#%% Spatially & temporally match SSTanom & DMS/P

SST_anom_ship = {}
DMS_anom_ship = {}
for d in SST_anom.keys():
    subset = uw_DMS[uw_DMS['time'].dt.strftime('%d') == d]
    subset = subset.loc[:,['lon','lat']].dropna()#.reset_index().drop('index',axis=1)
    SSTvals = []
    DMSvals = []
    anom_lats = []
    anom_lons = []
    for i,j in zip(subset['lon'], subset['lat']):
        latidx = (np.abs(SST_anom[d].index.values - j)).argmin()
        lonidx = (np.abs(SST_anom[d].columns.values - i)).argmin()
        
        SSTvals.append(SST_anom[d].iloc[latidx,lonidx])
        DMSvals.append(uw_DMS[(uw_DMS['lon']==i) &(uw_DMS['lat']==j)]['conc'].values[0])
        
        anom_lons.append(SST_anom[d].iloc[latidx,:].index[lonidx])
        anom_lats.append(SST_anom[d].iloc[:,lonidx].index[latidx])
        
    SST_anom_ship[d] = pd.DataFrame([anom_lons, anom_lats, SSTvals], index=['lons','lats','SST']).T
    DMS_anom_ship[d] = pd.DataFrame([anom_lons, anom_lats, DMSvals], index=['lons','lats','DMS']).T
    
SST_anom_ship = pd.concat(SST_anom_ship)
DMS_anom_ship = pd.concat(DMS_anom_ship)

SST_anom_ship = SST_anom_ship.set_index(['lons','lats'])
DMS_anom_ship = DMS_anom_ship.set_index(['lons','lats'])
DMS_ship_anom = pd.concat([SST_anom_ship, DMS_anom_ship], axis=1)


SST_anom_ship = {}
DMSP_anom_ship = {}
for d in SST_anom.keys():
    subset = uw_DMSP[uw_DMSP['time'].dt.strftime('%d') == d]
    subset = subset.loc[:,['lon','lat']].dropna()#.reset_index().drop('index',axis=1)
    SSTvals = []
    DMSPvals = []
    anom_lats = []
    anom_lons = []
    for i,j in zip(subset['lon'], subset['lat']):
        latidx = (np.abs(SST_anom[d].index.values - j)).argmin()
        lonidx = (np.abs(SST_anom[d].columns.values - i)).argmin()
        
        SSTvals.append(SST_anom[d].iloc[latidx,lonidx])
        DMSPvals.append(uw_DMSP[(uw_DMSP['lon']==i) & (uw_DMSP['lat']==j)]['conc'].values[0])
        
        anom_lons.append(SST_anom[d].iloc[latidx,:].index[lonidx])
        anom_lats.append(SST_anom[d].iloc[:,lonidx].index[latidx])
        
    SST_anom_ship[d] = pd.DataFrame([anom_lons, anom_lats, SSTvals], index=['lons','lats','SST']).T
    DMSP_anom_ship[d] = pd.DataFrame([anom_lons, anom_lats, DMSPvals], index=['lons','lats','DMSP']).T
    
SST_anom_ship = pd.concat(SST_anom_ship)
DMSP_anom_ship = pd.concat(DMSP_anom_ship)

SST_anom_ship = SST_anom_ship.set_index(['lons','lats'])
DMSP_anom_ship = DMSP_anom_ship.set_index(['lons','lats'])
DMSP_ship_anom = pd.concat([SST_anom_ship, DMSP_anom_ship], axis=1)


#%% load wind speeds

min_lon, max_lon, min_lat, max_lat = -146, -123, 48, 52

wind_dir = r"C:/Users/bcamc/OneDrive/Desktop/2022_cruise_data/wind_speeds/"
files = os.listdir(wind_dir)
wind_anom = {}
mask = {}
for file in tqdm(files):
    name = file.split('_')[-1].split('.')[0][-2:]
    data = xr.open_dataset(wind_dir+file)
    wind_lat = data.coords['lat'].values
    wind_lon = data.coords['lon'].values
    wind_lon = pd.Series(wind_lon).where(wind_lon<180, wind_lon-360).values
    wind = data.windspeed.values[0,0,:,:]
    
    wind = pd.DataFrame(wind, index=wind_lat, columns=wind_lon)
    wind.index.name = 'lat'
    wind.columns.name = 'lon'
    
    wind_anom[name] = wind.loc[min_lat:max_lat, min_lon:max_lon]
    data.close()
    del data

wind_anom_stack = pd.concat(wind_anom, names=['date'])


#%% Spatially & temporally match wind speeds & DMS/P

wind_anom_ship = {}
DMS_anom_ship = {}
wind_means = {}
for d in wind_anom.keys():
    subset = uw_DMS[uw_DMS['time'].dt.strftime('%d') == d]
    subset = subset.loc[:,['lon','lat']].dropna()#.reset_index().drop('index',axis=1)
    mean_windvals = []
    windvals = []
    DMSvals = []
    anom_lats = []
    anom_lons = []
    for i,j in zip(subset['lon'], subset['lat']):
        latidx = (np.abs(wind_anom[d].index.values - j)).argmin()
        lonidx = (np.abs(wind_anom[d].columns.values - i)).argmin()
        
        windvals.append(wind_anom[d].iloc[latidx,lonidx])
        DMSvals.append(uw_DMS[(uw_DMS['lon']==i) &(uw_DMS['lat']==j)]['conc'].values[0])
        
        anom_lons.append(wind_anom[d].iloc[latidx,:].index[lonidx])
        anom_lats.append(wind_anom[d].iloc[:,lonidx].index[latidx])
        
    wind_anom_ship[d] = pd.DataFrame([anom_lons, anom_lats, windvals], index=['lons','lats','wind']).T
    DMS_anom_ship[d] = pd.DataFrame([anom_lons, anom_lats, DMSvals], index=['lons','lats','DMS']).T
    
wind_anom_ship = pd.concat(wind_anom_ship)
DMS_anom_ship = pd.concat(DMS_anom_ship)

wind_anom_ship = wind_anom_ship.set_index(['lons','lats'])
DMS_anom_ship = DMS_anom_ship.set_index(['lons','lats'])
DMS_wind_ship_anom = pd.concat([wind_anom_ship, DMS_anom_ship], axis=1)
del wind_anom_ship, DMS_anom_ship

#%% Find coords and inds for Line P

stn_names = ['P1', 'P4', 'P12', 'P16', 'P20', 'P26']
DMS_coords = {}
for j, stn in enumerate(stn_names):
    ind = np.empty([len(DMS_ship_anom.index),2])
    for i,(lonind,latind) in enumerate(DMS_ship_anom.index):
        ind[i,0] = abs(lonind-LineP_stn_coords[j,1])
        ind[i,1] = abs(latind-LineP_stn_coords[j,0])
    DMS_coords[stn] = np.argmin(np.nansum(ind, axis=1))

DMSP_coords = {}
for j, stn in enumerate(stn_names):
    ind = np.empty([len(DMSP_ship_anom.index),2])
    for i,(lonind,latind) in enumerate(DMSP_ship_anom.index):
        ind[i,0] = abs(lonind-LineP_stn_coords[j,1])
        ind[i,1] = abs(latind-LineP_stn_coords[j,0])
    DMSP_coords[stn] = np.argmin(np.nansum(ind, axis=1))

# pull out uw coordinates
uw_coords_ = uw_DMS.loc[:,['lon','lat']].dropna()

# find OSP and restrict coords to just Line P
ind = np.empty([len(uw_coords_),2])
for i in range(len(uw_coords_)):
    ind[i,0] = abs(uw_coords_['lon'].iloc[i]-LineP_stn_coords[-1,1])
    ind[i,1] = abs(uw_coords_['lat'].iloc[i]-LineP_stn_coords[-1,0])
OSP_loc = np.argmin(np.nansum(ind, axis=1))
uw_coords = uw_coords_.iloc[:OSP_loc]
uw_coords_MVP = uw_coords_.iloc[OSP_loc:]

# Find Line P inds in model predictions
model_inds1 = []
for i,j in zip(DMS_ship_anom.iloc[:DMS_coords['P26'],:].index.get_level_values('lons').values, DMS_ship_anom.iloc[:DMS_coords['P26'],:].index.get_level_values('lats').values):
    ind = ((np.abs(models_combined.index.get_level_values('latbins') - j)+np.abs(models_combined.index.get_level_values('lonbins') - i))).argmin()
    model_inds1.append(ind)

# Find MVP inds in model predictions
model_inds2 = []
for i,j in zip(DMS_ship_anom.iloc[DMS_coords['P26']:,:].index.get_level_values('lons').values, DMS_ship_anom.iloc[DMS_coords['P26']:,:].index.get_level_values('lats').values):
    ind = ((np.abs(models_combined.index.get_level_values('latbins') - j)+np.abs(models_combined.index.get_level_values('lonbins') - i))).argmin()
    model_inds2.append(ind)

#%% Load historic Line P SSN data

files = os.listdir(linep_his_nitrate_dir_)

SSN_his = {}

#### Find SSN
for file in tqdm(files):
    linep_ctd = extract_profile(directory=linep_his_nitrate_dir_+os.sep+file,
                                data_names=['Nitrate_plus_Nitrite:Bottle [µmol/l]',])
    
    SSN = linep_ctd.loc[:,'Nitrate_plus_Nitrite:Bottle [µmol/l]'].where(linep_ctd.index.get_level_values('Pressure:CTD [dbar]')<=10, np.nan).dropna()
    SSN = SSN.reset_index().drop('level_0', axis=1)
    SSN = SSN.rename({'LOC:LATITUDE':'lat', 'LOC:LONGITUDE':'lon', 'LOC:STATION':'station', 'Nitrate_plus_Nitrite:Bottle [µmol/l]':'SSN'}, axis=1)
    SSN = SSN.groupby('station').mean().drop('Pressure:CTD [dbar]', axis=1)
    
    SSN = SSN.sort_values('lon')
    SSN = SSN.reset_index().set_index(['station','lon','lat'])
    
    SSN_his[file.split('-')[0]] = SSN.squeeze()

SSN_his = pd.concat(SSN_his)
SSN_his = SSN_his.reset_index().rename({'level_0':'date'}, axis=1).set_index(['date','station','lon','lat']).squeeze()


stations = [
    'Haro59',
    'JF2',
    'P1',
    'P2',
    'P3',
    'P4',
    'P5',
    'P6',
    'P7',
    'P8',
    'P9',
    'P10',
    'P11',
    'P12',
    'P13',
    'P14',
    'P15',
    'P16',
    'P17',
    'P18',
    'P19',
    'P20',
    'P21',
    'P22',
    'P23',
    'P24',
    'P25',
    'P35',
    'P26',
]

SSN_mean = {}
for stn in stations:
    SSN_mean[stn] = SSN_his.unstack('date').loc[stn].stack()

SSN_mean = pd.concat(SSN_mean).reset_index().rename({'level_0':'station', 0:'SSN'}, axis=1)
SSN_mean = SSN_mean.set_index(['station','lon','lat','date']).stack().squeeze().unstack('station')

#%% Load historic Line P ctd, calculate MLD

files = os.listdir(linep_his_ctd_dir_)

MLDs_his = {}

#### Find SSN
for file in tqdm(files):
    # 2010 data is a bad file - density only available below 700 m, so drop from dataset
    if file != '2010-014-ctd-cruise.csv':
        linep_ctd = extract_profile(directory=linep_his_ctd_dir_+os.sep+file,
                                    data_names=['Temperature:CTD [deg_C_(ITS90)]',
                                                'Salinity:CTD [PSS-78]',
                                                'Sigma-t:CTD [kg/m^3]',])
        #------------------------------------------------------------------------------
        #### Find MLDs
        sigma = linep_ctd.loc[:,'Sigma-t:CTD [kg/m^3]'].unstack('LOC:STATION')
        
        MLD_thres = 0.125
        
        MLDs = {}
        for col in sigma.columns:
            check = sigma.loc[:,col].sort_index().dropna()
            ref_ind = 0
            ind = np.argmin(abs((check - check.iloc[ref_ind]) - MLD_thres))
            MLDs[col] = check.iloc[[ind]]
        MLDs = pd.concat(MLDs).reset_index().drop('level_1', axis=1)
        MLDs = MLDs.rename({'level_0':'station', 'LOC:LATITUDE':'lat', 'LOC:LONGITUDE':'lon', 'Pressure:CTD [dbar]':'depth'}, axis=1)
        MLDs = MLDs.set_index(['station', 'lat', 'lon', 'depth']).squeeze()
        MLDs.name = 'sigma-t'
        
        MLDs = MLDs.sort_index(level='lon')
        MLDs = MLDs.reset_index().drop('sigma-t', axis=1)
        MLDs = MLDs.set_index(['station', 'lon', 'lat'])
        
        # convert the pressure values to depth (m) values
        MLDs = pd.DataFrame(abs(gsw.z_from_p(p=MLDs.values.flatten(),
                                             lat=MLDs.index.get_level_values('lat').values)),
                            index=MLDs.index)
        
        MLDs_his[file.split('-')[0]] = MLDs

MLDs_his = pd.concat(MLDs_his)
MLDs_his = MLDs_his.reset_index().rename({'level_0':'date'}, axis=1).set_index(['date','station','lon','lat']).squeeze()

stations = [
    'Haro59',
    'JF2',
    'P1',
    'P2',
    'P3',
    'P4',
    'P5',
    'P6',
    'P7',
    'P8',
    'P9',
    'P10',
    'P11',
    'P12',
    'P14',
    'P15',
    'P16',
    'P17',
    'P18',
    'P19',
    'P20',
    'P21',
    'P22',
    'P23',
    'P24',
    'P25',
    'P35',
    'P26',
]

MLD_mean = {}
for stn in stations:
    MLD_mean[stn] = MLDs_his.unstack('date').loc[stn].stack()

MLD_mean = pd.concat(MLD_mean).reset_index().rename({'level_0':'station', 0:'MLD'}, axis=1)
MLD_mean = MLD_mean.set_index(['station','lon','lat','date']).stack().squeeze().unstack('station')
MLD_mean = MLD_mean.loc[:,stations]

#%%  Load 2022 Line P CTD & SSN data

linep_ctd = extract_profile(directory=ctd_aug2022_dir,
                                data_names=['Temperature:CTD [deg_C_(ITS90)]',
                                            'Salinity:CTD [PSS-78]',
                                            'Sigma-t:CTD [kg/m^3]',])
#------------------------------------------------------------------------------
#### Find MLDs
sigma = linep_ctd.loc[:,'Sigma-t:CTD [kg/m^3]'].unstack('LOC:STATION')

MLD_thres = 0.125

MLDs = {}
for col in sigma.columns:
    check = sigma.loc[:,col].sort_index().dropna()
    # ref_ind = np.argmin(abs(check.index.get_level_values('Pressure:CTD [dbar]').values-10))
    ref_ind = 0
    ind = np.argmin(abs((check - check.iloc[ref_ind]) - MLD_thres))
    MLDs[col] = check.iloc[[ind]]
MLDs = pd.concat(MLDs).reset_index().drop('level_1', axis=1)
MLDs = MLDs.rename({'level_0':'station', 'LOC:LATITUDE':'lat', 'LOC:LONGITUDE':'lon', 'Pressure:CTD [dbar]':'depth'}, axis=1)
MLDs = MLDs.set_index(['station', 'lat', 'lon', 'depth']).squeeze()
MLDs.name = 'sigma-t'

MLDs = MLDs.sort_index(level='lon')
MLDs = MLDs.reset_index().drop('sigma-t', axis=1)
MLDs = MLDs.set_index(['station', 'lon', 'lat'])

# convert the pressure values to depth (m) values
MLDs = pd.DataFrame(abs(gsw.z_from_p(p=MLDs.values.flatten(),
                                     lat=MLDs.index.get_level_values('lat').values)),
                    index=MLDs.index)

# calculate the MLD anomaly along Line P
MLD_anom = pd.Series(MLDs.loc[stations].values.flatten()-MLD_mean.mean().loc[stations].values, index=MLDs.loc[stations].index)
#------------------------------------------------------------------------------
#### Find salinity

sal = linep_ctd.loc[:,'Salinity:CTD [PSS-78]'].where(linep_ctd.index.get_level_values('Pressure:CTD [dbar]')==10, np.nan).dropna()
sal = sal.reset_index().drop('level_0',axis=1)
sal = sal.rename({'LOC:LATITUDE':'lat', 'LOC:LONGITUDE':'lon', 'LOC:STATION':'station', 'Salinity:CTD [PSS-78]':'sal'}, axis=1).drop('Pressure:CTD [dbar]', axis=1)

sal = sal.groupby('station').mean()
sal = sal.sort_values('lon')
sal = sal.reset_index().set_index(['station','lon','lat'])

#------------------------------------------------------------------------------
#### Find SSN
linep_che = extract_profile(directory=linep_2022_nitrate_dir_,
                                data_names=['Nitrate_plus_Nitrite:Bottle [µmol/l]',])

SSN = linep_che.loc[:,'Nitrate_plus_Nitrite:Bottle [µmol/l]'].where(linep_che.index.get_level_values('Pressure:CTD [dbar]')<=10, np.nan).dropna()
SSN = SSN.reset_index().drop('level_0', axis=1)
SSN = SSN.rename({'LOC:LATITUDE':'lat', 'LOC:LONGITUDE':'lon', 'LOC:STATION':'station', 'Nitrate_plus_Nitrite:Bottle [µmol/l]':'SSN'}, axis=1)
SSN = SSN.groupby('station').mean().drop('Pressure:CTD [dbar]', axis=1)

SSN = SSN.sort_values('lon')
SSN = SSN.reset_index().set_index(['station','lon','lat'])

# calculate the SSN anomaly along Line P
SSN_anom = pd.Series(SSN.loc[stations].values.flatten()-SSN_mean.mean().loc[stations].values, index=SSN.loc[stations].index)
#------------------------------------------------------------------------------

#%% Extract and compute T/S depth anomalies (1956-1990)

#### run this script first to load historic data and average into a baseline
from extract_profile_data_1956_1990 import ctd_his

his_mean = ctd_his.set_index(['station','lat','lon','depth']).groupby(['station','depth']).mean(numeric_only=True)
his_sal_mean = his_mean.loc[:,'sal'].copy()
his_temp_mean = his_mean.loc[:,'temp']

# extracted matching stations between Aug 2022 CTD profiles and historic profiles
stations = ['P26', 'P35', 'P25', 'P24', 'P23', 'P22', 'P21', 'P20', 'P19',
        'P18', 'P17', 'P16', 'P15', 'P14', 'P13', 'P12', 'P11', 'P10',
        'P9', 'P8', 'P7', 'P6', 'P5', 'P4', 'P3', 'P2', 'P1',]

#### convert pressure to depth in 2022 data
linep_ts = linep_ctd.loc[idx[:,:,:,stations],['Temperature:CTD [deg_C_(ITS90)]', 'Salinity:CTD [PSS-78]']].copy()
linep_ts = linep_ts.reset_index().drop('level_0',axis=1)
linep_ts.loc[:,'depth'] = abs(gsw.conversions.z_from_p(linep_ts['Pressure:CTD [dbar]'].values,linep_ts['LOC:LATITUDE'].values))
linep_ts = linep_ts.set_index(['LOC:LATITUDE','LOC:LONGITUDE','depth','LOC:STATION'])

#### bin historic and 2022 data
his_temp_bin = {}
his_sal_bin = {}
temp_bin = {}
sal_bin = {}

bin_width = 1 # i.e. depth bins, in m

for stn in tqdm(stations):
    # historical data:
    his_temp_bin[stn] = bin1d(
        data=his_temp_mean.loc[stn],
        bin_width=bin_width,
        bin_name='depth')
    his_sal_bin[stn] = bin1d(
        data=his_sal_mean.loc[stn],
        bin_width=bin_width,
        bin_name='depth')
    # 2022 data:
    temp_bin[stn] = bin1d(
        data=linep_ts.loc[idx[:,:,:,stn],'Temperature:CTD [deg_C_(ITS90)]'].squeeze(),
        bin_width=bin_width,
        bin_name='depth')
    sal_bin[stn] = bin1d(
        data=linep_ts.loc[idx[:,:,:,stn],'Salinity:CTD [PSS-78]'],
        bin_width=bin_width,
        bin_name='depth')
# format historic data
his_temp_bin = pd.concat(his_temp_bin).reset_index().drop('level_1',axis=1).rename({'level_0':'station'}, axis=1).dropna(subset='depth').set_index(['station','depth']).squeeze()
his_sal_bin = pd.concat(his_sal_bin).reset_index().drop('level_1',axis=1).rename({'level_0':'station'}, axis=1).dropna(subset='depth').set_index(['station','depth']).squeeze()
# format 2022 data
temp_bin = pd.concat(temp_bin).reset_index().drop('level_1',axis=1).rename({'level_0':'station'}, axis=1).dropna(subset='depth').set_index(['station','depth']).squeeze()
sal_bin = pd.concat(sal_bin).reset_index().drop('level_1',axis=1).rename({'level_0':'station'}, axis=1).dropna(subset='depth').set_index(['station','depth']).squeeze()

#### get list of lons to reinsert from 2022 data
SSTA_depth_lons = temp_bin.copy().loc[:,'LOC:LONGITUDE'].reset_index().drop('depth', axis=1).set_index('station')
SSTA_depth_lons = SSTA_depth_lons.groupby('station').mean().loc[stations]

#### get matching depths
# sync up depths to upper 1000 m, and sort stations along transect
his_temp_bin = his_temp_bin.unstack('station')
his_temp_bin = his_temp_bin.loc[0:1000]
his_temp_bin = his_temp_bin.T.loc[stations].T

his_sal_bin = his_sal_bin.unstack('station')
his_sal_bin = his_sal_bin.loc[0:1000]
his_sal_bin = his_sal_bin.T.loc[stations].T

temp_bin = temp_bin.loc[:,'Temperature:CTD [deg_C_(ITS90)]'].unstack('station')
temp_bin = temp_bin.loc[0:1000]
temp_bin = temp_bin.T.loc[stations].T

sal_bin = sal_bin.loc[:,'Salinity:CTD [PSS-78]'].unstack('station')
sal_bin = sal_bin.loc[0:1000]
sal_bin = sal_bin.T.loc[stations].T

#### compute T/S anomalies by depth
temp_depth_anom = temp_bin-his_temp_bin
sal_depth_anom = sal_bin-his_sal_bin

temp_depth_anom.columns = SSTA_depth_lons.values[:,0]
sal_depth_anom.columns = SSTA_depth_lons.values[:,0]

temp_depth_anom = temp_depth_anom.stack()
temp_depth_anom = temp_depth_anom.reset_index().rename({'level_1':'lon', 0:'Temp'},axis=1).set_index(['depth','lon']).squeeze()

sal_depth_anom = sal_depth_anom.stack()
sal_depth_anom = sal_depth_anom.reset_index().rename({'level_1':'lon', 0:'Sal'},axis=1).set_index(['depth','lon']).squeeze()

#%% FINAL: Calculate N^2 (brunt-vaisala frequency)

SA = gsw.SA_from_SP(linep_ctd.loc[:,'Salinity:CTD [PSS-78]'],
                    pd.Series(linep_ctd.index.get_level_values('Pressure:CTD [dbar]'), index=linep_ctd.index),
                    pd.Series(linep_ctd.index.get_level_values('LOC:LONGITUDE').values, index=linep_ctd.index),
                   pd.Series(linep_ctd.index.get_level_values('LOC:LATITUDE').values, index=linep_ctd.index))
CT = gsw.CT_from_t(SA,
                   linep_ctd.loc[:,'Temperature:CTD [deg_C_(ITS90)]'],
                   pd.Series(linep_ctd.index.get_level_values('Pressure:CTD [dbar]'), index=linep_ctd.index),)

# returns N^2 and pressure mid points - save only first array
N_square = gsw.stability.Nsquared(SA,
                          CT,
                           pd.Series(linep_ctd.index.get_level_values('Pressure:CTD [dbar]'), index=linep_ctd.index),
                           lat=pd.Series(linep_ctd.index.get_level_values('LOC:LATITUDE').values, index=linep_ctd.index))

N_sq = pd.DataFrame(N_square[0], index=linep_ctd.index[:-1], columns=['N_sq'])
N_sq = N_sq.reset_index().drop('level_0',axis=1)
N_sq['Pressure:CTD [dbar]'] = N_square[1]

#### convert pressure to depth in 2022 data
N_sq.loc[:,'depth'] = abs(
    gsw.conversions.z_from_p(
        N_sq['Pressure:CTD [dbar]'].values,
        N_sq['LOC:LATITUDE'].values)
        )
N_sq = N_sq.set_index(['LOC:LATITUDE', 'LOC:LONGITUDE', 'Pressure:CTD [dbar]', 'depth', 'LOC:STATION'])
N_sq = N_sq.squeeze().groupby(['LOC:LATITUDE','LOC:LONGITUDE','Pressure:CTD [dbar]', 'depth', 'LOC:STATION']).mean()

#%% line P - extract 8d data (chl, bbp, PAR, kd)

dts_tz = pd.Series([pd.Timestamp(i, tz='UTC') for i in uw_DMS_ind['time']])

dirs_ = {'chl': r'H:/8_day/2022/chl',
         'PAR': r'H:/8_day/2022/PAR',
         'bbp': r'H:/8_day/2022/bbp',
         'Kd': r'H:/8_day/2022/Kd',
         'CDOM':r'H:/8_day/2022/CDOM',
         'nFLH':r'H:/8_day/2022/FLH',
         'SST':r'H:/8_day/2022/SST',
         'SSHA':r'H:/daily/2022/SSHA'}

var_names = {'chl': 'chlor_a',
         'PAR': 'par',
         'bbp': 'bbp_443',
         'Kd': 'Kd_490',
         'CDOM':'adg_443',
         'nFLH':'nflh',
         'SST':'sst',
         'SSHA':'SLA'}

matched_8d = {
    'datetime':[],
    'lat':[],
    'lon':[],
    'DMS':[],
    'chl':[],
    'PAR':[],
    'bbp':[],
    'Kd':[],
    'CDOM':[],
    'nFLH':[],
    'SST':[],
    'SSHA':[]}

for i, var_ in enumerate(dirs_.keys()):
    dir_ = dirs_[var_]
    files = os.listdir(dir_)
    for j, file in tqdm(enumerate(files)):
        var_name = var_names[var_]
        rawdata = xr.open_dataset(dir_+os.sep+file)
        # pull out PAR values
        if var_ == 'SSHA':
            lon = pd.Series(rawdata[var_name].Longitude.values)
            lon = lon.where(lon<180, lon-360).values
            data = pd.DataFrame(rawdata[var_name].values[0,:,:],
                                columns=lon,
                                index=rawdata[var_name].Latitude.values)
            # get the start and end dates of the PAR dataset
            d1 = pd.Timestamp(pd.Timestamp(rawdata.Time.values[0]), tz='utc')
            d2 = d1 + pd.Timedelta(5, 'd')
        else:
            lon = pd.Series(rawdata[var_name].lon.values)
            data = pd.DataFrame(rawdata[var_name].values,
                                columns=lon,
                                index=rawdata[var_name].lat.values)
            # get the start and end dates of the dataset
            d1 = pd.Timestamp(rawdata.attrs['time_coverage_start'])
            # for end dates: MODIS data has overlap between 8-day periods, so use the start date of the next 8-day window as the end date of the current window. For last file in directory, just use start and end dates listed in the netCDF file.
            if j == len(files)-1: # if last file in directory
                d2 = pd.Timestamp(rawdata.attrs['time_coverage_end'])
            else: # set end date as the start date of the next 8-day window
                df = xr.open_dataset(dir_+os.sep+files[j+1])
                d2 = pd.Timestamp(df.attrs['time_coverage_start'])
                df.close()
            
        data = data.T.sort_index().T
        # find the DMS values that have a timestamp within the 8d range of PAR data
        inds = np.argwhere(((dts_tz-d1) >= pd.Timedelta(0,'d')).values & ((dts_tz-d2) < pd.Timedelta(0,'d')).values).flatten()
        for ind in inds:
            latind = np.argmin(abs(uw_DMS_ind.iloc[ind].loc['lat'] - data.index.values))
            lonind = np.argmin(abs(uw_DMS_ind.iloc[ind].loc['lon'] - data.columns.values))
            matched_8d[var_].append(data.iloc[latind, lonind])
            if i == 0:
                matched_8d['datetime'].append(uw_DMS_ind.iloc[ind]['time'])
                matched_8d['lat'].append(data.index.values[latind])
                matched_8d['lon'].append(data.columns.values[lonind])
                matched_8d['DMS'].append(uw_DMS_ind.iloc[ind].loc['conc'])
        rawdata.close()

matched_8d = pd.DataFrame(matched_8d)

matched_8d.loc[:,'nFLH:Chl-a'] = matched_8d['nFLH']/matched_8d['chl']

#%% Find coords and inds for Line P

stn_names = ['P1', 'P4', 'P12', 'P16', 'P20', 'P26']

# Find Line P inds in model predictions
inds = []
for i,j in zip(uw_coords['lon'], uw_coords['lat']):
    if ~np.isnan(i) and ~np.isnan(j):
        ind = ((np.abs(models_combined.index.get_level_values('latbins') - j)+np.abs(models_combined.index.get_level_values('lonbins') - i))).argmin()
        inds.append(ind)
        
# Find MVP inds in model predictions
inds2 = []
for i,j in zip(uw_coords_MVP['lon'], uw_coords_MVP['lat']):
    if ~np.isnan(i) and ~np.isnan(j):
        ind = ((np.abs(models_combined.index.get_level_values('latbins') - j)+np.abs(models_combined.index.get_level_values('lonbins') - i))).argmin()
        inds2.append(ind)

#%% Interpolate CTD data to force models

# find the value corresponding to OSP / P26
matched_loc = matched_8d[matched_8d.loc[:,'datetime'] == pd.Timestamp('2022-08-19 01:38:18.491940')].index[0]
interp_index = matched_8d.loc[:matched_loc].copy().set_index(['datetime','lon','lat']).index

#------------------------------------------------------------------------------
# interpolate data
f = scipy.interpolate.interp1d(SSN.index.get_level_values('lon').values, SSN.values[:,0])
SSN_interpd = f(matched_8d.loc[:matched_loc, 'lon'])
SSN_interpd = pd.Series(SSN_interpd,
                        index=interp_index,
                        name='SSN')

f = scipy.interpolate.interp1d(sal.index.get_level_values('lon').values, sal.values[:,0])
sal_interpd = f(matched_8d.loc[:matched_loc, 'lon'])
sal_interpd = pd.Series(sal_interpd,
                        index=interp_index,
                        name='Salinity')



f = scipy.interpolate.interp1d(MLDs.index.get_level_values('lon').values, MLDs.values[:,0])
MLD_interpd = f(matched_8d.loc[:matched_loc, 'lon'])
MLD_interpd = pd.Series(MLD_interpd,
                        index=interp_index,
                        name='MLD')

# matched_8d_nona = pd.concat([matched_8d, MLD_interpd.reset_index()], axis=1)
# matched_8d_nona = matched_8d_nona.dropna()
matched_8d_nona = matched_8d.copy().dropna()

#%% Force ML models w/ aug 2022 data (8-day data)
# get the values up to OSP
X_linep = matched_8d.loc[:matched_loc,:].copy()
X_linep = X_linep.set_index(['datetime','lon', 'lat'])

#==============================================================================
#### calculate NPP using the VGPM algorithm
lats = X_linep.index.get_level_values('lat').values
yDays = pd.Series([pd.Timestamp(i).day_of_year for i in X_linep.index.get_level_values('datetime').values])
yDays = yDays.astype(float)
dayL = LatToDayLength(lats, yDays)
npp = calculate_NPP(X_linep.loc[:,'chl'], X_linep.loc[:,'PAR'], X_linep.loc[:,'SST'], dayL)
#### calculate NCP
X_linep.loc[:,'NCP'] = (8.57*npp)/(17.9+X_linep.loc[:,'SST'])
#==============================================================================

# insert interpolated data from Line P
X_linep.loc[:,'MLD'] = MLD_interpd.values
X_linep.loc[:,'SSN'] = SSN_interpd.values
X_linep.loc[:,'Salinity'] = sal_interpd.values
# rename to match training dataset
X_linep = X_linep.rename({'chl':'Chlorophyll a', 'CDOM':'a$_{cdm}$(443)'}, axis=1)

# get the closest matching wind speeds
ds = pd.Series(X_linep.index.get_level_values('datetime').strftime('%d'))
matched_winds = []
for d in wind_anom.keys():
    w_inds = ds[ds == d].index
    for i,j in zip(X_linep.iloc[w_inds,:].index.get_level_values('lon'), X_linep.iloc[w_inds,:].index.get_level_values('lat')):
        latidx = np.nanargmin(abs(wind_anom[d].index.values - j))
        lonidx = np.nanargmin(abs(wind_anom[d].columns.values - i))
        matched_winds.append(wind_anom[d].iloc[latidx, lonidx])
X_linep.loc[:,'Wind Speed'] = matched_winds

# Drop nans in DMS data
X_linep = X_linep.dropna(subset='DMS')
# seperate DMS data from input predictor data
y_linep = X_linep.loc[:,'DMS'].copy()

# Gap-fill nans with linear interpolation
X_linep = X_linep.interpolate(method='linear', axis=0)

# copy to run on literature algorithms
X_lit = X_linep.copy()
X_lit = X_lit.rename({'Chlorophyll a':'chl', 'Kd':'kd'}, axis=1)
X_lit.index = X_lit.index.rename({'lon':'lonbins','lat':'latbins'})

#%% Compute literature algorithms and compare

# X_lit = pd.concat([aug_PAR.reindex(X_anom.index).loc[8],
#                     aug_FLH_chl.reindex(X_anom.index).loc[8],
#                     aug_chl.reindex(X_anom.index).loc[8],
#                     aug_kd.reindex(X_anom.index).loc[8],
#                     aug_SST.reindex(X_anom.index).loc[8],
#                     aug_MLD.reindex(X_anom.index).loc[8],
#                     X_anom.loc[:,'SSN'].loc[8].to_frame()], axis=1)

#------------------------------------------------------------------------------

global_coefs = np.array([5.7, 55.8, 0.6])
def SD02_model(X, a,b,c):
    coefs = np.array([a,b,c])
    Chl = X.loc[:,['chl']].values
    MLD = X.loc[:,['MLD']].values
    Chl_MLD = Chl/MLD
    SD02 = np.empty([Chl.shape[0],Chl.shape[1]])
    for i, val in enumerate(Chl_MLD):
        if val < 0.02:
            SD02[i,0] = -np.log(MLD[i])+coefs[0]
        elif val >= 0.02:
            SD02[i,0] = coefs[1]*(Chl_MLD[i])+coefs[2]
    SD02 = SD02[:,0]
    return SD02

SD02 = SD02_model(X_lit, global_coefs[0], global_coefs[1], global_coefs[2])
SD02 = pd.Series(SD02, index=X_lit.index)

global_coefs = np.array([0.492,0.019])
def VS07_model(X, a, b):
    coefs = np.array([a,b])
    PAR = X.loc[:,['PAR']].values
    Kd = X.loc[:,['kd']].values
    MLD = X.loc[:,['MLD']].values
    z = MLD # surface depth in m
    SRD = (PAR/(Kd*MLD))*(1-np.exp(-Kd*z))
    VS07 = coefs[0]+(coefs[1]*SRD)
    VS07 = VS07[:,0]
    return VS07

VS07 = VS07_model(X_lit, global_coefs[0], global_coefs[1])
VS07 = pd.Series(VS07, index=X_lit.index)

# First run model with global coefs from paper:
global_coefs = np.array([-1.237,0.578,0.0180])
def G18_model(X,a,b,c):
    coefs = np.array([a,b,c])
    Kd = X.loc[:,['kd']].values.reshape(-1,1)
    MLD = X.loc[:,['MLD']].values
    Chl = X.loc[:,['chl']].values
    # Chl[Chl<=0.4] = 0.4
    # Chl[Chl>=60] = 60
    SST = X.loc[:,['SST']].values
    PAR = X.loc[:,['PAR']].values
    
    Z_eu = 4.6/Kd # euphotic layer depth
    Z_eu_MLD = Z_eu/MLD
    DMSPt = np.empty([MLD.shape[0], MLD.shape[1]])
    for i,val in enumerate(Z_eu_MLD):
        if val >= 1:
            DMSPt[i,0] = (1.70+(1.14*np.log10(Chl[i]))\
                              +(0.44*np.log10(Chl[i]**2))\
                                  +(0.063*SST[i])-(0.0024*(SST[i]**2)))
        elif val < 1:
            DMSPt[i,0] = (1.74+(0.81*np.log10(Chl[i]))+(0.60*np.log10(Z_eu_MLD[i])))
    G18 = coefs[0]+(coefs[1]*DMSPt)+(coefs[2]*PAR)
    G18 = 10**(G18[:,0])
    return G18

G18 = G18_model(X_lit, global_coefs[0],global_coefs[1],global_coefs[2])
G18 = pd.Series(G18, index=X_lit.index)

global_coefs = np.array([0.06346,0.1210,14.11,6.278])
def W07_model(X,a,b,c,d):
    coefs = np.array([a,b,c,d])
    L = X.index.get_level_values('latbins').values
    SST = X.loc[:,['SST']].values+273.15 # units are in Kelvin
    SSN = X.loc[:,['SSN']].values
    W07 = np.log(np.exp((coefs[0]*SST)-(coefs[1]*SSN)-(coefs[2]*np.cos(L.reshape(-1,1).astype(float)))-coefs[3]))
    W07 = W07[:,0]
    return W07

W07 = W07_model(X_lit, global_coefs[0], global_coefs[1],global_coefs[2],global_coefs[3])
W07 = pd.Series(W07, index=X_lit.index)

print(r2_score(y_linep, SD02))
print(r2_score(y_linep, VS07))
print(r2_score(y_linep, G18))
print(r2_score(y_linep, W07))

print(np.sqrt(metrics.mean_squared_error(np.sinh(y_linep), SD02)))
print(np.sqrt(metrics.mean_squared_error(np.sinh(y_linep), VS07)))
print(np.sqrt(metrics.mean_squared_error(np.sinh(y_linep), G18)))
print(np.sqrt(metrics.mean_squared_error(np.sinh(y_linep), W07)))

#%% Load in Line P experiment DMS data

DMS = {}
DMS['P4'] = pd.read_excel(incub_dir, sheet_name='P4_Init', skiprows=31, header=0).iloc[:15,:6].dropna()
DMS['P16'] = pd.read_excel(incub_dir, sheet_name='P16_Init', skiprows=31, header=0).iloc[:15,:6].dropna()
DMS['P20'] = pd.read_excel(incub_dir, sheet_name='P20_Init', skiprows=31, header=0).iloc[:15,:6].dropna()
DMS['P26'] = pd.read_excel(incub_dir, sheet_name='P26_Init', skiprows=31, header=0).iloc[:15,:6].dropna()

stns = ['P4', 'P16', 'P20', 'P26']
DMS_tref = {}
for stn in stns:
    DMS_tref[stn] = pd.Series(np.tile(pd.read_excel(incub_dir, sheet_name=stn+'_Init', header=None).iloc[0,1], len(DMS[stn])))

DMS['P12'] = pd.read_excel(incub_dir, sheet_name='P16_Fe', skiprows=47, header=0).iloc[:32,:6].dropna()
DMS['P16'] = pd.concat([DMS['P16'],
                        pd.read_excel(incub_dir, sheet_name='P16_Fe', skiprows=47, header=0).iloc[:32,:6].dropna()
                        ], axis=0)
DMS['P20'] = pd.concat([DMS['P20'],
                        pd.read_excel(incub_dir, sheet_name='P20_Fe', skiprows=47, header=0).iloc[:32,:6].dropna()
                        ], axis=0)
DMS['P26'] = pd.concat([DMS['P26'],
                        pd.read_excel(incub_dir, sheet_name='P26_Fe', skiprows=56, header=0).iloc[:40,:6].dropna()
                        ], axis=0)

for stn in DMS.keys():
    DMS[stn] = DMS[stn].reset_index()
    treatment = []
    time = []
    rep = []
    for i in range(DMS[stn].shape[0]):
        time.append(DMS[stn].loc[i,'Replicate'].split('_')[0])
        treatment.append(DMS[stn].loc[i,'Replicate'].split('_')[1])
        rep.append(DMS[stn].loc[i,'Replicate'].split('_')[2])
    DMS[stn] = DMS[stn].drop('Replicate', axis=1)
    DMS[stn].insert(loc=0,column='replicate', value=rep)
    DMS[stn].insert(loc=0,column='time', value=time)
    DMS[stn].insert(loc=0,column='treatment', value=treatment)
    DMS[stn] = DMS[stn].set_index(['treatment','time','replicate'])
    DMS[stn] = DMS[stn].drop('index', axis=1)
DMS = pd.concat(DMS).reset_index().rename({'level_0':'station'},axis=1).set_index(['station','treatment','time','replicate'])
DMS = DMS.rename({'Time':'Sampling Time'}, axis=1)

# get Line P expt dates to time match PAR
LineP_dates = {}
LineP_dates['P4'] = '2022-08-14'
LineP_dates['P12'] = '2022-08-16'
LineP_dates['P16'] = '2022-08-18'
LineP_dates['P20'] = '2022-08-20'
LineP_dates['P26'] = '2022-09-21'
stations = LineP_dates.keys()

dates = {}
for stn in stations:
    dates[stn] = [pd.to_datetime(LineP_dates[stn]+' '+i.strftime('%H:%M')) for i in DMS.loc[idx[stn,:],'Sampling Time']]
    dates[stn] = pd.Series(dates[stn]).astype('datetime64[ns]')
dates = pd.Series(pd.concat(dates).values, index=DMS.index)


# Get start times of experiments for reference
stations = DMS.index.get_level_values('station').unique().values
treatments = DMS.index.get_level_values('treatment').unique().values
tref = {}
for stn in stations:
    for treatment in treatments:
        try:
            tref[stn] = pd.Series(np.tile(pd.read_excel(incub_dir, sheet_name=stn+'_'+treatment, header=None).iloc[0,1], len(DMS.loc[stn])))
        except:
            pass
tref = pd.Series(pd.concat(tref).values, index=DMS.index)

for i in range(len(tref)):
    tref.iloc[i] = pd.to_datetime(dates.iloc[i].strftime('%Y-%m-%d')+' '+tref.iloc[i].strftime('%H:%M'))

#%% Get DMS/O/P turnover rates
#------------------------------------------------------------------------------
# DMS cycling
output = get_rates(data=DMS,
                    tref=tref,
                    t_unit='hr',
                    treatments=['Init','CRL','Fe','HL','DCMU'])
DMS_rates, DMS_rates_SE, DMS_raw_rates, DMS_hr, models, r2 = output['rates'], output['rate_SE'], output['raw_rates'], output['time-processed data'], output['models'], output['r2']
DMS_rates_sd = DMS_raw_rates.groupby(['station','treatment']).std()
print(DMS_rates)
#------------------------------------------------------------------------------

# sort from coast to offshelf
DMS_rates = DMS_rates.loc[(['P4','P16','P20','P26']),:]
DMS_rates_sd = DMS_rates_sd.loc[(['P4','P16','P20','P26']),:]
DMS_hr = DMS_hr.loc[(['P4','P16','P20','P26']),:]

#%% Extract experiment FRRF data

# choose parameters to extract
params = ['ChiSq', 'Fo', 'Fm', 'Fv', 'Fv/Fm', 'Fv/Fo','Sig', 'p', 'Alp1QA', 'Tau1QA', 'Alp2QA', 'Tau2QA', 'Alp3QA', 'Tau3QA', 'Alpha', 'carQ','carQt','p680Q', 'p680Qt','SNR_raw']

# Extract data
FRRF_raw_all = {}
for j, directory in enumerate(frrf_dirs_):
    FRRF_raw_all[frrf_dirs_[j].split('/')[-1]] = FRRF_extract(directory, params)

#%% Average across FRRF psuedoreps

idx = pd.IndexSlice
FRRF_data = {}
for stn in FRRF_raw_all:
    FRRF_raw = FRRF_raw_all[stn].copy()
    FRRF_raw = FRRF_raw.sort_index()
    # drop gain rows
    FRRF_raw.reset_index().dropna().set_index(['time','treatment','replicate','acclim','pseudorep'])

    # Filter out first two psudeoreps
    FRRF_raw.loc[idx[:,:,:,:,0:1],:] = np.nan
    # Remove low S/N values
    FRRF_raw[FRRF_raw.loc[:,'SNR_raw']<=10] = np.nan
    FRRF_raw = FRRF_raw.dropna()
    
    # Average across pseudoreps
    FRRF_mean = FRRF_raw.groupby(['time','treatment','replicate','acclim']).mean()
    FRRF_mean = FRRF_mean.reset_index().set_index(['acclim','treatment','time','replicate'])
    FRRF_data[stn] = FRRF_mean
    
FRRF_data_all = pd.concat(FRRF_data)
FRRF_data_all = FRRF_data_all.reset_index().rename({'level_0':'station'},axis=1).set_index(['station','acclim','treatment','time','replicate'])
FRRF_data_all = FRRF_data_all.dropna()
FRRF_data_all = FRRF_data_all.sort_index()

#%% Prep tracer / FRRF data to plot

DMS_nat_rates = DMS_rates.loc[idx[:,'Init'],'63_conc_nM'].droplevel('treatment').reindex(['P4','P16','P20','P26'])
DMS_rate_stns = pd.DataFrame(LineP_stn_coords[[1,3,4,5],:], columns=['lat','lon'], index=DMS_nat_rates.index)
DMS_nat_rates = pd.concat([DMS_rate_stns,DMS_nat_rates], axis=1)

DMS_nat_rates_sd = DMS_rates_sd.loc[idx[:,'Init'],'63_conc_nM'].droplevel('treatment')
DMS_nat_rates_sd = pd.concat([DMS_rate_stns,DMS_nat_rates_sd], axis=1)

DMS_tracer_rates = DMS_rates.loc[idx[:,'Init'],['66_conc_nM','69_conc_nM','71_conc_nM']].droplevel('treatment').reindex(['P4','P16','P20','P26'])
DMS_tracer_rates = pd.concat([DMS_rate_stns,DMS_tracer_rates], axis=1)

DMS_tracer_rates_sd = DMS_rates_sd.loc[idx[:,'Init'],['66_conc_nM','69_conc_nM','71_conc_nM']].droplevel('treatment')
DMS_tracer_rates_sd = pd.concat([DMS_rate_stns,DMS_tracer_rates_sd], axis=1)

FvFm_mean = FRRF_data_all.loc[idx[:,'dark','Init',:], 'Fv/Fm'].groupby(['station','acclim','treatment']).mean().droplevel(['acclim','treatment']).reindex(['P4','P16','P20','P26'])
FvFm_mean = pd.concat([DMS_rate_stns,FvFm_mean], axis=1)
FvFm_mean = FvFm_mean.reset_index().set_index(['station','lon','lat'])

FvFm_sd = FRRF_data_all.loc[idx[:,'dark','Init',:], 'Fv/Fm'].groupby(['station','acclim','treatment']).std().droplevel(['acclim','treatment']).reindex(['P4','P16','P20','P26'])
FvFm_sd= pd.concat([DMS_rate_stns,FvFm_sd], axis=1)
FvFm_sd = FvFm_sd.reset_index().set_index(['station','lon','lat'])

#%% Statistics

DMS_OSP_loc = uw_DMS[uw_DMS['time'] == pd.Timestamp('2022-08-19 02:04:58.767940')].index[0]

stations = [
    'P1',
    'P2',
    'P3',
    'P4',
    'P5',
    'P6',
    'P7',
    'P8',
    'P9',
    'P10',
    'P11',
    'P12',
    'P14',
    'P15',
    'P16',
    'P17',
    'P18',
    'P19',
    'P20',
    'P21',
    'P22',
    'P23',
    'P24',
    'P25',
    'P35',
    'P26',
]

# need to calculate DMS:chl ratios first (used later in plotting)
MHW_chl_matched = MHW_matched.loc[matched_8d_nona['datetime']].copy()
MHW_chl_matched['DMS:chl'] = pd.Series(MHW_chl_matched['DMS'].values / matched_8d_nona['chl'].values, index=MHW_chl_matched.index)

#------------------------------------------------------------------------------
#### Compute U-test for DMS and DMS:chl in ambient and MHW waters

# levenes test used to test equal variances - failed
scipy.stats.levene(MHW_chl_matched.mask(MHW_chl_matched['MHW']==0).dropna()['DMS'],
                   MHW_chl_matched.mask(MHW_chl_matched['MHW']==1).dropna()['DMS'])

# equal varainces assumption failed, use non-parameteric t-test
print('U-test DMS =', scipy.stats.mannwhitneyu(MHW_chl_matched.mask(MHW_chl_matched['MHW']==0).dropna()['DMS'],
                         MHW_chl_matched.mask(MHW_chl_matched['MHW']==1).dropna()['DMS']))

print('U-test DMS:chl =', scipy.stats.mannwhitneyu(MHW_chl_matched.mask(MHW_chl_matched['MHW']==0).dropna()['DMS:chl'],
                         MHW_chl_matched.mask(MHW_chl_matched['MHW']==1).dropna()['DMS:chl']))

#------------------------------------------------------------------------------
#### Compute correlations between DMS and MLD anomalies

# get values matching up with DMS data (practically excludes some early line stations in haro strait)
# index [1:] below excludes station P1, where there was a faulty DMS measurement
MLD_anom_inds = []
for i,j in zip(MLD_anom.loc[stations[1:]].index.get_level_values('lon'), MLD_anom.loc[stations[1:]].index.get_level_values('lat')):
    if ~np.isnan(i) and ~np.isnan(j):
        ind = np.nanargmin((np.abs(uw_DMS.iloc[:DMS_OSP_loc,:].loc[:,'lat'] - j)+np.abs(uw_DMS.iloc[:DMS_OSP_loc,:].loc[:,'lon'] - i)))
        MLD_anom_inds.append(ind)

# compute correlations
print('\nMLD ~ DMS; r =', pearsonr(MLD_anom.loc[stations[1:]], uw_DMS.loc[MLD_anom_inds, 'conc']))
print('MLD ~ DMS; rho =', spearmanr(MLD_anom.loc[stations[1:]], uw_DMS.loc[MLD_anom_inds, 'conc']))

#------------------------------------------------------------------------------
#### Compute correlation between SST and SSN anomalies

# using the in situ, rather than satellite-based, SST anomalies here
print('\nSSTA ~ SSNA; r =', pearsonr(temp_depth_anom.loc[0:10].groupby('lon').mean().loc[SSTA_depth_lons.loc[stations].values[:,0]].values,
                                   SSN_anom.loc[idx[stations,:]].values))

#------------------------------------------------------------------------------
#### Compute the correlations between SST anomalies and surface water haptophye abundance

print('\nSSTA ~ haptos; r =', pearsonr(temp_depth_anom.loc[0:10].groupby('lon').mean().loc[SSTA_depth_lons.loc[stations].values[:,0]],
                                    taxa_per.loc[stations,'Haptophytes']))

#%% Calculate analysis run time
analysis_end = timeit.default_timer()
analysis_runtime = analysis_end-analysis_start
print('\nAnalysis Runtime:')
print(str(round(analysis_runtime,5)),'secs')
print(str(round((analysis_runtime)/60,5)),'mins')
print(str(round((analysis_runtime)/3600,5)),'hrs')
