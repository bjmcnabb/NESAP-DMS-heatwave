# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:15:19 2020

@author: Brandon McNabb (bmcnabb@eoas.ubc.ca)
"""

#%% Start timer
import timeit
analysis_start = timeit.default_timer()
#%% Import Packages
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter
# from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, inset_axes
# from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import scipy
from scipy.stats import pearsonr#, spearmanr
# import cartopy
# import cartopy.crs as ccrs
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.decomposition import PCA
import os
# import sklearn
# from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split#, GridSearchCV, ShuffleSplit
# from sklearn.pipeline import Pipeline
# from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn import metrics
# from datetime import datetime
# from sklearn import tree
# from sklearn import linear_model
# import datetime
# import seaborn as sns
# import dask.array as da
# import dask.dataframe as dd
# from joblib import parallel_backend
# import cartopy.io.shapereader as shpreader
# import shapely.geometry as sgeom
# from shapely.ops import unary_union
# from shapely.prepared import prep
import joblib
# from joblib import Parallel, delayed 

from tqdm import tqdm

# import seawater as sw
# import gsw

# Custom taylor diagram script
from taylorDiagram import TaylorDiagram

#%% Define Region & File Paths

#### Spatial grid resolution (degrees):
grid = 0.25

#### Retrieve DMS data, call file directory
# All DMS data from the PMEL repository for summertime months
write_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/Sulfur/NESAP/NESAP_Data_res/'
PMEL = pd.read_csv('C:/Users/bcamc/OneDrive/Desktop/Python/Projects/Sulfur/NESAP/NESAP_Data_res/PMEL_Full_NEPacific.csv')

#### Define lat/lon constraints
min_lon, max_lon, min_lat, max_lat = -180, -122, 40, 61

#### Define destination to save figures
save_to_path = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/Sulfur/NESAP/Figures/ANN_RFR_NESAP_Final/'

#### Define bins
latbins = np.arange(min_lat,max_lat+grid,grid)
lonbins = np.arange(min_lon,max_lon+grid,grid)

#### Choose whether to build models or load them from memory
load_from_memory = True

#%% Switch to directory with nc files, if not already in it

dir_ = r'C:\Users\bcamc\OneDrive\Desktop\Python\projects\sulfur\NESAP\scripts'

if os.getcwd() != dir_:
    os.chdir(dir_)
#%% Load PMEL data
# filter out garbage data
PMEL = PMEL.replace(-999,np.nan)
#-----------------------------------------------------------------------------
# Extract variables
PMEL_lat = PMEL['Lat']
PMEL_lon = PMEL['Lon']
#-----------------------------------------------------------------------------
# Print metadata
print()
print('Coordinates for MIMS PMEL data:')
print('oW: ' + str([PMEL_lon.min(), PMEL_lon.max(), PMEL_lat.min(), PMEL_lat.max()]))
print('oE: ' + str([360+PMEL_lon.min(), 360+PMEL_lon.max(), PMEL_lat.min(), PMEL_lat.max()]))
print()
print('MIMS PMEL Date range:')
print(PMEL['DateTime'].min() + ' to ' + PMEL['DateTime'].max())
print()
print('Chosen satellite coordinates:')
print('-147, -122, 43, 60')
print('-180, -122, 40, 61')
print()

#%% Clean-up PMEL data
#-----------------------------------------------------------------------------
track_removed_data = np.empty(4)

# Remove NaNs (later, interpolate through!)
data_proc = PMEL.loc[:,['DateTime','Lat','Lon','swDMS','SST', 'SAL']]
track_removed_data[3] = data_proc.shape[0]

# Redefine columns as float data_proc type to be readable by binning functions:
data_proc['DateTime'] = pd.to_datetime(data_proc['DateTime']).values.astype('float64')

#-----------------------------------------------------------------------------

#### Bin the data

# Bin data as averages across 1-m bins by sampling date:
# data_proc = data_proc.groupby(['DateTime', pd.cut(idx, bins)]).mean()
to_bin = lambda x: np.round(x /grid) * grid
data_proc['latbins'] = data_proc.Lat.map(to_bin)
data_proc['lonbins'] = data_proc.Lon.map(to_bin)
data_proc = data_proc.groupby(['DateTime', 'latbins', 'lonbins']).mean()

# Rename binned columns + drop mean lat/lons:
data_proc = data_proc.drop(columns=['Lat','Lon'])
data_proc = data_proc.rename_axis(index=['DateTime', 'Lat', 'Lon'])

# Transform dates back from integers to datetime numbers:
data_proc.reset_index(inplace=True) # remove index specification on columns
data_proc['DateTime'] = pd.to_datetime(data_proc['DateTime'],format=None)

# Filter to restrict only to summertime months (june, july, august)
data_proc = data_proc[(data_proc['DateTime'].dt.month >= 6) & (data_proc['DateTime'].dt.month <= 8)]
# Pull unique dates from data
unique_dates = np.unique(data_proc['DateTime'].dt.strftime('%Y-%m'))

# Filter out dates older than satellite/argo data can be sourced for all variables:
if write_dir == 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/Sulfur/NESAP/NESAP_Data_res/':
    track_removed_data[2] = data_proc[data_proc['DateTime'].dt.strftime('%Y%m') < '1998-06'].shape[0]
    data_proc = data_proc[data_proc['DateTime'].dt.strftime('%Y%m') > '1998-06']
#-----------------------------------------------------------------------------
#### Reshape each variable by time x space for PCA

# Pivot to move lat/lon pairs to columns - this is still at mins resolution temporally
# reset_index pulls the dates back into a column
SST = data_proc.pivot(index='DateTime',columns=['Lat','Lon'], values='SST').reset_index()
SAL = data_proc.pivot(index='DateTime',columns=['Lat','Lon'], values='SAL').reset_index()
DMS = data_proc.pivot(index='DateTime',columns=['Lat','Lon'], values='swDMS').reset_index()

# Now bin rows into months
SST = SST.groupby(SST['DateTime'].dt.strftime('%m')).mean(numeric_only=True)
SAL = SAL.groupby(SAL['DateTime'].dt.strftime('%m')).mean(numeric_only=True)
DMS = DMS.groupby(DMS['DateTime'].dt.strftime('%m')).mean(numeric_only=True)
#------------------------------------------------------------------------------
#### Interpolate through the NaNs
func='inverse'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DMS

# stack to column vectors:
DMS_stack = DMS.stack(dropna=False).stack(dropna=False)
DMS_stack_dropna = DMS_stack.dropna()

# Get index and values to build interpolation function:
ind_from_raw = DMS_stack_dropna.index
ind_to_interp = DMS_stack.index
d = DMS_stack_dropna.values

# Build RBF interpolation function:
rbfinterp = scipy.interpolate.Rbf(np.array([n[0] for n in ind_from_raw]),
                                  np.array([n[1] for n in ind_from_raw]),
                                  np.array([n[2] for n in ind_from_raw]),
                                  d,function=func)

# Interpolate values:
DMS_interp = pd.DataFrame({'DateTime':np.array([n[0] for n in ind_to_interp]),
                            'Lon':np.array([n[1] for n in ind_to_interp]),
                            'Lat':np.array([n[2] for n in ind_to_interp]),
                            'DMS':rbfinterp(np.array([n[0] for n in ind_to_interp]),
                                              np.array([n[1] for n in ind_to_interp]),
                                              np.array([n[2] for n in ind_to_interp]))})

# Reshape and filter out negative interpolated values:
DMS_interp = DMS_interp.pivot(index='DateTime',columns=['Lat','Lon'], values='DMS').reindex_like(DMS)
DMS_interp[DMS_interp<0]=0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Salinity

# stack to column vectors:
SAL_stack = SAL.stack(dropna=False).stack(dropna=False)
SAL_stack_dropna = SAL_stack.dropna()

# Get index and values to build interpolation function:
ind_from_raw = SAL_stack_dropna.index
ind_to_interp = SAL_stack.index
d = SAL_stack_dropna.values

# Build RBF interpolation function:
rbfinterp = scipy.interpolate.Rbf(np.array([n[0] for n in ind_from_raw]),
                                  np.array([n[1] for n in ind_from_raw]),
                                  np.array([n[2] for n in ind_from_raw]),
                                  d,function=func)

# Interpolate values:
SAL_interp = pd.DataFrame({'DateTime':np.array([n[0] for n in ind_to_interp]),
                            'Lon':np.array([n[1] for n in ind_to_interp]),
                            'Lat':np.array([n[2] for n in ind_to_interp]),
                            'SAL':rbfinterp(np.array([n[0] for n in ind_to_interp]),
                                              np.array([n[1] for n in ind_to_interp]),
                                              np.array([n[2] for n in ind_to_interp]))})

# Reshape and filter out negative interpolated values:
SAL_interp = SAL_interp.pivot(index='DateTime',columns=['Lat','Lon'], values='SAL').reindex_like(DMS)
SAL_interp[SAL_interp<0]=0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SST

# stack to column vectors:
SST_stack = SST.stack(dropna=False).stack(dropna=False)
SST_stack_dropna = SST_stack.dropna()

# Get index and values to build interpolation function:
ind_from_raw = SST_stack_dropna.index
ind_to_interp = SST_stack.index
d = SST_stack_dropna.values

# Build RBF interpolation function:
rbfinterp = scipy.interpolate.Rbf(np.array([n[0] for n in ind_from_raw]),
                                  np.array([n[1] for n in ind_from_raw]),
                                  np.array([n[2] for n in ind_from_raw]),
                                  d,function=func)

# Interpolate values:
SST_interp = pd.DataFrame({'DateTime':np.array([n[0] for n in ind_to_interp]),
                            'Lon':np.array([n[1] for n in ind_to_interp]),
                            'Lat':np.array([n[2] for n in ind_to_interp]),
                            'SST':rbfinterp(np.array([n[0] for n in ind_to_interp]),
                                              np.array([n[1] for n in ind_to_interp]),
                                              np.array([n[2] for n in ind_to_interp]))})

# Reshape and filter out negative interpolated values:
SST_interp = SST_interp.pivot(index='DateTime',columns=['Lat','Lon'], values='SST').reindex_like(DMS)
SST_interp[SST_interp<0]=0

#-----------------------------------------------------------------------------
del DMS_stack,DMS_stack_dropna,SAL_stack,SAL_stack_dropna,SST_stack,SST_stack_dropna

#-----------------------------------------------------------------------------
# Format for reindexing:
# Remove leading zeros in index
SST.index = np.asarray(list(map(int, SST.index.str.lstrip('0').values)))
SST.columns.levels[0]
SST.index.name = 'datetime'
SAL.index = np.asarray(list(map(int, SAL.index.str.lstrip('0').values)))
SAL.index.name = 'datetime'
DMS.index = np.asarray(list(map(int, DMS.index.str.lstrip('0').values)))
DMS.index.name = 'datetime'

# print('Total data removed = '+'{:.2%}'.format(np.sum(track_removed_data[:3])/track_removed_data[3]))

#%% Retrieve Satellite data

# Run data processing script ('Satellite_data_processing.py') first + load data
# NOTE: Make sure file directory is correct in processing script!

# Import data
CHL_sat = pd.read_csv(write_dir+'CHL_sat_'+str(grid)+'.csv',index_col=[0], header=[0,1])

CHL_sat_interp = pd.read_csv(write_dir+'CHL_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
PAR_sat_interp = pd.read_csv(write_dir+'PAR_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
PIC_sat_interp = pd.read_csv(write_dir+'PIC_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
SSHA_sat_interp = pd.read_csv(write_dir+'SSHA_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
SST_sat_interp = pd.read_csv(write_dir+'SST_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
Kd_sat_interp = pd.read_csv(write_dir+'Kd_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
POC_sat_interp = pd.read_csv(write_dir+'POC_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
FLH_sat_interp = pd.read_csv(write_dir+'FLH_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
iPAR_sat_interp = pd.read_csv(write_dir+'iPAR_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
CDOM_sat_interp = pd.read_csv(write_dir+'CDOM_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
SSN_woa_interp = pd.read_csv(write_dir+'SSN_woa_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
NPP_sat_interp = pd.read_csv(write_dir+'NPP_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
MLD_interp = pd.read_csv(write_dir+'MLD_argo_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
SAL_argo_interp = pd.read_csv(write_dir+'SAL_argo_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
WSPD_sat_interp = pd.read_csv(write_dir+'WSPD_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
U_sat_interp = pd.read_csv(write_dir+'U_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])
V_sat_interp = pd.read_csv(write_dir+'V_sat_interp_'+str(grid)+'.csv',index_col=[0], header=[0,1])

etopo = pd.read_csv(write_dir+'etopo2_'+str(grid)+'.csv', index_col=[0,1])

# Convert lat/lons to float values
CHL_sat.columns = CHL_sat.columns.set_levels(CHL_sat.columns.levels[0].astype(float), level=0)
CHL_sat.columns = CHL_sat.columns.set_levels(CHL_sat.columns.levels[1].astype(float), level=1)
CHL_sat_interp.columns = CHL_sat_interp.columns.set_levels(CHL_sat_interp.columns.levels[0].astype(float), level=0)
CHL_sat_interp.columns = CHL_sat_interp.columns.set_levels(CHL_sat_interp.columns.levels[1].astype(float), level=1)
PAR_sat_interp.columns = PAR_sat_interp.columns.set_levels(PAR_sat_interp.columns.levels[0].astype(float), level=0)
PAR_sat_interp.columns = PAR_sat_interp.columns.set_levels(PAR_sat_interp.columns.levels[1].astype(float), level=1)
PIC_sat_interp.columns = PIC_sat_interp.columns.set_levels(PIC_sat_interp.columns.levels[0].astype(float), level=0)
PIC_sat_interp.columns = PIC_sat_interp.columns.set_levels(PIC_sat_interp.columns.levels[1].astype(float), level=1)
SSHA_sat_interp.columns = SSHA_sat_interp.columns.set_levels(SSHA_sat_interp.columns.levels[0].astype(float), level=0)
SSHA_sat_interp.columns = SSHA_sat_interp.columns.set_levels(SSHA_sat_interp.columns.levels[1].astype(float), level=1)
SST_sat_interp.columns = SST_sat_interp.columns.set_levels(SST_sat_interp.columns.levels[0].astype(float), level=0)
SST_sat_interp.columns = SST_sat_interp.columns.set_levels(SST_sat_interp.columns.levels[1].astype(float), level=1)
Kd_sat_interp.columns = Kd_sat_interp.columns.set_levels(Kd_sat_interp.columns.levels[0].astype(float), level=0)
Kd_sat_interp.columns = Kd_sat_interp.columns.set_levels(Kd_sat_interp.columns.levels[1].astype(float), level=1)
POC_sat_interp.columns = POC_sat_interp.columns.set_levels(POC_sat_interp.columns.levels[0].astype(float), level=0)
POC_sat_interp.columns = POC_sat_interp.columns.set_levels(POC_sat_interp.columns.levels[1].astype(float), level=1)
FLH_sat_interp.columns = FLH_sat_interp.columns.set_levels(FLH_sat_interp.columns.levels[0].astype(float), level=0)
FLH_sat_interp.columns = FLH_sat_interp.columns.set_levels(FLH_sat_interp.columns.levels[1].astype(float), level=1)
iPAR_sat_interp.columns = iPAR_sat_interp.columns.set_levels(iPAR_sat_interp.columns.levels[0].astype(float), level=0)
iPAR_sat_interp.columns = iPAR_sat_interp.columns.set_levels(iPAR_sat_interp.columns.levels[1].astype(float), level=1)
CDOM_sat_interp.columns = CDOM_sat_interp.columns.set_levels(CDOM_sat_interp.columns.levels[0].astype(float), level=0)
CDOM_sat_interp.columns = CDOM_sat_interp.columns.set_levels(CDOM_sat_interp.columns.levels[1].astype(float), level=1)
SSN_woa_interp.columns = SSN_woa_interp.columns.set_levels(SSN_woa_interp.columns.levels[0].astype(float), level=0)
SSN_woa_interp.columns = SSN_woa_interp.columns.set_levels(SSN_woa_interp.columns.levels[1].astype(float), level=1)
NPP_sat_interp.columns = NPP_sat_interp.columns.set_levels(NPP_sat_interp.columns.levels[0].astype(float), level=0)
NPP_sat_interp.columns = NPP_sat_interp.columns.set_levels(NPP_sat_interp.columns.levels[1].astype(float), level=1)
MLD_interp.columns = MLD_interp.columns.set_levels(MLD_interp.columns.levels[0].astype(float), level=0)
MLD_interp.columns = MLD_interp.columns.set_levels(MLD_interp.columns.levels[1].astype(float), level=1)
SAL_argo_interp.columns = SAL_argo_interp.columns.set_levels(SAL_argo_interp.columns.levels[0].astype(float), level=0)
SAL_argo_interp.columns = SAL_argo_interp.columns.set_levels(SAL_argo_interp.columns.levels[1].astype(float), level=1)
WSPD_sat_interp.columns = WSPD_sat_interp.columns.set_levels(WSPD_sat_interp.columns.levels[0].astype(float), level=0)
WSPD_sat_interp.columns = WSPD_sat_interp.columns.set_levels(WSPD_sat_interp.columns.levels[1].astype(float), level=1)
U_sat_interp.columns = U_sat_interp.columns.set_levels(U_sat_interp.columns.levels[0].astype(float), level=0)
U_sat_interp.columns = U_sat_interp.columns.set_levels(U_sat_interp.columns.levels[1].astype(float), level=1)
V_sat_interp.columns = V_sat_interp.columns.set_levels(V_sat_interp.columns.levels[0].astype(float), level=0)
V_sat_interp.columns = V_sat_interp.columns.set_levels(V_sat_interp.columns.levels[1].astype(float), level=1)

#%% Estimate NCP
# Algorithm from Li & Cassar (2016)
NCP = (8.57*NPP_sat_interp)/(17.9+SST_sat_interp)

#%% Calculate NPQ-corrected fluorescence yield (Phi)
etopo_unstack = pd.Series(pd.concat([etopo]*3).values[:,0], index=CHL_sat_interp.stack().stack().index).unstack().unstack()

# Algorithm from Beherenfield et al. (2009)
alpha = 0.0147*(CHL_sat_interp**-0.316)
phi = (FLH_sat_interp/(CHL_sat_interp*alpha*100))
phi_cor = (FLH_sat_interp/(CHL_sat_interp*alpha*100))*(iPAR_sat_interp/np.nanmean(iPAR_sat_interp))
#%% Concatenate DMS, SAL
# stack variables dimensionally into 3d
# should be: time x space x variable
seq = [DMS_interp.values[...,np.newaxis],
        SAL_interp.values[...,np.newaxis]]
seq_names = ['DMS','SAL'] # IMP: assign these names for later indexing
PCA_var = np.concatenate(seq, axis=2)
OG_PCA_var = PCA_var

# Save these
seq_names_OG = seq_names

#%% Concatenate Satellite data

seq = [SST_sat_interp.values[...,np.newaxis],
       CHL_sat_interp.values[...,np.newaxis],
       PAR_sat_interp.values[...,np.newaxis],
       PIC_sat_interp.values[...,np.newaxis],
       Kd_sat_interp.values[...,np.newaxis],
       SSHA_sat_interp.values[...,np.newaxis],
       WSPD_sat_interp.values[...,np.newaxis],
       NPP_sat_interp.values[...,np.newaxis],
       FLH_sat_interp.values[...,np.newaxis]/CHL_sat_interp.values[...,np.newaxis],
       CDOM_sat_interp.values[...,np.newaxis],
       MLD_interp.values[...,np.newaxis],
       SAL_argo_interp.values[...,np.newaxis],
       SSN_woa_interp.values[...,np.newaxis],
       NCP.values[...,np.newaxis]]
seq_names = ['SST','Chlorophyll a','PAR','Calcite (PIC)','Kd (490 nm)','SSHA', 'Wind Speed','NPP', 'nFLH:Chl-a',r'a$_{cdm}$(443)','MLD','Salinity','SSN','NCP'] # IMP: assign these names for later indexing
PCA_var = np.concatenate(seq, axis=2)
sat_PCA_var = PCA_var

# Save these
seq_names_sat = seq_names

#%% Select Training/Validation Data

# reshape satelite data back to column vectors to input into models
model_input2=np.empty([])
i=0
# input_ = sat_data_rec
input_ = sat_PCA_var
for var in range(input_.shape[2]):
    data_reshape = pd.DataFrame(input_[:,:,var],
                                index=CHL_sat.index,
                                columns=CHL_sat.columns).stack().stack()
    # data_reshape.rename(seq_names[var])
    if i == 0:
        model_input2 = data_reshape
    else:
        model_input2 = pd.concat([model_input2,data_reshape],axis=1)
    i+=1

# Reshape DMS, Salinity and SST (cruise measurments) as column vectors:
model_input1 = pd.concat([DMS.stack().stack().reindex_like(model_input2),SAL.stack().stack().reindex_like(model_input2)], axis=1)

# Reassign variable names to dataframe columns:
model_input1.columns=seq_names_OG[:]
model_input2.columns=seq_names_sat[:]

# Create dataframe for bathymetry data (extracted in processing script)
bathy = pd.DataFrame(pd.concat([etopo]*3).values, index=model_input2.index)
bathy[bathy>0] = np.nan # filter out land values
bathy.columns = ['Bathymetry']

y = model_input1.iloc[:,0] # DMS
y[y == 0] = np.nan # set neg. values to nan

# Set-up dataframe with predictors to be filtered:
X_full = pd.concat([y, model_input1.iloc[:,1:], bathy, model_input2], axis=1, sort=False) # rest of variables

# Filter out negative SSN values
X_full.loc[:,'SSN'][X_full.loc[:,'SSN']<0] = 0

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Drop bathymetry data:
X = X_full.drop('Bathymetry', axis=1)
# X = X_full
# Drop nans and restrict to satellite data:
X_full = X_full.iloc[:,2:].dropna()
X_full = X_full.iloc[:,1:]
# restrict indices to match measured DMS values:
X = X.drop('SAL', axis=1).dropna()
# drop sal for now to test full forecast:
X = X.iloc[:,1:]
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Average predictors to regional mean values
X_full_reg_mean = X_full.groupby(['lonbins','latbins']).mean()
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~


# Drop nans from DMS values
y = y.dropna()
# Apply IHS transformation to DMS data
y = np.arcsinh(y)
# y = np.log(y)

# split the 80:20 of the data for training:testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('Proportion of training data = %.2f' % round(X_train.shape[0]/X.shape[0],2))
print('Proportion of testing data = %.2f' % round(X_test.shape[0]/X.shape[0],2))
# print('Proportion of validation data = %.2f' % round(X_vali.shape[0]/X.shape[0],2))

#-----------------------------------------------------------------------------
# Remove variables but save for use in running literature models

lit_X_test = X_test
lit_X = X
lit_X_full = X_full

NPP_test = X_test.loc[:,['NPP']]
PIC_test = X_test.loc[:,['Calcite (PIC)']]
Chl_test = X_test.loc[:,['Chlorophyll a']]
Kd_test = X_test.loc[:,['Kd (490 nm)']]

NPP_x = X.loc[:,['NPP']]
PIC_x = X.loc[:,['Calcite (PIC)']]
Chl_x = X.loc[:,['Chlorophyll a']]
Kd_x = X.loc[:,['Kd (490 nm)']]

NPP_full = X_full.loc[:,['NPP']]
PIC_full = X_full.loc[:,['Calcite (PIC)']]
Chl_full = X_full.loc[:,['Chlorophyll a']]
Kd_full = X_full.loc[:,['Kd (490 nm)']]

# Model Biological Configuration:
# NCP w/o Kd
X_full = X_full.drop(['NPP','Calcite (PIC)','Chlorophyll a','Kd (490 nm)'], axis=1)
X = X.drop(['NPP','Calcite (PIC)','Chlorophyll a','Kd (490 nm)'], axis=1)
X_train = X_train.drop(['NPP','Calcite (PIC)','Chlorophyll a','Kd (490 nm)'], axis=1)
X_test = X_test.drop(['NPP','Calcite (PIC)','Chlorophyll a','Kd (490 nm)'], axis=1)
X_full_reg_mean = X_full_reg_mean.drop(['NPP','Calcite (PIC)','Chlorophyll a','Kd (490 nm)'], axis=1)
#-----------------------------------------------------------------------------

# standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

#%% RFR - Build RFR model
#-----------------------------------------------------------------------------
#### Define model
start = timeit.default_timer() # start the clock

nfeatures = np.min(X_train.shape)
# nfeatures = 7
if load_from_memory is False:
    RFR_model = RandomForestRegressor(n_estimators=1000,
                                      max_features=nfeatures,
                                      min_samples_leaf=1, # 1 is default
                                      max_depth=25, # None is default
                                      # ccp_alpha=0, # 0 is default
                                      n_jobs=-1, # use all core processors in computer (i.e. speed up computation)
                                      random_state=0,# this just seeds the randomization of the ensemble models each time
                                      bootstrap=True,
                                      oob_score=False,
                                      verbose=False)
    # fit the model to the training data
    RFR_model.fit(X_train, y_train)
    # save model
    joblib.dump(RFR_model, write_dir[:61]+'RFR_model'+'_'+str(grid)+'.joblib')
else:
    RFR_model = joblib.load(write_dir[:61]+'RFR_model'+'_'+str(grid)+'.joblib')

#-----------------------------------------------------------------------------
#### Validate the model
y_pred = RFR_model.predict(X_test)

# score = RFR_model.score(X,y)
n_features = RFR_model.n_features_in_
importances = RFR_model.feature_importances_

# Calculate the absolute errors
errors = abs(y_pred - y_test)
print('Average absolute error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = (errors / y_test) * 100
print('Average absolute percent error:', round(np.mean(mape), 2))

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
end = timeit.default_timer() # stop the clock

RFR_model_R2 = RFR_model.score(X_test,y_test)
y_pred_subset = RFR_model.predict(pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)) 
#-----------------------------------------------------------------------------
#### Evaluate the model
print()
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print('             RFR Model Results         ')
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
RFR_execution_time = end-start
print('\nExecution time:')
print(str(round(RFR_execution_time,5)),'seconds') 
print(str(round((RFR_execution_time)/60,5)),'mins')
print(str(round((RFR_execution_time)/3600,5)),'hrs')
print('Number of trees in ensemble:',str(RFR_model.n_estimators))
print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, y_pred),4))
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, y_pred),4))
print('Training Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_train, RFR_model.predict(X_train))),4))
print("Training accuracy (R^2): %0.3f" % RFR_model.score(X_train, y_train))
print('Testing Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),4))
print("Testing accuracy (R^2): %0.3f" % RFR_model.score(X_test, y_test))
print('- - - - - - - - - - - -')
print('Full model R^2: %0.3f' % RFR_model_R2)
print('Full model RMSE: %0.3f' % np.sqrt(metrics.mean_squared_error(y_test, RFR_model.predict(X_test))))
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print()

#%% RFR - Model DMS values + reshape for plotting
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# extract the coords from MultiIndex for plotting
date_ind = np.asarray([x[0] for x in y.index])
lon_plt = np.asarray([x[1] for x in y.index])
lat_plt = np.asarray([x[2] for x in y.index])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RFR_model prediction of DMS values
y_pred = RFR_model.predict(pd.DataFrame(scaler.transform(X_full), index=X_full.index, columns=X_full.columns))
y_pred_regional = RFR_model.predict(pd.DataFrame(scaler.transform(X_full_reg_mean), index=X_full_reg_mean.index, columns=X_full_reg_mean.columns)) 

y_pred_raw = [tree.predict(scaler.transform(X_full)) for tree in RFR_model.estimators_]

y_pred_sd = np.std([tree.predict(scaler.transform(X_full)) for tree in RFR_model.estimators_],axis=0)
y_pred_sd = pd.Series(y_pred_sd,index=[X_full.index.get_level_values('datetime'),
                                        X_full.index.get_level_values('lonbins'),
                                        X_full.index.get_level_values('latbins')], name='DMS_sd')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Reshape data to plotting grids by time point

# y_pred values:
# need to shape as dataframe first
y_pred_stack = pd.Series(y_pred,index=[X_full.index.get_level_values('datetime'),
                                        X_full.index.get_level_values('lonbins'),
                                        X_full.index.get_level_values('latbins')], name='DMS')
y_pred_stack.index.set_names(('DateTime', 'Lon', 'Lat'), inplace=True)

y_pred_unstack = y_pred_stack.unstack('Lon')
y_preds = []
for ind in np.unique(y_pred_unstack.index.get_level_values('DateTime')):
    y_preds.append(y_pred_unstack.loc[ind])


# y values:
y.index.set_names(('DateTime','Lon','Lat'), inplace=True)
y_unstack = y.unstack('Lon')
y_unstack = y_unstack.reindex_like(y_pred_unstack)

ys = []
for ind in np.unique(y_unstack.index.get_level_values('DateTime')):
    ys.append(y_unstack.loc[ind])
#-----------------------------------------------------------------------------
# Calculate mean regional DMS across timeseries
# Predicted:
y_pred_full_unstack = y_pred_unstack.unstack('Lat').mean(axis=0)
y_pred_mean = y_pred_full_unstack.unstack('Lon')
# measured:
y_meas_full_unstack = y_unstack.unstack('Lat').mean(axis=0)
y_meas_mean = y_meas_full_unstack.unstack('Lon')
#-----------------------------------------------------------------------------
print()
print('Modelled R2:')
print(RFR_model_R2)
print()
print('Mean [DMS]')
print(str(round(np.mean(np.sinh(y_pred_stack).values),2))+' +/- '+str(round(np.std(np.sinh(y_pred_stack).values),2)))


#%% ANN - Build Artifical Neural Network ensemble of Multilayer Perceptron models
start = timeit.default_timer() # start the clock
# ANN_save_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/py_eosc510/Final Project/ANN_models/ANN_model_'
ANN_save_dir = 'C:/Users/bcamc/OneDrive/Desktop/Python/Projects/Sulfur/NESAP/ANN_models/ANN_model_'
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
def ANN_ensemble_builder(X_train, y_train, num_models=10):
    # Define number of models in ensemble
    ensemble = num_models
    
    # Define hyperparameters:
    num_hidden_neurons = 30
    num_hidden_layers = 2
    max_iter = int(1e9) # max number of iterations (epochs) to run
    alpha = 0.003 # Regularization parameter - penalize to reduce overfitting
    validation_fraction = 0.2 # Percentage of data to leave for internal validation
    batch_size = 32 # size of 'mini batches' used by stochastic optimizer
    solver = 'lbfgs' # use quasi-Newton optimizer optimization method
    # solver = 'adam'
    activation = 'logistic' # activation function to use for the hidden layer
    learning_rate_init = 0.001 # learning rate - equals step size for optimization method
    tol = 1e-6 # threshold for loss function to improve by before training is stopped
    max_fun = int(1.5e5)
    
    # Define list to store models in (the ensemble):
    ANN_ensemble = []
    
    # Iterate to create ensemble:
    for member in tqdm(range(ensemble)):
        # Define MLP model:
        MLP_model = MLPRegressor(hidden_layer_sizes=(num_hidden_neurons,num_hidden_layers), 
                             verbose=False,
                             max_iter=max_iter,
                             alpha=alpha,
                             early_stopping = True,
                             validation_fraction = validation_fraction,
                             batch_size = batch_size,
                             solver = solver,
                             activation = activation,
                             learning_rate_init = learning_rate_init,
                             tol=tol,
                               # random_state=0, # used in selecting design framework; comment out for running full ensemble
                             max_fun=max_fun)
        # Fit the model:
        MLP_model.fit(X_train,y_train)
        # Store model for use in ensemble later
        ANN_ensemble.append(MLP_model)
        
        # print('Model #'+str(member+1)+' of '+str(ensemble)+' models fitted')
    return ANN_ensemble
#-----------------------------------------------------------------------------
print('\nEnsemble Function Built!')

# Build the ensemble:

# ~ ~ ~ ~ ~ ~ ~ ~
num_models = 1000
# ~ ~ ~ ~ ~ ~ ~ ~

# Determine whether to build models or load prebuilt models
if load_from_memory is True:
    # load models from files
    print('\nNow building the ensemble...')
    ANN_ensemble = [joblib.load(ANN_save_dir+str(grid)+'_'+str(i+1)+'.joblib') for i in np.arange(0,num_models,1)]
else:
    print('\nNow loading the ensemble...')
    # with parallel_backend('threading', n_jobs=-1):
    ANN_ensemble = ANN_ensemble_builder(X_train, y_train, num_models=num_models)
    #------------------------------------------------------------------------------
    # save models to files
    for i, model in enumerate(ANN_ensemble):
        joblib.dump(model, ANN_save_dir+str(grid)+'_'+str(i+1)+'.joblib')
#-----------------------------------------------------------------------------
# Predict from training and test dataset from ensemble models - use this to assess fit in plots later on:
y_train_pred = [ANN_ensemble[model].predict(X_train) for model in range(len(ANN_ensemble))]
y_test_pred = [ANN_ensemble[model].predict(X_test) for model in range(len(ANN_ensemble))]

# Average through ensemble predictions to get final modelled values:
ANN_y_train_predict = np.mean([ANN_ensemble[model].predict(X_train) for model in range(len(ANN_ensemble))], axis=0)
ANN_y_test_predict = np.mean([ANN_ensemble[model].predict(X_test) for model in range(len(ANN_ensemble))], axis=0)
ANN_y_predict = np.mean([ANN_ensemble[model].predict(pd.DataFrame(scaler.transform(X_full),
                                                                  index=X_full.index,
                                                                  columns=X_full.columns)) for model in range(len(ANN_ensemble))], axis=0)

ANN_predict_raw = [ANN_ensemble[model].predict(pd.DataFrame(scaler.transform(X_full),
                                                            index=X_full.index,
                                                            columns=X_full.columns)) for model in range(len(ANN_ensemble))]

ANN_y_sd = np.std([ANN_ensemble[model].predict(pd.DataFrame(scaler.transform(X_full),
                                                            index=X_full.index,
                                                            columns=X_full.columns)) for model in range(len(ANN_ensemble))], axis=0)
ANN_y_sd = pd.DataFrame(ANN_y_sd, index=X_full.index)
#-----------------------------------------------------------------------------
# Calculate correlation coefficients
ANN_corrcoefs = np.empty([len(y_test_pred)])
for i, model in enumerate(y_test_pred):
    rs = pearsonr(y_test_pred[i], y_test)
    ANN_corrcoefs[i] = rs[0]
#-----------------------------------------------------------------------------
# Reshape predictions
ANN_y_pred = pd.DataFrame(ANN_y_predict, index=X_full.index)

ANN_y_pred_unstack = ANN_y_pred.unstack('lonbins')
ANN_y_pred_unstack.columns = ANN_y_pred_unstack.columns.droplevel(0)
ANN_y_preds = []
for ind in np.unique(ANN_y_pred_unstack.index.get_level_values('datetime')):
    ANN_y_preds.append(ANN_y_pred_unstack.loc[ind])

ANN_y_pred_full_unstack = ANN_y_pred_unstack.unstack('latbins').mean(axis=0)
ANN_y_pred_mean = ANN_y_pred_full_unstack.unstack('lonbins')
#-----------------------------------------------------------------------------
end = timeit.default_timer() # stop the clock
ANN_execution_time = end-start
#-----------------------------------------------------------------------------
#### Evaluate the model
print()
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print('             ANN Model Results         ')
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print('\nExecution time:')
print(str(round(ANN_execution_time,5)),'seconds') 
print(str(round((ANN_execution_time)/60,5)),'mins')
print(str(round((ANN_execution_time)/3600,5)),'hrs')
print('Number of models in ensemble:',str(num_models))
print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(y_test, ANN_y_test_predict),4))
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(y_test, ANN_y_test_predict),4))
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, ANN_y_test_predict)),4))
# print('Mean Prediction Accuracy (100-MAPE):', round(accuracy, 2), '%')
ANN_model_R2 = r2_score(y_test,ANN_y_test_predict)
print('Training Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_train, ANN_y_train_predict)),4))
print("Training accuracy (R^2): %0.3f" % r2_score(y_train,ANN_y_train_predict))
print('Testing Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(y_test, ANN_y_test_predict)),4))
print("Testing accuracy (R^2): %0.3f" % ANN_model_R2)
print('~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print()

#------------------------------------------------------------------------------
print('Mean [DMS]')
print(str(round(np.mean(np.sinh(ANN_y_pred).values),2))+' +/- '+str(round(np.std(np.sinh(ANN_y_pred).values),2)))

#%% RFR - Calculate back-transformed DMS air-sea flux rates
# #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# # See Goddijn-Murphy et al. (2012) and Simo & Dachs (2002):
# #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# #                          ===================
# #                               Modelled
# #                          ===================
# #-----------------------------------------------------------------------------
# t = X_full.loc[:,['SST']].values # SST (oC)
# u = X_full.loc[:,['Wind Speed']].values # Wind Speed (m s^-1)
# #-----------------------------------------------------------------------------
# # Schmidt number (cm^2 sec^-1):
# k_dms = (2.1*u)-2.8
# #-----------------------------------------------------------------------------
# # Flux rates (umol m^-2 d^-1):
# # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

# # Note - back-transformed here: 
# flux_pred = k_dms*(np.sinh(y_pred).reshape(-1,1))*0.24 # 0.24 converts hr^-1 to d^-1 (& cm to m)

# # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# flux_pred = pd.DataFrame(flux_pred,index=X_full.loc[:,['SST']].index)
# flux_pred.columns=['DMS flux']
# flux_pred_unstack = flux_pred.unstack('lonbins')
# flux_pred_unstack.columns = flux_pred_unstack.columns.droplevel(0)
# flux_preds = []
# for ind in np.unique(flux_pred_unstack.index.get_level_values('datetime')):
#     flux_preds.append(flux_pred_unstack.loc[ind])
# #-----------------------------------------------------------------------------
# #                          ===================
# #                                Actual
# #                          ===================
# #-----------------------------------------------------------------------------
# t = X.loc[:,['SST']].values # SST (oC)
# u = X.loc[:,['Wind Speed']].values # Wind Speed (m s^-1)
# #-----------------------------------------------------------------------------
# # Schmidt number (cm^2 sec^-1):
# k_dms = (2.1*u)-2.8
# #-----------------------------------------------------------------------------
# # Flux rates (umol m^-2 d^-1):
# # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# # Note - back-transformed here: 
# flux_meas = k_dms*(np.sinh(y).values.reshape(-1,1))*0.24 # converts to d^-1 (& cm to m)

# # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# flux_meas = pd.DataFrame(flux_meas,index=X.loc[:,['SST']].index)
# flux_meas.columns=['DMS flux']
# flux_meas_unstack = flux_meas.unstack('lonbins')
# flux_meas_unstack.columns = flux_meas_unstack.columns.droplevel(0)
# flux_meas_unstack = flux_meas_unstack.reindex_like(flux_pred_unstack)
# fluxes = []
# for ind in np.unique(flux_meas_unstack.index.get_level_values('datetime')):
#     fluxes.append(flux_meas_unstack.loc[ind])
# #-----------------------------------------------------------------------------
# #                          ===================
# #                            Regional Fluxes
# #                          ===================
# #-----------------------------------------------------------------------------
# # average/unstack wind field values across timeseries
# U_mean = U_sat_interp.mean(axis=0).unstack('lonbins')
# V_mean = V_sat_interp.mean(axis=0).unstack('lonbins')
# WSPD_mean = WSPD_sat_interp.mean(axis=0).unstack('lonbins')
# #-----------------------------------------------------------------------------
# # convert wind vectors to lists
# WSPD_unstack = WSPD_sat_interp[:-1].stack('latbins')
# WSPDs = []
# for ind in np.unique(WSPD_unstack.index.get_level_values('datetime')):
#     WSPDs.append(WSPD_unstack.loc[ind])

# U_unstack = U_sat_interp[:-1].stack('latbins')
# Us = []
# for ind in np.unique(U_unstack.index.get_level_values('datetime')):
#     Us.append(U_unstack.loc[ind])

# V_unstack = V_sat_interp[:-1].stack('latbins')
# Vs = []
# for ind in np.unique(V_unstack.index.get_level_values('datetime')):
#     Vs.append(V_unstack.loc[ind])
# #-----------------------------------------------------------------------------
# # For modelled flux rates...
# flux_pred_unstack = flux_pred.unstack('lonbins').unstack('latbins')
# # ...Average across time (plot spatially)
# flux_pred_unstack_mean = flux_pred_unstack.mean(axis=0).droplevel(0)
# flux_pred_mean = flux_pred_unstack_mean.unstack('lonbins')
# #-----------------------------------------------------------------------------
# # For actual flux rates...
# flux_meas_unstack = flux_meas.unstack('lonbins').unstack('latbins').reindex_like(flux_pred_unstack)
# # ...Average across time (plot spatially)
# flux_meas_unstack_mean = flux_meas_unstack.mean(axis=0).droplevel(0)
# flux_meas_mean = flux_meas_unstack_mean.unstack('lonbins')
# #-----------------------------------------------------------------------------
# # Define constants:
# # See this post: https://stackoverflow.com/questions/47894513/checking-if-a-geocoordinate-point-is-land-or-ocean-with-cartopy
# land_shp_fname = shpreader.natural_earth(resolution='50m',
#                                        category='physical', name='land')

# land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
# land = prep(land_geom)

# # Function for determining whether a grid cell is covered by land:
# def is_land(x, y):
#     return land.contains(sgeom.Point(x, y))

# # Check whether grid cell is on land or at sea:
# check=np.empty([flux_pred_mean.index.size,flux_pred_mean.columns.size])
# for i in range(len(flux_pred_mean.index.values)):
#     for j in range(len(flux_pred_mean.columns.values)):
#         check[i,j] = is_land(flux_pred_mean.columns.values[j],flux_pred_mean.index.values[i])

# # percentage of study region area attributted to ocean:
# frac_ocean = 1-check[check>0].size/check.size 
# #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# # Constants:
# A = ((max_lat-min_lat)*111*1000)*((max_lon-min_lon)*111*1000) # total regional area
# A_ocean = A*frac_ocean # fraction of total area covered by ocean
# S_mol_mass = 32.06 # molar mass of sulfur
# #-----------------------------------------------------------------------------

# # Mean regional modelled flux (convert to Tg for summertime (~92 days))
# flux_pred_reg_mean = (flux_pred_unstack.mean().mean()*S_mol_mass*A_ocean*92)/(1e6*1e12)

# # Mean regional measured flux (convert to Tg for summertime (~92 days))
# flux_meas_reg_mean = (flux_meas_unstack.mean().mean()*S_mol_mass*A_ocean*92)/(1e6*1e12)

# # Regional fluxes per month (convert to Tg for summertime (~92 days))
# flux_pred_regs = []
# for ind,df in enumerate(flux_preds):
#     flux_pred_regs.append((flux_preds[ind].mean().mean()*S_mol_mass*A_ocean*92)/(1e6*1e12))
# flux_meas_regs = []
# for ind,df in enumerate(fluxes):
#     flux_meas_regs.append((fluxes[ind].mean().mean()*S_mol_mass*A_ocean*92)/(1e6*1e12))
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# print('Mean DMS flux')
# print(str(round(np.mean(flux_pred.values),2))+' +/- '+str(round(np.std(flux_pred.values),2)))

#%% ANN - Calculate back-transformed DMS air-sea flux rates
# #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# # See Goddijn-Murphy et al. (2012) and Simo & Dachs (2002):
# #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# #                          ===================
# #                               Modelled
# #                          ===================
# #-----------------------------------------------------------------------------
# t = X_full.loc[:,['SST']].values # SST (oC)
# u = X_full.loc[:,['Wind Speed']].values # Wind Speed (m s^-1)
# #-----------------------------------------------------------------------------
# # Schmidt number (cm^2 sec^-1):
# k_dms = (2.1*u)-2.8 # in cm hr-1
# #-----------------------------------------------------------------------------
# # Flux rates (umol m^-2 d^-1):
# # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

# # Note - back-transformed here: 
# ANN_flux_pred = k_dms*(np.sinh(ANN_y_pred))*0.24 # 0.24 converts hr^-1 to d^-1 (& cm to m)

# # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# ANN_flux_pred = pd.DataFrame(ANN_flux_pred,index=X_full.loc[:,['SST']].index)
# ANN_flux_pred.columns=['DMS flux']
# ANN_flux_pred_unstack = ANN_flux_pred.unstack('lonbins')
# ANN_flux_pred_unstack.columns = ANN_flux_pred_unstack.columns.droplevel(0)
# ANN_flux_preds = []
# for ind in np.unique(ANN_flux_pred_unstack.index.get_level_values('datetime')):
#     ANN_flux_preds.append(ANN_flux_pred_unstack.loc[ind])
# #-----------------------------------------------------------------------------
# #                          ===================
# #                                Actual
# #                          ===================
# #-----------------------------------------------------------------------------
# t = X.loc[:,['SST']].values # SST (oC)
# u = X.loc[:,['Wind Speed']].values # Wind Speed (m s^-1)
# #-----------------------------------------------------------------------------
# # Schmidt number (cm^2 sec^-1):
# k_dms = (2.1*u)-2.8
# #-----------------------------------------------------------------------------
# # Flux rates (umol m^-2 d^-1):
# # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

# # Note - back-transformed here: 
# flux_meas = k_dms*(np.sinh(y).values.reshape(-1,1))*0.24 # converts to d^-1 (& cm to m)

# # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# flux_meas = pd.DataFrame(flux_meas,index=X.loc[:,['SST']].index)
# flux_meas.columns=['DMS flux']
# flux_meas_unstack = flux_meas.unstack('lonbins')
# flux_meas_unstack.columns = flux_meas_unstack.columns.droplevel(0)
# flux_meas_unstack = flux_meas_unstack.reindex_like(ANN_flux_pred_unstack)
# fluxes = []
# for ind in np.unique(flux_meas_unstack.index.get_level_values('datetime')):
#     fluxes.append(flux_meas_unstack.loc[ind])
# #-----------------------------------------------------------------------------
# #                          ===================
# #                            Regional Fluxes
# #                          ===================
# #-----------------------------------------------------------------------------
# # average/unstack wind field values across timeseries
# U_mean = U_sat_interp.mean(axis=0).unstack('lonbins')
# V_mean = V_sat_interp.mean(axis=0).unstack('lonbins')
# WSPD_mean = WSPD_sat_interp.mean(axis=0).unstack('lonbins')
# #-----------------------------------------------------------------------------
# # convert wind vectors to lists
# WSPD_unstack = WSPD_sat_interp[:-1].stack('latbins')
# WSPDs = []
# for ind in np.unique(WSPD_unstack.index.get_level_values('datetime')):
#     WSPDs.append(WSPD_unstack.loc[ind])

# U_unstack = U_sat_interp[:-1].stack('latbins')
# Us = []
# for ind in np.unique(U_unstack.index.get_level_values('datetime')):
#     Us.append(U_unstack.loc[ind])

# V_unstack = V_sat_interp[:-1].stack('latbins')
# Vs = []
# for ind in np.unique(V_unstack.index.get_level_values('datetime')):
#     Vs.append(V_unstack.loc[ind])
# #-----------------------------------------------------------------------------
# # For modelled flux rates...
# ANN_flux_pred_unstack = ANN_flux_pred.unstack('lonbins').unstack('latbins')
# # ...Average across time (plot spatially)
# ANN_flux_pred_unstack_mean = ANN_flux_pred_unstack.mean(axis=0).droplevel(0)
# ANN_flux_pred_mean = ANN_flux_pred_unstack_mean.unstack('lonbins')
# #-----------------------------------------------------------------------------
# # For actual flux rates...
# flux_meas_unstack = flux_meas.unstack('lonbins').unstack('latbins').reindex_like(ANN_flux_pred_unstack)
# # ...Average across time (plot spatially)
# flux_meas_unstack_mean = flux_meas_unstack.mean(axis=0).droplevel(0)
# flux_meas_mean = flux_meas_unstack_mean.unstack('lonbins')
# #-----------------------------------------------------------------------------
# # Define constants:
# # See this post: https://stackoverflow.com/questions/47894513/checking-if-a-geocoordinate-point-is-land-or-ocean-with-cartopy
# land_shp_fname = shpreader.natural_earth(resolution='50m',
#                                        category='physical', name='land')

# land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
# land = prep(land_geom)

# # Function for determining whether a grid cell is covered by land:
# def is_land(x, y):
#     return land.contains(sgeom.Point(x, y))

# # Check whether grid cell is on land or at sea:
# check=np.empty([ANN_flux_pred_mean.index.size,ANN_flux_pred_mean.columns.size])
# for i in range(len(ANN_flux_pred_mean.index.values)):
#     for j in range(len(ANN_flux_pred_mean.columns.values)):
#         check[i,j] = is_land(ANN_flux_pred_mean.columns.values[j],ANN_flux_pred_mean.index.values[i])

# # percentage of study region area attributted to ocean:
# frac_ocean = 1-check[check>0].size/check.size 
# #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
# # Constants:
# A = ((max_lat-min_lat)*111*1000)*((max_lon-min_lon)*111*1000) # total regional area (m3)
# A_ocean = A*frac_ocean # fraction of total area covered by ocean
# S_mol_mass = 32.06 # molar mass of sulfur (g/mol)
# #-----------------------------------------------------------------------------

# # Mean regional modelled flux (convert to Tg for summertime (~92 days))
# ANN_flux_pred_reg_mean = (ANN_flux_pred_unstack.mean().mean()*S_mol_mass*A_ocean*92)/(1e6*1e12)

# # Mean regional measured flux (convert to Tg for summertime (~92 days))
# flux_meas_reg_mean = (flux_meas_unstack.mean().mean()*S_mol_mass*A_ocean*92)/(1e6*1e12)

# # Regional fluxes per month (convert to Tg for summertime (~92 days))
# ANN_flux_pred_regs = []
# for ind,df in enumerate(ANN_flux_preds):
#     ANN_flux_pred_regs.append((ANN_flux_preds[ind].mean().mean()*S_mol_mass*A_ocean*92)/(1e6*1e12))
# flux_meas_regs = []
# for ind,df in enumerate(fluxes):
#     flux_meas_regs.append((fluxes[ind].mean().mean()*S_mol_mass*A_ocean*92)/(1e6*1e12))
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #### Calculate total S emitted per individual model in ensemble to generate uncertainity range:
# print('ANN Fluxes:')
# print(str(round(np.mean(np.mean(((ANN_flux_pred_unstack*S_mol_mass*A_ocean*92)/(1e6*1e12))*(365/92))),2))+'+/-'+str(round(np.std(np.std(((ANN_flux_pred_unstack*S_mol_mass*A_ocean*92)/(1e6*1e12))*(365/92))),2))+' TgS yr^-1')
# print(str(round(np.mean(np.mean(((ANN_flux_pred_unstack*S_mol_mass*A_ocean*92)/(1e6*1e12)))),2))+'+/-'+str(round(np.std(np.std(((ANN_flux_pred_unstack*S_mol_mass*A_ocean*92)/(1e6*1e12)))),2))+' TgS')
# print()
# print('Mean DMS flux')
# print(str(round(np.mean(ANN_flux_pred.values),2))+' +/- '+str(round(np.std(ANN_flux_pred.values),2)))

#%% Calculate analysis run time
analysis_end = timeit.default_timer()
analysis_runtime = analysis_end-analysis_start
print('Analysis Runtime:')
print(str(round(analysis_runtime,5)),'secs')
print(str(round((analysis_runtime)/60,5)),'mins')
print(str(round((analysis_runtime)/3600,5)),'hrs')
