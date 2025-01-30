# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:16:33 2024

@author: bcamc
"""
#%% Import packages
import pandas as pd
import numpy as np
import lvm_read
from sklearn import linear_model

from tqdm import tqdm

#%%
#------------------------------------------------------------------------------
# load & process Line P uw data
uwDMS_raw = pd.read_excel('C:/Users/bcamc/OneDrive/Desktop/2022_cruise_data/OSSCAR_peak_areas_LineP.xlsx', sheet_name='raw_data')
total_elapsed_sec = (uwDMS_raw.loc[:,'elapsed_hr'].astype(float)*3600)+(uwDMS_raw.loc[:,'elapsed_min'].astype(float)*60)+(uwDMS_raw.loc[:,'elapsed_sec'].astype(float))
uwDMS_raw.insert(loc=6, column='total_elapsed_sec', value=total_elapsed_sec)

# add timestamp for each value
cimspath = r'C:/Users/bcamc/OneDrive/Desktop/2022_cruise_data/CIMS_Data/'
headers = ['Latitude', 'Longitude', 'Julian_day', 'Hour', 'Minute', 'Second', 'Elapsed_time', 'DMS', 'Comments']
timestamps = []
lats = []
lons = []
for i,file in tqdm(enumerate(uwDMS_raw['Filename'])):
    # parse time
    lvmfile = lvm_read.read(cimspath+file, read_from_pickle=True)
    date = lvmfile['Date']
    time = lvmfile['Time']
    start_timestamp = pd.Timestamp(date+' '+time)
    
    # index data
    lvmfile = lvmfile[0]['data']
    lvmfile = pd.DataFrame(lvmfile, columns=headers)
    ind = np.argmin(abs(np.round(lvmfile['Elapsed_time'], decimals=3)-np.round(uwDMS_raw['total_elapsed_sec'].iloc[i], decimals=3)))
    # ind = np.argmin(abs(lvmfile['Elapsed_time']-uwDMS_raw['total_elapsed_sec'].iloc[i]))    
    
    # save parameters
    # timestamps.append((start_timestamp+pd.to_timedelta(lvmfile.iloc[ind].loc['Elapsed_time'], unit='s')))
    timestamps.append((start_timestamp+pd.to_timedelta(uwDMS_raw['elapsed_hr'].iloc[i], 'hour')+pd.to_timedelta(uwDMS_raw['elapsed_min'].iloc[i], 'min')+pd.to_timedelta(uwDMS_raw['elapsed_sec'].iloc[i], 'sec')))
    lats.append(lvmfile.iloc[ind].loc['Latitude'])
    lons.append(lvmfile.iloc[ind].loc['Longitude'])

# concatenate values
timestamps = pd.Series(timestamps, name='Timestamp')
lats = pd.Series(lats, name='lat')
lons = pd.Series(lons, name='lon')

# drop 0 values for now
lats = lats.replace(0, np.nan)
lons = lons.replace(0, np.nan)

# format coordinates properly
lats_og = [str((vals-4800)/100).split('.') for i,vals in enumerate(lats)]
lats_degrees = [float(vals[0])+48 for i,vals in enumerate(lats_og)]
lats_dec_mins = [(float('0.'+vals[1])*100)/60 if vals[0]!='nan' else vals[0] for i,vals in enumerate(lats_og)]
lats_fixed = np.asarray(lats_degrees).astype(float)+np.asarray(lats_dec_mins).astype(float)

lons_og = [str((vals-12300)/100).split('.') for i,vals in enumerate(lons)]
lons_degrees = [float(vals[0])+123 for i,vals in enumerate(lons_og)]
lons_dec_mins = [(float('0.'+vals[1])*100)/60 if vals[0]!='nan' else vals[0] for i,vals in enumerate(lons_og)]
lons_fixed = np.asarray(lons_degrees).astype(float)+np.asarray(lons_dec_mins).astype(float)

# insert coords into raw dataset
uwDMS_raw = uwDMS_raw.drop(labels=['elapsed_hr', 'elapsed_min','elapsed_sec','total_elapsed_sec'], axis=1)
uwDMS_raw.insert(loc=1, column='lat', value=lats_fixed)
uwDMS_raw.insert(loc=1, column='lon', value=lons_fixed)
uwDMS_raw.insert(loc=1, column='time', value=timestamps)

# convert lons from E to W (i.e. * by -1)
uwDMS_raw.lon = uwDMS_raw.lon*-1

# index out std curve values + fit and define coef/intercept
uwDMS_std_idx = uwDMS_raw.loc[:,['time','peak_area','peak_height','std_flag']].dropna()
def std_vals(std):
    vals = [std*(5/5),
            std*(4/5),
            std*(3/5),
            std*(2/5),
            std*(1/5),
            std*(0/5),
            ]
    return np.asarray(vals)

std_areas = {}
std_heights = {}
std_fits = {}
linep_std_data = {}
stds = np.tile(30,16)
stds[6] = 10 # 15th used a 10 nM standard
for i,val in enumerate(uwDMS_std_idx['std_flag'].unique()):
    std_concs = pd.Series(std_vals(stds[i]), name='std_concs')
    std_areas[val] = pd.Series(uwDMS_std_idx[uwDMS_std_idx['std_flag'] == val]['peak_area'], name='peak_area')
    std_heights[val] = pd.Series(uwDMS_std_idx[uwDMS_std_idx['std_flag'] == val]['peak_height'], name='peak_height')
    time = uwDMS_std_idx[uwDMS_std_idx['std_flag'] == val]['time'].reset_index().drop('index',axis=1)
    std_data = pd.concat([time, std_concs, std_areas[val].reset_index(), std_heights[val].reset_index()], axis=1).drop('index', axis=1).dropna()
    linep_std_data[val] = std_data
    
    lm = linear_model.LinearRegression()
    lm.fit(std_data['peak_area'].values.reshape(-1,1),
           std_data['std_concs'].values.reshape(-1,1))
    r2 = lm.score(std_data['peak_area'].values.reshape(-1,1),
           std_data['std_concs'].values.reshape(-1,1))
    
    std_fits[val] = [lm.coef_[0][0], lm.intercept_[0]]

# calculate DMS/P concentrations
uw = uwDMS_raw[uwDMS_raw['std_flag'].isna()]
conc = []
for i in range(len(uw)):
    coef_, intercept_ = std_fits[uw.loc[:,'use_std'].iloc[i]]
    conc.append((coef_*uw.loc[:,'peak_area'].iloc[i])+intercept_)
uw.insert(loc=4, column='conc', value=conc)

# pull out la perouse data - concatenate later with other datasets
uw = uw.reset_index().drop('index', axis=1)

uw_lap = uw.iloc[715:,:]
uw = uw.iloc[:714,:]


# concantenate relevant metadata files
metadata1 = pd.read_csv('C:/Users/bcamc/OneDrive/Desktop/2022_cruise_data/OSSCAR_metadata/OSSCAR_sample_metadataTully_LineP_2022_curve1INTEGRALS.TXT', sep=",", header=0, skipinitialspace=True)
metadata2 = pd.read_csv('C:/Users/bcamc/OneDrive/Desktop/2022_cruise_data/OSSCAR_metadata/OSSCAR_sample_metadataTully_LineP_2022INTEGRALS.TXT', sep=",", header=0, skipinitialspace=True)
metadata = pd.concat([metadata1,metadata2],axis=0)
del metadata1, metadata2
# format time, sample flags, & strip out headers throughout
metadata = metadata.reset_index().drop('index', axis=1)
metadata = metadata[metadata['Datestamp']!='Datestamp']
metadata.insert(loc=0,
                column='time',
                value=[pd.Timestamp(i.split(' ')[0]+' '+i.split(' ')[2]+i.split(' ')[3]) for i in metadata['Datestamp']])
metadata = metadata.drop('Datestamp', axis=1)
metadata = metadata.loc[:,['time', 'Sample Type ( DMS)', 'Unnamed: 19']]
metadata.loc[:,'Sample Type ( DMS)'] = metadata.loc[:,'Sample Type ( DMS)'].astype(int)
metadata = metadata.reset_index().drop('index', axis=1)
# correct for offset in time due to chromatography column
metadata['time'] = metadata['time']+pd.to_timedelta(10,'min')
metadata = metadata.rename({'Unnamed: 19':'Comment'}, axis=1)


# correct for time offset for chromatography with DMSP values (~11 mins) - needed to search + assign sample IDs
timepts = [t for t in metadata['time'].unique()]
inds = {}
for i in range(len(metadata)):
    try:
        _,ind = np.argwhere([metadata['time'] == timepts[i]])
        inds[ind[1]] = timepts[i]+pd.to_timedelta(11,'min') # pull out duplicate value 
    except:
        pass

for i in inds.keys():
    metadata.loc[i,'time'] = inds[i]

# pull out sample IDs and insert them into main dataset
inds = []
failed_locs = []
for i in range(len(uw)):
    delta = abs(uw['time'].iloc[i]-metadata['time'])
    ind = np.nanargmin(delta) # EDIT
    if delta.iloc[ind] > pd.to_timedelta(5,'min'):
        # value is not a close enough match, assign nan
        failed_locs.append(i)
        inds.append(np.nan)
    else:
        if metadata.iloc[ind].loc['Comment'] != 'Underway\t':
            inds.append(np.nan)
        else:
            inds.append(metadata.iloc[ind].loc['Sample Type ( DMS)'])
print(len(failed_locs))
uw.insert(4,'sample',inds)

# now recorrect offset - time/coord match pairs of DMS/P samples by overwriting offset on DMSP
for i in tqdm(uw.index):
    if uw.loc[i,'sample'] == 0:
        try:
            if uw.loc[i+1,'sample'] == 2:
                uw.loc[i+1,'time'] = uw.loc[i,'time']
                uw.loc[i+1,'lon'] = uw.loc[i,'lon']
                uw.loc[i+1,'lat'] = uw.loc[i,'lat']
        except:
            pass
    else:
        pass        


# get bridge lat/lons to fill in gaps from missing TSG data
lineP_coords = pd.read_csv('C:/Users/bcamc/OneDrive/Desktop/2022_cruise_data/LineP_GPS_2022-008.TXT', skiprows=8, header=None, converters={1: str})
lineP_coords.columns = ['date','time','lat','lon']
timestamps = []
for i in tqdm(range(len(lineP_coords))):
    timestamps.append(pd.to_datetime(str(lineP_coords['date'].iloc[i])+' '+str(lineP_coords['time'].iloc[i]), utc=True))

# convert bridge coords to series and shift TZ back to PDT, then remove TZ to compare
timestamps = pd.Series(timestamps)
timestamps = timestamps.dt.tz_convert('US/Pacific')
timestamps = timestamps.dt.tz_localize(None)

lineP_coords.insert(0,'datetime',timestamps)
lineP_coords = lineP_coords.drop(['date','time'],axis=1)

# subset out the nearest matching timestamps
failed_locs = []
inds = np.empty((len(uw),2))
for i in range(len(uw)):
    delta = abs(uw['time'].iloc[i]-lineP_coords['datetime'])
    ind = np.nanargmin(delta) # EDIT
    
    if delta.iloc[ind] > pd.to_timedelta(5,'min'):
        # value is not a close enough match, assign nan
        failed_locs.append(i)
        inds[i,:] = np.array([np.nan, np.nan])
    else:
        inds[i,:] = lineP_coords.iloc[ind,1:].values
print(len(failed_locs))
# add these to the dataframe
coords = pd.DataFrame(inds, columns=['bridge_lat', 'bridge_lon'])
uw = pd.concat([uw, coords], axis=1)

# fill nans in with the new coord values
uw.lon.fillna(uw.bridge_lon, inplace=True)
uw.lat.fillna(uw.bridge_lat, inplace=True)

# Manually asign sample IDs where above algorithm failed
uw.loc[21,'sample'] = 2
uw.loc[23,'sample'] = 2
uw.loc[74:88, 'sample'] = [0,2,0,2,0,2,0,2,0,2,0,2,0,2,0]
uw.loc[120,'sample'] = 0
uw.loc[195,'sample'] = 0
uw.loc[242,'sample'] = 0
uw.loc[263:264,'sample'] = 0

# extract variables
uw_DMS = uw[uw['sample']==0]
uw_DMSP = uw[uw['sample']==2]
uw_DMS.reset_index(inplace=True)
uw_DMSP.reset_index(inplace=True)

#### Correct for some erroneous points 

# remove values below detection limit
uw_DMS = uw_DMS.where(uw_DMS['conc']>0, np.nan) 
# pressure spike - discard point sample
uw_DMS.loc[15,'conc'] = np.nan # pressure spike - discard point sample
uw_DMS.loc[114,'conc'] = np.nan # pressure spike - discard point sample
# rinsing failure - samples were too high:
uw_DMS = uw_DMS.mask(uw_DMS.loc[:,'time']<pd.Timestamp('2022-08-11 15:10:00.0'))
uw_DMSP = uw_DMSP.mask(uw_DMSP.loc[:,'time']<pd.Timestamp('2022-08-11 15:10:00.0'))
# precipitate build-up - contamination of DMSP in DMS samples, remove:
uw_DMS = uw_DMS.mask((uw_DMS.loc[:,'time']>=pd.Timestamp('2022-08-13 11:56:48.082501')) & (uw_DMS.loc[:,'time']<=pd.Timestamp('2022-08-13 13:08:33.023128')))
uw_DMSP = uw_DMSP.mask((uw_DMSP.loc[:,'time']>=pd.Timestamp('2022-08-13 11:56:48.082501')) & (uw_DMSP.loc[:,'time']<=pd.Timestamp('2022-08-13 13:08:33.023128')))
# switching from discrete to underway plumbing failed: 
uw_DMS = uw_DMS.mask((uw_DMS.loc[:,'time']>=pd.Timestamp('2022-08-14 20:53:00')) & (uw_DMS.loc[:,'time']<=pd.Timestamp('2022-08-14 21:14:00')))
uw_DMSP = uw_DMSP.mask((uw_DMSP.loc[:,'time']>=pd.Timestamp('2022-08-14 20:53:00')) & (uw_DMSP.loc[:,'time']<=pd.Timestamp('2022-08-14 21:14:00')))
# heat test failed - samples did not elute properly:
uw_DMS = uw_DMS.mask((uw_DMS.loc[:,'time']>=pd.Timestamp('2022-08-16 17:00:00')) & (uw_DMS.loc[:,'time']<=pd.Timestamp('2022-08-16 19:00:00')))
uw_DMSP = uw_DMSP.mask((uw_DMSP.loc[:,'time']>=pd.Timestamp('2022-08-16 17:00:00')) & (uw_DMSP.loc[:,'time']<=pd.Timestamp('2022-08-16 19:00:00')))

# adjust timezones
uw_DMS.loc[:,'time'] = uw_DMS.loc[:,'time'].dt.tz_localize('US/Pacific')
uw_DMS.loc[:,'time'] = uw_DMS.loc[:,'time'].dt.tz_convert('US/Pacific')
uw_DMS.loc[:,'time'] = uw_DMS.loc[:,'time'].dt.tz_localize(None)

uw_DMSP.loc[:,'time'] = uw_DMSP.loc[:,'time'].dt.tz_localize('US/Pacific')
uw_DMSP.loc[:,'time'] = uw_DMSP.loc[:,'time'].dt.tz_convert('US/Pacific')
uw_DMSP.loc[:,'time'] = uw_DMSP.loc[:,'time'].dt.tz_localize(None)

#------------------------------------------------------------------------------
