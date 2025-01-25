# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:01:30 2021

@author: Brandon McNabb
"""

def FRRF_extract(directory, params=['Fo','Fm', 'Fv/Fm'], treatments=['Init','CRL','HL','Fe','DCMU'], uw=False):
    """
    Function to extract Solinese FRRF data into a pandas dataframe from the raw data subfolders and files.

    Parameters
    ----------
    directory : str
        Parent file directory containing all subfolders of data files.
    params : List, optional
        The list of fit parameters to extract from each file. The default is ['Fo','Fm', 'Fv/Fm'].

    Returns
    -------
    FRRF_raw : Dataframe
        The compiled raw data.

    """
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    # start a master dataframe to add all the raw data to
    FRRF_raw = pd.DataFrame()
    
    if uw is True:
        k = 0
    # Loop through directory, extracting data from the pertinent files
    for dir_,_,filenames in tqdm(os.walk(directory)):
        # get the current subfolder
        subfolder = dir_.split(os.sep)[-1]
        # loop through filenames
        for file in filenames:
            # specifies to only open files with the saved "fit" parameters (ignore data & gain files)
            if file.split('_')[2] != 'set' and file.split('_')[-1] == 'fit.csv':
                # load/reassign data as a temporary dataframe
                temp = pd.read_csv(dir_+'/'+file, skipinitialspace=True).drop(0,axis=0)
                # acclim = [i.split("\\")[-1].split('01')[0].split('_')[0] for i in temp.iloc[:,0]]
                acclim = temp.loc[:,'Light_1'].copy().astype(float)
                # acclim = pd.Series([float(i) for i in acclim])
                for i,val in enumerate(acclim):
                    if val == 0:
                        acclim.loc[i+1] = 'dark'
                    else:
                        acclim.loc[i+1] = 'light'
                # remove spaces in column names
                temp.columns = [temp.columns[i].replace(' ','') for i in range(len(temp.columns))]
                # assign sample IDs as index
                temp.insert(loc=0,column='time',value=subfolder)
                # find characters in the filename corresponding to the treatment
                position = []
                for j in treatments:
                    for string_position, i in enumerate(file.split('_')[:-1]):
                        if any(j in s for s in [i]):
                            position.append(string_position)
                # now filter out the integers, so it is just the treatment name
                temp.insert(loc=1,column='treatment',value=''.join(filter(str.isalpha, file.split('_')[position[0]])))
                # for the replicate number, extract by filtering characters out of the substring from the filename
                if uw is True:
                    temp.insert(loc=2,column='replicate',value=''.join(filter(str.isdigit, file.split('_')[1])))
                else:
                    temp.insert(loc=2,column='replicate',value=''.join(filter(str.isdigit, file.split('_')[position[0]])))
                temp.insert(loc=3,column='acclim',value=acclim)

                # temp.insert(loc=0,column='time',value=subfolder)
                # temp.insert(loc=1,column='treatment',value=file.split('_')[0])
                # temp.insert(loc=2,column='replicate',value=file.split('_')[1])
                # temp.insert(loc=3,column='acclim',value=acclim)
                # temp.insert(loc=4,column='pseudorep',value=temp.index)
                
                temp = temp.reset_index().drop('index',axis=1)
                temp.loc[:,'Light_1'] = temp.loc[:,'Light_1'].astype(float)
                i,j = 0,0
                counter = []
                for z in temp.index:
                    if (j >= 2) and (temp.loc[z,'acclim'] == 'dark'):
                        i,j = 0,0
                        if temp.loc[z,'Light_1'] == float(0):
                            counter.append(i)
                            i += 1
                        elif temp.loc[z,'Light_1'] != 0:
                            counter.append(j)
                            j += 1
                    else:
                        if temp.loc[z,'Light_1'] == float(0):
                            counter.append(i)
                            i += 1
                        elif temp.loc[z,'Light_1'] != 0:
                            counter.append(j)
                            j += 1
                temp.insert(loc=4,column='pseudorep',value=pd.Series(counter))
                # assign a smaple number for underway data
                if uw is True:
                    sample_num = []
                    for z in temp.index:
                        try:
                            if (temp.loc[z,'Light_1'] == 0) and (temp.loc[z-1,'Light_1'] == 30):
                                k += 1
                                sample_num.append(k)
                            else:
                                sample_num.append(k)
                        except:
                            sample_num.append(k) # case for the first index is 0
                    temp.insert(loc=4, column='sample',value=pd.Series(sample_num))
                
                # drop gain values & associated rows
                temp = temp[temp.loc[:,'acclim']!='gain']
                temp = temp[temp['acclim'].notna()]
                # set index
                temp = temp.set_index(['time','treatment','replicate','acclim','pseudorep'])
                # extract important parameters from the file (+ convert values from strings to floating point integers)
                # If extracting time values, add after converting to floats
                if any("TIME" in s for s in params):
                    params_subset = params.copy()
                    params_subset.remove('TIME')
                    subset = temp.loc[:,params_subset]
                    # filter out any dashes as nans before setting values as floats
                    subset = subset.loc[:,params_subset].replace('------ ', np.nan)
                    subset = subset.astype(float)
                    subset.insert(loc=1,column='TIME',value=temp.loc[:,'TIME'])
                else:
                    subset = temp.loc[:,params]
                    # filter out any dashes as nans before setting values as floats
                    subset = subset.loc[:,params].replace('------ ', np.nan)
                    subset = subset.astype(float)
                if uw is True:
                    subset.insert(loc=0, column='sample', value=temp['sample'])
                # add to the master dataframe row-wise
                FRRF_raw = pd.concat([FRRF_raw, subset], axis=0)
    return FRRF_raw