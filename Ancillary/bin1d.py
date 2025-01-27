# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 18:10:29 2022

@author: Brandon McNabb
"""

def bin1d(data, bin_width, bin_name, bin_loc=0):
    """
    1-D binning function. Takes either a 2-D numpy array, Pandas Series,
    or Pandas DataFrame objects and bins all columns by the specified column.
    Requires the Numpy and Pandas packages to be installed.
    
    WARNING: all columns in the returned dataframe will be binned by this function.
    Be mindful that any columns extracted from the results are only those that
    were intended to be binned.

    Parameters
    ----------
    data : 2-D array, Series, or DataFrame
        Input data to be binned. If a Series, assumes binning is intended on
        the index. Multiindex Series not currently tested/supported.
    bin_width : int
        The width of the bins (ex. '1' for 1-m depth intervals). 
    bin_name : str
        The name of the column to bin by (ex. 'Depth').
    bin_loc : int, optional
        If using a numpy array, specify the column indexer that bin_name will be assigned to
        and the data binned by. The default is 0.

    Raises
    ------
    ValueError
        Notifies user that data is incorrectly formatted (i.e. bin_name is not present).

    Returns
    -------
    data_binned : DataFrame
        Returns a pandas DataFrame consisting of the orginal data binned by bin_name.

    """
    
    import numpy as np
    import pandas as pd
    
    if isinstance(data, pd.Series):
        if any(bin_name in ind for ind in data.index.names):
            data = data.reset_index()
        else:
            raise ValueError(f'{bin_name} is not in index of Series object.')
    elif isinstance(data, pd.DataFrame):
        if any(bin_name in ind for ind in data.index.names):
            data = data.reset_index()
        elif any(bin_name in ind for ind in data.columns):
            pass
        else:
            raise ValueError(f'{bin_name} is not in index or columns of DataFrame object.')
    else:
        header = {}
        header[bin_name] = data[:,bin_loc]
        header['data'] = data[:,data.shape[1]-(bin_loc+1)]
        data = pd.DataFrame(header)
    
    # define bins
    bins = np.arange(0, np.floor(data[bin_name].max())+bin_width,bin_width)
    # Bin data
    data_binned = data.groupby([pd.cut(data[bin_name], bins)]).mean(numeric_only=True)
    data_binned = data_binned.rename_axis(index=['Bins']).reset_index().drop('Bins', axis=1)
    data_binned[bin_name] = np.ceil(data_binned[bin_name])
    return data_binned
