# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:32:51 2024

@author: bcamc
"""

def get_rates(data, tref, t_unit='hr', treatments=['CRL','Fe']):
    import sys
    from sklearn.metrics import r2_score
    import pandas as pd
    import statsmodels.api as sm
    from tqdm import tqdm
    
    data = data.copy()
    tref = tref.copy()
    # Get times in days or hrs
    idx = pd.IndexSlice
    for i in range(len(data)):
        # data.iloc[i].loc['Sampling Time'] = pd.to_datetime(tref.iloc[i].strftime('%Y-%m-%d')+' '+data.iloc[i].loc['Sampling Time'].strftime('%H:%M')) 
        data.iat[i, data.columns.get_loc('Sampling Time')] = pd.to_datetime(tref.iloc[i].strftime('%Y-%m-%d')+' '+data.iloc[i].loc['Sampling Time'].strftime('%H:%M')) 
    if t_unit == 'd':
        data['Sampling Time'] = pd.Series((data['Sampling Time'] - tref).dt.total_seconds() / 3600 / 24)
    if t_unit == 'hr':
        data['Sampling Time'] = pd.Series((data['Sampling Time'] - tref).dt.total_seconds() / 3600)
    if t_unit not in {'hr','d'}:
        sys.exit("Error: Time must be hourly ('hr') or daily ('d')")
    # rename column
    data = data.rename({'Sampling Time':'hours'}, axis=1)
    data = data.astype(float)  
    
    # Get hourly rates
    rates = pd.DataFrame(index=data.index,columns=data.columns)
    SE = pd.DataFrame(index=data.index,columns=data.columns)
    r2 = pd.DataFrame(index=data.index,columns=data.columns)
    
    print('Regressing tracer yields & calculating rates...')
    models = {}
    for stn in tqdm(data.index.get_level_values('station').unique()):
        for tracer in data.drop('hours',axis=1).columns:
            for treatment in treatments:
                for rep in data.index.get_level_values('replicate').unique():
                    try:
                        # lm = linear_model.LinearRegression().fit(data.loc[idx[stn,treatment,:,rep],'hours'].to_frame(),
                        #                                          data.loc[idx[stn,treatment,:,rep],tracer].to_frame())
                        # rates.loc[idx[stn,treatment,:,rep],tracer] =  lm.coef_[0][0]
                        
                        # run linear regression to determine rates
                        lm = sm.OLS(data.loc[idx[stn,treatment,:,rep],tracer].to_frame(),
                                    sm.add_constant(data.loc[idx[stn,treatment,:,rep],'hours'].to_frame())).fit()
                        r2 =  r2_score(data.loc[idx[stn,treatment,:,rep],tracer].to_frame(), 
                                       lm.predict(sm.add_constant(data.loc[idx[stn,treatment,:,rep],'hours'].to_frame())))
                        
                        # significance threshold: if r2 < 0.5, rate is considered below detection (Asher et al.2017, doi: 10.1002/lno.10379)
                        if r2 >= 0.5:
                            rates.loc[idx[stn,treatment,:,rep],tracer] =  lm.params[1]
                            SE.loc[idx[stn,treatment,:,rep],tracer] = lm.bse[1]
                            r2.loc[idx[stn,treatment,:,rep],tracer] = r2
                            models[(stn, treatment)] = lm
                    except:
                        pass
                
    mean_rates = rates.groupby(['station','treatment']).mean().drop('hours',axis=1)
    mean_SE = SE.groupby(['station','treatment']).mean().drop('hours',axis=1)
    rates = rates.groupby(['station','treatment','replicate']).mean().drop('hours', axis=1)
    return {'rates':mean_rates, 'rate_SE':mean_SE, 'raw_rates':rates, 'time-processed data':data, 'models':models, 'r2':r2}


# older version - from la perouse 2023 file

# def get_rates(data, tref, treatments=['CRL','Fe']):
#     data = data.copy()
#     tref = tref.copy()
#     # Get times in daysr
#     idx = pd.IndexSlice
#     for i in range(len(data)):
#         # data.iloc[i].loc['Sampling Time'] = pd.to_datetime(tref.iloc[i].strftime('%Y-%m-%d')+' '+data.iloc[i].loc['Sampling Time'].strftime('%H:%M')) 
#         data.iat[i, data.columns.get_loc('Sampling Time')] = pd.to_datetime(tref.iloc[i].strftime('%Y-%m-%d')+' '+data.iloc[i].loc['Sampling Time'].strftime('%H:%M')) 
#     data['Sampling Time'] = pd.Series((data['Sampling Time'] - tref).dt.total_seconds() / 3600 / 24)
#     # rename column
#     data = data.rename({'Sampling Time':'hours'}, axis=1)
#     data = data.astype(float)    
    
#     # Get hourly rates
#     rates = pd.DataFrame(index=data.index,columns=data.columns)
#     SD = pd.DataFrame(index=data.index,columns=data.columns)
#     for stn in data.index.get_level_values('station').unique():
#         for tracer in data.drop('hours',axis=1).columns:
#             for treatment in treatments:
#                 for rep in data.index.get_level_values('replicate').unique():
#                     try:
#                         lm = sm.OLS(data.loc[idx[stn,treatment,:,rep],tracer].to_frame(),
#                                     sm.add_constant(data.loc[idx[stn,treatment,:,rep],'hours'].to_frame())).fit()
#                         # lm = linear_model.LinearRegression().fit(data.loc[idx[stn,treatment,:,rep],tracer].to_frame(),
#                         #                                          data.loc[idx[stn,treatment,:,rep],'hours'].to_frame())
#                         rates.loc[idx[stn,treatment,:,rep],tracer] =  lm.params[1]
#                         SD.loc[idx[stn,treatment,:,rep],tracer] = lm.bse[1]
#                     except:
#                         pass
                
#     mean_rates = rates.groupby(['station','treatment']).mean().drop('hours',axis=1)
#     mean_SD = SD.groupby(['station','treatment']).mean().drop('hours',axis=1)
#     rates = rates.groupby(['station','treatment','replicate']).mean().drop('hours', axis=1)
#     return mean_rates, mean_SD, rates, data