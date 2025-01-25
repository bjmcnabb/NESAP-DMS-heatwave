# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:36:53 2023

@author: bcamc
"""

def LatToDayLength(lats, yDays):
    import numpy as np
    import pandas as pd
    dayL = []
    for lat, yDay in zip(lats,yDays):
        # get lat in radians
        gamma = (lat / 180.0) * np.pi
        
        # convert date into an angle
        psi = (yDay / 365.0) * 2.0 * np.pi
        
        # calc solar declination
        # Kirk page 35
        solarDec = (0.39637 \
                    - 22.9133 * np.cos(psi) \
                    + 4.02543 * np.sin(psi) \
                    - 0.38720 * np.cos(2*psi) \
                    + 0.05200 * np.sin(2*psi)) * np.pi / 180.0
        
        r = -np.tan(gamma) * np.tan(solarDec)
        if r <= -1:
           dayL.append(24.0)
        else:
            if abs(r) < 1:
                dayL.append(24.0 * np.arccos(r) / np.pi)
            else:
                dayL.append(0)
    
    # return final values as either series or array
    if isinstance(lats, pd.Series) or isinstance(yDays, pd.Series):
        if isinstance(lats, pd.Series):
            dayL = pd.Series(dayL, index=lats.index)
        else:
            dayL = pd.Series(dayL, index=yDays.index)
    else:
        dayL = np.asarray(dayL)
        
    return dayL

def calculate_NPP(chls, irrs, ssts, dayLs):
    import numpy as np
    import pandas as pd
    npp = []
    for chl, irr, sst, dayL in zip(chls, irrs, ssts, dayLs):
        # Calculate chl_tot from Satellite Surface Chlorophyll Data.
        if chl < 1.0:
            chl_tot = 38.0 * chl**0.425
        else:
            chl_tot = 40.2 * chl**0.425
        
        # calculates euphotic depth (z_eu) with Morel's Case I model
        z_eu = 200 * chl_tot**-0.293
        
        if z_eu <= 102.0:
            z_eu = 568.2 * chl_tot**-0.746
        
        # Calculate the Pb_opt from satellite sea surface temperature (sst).
        if sst < -10.0:
            pb_opt = 0.00 
        elif sst < -1.0:
           pb_opt = 1.13
        elif sst >  28.5:
            pb_opt = 4.00
        else:
            pb_opt = 1.2956 + 2.749e-1*sst + 6.17e-2*sst**2 \
                - 2.05e-2*sst**3 + 2.462e-3*sst**4 - 1.348e-4*sst**5 \
                    + 3.4132e-6*sst**6 - 3.27e-8*sst**7
        
        # calculate the irradiance (PAR) function
        irrFunc = 0.66125 * irr / (irr + 4.1)
        
        # return the primary production calculation
        npp.append(pb_opt * chl * dayL * irrFunc * z_eu)
    
    if isinstance(chls, pd.Series) or isinstance(irrs, pd.Series) or isinstance(ssts, pd.Series) or isinstance(dayLs, pd.Series):
        if isinstance(chls, pd.Series):
            npp = pd.Series(npp, index=chls.index)
        elif isinstance(irrs, pd.Series):
            npp = pd.Series(npp, index=irrs.index)
        elif isinstance(ssts, pd.Series):
            npp = pd.Series(npp, index=ssts.index)
        else:
            npp = pd.Series(npp, index=dayLs.index)
    else:
        npp = np.asarray(npp)
    
    return npp
    
    