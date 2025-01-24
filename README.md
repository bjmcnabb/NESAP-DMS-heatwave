# NESAP-DMS-heatwave
Scripts providing post-processing and analysis of DMS/O/P cycling & concentrations, and other ancillary data, collected along Line P during a marine heatwave in August 2022.

The main scripts used are the following:
- DMS_heatwave_analysis.py: loads relevant datasets, runs models and statistics.
- DMS_heatwave_figures.py: produces main figure 1-4 and extended data figures 1-6.

Other required code are provided in the following scripts:
- extract_profile_data_1956_1990.py: script provides an iterative function to read historic 1956-1990 profile T-S data (.bot and .ctd files, provided by the Institute of Ocean Sciences at waterproperties.com). Note this function is not comprehensive in extracting metadata; it only extracts neccesary data.
- extract_profile_data_2007_2022.py: function to read modern (2007-2022) CTD and water property data (provided by the Institute of Ocean Sciences at waterproperties.com).
- MHW_get_clim_stats.py: iterative script to load, currate and get descriptive statistics (mean, SD) from historical SST data, which are used to compute the baseline for calculating SST anomalies in Fig. 2.
- NESAP_build_models.py: streamlined code reporduced from https://github.com/bjmcnabb/DMS_Climatology/tree/main/NESAP, which builds the RFR and ANN models as described in McNabb & Tortell (2022).
- FRRF_data_extraction.py: provides an iterative function to extract user-selected physiological parameters obtained from the curve-fitting outputs of the LIFT software.
- process_uw_DMS_2022.py: script to extract underway DMS and DMSP concentrations from integrated CIMS peaks. 
- turnover_rates.py: provides a wrapper function that iteratively runs linear regression on time-course, isotopically-labelled DMS/O/P concentrations obtained from the incubations described in text.
- VGPM_NPP_toolbox.py: provides two helper functions "LatToDayLength" and "calculate_NPP" which are Python ports of the code provided with the VGPM product webpage (http://orca.science.oregonstate.edu/vgpm.code.php).



