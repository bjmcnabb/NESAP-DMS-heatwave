# NESAP-DMS-heatwave
Code accompanying the manuscript "Exceptionally high dimethyl sulfide concentrations during a marine heatwave in the NE subarctic Pacific" (McNabb *et al.* 2025, *in prep.*). These scripts provide post-processing and analysis of DMS/O/P cycling & concentrations, and other ancillary data, collected along Line P during a marine heatwave in August 2022.

The main scripts used are the following:
- <ins>DMS_heatwave_analysis.py</ins>: loads the relevant datasets and runs data analysis, including model runs and statistics.
- <ins>DMS_heatwave_figures.py</ins>: produces main figures 1-4 and extended data figures 1-6.

| Script  | Description |
| ------------- | ------------- |
| DMS_heatwave_analysis.py | loads the relevant datasets and runs data analysis, including model runs and statistics. |
| DMS_heatwave_figures.py | produces main figures 1-4 and extended data figures 1-6. |

Other required code are provided in the following scripts:
| Script  | Description |
| ------------- | ------------- |
| process_uw_DMS_2022.py  | script to extract underway DMS and DMSP concentrations from integrated CIMS peaks.   |
| turnover_rates.py  | provides a wrapper function that iteratively runs linear regression on time-course, isotopically-labelled DMS/O/P concentrations obtained from the incubations described in text.  |
| NESAP_build_models.py | streamlined code reproduced from https://github.com/bjmcnabb/DMS_Climatology/tree/main/NESAP, which builds the RFR and ANN models as described in [McNabb & Tortell (2022)](https://bg.copernicus.org/articles/19/1705/2022/). |
| FRRF_data_extraction.py | provides an iterative function to extract user-selected physiological parameters obtained from the curve-fitting outputs of the LIFT software. |
| extract_profile_data_1956_1990.py | provides an iterative function to read historic 1956-1990 profile T-S data (.bot and .ctd files, provided by the Institute of Ocean Sciences at https://wwww.waterproperties.com). Note this function is not comprehensive in extracting metadata; it only extracts neccesary data. |
| extract_profile_data_2007_2022.py | function to read modern (2007-2022) CTD and water property data (provided by the Institute of Ocean Sciences at https://wwww.waterproperties.com). |
| MHW_get_clim_stats.py | iterative script to load, currate and get descriptive statistics (mean, SD) from historical SST data, which are used to compute the temperature baseline for calculating SST anomalies in Fig. 2a. |
| VGPM_NPP_toolbox.py | provides two helper functions "LatToDayLength" and "calculate_NPP", which are Python ports of the code written and provided by the VGPM product maintainers (see http://orca.science.oregonstate.edu/vgpm.code.php) that are used to calculate NPP from satellite data. |
| bin1d.py | function that takes either 1-D or 2-D data and bins it along the column axis. Dependencies include numpy and pandas. |

