#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
#from shapely.geometry import Polygon
import statistics as stat
from pyts.decomposition import SingularSpectrumAnalysis
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

####-----------------------------------###
### EAST ASIAN SUMMER MONSOON IN CMIP6 ###
####-----------------------------------###
# author: Anja Katzenberger, anja.katzenberger@pik-potsdam.de

# This code reproduces the results as published in the manuscript


#%%---------
#   DIRECTORIES
#-----------

# china mask 
china_dir = r"C:\Users\anjaka\Nextcloud\PhD\03_CMIP6_china\data\data_new\china_mask_remapcon_asia.nc"

# where to save the results 
dir_save = r"C:\Users\anjaka\Nextcloud\PhD\03_CMIP6_china\figures\fig_new"

# CMIP6 data
dir_hist = r"C:\Users\anjaka\Nextcloud\PhD\03_CMIP6_china\data\data_new\EASM_new\historical"
dir_ssp126 = r"C:\Users\anjaka\Nextcloud\PhD\03_CMIP6_china\data\data_new\EASM_new\ssp126"
dir_ssp245 = r"C:\Users\anjaka\Nextcloud\PhD\03_CMIP6_china\data\data_new\EASM_new\ssp245"
dir_ssp370 = r"C:\Users\anjaka\Nextcloud\PhD\03_CMIP6_china\data\data_new\EASM_new\ssp370"
dir_ssp585 = r"C:\Users\anjaka\Nextcloud\PhD\03_CMIP6_china\data\data_new\EASM_new\ssp585"

# TOP6 CMIP6 models
good_models = ['AWI-CM-1-1-MR',
 'EC-Earth3',
 'GFDL-CM4',
 'IPSL-CM6A-LR',
 'MPI-ESM1-2-LR',
 'MRI-ESM2-0']

# All CMIP6 models 
models_analysis = ['CNRM-CM6-1', 'CNRM-ESM2-1', 'BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'FGOALS-f3-L', 'FGOALS-g3', 'UKESM1-0-LL', 'AWI-CM-1-1-MR', 'GFDL-ESM4', 'GFDL-CM4', 'IITM-ESM', 'CanESM5', 'CanESM5-CanOE', 'E3SM-1-1',  'KACE-1-0-G', 'INM-CM5-0', 'INM-CM4-8', 'TaiESM1', 'EC-Earth3-CC', 'EC-Earth3', 'CMCC-ESM2', 'CMCC-CM2-SR5', 'ACCESS-ESM1-5', 'MRI-ESM2-0', 'ACCESS-CM2', 'NESM3', 'MIROC-ES2L', 'MIROC6', 'IPSL-CM6A-LR', 'NorESM2-MM', 'FIO-ESM-2-0', 'MPI-ESM1-2-LR']
#'CAMS-CSM1-0',

startyear_ref = 1995 # start year of reference period for general anaylsis
startyear_ref_std = 1965 # adapted start year of reference period for variability anaylsis
endyear_ref = 2014

startyear_fut = 2081
startyear_fut_std = 2050
endyear_fut = 2100


country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')

#%%----------------------------
### SSP Color Code
#----------------------------

# Color code according to IPCC framework
col_ssp119 = '#1e9583'
col_ssp126 = '#1d3354'
col_ssp245 = '#e9dc3d'
col_ssp370 = '#f11111'
col_ssp434 = '#63bce4'
col_ssp460 = '#e78731'
col_ssp534over = '#996dc8'
col_ssp585 = '#830b22'




#%%---------------
# CMIP6 RAINFALL DATA
#---------------
# CMIP6 data, monthly

# Use climate data operators (CDOs) for preprocessing
# remaped to 1°x1° (CDO remapcon)
# land only 
# 100-150°E, 20-50°N
# multiplied with the mask from above to get the monsoon area only
# JJA yearmean

### HIST 

files_hist = os.listdir(dir_hist)
files_hist_pr = [files_hist for files_hist in files_hist if files_hist.startswith("pr") and files_hist.endswith("mask2.nc")]

hist_pr_timeseries = []
hist_pr_mean = []
hist_pr_std = []
hist_pr_spatial = []

hist_pr_model = []
hist_pr_center = []


for i in range(0,len(files_hist_pr)):
    data = xr.open_dataset(dir_hist + "/" + files_hist_pr[i])
    model = files_hist_pr[i].split("_")[2]
    if model in models_analysis:
        print(model)
        hist_pr_model.append(model)
        center = files_hist_pr[i].split("_")[1]
        hist_pr_center.append(center)
        pr = data["pr"]*86400 # transforming to mm/day

        # timeseries of mean
        pr_mean = pr.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(pr_mean.lat))
        pr_mean = pr_mean.weighted(weights).mean(dim=['lat'])
        hist_pr_timeseries.append(pr_mean)
        
        # pr mean 1995-2014
        pr_mean_ref = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_ref) & (pr_mean['time.year'] <= endyear_ref))
        pr_mean_reff = pr_mean_ref.mean(dim = ["time"]).data.round(2)
        hist_pr_mean.append(pr_mean_reff)
        
        # pr std 1965-2015
        pr_mean_ref_std = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_ref_std) & (pr_mean['time.year'] <= endyear_ref))
        hist_pr_std.append(pr_mean_ref_std.std(dim = ["time"]).data.round(2))
    
        # pr spatial average 1994-2015
        pr_ref = pr.sel(time=(pr['time.year'] >= startyear_ref) & (pr['time.year'] <= endyear_ref))
        pr_ref = pr_ref.mean(dim=["time"])
        hist_pr_spatial.append(pr_ref)
        


### SSP126 

files_ssp126 = os.listdir(dir_ssp126)
files_ssp126_pr = [files_ssp126 for files_ssp126 in files_ssp126 if files_ssp126.startswith("pr") and files_ssp126.endswith("mask2.nc")]

ssp126_pr_timeseries = []
ssp126_pr_mean = []
ssp126_pr_std = []
ssp126_pr_spatial = []

ssp126_pr_model = []
ssp126_pr_center = []


for i in range(0,len(files_ssp126_pr)):
    data = xr.open_dataset(dir_ssp126 + "/" + files_ssp126_pr[i])
    model = files_ssp126_pr[i].split("_")[2]
    if model in models_analysis:
        print(model)
        ssp126_pr_model.append(model)
        center = files_ssp126_pr[i].split("_")[1]
        ssp126_pr_center.append(center)
        pr = data["pr"]*86400 # transforming to mm/day

        # timeseries of mean
        pr_mean = pr.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(pr_mean.lat))
        pr_mean = pr_mean.weighted(weights).mean(dim=['lat'])
        ssp126_pr_timeseries.append(pr_mean)
        
        # pr mean 2081-2100
        pr_mean_ref = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_fut) & (pr_mean['time.year'] <= endyear_fut))
        pr_mean_reff = pr_mean_ref.mean(dim = ["time"]).data.round(2)
        ssp126_pr_mean.append(pr_mean_reff)
        
        # pr std 2050-2100
        pr_mean_ref_std = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_fut_std) & (pr_mean['time.year'] <= endyear_fut))
        ssp126_pr_std.append(pr_mean_ref_std.std(dim = ["time"]).data.round(2))
    
        # pr spatial average 2081-2100
        pr_ref = pr.sel(time=(pr['time.year'] >= startyear_fut) & (pr['time.year'] <= endyear_fut))
        pr_ref = pr_ref.mean(dim=["time"])
        ssp126_pr_spatial.append(pr_ref)
        




### ssp245 

files_ssp245 = os.listdir(dir_ssp245)
files_ssp245_pr = [files_ssp245 for files_ssp245 in files_ssp245 if files_ssp245.startswith("pr") and files_ssp245.endswith("mask2.nc")]

ssp245_pr_timeseries = []
ssp245_pr_mean = []
ssp245_pr_std = []
ssp245_pr_spatial = []

ssp245_pr_model = []
ssp245_pr_center = []


for i in range(0,len(files_ssp245_pr)):
    data = xr.open_dataset(dir_ssp245 + "/" + files_ssp245_pr[i])
    model = files_ssp245_pr[i].split("_")[2]
    if model in models_analysis:
        print(model)
        ssp245_pr_model.append(model)
        center = files_ssp245_pr[i].split("_")[1]
        ssp245_pr_center.append(center)
        pr = data["pr"]*86400 # transforming to mm/day

        # timeseries of mean
        pr_mean = pr.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(pr_mean.lat))
        pr_mean = pr_mean.weighted(weights).mean(dim=['lat'])
        ssp245_pr_timeseries.append(pr_mean)
        
        # pr mean 2081-2100
        pr_mean_ref = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_fut) & (pr_mean['time.year'] <= endyear_fut))
        pr_mean_reff = pr_mean_ref.mean(dim = ["time"]).data.round(2)
        ssp245_pr_mean.append(pr_mean_reff)
        
        # pr std 2050-2100
        pr_mean_ref_std = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_fut_std) & (pr_mean['time.year'] <= endyear_fut))
        ssp245_pr_std.append(pr_mean_ref_std.std(dim = ["time"]).data.round(2))
    
        # pr spatial average 2081-2100
        pr_ref = pr.sel(time=(pr['time.year'] >= startyear_fut) & (pr['time.year'] <= endyear_fut))
        pr_ref = pr_ref.mean(dim=["time"])
        ssp245_pr_spatial.append(pr_ref)
        


### ssp370 

files_ssp370 = os.listdir(dir_ssp370)
files_ssp370_pr = [files_ssp370 for files_ssp370 in files_ssp370 if files_ssp370.startswith("pr") and files_ssp370.endswith("mask2.nc")]

ssp370_pr_timeseries = []
ssp370_pr_mean = []
ssp370_pr_std = []
ssp370_pr_spatial = []

ssp370_pr_model = []
ssp370_pr_center = []


for i in range(0,len(files_ssp370_pr)):
    data = xr.open_dataset(dir_ssp370 + "/" + files_ssp370_pr[i])
    model = files_ssp370_pr[i].split("_")[2]
    if model in models_analysis:
        print(model)
        ssp370_pr_model.append(model)
        center = files_ssp370_pr[i].split("_")[1]
        ssp370_pr_center.append(center)
        pr = data["pr"]*86400 # transforming to mm/day

        # timeseries of mean
        pr_mean = pr.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(pr_mean.lat))
        pr_mean = pr_mean.weighted(weights).mean(dim=['lat'])
        ssp370_pr_timeseries.append(pr_mean)
        
        # pr mean 2081-2100
        pr_mean_ref = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_fut) & (pr_mean['time.year'] <= endyear_fut))
        pr_mean_reff = pr_mean_ref.mean(dim = ["time"]).data.round(2)
        ssp370_pr_mean.append(pr_mean_reff)
        
        # pr std 2050-2100
        pr_mean_ref_std = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_fut_std) & (pr_mean['time.year'] <= endyear_fut))
        ssp370_pr_std.append(pr_mean_ref_std.std(dim = ["time"]).data.round(2))
    
        # pr spatial average 2081-2100
        pr_ref = pr.sel(time=(pr['time.year'] >= startyear_fut) & (pr['time.year'] <= endyear_fut))
        pr_ref = pr_ref.mean(dim=["time"])
        ssp370_pr_spatial.append(pr_ref)
        


### ssp585 

files_ssp585 = os.listdir(dir_ssp585)
files_ssp585_pr = [files_ssp585 for files_ssp585 in files_ssp585 if files_ssp585.startswith("pr") and files_ssp585.endswith("mask2.nc")]

ssp585_pr_timeseries = []
ssp585_pr_mean = []
ssp585_pr_std = []
ssp585_pr_spatial = []

ssp585_pr_model = []
ssp585_pr_center = []


for i in range(0,len(files_ssp585_pr)):
    data = xr.open_dataset(dir_ssp585 + "/" + files_ssp585_pr[i])
    model = files_ssp585_pr[i].split("_")[2]
    if model in models_analysis:
        print(model)
        ssp585_pr_model.append(model)
        center = files_ssp585_pr[i].split("_")[1]
        ssp585_pr_center.append(center)
        pr = data["pr"]*86400 # transforming to mm/day

        # timeseries of mean
        pr_mean = pr.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(pr_mean.lat))
        pr_mean = pr_mean.weighted(weights).mean(dim=['lat'])
        ssp585_pr_timeseries.append(pr_mean)
        
        # pr mean 2081-2100
        pr_mean_ref = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_fut) & (pr_mean['time.year'] <= endyear_fut))
        pr_mean_reff = pr_mean_ref.mean(dim = ["time"]).data.round(2)
        ssp585_pr_mean.append(pr_mean_reff)
        
        # pr std 2050-2100
        pr_mean_ref_std = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_fut_std) & (pr_mean['time.year'] <= endyear_fut))
        ssp585_pr_std.append(pr_mean_ref_std.std(dim = ["time"]).data.round(2))
    
        # pr spatial average 2081-2100
        pr_ref = pr.sel(time=(pr['time.year'] >= startyear_fut) & (pr['time.year'] <= endyear_fut))
        pr_ref = pr_ref.mean(dim=["time"])
        ssp585_pr_spatial.append(pr_ref)
        




### Merging timeseries hist + scenario
ssp126_pr_timeseries_me = []
for i in range(len(ssp126_pr_model)):
    index = hist_pr_model.index(ssp126_pr_model[i])
    ssp126_pr_timeseries_m = np.concatenate((hist_pr_timeseries[index], ssp126_pr_timeseries[i]))
    ssp126_pr_timeseries_me.append(ssp126_pr_timeseries_m)
    # plt.figure()
    # plt.plot(ssp126_pr_timeseries_m)

ssp245_pr_timeseries_me = []
for i in range(len(ssp245_pr_model)):
    index = hist_pr_model.index(ssp245_pr_model[i])
    ssp245_pr_timeseries_m = np.concatenate((hist_pr_timeseries[index], ssp245_pr_timeseries[i]))
    ssp245_pr_timeseries_me.append(ssp245_pr_timeseries_m)
    #plt.figure()
    #plt.plot(ssp245_pr_timeseries_m)

ssp370_pr_timeseries_me = []
for i in range(len(ssp370_pr_model)):
    index = hist_pr_model.index(ssp370_pr_model[i])
    ssp370_pr_timeseries_m = np.concatenate((hist_pr_timeseries[index], ssp370_pr_timeseries[i]))
    ssp370_pr_timeseries_me.append(ssp370_pr_timeseries_m)
    #plt.figure()
    #plt.plot(ssp370_pr_timeseries_m)

ssp585_pr_timeseries_me = []
for i in range(len(ssp585_pr_model)):
    index = hist_pr_model.index(ssp585_pr_model[i])
    ssp585_pr_timeseries_m = np.concatenate((hist_pr_timeseries[index], ssp585_pr_timeseries[i]))
    ssp585_pr_timeseries_me.append(ssp585_pr_timeseries_m)
    #plt.figure()
    #plt.plot(ssp585_pr_timeseries_m)


        
#%%##_________________________________________________________
### SCENARIOPLOT
###___________________________________________________________

# averaging for each model over 20 years
    
# period = 20

# data_ssp126_av = []
# for m in range(len(ssp126_pr_timeseries)):
#     modelm = ssp126_pr_timeseries[m].data
#     modelm_av = []
#     for i in range(len(ssp126_pr_timeseries[0]) - period):
#         modelm_av.append(stat.mean(modelm[i : ((period - 1) + i)]))
#     data_ssp126_av.append(modelm_av)  
       
# Singular Spectrum Analysis
L = 20 # window_size

data_ssa_hist = []
for i in range(len(hist_pr_timeseries)):
    F = hist_pr_timeseries[i].data
    F_arr = np.array(F)
    F_in = np.array([F_arr])
    ssa = SingularSpectrumAnalysis(window_size = L)
    X_ssa = ssa.transform(F_in)
    data_ssa_hist.append(X_ssa[0])

data_ssa_ssp126 = []
for i in range(len(ssp126_pr_timeseries)):
    F = ssp126_pr_timeseries_me[i].data
    F_arr = np.array(F)
    F_in = np.array([F_arr])
    ssa = SingularSpectrumAnalysis(window_size = L)
    X_ssa = ssa.transform(F_in)
    data_ssa_ssp126.append(X_ssa[0])

for i in range(0,15):
    index = hist_pr_model.index(ssp126_pr_model[i])
    plt.plot(np.concatenate((data_ssa_hist[index], data_ssa_ssp126[i])))
    #plt.plot(range(1849,2015),data_hist_dif_av[index])
    


data_ssa_ssp245 = []
for i in range(len(ssp245_pr_timeseries)):
    F = ssp245_pr_timeseries_me[i].data
    F_arr = np.array(F)
    F_in = np.array([F_arr])
    ssa = SingularSpectrumAnalysis(window_size = L)
    X_ssa = ssa.transform(F_in)
    data_ssa_ssp245.append(X_ssa[0])

data_ssa_ssp370 = []
for i in range(len(ssp370_pr_timeseries)):
    F = ssp370_pr_timeseries_me[i].data
    F_arr = np.array(F)
    F_in = np.array([F_arr])
    ssa = SingularSpectrumAnalysis(window_size = L)
    X_ssa = ssa.transform(F_in)
    data_ssa_ssp370.append(X_ssa[0])

data_ssa_ssp585 = []
for i in range(len(ssp585_pr_timeseries)):
    F = ssp585_pr_timeseries_me[i].data
    F_arr = np.array(F)
    F_in = np.array([F_arr])
    ssa = SingularSpectrumAnalysis(window_size = L)
    X_ssa = ssa.transform(F_in)
    data_ssa_ssp585.append(X_ssa[0])
  

###HIST
# calculate difference to reference period 1995-2015 of the same model (av)
# data_hist_dif_av = []
# for i in range(len(data_ssa_hist)):
#     modeli = data_ssa_hist[i]
#     modeli_name = hist_pr_model[i]
#     index = hist_pr_model.index(modeli_name)    
#     modeli_dif = []
#     for s in range(len(modeli)):
#         modeli_dif.append(modeli[s] - stat.mean(data_ssa_hist[index][-20:]))
#     data_hist_dif_av.append(modeli_dif)

# # calculate mean and stdev over all models for each year (av)
# mean_of_models_hist_av = []
# std_of_models_hist_av = []
# for i in range(len(data_hist_dif_av[0])-1):
#     print(i)
#     yearpoints = [model[i] for model in data_hist_dif_av]
#     mean_of_models_hist_av.append(stat.mean(yearpoints))
#     std_of_models_hist_av.append(stat.stdev(yearpoints))

# # calculate background areas (mean plus/minus std) (av)
# hist_minus_av = []
# hist_plus_av = []
# for i in range(len(mean_of_models_hist_av)):
#     hist_minus_av.append(mean_of_models_hist_av[i] - std_of_models_hist_av[i])
#     hist_plus_av.append(mean_of_models_hist_av[i] + std_of_models_hist_av[i])




###SSP126
# calculate difference to reference period 1995-2015 of the same model (av)
data_ssp126_dif_av = []
for i in range(len(data_ssa_ssp126)):
    modeli = data_ssa_ssp126[i]
    modeli_dif = []
    for s in range(len(modeli)):
        modeli_dif.append(modeli[s] - stat.mean(data_ssa_ssp126[i][145:165])) 
    data_ssp126_dif_av.append(modeli_dif)




# calculate mean and stdev over all models for each year (av)
mean_of_models_ssp126_av = []
std_of_models_ssp126_av = []
for i in range(len(data_ssp126_dif_av[0])-1):
    print(i)
    yearpoints = [model[i] for model in data_ssp126_dif_av]
    mean_of_models_ssp126_av.append(stat.mean(yearpoints))
    std_of_models_ssp126_av.append(stat.stdev(yearpoints))

# calculate background areas (mean plus/minus std) (av)
ssp126_minus_av = []
ssp126_plus_av = []
for i in range(len(mean_of_models_ssp126_av)):
    ssp126_minus_av.append(mean_of_models_ssp126_av[i] - std_of_models_ssp126_av[i])
    ssp126_plus_av.append(mean_of_models_ssp126_av[i] + std_of_models_ssp126_av[i])


###SSP245
# calculate difference to reference period 1995-2015 of the same model (av)
data_ssp245_dif_av = []
for i in range(len(data_ssa_ssp245)):
    modeli = data_ssa_ssp245[i]   
    modeli_dif = []
    for s in range(len(modeli)):
        modeli_dif.append(modeli[s] - stat.mean(data_ssa_ssp245[i][145:165])) 
    data_ssp245_dif_av.append(modeli_dif)

# calculate mean and stdev over all models for each year (av)
mean_of_models_ssp245_av = []
std_of_models_ssp245_av = []
for i in range(len(data_ssp245_dif_av[0])-1):
    print(i)
    yearpoints = [model[i] for model in data_ssp245_dif_av]
    mean_of_models_ssp245_av.append(stat.mean(yearpoints))
    std_of_models_ssp245_av.append(stat.stdev(yearpoints))

# calculate background areas (mean plus/minus std) (av)
ssp245_minus_av = []
ssp245_plus_av = []
for i in range(len(mean_of_models_ssp245_av)):
    ssp245_minus_av.append(mean_of_models_ssp245_av[i] - std_of_models_ssp245_av[i])
    ssp245_plus_av.append(mean_of_models_ssp245_av[i] + std_of_models_ssp245_av[i])



###SSP370
# calculate difference to reference period 1995-2015 of the same model (av)
data_ssp370_dif_av = []
for i in range(len(data_ssa_ssp370)):
    modeli = data_ssa_ssp370[i]   
    modeli_dif = []
    for s in range(len(modeli)):
        modeli_dif.append(modeli[s] - stat.mean(data_ssa_ssp370[i][145:165])) 
    data_ssp370_dif_av.append(modeli_dif)

# calculate mean and stdev over all models for each year (av)
mean_of_models_ssp370_av = []
std_of_models_ssp370_av = []
for i in range(len(data_ssp370_dif_av[0])-1):
    print(i)
    yearpoints = [model[i] for model in data_ssp370_dif_av]
    mean_of_models_ssp370_av.append(stat.mean(yearpoints))
    std_of_models_ssp370_av.append(stat.stdev(yearpoints))

# calculate background areas (mean plus/minus std) (av)
ssp370_minus_av = []
ssp370_plus_av = []
for i in range(len(mean_of_models_ssp370_av)):
    ssp370_minus_av.append(mean_of_models_ssp370_av[i] - std_of_models_ssp370_av[i])
    ssp370_plus_av.append(mean_of_models_ssp370_av[i] + std_of_models_ssp370_av[i])



###SSP585
# calculate difference to reference period 1995-2015 of the same model (av)
data_ssp585_dif_av = []
for i in range(len(data_ssa_ssp585)):
    modeli = data_ssa_ssp585[i]   
    modeli_dif = []
    for s in range(len(modeli)):
        modeli_dif.append(modeli[s] - stat.mean(data_ssa_ssp585[i][145:165])) 
    data_ssp585_dif_av.append(modeli_dif)

# calculate mean and stdev over all models for each year (av)
mean_of_models_ssp585_av = []
std_of_models_ssp585_av = []
for i in range(len(data_ssp585_dif_av[0])-1):
    print(i)
    yearpoints = [model[i] for model in data_ssp585_dif_av]
    mean_of_models_ssp585_av.append(stat.mean(yearpoints))
    std_of_models_ssp585_av.append(stat.stdev(yearpoints))

# calculate background areas (mean plus/minus std) (av)
ssp585_minus_av = []
ssp585_plus_av = []
for i in range(len(mean_of_models_ssp585_av)):
    ssp585_minus_av.append(mean_of_models_ssp585_av[i] - std_of_models_ssp585_av[i])
    ssp585_plus_av.append(mean_of_models_ssp585_av[i] + std_of_models_ssp585_av[i])


# Plot
plt.figure(figsize=(10, 7))

plt.plot(np.arange(2015, 2100, 1), mean_of_models_ssp126_av[-85:], color = col_ssp126, label = "SSP1-2.6",linewidth = 2)
plt.plot(np.arange(2015, 2100, 1), ssp126_minus_av[-85:], color = col_ssp126, alpha = 0.5)
plt.plot(np.arange(2015, 2100, 1), ssp126_plus_av[-85:], color = col_ssp126, alpha = 0.5)
plt.fill_between(np.arange(2015, 2100, 1), ssp126_minus_av[-85:], ssp126_plus_av[-85:], alpha = 0.2, color = col_ssp126)

plt.plot(np.arange(2015, 2100, 1), mean_of_models_ssp245_av[-85:], color = col_ssp245, label = "SSP2-4.5",linewidth = 2)
plt.plot(np.arange(2015, 2100, 1), ssp245_minus_av[-85:], color = col_ssp245, alpha = 0.5)
plt.plot(np.arange(2015, 2100, 1), ssp245_plus_av[-85:], color = col_ssp245, alpha = 0.5)
plt.fill_between(np.arange(2015, 2100, 1), ssp245_minus_av[-85:], ssp245_plus_av[-85:], alpha = 0.2, color = col_ssp245)

plt.plot(np.arange(2015, 2100, 1), mean_of_models_ssp370_av[-85:], color = col_ssp370, label = "SSP3-7.0",linewidth = 2)
plt.plot(np.arange(2015, 2100, 1), ssp370_minus_av[-85:], color = col_ssp370, alpha = 0.5)
plt.plot(np.arange(2015, 2100, 1), ssp370_plus_av[-85:], color = col_ssp370, alpha = 0.5)
plt.fill_between(np.arange(2015, 2100, 1), ssp370_minus_av[-85:], ssp370_plus_av[-85:], alpha = 0.2, color = col_ssp370)

plt.plot(np.arange(2015, 2100, 1), mean_of_models_ssp585_av[-85:], color = col_ssp585, label = "SSP5-8.5",linewidth = 2)
plt.plot(np.arange(2015, 2100, 1), ssp585_minus_av[-85:], color = col_ssp585, alpha = 0.5)
plt.plot(np.arange(2015, 2100, 1), ssp585_plus_av[-85:], color = col_ssp585, alpha = 0.5)
plt.fill_between(np.arange(2015, 2100, 1), ssp585_minus_av[-85:], ssp585_plus_av[-85:], alpha = 0.2, color = col_ssp585)

plt.plot(np.arange(1850, 2015, 1), mean_of_models_ssp245_av[:165], color = "black", label = "historical",linewidth = 2)
plt.plot(np.arange(1850, 2015, 1), ssp245_minus_av[:165], color = "black", alpha = 0.5)
plt.plot(np.arange(1850, 2015, 1), ssp245_plus_av[:165], color = "black", alpha = 0.5)
plt.fill_between(np.arange(1850, 2015, 1), ssp245_minus_av[:165], ssp245_plus_av[:165], alpha = 0.2, color = "black")


plt.axhline(0, color='black', linestyle = '-', linewidth = 0.9)
plt.axvline(2014.5, color='black', linestyle = '-', linewidth = 0.9)
plt.legend(loc='upper left',frameon = False)
plt.ylabel("mm/day")
plt.axvspan(1995, 2015, facecolor='grey', alpha=0.085)

plt.savefig(dir_save + '/Scenarioplot.pdf', bbox_inches='tight')
#plt.show()

        
        
        
        

        
#%%##_________________________________________________________
### SCENARIOPLOT - TOP6 models only
###___________________________________________________________

# averaging for each model over 20 years
    
# period = 20

# Singular Spectrum Analysis
L = 20 # window_size

hist_pr_timeseries_good = []
hist_pr_model_good = []
data_ssa_hist = []
for i in range(len(hist_pr_timeseries)):
    if hist_pr_model[i] in good_models: 
        hist_pr_model_good.append(hist_pr_model[i])
        hist_pr_timeseries_good.append(hist_pr_timeseries[i])
        F = hist_pr_timeseries[i]
        F_arr = np.array(F)
        F_in = np.array([F_arr])
        ssa = SingularSpectrumAnalysis(window_size = L)
        X_ssa = ssa.transform(F_in)
        data_ssa_hist.append(X_ssa[0])

ssp126_pr_model_good = []
ssp126_pr_timeseries_good = []
data_ssa_ssp126 = []
for i in range(len(ssp126_pr_timeseries)):
    if ssp126_pr_model[i] in good_models: 
        ssp126_pr_model_good.append(ssp126_pr_model[i])
        ssp126_pr_timeseries_good.append(ssp126_pr_timeseries_me[i])
        
        F = ssp126_pr_timeseries_me[i]
        F_arr = np.array(F)
        F_in = np.array([F_arr])
        ssa = SingularSpectrumAnalysis(window_size = L)
        X_ssa = ssa.transform(F_in)
        data_ssa_ssp126.append(X_ssa[0])
        

ssp245_pr_model_good = []
ssp245_pr_timeseries_good = []
data_ssa_ssp245 = []
for i in range(len(ssp245_pr_timeseries)):
    if ssp245_pr_model[i] in good_models: 
        ssp245_pr_model_good.append(ssp245_pr_model[i])
        ssp245_pr_timeseries_good.append(ssp245_pr_timeseries_me[i])
        
        F = ssp245_pr_timeseries_me[i]
        F_arr = np.array(F)
        F_in = np.array([F_arr])
        ssa = SingularSpectrumAnalysis(window_size = L)
        X_ssa = ssa.transform(F_in)
        data_ssa_ssp245.append(X_ssa[0])
        
ssp370_pr_model_good = []
ssp370_pr_timeseries_good = []
data_ssa_ssp370 = []
for i in range(len(ssp370_pr_timeseries)):
    if ssp370_pr_model[i] in good_models: 
        ssp370_pr_model_good.append(ssp370_pr_model[i])
        ssp370_pr_timeseries_good.append(ssp370_pr_timeseries_me[i])
        
        F = ssp370_pr_timeseries_me[i]
        F_arr = np.array(F)
        F_in = np.array([F_arr])
        ssa = SingularSpectrumAnalysis(window_size = L)
        X_ssa = ssa.transform(F_in)
        data_ssa_ssp370.append(X_ssa[0])
        
ssp585_pr_model_good = []
ssp585_pr_timeseries_good = []
data_ssa_ssp585 = []
for i in range(len(ssp585_pr_timeseries)):
    if ssp585_pr_model[i] in good_models: 
        ssp585_pr_model_good.append(ssp585_pr_model[i])
        ssp585_pr_timeseries_good.append(ssp585_pr_timeseries_me[i])
        
        F = ssp585_pr_timeseries_me[i]
        F_arr = np.array(F)
        F_in = np.array([F_arr])
        ssa = SingularSpectrumAnalysis(window_size = L)
        X_ssa = ssa.transform(F_in)
        data_ssa_ssp585.append(X_ssa[0])
        
        # plt.figure()
        # plt.plot(range(1850,2100),ssp585_pr_timeseries_me[i][:250])
        # plt.plot(range(1850,2100),X_ssa[0][:250])
        # plt.title(ssp585_pr_model[i])





###SSP126
# calculate difference to reference period 1995-2015 of the same model (av)
data_ssp126_dif_av = []
for i in range(len(data_ssa_ssp126)):
    modeli = data_ssa_ssp126[i]
    modeli_dif = []
    for s in range(len(modeli)):
        modeli_dif.append(modeli[s] - stat.mean(data_ssa_ssp126[i][145:165])) 
    data_ssp126_dif_av.append(modeli_dif)




# calculate mean and stdev over all models for each year (av)
mean_of_models_ssp126_av = []
std_of_models_ssp126_av = []
for i in range(len(data_ssp126_dif_av[0])-1):
    print(i)
    yearpoints = [model[i] for model in data_ssp126_dif_av]
    mean_of_models_ssp126_av.append(stat.mean(yearpoints))
    std_of_models_ssp126_av.append(stat.stdev(yearpoints))

# calculate background areas (mean plus/minus std) (av)
ssp126_minus_av = []
ssp126_plus_av = []
for i in range(len(mean_of_models_ssp126_av)):
    ssp126_minus_av.append(mean_of_models_ssp126_av[i] - std_of_models_ssp126_av[i])
    ssp126_plus_av.append(mean_of_models_ssp126_av[i] + std_of_models_ssp126_av[i])


###SSP245
# calculate difference to reference period 1995-2015 of the same model (av)
data_ssp245_dif_av = []
for i in range(len(data_ssa_ssp245)):
    modeli = data_ssa_ssp245[i]   
    modeli_dif = []
    for s in range(len(modeli)):
        modeli_dif.append(modeli[s] - stat.mean(data_ssa_ssp245[i][145:165])) 
    data_ssp245_dif_av.append(modeli_dif)

# calculate mean and stdev over all models for each year (av)
mean_of_models_ssp245_av = []
std_of_models_ssp245_av = []
for i in range(len(data_ssp245_dif_av[0])-1):
    print(i)
    yearpoints = [model[i] for model in data_ssp245_dif_av]
    mean_of_models_ssp245_av.append(stat.mean(yearpoints))
    std_of_models_ssp245_av.append(stat.stdev(yearpoints))

# calculate background areas (mean plus/minus std) (av)
ssp245_minus_av = []
ssp245_plus_av = []
for i in range(len(mean_of_models_ssp245_av)):
    ssp245_minus_av.append(mean_of_models_ssp245_av[i] - std_of_models_ssp245_av[i])
    ssp245_plus_av.append(mean_of_models_ssp245_av[i] + std_of_models_ssp245_av[i])



###SSP370
# calculate difference to reference period 1995-2015 of the same model (av)
data_ssp370_dif_av = []
for i in range(len(data_ssa_ssp370)):
    modeli = data_ssa_ssp370[i]   
    modeli_dif = []
    for s in range(len(modeli)):
        modeli_dif.append(modeli[s] - stat.mean(data_ssa_ssp370[i][145:165])) 
    data_ssp370_dif_av.append(modeli_dif)

# calculate mean and stdev over all models for each year (av)
mean_of_models_ssp370_av = []
std_of_models_ssp370_av = []
for i in range(len(data_ssp370_dif_av[0])-1):
    print(i)
    yearpoints = [model[i] for model in data_ssp370_dif_av]
    mean_of_models_ssp370_av.append(stat.mean(yearpoints))
    std_of_models_ssp370_av.append(stat.stdev(yearpoints))

# calculate background areas (mean plus/minus std) (av)
ssp370_minus_av = []
ssp370_plus_av = []
for i in range(len(mean_of_models_ssp370_av)):
    ssp370_minus_av.append(mean_of_models_ssp370_av[i] - std_of_models_ssp370_av[i])
    ssp370_plus_av.append(mean_of_models_ssp370_av[i] + std_of_models_ssp370_av[i])



###SSP585
# calculate difference to reference period 1995-2015 of the same model (av)
data_ssp585_dif_av = []
for i in range(len(data_ssa_ssp585)):
    modeli = data_ssa_ssp585[i]   
    modeli_dif = []
    for s in range(len(modeli)):
        modeli_dif.append(modeli[s] - stat.mean(data_ssa_ssp585[i][145:165])) 
    data_ssp585_dif_av.append(modeli_dif)

# calculate mean and stdev over all models for each year (av)
mean_of_models_ssp585_av = []
std_of_models_ssp585_av = []
for i in range(len(data_ssp585_dif_av[0])-1):
    print(i)
    yearpoints = [model[i] for model in data_ssp585_dif_av]
    mean_of_models_ssp585_av.append(stat.mean(yearpoints))
    std_of_models_ssp585_av.append(stat.stdev(yearpoints))

# calculate background areas (mean plus/minus std) (av)
ssp585_minus_av = []
ssp585_plus_av = []
for i in range(len(mean_of_models_ssp585_av)):
    ssp585_minus_av.append(mean_of_models_ssp585_av[i] - std_of_models_ssp585_av[i])
    ssp585_plus_av.append(mean_of_models_ssp585_av[i] + std_of_models_ssp585_av[i])


# Plot
plt.figure(figsize=(10, 7))

plt.plot(np.arange(2015, 2100, 1), mean_of_models_ssp126_av[-85:], color = col_ssp126, label = "SSP1-2.6",linewidth = 2)
plt.plot(np.arange(2015, 2100, 1), ssp126_minus_av[-85:], color = col_ssp126, alpha = 0.5)
plt.plot(np.arange(2015, 2100, 1), ssp126_plus_av[-85:], color = col_ssp126, alpha = 0.5)
plt.fill_between(np.arange(2015, 2100, 1), ssp126_minus_av[-85:], ssp126_plus_av[-85:], alpha = 0.2, color = col_ssp126)

plt.plot(np.arange(2015, 2100, 1), mean_of_models_ssp245_av[-85:], color = col_ssp245, label = "SSP2-4.5",linewidth = 2)
plt.plot(np.arange(2015, 2100, 1), ssp245_minus_av[-85:], color = col_ssp245, alpha = 0.5)
plt.plot(np.arange(2015, 2100, 1), ssp245_plus_av[-85:], color = col_ssp245, alpha = 0.5)
plt.fill_between(np.arange(2015, 2100, 1), ssp245_minus_av[-85:], ssp245_plus_av[-85:], alpha = 0.2, color = col_ssp245)

plt.plot(np.arange(2015, 2100, 1), mean_of_models_ssp370_av[-85:], color = col_ssp370, label = "SSP3-7.0",linewidth = 2)
plt.plot(np.arange(2015, 2100, 1), ssp370_minus_av[-85:], color = col_ssp370, alpha = 0.5)
plt.plot(np.arange(2015, 2100, 1), ssp370_plus_av[-85:], color = col_ssp370, alpha = 0.5)
plt.fill_between(np.arange(2015, 2100, 1), ssp370_minus_av[-85:], ssp370_plus_av[-85:], alpha = 0.2, color = col_ssp370)

plt.plot(np.arange(2015, 2100, 1), mean_of_models_ssp585_av[-85:], color = col_ssp585, label = "SSP5-8.5",linewidth = 2)
plt.plot(np.arange(2015, 2100, 1), ssp585_minus_av[-85:], color = col_ssp585, alpha = 0.5)
plt.plot(np.arange(2015, 2100, 1), ssp585_plus_av[-85:], color = col_ssp585, alpha = 0.5)
plt.fill_between(np.arange(2015, 2100, 1), ssp585_minus_av[-85:], ssp585_plus_av[-85:], alpha = 0.2, color = col_ssp585)

plt.plot(np.arange(1850, 2015, 1), mean_of_models_ssp245_av[:165], color = "black", label = "historical",linewidth = 2)
plt.plot(np.arange(1850, 2015, 1), ssp245_minus_av[:165], color = "black", alpha = 0.5)
plt.plot(np.arange(1850, 2015, 1), ssp245_plus_av[:165], color = "black", alpha = 0.5)
plt.fill_between(np.arange(1850, 2015, 1), ssp245_minus_av[:165], ssp245_plus_av[:165], alpha = 0.2, color = "black")


plt.axhline(0, color='black', linestyle = '-', linewidth = 0.9)
plt.axvline(2014.5, color='black', linestyle = '-', linewidth = 0.9)
plt.legend(loc='upper left',frameon = False)
plt.ylabel("mm/day")
plt.axvspan(1995, 2015, facecolor='grey', alpha=0.085)

plt.savefig(dir_save + '/Scenarioplot_TOP6.pdf', bbox_inches='tight')
#plt.show()

        
        
        
        


#%%---------------
# TIMESERIES
#---------------

hist_pr_model_order = []
hist_pr_timeseries_order = []
for i in range(0,6):
    wanted = good_models[i]
    print(wanted)
    for j in range(0,6):
        if hist_pr_model_good[j] == wanted:
            print(hist_pr_model_good[j])
            hist_pr_model_order.append(hist_pr_model_good[j])
            hist_pr_timeseries_order.append(hist_pr_timeseries_good[j])

ssp126_pr_model_order = []
ssp126_pr_timeseries_order = []
data_ssa_ssp126_order = []
for i in range(0,6):
    wanted = good_models[i]
    print(wanted)
    for j in range(0,len(ssp126_pr_model_good)):
        if ssp126_pr_model_good[j] == wanted:
            print(ssp126_pr_model_good[j])
            ssp126_pr_model_order.append(ssp126_pr_model_good[j])
            ssp126_pr_timeseries_order.append(ssp126_pr_timeseries_good[j])
            data_ssa_ssp126_order.append(data_ssa_ssp126[j])


ssp245_pr_model_order = []
ssp245_pr_timeseries_order = []
data_ssa_ssp245_order = []
for i in range(0,6):
    wanted = good_models[i]
    print(wanted)
    for j in range(0,len(ssp245_pr_model_good)):
        if ssp245_pr_model_good[j] == wanted:
            print(ssp245_pr_model_good[j])
            ssp245_pr_model_order.append(ssp245_pr_model_good[j])
            ssp245_pr_timeseries_order.append(ssp245_pr_timeseries_good[j])
            data_ssa_ssp245_order.append(data_ssa_ssp245[j])


ssp370_pr_model_order = []
ssp370_pr_timeseries_order = []
data_ssa_ssp370_order = []
for i in range(0,6):
    wanted = good_models[i]
    print(wanted)
    for j in range(0,len(ssp370_pr_model_good)):
        if ssp370_pr_model_good[j] == wanted:
            print(ssp370_pr_model_good[j])
            ssp370_pr_model_order.append(ssp370_pr_model_good[j])
            ssp370_pr_timeseries_order.append(ssp370_pr_timeseries_good[j])
            data_ssa_ssp370_order.append(data_ssa_ssp370[j])




ssp585_pr_model_order = []
ssp585_pr_timeseries_order = []
data_ssa_ssp585_order = []
for i in range(0,6):
    wanted = good_models[i]
    print(wanted)
    for j in range(0,6):
        if ssp585_pr_model_good[j] == wanted:
            print(ssp585_pr_model_good[j])
            ssp585_pr_model_order.append(ssp585_pr_model_good[j])
            ssp585_pr_timeseries_order.append(ssp585_pr_timeseries_good[j])
            data_ssa_ssp585_order.append(data_ssa_ssp585[j])



# Creating plots 
plt.style.use('classic')
fig, axs = plt.subplots(2,3,figsize = (15,7))
index = 0
for i in [0,1]:
    for j in [0,1,2]:
        axs[i,j].plot(np.arange(1850,2015, 1), ssp585_pr_timeseries_order[index][0:165], color = "gray", alpha = 0.35)
        axs[i,j].plot(np.arange(1850, 2015, 1), data_ssa_ssp585_order[index][0:165], color = "gray", label = "historical")
 

        if index not in [2,3,4,5]: 
            axs[i,j].plot(np.arange(1850, 2015, 1), data_ssa_ssp126_order[index][0:165], color = "gray")
 
            axs[i,j].plot(np.arange(2015, 2100, 1), ssp126_pr_timeseries_order[index][164:164+85], color = col_ssp126, alpha = 0.35)
            axs[i,j].plot(np.arange(2015, 2100, 1), data_ssa_ssp126_order[index][165:165+85], color = col_ssp126)

        if index in [3,4,5]:    
            axs[i,j].plot(np.arange(1850, 2015, 1), data_ssa_ssp126_order[index-1][0:165], color = "gray")
 
            
            axs[i,j].plot(np.arange(2015, 2100, 1), ssp126_pr_timeseries_order[index-1][164:164+85], color = col_ssp126, alpha = 0.35)
            axs[i,j].plot(np.arange(2015, 2100, 1), data_ssa_ssp126_order[index-1][165:165+85], color = col_ssp126, label = "SSP1-2.6")

        axs[i,j].plot(np.arange(2015,2100, 1), ssp585_pr_timeseries_order[index][164:164+85], color = col_ssp585, alpha = 0.35)
        axs[i,j].plot(np.arange(2015, 2100, 1), data_ssa_ssp585_order[index][165:165+85], color = col_ssp585, label = "SSP5-8.5")
  
            
        # axs[i,j].plot(np.arange(2015, 2100, 1), ssp245_pr_timeseries[index][-85:], color = col_ssp245, alpha = 0.35)
        # axs[i,j].plot(np.arange(2015, 2100, 1), data_ssa_ssp245[index][-85:], color = col_ssp245)
        
        # axs[i,j].plot(np.arange(2015, 2100, 1), ssp370_pr_timeseries[index][-85:], color = col_ssp370, alpha = 0.35)
        # axs[i,j].plot(np.arange(2015, 2100, 1), data_ssa_ssp370[index][-85:], color = col_ssp370)
        
        # axs[i,j].plot(np.arange(1849, 2101, 1), ssp585_pr_timeseries_me[index], color = col_ssp585, alpha = 0.35)
        # axs[i,j].plot(np.arange(1849, 2101, 1), data_ssa_ssp585[index], color = col_ssp585)
        
        axs[i,j].set_title(ssp585_pr_model_order[index], fontsize = 10, pad = 1)
        axs[i,j].set_ylim(4,7.5)
        #axs[i,j].set_xlim(1850,2100)
        axs[i,j].tick_params(axis='both', which='major', labelsize=8)
        axs[i,j].set_xticks(np.arange(1850, 2100, 100))
        axs[i,j].set_yticks(np.arange(4, 8, 1))
        #axs[i,j].axhline(data_means_1850_2015[index] + data_std_1850_2015[index], color='grey', linestyle = '-', linewidth = 0.5)
        #axs[i,j].axhline(data_means_1850_2015[index] - data_std_1850_2015[index], color='grey', linestyle = '-', linewidth = 0.5)
        index = index + 1
for ax in axs.flat:
    ax.label_outer()
plt.legend(frameon=False, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize='small')
fig.text(0.08, 0.5, 'Monsoon rainfall (mm/day)', va='center', rotation='vertical')
plt.savefig(dir_save + 'Timeseries_good.pdf',bbox_to_inches = "tight")
   

        

#%%---------------
# MEAN DIFFERENCES
#---------------

ssp126_div_2021 = []
ssp126_div_2041 = []
ssp126_div_2061 = []
ssp126_div_2081 = []
for i in range(0,len(ssp126_pr_timeseries_order)):
    ssp126_div_2081.append((stat.mean(ssp126_pr_timeseries_order[i][233:252]) - stat.mean(ssp126_pr_timeseries_order[i][147:166]))/stat.mean(ssp126_pr_timeseries_order[i][147:166])*100)
    ssp126_div_2061.append((stat.mean(ssp126_pr_timeseries_order[i][213:232]) - stat.mean(ssp126_pr_timeseries_order[i][147:166]))/stat.mean(ssp126_pr_timeseries_order[i][147:166])*100)
    ssp126_div_2041.append((stat.mean(ssp126_pr_timeseries_order[i][193:212]) - stat.mean(ssp126_pr_timeseries_order[i][147:166]))/stat.mean(ssp126_pr_timeseries_order[i][147:166])*100)
    ssp126_div_2021.append((stat.mean(ssp126_pr_timeseries_order[i][177:192]) - stat.mean(ssp126_pr_timeseries_order[i][147:166]))/stat.mean(ssp126_pr_timeseries_order[i][147:166])*100)
        


ssp245_div_2021 = []
ssp245_div_2041 = []
ssp245_div_2061 = []
ssp245_div_2081 = []
for i in range(0,len(ssp245_pr_timeseries_order)):
    ssp245_div_2081.append((stat.mean(ssp245_pr_timeseries_order[i][233:252]) - stat.mean(ssp245_pr_timeseries_order[i][147:166]))/stat.mean(ssp245_pr_timeseries_order[i][147:166])*100)
    ssp245_div_2061.append((stat.mean(ssp245_pr_timeseries_order[i][213:232]) - stat.mean(ssp245_pr_timeseries_order[i][147:166]))/stat.mean(ssp245_pr_timeseries_order[i][147:166])*100)
    ssp245_div_2041.append((stat.mean(ssp245_pr_timeseries_order[i][193:212]) - stat.mean(ssp245_pr_timeseries_order[i][147:166]))/stat.mean(ssp245_pr_timeseries_order[i][147:166])*100)
    ssp245_div_2021.append((stat.mean(ssp245_pr_timeseries_order[i][177:192]) - stat.mean(ssp245_pr_timeseries_order[i][147:166]))/stat.mean(ssp245_pr_timeseries_order[i][147:166])*100)
        
    

ssp370_div_2021 = []
ssp370_div_2041 = []
ssp370_div_2061 = []
ssp370_div_2081 = []
for i in range(0,len(ssp370_pr_timeseries_order)):
    ssp370_div_2081.append((stat.mean(ssp370_pr_timeseries_order[i][233:252]) - stat.mean(ssp370_pr_timeseries_order[i][147:166]))/stat.mean(ssp370_pr_timeseries_order[i][147:166])*100)
    ssp370_div_2061.append((stat.mean(ssp370_pr_timeseries_order[i][213:232]) - stat.mean(ssp370_pr_timeseries_order[i][147:166]))/stat.mean(ssp370_pr_timeseries_order[i][147:166])*100)
    ssp370_div_2041.append((stat.mean(ssp370_pr_timeseries_order[i][193:212]) - stat.mean(ssp370_pr_timeseries_order[i][147:166]))/stat.mean(ssp370_pr_timeseries_order[i][147:166])*100)
    ssp370_div_2021.append((stat.mean(ssp370_pr_timeseries_order[i][177:192]) - stat.mean(ssp370_pr_timeseries_order[i][147:166]))/stat.mean(ssp370_pr_timeseries_order[i][147:166])*100)
        

ssp585_div_2021 = []
ssp585_div_2041 = []
ssp585_div_2061 = []
ssp585_div_2081 = []
for i in range(0,len(ssp585_pr_timeseries_order)):
    ssp585_div_2081.append((stat.mean(ssp585_pr_timeseries_order[i][233:252]) - stat.mean(ssp585_pr_timeseries_order[i][147:166]))/stat.mean(ssp585_pr_timeseries_order[i][147:166])*100)
    ssp585_div_2061.append((stat.mean(ssp585_pr_timeseries_order[i][213:232]) - stat.mean(ssp585_pr_timeseries_order[i][147:166]))/stat.mean(ssp585_pr_timeseries_order[i][147:166])*100)
    ssp585_div_2041.append((stat.mean(ssp585_pr_timeseries_order[i][193:212]) - stat.mean(ssp585_pr_timeseries_order[i][147:166]))/stat.mean(ssp585_pr_timeseries_order[i][147:166])*100)
    ssp585_div_2021.append((stat.mean(ssp585_pr_timeseries_order[i][177:192]) - stat.mean(ssp585_pr_timeseries_order[i][147:166]))/stat.mean(ssp585_pr_timeseries_order[i][147:166])*100)
    
#%%
ssp126_pr_model_order.reverse()
ssp126_div_2081.reverse()

ssp245_pr_model_order.reverse()
ssp245_div_2081.reverse()    

ssp370_pr_model_order.reverse()
ssp370_div_2081.reverse()
    
ssp585_pr_model_order.reverse()
ssp585_div_2081.reverse()
    
#%%
fig, axs = plt.subplots(1)
plt.style.use('classic')

x = np.arange(len(ssp585_pr_model_order))  # the label locations
barwidth = 0.2  # Adjust the bar width as needed

# Plotting bars for ssp585
axs.barh(x + 0.3, ssp585_div_2081, barwidth, color=col_ssp585, label='SSP5-8.5')

# Plotting bars for ssp370 if the model is present, else plotting empty bars
for i, model in enumerate(ssp585_pr_model_order):
    if model in ssp370_pr_model_order:
        index_ssp370 = ssp370_pr_model_order.index(model)
        axs.barh(x[i] + 0.1, ssp370_div_2081[index_ssp370], barwidth, color=col_ssp370, label='SSP3-7.0' if i == 0 else "")

# Plotting bars for ssp245
axs.barh(x - 0.1, ssp245_div_2081, barwidth, color=col_ssp245, label='SSP2-4.5')

# Plotting bars for ssp126 if the model is present, else plotting empty bars
for i, model in enumerate(ssp585_pr_model_order):
    if model in ssp126_pr_model_order:
        index_ssp126 = ssp126_pr_model_order.index(model)
        axs.barh(x[i] - 0.3, ssp126_div_2081[index_ssp126], barwidth, color=col_ssp126, label='SSP1-2.6' if i == 0 else "")

axs.set_ylim([-1, len(ssp585_pr_model_order)])
axs.set_xlim([0, 25])
plt.xlabel("$\Delta$ Monsoon Rainfall (2081-2100; %)")
axs.set_yticks(x)
axs.set_yticklabels(ssp585_pr_model_order)
axs.grid(b=None, which='major', axis='y', color='grey', linestyle=':', linewidth=0.5)

# Add legend for scenario colors outside the loop
axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, frameon = False)  # Adjust the parameters as needed

plt.tight_layout()


plt.savefig(dir_save + 'Barplots_good_2081.pdf',bbox_to_inches = "tight")
   

#%%

results = {}

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']  # Add other scenarios if needed
periods = ['2021-2040', '2041-2060', '2061-2080', '2081-2100']

# Populate the dictionary with the calculated values
for scenario in scenarios:
    scenario_results = []
    for period in periods:
        key = f'{scenario}_{period}'
        column_year = period.split('-')[0]  # Extract the starting year from the period
        column = f'{scenario}_div_{column_year}'  # Construct the variable name
        values = globals()[column]  # Assuming you have the variables defined
        min_val = np.min(values).round(1)
        mean_val = np.mean(values).round(1)
        max_val = np.max(values).round(1)

        scenario_results.extend([min_val, mean_val, max_val])

    results[scenario] = scenario_results

# Create a DataFrame from the results dictionary
df = pd.DataFrame(results, index=pd.MultiIndex.from_product([periods, ['Min', 'Mean', 'Max']], names=['Period', 'Stat']))

# Transpose the DataFrame for the desired format
df = df.T


excel_filename = dir_save + 'Mean_changes.xlsx'
df.to_excel(excel_filename)

# Display the DataFrame
print(df)




#%%---------------
# SPATIAL CHANGE
#---------------

# hist_pr_spatial
# hist_pr_model 

# ssp585_pr_model_good
# ssp585_pr_spatial


hist_pr_model_spatialorder = []
hist_pr_spatial_spatialorder = []
for i in range(0,6):
    wanted = good_models[i]
    print(wanted)
    for j in range(0,len(hist_pr_model)):
        if hist_pr_model[j] == wanted:
            print(hist_pr_model[j])
            hist_pr_model_spatialorder.append(hist_pr_model[j])
            hist_pr_spatial_spatialorder.append(hist_pr_spatial[j])

ssp126_pr_model_spatialorder = []
ssp126_pr_spatial_spatialorder = []
for i in range(0,6):
    wanted = good_models[i]
    print(wanted)
    for j in range(0,len(ssp126_pr_model)):
        if ssp126_pr_model[j] == wanted:
            print(ssp126_pr_model[j])
            ssp126_pr_model_spatialorder.append(ssp126_pr_model[j])

            ssp126_pr_spatial_spatialorder.append(ssp126_pr_spatial[j])

ssp245_pr_model_spatialorder = []
ssp245_pr_spatial_spatialorder = []
for i in range(0,6):
    wanted = good_models[i]
    print(wanted)
    for j in range(0,len(ssp245_pr_model)):
        if ssp245_pr_model[j] == wanted:
            print(ssp245_pr_model[j])
            ssp245_pr_model_spatialorder.append(ssp245_pr_model[j])

            ssp245_pr_spatial_spatialorder.append(ssp245_pr_spatial[j])

ssp370_pr_model_spatialorder = []
ssp370_pr_spatial_spatialorder = []
for i in range(0,6):
    wanted = good_models[i]
    print(wanted)
    for j in range(0,len(ssp370_pr_model)):
        if ssp370_pr_model[j] == wanted:
            print(ssp370_pr_model[j])
            ssp370_pr_model_spatialorder.append(ssp370_pr_model[j])

            ssp370_pr_spatial_spatialorder.append(ssp370_pr_spatial[j])


ssp585_pr_model_spatialorder = []
ssp585_pr_spatial_spatialorder = []
for i in range(0,6):
    wanted = good_models[i]
    print(wanted)
    for j in range(0,len(ssp585_pr_model)):
        if ssp585_pr_model[j] == wanted:
            print(ssp585_pr_model[j])
            ssp585_pr_model_spatialorder.append(ssp585_pr_model[j])

            ssp585_pr_spatial_spatialorder.append(ssp585_pr_spatial[j])




ssp245_pr_spatial_spatialorder_dif_rel = []
ssp585_pr_spatial_spatialorder_dif_rel = []
ssp245_pr_spatial_spatialorder_dif = []
ssp585_pr_spatial_spatialorder_dif = []
for i in range(0,6): 
    ssp585_pr_spatial_spatialorder_dif.append(ssp585_pr_spatial_spatialorder[i] - hist_pr_spatial_spatialorder[i])
    ssp245_pr_spatial_spatialorder_dif.append(ssp245_pr_spatial_spatialorder[i] - hist_pr_spatial_spatialorder[i])
    ssp585_pr_spatial_spatialorder_dif_rel.append((ssp585_pr_spatial_spatialorder[i] - hist_pr_spatial_spatialorder[i])/hist_pr_spatial_spatialorder[i])
    ssp245_pr_spatial_spatialorder_dif_rel.append((ssp245_pr_spatial_spatialorder[i] - hist_pr_spatial_spatialorder[i])/hist_pr_spatial_spatialorder[i])

ssp126_pr_spatial_spatialorder_dif_rel = []
ssp126_pr_spatial_spatialorder_dif = []
for i in range(0,5): 
    ssp126_pr_spatial_spatialorder_dif.append(ssp126_pr_spatial_spatialorder[i] - hist_pr_spatial_spatialorder[i])
    ssp126_pr_spatial_spatialorder_dif_rel.append((ssp126_pr_spatial_spatialorder[i] - hist_pr_spatial_spatialorder[i])/hist_pr_spatial_spatialorder[i])

ssp370_pr_spatial_spatialorder_dif_rel = []
ssp370_pr_spatial_spatialorder_dif = []
for i in range(0,4): 
    ssp370_pr_spatial_spatialorder_dif.append(ssp370_pr_spatial_spatialorder[i] - hist_pr_spatial_spatialorder[i])
    ssp370_pr_spatial_spatialorder_dif_rel.append((ssp370_pr_spatial_spatialorder[i] - hist_pr_spatial_spatialorder[i])/hist_pr_spatial_spatialorder[i])



# Create TOP6 Multi model mean

# Concatenate the list of DataArrays along a new dimension 'scenario'
ssp126_pr_spatial_spatialorder_dif_con = xr.concat(ssp126_pr_spatial_spatialorder_dif, dim='models')
ssp245_pr_spatial_spatialorder_dif_con = xr.concat(ssp245_pr_spatial_spatialorder_dif, dim='models')
ssp370_pr_spatial_spatialorder_dif_con = xr.concat(ssp370_pr_spatial_spatialorder_dif, dim='models')
ssp585_pr_spatial_spatialorder_dif_con = xr.concat(ssp585_pr_spatial_spatialorder_dif, dim='models')

ssp126_pr_spatial_spatialorder_dif_rel_con = xr.concat(ssp126_pr_spatial_spatialorder_dif_rel, dim='models')
ssp245_pr_spatial_spatialorder_dif_rel_con = xr.concat(ssp245_pr_spatial_spatialorder_dif_rel, dim='models')
ssp370_pr_spatial_spatialorder_dif_rel_con = xr.concat(ssp370_pr_spatial_spatialorder_dif_rel, dim='models')
ssp585_pr_spatial_spatialorder_dif_rel_con = xr.concat(ssp585_pr_spatial_spatialorder_dif_rel, dim='models')


# Calculate the mean across all DataArrays
ssp126_pr_spatial_top6 = ssp126_pr_spatial_spatialorder_dif_con.mean(dim='models')
ssp245_pr_spatial_top6 = ssp245_pr_spatial_spatialorder_dif_con.mean(dim='models')
ssp370_pr_spatial_top6 = ssp370_pr_spatial_spatialorder_dif_con.mean(dim='models')
ssp585_pr_spatial_top6 = ssp585_pr_spatial_spatialorder_dif_con.mean(dim='models')

ssp126_pr_spatial_top6_rel = ssp126_pr_spatial_spatialorder_dif_rel_con.mean(dim='models')
ssp245_pr_spatial_top6_rel = ssp245_pr_spatial_spatialorder_dif_rel_con.mean(dim='models')
ssp370_pr_spatial_top6_rel = ssp370_pr_spatial_spatialorder_dif_rel_con.mean(dim='models')
ssp585_pr_spatial_top6_rel = ssp585_pr_spatial_spatialorder_dif_rel_con.mean(dim='models')


#%%---------------
# SPATIAL CHANGE - INDIVIDUAL MODELS SSP585
#---------------

fig, axs = plt.subplots(2, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 7))
plt.subplots_adjust(wspace=0.15, hspace=0.01)

axs = axs.flatten()  # Flatten the array for easier iteration

for i, ax in enumerate(axs):
    cf = ax.contourf(ssp585_pr_spatial_spatialorder_dif[i].lon, ssp585_pr_spatial_spatialorder_dif[i].lat, ssp585_pr_spatial_spatialorder_dif[i], 
                     transform=ccrs.PlateCarree(), cmap="RdBu", levels = np.arange(-3, 3.5, 0.5), extend = "both")
    ax.coastlines('10m')
    ax.add_feature(country_borders, edgecolor='black')
    xticks = np.linspace(100, 147, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels(xticks, fontsize=7)
    ax.set_title(ssp585_pr_model_spatialorder[i])
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    yticks = np.linspace(25, 45, 3)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_yticklabels(yticks, fontsize=7)
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

cbar_ax = fig.add_axes([0.31, 0.06, 0.4, 0.02])  # Adjust the position and size of the colorbar
cbar = plt.colorbar(cf, cax=cbar_ax, orientation='horizontal')
cbar.set_label('$\Delta$ Monsoon rainfall (mm/day)')

plt.tight_layout(rect=[0, 0.3, 1, 0.97],h_pad = 0.1)  # Adjust layout to prevent title overlap
plt.savefig(dir_save + "cmip6_spatial_change_ssp585.pdf",bbox_inches = "tight")


#%%---------------
# SPATIAL CHANGE - TOP6 Mean
#---------------


country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
cf = axs.contourf(ssp585_pr_spatial_top6.lon,ssp585_pr_spatial_top6.lat,ssp585_pr_spatial_top6,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),extend = 'both', levels = np.arange(-2, 2.5, 0.25))
#axs.set_title('SSP5-8.5', fontsize = 10, pad = 1)
axs.coastlines('10m')
axs.add_feature(country_borders, edgecolor='black')
xticks = np.linspace(100, 147, 5)
axs.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
axs.set_xticklabels(xticks,fontsize=7)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axs.xaxis.set_major_formatter(lon_formatter)       
axs.set_title("SSP5-8.5")
yticks = np.linspace(25, 45, 3)
axs.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
axs.set_yticklabels(yticks,fontsize=7)
lat_formatter = LatitudeFormatter()
axs.yaxis.set_major_formatter(lat_formatter)   
cbar = fig.colorbar(cf,fraction=0.028, pad=0.04)
cbar.set_label(' $\Delta$ Monsoon Rainfall (mm/day)', rotation=270,labelpad=15)
plt.savefig(dir_save + 'Spatial_change_ssp585.pdf',bbox_inches = "tight")

country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
cf = axs.contourf(ssp370_pr_spatial_top6.lon,ssp370_pr_spatial_top6.lat,ssp370_pr_spatial_top6,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),extend = 'both', levels = np.arange(-2, 2.5, 0.25))
#axs.set_title('SSP5-8.5', fontsize = 10, pad = 1)
axs.coastlines('10m')
axs.add_feature(country_borders, edgecolor='black')
xticks = np.linspace(100, 147, 5)
axs.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
axs.set_xticklabels(xticks,fontsize=7)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axs.xaxis.set_major_formatter(lon_formatter)       
axs.set_title("SSP3-7.0")
yticks = np.linspace(25, 45, 3)
axs.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
axs.set_yticklabels(yticks,fontsize=7)
lat_formatter = LatitudeFormatter()
axs.yaxis.set_major_formatter(lat_formatter)   
cbar = fig.colorbar(cf,fraction=0.028, pad=0.04)
cbar.set_label(' $\Delta$ Monsoon Rainfall (mm/day)', rotation=270,labelpad=15)
plt.savefig(dir_save + 'Spatial_change_ssp370.pdf',bbox_inches = "tight")


country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
cf = axs.contourf(ssp245_pr_spatial_top6.lon,ssp245_pr_spatial_top6.lat,ssp245_pr_spatial_top6,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),extend = 'both', levels = np.arange(-2, 2.5, 0.25))
#axs.set_title('SSP5-8.5', fontsize = 10, pad = 1)
axs.coastlines('10m')
axs.add_feature(country_borders, edgecolor='black')
xticks = np.linspace(100, 147, 5)
axs.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
axs.set_xticklabels(xticks,fontsize=7)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axs.xaxis.set_major_formatter(lon_formatter)       
axs.set_title("SSP2-4.5")
yticks = np.linspace(25, 45, 3)
axs.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
axs.set_yticklabels(yticks,fontsize=7)
lat_formatter = LatitudeFormatter()
axs.yaxis.set_major_formatter(lat_formatter)   
cbar = fig.colorbar(cf,fraction=0.028, pad=0.04)
cbar.set_label(' $\Delta$ Monsoon Rainfall (mm/day)', rotation=270,labelpad=15)
plt.savefig(dir_save + 'Spatial_change_ssp245.pdf')


country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
cf = axs.contourf(ssp126_pr_spatial_top6.lon,ssp126_pr_spatial_top6.lat,ssp126_pr_spatial_top6,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),extend = 'both', levels = np.arange(-2, 2.5, 0.25))
#axs.set_title('SSP5-8.5', fontsize = 10, pad = 1)
axs.coastlines('10m')
axs.add_feature(country_borders, edgecolor='black')
xticks = np.linspace(100, 147, 5)
axs.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
axs.set_xticklabels(xticks,fontsize=7)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axs.xaxis.set_major_formatter(lon_formatter)       
axs.set_title("SSP1-2.6")
yticks = np.linspace(25, 45, 3)
axs.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
axs.set_yticklabels(yticks,fontsize=7)
lat_formatter = LatitudeFormatter()
axs.yaxis.set_major_formatter(lat_formatter)   
cbar = fig.colorbar(cf,fraction=0.028, pad=0.04)
cbar.set_label(' $\Delta$ Monsoon Rainfall (mm/day)', rotation=270,labelpad=15)
plt.savefig(dir_save + 'Spatial_change_ssp126.pdf',bbox_inches = "tight")

#%%
# Create a figure with 2x2 layout and adjusted gridspec
fig = plt.figure()
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], wspace=0.2)

# Define SSP scenarios and corresponding data
ssp_scenarios = ['SSP5-8.5', 'SSP3-7.0', 'SSP2-4.5', 'SSP1-2.6']
ssp_data = [ssp585_pr_spatial_top6, ssp370_pr_spatial_top6, ssp245_pr_spatial_top6, ssp126_pr_spatial_top6]

# Iterate over subplots using the gridspec
for i, (scenario, data) in enumerate(zip(ssp_scenarios, ssp_data)):
    row, col = divmod(i, 2)  # Divide i by 2 to get row and column indices
    ax = plt.subplot(gs[row, col], projection=ccrs.PlateCarree())
    cf = ax.contourf(data.lon, data.lat, data, cmap='RdBu', transform=ccrs.PlateCarree(), extend='both', levels=np.arange(-2, 2.5, 0.25))
    ax.coastlines('10m')
    ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none'), edgecolor='black')
    xticks = np.linspace(100, 147, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels(xticks, fontsize=7)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.set_title(scenario)
    yticks = np.linspace(25, 45, 3)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_yticklabels(yticks, fontsize=7)
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)


# Add colorbar below the panels
cbar_ax = fig.add_axes([0.27, 0.07, 0.5, 0.025])  # [left, bottom, width, height]
cbar = plt.colorbar(cf, cax=cbar_ax, orientation='horizontal', label='$\Delta$ Monsoon rainfall (mm/day)')


plt.subplots_adjust(hspace=0.05)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(dir_save + 'Spatial_change_combined.pdf', bbox_inches="tight")
plt.show()




#%%---------------
# CHINA ONLY
#---------------




china = xr.open_dataset(china_dir)
china_mask = china["China"]

ssp126_pr_spatial_top6_china = ssp126_pr_spatial_top6_rel * china_mask
ssp126_pr_spatial_top6_china = ssp126_pr_spatial_top6_china.mean(dim="lon")
weights = np.cos(np.deg2rad(ssp126_pr_spatial_top6_china.lat))
ssp126_pr_spatial_top6_china = ssp126_pr_spatial_top6_china.weighted(weights).mean(dim=['lat'])
  
ssp245_pr_spatial_top6_china = ssp245_pr_spatial_top6_rel * china_mask
ssp245_pr_spatial_top6_china = ssp245_pr_spatial_top6_china.mean(dim="lon")
weights = np.cos(np.deg2rad(ssp245_pr_spatial_top6_china.lat))
ssp245_pr_spatial_top6_china = ssp245_pr_spatial_top6_china.weighted(weights).mean(dim=['lat'])
  
ssp370_pr_spatial_top6_china = ssp370_pr_spatial_top6_rel * china_mask
ssp370_pr_spatial_top6_china = ssp370_pr_spatial_top6_china.mean(dim="lon")
weights = np.cos(np.deg2rad(ssp370_pr_spatial_top6_china.lat))
ssp370_pr_spatial_top6_china = ssp370_pr_spatial_top6_china.weighted(weights).mean(dim=['lat'])
  
ssp585_pr_spatial_top6_china = ssp585_pr_spatial_top6_rel * china_mask
ssp585_pr_spatial_top6_china = ssp585_pr_spatial_top6_china.mean(dim="lon")
weights = np.cos(np.deg2rad(ssp585_pr_spatial_top6_china.lat))
ssp585_pr_spatial_top6_china = ssp585_pr_spatial_top6_china.weighted(weights).mean(dim=['lat'])
  
    
  





#%%
### INTERANNUAL VARIABILITY
### ---------------------------

ssp126_std = []
for i in range(0,5): 
    ssp126_timeseries_detrended = ssp126_pr_timeseries_order[i]-data_ssa_ssp126_order[i]
    ssp126_std.append(stat.stdev(ssp126_timeseries_detrended[202:252])/stat.stdev(ssp126_timeseries_detrended[117:167])*100-100)
    
ssp245_std = []
for i in range(0,6): 
    ssp245_timeseries_detrended = ssp245_pr_timeseries_order[i]-data_ssa_ssp245_order[i]
    ssp245_std.append(stat.stdev(ssp245_timeseries_detrended[202:252])/stat.stdev(ssp245_timeseries_detrended[117:167])*100-100)
    
ssp370_std = []
for i in range(0,4): 
    ssp370_timeseries_detrended = ssp370_pr_timeseries_order[i]-data_ssa_ssp370_order[i]
    ssp370_std.append(stat.stdev(ssp370_timeseries_detrended[202:252])/stat.stdev(ssp370_timeseries_detrended[117:167])*100-100)

ssp585_std = []
for i in range(0,6): 
    ssp585_timeseries_detrended = ssp585_pr_timeseries_order[i]-data_ssa_ssp585_order[i]
    ssp585_std.append(stat.stdev(ssp585_timeseries_detrended[202:252])/stat.stdev(ssp585_timeseries_detrended[117:167])*100-100)
    
    
 
fig, axs = plt.subplots(1)
plt.style.use('classic')

x = np.arange(6)  # the label locations
barwidth = 0.2  # Adjust the bar width as needed

# Plotting bars for ssp585
axs.barh(x + 0.3, ssp585_std, barwidth, color=col_ssp585, label='SSP5-8.5')

# Plotting bars for ssp370 if the model is present, else plotting empty bars
for i, model in enumerate(ssp585_pr_model_order):
    if model in ssp370_pr_model_order:
        index_ssp370 = ssp370_pr_model_order.index(model)
        axs.barh(x[i] + 0.1, ssp370_std[index_ssp370], barwidth, color=col_ssp370, label='SSP3-7.0' if i == 0 else "")

# Plotting bars for ssp245
axs.barh(x - 0.1, ssp245_std, barwidth, color=col_ssp245, label='SSP2-4.5')

# Plotting bars for ssp126 if the model is present, else plotting empty bars
for i, model in enumerate(ssp585_pr_model_order):
    if model in ssp126_pr_model_order:
        index_ssp126 = ssp126_pr_model_order.index(model)
        axs.barh(x[i] - 0.3, ssp126_std[index_ssp126], barwidth, color=col_ssp126, label='SSP1-2.6' if i == 0 else "")

axs.set_ylim([-1, 6])
axs.set_xlim([-25, 80])
plt.axvline(0, color = "black")
plt.xlabel("$\Delta$ Interannual variability (%)")
axs.set_yticks(x)
axs.set_yticklabels(ssp585_pr_model_order)
axs.grid(b=None, which='major', axis='y', color='grey', linestyle=':', linewidth=0.5)

# Add legend for scenario colors outside the loop
axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, frameon = False)  # Adjust the parameters as needed

plt.tight_layout()


plt.savefig(dir_save + 'STD_good.pdf',bbox_to_inches = "tight")
   



#%%
### EXTREME WET SEASONS - ONLY TOP
### ---------------------------
window = 50

### 90 % percentil
# calculate percentil for 1965 - 2015
percentil90_ssp126 = []
for i in range(len(ssp126_pr_timeseries_order)):
    percentil90_ssp126.append(np.percentile(ssp126_pr_timeseries_order[i][117:167],90))

### Check: how many models have values smaller than 90% percentil in 1965-2015 (only for checking)
belows = []
aboves = []
for i in range(len(ssp126_pr_timeseries_order)): # going through all models
    below = 0
    above = 0
    for myear in range(117,167): # for each model going through all 50 years
        if ssp126_pr_timeseries_order[i][myear] < percentil90_ssp126[i]:
            below = below + 1
        if ssp126_pr_timeseries_order[i][myear] > percentil90_ssp126[i]:
            above = above + 1
    belows.append(below)
    aboves.append(above)
    
###
belows = []
aboves = []
for i in range(len(ssp126_pr_timeseries_order)): # going through all models
    below = 0
    above = 0
    for myear in range(202,252): # for each model going through all 50 years
        if ssp126_pr_timeseries_order[i][myear] < percentil90_ssp126[i]:
            below = below + 1
        if ssp126_pr_timeseries_order[i][myear] > percentil90_ssp126[i]:
            above = above + 1
    belows.append(below)
    aboves.append(above)

### 90 % percentil
# calculate percentil for 1965 - 2015
percentil90_ssp245 = []
for i in range(len(ssp245_pr_timeseries_order)):
    percentil90_ssp245.append(np.percentile(ssp245_pr_timeseries_order[i][117:167],90))

### Check: how many models have values smaller than 90% percentil in 1965-2015 (only for checking)
belows = []
aboves = []
for i in range(len(ssp245_pr_timeseries_order)): # going through all models
    below = 0
    above = 0
    for myear in range(117,167): # for each model going through all 50 years
        if ssp245_pr_timeseries_order[i][myear] < percentil90_ssp245[i]:
            below = below + 1
        if ssp245_pr_timeseries_order[i][myear] > percentil90_ssp245[i]:
            above = above + 1
    belows.append(below)
    aboves.append(above)
    
###
belows = []
aboves = []
for i in range(len(ssp245_pr_timeseries_order)): # going through all models
    below = 0
    above = 0
    for myear in range(202,252): # for each model going through all 50 years
        if ssp245_pr_timeseries_order[i][myear] < percentil90_ssp245[i]:
            below = below + 1
        if ssp245_pr_timeseries_order[i][myear] > percentil90_ssp245[i]:
            above = above + 1
    belows.append(below)
    aboves.append(above)



### 90 % percentil
# calculate percentil for 1965 - 2015
percentil90_ssp370 = []
for i in range(len(ssp370_pr_timeseries_order)):
    percentil90_ssp370.append(np.percentile(ssp370_pr_timeseries_order[i][117:167],90))

### Check: how many models have values smaller than 90% percentil in 1965-2015 (only for checking)
belows = []
aboves = []
for i in range(len(ssp370_pr_timeseries_order)): # going through all models
    below = 0
    above = 0
    for myear in range(117,167): # for each model going through all 50 years
        if ssp370_pr_timeseries_order[i][myear] < percentil90_ssp370[i]:
            below = below + 1
        if ssp370_pr_timeseries_order[i][myear] > percentil90_ssp370[i]:
            above = above + 1
    belows.append(below)
    aboves.append(above)
    
###
belows = []
aboves = []
for i in range(len(ssp370_pr_timeseries_order)): # going through all models
    below = 0
    above = 0
    for myear in range(202,252): # for each model going through all 50 years
        if ssp370_pr_timeseries_order[i][myear] < percentil90_ssp370[i]:
            below = below + 1
        if ssp370_pr_timeseries_order[i][myear] > percentil90_ssp370[i]:
            above = above + 1
    belows.append(below)
    aboves.append(above)



### 90 % percentil
# calculate percentil for 1965 - 2015
percentil90_ssp585 = []
for i in range(len(ssp585_pr_timeseries_order)):
    percentil90_ssp585.append(np.percentile(ssp585_pr_timeseries_order[i][117:167],90))

### Check: how many models have values smaller than 90% percentil in 1965-2015 (only for checking)
belows = []
aboves = []
for i in range(len(ssp585_pr_timeseries_order)): # going through all models
    below = 0
    above = 0
    for myear in range(117,167): # for each model going through all 50 years
        if ssp585_pr_timeseries_order[i][myear] < percentil90_ssp585[i]:
            below = below + 1
        if ssp585_pr_timeseries_order[i][myear] > percentil90_ssp585[i]:
            above = above + 1
    belows.append(below)
    aboves.append(above)
    
###
belows = []
aboves = []
for i in range(len(ssp585_pr_timeseries_order)): # going through all models
    print(i)
    below = 0
    above = 0
    for myear in range(202,252): # for each model going through all 50 years
        if ssp585_pr_timeseries_order[i][myear] < percentil90_ssp585[i]:
            below = below + 1
        if ssp585_pr_timeseries_order[i][myear] > percentil90_ssp585[i]:
            above = above + 1
    belows.append(below)
    aboves.append(above)


#%%
### EXTREME WET SEASONS - ALL MODELS
### ---------------------------
window = 50

### 90 % percentil
# calculate percentil for 1965 - 2015
percentil90_ssp585 = []
for i in range(len(ssp585_pr_timeseries_me)):
    percentil90_ssp585.append(np.percentile(ssp585_pr_timeseries_me[i][117:167],90))

### Check: how many models have values smaller than 90% percentil in 1965-2015 (only for checking)
belows = []
aboves = []
for i in range(len(ssp585_pr_timeseries_me)): # going through all models
    below = 0
    above = 0
    for myear in range(117,167): # for each model going through all 50 years
        if ssp585_pr_timeseries_me[i][myear] < percentil90_ssp585[i]:
            below = below + 1
        if ssp585_pr_timeseries_me[i][myear] > percentil90_ssp585[i]:
            above = above + 1
    belows.append(below)
    aboves.append(above)
    
###
belows = []
aboves = []
for i in range(len(ssp585_pr_timeseries_me)): # going through all models
    below = 0
    above = 0
    for myear in range(202,252): # for each model going through all 50 years
        if ssp585_pr_timeseries_me[i][myear] < percentil90_ssp585[i]:
            below = below + 1
        if ssp585_pr_timeseries_me[i][myear] > percentil90_ssp585[i]:
            above = above + 1
    belows.append(below)
    aboves.append(above)

# how many models have values smaller than 90% percentil in 2050-2100 per year
aboves_per_year = []
aboves_per_year_rel = []
for i in range(len(ssp585_pr_timeseries_me)): # going through all models
    above = 0
    aboves_per_year_in_one_model = []
    aboves_per_year_in_one_model_rel = []
    for year in range(151): # 1950-2100 backwards 50 years
        perioddata = ssp585_pr_timeseries_me[i][(50 + year) : (50 + window + year)]
        exceeding = len([y for y in perioddata if y > percentil90_ssp585[i]])
        aboves_per_year_in_one_model.append(exceeding)
        aboves_per_year_in_one_model_rel.append(exceeding/window)
    aboves_per_year.append(aboves_per_year_in_one_model)
    aboves_per_year_rel.append(aboves_per_year_in_one_model_rel)
    
    
# how many models have values smaller than 90% percentil in 2050-2100 per year
aboves_per_year_10mov = []
for model_i in range(32): # going through all models
    aboves_per_year_in_one_model_10mov = []
    for year in range(151): # 1950-2100 backwards 50 years
        aboves_per_year_in_one_model_10mov.append(stat.mean(aboves_per_year[model_i][year:(year + 10)]))
    aboves_per_year_10mov.append(aboves_per_year_in_one_model_10mov)


median_rel = []
for i in range(len(aboves_per_year_rel[0])):
    yearpoints = [model[i] for model in aboves_per_year_rel]
    median_rel.append(stat.median(yearpoints))

median_abs = []
for i in range(len(aboves_per_year[0])):
    yearpoints = [model[i] for model in aboves_per_year]
    median_abs.append(stat.median(yearpoints))


median_abs_10mov = []
for i in range(len(aboves_per_year_10mov[0])):
    yearpoints = [model[i] for model in aboves_per_year_10mov]
    median_abs_10mov.append(stat.median(yearpoints))

        
# extracting good models to calculate the mean
aboves_per_year_good = []
for s in range(len(aboves_per_year)):
    if ssp585_pr_model[s] in good_models:
        aboves_per_year_good.append(aboves_per_year[s])

median_good = []
for i in range(len(aboves_per_year_good[0])):
    yearpoints = [model[i] for model in aboves_per_year_good]
    median_good.append(stat.median(yearpoints))


# moving average # version 2
# - for all models
aboves_per_year_movav = []
window = 10
for s in range(len(aboves_per_year)):
   aboves_per_year_movav_s = []
   for j in range(len(aboves_per_year[s]) - window):
       aboves_per_year_movav_s.append(stat.mean(aboves_per_year[s][j:(j + window)]))
   aboves_per_year_movav.append(aboves_per_year_movav_s)

median_abs_movav = []
for i in range(len(aboves_per_year_movav[0])):
    yearpoints = [model[i] for model in aboves_per_year_movav]
    median_abs_movav.append(stat.median(yearpoints))

# - for good models
aboves_per_year_good_movav = []
for s in range(len(aboves_per_year_movav)):
    if ssp585_pr_model[s] in good_models:
        aboves_per_year_good_movav.append(aboves_per_year_movav[s])

median_good_movav = []
for i in range(len(aboves_per_year_good_movav[0])):
    yearpoints = [model[i] for model in aboves_per_year_good_movav]
    median_good_movav.append(stat.median(yearpoints))

#%%moving average
plt.style.use('classic')
fig, ax = plt.subplots()
ax2 = ax.twinx()
start = 1950 + window/2-1
end = 2101 - window/2-1
for s in range(len(aboves_per_year)):
    if ssp585_pr_model[s] in good_models:
        ax.plot(np.arange(start,end,1), aboves_per_year_movav[s][0:150], color = "darkblue", label = "models within 2 std", alpha=0.4)
    else:
        ax.plot(np.arange(start,end,1), aboves_per_year_movav[s][0:150], color = "darkorange", label = "models outside 2 std",alpha=0.4)

ax.plot(np.arange(start,end,1), median_good_movav, color = "darkblue", linewidth = 3)
ax.plot(np.arange(start,end,1), median_abs_movav, color = "darkred", linewidth = 3)

plt.xlim(1950,2100)
plt.xticks(np.arange(1950,2101,50))
ax.set_xticklabels(np.arange(1950,2101,50))
ax.set_ylabel("Number of very wet summer seasons in the past 50 years")
ax2.set_ylabel("Fraction of very wet summer seasons in the past 50 years")
#ax.set_yticklabels(np.arange(0,50,5))
ax.set_ylim(0, 50)

ax2.set_ylim(0, 1)
blue_patch = mpatches.Patch(color='darkblue', label='TOP6 models', alpha = 0.4)
grey_patch = mpatches.Patch(color='darkorange', label='Other models', alpha = 0.4)
red_patch = mpatches.Patch(color='darkred', label='Median of all models')
blue_patch2 = mpatches.Patch(color='darkblue', label='Median of TOP6')
plt.legend(handles=[blue_patch2,red_patch,blue_patch, grey_patch],loc=2, prop={'size': 9},frameon=False)
ax.axhline(5, color="grey", linewidth=2, alpha=0.6,linestyle="--")
plt.xlabel("Years")
ax2.plot([],[])

plt.savefig(dir_save + 'Extremes_timeseries.pdf', bbox_inches = "tight")

#%% 
### GMT Dependence 
# -----------------------


files_hist = os.listdir(dir_hist)
files_hist_tas = [files_hist for files_hist in files_hist if files_hist.startswith("tas") and files_hist.endswith("mask2.nc")]

hist_tas_timeseries = []
hist_tas_mean = []
hist_tas_std = []
hist_tas_spatial = []

hist_tas_model = []
hist_tas_center = []


for i in range(0,len(files_hist_tas)):
    data = xr.open_dataset(dir_hist + "/" + files_hist_tas[i])
    model = files_hist_tas[i].split("_")[2]
    if model in good_models:
        print(model)
        hist_tas_model.append(model)
        center = files_hist_tas[i].split("_")[1]
        hist_tas_center.append(center)
        tas = data["tas"]-273.15 # transforming to °C

        # timeseries of mean
        tas_mean = tas.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(tas_mean.lat))
        tas_mean = tas_mean.weighted(weights).mean(dim=['lat'])
        hist_tas_timeseries.append(tas_mean)
        
        # tas mean 1995-2014
        tas_mean_ref = tas_mean.sel(time=(tas_mean['time.year'] >= startyear_ref) & (tas_mean['time.year'] <= endyear_ref))
        tas_mean_reff = tas_mean_ref.mean(dim = ["time"]).data.round(2)
        hist_tas_mean.append(tas_mean_reff)
        
        
        
        
        
        


### SSP126 

files_ssp126 = os.listdir(dir_ssp126)
files_ssp126_tas = [files_ssp126 for files_ssp126 in files_ssp126 if files_ssp126.startswith("tas") and files_ssp126.endswith("mask2.nc")]

ssp126_tas_timeseries = []
ssp126_tas_mean = []
ssp126_tas_model = []
ssp126_tas_center = []

for i in range(0,len(files_ssp126_tas)):
    data = xr.open_dataset(dir_ssp126 + "/" + files_ssp126_tas[i])
    model = files_ssp126_tas[i].split("_")[2]
    if model in good_models:
        print(model)
        ssp126_tas_model.append(model)
        center = files_ssp126_tas[i].split("_")[1]
        ssp126_tas_center.append(center)
        tas = data["tas"]-273 # transforming to mm/day

        # timeseries of mean
        tas_mean = tas.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(tas_mean.lat))
        tas_mean = tas_mean.weighted(weights).mean(dim=['lat'])
        ssp126_tas_timeseries.append(tas_mean)
        
        # tas mean 2081-2100
        tas_mean_ref = tas_mean.sel(time=(tas_mean['time.year'] >= startyear_fut) & (tas_mean['time.year'] <= endyear_fut))
        tas_mean_reff = tas_mean_ref.mean(dim = ["time"]).data.round(2)
        ssp126_tas_mean.append(tas_mean_reff)
        
        


### ssp245 

files_ssp245 = os.listdir(dir_ssp245)
files_ssp245_tas = [files_ssp245 for files_ssp245 in files_ssp245 if files_ssp245.startswith("tas") and files_ssp245.endswith("mask2.nc")]

ssp245_tas_timeseries = []
ssp245_tas_mean = []
ssp245_tas_model = []
ssp245_tas_center = []

for i in range(0,len(files_ssp245_tas)):
    data = xr.open_dataset(dir_ssp245 + "/" + files_ssp245_tas[i])
    model = files_ssp245_tas[i].split("_")[2]
    if model in good_models:
        print(model)
        ssp245_tas_model.append(model)
        center = files_ssp245_tas[i].split("_")[1]
        ssp245_tas_center.append(center)
        tas = data["tas"]-273 # transforming to mm/day

        # timeseries of mean
        tas_mean = tas.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(tas_mean.lat))
        tas_mean = tas_mean.weighted(weights).mean(dim=['lat'])
        ssp245_tas_timeseries.append(tas_mean)
        
        # tas mean 2081-2100
        tas_mean_ref = tas_mean.sel(time=(tas_mean['time.year'] >= startyear_fut) & (tas_mean['time.year'] <= endyear_fut))
        tas_mean_reff = tas_mean_ref.mean(dim = ["time"]).data.round(2)
        ssp245_tas_mean.append(tas_mean_reff)
        
        
        
        
        
### ssp370 

files_ssp370 = os.listdir(dir_ssp370)
files_ssp370_tas = [files_ssp370 for files_ssp370 in files_ssp370 if files_ssp370.startswith("tas") and files_ssp370.endswith("mask2.nc")]

ssp370_tas_timeseries = []
ssp370_tas_mean = []
ssp370_tas_model = []
ssp370_tas_center = []

for i in range(0,len(files_ssp370_tas)):
    data = xr.open_dataset(dir_ssp370 + "/" + files_ssp370_tas[i])
    model = files_ssp370_tas[i].split("_")[2]
    if model in good_models:
        print(model)
        ssp370_tas_model.append(model)
        center = files_ssp370_tas[i].split("_")[1]
        ssp370_tas_center.append(center)
        tas = data["tas"]-273 # transforming to mm/day

        # timeseries of mean
        tas_mean = tas.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(tas_mean.lat))
        tas_mean = tas_mean.weighted(weights).mean(dim=['lat'])
        ssp370_tas_timeseries.append(tas_mean)
        
        # tas mean 2081-2100
        tas_mean_ref = tas_mean.sel(time=(tas_mean['time.year'] >= startyear_fut) & (tas_mean['time.year'] <= endyear_fut))
        tas_mean_reff = tas_mean_ref.mean(dim = ["time"]).data.round(2)
        ssp370_tas_mean.append(tas_mean_reff)




### ssp585 

files_ssp585 = os.listdir(dir_ssp585)
files_ssp585_tas = [files_ssp585 for files_ssp585 in files_ssp585 if files_ssp585.startswith("tas") and files_ssp585.endswith("mask2.nc")]

ssp585_tas_timeseries = []
ssp585_tas_mean = []
ssp585_tas_model = []
ssp585_tas_center = []

for i in range(0,len(files_ssp585_tas)):
    data = xr.open_dataset(dir_ssp585 + "/" + files_ssp585_tas[i])
    model = files_ssp585_tas[i].split("_")[2]
    if model in good_models:
        print(model)
        ssp585_tas_model.append(model)
        center = files_ssp585_tas[i].split("_")[1]
        ssp585_tas_center.append(center)
        tas = data["tas"]-273 # transforming to mm/day

        # timeseries of mean
        tas_mean = tas.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(tas_mean.lat))
        tas_mean = tas_mean.weighted(weights).mean(dim=['lat'])
        ssp585_tas_timeseries.append(tas_mean)
        
        # tas mean 2081-2100
        tas_mean_ref = tas_mean.sel(time=(tas_mean['time.year'] >= startyear_fut) & (tas_mean['time.year'] <= endyear_fut))
        tas_mean_reff = tas_mean_ref.mean(dim = ["time"]).data.round(2)
        ssp585_tas_mean.append(tas_mean_reff)



### Merge tas hist with scenarios
ssp126_tas_timeseries_order = []
ssp126_tas_model_order = []
for i in range(len(ssp126_tas_model)):
    index = hist_tas_model.index(ssp126_tas_model[i])
    ssp126_tas_model_order.append(ssp126_tas_model[i])
    ssp126_tas_timeseries_m = np.concatenate((hist_tas_timeseries[index], ssp126_tas_timeseries[i]))
    ssp126_tas_timeseries_order.append(ssp126_tas_timeseries_m)

ssp245_tas_timeseries_order = []
ssp245_tas_model_order = []
for i in range(len(ssp245_tas_model)):
    index = hist_tas_model.index(ssp245_tas_model[i])
    ssp245_tas_model_order.append(ssp245_tas_model[i])
    ssp245_tas_timeseries_m = np.concatenate((hist_tas_timeseries[index],ssp245_tas_timeseries[i]))
    ssp245_tas_timeseries_order.append(ssp245_tas_timeseries_m)

ssp370_tas_timeseries_order = []
ssp370_tas_model_order = []
for i in range(len(ssp370_tas_model)):
    index = hist_tas_model.index(ssp370_tas_model[i])
    ssp370_tas_model_order.append(ssp370_tas_model[i])
    ssp370_tas_timeseries_m = np.concatenate((hist_tas_timeseries[index], ssp370_tas_timeseries[i]))
    ssp370_tas_timeseries_order.append(ssp370_tas_timeseries_m)

ssp585_tas_timeseries_order = []
ssp585_tas_model_order = []
for i in range(len(ssp585_tas_model)):
    index = hist_tas_model.index(ssp585_tas_model[i])
    ssp585_tas_model_order.append(ssp585_tas_model[i])
    ssp585_tas_timeseries_m = np.concatenate((hist_tas_timeseries[index], ssp585_tas_timeseries[i]))
    ssp585_tas_timeseries_order.append(ssp585_tas_timeseries_m)


#%%

### calculate change in mean precipitation and temperature since 1995-2014 for each model

# creates list, each entry is one list for one model giving delta changes for 16 periods
period_delta = 20 # how long are the single periods
s_plus = 5 # how big are the steps to the next period
r_val = 17 # number of steps
y = 165 # Which year to end the first period: 1850 + y 


# pr
delta_pr_ssp126 = []
for i in range(len(ssp126_pr_model_order)):
    delta_for_model_i = []
    for s in range(r_val): # 165-251
        delta_for_model_i.append(stat.mean(ssp126_pr_timeseries_order[i][(y + s * s_plus - period_delta):(y + s  * s_plus)]) - stat.mean(ssp126_pr_timeseries_order[i][145:165]))    
    delta_pr_ssp126.append(delta_for_model_i)

delta_tas_ssp126 = []
for i in range(len(ssp126_tas_model_order)):
    delta_for_model_i = []
    for s in range(r_val): # 165-251
        delta_for_model_i.append(stat.mean(ssp126_tas_timeseries_order[i][(y + s * s_plus - period_delta):(y + s  * s_plus)]) - stat.mean(ssp126_tas_timeseries_order[i][145:165]))    
    delta_tas_ssp126.append(delta_for_model_i)



delta_pr_ssp245 = []
for i in range(len(ssp245_pr_model_order)):
    delta_for_model_i = []
    for s in range(r_val): # 165-251
        delta_for_model_i.append(stat.mean(ssp245_pr_timeseries_order[i][(y + s * s_plus - period_delta):(y + s  * s_plus)]) - stat.mean(ssp245_pr_timeseries_order[i][145:165]))    
    delta_pr_ssp245.append(delta_for_model_i)

delta_tas_ssp245 = []
for i in range(len(ssp245_tas_model_order)):
    delta_for_model_i = []
    for s in range(r_val): # 165-251
        delta_for_model_i.append(stat.mean(ssp245_tas_timeseries_order[i][(y + s * s_plus - period_delta):(y + s  * s_plus)]) - stat.mean(ssp245_tas_timeseries_order[i][145:165]))    
    delta_tas_ssp245.append(delta_for_model_i)



delta_pr_ssp370 = []
for i in range(len(ssp370_pr_model_order)):
    delta_for_model_i = []
    for s in range(r_val): # 165-251
        delta_for_model_i.append(stat.mean(ssp370_pr_timeseries_order[i][(y + s * s_plus - period_delta):(y + s  * s_plus)]) - stat.mean(ssp370_pr_timeseries_order[i][145:165]))    
    delta_pr_ssp370.append(delta_for_model_i)

delta_tas_ssp370 = []
for i in range(len(ssp370_tas_model_order)):
    delta_for_model_i = []
    for s in range(r_val): # 165-251
        delta_for_model_i.append(stat.mean(ssp370_tas_timeseries_order[i][(y + s * s_plus - period_delta):(y + s  * s_plus)]) - stat.mean(ssp370_tas_timeseries_order[i][145:165]))    
    delta_tas_ssp370.append(delta_for_model_i)



delta_pr_ssp585 = []
for i in range(len(ssp585_pr_model_order)):
    delta_for_model_i = []
    for s in range(r_val): # 165-251
        delta_for_model_i.append(stat.mean(ssp585_pr_timeseries_order[i][(y + s * s_plus - period_delta):(y + s  * s_plus)]) - stat.mean(ssp585_pr_timeseries_order[i][145:165]))    
    delta_pr_ssp585.append(delta_for_model_i)

delta_tas_ssp585 = []
for i in range(len(ssp585_tas_model_order)):
    delta_for_model_i = []
    for s in range(r_val): # 165-251
        delta_for_model_i.append(stat.mean(ssp585_tas_timeseries_order[i][(y + s * s_plus - period_delta):(y + s  * s_plus)]) - stat.mean(ssp585_tas_timeseries_order[i][145:165]))    
    delta_tas_ssp585.append(delta_for_model_i)



#%%
### Bring in same order 

delta_tas_ssp126_sort = []
ssp126_tas_model_order_sort = []
for i in range(0,6):
    wanted = good_models[i]
    for j in range(0,len(ssp126_tas_model_order)):
        if ssp126_tas_model_order[j] == wanted:
            ssp126_tas_model_order_sort.append(ssp126_tas_model_order[j])
            delta_tas_ssp126_sort.append(delta_tas_ssp126[j])
            
delta_pr_ssp126_sort = []
ssp126_pr_model_order_sort = []
for i in range(0,6):
    wanted = good_models[i]
    for j in range(0,len(ssp126_pr_model_order)):
        if ssp126_pr_model_order[j] == wanted:
            ssp126_pr_model_order_sort.append(ssp126_pr_model_order[j])
            delta_pr_ssp126_sort.append(delta_pr_ssp126[j])
            

delta_tas_ssp245_sort = []
ssp245_tas_model_order_sort = []
for i in range(0,6):
    wanted = good_models[i]
    for j in range(0,len(ssp245_tas_model_order)):
        if ssp245_tas_model_order[j] == wanted:
            ssp245_tas_model_order_sort.append(ssp245_tas_model_order[j])
            delta_tas_ssp245_sort.append(delta_tas_ssp245[j])
            
delta_pr_ssp245_sort = []
ssp245_pr_model_order_sort = []
for i in range(0,6):
    wanted = good_models[i]
    for j in range(0,len(ssp245_pr_model_order)):
        if ssp245_pr_model_order[j] == wanted:
            ssp245_pr_model_order_sort.append(ssp245_pr_model_order[j])
            delta_pr_ssp245_sort.append(delta_pr_ssp245[j])
            

delta_tas_ssp370_sort = []
ssp370_tas_model_order_sort = []
for i in range(0,6):
    wanted = good_models[i]
    for j in range(0,len(ssp370_tas_model_order)):
        if ssp370_tas_model_order[j] == wanted:
            ssp370_tas_model_order_sort.append(ssp370_tas_model_order[j])
            delta_tas_ssp370_sort.append(delta_tas_ssp370[j])
            
delta_pr_ssp370_sort = []
ssp370_pr_model_order_sort = []
for i in range(0,6):
    wanted = good_models[i]
    for j in range(0,len(ssp370_pr_model_order)):
        if ssp370_pr_model_order[j] == wanted:
            ssp370_pr_model_order_sort.append(ssp370_pr_model_order[j])
            delta_pr_ssp370_sort.append(delta_pr_ssp370[j])
            

delta_tas_ssp585_sort = []
ssp585_tas_model_order_sort = []
for i in range(0,6):
    wanted = good_models[i]
    for j in range(0,len(ssp585_tas_model_order)):
        if ssp585_tas_model_order[j] == wanted:
            ssp585_tas_model_order_sort.append(ssp585_tas_model_order[j])
            delta_tas_ssp585_sort.append(delta_tas_ssp585[j])
            
delta_pr_ssp585_sort = []
ssp585_pr_model_order_sort = []
for i in range(0,6):
    wanted = good_models[i]
    for j in range(0,len(ssp585_pr_model_order)):
        if ssp585_pr_model_order[j] == wanted:
            ssp585_pr_model_order_sort.append(ssp585_pr_model_order[j])
            delta_pr_ssp585_sort.append(delta_pr_ssp585[j])
            

#%%

# delta_tas_ord_ssp585 = []
# delta_pr_ord_ssp585 = []
# delta_pr_tas_ord_ssp585 = []
# delta_pr_tas_ord_ssp585_rel = []
# for t in range(len(tas_models_ssp585)):
#     for p in range(len(models_ssp585)):
#         if models_ssp585[p] == tas_models_ssp585[t] and tas_models_ssp585[t] in good_models:
#             delta_pr_ord_ssp585.append(delta_ssp585[p])
#             delta_tas_ord_ssp585.append(delta_tas_ssp585[t])
#             quot = delta_ssp585[p][16]/delta_tas_ssp585[t][16]
#             quot_rel = (delta_ssp585[p][16]/means_hist_ssp585[p])/delta_tas_ssp585[t][16]
#             delta_pr_tas_ord_ssp585.append(quot)
#             delta_pr_tas_ord_ssp585_rel.append(quot_rel)

    
    
    
# Dependence on GMT
# min(delta_pr_tas_ord_ssp585_rel)
# max(delta_pr_tas_ord_ssp585_rel)
# stat.mean(delta_pr_tas_ord_ssp585_rel)

# min(delta_pr_tas_ord_ssp585)
# max(delta_pr_tas_ord_ssp585)
# stat.mean(delta_pr_tas_ord_ssp585)


# Creating GMTplot for all good models
plt.figure()
l = 1
for i in range(len(delta_tas_ssp585_sort)):
    plt.plot([0]+delta_tas_ssp585_sort[i],[0]+delta_pr_ssp585_sort[i],color = col_ssp585, linewidth = l)

for i in range(len(delta_tas_ssp370_sort)):
    plt.plot([0]+delta_tas_ssp370_sort[i],[0]+delta_pr_ssp370_sort[i],color = col_ssp370, linewidth = l)

for i in range(len(delta_tas_ssp245_sort)):
    plt.plot([0]+delta_tas_ssp245_sort[i],[0]+delta_pr_ssp245_sort[i],color = col_ssp245, linewidth = l)

for i in range(len(delta_tas_ssp126_sort)):
    plt.plot([0]+delta_tas_ssp126_sort[i],[0]+delta_pr_ssp126_sort[i],color = col_ssp126, linewidth = l)

plt.xlabel("$\Delta$ GMT (K)")
plt.ylabel("$\Delta$ JJA mean rainfall (mm/day)")
#plt.xlim([0,8.2])
plt.axhline(0, color='black', linestyle = '-', linewidth = 0.5)
plt.savefig(dir_save + '/GMTplot_good.pdf', bbox_inches='tight')
plt.show()



#%%

## calculate model mean


tas_series_ssp126 = []
pr_series_ssp126 = []
for i in range(len(delta_tas_ssp126_sort[0])):
    tas_year_i = stat.mean(item[i] for item in delta_tas_ssp126_sort)
    pr_year_i = stat.mean(itemm[i] for itemm in delta_pr_ssp126_sort)
    tas_series_ssp126.append(tas_year_i)
    pr_series_ssp126.append(pr_year_i)
    
tas_series_ssp245 = []
pr_series_ssp245 = []
for i in range(len(delta_tas_ssp245_sort[0])):
    tas_year_i = stat.mean(item[i] for item in delta_tas_ssp245_sort)
    pr_year_i = stat.mean(itemm[i] for itemm in delta_pr_ssp245_sort)
    tas_series_ssp245.append(tas_year_i)
    pr_series_ssp245.append(pr_year_i)
    
tas_series_ssp370 = []
pr_series_ssp370 = []
for i in range(len(delta_tas_ssp370_sort[0])):
    tas_year_i = stat.mean(item[i] for item in delta_tas_ssp370_sort)
    pr_year_i = stat.mean(itemm[i] for itemm in delta_pr_ssp370_sort)
    tas_series_ssp370.append(tas_year_i)
    pr_series_ssp370.append(pr_year_i)
    
tas_series_ssp585 = []
pr_series_ssp585 = []
for i in range(len(delta_tas_ssp585_sort[0])):
    tas_year_i = stat.mean(item[i] for item in delta_tas_ssp585_sort)
    pr_year_i = stat.mean(itemm[i] for itemm in delta_pr_ssp585_sort)
    tas_series_ssp585.append(tas_year_i)
    pr_series_ssp585.append(pr_year_i)

 
### Creating GMT plot for multi-model mean of good models 
lw = 2
plt.figure()
plt.plot(tas_series_ssp126, pr_series_ssp126, color = col_ssp126, label = "SSP1-2.6",linewidth = lw)
plt.plot(tas_series_ssp245, pr_series_ssp245, color = col_ssp245, label = "SSP2-4.5",linewidth = lw)
plt.plot(tas_series_ssp370, pr_series_ssp370, color = col_ssp370, label = "SSP3-7.0",linewidth = lw)
plt.plot(tas_series_ssp585, pr_series_ssp585, color = col_ssp585, label = "SSP5-8.5",linewidth = lw)
plt.legend(loc='best', frameon = False)
plt.xlabel("$\Delta$ GMT (K)")
plt.ylabel("$\Delta$ JJA mean rainfall (mm/day)")
plt.ylim([- 0.2 , 1.4])
plt.xlim([0 , 6])
plt.axhline(0, color='black', linestyle = '-', linewidth = 0.5)
plt.savefig(dir_save + '/GMTplot_good_mean.pdf', bbox_inches='tight')
plt.show()




#%%
# Panel 1: Plot for all good models
l = 1
fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

for i in range(len(delta_tas_ssp585_sort)):
    axs[0].plot([0] + delta_tas_ssp585_sort[i], [0] + delta_pr_ssp585_sort[i], color=col_ssp585, linewidth=l)

for i in range(len(delta_tas_ssp370_sort)):
    axs[0].plot([0] + delta_tas_ssp370_sort[i], [0] + delta_pr_ssp370_sort[i], color=col_ssp370, linewidth=l)

for i in range(len(delta_tas_ssp245_sort)):
    axs[0].plot([0] + delta_tas_ssp245_sort[i], [0] + delta_pr_ssp245_sort[i], color=col_ssp245, linewidth=l)

for i in range(len(delta_tas_ssp126_sort)):
    axs[0].plot([0] + delta_tas_ssp126_sort[i], [0] + delta_pr_ssp126_sort[i], color=col_ssp126, linewidth=l)

axs[0].set_xlabel("$\Delta$ GMT (K)")
axs[0].set_ylabel("$\Delta$ JJA mean rainfall (mm/day)")
# axs[0].set_xlim([0, 8.2])
axs[0].axhline(0, color='black', linestyle='-', linewidth=0.5)

# Panel 2: Multi-model mean plot
lw = 2
axs[1].plot(tas_series_ssp126, pr_series_ssp126, color=col_ssp126, label="SSP1-2.6", linewidth=lw)
axs[1].plot(tas_series_ssp245, pr_series_ssp245, color=col_ssp245, label="SSP2-4.5", linewidth=lw)
axs[1].plot(tas_series_ssp370, pr_series_ssp370, color=col_ssp370, label="SSP3-7.0", linewidth=lw)
axs[1].plot(tas_series_ssp585, pr_series_ssp585, color=col_ssp585, label="SSP5-8.5", linewidth=lw)
axs[1].legend(loc='best', frameon=False)
axs[1].set_xlabel("$\Delta$ GMT (K)")
axs[1].set_ylabel("$\Delta$ JJA mean rainfall (mm/day)")
axs[1].set_ylim([-0.2, 1.4])
axs[1].set_xlim([0, 6])
axs[1].axhline(0, color='black', linestyle='-', linewidth=0.5)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(dir_save + '/combined_GMT_plots.pdf', bbox_inches='tight')
plt.show()

#%%
dif = []
for i in range(0,6):
    dif.append(delta_pr_ssp585_sort[i][16]/delta_tas_ssp585_sort[i][16])

