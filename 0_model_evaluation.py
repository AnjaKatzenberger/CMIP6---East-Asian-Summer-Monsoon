#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
from shapely.geometry import Polygon


####-----------------------------------###
### EAST ASIAN SUMMER MONSOON IN CMIP6 ###
####-----------------------------------###
# author: Anja Katzenberger, anja.katzenberger@pik-potsdam.de

# This code reproduces the results in chapter 3.1. Model Evaluation: 
# Evaluation regarding the mean, standard deviation, spatial distribution and 850hPa wind


#%%---------
#   DIRECTORIES
#-----------

# where to save the results 
dir_save = "/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/figures/fig_new/"

# where to save data output (e.g. mask, model evaluation table)
dir_save_data = "/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/data_new/"

# CMIP6 data
dir_hist = "/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/data_new/EASM_new/historical"

# CMIP6 hist wind data
udir = "/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/data_new/ua_output"
vdir = "/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/data_new/va_output"



# GPCC precipitation reanalysis data (1° x 1° resolution, monthly, 1891-2019, extracted for 100-150°E, 20-50°N)
gpcc_dir = "/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/data_new/GPCC_monthly_1891_2019_10_precip_China2.nc"

# JRA55 wind data
wind_ref = "/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/data_new/JRA55_JJA_yearmean_timmean.nc"


#%%---------
#  PREPARATION
#-----------

startyear_ref = 1995 # start year of reference period for general anaylsis
startyear_ref_std = 1965 # adapted start year of reference period for variability anaylsis
endyear_ref = 2014

country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')

models_analysis = ['CNRM-CM6-1', 'CNRM-ESM2-1', 'BCC-CSM2-MR', 'CESM2', 'CESM2-WACCM', 'FGOALS-f3-L', 'FGOALS-g3', 'UKESM1-0-LL', 'AWI-CM-1-1-MR', 'GFDL-ESM4', 'GFDL-CM4', 'IITM-ESM', 'CanESM5', 'CanESM5-CanOE', 'E3SM-1-1', 'CAMS-CSM1-0', 'KACE-1-0-G', 'INM-CM5-0', 'INM-CM4-8', 'TaiESM1', 'EC-Earth3-CC', 'EC-Earth3', 'CMCC-ESM2', 'CMCC-CM2-SR5', 'ACCESS-ESM1-5', 'MRI-ESM2-0', 'ACCESS-CM2', 'NESM3', 'MIROC-ES2L', 'MIROC6', 'IPSL-CM6A-LR', 'NorESM2-MM', 'FIO-ESM-2-0', 'MPI-ESM1-2-LR']


#%%---------
#   MASK
#-----------
# create mask with 1 if JJA rainfall minus DJF rainfall  > 2 mm/day based on GPCC data

# GPCC precipitation data 
ds_gpcc = xr.open_dataset(gpcc_dir)
pr_gpcc = ds_gpcc['precip']
pr_gpcc = pr_gpcc / pr_gpcc.time.dt.days_in_month # transform from mm/month to mm/day
pr_gpcc_ref = pr_gpcc.sel(time=slice(str(startyear_ref) + '-01-01', str(endyear_ref) + '-12-31'))

# Create mask for all grid cells with precip_JJA - precip_DJF > 2 mm/day
pr_gpcc_ref_jja = pr_gpcc_ref.sel(time=pr_gpcc_ref['time.month'].isin([6, 7, 8]))
pr_gpcc_ref_jja_mean = pr_gpcc_ref_jja.mean(dim=['time'])
pr_gpcc_ref_djf = pr_gpcc_ref.sel(time=pr_gpcc_ref['time.month'].isin([12, 1, 2]))
pr_gpcc_ref_djf_mean = pr_gpcc_ref_djf.mean(dim=['time'])
pr_gpcc_diff = pr_gpcc_ref_jja_mean - pr_gpcc_ref_djf_mean

# Set values greater than 2 to 1, remaining to nan
maskk = pr_gpcc_diff.where(pr_gpcc_diff > 2,  np.nan) # if condition is not met, value is set to np.nan 
mask = maskk.where(maskk.isnull(), 1)
mask.name = "mask"
mask.to_netcdf("mask.nc", mode='w')
#mask = mask.reindex(lat=mask['lat'][::-1])
### You might need to apply "cdo invertlat mask_old.nc mask_invertlat.nc" before applying the mask to CMIP6 data

# Create mask plot 

x_up = 160
x_low = 70
y_up = 60
y_low = 0 

country_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})
cf = axs.contourf(mask.lon, mask.lat, mask, cmap='RdBu', transform=cartopy.crs.PlateCarree(),alpha = 0.7)
axs.add_feature(country_borders, edgecolor='black', linestyle = "-",linewidth = 0.5)
axs.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')  # Set facecolor to green for land
xticks = np.linspace(x_low, x_up, 5)
axs.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
axs.set_xticklabels(xticks, fontsize=7)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axs.xaxis.set_major_formatter(lon_formatter)
yticks = np.linspace(y_low,y_up, 3)
axs.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
axs.set_yticklabels(yticks, fontsize=7)
lat_formatter = LatitudeFormatter()
axs.yaxis.set_major_formatter(lat_formatter)

# Adding poly_show to the plot
poly_used = Polygon([(99.7, 21.2), (99.7, 50.2), (150, 50.2), (150, 21.2), (99.7, 21.2)])
poly_show = Polygon([(x_low, y_low), (x_low, y_up), (x_up, y_up), (x_up, y_low), (x_low, y_low)])
x, y = poly_show.exterior.xy
axs.add_patch(plt.Polygon(xy=list(zip(x, y)), color='none', edgecolor='black', linewidth=2, transform=ccrs.PlateCarree()))

# Adding a red frame around poly_used
x_used, y_used = poly_used.exterior.xy
axs.add_patch(plt.Polygon(xy=list(zip(x_used, y_used)), fill=None, edgecolor='darkred', linewidth=2, transform=ccrs.PlateCarree()))

plt.savefig(dir_save_data + 'area_of_interest_EASM.pdf', bbox_inches='tight')

plt.show()



#%%---------
#   RAINFALL REFERENCE: GPCC
#-----------

# Apply mask to create map of masked GPCC climatology
pr_gpcc_ref_jja_mask = pr_gpcc_ref_jja_mean * mask

# Calculate timeseries of mean 1965-2014, mean, std
pr_gpcc_ref_jja_timeseries = pr_gpcc_ref_jja.groupby('time.year').mean(dim='time')
pr_gpcc_ref_jja_timeseries_mean  = pr_gpcc_ref_jja_timeseries.mean(dim = ["lon"])
weights = np.cos(np.deg2rad(pr_gpcc_ref_jja_timeseries_mean.lat))
pr_gpcc_ref_jja_timeseries_mean = pr_gpcc_ref_jja_timeseries_mean.weighted(weights).mean(dim=['lat'])

pr_gpcc_ref_jja_mask_mean  = pr_gpcc_ref_jja_mask.mean(dim = ["lon"])
weights = np.cos(np.deg2rad(pr_gpcc_ref_jja_mask.lat))
pr_gpcc_ref_jja_mask_mean = pr_gpcc_ref_jja_mask_mean.weighted(weights).mean(dim=['lat'])


#pr_gpcc_mean = pr_gpcc_ref_jja_timeseries_mean.mean(dim=["year"])
print(pr_gpcc_ref_jja_mask_mean)
pr_gpcc_mean = pr_gpcc_ref_jja_mask_mean

pr_gpcc_ref_std = pr_gpcc.sel(time=slice(str(startyear_ref_std) + '-01-01', str(endyear_ref) + '-12-31'))
pr_gpcc_ref_jja_std = pr_gpcc_ref_std.sel(time=pr_gpcc_ref_std['time.month'].isin([6, 7, 8]))
pr_gpcc_ref_jja_timeseries_std = pr_gpcc_ref_jja_std.groupby('time.year').mean(dim='time')
pr_gpcc_ref_jja_timeseries_std = pr_gpcc_ref_jja_timeseries_std * mask
pr_gpcc_ref_jja_timeseries_mean_std  = pr_gpcc_ref_jja_timeseries_std.mean(dim = ["lon"])
weights = np.cos(np.deg2rad(pr_gpcc_ref_jja_timeseries_mean_std.lat))
pr_gpcc_ref_jja_timeseries_mean_std = pr_gpcc_ref_jja_timeseries_mean_std.weighted(weights).mean(dim=['lat'])
pr_gpcc_std = pr_gpcc_ref_jja_timeseries_mean_std.std(dim=["year"])
print(pr_gpcc_std)
# pr_gpcc_std = 0.30 mm/day


### plot GPCC rainfall distribution
country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
cf = axs.contourf(pr_gpcc_ref_jja.lon,pr_gpcc_ref_jja.lat,pr_gpcc_ref_jja_mask,cmap = 'Blues',transform=cartopy.crs.PlateCarree(),extend = 'max', levels = range(0,11))
#axs.set_title('SSP5-8.5', fontsize = 10, pad = 1)
axs.coastlines('10m')
axs.add_feature(country_borders, edgecolor='black')
xticks = np.linspace(100, 147, 5)
axs.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
axs.set_xticklabels(xticks,fontsize=7)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axs.xaxis.set_major_formatter(lon_formatter)       
axs.set_title("GPCC")
yticks = np.linspace(25, 45, 3)
axs.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
axs.set_yticklabels(yticks,fontsize=7)
lat_formatter = LatitudeFormatter()
axs.yaxis.set_major_formatter(lat_formatter)   
cbar = fig.colorbar(cf,fraction=0.028, pad=0.04)
cbar.set_label('JJA mean rainfall (mm/day)', rotation=270,labelpad=15)
plt.savefig(dir_save + 'GPCC_1995_2014.pdf')


### plot JJA-DJF
# country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
# plt.style.use('classic')
# fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
# cf = axs.contourf(pr_gpcc_diff.lon,pr_gpcc_diff.lat,pr_gpcc_diff,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),extend = 'both')
# #axs.set_title('SSP5-8.5', fontsize = 10, pad = 1)
# axs.coastlines('10m')
# axs.add_feature(country_borders, edgecolor='black')
# xticks = np.linspace(101, 140, 5)
# axs.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
# axs.set_xticklabels(xticks,fontsize=7)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# axs.xaxis.set_major_formatter(lon_formatter)       
# yticks = np.linspace(25, 45, 3)
# axs.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
# axs.set_yticklabels(yticks,fontsize=7)
# lat_formatter = LatitudeFormatter()
# axs.yaxis.set_major_formatter(lat_formatter)   
# cbar = fig.colorbar(cf,fraction=0.028, pad=0.04)
# cbar.set_label('JJA -DJF mean rainfall (mm/day)', rotation=270,labelpad=15)
#plt.savefig(dir_save + 'GPCC_1995_2014_dif.pdf')



#%%---------------
# CMIP6 RAINFALL DATA
#---------------
# CMIP6 historical data, monthly

# Use climate data operators (CDOs) for preprocessing
# remaped to 1°x1° (CDO remapcon)
# land only 
# 100-150°E, 20-50°N
# multiplied with the mask from above to get the monsoon area only
# JJA yearmean


files_hist = os.listdir(dir_hist)
files_hist_pr = [files_hist for files_hist in files_hist if files_hist.startswith("pr") and files_hist.endswith("mask2.nc")]

hist_pr_timeseries = []
hist_pr_mean = []
hist_pr_std = []
hist_pr_model = []
hist_pr_center = []
hist_pr_spatial = []
rmse = []
rmse_tot = []

m = 0
for i in range(0,len(files_hist_pr)):
    data = xr.open_dataset(dir_hist + "/" + files_hist_pr[i])
    model = files_hist_pr[i].split("_")[2]
    if model in models_analysis:
        m = m+1
        print(model)
        hist_pr_model.append(model)
        center = files_hist_pr[i].split("_")[1]
        hist_pr_center.append(center)
        pr = data["pr"]*86400 # transforming to mm/day

        pr_mean = pr.mean(dim = ["lon"])
        weights = np.cos(np.deg2rad(pr_mean.lat))
        pr_mean = pr_mean.weighted(weights).mean(dim=['lat'])
        hist_pr_timeseries.append(pr_mean.data)
        
        pr_mean_ref = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_ref) & (pr_mean['time.year'] <= endyear_ref))
        pr_mean_reff = pr_mean_ref.mean(dim = ["time"]).data.round(2)
        hist_pr_mean.append(pr_mean_reff)
        
        pr_mean_ref_std = pr_mean.sel(time=(pr_mean['time.year'] >= startyear_ref_std) & (pr_mean['time.year'] <= endyear_ref))
        hist_pr_std.append(pr_mean_ref_std.std(dim = ["time"]).data.round(2))
    
        pr_ref = pr.sel(time=(pr['time.year'] >= startyear_ref) & (pr['time.year'] <= endyear_ref))
        pr_ref = pr_ref.mean(dim=["time"])
        hist_pr_spatial.append(pr_ref)
        
        pr_ref_m = pr_ref.mean(dim=["lon"])
        weights = np.cos(np.deg2rad(pr_ref_m.lat))
        pr_ref_m = pr_ref_m.mean(dim=["lat"])
        
        ### RMSE
        pr_gpcc_ref_jja_mask = pr_gpcc_ref_jja_mask.reindex(lat=pr_gpcc_ref_jja_mask['lat'][::-1])
        pr_gpcc_ref_jja_mask_aligned = pr_gpcc_ref_jja_mask.reindex(lat=pr_ref['lat'], method='nearest')
        pr_gpcc_ref_jja_mask_aligned = pr_gpcc_ref_jja_mask.reindex(lon=pr_ref['lon'], method='nearest')

        diff = (pr_ref - pr_mean_reff.item()) - (pr_gpcc_ref_jja_mask_aligned - pr_gpcc_mean)
        
        diff_squared = diff**2
        mean_squared_diff = np.nanmean(diff_squared)
        rmses = np.sqrt(mean_squared_diff)
        rmse.append(rmses.round(2))




#%%---------------
# EVALUATION
#---------------

#---------------
### MEAN
#---------------

# ordering according to means
hist_pr_mean_sort = [x for x,_ in sorted(zip(hist_pr_mean, hist_pr_mean))]
hist_pr_std_sort = [x for _,x in sorted(zip(hist_pr_mean, hist_pr_std))]
rmse_sort = [x for _,x in sorted(zip(hist_pr_mean, rmse))]
hist_pr_spatial_sort = [x for _,x in sorted(zip(hist_pr_mean, hist_pr_spatial))]

hist_pr_model_sort = [x for _,x in sorted(zip(hist_pr_mean, hist_pr_model))]
hist_pr_center_sort = [x for _,x in sorted(zip(hist_pr_mean, hist_pr_center))]

# Observation data (GPCC)
ra_mean_hist =  pr_gpcc_mean
ra_std_hist =  pr_gpcc_std

# Plot
plt.style.use('classic')
plt.rcParams['ytick.major.pad']='3' # space between axis and labels
fig,ax = plt.subplots(1,1)
y = np.arange(len(hist_pr_model_sort))
plt.grid(b = None, which = 'major', axis = 'y', color = 'grey', linestyle = ':', linewidth = 0.5)
plt.errorbar(hist_pr_mean_sort, y , xerr = hist_pr_std_sort, fmt = 'ko', capsize = 2, capthick = 1, marker = '.', elinewidth = 0.4, ecolor = "black")
plt.axvline(ra_mean_hist, color = 'black', linewidth = 0.5)
plt.axvline(ra_mean_hist -  2 * ra_std_hist, color='black', linestyle = '--', linewidth = 0.5)
plt.axvline(ra_mean_hist +  2 * ra_std_hist, color='black', linestyle = '--', linewidth = 0.5)
plt.axvspan(ra_mean_hist -  2 * ra_std_hist, ra_mean_hist + 2 * ra_std_hist, color = "black", alpha = 0.05)
ax.set_xlabel('JJA Mean Rainfall (mm/day)') 
plt.ylim([0 - 1 ,len(hist_pr_model_sort) + 0.3])
ax.set_yticks(y)  
ax.set_yticklabels(hist_pr_model_sort)  
ax.xaxis.set_label_position('top') 
plt.tight_layout()
#ax.legend(numpoints = 1, fontsize=12, loc = "upper left")
plt.savefig(dir_save + '/evaluation.pdf',bbox_inches = 'tight')

good_models_mean = [model for model, mean in zip(hist_pr_model_sort, hist_pr_mean_sort) if (ra_mean_hist - 2 * ra_std_hist) <= mean <= (ra_mean_hist + 2 * ra_std_hist)]



#%%---------------
### INTERANNUAL VARIABILITY
#---------------

good_models_std = [model for model, std in zip(hist_pr_model_sort, hist_pr_std_sort) if (ra_std_hist - 0.5 * ra_std_hist) <= std <= (ra_std_hist + 0.5 * ra_std_hist)]



#%%---------------
### RMSE
#---------------

good_models_rmse = [model for model,rmse_sort in zip(hist_pr_model_sort, rmse_sort) if rmse_sort <= 2 ]



#%%---------------
### CREATE OVERVIEW TABLE
#---------------

# Convert the lists to sets
set1 = set(good_models_mean)
set2 = set(good_models_std)
set3 = set(good_models_rmse)

# Find the common models among all subsets
good_models = set1.intersection(set2,set3)#, set3, set4)

# Define the path for saving the Excel file
excel_file_path = dir_save_data + '/Model_evaluation.xlsx'

# Create a DataFrame
data = {'Model': hist_pr_model_sort, 'MEAN': hist_pr_mean_sort, "STD": hist_pr_std_sort, "CRMSE": rmse_sort}
df = pd.DataFrame(data)
df = df.iloc[::-1].reset_index(drop=True) # reverse
 
# Save the styled DataFrame to an Excel file
df.to_excel(excel_file_path, index=False, engine='openpyxl')

# good_models = ['AWI-CM-1-1-MR',
#  'BCC-CSM2-MR',
#  'EC-Earth3',
#  'GFDL-CM4',
#  'IPSL-CM6A-LR',
#  'MPI-ESM1-2-LR']
# we removed  'EC-Earth3-CC' to focus on only one model per model center
# given that the intermodel-dependencies between models from the same center would create a bias of the MME


#%%---------------
### SPATIAL
#---------------
#indices = [i for i, model in enumerate(hist_pr_model_sort) if model in good_models]
indices = [8,13,7,12,11,6]
selected_models = [hist_pr_model_sort[i] for i in indices]
selected_pr = [hist_pr_spatial_sort[i] for i in indices]

fig, axs = plt.subplots(2, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 7))
plt.subplots_adjust(wspace=0.15, hspace=0.01)

axs = axs.flatten()  # Flatten the array for easier iteration

for i, ax in enumerate(axs):
    cf = ax.contourf(selected_pr[i].lon, selected_pr[i].lat, selected_pr[i], 
                     transform=ccrs.PlateCarree(), cmap="Blues", levels = range(0,11), extend = "max")
    ax.coastlines('10m')
    ax.add_feature(country_borders, edgecolor='black')
    xticks = np.linspace(100, 147, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels(xticks, fontsize=7)
    ax.set_title(selected_models[i])
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    yticks = np.linspace(25, 45, 3)
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_yticklabels(yticks, fontsize=7)
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)

cbar_ax = fig.add_axes([0.31, 0.06, 0.4, 0.02])  # Adjust the position and size of the colorbar
cbar = plt.colorbar(cf, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Monsoon rainfall (mm/day)')

plt.tight_layout(rect=[0, 0.3, 1, 0.97],h_pad = 0.1)  # Adjust layout to prevent title overlap
plt.savefig(dir_save + "cmip6_spatial.pdf",bbox_inches = "tight")


#%%-------------
### WIND REANALYSIS
#---------------

### JRA55 WIND
wind = xr.open_dataset(wind_ref)
u_ref = wind['u-component_of_wind_hybrid_Average'].isel(time=0, hybrid=0)
v_ref = wind['v-component_of_wind_hybrid_Average'].isel(time=0, hybrid=0)
lat = wind['latitude']
lon = wind['longitude']

wind_speed = np.sqrt(u_ref**2 + v_ref**2)

# Plotting
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6))
ax.set_extent([60, 159, 1, 59], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='--')
ax.add_feature(cfeature.LAND, edgecolor='black')

# Create the filled contour plot
contourf_plot = ax.contourf(lon, lat, wind_speed.values, transform=ccrs.PlateCarree(), cmap=plt.cm.RdYlBu_r,
                            levels=range(0,16), extend='max')

# Show every 6th u and v vector using quiver
stride = 6  # Change this value as needed
quiver_plot = ax.quiver(lon[::stride], lat[::stride], u_ref.values[::stride, ::stride],
                        v_ref.values[::stride, ::stride], transform=ccrs.PlateCarree(), scale=200,
                        color='black')

# Add a colorbar for the contourf plot
cbar = plt.colorbar(contourf_plot, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
cbar.set_label('Wind Speed (m/s)')

ax.set_xticks(np.arange(60, 159, 20))  
ax.set_yticks(np.arange(10, 59, 10))  
ax.set_xticklabels(np.arange(60, 159, 20))
ax.set_yticklabels(np.arange(10, 59, 10))

ax.set_title("JRA55 Reanalysis")

plt.savefig(dir_save + 'evaluation_wind_JRA55.pdf', bbox_inches='tight')



#%%-------------
### CMIP WIND
#----------------

file_list_u = os.listdir(udir)
file_list_v = os.listdir(vdir)

file_list_u.sort()
file_list_v.sort()

# Plotting
fig, axs = plt.subplots(nrows=2, ncols=3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(14, 7))
fig.subplots_adjust(hspace=0.04)  # Adjust the vertical spacing between subplots
fig.subplots_adjust(bottom=0.05)
fig.subplots_adjust(wspace=0.1)

axs = axs.flatten()  # Flatten the array of subplots

for i in range(2):
    for j in range(3):
        u_file_path = udir + "/" + file_list_u[i * 3 + j]
        v_file_path = vdir + "/" + file_list_v[i * 3 + j]

        # Load u and v data from NetCDF
        u_data = xr.open_dataset(u_file_path)
        v_data = xr.open_dataset(v_file_path)

        # Extract u, v, lon, and lat data (wind at 850hPa)
        ua = u_data['ua'].isel(time=0, plev=0)
        va = v_data['va'].isel(time=0, plev=0)

        lon = u_data['lon']
        lat = u_data['lat']

        # Calculate wind speed from u and v components
        wind_speed = np.sqrt(ua**2 + va**2)

        if np.isnan(wind_speed.values).any():
            print("Warning: There are NaN values in the wind_speed array. These will be replaced with a specified value.")

        # Extract the subplot axis
        ax = axs[i * 3 + j]  # Use the flattened index

        # Set up the subplot
        ax.set_extent([60, 159, 1, 59], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle='--')
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='gray')  # Set facecolor to green for land

        if wind_speed.isnull().any():
            print("Warning: There are NaN values in the wind_speed array. These will be replaced with a specified value.")

        # Create the filled contour plot
        contourf_plot = ax.contourf(lon, lat, wind_speed.values, transform=ccrs.PlateCarree(), cmap=plt.cm.RdYlBu_r,
                                    levels=range(0,16), extend='max')

        # Show every 3rd u and v vector using quiver
        stride = [4, 4, 4, 4, 4, 4]
        quiver_plot = ax.quiver(lon[::stride[i * 3 + j]], lat[::stride[i * 3 + j]], ua.values[::stride[i * 3 + j], ::stride[i * 3 + j]],
                                va.values[::stride[i * 3 + j], ::stride[i * 3 + j]], transform=ccrs.PlateCarree(), scale=200,
                                color='black')

        # Add a colorbar for the contourf plot
       # cbar = plt.colorbar(contourf_plot, ax=ax, orientation='vertical', pad=0.05)
       # cbar.set_label('Wind Speed (m/s)')

        # Set x-axis and y-axis ticks
        ax.set_xticks(np.arange(60, 159, 20))  # Adjust the range and interval as needed
        ax.set_yticks(np.arange(10, 59, 10))  # Adjust the range and interval as needed

        # Set x-axis and y-axis tick labels
        ax.set_xticklabels(np.arange(60, 159, 20), fontsize=8)
        ax.set_yticklabels(np.arange(10, 59, 10), fontsize=8)

        ax.set_title(file_list_u[i * 3 + j].split("_")[2])

# Create a common colorbar for all subplots
fig.subplots_adjust(bottom=0.2)  # Adjust the bottom margin to make room for the colorbar
cbar_ax = fig.add_axes([0.31, 0.15, 0.4, 0.02])  # Adjust the position and size of the colorbar
cbar = plt.colorbar(contourf_plot, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Wind Speed (m/s)')

plt.savefig(dir_save + 'evaluation_wind.pdf', bbox_inches='tight')
#plt.show()

