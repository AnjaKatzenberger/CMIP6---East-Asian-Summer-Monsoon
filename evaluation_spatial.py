#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####-----------------------------------###
### EAST ASIAN SUMMER MONSOON IN CMIP6 ###
####-----------------------------------###
# Anja Katzenberger, anja.katzenberger@pik-potsdam.de

import cartopy
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs


#%%----------------------------
### Directories
#----------------------------

# Directory with data
dir = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data'
# Directory with reanalysis data
dir_reanalysis = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/pr_W5E5v2.0_1995-2014_JJA_yearmean_remap_mask_asia.nc'
# Directory of multi model mean
dir_div = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/pr_bestmodels_ssp585_1995-2014_timmean_sub_div.nc'

# Directory where figures will be saved
dir_save = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/figures/'

#%%----------------------------
### Precipitation from reanalysis data (W5E5) for 1995-2014
#----------------------------

# get national borders
country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')

# Get data
data = nc.Dataset(dir_reanalysis)
pr = data['pr'][:]
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
        
# Plot 
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
cf = axs.contourf(lons,lats,data['pr'][0,:,:]*86400,cmap = 'Blues',transform=cartopy.crs.PlateCarree(),levels = [0,1,2,3,4,5,6,7,8,9,10,11,12],extend = 'max')
#axs.set_title('W5E5 Reanalysis Data', fontsize = 10, pad = 1)
axs.coastlines('10m')
axs.add_feature(country_borders, edgecolor='black')
xticks = np.linspace(100, 140, 5)
axs.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
axs.set_xticklabels(xticks,fontsize=7)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axs.xaxis.set_major_formatter(lon_formatter)
yticks = np.linspace(25, 45, 3)
axs.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
axs.set_yticklabels(yticks,fontsize=7)
lat_formatter = LatitudeFormatter()
axs.yaxis.set_major_formatter(lat_formatter)
cbar = fig.colorbar(cf,fraction=0.028, pad=0.04)
cbar.set_label('JJA mean rainfall (mm/day)', rotation=270,labelpad=15)
plt.savefig(dir_save + 'Spatial_1995_2014_reanalysis.pdf')




#%%----------------------------
### Difference between CMIP6 Multi-model-mean and Reanalysis data (W5E5) for 1995-2014
#----------------------------

# Get borders
country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')

# Get data
data = nc.Dataset(dir_div)
pr = data['pr'][:]
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]

# Create list with levels for plot
r = [-1.1,-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]

plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})
cf = axs.contourf(lons,lats,data['pr'][0,:,:]*100,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),levels = np.array(r)*100)
#axs.set_title('W5E5 Reanalysis Data', fontsize = 10, pad = 1)
axs.coastlines('10m')
axs.add_feature(country_borders, edgecolor='black')
xticks = np.linspace(100, 140, 5)
axs.set_xticks(xticks, crs=cartopy.crs.PlateCarree())
axs.set_xticklabels(xticks,fontsize=7)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
axs.xaxis.set_major_formatter(lon_formatter)
yticks = np.linspace(25, 45, 3)
axs.set_yticks(yticks, crs=cartopy.crs.PlateCarree())
axs.set_yticklabels(yticks,fontsize=7)
lat_formatter = LatitudeFormatter()
axs.yaxis.set_major_formatter(lat_formatter)
cbar = fig.colorbar(cf,fraction=0.028, pad=0.04)
cbar.set_label('Difference [%]', rotation=270,labelpad=15)

plt.savefig(dir_save + 'Spatial_1995_2014_diff.pdf')
   


#%%----------------------------
### Individual models that capture historic rainfall wthin two standard deviations (group A)
#----------------------------

# List of 'good models'
good_models =  ['UKESM1-0-LL', 'NorESM2-MM', 'ACCESS-CM2', 'KACE-1-0-G', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'E3SM-1-1', 'EC-Earth3', 'MPI-ESM1-2-LR', 'EC-Earth3-CC', 'GFDL-CM4', 'AWI-CM-1-1-MR', 'MRI-ESM2-0', 'GFDL-ESM4', 'BCC-CSM2-MR']
good_centers = ['MOHC', 'NCC', 'CSIRO-ARCCSS', 'NIMS-KMA', 'CNRM-CERFACS', 'CNRM-CERFACS', 'IPSL', 'E3SM-Project', 'EC-Earth-Consortium', 'MPI-M', 'EC-Earth-Consortium', 'NOAA-GFDL', 'AWI', 'MRI', 'NOAA-GFDL', 'BCC']

# Get borders
country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')

# Plot
plt.style.use('classic')
fig, axs = plt.subplots(4,4,subplot_kw={'projection': ccrs.PlateCarree()} ,gridspec_kw = {'wspace':0.1, 'hspace':-0.4})

index = 0

for i in range(4):
    for j in range(4):
        
        print(good_models[index])
        file = 'pr_' + good_centers[index] + '_' + good_models[index] + '_ssp585_1850-2100_JJA_1850_2100_yearmean_remap_mask_asia_timmean_selyear.nc'
        data = nc.Dataset(dir + '/' + file)
        pr = data['pr'][:]
        lats = data.variables['lat'][:]
        lons = data.variables['lon'][:]
        

        cf = axs[i,j].contourf(lons,lats,data['pr'][0,:,:]*86400,cmap = 'Blues',transform=cartopy.crs.PlateCarree(),levels = [0,1,2,3,4,5,6,7,8,9,10,11,12],extend = 'max')
        axs[i,j].set_title(good_models[index], fontsize = 10, pad = 1)
        axs[i,j].coastlines('10m')
        axs[i,j].add_feature(country_borders, edgecolor='black')

        if i == 3: 
            xticks = np.linspace(100, 140, 3)
            axs[i,j].set_xticks(xticks, crs=cartopy.crs.PlateCarree())
            axs[i,j].set_xticklabels(xticks,fontsize=5)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            axs[i,j].xaxis.set_major_formatter(lon_formatter)
            
        if j == 0: 
            yticks = np.linspace(25, 45, 3)
            axs[i,j].set_yticks(yticks, crs=cartopy.crs.PlateCarree())
            axs[i,j].set_yticklabels(yticks,fontsize=5)
            lat_formatter = LatitudeFormatter()
            axs[i,j].yaxis.set_major_formatter(lat_formatter)
            
        index = index + 1
            
cbar = fig.colorbar(cf, ax=axs.ravel().tolist(),fraction=0.031, pad=0.04)
cbar.set_label('JJA mean rainfall (mm/day)', rotation=270)

plt.savefig(dir_save + 'Spatial_1995_2014.pdf')




#%%----------------------------
### Other individual models that do not capture historic rainfall wthin two standard deviations (group B)
#----------------------------

# List of 'other' models
other_models =  ['INM-CM4-8', 'INM-CM5-0', 'MIROC-ES2L', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'FIO-ESM-2-0', 'CESM2', 'CESM2-WACCM', 'TaiESM1', 'MIROC6', 'ACCESS-ESM1-5', 'CanESM5', 'CanESM5-CanOE', 'NESM3', 'FGOALS-f3-L', 'IITM-ESM', 'FGOALS-g3', 'CAMS-CSM1-0']
other_centers = ['INM', 'INM', 'MIROC', 'CMCC', 'CMCC', 'FIO-QLNM', 'NCAR', 'NCAR', 'AS-RCEC', 'MIROC', 'CSIRO', 'CCCma', 'CCCma', 'NUIST', 'CAS', 'CCCR-IITM', 'CAS', 'CAMS']

country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(5,4,subplot_kw={'projection': ccrs.PlateCarree()} ,gridspec_kw = {'wspace':0.02, 'hspace':0.25})
index = 0
for i in range(5):
    for j in range(4):
        if index != 18 and index!=19: 
            print(other_models[index])
            file = 'pr_' + other_centers[index] + '_' + other_models[index] + '_ssp585_1850-2100_JJA_1850_2100_yearmean_remap_mask_asia_timmean_selyear.nc'
            data = nc.Dataset(dir + '/' + file)
            pr = data['pr'][:]
            lats = data.variables['lat'][:]
            lons = data.variables['lon'][:]
            
    
            cf = axs[i,j].contourf(lons,lats,data['pr'][0,:,:]*86400,cmap = 'Blues',transform=cartopy.crs.PlateCarree(),levels = [0,1,2,3,4,5,6,7,8,9,10,11,12],extend = 'max')
            axs[i,j].set_title(other_models[index], fontsize = 10, pad = 1)
            axs[i,j].coastlines('10m')
            axs[i,j].add_feature(country_borders, edgecolor='black')
    
            if i == 4: 
                xticks = np.linspace(100, 140, 3)
                axs[i,j].set_xticks(xticks, crs=cartopy.crs.PlateCarree())
                axs[i,j].set_xticklabels(xticks,fontsize=5)
                lon_formatter = LongitudeFormatter(zero_direction_label=True)
                axs[i,j].xaxis.set_major_formatter(lon_formatter)
                
            if j == 0: 
                yticks = np.linspace(25, 45, 3)
                axs[i,j].set_yticks(yticks, crs=cartopy.crs.PlateCarree())
                axs[i,j].set_yticklabels(yticks,fontsize=5)
                lat_formatter = LatitudeFormatter()
                axs[i,j].yaxis.set_major_formatter(lat_formatter)
            
        if index == 18 or index == 19:
            axs[i,j].set_visible(False)
            
        index = index + 1
    
cbar = fig.colorbar(cf, ax=axs.ravel().tolist(),fraction=0.031, pad=0.04)
cbar.set_label('JJA mean rainfall (mm/day)', rotation=270)
plt.savefig(dir_save + 'Spatial_1995_2014_others.pdf')
   


