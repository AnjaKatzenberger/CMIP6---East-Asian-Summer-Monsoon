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
import cartopy
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec


# Directories
dir = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data'
dir_save = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/figures/'
dir_div = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/pr_bestmodels_ssp585_1995-2014_timmean_sub_div.nc'


dir_change = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/pr_bestmodels_ssp585_2081-2100_1995-2014_change.nc'
dir_change_ssp585 = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/pr_allscenariomodels_ssp585_2081-2100_1995-2014_change.nc'
dir_change_ssp370 = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/pr_allscenariomodels_ssp370_2081-2100_1995-2014_change.nc'
dir_change_ssp245 = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/pr_allscenariomodels_ssp245_2081-2100_1995-2014_change.nc'
dir_change_ssp126 = '/Users/anjakatzenberger/Dokumente/PhD/03_CMIP6_china/data/pr_allscenariomodels_ssp126_2081-2100_1995-2014_change.nc'


#%% 

### Precipitation change between 2081-2100 and 1995-2015 (SSP585) multi-model-mean

country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})

data = nc.Dataset(dir_change)
pr = data['pr'][:]
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
        

cf = axs.contourf(lons,lats,data['pr'][0,:,:]*86400,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),levels = [-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],extend = 'both')
axs.set_title('SSP5-8.5', fontsize = 10, pad = 1)
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

plt.savefig(dir_save + 'Spatial_change_ssp585_mmm.pdf')


#%% 

### Precipitation change between 2081-2100 and 1995-2015 (SSP585) multi-model-mean
# as above but including only the models that are available in all scenarios

country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})

data = nc.Dataset(dir_change_ssp585)
pr = data['pr'][:]
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
        

cf = axs.contourf(lons,lats,data['pr'][0,:,:]*86400,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),levels = [-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],extend = 'both')
axs.set_title('SSP5-8.5', fontsize = 10, pad = 1)
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

plt.savefig(dir_save + 'Spatial_change_ssp585_mmm_allscenariomodels.pdf')


#%% 
### Precipitation change between 2081-2100 and 1995-2015 (SSP370) multi-model-mean
# including only the models that are available in all scenarios


country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})

data = nc.Dataset(dir_change_ssp370)
pr = data['pr'][:]
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
        

cf = axs.contourf(lons,lats,data['pr'][0,:,:]*86400,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),levels = [-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],extend = 'both')
axs.set_title('SSP3-7.0', fontsize = 10, pad = 1)
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

plt.savefig(dir_save + 'Spatial_change_ssp370_mmm_allscenariomodels.pdf')


#%% 
### Precipitation change between 2081-2100 and 1995-2015 (SSP245) multi-model-mean
# including only the models that are available in all scenarios


country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})

data = nc.Dataset(dir_change_ssp245)
pr = data['pr'][:]
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
        

cf = axs.contourf(lons,lats,data['pr'][0,:,:]*86400,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),levels = [-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],extend = 'both')
axs.set_title('SSP2-4.5', fontsize = 10, pad = 1)
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

plt.savefig(dir_save + 'Spatial_change_ssp245_mmm_allscenariomodels.pdf')


#%% 
### Precipitation change between 2081-2100 and 1995-2015 (SSP126) multi-model-mean
# including only the models that are available in all scenarios


country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(1,1,subplot_kw={'projection': ccrs.PlateCarree()})

data = nc.Dataset(dir_change_ssp126)
pr = data['pr'][:]
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
        

cf = axs.contourf(lons,lats,data['pr'][0,:,:]*86400,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),levels = [-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2],extend = 'both')
axs.set_title('SSP1-2.6', fontsize = 10, pad = 1)
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

plt.savefig(dir_save + 'Spatial_change_ssp126_mmm_allscenariomodels.pdf')



#%%

# Figure for models that capture historic rainfall wthin two standard deviations (group A)

good_models =  ['UKESM1-0-LL', 'NorESM2-MM', 'ACCESS-CM2', 'KACE-1-0-G', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'E3SM-1-1', 'EC-Earth3', 'MPI-ESM1-2-LR', 'EC-Earth3-CC', 'GFDL-CM4', 'AWI-CM-1-1-MR', 'MRI-ESM2-0', 'GFDL-ESM4', 'BCC-CSM2-MR']
good_centers = ['MOHC', 'NCC', 'CSIRO-ARCCSS', 'NIMS-KMA', 'CNRM-CERFACS', 'CNRM-CERFACS', 'IPSL', 'E3SM-Project', 'EC-Earth-Consortium', 'MPI-M', 'EC-Earth-Consortium', 'NOAA-GFDL', 'AWI', 'MRI', 'NOAA-GFDL', 'BCC']

country_borders = cfeature.NaturalEarthFeature(category='cultural',name='admin_0_boundary_lines_land',scale='50m',facecolor='none')
plt.style.use('classic')
fig, axs = plt.subplots(4,4,subplot_kw={'projection': ccrs.PlateCarree()} ,gridspec_kw = {'wspace':0.1, 'hspace':-0.4})

index = 0

for i in range(4):
    for j in range(4):
        
        print(good_models[index])
        file = 'pr_' + good_centers[index] + '_' + good_models[index] + '_ssp585_1850-2100_JJA_1850_2100_yearmean_remap_mask_asia_timmean_2081-2100_timmean_2081-2100_diff.nc'
        data = nc.Dataset(dir + '/' + file)
        pr = data['pr'][:]
        lats = data.variables['lat'][:]
        lons = data.variables['lon'][:]

        cf = axs[i,j].contourf(lons,lats,data['pr'][0,:,:]*86400,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),levels = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3],extend = 'both')
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

plt.savefig(dir_save + 'Spatial_change_ssp585_good.pdf')
   
#%%

# Figure for other models (group B)

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
            file = 'pr_' + other_centers[index] + '_' + other_models[index] + '_ssp585_1850-2100_JJA_1850_2100_yearmean_remap_mask_asia_timmean_2081-2100_timmean_2081-2100_diff.nc'
            data = nc.Dataset(dir + '/' + file)
            pr = data['pr'][:]
            lats = data.variables['lat'][:]
            lons = data.variables['lon'][:]
            
    
            cf = axs[i,j].contourf(lons,lats,data['pr'][0,:,:]*86400,cmap = 'RdBu',transform=cartopy.crs.PlateCarree(),levels = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3],extend = 'both')
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

plt.savefig(dir_save + 'Spatial_change_ssp585_others.pdf')
   

