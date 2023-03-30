#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:10:00 2022

@author: anjakatzenberger
"""



###############################################################
#       DATA ANALYSIS - VERY WET MONSOON SEASONS IN INDIA     #
#      RESULTS PUBLISHED IN GEOPHYSICAL RESEARCH LETTERS      #
#                    Code by Anja Katzenberger                #
###############################################################  
#Link to publication: 
#https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022GL098856

# Python Code for producing fig S1, Supplementary Information

#### Other example from which code is derived: 
#https://towardsdatascience.com/what-is-new-in-geopandas-0-70-dda0ddc90978

import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.patches as patches

#### Polygons 
# which polygon has been used for the analysis (bottom left, top left, top right, bottom, right, bottom left)
poly_used = Polygon([(100, 20), (100, 50), (150, 50), (150, 20), (100, 20)])
#poly_show = Polygon([(70, -10), (70, 70), (179, 70), (170, -10), (70, -10)])

### Map
# Get world map
world = gpd.read_file(  
     gpd.datasets.get_path("naturalearth_lowres"))

# Display world map
#world.plot(color="lightgrey")

# Potentially choose countries
#countries[countries["name"] == "India"].plot(color="lightgrey",ax=ax)

# Extract framing map (world map or larger region around area of interest)
clipped = gpd.clip(world, poly_used) #### poly_show here produces map as in area of interest figure
#world.plot(ax=ax,color="lightgrey")

fig, ax = plt.subplots(figsize=(12,10))
ax.margins(0)

clipped.plot(ax=ax, color="lightgrey")
# Create polygon of area of interest
polygon = gpd.GeoDataFrame([1], geometry=[poly_used], crs=world.crs)


dir = '/home/anjaka/EASM/ssp585'
file = 'pr_MOHC_UKESM1-0-LL_ssp585_1850-2100_JJA_1850_2100_yearmean_remap_mask_asia_timmean_selyear.nc'


data = nc.Dataset(dir + '/' + file)
pr = data['pr'][:]
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]


plt.contourf(lons,lats,data['pr'][0,:,:]*86400)
plt.colorbar()

#countries[countries["name"] == "India"].plot(color="lightgrey",ax=ax)
polygon.boundary.plot(ax=ax, color="black")
#plt.ylabel('Latitude')
#plt.xlabel('Longitude')
plt.savefig('/home/anjaka/area_of_interest_EASM.pdf', bbox_inches='tight')
plt.show()
plt.close()
