
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:20:17 2025

@author: venkatesh.panchariya
"""

#This code can be used to get the data from NetCDF files for any area of interest using shape file (.shp). It is prepared taking help from ChatGPT, and online sources while working with .nc files.
#Resources were provided by the University of Trento, Italy. Suggestions to improve this script are welcomed at venkatesh.panchariya@unitn.it

import xarray as xr
import geopandas as gpd
from shapely.geometry import shape, Point
import numpy as np
#import dask.array as da
import fiona
#from datetime import timedelta
#import calendar
import pandas as pd
import os
import matplotlib.pyplot as plt


# %%

# Step 1: Load the shapefile

# Load the catchment boundary shapefile
catchment_shapefile = "path_to_your_shapefile"

#If your geopandas version is beyond 1.0, uncomment following line to read the shapefile. Otherwise follow the below steps using Fiona engine

#catchment_gdf = gpd.read_file(catchment_shapefile)

# Open the shapefile using Fiona
with fiona.open(catchment_shapefile) as src:
    crs = src.crs  # Extract CRS
    data = [
        {**feat["properties"], "geometry": shape(feat["geometry"])}  # Merge properties into the dict
        for feat in src
    ]

# Convert to GeoDataFrame
catchment_gdf = gpd.GeoDataFrame(data, geometry="geometry")
# Manually set CRS (mimics gpd.read_file)
catchment_gdf.set_crs(crs, inplace=True)

# Ensure data types are correctly interpreted
catchment_gdf = catchment_gdf.astype({"fid": float, "DN": int})  # Modify based on actual data

print(catchment_gdf.head())
print(catchment_gdf.columns)


# Plot the shapefile
fig, ax = plt.subplots(figsize=(8, 6))
catchment_gdf.plot(ax=ax, edgecolor="black", facecolor="none")
plt.title("Catchment Shapefile Boundary")
plt.show()

#%% 

#Step2: Load the .nc file

# Load the .nc file to check its CRS
nc_file = "path_to_your_nc_file"

dataset_var = xr.open_dataset(nc_file)

#To see the metadata of .nc file 
print(dataset_var.attrs)
#variable details
print(dataset_var.variables.keys()) 

#Know dimension
if 'variable_name_in_metadata' in dataset_var.variables:
    print(dataset_var.variables['variable_name_in_metadata'][:])  # Details about the dimensions/variables


# Extract the CRS of the NetCDF file. Two alternatives
#1.
nc_crs = dataset_var.attrs.get("crs", "EPSG:32632")  # Default to WGS84 if no CRS is specified
print(nc_crs)

#2.
#nc_crs = dataset['transverse_mercator'].attrs['spatial_ref']
#print(nc_crs)


# Check if the shapefile CRS matches the NetCDF CRS, and reproject it to match the .nc file's CRS
if not catchment_gdf.crs.equals(nc_crs):
    # Reproject the catchment shapefile to match the .nc file's CRS
    catchment_gdf = catchment_gdf.to_crs(nc_crs)
else:
    print("CRS of shapefile matches NetCDF file. Skipping reprojection.")


#%% 

# Step 3: Extract spatial data from the .nc file

# Assume your .nc file has lat/lon coordinates or projected x/y coordinates
lats = dataset_var['y'].values
lons = dataset_var['x'].values

#Assume that you are reading .nc file for Daily Precipitation (mm/day), named as 'precipitation' in metadata 
P = dataset_var['precipitation']  # Replace with the actual variable name

# Create a grid of points corresponding to the .nc file's coordinates
lon_grid, lat_grid = np.meshgrid(lons, lats)


#To get the corner points of the bounding box of area covered in .nc file
print(f"Longitude range: {lon_grid.min()} to {lon_grid.max()}")
print(f"Latitude range: {lat_grid.min()} to {lat_grid.max()}")

#%%
# Step 4: Apply the mask to extract the data for area of interest. Here, mask the .nc data using the catchment boundary.

# Create a mask for the catchment boundary
catchment_mask = np.zeros(lon_grid.shape, dtype=bool)

# Iterate through catchment polygons to create the mask. 
#catchment_mask would be a boolean array. 'True' values corresponds to the points within the catchment area while value corresponds to outside points would be 'False'

for _, row in catchment_gdf.iterrows():
    geom = row['geometry']
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            point = Point(lon_grid[i, j], lat_grid[i, j])
            if geom.contains(point):
                catchment_mask[i, j] = True


## Set bounding box for catchment shape file using catchment_mask: to be used as plot limits in the following loop
# Find indices where mask is True
true_indices = np.argwhere(catchment_mask)  # Get row, col indices of True values

# Get bounding box indices
row_min, col_min = true_indices.min(axis=0)
row_max, col_max = true_indices.max(axis=0)

# Convert indices to real-world coordinates
lon_min, lon_max = lon_grid[row_min, col_min], lon_grid[row_max, col_max]
lat_min, lat_max = lat_grid[row_min, col_min], lat_grid[row_max, col_max]


#%% 
# Step 5: Get the variable values at valid points i.e. for points within the area of interest.

#Based on the computing power of your system (memory) use one of the following five ways

#1.    
P_array = np.where(catchment_mask, P.values, np.nan)  # recommended. On systems with low computing power their would be a 'kernel restart' problem. 

#Uncomment following line: To check the array at any random time-steps (from first time-step onwards. Given that first dimension represents 'time', here 'days')
#P = P_array[180,:,:]

#2.    #masked_variable = variable.where(catchment_mask, np.nan)      # On systems with low computing power their would be a 'kernel restart' problem.

#3.    from scipy.sparse import csr_matrix                                   
       #masked_sparse = csr_matrix(P.values * catchment_mask)  #      

#4.    masked_variable = np.empty(P.shape)  # Preallocate the output array     # On systems with low computing power their would be a 'kernel restart' problem.
       #masked_variable.fill(np.nan)  # Initialize with NaNs

       #chunk_size = 500

        #for i in range(0, P.shape[0], chunk_size):
            #for j in range(0, P.shape[1], chunk_size):
                #sub_mask = catchment_mask[i:i+chunk_size, j:j+chunk_size]
                #sub_variable = P.values[i:i+chunk_size, j:j+chunk_size]
                #masked_variable[i:i+chunk_size, j:j+chunk_size] = np.where(sub_mask, sub_variable, np.nan)
                
#5. Use of dask library. Recommended for parallel processing of data in suitable chunk size. On systems with low computing power may encounter a 'kernel restart' problem.
    #chunk_size = 20

    #Convert arrays into dask arrays
    #variable_dask = da.from_array(P.values, chunks = (chunk_size, chunk_size))
    #catchment_mask_dask = da.from_array(catchment_mask, chunks = (chunk_size, chunk_size))

#Check the dimensions of masked array (2D) and extracted variable (should be 3D here)
print("catchment_mask shape:", catchment_mask.shape)
print("P_array shape:", P_array.shape)


#%%
# Step 6: Process data for points within the catchment area and visualize the spatial distribution of variable of interest for each month. 
#Below example of for loop is to get the area averaged monthly values for variable of interest

# Assuming the variable extracted from mask-array is an xarray.DataArray with dimensions (time, lat, lon)

# Get unique months. Here assume first slice of data in your .nc file corresponds to 1st January 1980
start_day = '1980-01-01'
time_index = pd.date_range(start = start_day, periods=P_array.shape[0], freq ='D')  #freq = 'D' is for daily values
unique_months = time_index.to_period("M").unique() #'M' corresponds to 'month' in index column

#Initiate an empty array to store area-averaged (averaged over all valid points) monthly values for variable of interest (here catchment averaged monthly precipitation, mm/month)

monthly_avg_results = []

#Following loop would aggregate and process data with respect to each month in sequence

for month in unique_months:
    # Select data for the current month
    month_mask = time_index.to_period("M") == month  #returns boolean array: being True for the days in current month 
    Ps_in_month = P_array[month_mask,:,:]  #3D array corresponds to values for current month
        
    # Compute cumulative sum **only within the month**
    cumulative_variable = np.nansum(Ps_in_month, axis=0)     
    
    #** To get the areal average. Returns a scalar (single value) **
    
    #Mean operation using mask-array: outside values are being ignored(as cumulative sum over nan gives zero values). It considers inside zero values if any. 
    #check the length of 1D array after the mask (i.e. length of 'cumulative_variable[catchment_mask]') and count of True values inside catchment_mask (Tc = np.count_nonzero(catchment_mask)). It should match.
    regional_avg = np.nanmean(cumulative_variable[catchment_mask]) 
    
    
    # **Plot Spatial Distribution**
    
    # Apply catchment mask. 
    cumulative_variable_masked = np.where(catchment_mask, cumulative_variable, np.nan)
    
    # **Plot Spatial Distribution with Latitude & Longitude**
    plt.figure(figsize=(10, 8))
    #plt.figure(figsize=(8, 6))
    
    # Use pcolormesh with actual lat/lon values
    plt.pcolormesh(lon_grid, lat_grid, cumulative_variable_masked, cmap='viridis', shading='auto')
    #plt.pcolormesh(cumulative_variable_masked, cmap='viridis', shading='auto')  # Use `shading='auto'` for better display
    plt.colorbar(label="Cumulative Value")  # Colorbar for scale
    
    # Improve scaling using real-world coordinates
    plt.xlim(lon_min, lon_max)
    plt.ylim(lat_min, lat_max)
   
    plt.title(f"Spatial Distribution for {month.strftime('%Y-%m')}")
    plt.xlabel("Longitude Grid Index")
    plt.ylabel("Latitude Grid Index")
    #plt.savefig(f"spatial_distribution_{month.strftime('%Y-%m')}.png")  # Save plot
    plt.show()
    plt.close('all')
    
    # Store result
    monthly_avg_results.append([month.strftime('%Y-%m'), regional_avg])


#%% 
#Step 7: Store processed data into .csv file

# Combine results into a new xarray DataArray
df_monthly_avg = pd.DataFrame(monthly_avg_results, columns=['Month','Catchment Average (mm/month)'])
# Display result
print(df_monthly_avg)

output_path = 'output_directory'
output_file = 'output_file_name.csv'
save = os.path.join(output_path, output_file)
df_monthly_avg.to_csv(save, index=False)
print('Area-averaged monthly values have saved into .csv')




