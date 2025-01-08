#!/usr/bin/env python
# coding: utf-8

# Per pixel phenology modelling for Australia.

import os
import sys
import dask
import scipy
import warnings
import dask.array
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from dask import delayed
from odc.geo.xr import assign_crs

import sys
sys.path.append('/g/data/os22/chad_tmp/Aus_phenology/src')
from phenology_pixel_circular import _preprocess, xr_phenometrics, phenology_circular_trends, circular_mean_and_stddev, regression_attribution, IOS_analysis
sys.path.append('/g/data/os22/chad_tmp/AusEFlux/src/')
from _utils import start_local_dask
sys.path.append('/g/data/os22/chad_tmp/AusEFlux/src/')
from _utils import round_coords

## varibles for script
n_workers=102
memory_limit='450GiB'
integral_var = 'IOS'
regress_var = 'vPOS'
modelling_vars=['co2', 'srad', 'rain', 'tavg', 'vpd']
results_path = '/g/data/os22/chad_tmp/Aus_phenology/results/combined_tiles/'
template_path='/g/data/os22/chad_tmp/Aus_phenology/data/templates/'

n = os.getenv('TILENAME')

#define meta function
def phenometrics_etal(
    n,
    results_path,
    integral_var,
    regress_var,
    template_path,
    modelling_vars,
):
    print('Working on tile', n)

    #open data
    d_path = f'/g/data/os22/chad_tmp/Aus_phenology/data/tiled_data/NDVI_{n}.nc'
    dd_path = f'/g/data/os22/chad_tmp/Aus_phenology/data/tiled_data/COVARS_{n}.nc'
    ss_path = f'/g/data/os22/chad_tmp/Aus_phenology/data/tiled_data/SS_{n}.nc'
    # ss_path = f'/g/data/xc0/project/AusEFlux/data/ndvi_of_baresoil_5km.nc'
    d = assign_crs(xr.open_dataarray(d_path), crs='epsg:4326')
    dd = assign_crs(xr.open_dataset(dd_path), crs='epsg:4326')
    ss = assign_crs(xr.open_dataset(ss_path)['NDVI'], crs='epsg:4326')
    ss.name = 'NDVI'
    
    # transform the data and return all the objects we need. This code smooth and
    # interpolates the data, then stacks the pixels into a spatial index
    d, dd, ss, Y, idx_all_nan, nan_mask, shape = _preprocess(d, dd, ss)
    
    # Open templates array which we'll use whenever we encounter an all-NaN index
    # This speeds up the analysis by not running pixels that are empty.
    # Created the template using one of the output results
    # bb = xr.full_like(results[0], fill_value=np.nan, dtype='float32')
    template_path='/g/data/os22/chad_tmp/Aus_phenology/data/templates/'
    phen_template = xr.open_dataset(f'{template_path}template.nc')
    ios_template = xr.open_dataset(f'{template_path}template_integral_parcorr.nc')

    #now we start the real proceessing
    results=[]
    for i in range(shape[1]): #loop through all spatial indexes.
    
        #select pixel
        data = Y.isel(spatial=i)
        
        # First, check if spatial index has data. If its one of 
        # the all-NaN indexes then return xarray filled with -99 values
        if i in idx_all_nan:
            xx = phen_template.copy() #use our template    
            xx['latitude'] = [data.latitude.values.item()] #update coords
            xx['longitude'] = [data.longitude.values.item()]
        
        else:
            #run the phenometrics
            xx = xr_phenometrics(data,
                              rolling=90,
                              distance=90,
                              prominence='auto',
                              plateau_size=10,
                              amplitude=0.20,
                              soil_signal = ss.sel(
                                 latitude=data.latitude.values.item(),
                                 longitude=data.longitude.values.item()
                                    )
                             )
    
        #append results, either data or all-zeros
        results.append(xx)
    
    # bring into memory.
    results = dask.compute(results)[0]
        
    ### ----Summarise phenology with a median------------
    if os.path.exists(f'{results_path}mean_phenology_perpixel_{n}.nc'):
        pass
    else:
        p_average = [circular_mean_and_stddev(x) for x in results]
        p_average = dask.compute(p_average)[0]
        p_average = xr.combine_by_coords(p_average)
        
        #remove NaN areas that have a fill value
        p_average = p_average.where(p_average>-99).astype('float32')
        p_average = p_average.where(~np.isnan(p_average.vPOS)) #and again for the n_seasons layer
        p_average = assign_crs(p_average, crs='EPSG:4326') # add geobox
    
        #export results
        p_average.to_netcdf(f'{results_path}mean_phenology_perpixel_{n}.nc')
    
    ## ----Find the trends in phenology--------------
    if os.path.exists(f'{results_path}trends_phenology_perpixel_{n}.nc'):
        pass
    else:
    
        trend_vars = ['POS','vPOS','TOS','vTOS','AOS','SOS','vSOS','EOS',
                      'vEOS','LOS','LOC', 'IOS','IOC','ROG','ROS'
                      ]
        p_trends = [phenology_circular_trends(x, trend_vars) for x in results]
        p_trends = dask.compute(p_trends)[0]
        p_trends = xr.combine_by_coords(p_trends)
        
        #remove NaNs
        p_trends = p_trends.where(~np.isnan(p_average.vPOS)).astype('float32')
        
        # assign crs and export
        p_trends = assign_crs(p_trends, crs='EPSG:4326')
        p_trends.to_netcdf(f'{results_path}trends_phenology_perpixel_{n}.nc')
    
    # ----Partial correlation analysis etc. on IOS -----------------
    if os.path.exists(f'{results_path}{integral_var}_analysis_perpixel_{n}.nc'):
        pass
    else:
        p_parcorr = []
        for pheno in results:            
            corr = IOS_analysis(pheno, ios_template, pheno_var=integral_var)
            p_parcorr.append(corr)
        
        p_parcorr = dask.compute(p_parcorr)[0]
        p_parcorr = xr.combine_by_coords(p_parcorr).astype('float32')
        p_parcorr.to_netcdf(f'{results_path}{integral_var}_analysis_perpixel_{n}.nc')
    
    # # -----regression attribution iterate-------------------------------
    # for model_type in ['delta_slope', 'PLS', 'PCMCI', 'ML']:
        
    #     if os.path.exists(f'{results_path}attribution_{regress_var}_{model_type}_perpixel_{n}.nc'):
    #         pass
    #     else:
            
    #         regress_template = xr.open_dataset(f'{template_path}template_{model_type}.nc').sel(feature=modelling_vars)
        
    #         p_attribution = []    
    #         for pheno in results:
    #             lat = pheno.latitude.item()
    #             lon = pheno.longitude.item()
                
    #             fi = regression_attribution(pheno,
    #                                X=dd.sel(latitude=lat, longitude=lon),
    #                                template=regress_template,
    #                                model_type=model_type,
    #                                pheno_var=regress_var,
    #                                modelling_vars=modelling_vars,
    #                               )
    #             p_attribution.append(fi)
            
    #         p_attribution = dask.compute(p_attribution)[0]
    #         p_attribution = xr.combine_by_coords(p_attribution).astype('float32')
    #         p_attribution.to_netcdf(f'{results_path}attribution_{regress_var}_{model_type}_perpixel_{n}.nc')

#run function
if __name__ == '__main__':
    #start a dask client
    start_local_dask(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=memory_limit
                    )

    # Run meta function
    phenometrics_etal(
        n=n,
        results_path=results_path,
        template_path=template_path,
        integral_var=integral_var,
        regress_var=regress_var,
        modelling_vars=modelling_vars,
    )

