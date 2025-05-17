
'''
Robust phenometrics for Australian ecosystems
Works per pixel by looping through spatial coordinates
and parallelizes with dask.delayed.
'''

import os
import sys
import math
import shap
import dask
import scipy
import numpy as np
import xarray as xr
import pandas as pd
import pingouin as pg
from scipy import stats
from datetime import datetime
from scipy.stats import circmean, circstd, false_discovery_control
from scipy.stats import pearsonr, chi2
from collections import namedtuple
from odc.geo.xr import assign_crs
from odc.algo._percentile import xr_quantile

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.robust_parcorr import RobustParCorr

import pymannkendall as mk
from pymannkendall.pymannkendall import __preprocessing, __missing_values_analysis, __mk_score, __variance_s, __z_score, __p_value, sens_slope

def _preprocess(d, dd, ss):
    """
    d = ndvi data
    dd = covariables for regression modelling.
    ss = soil signal xarray dataarray
    
    """
    ### --------Handle NaNs---------
    # Due to issues with xarray quadratic interpolation, we need to remove
    # every NaN or else the daily interpolation function will fail
    
    ## remove last ~6 timesteps that are all-NaN (from S-G smoothing).
    times_to_keep = d.mean(['latitude','longitude']).dropna(dim='time',how='any').time
    d = d.sel(time=times_to_keep)
    
    # Find where NaNs are >10 % of data, will use this mask to remove pixels later.
    # and include any NaNs in the climate data.
    ndvi_nan_mask = np.isnan(d).sum('time') >= len(d.time) / 10
    clim_nan_mask = dd[['rain','vpd','tavg','srad']].to_array().isnull().any('variable')
    clim_nan_mask = (clim_nan_mask.sum('time')>0)
    soil_nan_mask = np.isnan(ss)
    nan_mask = (clim_nan_mask | ndvi_nan_mask | soil_nan_mask)

    d = d.where(~nan_mask)
    dd = dd.where(~nan_mask)
    ss = ss.where(~nan_mask)
    
    #fill the mostly all NaN slices with a fill value
    d = xr.where(nan_mask, -99, d)
    
    #interpolate away any remaining NaNs
    d = d.interpolate_na(dim='time', method='cubic', fill_value="extrapolate")
    
    #now we can finally interpolate to daily
    d = d.resample(time='1D').interpolate(kind='quadratic').astype('float32')
    
    #stack spatial indexes, this makes it easy to loop through data
    y_stack = d.stack(spatial=('latitude', 'longitude'))
    Y = y_stack.transpose('time', 'spatial')
    
    # We also need the shape of the stacked array
    shape = y_stack.values.shape
    
    # find spatial indexes where values are mostly NaN (mostly land-sea mask)
    # This is where the nan_mask we created earlier = True
    idx_all_nan = np.where(nan_mask.stack(spatial=('latitude', 'longitude'))==True)[0]

    return d, dd, ss, Y, idx_all_nan, nan_mask, shape

def _extract_peaks_troughs(da,
                      rolling=90,
                      distance=90,
                      prominence='auto',
                      plateau_size=10
                     ):
    
    """
    Identifying peaks and troughs in a vegetation index time series
    
    The algorithm builds upon those described by Broich et al. (2014).  
    The steps are as follows:
    
    1. Calculate rolling minimums
    2. Calculate rolling maximums
    3. Using scipy.signal.find_peaks, identify peaks and troughs in rolling max
        and min time-series using a minimum distance between peaks, a peak plateau size,
        and the peak/trough prominence.
    4. Remove peaks or troughs where troughs (peaks) occur sequentially without
       a peak (trough) between them. So enforce the pattern peak-valley-peak-valley etc.
       This is achieved by taking the maximum (minimum) peak (trough) where two peaks (troughs)
       occur sequentially.
    
    returns:
    --------
    A pandas.dataframe with peaks and trough identified by the time stamp they occur.
    
    """
    #check its an array
    if not isinstance(da, xr.DataArray):
        raise TypeError(
            "This function only excepts an xr.DataArray"
        )
    # doesn't matter what we call the variable just
    # need it to be predicatable
    da.name = 'NDVI'
    
    #ensure da has only time coordinates
    coords = list(da.coords)
    coords.remove('time')
    da = da.drop_vars(coords).squeeze()

    #calculate rolling min/max to find local minima maxima
    roll_max = da.rolling(time=rolling, min_periods=1, center=True).max()
    roll_min = da.rolling(time=rolling, min_periods=1, center=True).min()
    
    if prominence=='auto':
        #dynamically determine how prominent a peak needs to be
        #based upon typical range of seasonal cycle
        clim = da.groupby('time.month').mean()
        _range = (clim.max() - clim.min()).values.item()
        if _range>=0.05:
            prominence = 0.01
        if _range<0.05:
            prominence = 0.005
    
    #find peaks and valleys
    peaks = scipy.signal.find_peaks(roll_max.data,
                        distance=distance,
                        prominence=prominence,
                        plateau_size=plateau_size)[0]
    
    troughs = scipy.signal.find_peaks(roll_min.data*-1,#invert
                        distance=distance,
                        prominence=prominence,
                        plateau_size=plateau_size)[0]
    
    #--------------cleaning-------
    # Identify where two peaks or two valleys occur one after another and remove.
    # i.e. enforcing the pattern peak-valley-peak-valley etc.
    # First get the peaks and troughs into a dataframe with matching time index
    df = da.to_dataframe()
    df['peaks'] = da.isel(time=peaks).to_dataframe()
    df['troughs'] = da.isel(time=troughs).to_dataframe()
    df_peaks_troughs = df.drop('NDVI', axis=1).dropna(how='all')
    
    #find where two peaks or two troughs occur sequentially
    peaks_num_nans = df_peaks_troughs.peaks.isnull().rolling(2).sum()
    troughs_num_nans = df_peaks_troughs.troughs.isnull().rolling(2).sum()
    
    # Grab the indices where the rolling sum of NaNs equals 2.
    # The labelling is inverted here because two NaNs in the trough column
    # mean two peaks occur concurrently, and vice versa
    idx_consecutive_peaks = troughs_num_nans[troughs_num_nans==2.0]
    idx_consecutive_troughs = peaks_num_nans[peaks_num_nans==2.0]
    
    # Loop through locations with two sequential peaks and drop
    # the smaller of the two peaks
    for idx in idx_consecutive_peaks.index:
        
        loc = df_peaks_troughs.index.get_loc(idx)
        df = df_peaks_troughs.iloc[[loc-1,loc]]
        min_peak_to_drop = df.idxmin(skipna=True).peaks
        df_peaks_troughs = df_peaks_troughs.drop(min_peak_to_drop)
    
    # Loop through locations with two sequential troughs and drop
    # the higher of the two troughs (less prominent trough)
    for idx in idx_consecutive_troughs.index:
        
        loc = df_peaks_troughs.index.get_loc(idx)
        df = df_peaks_troughs.iloc[[loc-1,loc]]
        min_trough_to_drop = df.idxmax(skipna=True).troughs
        
        df_peaks_troughs = df_peaks_troughs.drop(min_trough_to_drop)
    
    return df_peaks_troughs

@dask.delayed
def xr_phenometrics(da,
             rolling=90,
             distance=90,
             prominence='auto',
             plateau_size=10,
             amplitude=0.20,  
             verbose=True,
             soil_signal=0.141
            ):
    """
    Calculate statistics that describe the phenology cycle of
    a vegetation condition time series.
    
    Identifies the start and end points of each cycle using 
    the `seasonal amplitude` method. When the vegetation time series reaches
    20 % of the seasonal amplitude between the first minimum and the peak,
    and the peak and the second minimum.
    
    To ensure we are measuring only full cycles we enforce the time series to
    start and end with a trough.
    
    Phenometrics calculated:
        * ``'SOS'``: DOY of start of season
        * ``'POS'``: DOY of peak of season
        * ``'EOS'``: DOY of end of season
        * ``'TOS'``: DOY of the minimum at the beginning of cycle (left of peak)
        * ``'vSOS'``: Value at start of season
        * ``'vPOS'``: Value at peak of season
        * ``'vEOS'``: Value at end of season
        * ``'vTOS'``: Value at the beginning of cycle (left of peak)
        * ``'LOS'``: Length of season (DOY)
        * ``'LOC'``: Length of cycle (trough to trough) (DOY)
        * ``'AOS'``: Amplitude of season (in value units)
        * ``'IOS'``: Integral of season (in value units, minus soil signal)
        * ``'IOC'``: Integral of cycle (in value units, minus soil signal)
        * ``'ROG'``: Rate of greening (value units per day)
        * ``'ROS'``: Rate of senescence (value units per day)
    
    Returns:
    --------
    Xarray dataset with phenometrics. Xarray is a 1D array.
    
    """
    
    #Extract peaks and troughs in the timeseries
    peaks_troughs = _extract_peaks_troughs(
                         da,
                         rolling=rolling,
                         distance=distance,
                         prominence=prominence,
                         plateau_size=plateau_size)
    
    # start the timeseries with trough
    if np.isnan(peaks_troughs.iloc[0].troughs):
        p_t = peaks_troughs.iloc[1:]
    else:
        p_t = peaks_troughs
    
    # end the timeseries with trough
    if np.isnan(p_t.iloc[-1].troughs)==True:
        p_t = p_t.iloc[0:-1]
        
    # Store phenology stats
    pheno = {}

    if isinstance(soil_signal, float):
        pass

    if isinstance(soil_signal, xr.DataArray):
       soil_signal = soil_signal.values.item()
    
    peaks_only = p_t.peaks.dropna()
    for peaks, idx in zip(peaks_only.index, range(0,len(peaks_only))):
        # First we extract the trough times either side of the peak
        start_time = p_t.iloc[p_t.index.get_loc(peaks)-1].name
        end_time = p_t.iloc[p_t.index.get_loc(peaks)+1].name
    
        # now extract the NDVI time series for the cycle
        ndvi_cycle = da.sel(time=slice(start_time, end_time)).squeeze()

        # add the stats to this dict
        vars = {}
       
        # --Extract phenometrics---------------------------------
        # Peaks
        pos = ndvi_cycle.idxmax(skipna=True)
        vars['POS_year'] = pos.dt.year #so we can keep track
        vars['POS'] = pos.dt.dayofyear.values
        vars['vPOS'] = ndvi_cycle.max().values

        # Troughs
        #we want the trough values from the beginning of the season only (left side)
        vars['TOS_year'] =  p_t.iloc[p_t.index.get_loc(peaks)-1].name.year
        vars['TOS'] = p_t.iloc[p_t.index.get_loc(peaks)-1].name.dayofyear
        vars['vTOS'] = p_t.iloc[p_t.index.get_loc(peaks)-1].troughs
        vars['AOS'] = (vars['vPOS'] - vars['vTOS'])
        
        # SOS  
        # Find the greening cycle (left of the POS)
        greenup = ndvi_cycle.where(ndvi_cycle.time <= pos)
        # Find absolute distance between 20% of the AOS and the values of the greenup, then
        # find the NDVI value that's closest to 20% of AOS, this is our SOS date
        sos = np.abs(greenup - (vars['AOS'] * amplitude + vars['vTOS'])).idxmin(skipna=True)
        vars['SOS_year'] = sos.dt.year #so we can keep track
        vars['SOS'] = sos.dt.dayofyear.values
        vars['vSOS'] = ndvi_cycle.sel(time=sos).values
        
        # EOS
        # Find the senescing cycle (right of the POS)
        browning = ndvi_cycle.where(ndvi_cycle.time >= ndvi_cycle.idxmax(skipna=True))
        # Find absolute distance between 20% of the AOS and the values of the browning, then
        # find the NDVI value that's closest to 20% of AOS, this is our EOS date
        ampl_browning = browning.max() - browning.min()
        eos = np.abs(browning - (ampl_browning * amplitude + browning.min())).idxmin(skipna=True)
        vars['EOS_year'] = eos.dt.year #so we can keep track
        vars['EOS'] = eos.dt.dayofyear.values
        vars['vEOS'] = ndvi_cycle.sel(time=eos).values
    
        # LOS
        los = (pd.to_datetime(eos.values) - pd.to_datetime(sos.values)).days
        vars['LOS'] = los

        # LOC
        loc = (end_time - start_time).days
        vars['LOC'] = loc
    
        #Integral of season
        ios = ndvi_cycle.sel(time=slice(sos, eos))
        ios = ios-soil_signal
        ios = ios.integrate(coord='time', datetime_unit='D')
        vars['IOS'] = ios

        #Integral of cycle
        ioc = ndvi_cycle-soil_signal
        ioc = ioc.integrate(coord='time', datetime_unit='D')
        vars['IOC'] = ioc
    
        # Rate of growth and sensecing (NDVI per day)
        vars['ROG'] = (vars['vPOS'] - vars['vSOS']) / ((pd.to_datetime(pos.values) - pd.to_datetime(sos.values)).days)
        vars['ROS'] = (vars['vEOS'] - vars['vPOS']) / ((pd.to_datetime(eos.values) - pd.to_datetime(pos.values)).days)

        # Ratio between green-up time and senescence time (Meng et al. 2024)
        # vars['VGU/VSS'] = (vars['POS']-vars['SOS'])/(vars['EOS']-vars['POS'])
        
        pheno[idx] = vars

    # convert to xarray
    ds = pd.DataFrame(pheno).astype('float32').transpose().to_xarray()

    # tidy up and add spatial coords.
    ds = ds.astype(np.float32)
    lat = da.latitude.item()
    lon = da.longitude.item()
    ds.assign_coords(latitude=lat, longitude=lon)
    for var in ds.data_vars:
        ds[var] = ds[var].expand_dims(latitude=[lat], longitude = [lon])
    
    return ds

@dask.delayed
def circular_mean_and_stddev(ds):
    
    # count number of seasons
    n_seasons = len(ds.index)
    
    # and how many years was this over?
    a = ds.POS_year.isel(index=0).values.item()
    b = ds.POS_year.isel(index=-1).values.item()
    
    if np.isnan(a):
        n_years=39
    else:
        n_years = len([i for i in range(int(a),int(b)+1)])
    
    #Now check if its nodata and do a quick mean so we return nodata
    if np.isnan(ds.vPOS.isel(index=0).values.item()):
        dd = ds.mean('index')
        
    else: 
        #calculate circular statistics for seasonal timings
        circ_stats = []
        for var in ['SOS', 'POS', 'EOS', 'TOS']:
            
            data = pd.DataFrame({
                "year": ds[f'{var}_year'].squeeze().values,
                "day_of_year": ds[var].squeeze().values
            })
        
            # Number of days in a year (adjusting for leap years)
            data['days_in_year'] = data['year'].apply(lambda y: 366 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 365)
            
            # Convert day-of-year to circular coordinates doy / 365 * 2 * np.pi
            data['theta'] = data['day_of_year']*((2*np.pi)/data['days_in_year'])
            data['theta_unwrap'] = np.unwrap(data['theta'])
            
            # Calculate circular mean and stddev, convert back to DOY
            circular_mean = circmean(data['theta'], nan_policy='omit')
            circular_std = circstd(data['theta'], nan_policy='omit')
            circular_mean_doy = circular_mean / (2 * np.pi) * 365
            circular_std_doy = circular_std / (2 * np.pi) * 365
        
            df = pd.DataFrame({
                f'{var}': [circular_mean_doy],
                f'{var}_std': [circular_std_doy],
                })
        
            circ_stats.append(df)

        # For the other variables use Kirill's quantile function (fast median stats)
        dd_circ = pd.concat(circ_stats, axis=1).to_xarray().squeeze().expand_dims(latitude=ds.latitude,longitude=ds.longitude).drop_vars('index')
        ds = ds.transpose('index', 'latitude', 'longitude')
        other_vars=['vTOS','vSOS','vPOS','vEOS','AOS','LOS','IOS','IOC','LOC','ROG','ROS']
        dd_median = xr_quantile(ds[other_vars], quantiles=[0.5], nodata=np.nan)
        dd_median = dd_median.sel(quantile=0.5).drop_vars('quantile')
        
        dd = xr.merge([dd_median, dd_circ])

    # add new variable with number of seasons
    dd['n_seasons'] = n_seasons
    dd['n_seasons'] = dd['n_seasons'].expand_dims(
        latitude=dd.latitude,
        longitude=dd.longitude
    )
    
    dd['n_years'] = n_years
    dd['n_years'] = dd['n_years'].expand_dims(
        latitude=dd.latitude,
        longitude=dd.longitude
    )
    return dd

def remove_circular_outliers_and_unwrap(
    angles,
    n_sigma=2.0
): 
    """
    Detect outliers in circular data using circular statistics,
    then apply numpy.unwrap() so we can calculate linear trends
    on the resulting dataset.
    
    Parameters:
    -----------
    angles : array-like
        Angular data in radians
    n_sigma : float, optional (default=2.0)
        Number of circular standard deviations to use for outlier detection
        
    Returns:
    --------
    result : Data with outliers replaced with NaNs and np.unwrap applied
    
    """
    data_copy = angles.copy()
    angles = np.asarray(angles)
    
    #circular statistics
    mean_angle = circmean(angles)
    std_angle = circstd(angles)
    
    #circular distances from mean, using complex numbers
    distances = np.abs(np.angle(np.exp(1j * (angles - mean_angle))))
    
    # Identify outliers based on circular std. dev.
    outlier_mask = distances > (n_sigma * std_angle)
    
    # Remove outliers
    clean_data = data_copy[~outlier_mask]
    
    # Apply unwrap to clean data
    unwrapped_clean = np.unwrap(clean_data)
    
    # Create output array filled with nan
    result = np.full_like(data_copy, np.nan, dtype=float)
    
    # Fill in the unwrapped values at non-outlier positions
    result[~outlier_mask] = unwrapped_clean
    
    return result

def mk_with_slopes(x_old, alpha = 0.05):
    """
    This function checks the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert 1987).
    This was modified from the "pymannkendall" library to return fewer statistics which makes
    it a little more robust.
    
    Input:
        x: a vector (list, numpy array or pandas series) data
        alpha: significance level (0.05 default)
    Output:
        p: p-value of the significance test
        slope: Theil-Sen estimator/slope
        intercept: intercept of Kendall-Theil Robust Line
        
    """
    res = namedtuple('Mann_Kendall_Test', ['p','slope','intercept'])
    x, c = __preprocessing(x_old)
    x, n = __missing_values_analysis(x, method = 'skip')
    
    s = __mk_score(x, n)
    var_s = __variance_s(x, n)
    
    z = __z_score(s, var_s)
    p, h, trend = __p_value(z, alpha)
    slope, intercept = sens_slope(x_old)

    return res(p, slope, intercept)

@dask.delayed
def phenology_circular_trends(ds, vars, n_sigma=2):
    """
    Calculate robust statistics over phenology
    time series using Mann-Kendal/Theil-Sen.
    
    Accounting for circular variables like dayofyear by
    converting to radians, unwrapping result, and then
    applying trend analysis.
    
    """
    slopes=[]
    p_values=[]
    for var in vars:
        if any(var in x for x in ['SOS', 'POS', 'EOS', 'TOS']):
            # If variabes are circular, then convert to radians
            data = pd.DataFrame({
                "year": ds[f'{var}_year'].squeeze().values,
                "day_of_year": ds[var].squeeze().values
                })
    
            # Number of days in a year (adjusting for leap years)
            data['days_in_year'] = data['year'].apply(lambda y: 366 if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0) else 365)
            
            # Convert day-of-year to circular coordinates: doy / 365 * 2 * np.pi
            data['theta'] = data['day_of_year']*((2*np.pi)/data['days_in_year'])
            
            # Then unwrap to deal with calendar crossing and do linear trend on unwrapped theta
            #  We will try to remove outliers so we don't introducing discontunties in the time series.
            data['theta_unwrap'] = remove_circular_outliers_and_unwrap(data['theta'], n_sigma=n_sigma)
            p_value, slope, intercept = mk_with_slopes(data['theta_unwrap'])
            slope_doy = slope * 365 / (2 * np.pi)

            # if slope is unreasonably large then the unwrapping has
            # likely failed. Retry with a more aggressive outlier filter
            if np.abs(slope_doy) > 2.5:
                data['theta_unwrap'] = remove_circular_outliers_and_unwrap(data['theta'], n_sigma=1.5)
                p_value, slope, intercept = mk_with_slopes(data['theta_unwrap'])
                slope_doy = slope * 365 / (2 * np.pi)

                #----ignore--------------------------------------------
                # if slope is still unreasonably large then fit a linear-circular
                # regression model of the type y=mu+2atan(Bx) and return the
                # beta coefficient as a substitute for the linear coefficient.
                # Beta is not strictly the same as a linear coefficient but the trend
                # direction will be correct and magnitudes will likley be an underestimate of
                # the true linear coefficient. The linear approach only fails on 1-2% of 
                # pixels across Australia.
                # if np.abs(slope_doy) > 2.5:
                #     x = data.index.values # values from 1-~39
                #     x = x-np.mean(x) #centre around 0
                #     y_obs = data['theta'].where(~np.isnan(data['theta_unwrap']))
                #     weights = np.ones_like(x)  # Equal weights for simplicity
                    
                #     # Fit the LC regression model
                #     optimal_params, objective_value = fit_circular_model(x, y_obs, weights)
                #     fitted_mu, fitted_beta = optimal_params
                #     slope_doy = fitted_beta * 365 / (2 * np.pi)
                    
                #     # Compute p-value for beta
                #     p_value = calculate_beta_significance(x, y_obs, weights, kappa=1, wls_weight=0.5)
                # -------------------------------------------------------
                
                # if it still fails return nans
                if np.abs(slope_doy) > 2.5:
                    slope_doy=np.nan
                    p_value = np.nan
                    
            #create dataframe
            df = pd.DataFrame({
                f'{var}_slope': [slope_doy],
                f'{var}_p_value': [p_value],
                })

            #convert to xarray
            dss = df.to_xarray().squeeze().expand_dims(latitude=ds.latitude,longitude=ds.longitude).drop_vars('index')

            #add variables to lists
            slopes.append(dss[f'{var}_slope'])
            p_values.append(dss[f'{var}_p_value'])
        
        else:
            # for the other variables take the simple robust slope
            # no need to transform into radians
            out = xr.apply_ufunc(mk_with_slopes,
                          ds[var],
                          input_core_dims=[["index"]],
                          output_core_dims=[[],[],[],],
                          vectorize=True)
            
            #grab the slopes and p-value
            p = out[0].rename(var+'_p_value')
            s = out[1].rename(var+'_slope')
            # i = out[2].rename(var+'_intercept')
        
            slopes.append(s)
            p_values.append(p)
            # intercept.append(i)
    
        #merge all the variables
        slopes_xr = xr.merge(slopes)
        p_values_xr = xr.merge(p_values)
        # intercept_xr = xr.merge(intercept)

    #export a dataset
    return xr.merge([slopes_xr,p_values_xr]).astype('float32')


def parcorr_with_circular_vars(data, circular_vars, linear_vars, target_var):
    """
    Calculate partial correlations between variables and a target properly handling
    circular variables by decomposing them into sine and cosine components and 
    using the embedding approach
    
    Parameters:
    data: pd.Dataframe
    circular_vars: List of column names for circular variables
    linear_vars: List of column names for linear variables
    target_var: Name of the target variable
    
    Returns:
    pd.DataFrame: Partial correlations with `target_var`
    
    """
    # Convert circular variables to sine and cosine components
    expanded_data = data.copy()
    
    for var in circular_vars:
        # Convert DOY to radians
        radians = 2 * np.pi * (data[var] - 1) / 366
        
        # Create sine and cosine components
        expanded_data[f"{var}_sin"] = np.sin(radians)
        expanded_data[f"{var}_cos"] = np.cos(radians)
        
        # Drop original circular variable
        expanded_data = expanded_data.drop(columns=[var])
    
    # Create list of all predictor variables
    circular_components = [f"{var}_{comp}" for var in circular_vars 
                           for comp in ['sin', 'cos']]

    all_predictors = circular_components + linear_vars
    expanded_data = expanded_data[[target_var]+all_predictors]
    
    # Calculate partial correlation between all vars
    results_df = pg.pairwise_corr(expanded_data,
            columns=[target_var], padjust='none') #[[target_var],all_predictors]
    
    # # Process final results
    final_results = []
    
    # Process linear variables directly
    for var in linear_vars:
        row = results_df[results_df['Y'] == var].iloc[0]

        final_results.append({
            'Y': var,
            'r': row['r'],
            # 'p': row['p-unc']
        })
    
    # Combine sine and cosine components for circular variables
    for var in circular_vars:
        sin_row = results_df[results_df['Y'] == f"{var}_sin"].iloc[0]
        cos_row = results_df[results_df['Y'] == f"{var}_cos"].iloc[0]
        n = len(expanded_data[f"{var}_cos"])
        
        # Calculate magnitude of correlation, reimplementation of here: 
        # https://pingouin-stats.org/build/html/_modules/pingouin/circular.html#circ_corrcl
        rcs = pearsonr(expanded_data[f"{var}_sin"], expanded_data[f"{var}_cos"])[0]
        rxc = cos_row['r']
        rxs = sin_row['r']
       
        r = np.sqrt((rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1-rcs**2))
       
        # Compute p-value
        # p_value = chi2.sf(n * r**2, 2)
        
        final_results.append({
            'Y': var,
            'r': r,
            # 'p': p_value
        })
    
    return pd.DataFrame(final_results).set_index('Y').transpose().reset_index(drop=False)


@dask.delayed
def IOS_analysis(
    pheno, #pheno data
    template,
    pheno_var='IOS',
    rolling=1,
    linear_vars=['vPOS','vSOS','vEOS'],
    circular_vars=['SOS','POS','EOS']
):  
    """
    Find the partial correlation coefficients between IOS and
    other seasonality metrics. 
    """
    if pheno_var=='IOS':
        linear_vars=linear_vars+['LOS']

    if pheno_var=='IOC':
        linear_vars=linear_vars+['vTOS','LOC']
        
    #check if this is a no-data pixel
    if np.isnan(pheno.vPOS.isel(index=0).values.item()):
        p_corr = template.copy() #use our template    
        p_corr['latitude'] = [pheno.latitude.values.item()] #update coords
        p_corr['longitude'] = [pheno.longitude.values.item()]
        
        if pheno_var=='IOC': #rename the template LOS to LOC
            p_corr = p_corr.rename({'LOS':'LOC'})
            p_corr['vTOS'] = p_corr['vPOS'] #add the extra variable 
    
    else:
        #rolling means to squeeze out IAV (leaving this here as configurable but not smoothing data anymore)
        df = pheno.squeeze().to_dataframe().drop(['latitude','longitude'], axis=1).rolling(rolling).mean().dropna()

        #--------------partial correlation with IOS -----
        p_corr = parcorr_with_circular_vars(df, circular_vars, linear_vars, pheno_var)
    
        # tidy up
        lat = pheno.latitude.item()
        lon = pheno.longitude.item()
        p_corr = p_corr.to_xarray().squeeze().expand_dims(latitude=[lat],longitude=[lon]).drop_vars('index')

    return p_corr


@dask.delayed
def regression_attribution(
    pheno, #pheno data
    X, #covariables,
    template,
    model_type='PLS',
    pheno_var='vPOS',
    rolling=1,
    modelling_vars=['srad','co2','rain','tavg','vpd']
):
    """
    Develop regression models between a phenological
    variable and modelling covariables such as climate data. 

    returns:
    --------
    An xarray dataset with regression coefficients, along with 
    slope, p-values, and r2 values for actual and predicted.
    """
    
    #-------Get phenometrics and covariables in the same frame-------
    #check if this is a no-data pixel
    if np.isnan(pheno.vPOS.isel(index=0).values.item()):
        fi = template.copy() #use our template    
        fi['latitude'] = [pheno.latitude.values.item()] #update coords
        fi['longitude'] = [pheno.longitude.values.item()]

    else:
        pheno_df = pheno.squeeze().to_dataframe()
        lat = pheno.latitude.item()
        lon = pheno.longitude.item()

        if pheno_var=='vPOS':

            # Summarise climate data over greening cycle SOS-POS
            start = [pd.Timestamp(datetime.strptime(f'{int(y)} {int(doy)}', '%Y %j'))
                     for y,doy in zip(pheno_df.SOS_year, pheno_df.SOS)]
            end = [pd.Timestamp(datetime.strptime(f'{int(y)} {int(doy)}', '%Y %j'))
                     for y,doy in zip(pheno_df.POS_year, pheno_df.POS)]

        if pheno_var == 'IOS':
            # Summarise climate data over length of season for IOS (SOS to EOS)
            start = [pd.Timestamp(datetime.strptime(f'{int(y)} {int(doy)}', '%Y %j'))
                     for y,doy in zip(pheno_df.SOS_year, pheno_df.SOS)]
            end = [pd.Timestamp(datetime.strptime(f'{int(y)} {int(doy)}', '%Y %j'))
                   for y,doy in zip(pheno_df.EOS_year, pheno_df.EOS)]

        #iterate through each season and summarise climate data.
        # Use peak-of-year for labelling coordinates
        clim=[]
        for s,e,p in zip(start, end, pheno_df.POS_year):
            c = X.sel(time=slice(s,e)) #select each season
            r = c['rain']
            r = r.sum('time')
            c = c[[x for x in modelling_vars if x != 'rain']]
            c = c.mean('time')
            c['rain'] = r
            c = c.drop_vars('spatial_ref')
            c = c.assign_coords(year=p)
            clim.append(c)

        #join back into a time series xarray
        c = xr.concat(clim, dim='year')
        
        # index of our metric with the years
        pheno_df[pheno_var].index = pheno_df['POS_year'].values
        pheno_df[pheno_var].index.name = 'year'
    
        #get all xarrays into the same dims etc.
        pheno_variable = pheno_df[pheno_var].to_xarray()

        #now add our phenometric to the covars- so we have a neat object to work with
        c[pheno_var] = pheno_variable
        
        #--------------- modelling---------------------------------------------------
        # fit rolling annuals to remove some of the IAV (controllable)
        df = c.rolling(year=rolling, min_periods=rolling).mean().to_dataframe().dropna()

        #fit a model with all vars
        x = df[modelling_vars]
        y = df[pheno_var]        

        if model_type=='ML':
            #fit a RF model with all vars
            rf = RandomForestRegressor(n_estimators=100).fit(x, y)
            
            # use SHAP to extract importance
            explainer = shap.Explainer(rf)
            shap_values = explainer(x)
            
            #get SHAP values into a neat DF
            df_shap = pd.DataFrame(data=shap_values.values,columns=x.columns)
            df_fi = pd.DataFrame(columns=['feature','importance'])
            for col in df_shap.columns:
                importance = df_shap[col].abs().mean()
                df_fi.loc[len(df_fi)] = [col,importance]
    
            # Tidy up into dataset
            fi = df_fi.set_index('feature').to_xarray().expand_dims(latitude=[lat],longitude=[lon])

        if model_type=='PLS':
            lr = PLSRegression().fit(x, y)
            prediction = lr.predict(x)
            r2_all = r2_score(y, prediction)
            
            # Find the robust slope of actual
            result_actual = mk.original_test(y, alpha=0.05)
            p_actual = result_actual.p
            s_actual = result_actual.slope
            i_actual = result_actual.intercept
            
            #calculate slope of predicted variable with all params
            result_prediction = mk.original_test(prediction, alpha=0.05)
            p_prediction = result_prediction.p
            s_prediction = result_prediction.slope
            i_prediction = result_prediction.intercept
        
            #get the PLS coefficients
            fi = pd.Series(dict(zip(list(x.columns), list(lr.coef_.reshape(len(x.columns)))))).to_frame()
            fi = fi.rename({0:'PLS_coefficent'},axis=1)
            fi = fi.reset_index().rename({'index':'feature'},axis=1).set_index('feature')
            
            # create tidy df with all stats
            # fi['phenometric'] = 'vPOS'
            fi['slope_actual'] = s_actual
            fi['slope_modelled'] = s_prediction
            fi['p_actual'] = p_actual
            fi['p_modelled'] = p_prediction
            fi['i_actual'] = i_actual
            fi['i_modelled'] = i_prediction
            fi['r2'] = r2_all
    
            fi = fi.to_xarray().squeeze().expand_dims(latitude=[lat],longitude=[lon])

        if model_type=='PCMCI':
            # for PCMCI we need the whole dataframe with pheno_var.
            # Ensure pheno_var is the first column in the dataframe
            df = df[[pheno_var] + modelling_vars]
            
            # Initialize dataframe object, specify time axis and variable names
            dataframe = pp.DataFrame(df.values, 
                                     datatime=df.index, 
                                     var_names=df.columns)
            
            # # Make solar radiation have no parents.
            # idx_srad = df.columns.get_loc("srad")
            # tau_max=0
            # T, N = df.values.shape
            # link_assumptions_with_absent_links= {idx_srad:{(i, -tau):'' for i in range(N) for tau in range(0, tau_max+1) if not (i==idx_srad and tau==0)}}
            # link_assumptions =  PCMCI.build_link_assumptions(link_assumptions_with_absent_links,
            #                                    n_component_time_series=N,
            #                                    tau_max=tau_max,
            #                                    tau_min=0)
            #initiate PCMCI
            pcmci = PCMCI(
                dataframe=dataframe, 
                cond_ind_test=RobustParCorr(),
                verbosity=0)
            
            #Run PCMCI Plus
            pcmci.verbosity = 0
            results = pcmci.run_pcmciplus(tau_max=0, pc_alpha=0.05,
                                # link_assumptions=link_assumptions
                                         )

            ## because 'pheno_var' is the first column in the dataframe, if we want to extract
            # the MCI values for vPOS we index for the first array in val_matrix
            fi = pd.DataFrame(results['val_matrix'][0]).rename({0:'PCMCI'}, axis=1).reset_index(drop=True).set_index(df.columns)
            fi['p_values'] = pd.DataFrame(results['p_matrix'][0]).rename({0:'p_values'}, axis=1).reset_index(drop=True).set_index(df.columns)
            fi = fi.reset_index().rename({'index':'feature'},axis=1).set_index('feature').drop('vPOS')

            #tidy up
            fi = fi.to_xarray().squeeze().expand_dims(latitude=[lat],longitude=[lon])

        if model_type=='delta_slope':
            lr = PLSRegression().fit(x, y)
            prediction = lr.predict(x)
            r2_all = r2_score(y, prediction)
    
            # Find the robust slope of actual
            result_actual = mk.original_test(y, alpha=0.05) #
            p_actual = result_actual.p
            s_actual = result_actual.slope
            i_actual = result_actual.intercept
            
            #calculate slope of predicted variable with all params
            result_prediction = mk.original_test(prediction, alpha=0.05) #
            p_prediction = result_prediction.p
            s_prediction = result_prediction.slope
            i_prediction = result_prediction.intercept
    
            # now fit a model without a given variable
            # and calculate the slope of the phenometric
            r_delta={}
            s_delta={}
            for v in modelling_vars:
                #set variable of interest as a constant value 
                constant = x[v].iloc[0:1].mean() #average of first 5 years
                xx = x.drop(v, axis=1)
                xx[v] = constant
            
                #model and determine slope
                lrr = PLSRegression().fit(xx, y)
                pred = lrr.predict(xx)
                r2 = r2_score(y, pred)
                
                result_p = mk.original_test(pred, alpha=0.1)
                s_p = result_p.slope
    
                #determine the eucliden distance between
                #modelled slope and actual slope (and r2)
                s_delta[v] = math.dist((s_prediction,), (s_p,))
                r_delta[v] = math.dist((r2_all,), (r2,))
    
            #determine most important feature
            s_delta = pd.Series(s_delta)
            r_delta = pd.Series(r_delta)
            fi = pd.concat([s_delta, r_delta], axis=1).rename({0:'delta_slope', 1:'delta_r2'}, axis=1)
            # fi = fi.loc[[fi['delta_slope'].idxmax()]]
            fi = fi.reset_index().rename({'index':'feature'},axis=1).set_index('feature')
    
            #create tidy df
            fi['slope_actual'] = s_actual
            fi['slope_modelled'] = s_prediction
            fi['p_actual'] = p_actual
            fi['p_modelled'] = p_prediction
            fi['i_actual'] = i_actual
            fi['i_modelled'] = i_prediction
            fi['r2'] = r2_all
            fi = fi.to_xarray().squeeze().expand_dims(latitude=[lat],longitude=[lon])
            
    return fi



# this code for resampling time rather than rolling means
# df['time'] = [datetime.strptime(f'{int(y)} {int(doy)}', '%Y %j') for y,doy in zip(df['POS_year'], df['POS'])]
# df = df.set_index('time').resample('3YE').mean()

# # Summarise climate data over SOS to the POS
        # peaks = [pd.Timestamp(datetime.strptime(f'{int(y)} {int(doy)}', '%Y %j'))
        #          for y,doy in zip(pheno_df.POS_year, pheno_df.POS)]
    
        # #iterate through each POS and summarise climate data.
        # # Use peak-of-year for labelling coordinates
        # clim=[]
        # for p,y in zip(peaks, pheno_df.POS_year):
        #     #subtract months to find the month-range
        #     b = p - pd.DateOffset(months=1)
        #     s,e = b.strftime('%Y-%m'), p.strftime('%Y-%m')
        #     c = X.sel(time=slice(s,e)) #select months before peak
        #     r = c['rain']
        #     r = r.sum('time')
        #     c = c[[x for x in modelling_vars if x != 'rain']]
        #     c = c.mean('time')
        #     c['rain'] = r
        #     c = c.drop_vars('spatial_ref')
        #     c = c.assign_coords(year=y)
        #     clim.append(c)






