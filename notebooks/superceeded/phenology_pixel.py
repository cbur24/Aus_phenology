
'''
Robust phenometrics for Australian ecosystems
Works per pixel by looping through spatial coordinates
and parallelizes with dask.delayed.
'''

import os
import sys
import math
import shap
import scipy
import numpy as np
import xarray as xr
import pandas as pd
import pingouin as pg
from scipy import stats
from datetime import datetime
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

import dask


def months_filter(month, start, end):
    return (month >= start) & (month <= end)

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
    20% of the seasonal amplitude between the first minimum and the peak,
    and the peak and the second minimum.
    
    To ensure we are measuring only full cycles we enforce the time series to
    start and end with a trough.
    
    Phenometrics calculated:
        * ``'SOS'``: DOY of start of season
        * ``'POS'``: DOY of peak of season
        * ``'EOS'``: DOY of end of season
        * ``'vSOS'``: Value at start of season
        * ``'vPOS'``: Value at peak of season
        * ``'vEOS'``: Value at end of season
        * ``'TOS'``: DOY of the minimum at the beginning of cycle (left of peak)
        * ``'vTOS'``: Value at the beginning of cycle (left of peak)
        * ``'LOS'``: Length of season (DOY)
        * ``'LOC'``: Length of cycle (trough to trough) (DOY)
        * ``'AOS'``: Amplitude of season (in value units)
        * ``'IOS'``: Integral of season (in value units, minus soil signal)
        * ``'IOC'``: Integral of cycle (in value units, minus soil signal)
        * ``'ROG'``: Rate of greening (value units per day)
        * ``'ROS'``: Rate of senescence (value units per day)
        * ``'LOS*vPOS'```: Product of LOS times vPOS, approximates IOS
        * ``'IOS/(LOS*vPOS)'``: Ratio of IOS to the product of LOS*vPOS
    
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

        #----some higher order stats------
        # Multiple of LOS*vPOS (estimates IOS)
        vars['LOS*vPOS'] = vars['LOS'] * vars['vPOS']

        # Ratio of IOS to the LOS*vPOS
        vars['IOS:(LOS*vPOS)'] = vars['IOS'] / vars['LOS*vPOS']

        # Multiple of LOC*vPOS (estimates IOC)
        vars['LOC*vPOS'] = vars['LOC'] * vars['vPOS']

        # Ratio of IOS to the LOS*vPOS
        vars['IOC:(LOC*vPOS)'] = vars['IOC'] / vars['LOC*vPOS']
        
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
def phenology_trends(ds, vars):
    """
    Calculate robust statistics over phenology
    time series using MannKendal/Theil-Sen.
    """
    slopes=[]
    p_values=[]
    intercept=[]
    for var in vars:

        #apply mankendall over 'index' dimension
        #this return three variables 
        out = xr.apply_ufunc(mk_with_slopes,
                      ds[var],
                      input_core_dims=[["index"]],
                      output_core_dims=[[],[],[],],
                      vectorize=True)
        
        #grab the slopes and p-value
        p = out[0].rename(var+'_p_value')
        s = out[1].rename(var+'_slope')
        i = out[2].rename(var+'_intercept')
        
        slopes.append(s)
        p_values.append(p)
        intercept.append(i)

    #merge all the variables
    slopes_xr = xr.merge(slopes)
    p_values_xr = xr.merge(p_values)
    intercept_xr = xr.merge(intercept)

    #export a dataset
    return xr.merge([slopes_xr,p_values_xr, intercept_xr]).astype('float32')

@dask.delayed
def _mean(ds):
    # count number of seasons
    # and how many years was this over?
    n_seasons = len(ds.index)
    
    a = ds.POS_year.isel(index=0).values.item()
    b = ds.POS_year.isel(index=-1).values.item()
    
    if np.isnan(a):
        n_years=39
    else:
        n_years = len([i for i in range(int(a),int(b)+1)])
    
    #first check if its nodata and do a quick mean
    # if ds.vPOS.isel(index=0).values.item() == -99.0:
    if np.isnan(ds.vPOS.isel(index=0).values.item()):
        dd = ds.mean('index')
        
    else: # use Kirill's much much faster quantile function
        ds = ds.transpose('index', 'latitude', 'longitude')
        dd = xr_quantile(ds, quantiles=[0.5], nodata=np.nan)
        dd = dd.sel(quantile=0.5).drop_vars('quantile')

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

@dask.delayed
def IOS_analysis(
    pheno, #pheno data
    template,
    pheno_var='IOS',
    rolling=1,
    modelling_vars=['vPOS','vSOS','vEOS','SOS','POS','EOS']
):  
    """
    Find the partial correlation coefficients between IOS(C) and
    other seasonality metrics. 
    """
    if pheno_var=='IOS':
        modelling_vars=modelling_vars+['LOS']

    if pheno_var=='IOC':
        modelling_vars=modelling_vars+['vTOS','LOC']
        
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

        #--------------partial correlation with IOC -----
        p_corr = pg.pairwise_corr(df,
                         columns=[[pheno_var], modelling_vars]) 
            
        p_corr = p_corr[['Y','r']].set_index('Y').transpose().reset_index(drop=True)
        
        # --------Slope and pearson R between IOC and LOS*vPOS ---------
        # result = stats.linregress(df['LOS*vPOS'], df[pheno_var])
        # slope = result.slope
        # pearson_r = result.rvalue
        # p_corr[f'slope_{pheno_var}_vs_LOS*vPOS'] = slope
        # p_corr[f'pearson_r_{pheno_var}_vs_LOS*vPOS'] = pearson_r

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
    rolling=5,
    modelling_vars=['srad','co2','rain','tavg','vpd']
):
    """
    Develop regression models or between a phenological
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

        #now add our phenometric to the covars-we have a neat object to work with
        c[pheno_var] = pheno_variable
        
        #--------------- modelling---------------------------------------------------
        # fit rolling annuals to remove some of the IAV (only interested in trends)
        df = c.rolling(year=rolling, min_periods=rolling).mean().to_dataframe().dropna()

        #fit a model with all vars
        x = df[modelling_vars]
        y = df[pheno_var]        

        # print(x)
        # print(y)
        
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

# @dask.delayed
# def pls_phenology_modelling(data, #NDVI data
#              X, #covariables             
#              rolling=90,
#              distance=90,
#              plateau_size=10,
#              prominence='auto', 
#              verbose=True
#         ):
#     #-----Calculate vPOS and POS--------------
#     #Extract peaks and troughs in the timeseries
#     peaks_troughs = _extract_peaks_troughs(
#                          data,
#                          rolling=rolling,
#                          distance=distance,
#                          prominence=prominence,
#                          plateau_size=plateau_size
#                             )
    
#     # start the timeseries with trough
#     if np.isnan(peaks_troughs.iloc[0].troughs):
#         p_t = peaks_troughs.iloc[1:]
#     else:
#         p_t = peaks_troughs
    
#     # end the timeseries with trough
#     if np.isnan(p_t.iloc[-1].troughs)==True:
#         p_t = p_t.iloc[0:-1]
        
#     # Store phenology stats
#     pheno = {}
    
#     peaks_only = p_t.peaks.dropna()
#     for peaks, idx in zip(peaks_only.index, range(0,len(peaks_only))):
#         # First we extract the trough times either side of the peak
#         start_time = p_t.iloc[p_t.index.get_loc(peaks)-1].name
#         end_time = p_t.iloc[p_t.index.get_loc(peaks)+1].name
    
#         # now extract the NDVI time series for the cycle
#         ndvi_cycle = data.sel(time=slice(start_time, end_time))
        
#         # add the stats to this
#         vars = {}
       
#         # --Extract phenometrics---------------------------------
#         pos = ndvi_cycle.idxmax(skipna=True)
#         vars['POS_year'] = pos.dt.year #so we can keep track
#         vars['POS'] = pos.dt.dayofyear.values
#         vars['vPOS'] = ndvi_cycle.max().values
#         pheno[idx] = vars

#     pheno = pd.DataFrame(pheno).astype('float32').transpose()  
    
#     #-------Get phenometrics and covariables in the same frame-------
#     def months_filter(month, start, end):
#         return (month >= start) & (month <= end)

#     #find average time for POS - just use a random year
#     mean_pos = pd.Timestamp(datetime.strptime(f'{2000} {int(pheno.POS.mean())}', '%Y %j'))

#     #subtract 2 months to find the month-range for summarising climate
#     months_before = mean_pos - pd.DateOffset(months=1)
#     m_r = months_before.month, mean_pos.month

#     #now we meed to index the covariable data by the range of months
#     trimmed_climate = X.sel(time=months_filter(X.time.dt.month, m_r[0], m_r[-1]))
    
#     #calculate annual climate summary stats
#     rain=trimmed_climate['rain']
#     rain=rain.groupby('time.year').sum()
#     trimmed_climate = trimmed_climate[['co2', 'srad', 'tavg', 'vpd']]
#     annual_climate = trimmed_climate.groupby('time.year').mean()
#     annual_climate = annual_climate.sel(year=pheno['POS_year'].values)
#     annual_climate['rain'] = rain.sel(year=pheno['POS_year'].values)

#     # join our POS metric to the climate data after updating
#     # index of our metric with the years
#     pheno['vPOS'].index = pheno['POS_year'].values
#     pheno['vPOS'].index.name = 'year'

#     #get all xarrays into the same dims etc.
#     lat = data.latitude.item()
#     lon = data.longitude.item()
#     pheno_vPOS = pheno['vPOS'].to_xarray().expand_dims(latitude=[lat],longitude=[lon])
#     annual_climate = annual_climate.drop_vars(['spatial', 'latitude', 'longitude']).squeeze()    
#     annual_climate.assign_coords(latitude=lat, longitude=lon)
    
#     for var in annual_climate.data_vars:
#         annual_climate[var] = annual_climate[var].expand_dims(latitude=[lat], longitude = [lon])

#     #now add our phenometric to the covars-we have a neat object to work with
#     annual_climate['vPOS'] = pheno_vPOS

#     #---------------PLS modelling----------------------------------------------------------------------
#     # fit PLS on rolling annuals to remove some of the IAV (only interested in trends)
#     df = annual_climate.squeeze().rolling(year=5, min_periods=5).mean().drop_vars('spatial_ref').to_dataframe()
#     df = df.dropna()
            
#     #fit a model with all vars
#     x = df[['co2','srad', 'rain','tavg', 'vpd']]
#     y = df['vPOS']
#     lr = PLSRegression().fit(x, y)
#     prediction = lr.predict(x)
#     r2_all = r2_score(y, prediction)
    
#     # Find the robust slope of actual
#     result_actual = mk.original_test(y, alpha=0.05)
#     p_actual = result_actual.p
#     s_actual = result_actual.slope
#     i_actual = result_actual.intercept
    
#     #calculate slope of predicted variable with all params
#     result_prediction = mk.original_test(prediction, alpha=0.05)
#     p_prediction = result_prediction.p
#     s_prediction = result_prediction.slope
#     i_prediction = result_prediction.intercept

#     #get the PLS coefficients
#     fi = pd.Series(dict(zip(list(x.columns), list(lr.coef_.reshape(len(x.columns)))))).to_frame()
#     fi = fi.rename({0:'PLS_coefficent'},axis=1)
#     fi = fi.reset_index().rename({'index':'feature'},axis=1).set_index('feature')
    
#     # create tidy df with all stats
#     # fi['phenometric'] = 'vPOS'
#     fi['slope_actual'] = s_actual
#     fi['slope_modelled'] = s_prediction
#     fi['p_actual'] = p_actual
#     fi['p_modelled'] = p_prediction
#     fi['i_actual'] = i_actual
#     fi['i_modelled'] = i_prediction
#     fi['r2'] = r2_all

#     fi = fi.to_xarray().squeeze().expand_dims(latitude=[lat],longitude=[lon])
#     return fi








