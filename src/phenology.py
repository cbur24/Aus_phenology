
'''
Robust phenometrics for Australian ecosystems
Works on 1D timeseries only.

'''

import calendar
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal
import warnings
warnings.simplefilter('ignore')

def extract_peaks_troughs(input_dict, rolling=90, distance=6, prominence='auto', plateau_size=2):
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
    A dictionary with keys the same as the input and the values a pandas.dataframe with peaks
    and trough identified by the time stamp they occur.
    
    """
    peaks_troughs = {}
    for k in input_dict.keys():
        #calculate rolling min/max to find local minima maxima
        roll_max = input_dict[k].rolling(time=90, min_periods=1, center=True).max()
        roll_min = input_dict[k].rolling(time=90, min_periods=1, center=True).min()
    
        if prominence=='auto':
            #dynamically determine how prominent a peak needs to be
            #based upon typical range of seasonal cycle
            clim = input_dict[k].groupby('time.month').mean()
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
        # i.e. enforcing the pattern peak-vally-peak-valleys etc.
        # First get the peaks and troughs into a dataframe with matching time index 
        df = input_dict[k].drop_vars('spatial_ref').to_dataframe()
        df['peaks'] = input_dict[k].isel(time=peaks).drop_vars('spatial_ref').to_dataframe()
        df['troughs'] = input_dict[k].isel(time=troughs).drop_vars('spatial_ref').to_dataframe()
        df_peaks_troughs = df.drop('NDVI',axis=1).dropna(how='all')

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
            
            min_peak_to_drop = df.idxmin().peaks
            df_peaks_troughs = df_peaks_troughs.drop(min_peak_to_drop)
        
        # Loop through locations with two sequential troughs and drop
        # the higher of the two troughs (less prominent trough)
        for idx in idx_consecutive_troughs.index:
            
            loc = df_peaks_troughs.index.get_loc(idx)
            df = df_peaks_troughs.iloc[[loc-1,loc]]
            
            min_trough_to_drop = df.idxmax().troughs
            df_peaks_troughs = df_peaks_troughs.drop(min_trough_to_drop)
        
        #add results to dict for later
        peaks_troughs[k] = df_peaks_troughs

    return peaks_troughs


def phenometrics(input_dict, rolling=90,distance=180, prominence='auto', plateau_size=2, amplitude=0.20):
    """
    Calculate statistics that describe the phenology cycle of
    a vegetation condition time series.
    
    Identifies the start and end points of each cycle using 
    the `seasonal amplitude` method. When the vegetation time series reaches
    20% of the sesonal amplitude between the first minimum and the peak,
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
        * ``'AOS'``: Amplitude of season (in value units)
        * ``'IOS'``: Integral of season (in value units)
        * ``'ROG'``: Rate of greening (value units per day)
        * ``'ROS'``: Rate of senescence (value units per day)
    
    returns:
    --------
    Dictionary where keys are the labels of the polygons, and values
    are Pandas.Dataframe containing phenometrics.
    
    """

    #first extract peaks and troughs in the timeseries
    peaks_troughs = extract_peaks_troughs(input_dict,
                         rolling=rolling,
                         distance=distance,
                         prominence=prominence,
                         plateau_size=plateau_size)

    #loop through dictionary and extract phenometrics
    eco_regions_phenometrics = {}
    i=0
    for k,v in input_dict.items():
        print("Feature {:02}/{:02}\r".format(i + 1, len(range(0, len(input_dict)))), end="")
    
        # start the timeseries with trough
        if np.isnan(peaks_troughs[k].iloc[0].troughs):
            p_t = peaks_troughs[k].iloc[1:]
        else:
            p_t = peaks_troughs[k]
    
        # end the timeseries with trough
        if np.isnan(p_t.iloc[-1].troughs)==True:
            p_t = p_t.iloc[0:-1]
            
        # Store phenology stats
        phenometrics = {}
        
        peaks_only = p_t.peaks.dropna()
        for peaks, idx in zip(peaks_only.index, range(0,len(peaks_only))):
            
            # First we extract the trough times either side of the peak
            start_time = p_t.iloc[p_t.index.get_loc(peaks)-1].name
            end_time = p_t.iloc[p_t.index.get_loc(peaks)+1].name
    
            # now extract the NDVI time series for the cycle
            ndvi_cycle = input_dict[k].sel(time=slice(start_time, end_time))
            
            # add the stats to this
            vars = {}
           
            # --Extract phenometrics---------------------------------
            pos = ndvi_cycle.idxmax()
            vars['POS_year'] = pos.dt.year #so we can keep track
            vars['POS'] = pos.dt.dayofyear.values
            vars['vPOS'] = ndvi_cycle.max().values
            #we want the trough values from the beginning of the season only (left side)
            vars['TOS_year'] =  p_t.iloc[p_t.index.get_loc(peaks)-1].name.year
            vars['TOS'] = p_t.iloc[p_t.index.get_loc(peaks)-1].name.dayofyear
            vars['vTOS'] = p_t.iloc[p_t.index.get_loc(peaks)-1].troughs
            vars['AOS'] = (vars['vPOS'] - vars['vTOS'])
            
            #SOS ------ 
            # Find the greening cycle (left of the POS)
            greenup = ndvi_cycle.where(ndvi_cycle.time <= pos)
            # Find absolute distance between 20% of the AOS and the values of the greenup, then
            # find the NDVI value that's closest to 20% of AOS, this is our SOS date
            sos = np.abs(greenup - (vars['AOS'] * amplitude + vars['vTOS'])).idxmin()
            vars['SOS_year'] = sos.dt.year #so we can keep track
            vars['SOS'] = sos.dt.dayofyear.values
            vars['vSOS'] = ndvi_cycle.sel(time=sos).values
            
            #EOS ------
            # Find the senescing cycle (right of the POS)
            browning = ndvi_cycle.where(ndvi_cycle.time >= ndvi_cycle.idxmax())
            # Find absolute distance between 20% of the AOS and the values of the browning, then
            # find the NDVI value that's closest to 20% of AOS, this is our EOS date
            ampl_browning = browning.max() - browning.min()
            eos = np.abs(browning - (ampl_browning * amplitude + browning.min())).idxmin()
            vars['EOS_year'] = eos.dt.year #so we can keep track
            vars['EOS'] = eos.dt.dayofyear.values
            vars['vEOS'] = ndvi_cycle.sel(time=eos).values
    
            # LOS ---
            los = (pd.to_datetime(eos.values) - pd.to_datetime(sos.values)).days
            vars['LOS'] = los

            #Integral of season
            ios = ndvi_cycle.sel(time=slice(sos, eos))
            ios = ios.integrate(coord='time', datetime_unit='D')
            vars['IOS'] = ios
    
            # Rate of growth and sensecing (NDVI per day)
            vars['ROG'] = (vars['vPOS'] - vars['vSOS']) / ((pd.to_datetime(pos.values) - pd.to_datetime(sos.values)).days)
            vars['ROS'] = (vars['vEOS'] - vars['vPOS']) / ((pd.to_datetime(eos.values) - pd.to_datetime(pos.values)).days) 
    
            phenometrics[idx] = vars
           
        #add metric back to ecoregions dict
        eco_regions_phenometrics[k] = pd.DataFrame(phenometrics).astype('float32').transpose()
        i+=1
        
    return eco_regions_phenometrics



