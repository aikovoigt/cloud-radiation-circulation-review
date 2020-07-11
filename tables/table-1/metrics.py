# Helper functions for calculating ITCZ position and Jet position and strength

import numpy as np

#####################################################################################
# ITCZ position
#####################################################################################

def _reorder_south2north(data, lat):
    # if latitude is not indexed from SP to NP, then reorder
    if lat[0]>lat[1]:
        lat = lat[::-1]
        data  = data[::-1]
    return data, lat

def get_itczposition(pr, lat):
    # calculate ITCZ position as precipitation centroid between 20 deg N/S
    # see Harrop et al. (2018), GRL, doi:10.1029/2018GL080772
    #
    # input
    # - pr is precipitation
    # - lat is latitude in degrees
    
    pr, lat = _reorder_south2north(pr, lat)
    # interpolate lat and pr onto 0.1 lat grid
    lati  = np.arange(-20.0, 20.0, 0.1)
    pri   = np.interp(lati, lat, pr)
    areai = np.cos(lati*np.pi/180)
    return np.nansum(lati * areai * pri) / np.nansum(areai * pri)

#####################################################################################
# Extratropical jet position (and strength)
#####################################################################################

def _func_fit_quadratic(x, p0, p1, p2):
    return p0+p1*x+p2*x**2

def get_eddyjetlatint(u, lat):
    from scipy.optimize import curve_fit 
    
    # make sure that lat is ordered from SP to NP; otherwise
    # np.arange does not work to create latint
    if lat[0] > lat[1]:
        lat = lat[::-1]
        u = u[::-1]

    # Southern Hemisphere
    indlat_sh = np.squeeze(np.array(np.where((lat<-25.0) & (lat>-70.0))))
    # find index of maximum value of u within the defined latitude range
    maxlat_index = np.unravel_index(np.argmax(u[indlat_sh]),
                                    u[indlat_sh].shape)  + indlat_sh[0]
    maxlat = maxlat_index[0]
    
    # do quadratic fit around the maximum, interpolate latitude and u-wind
    latint=np.arange(lat[maxlat-1], lat[maxlat+1], 0.01)
    uint  = np.interp(latint, lat[indlat_sh], u[indlat_sh])
    p, _  = curve_fit(_func_fit_quadratic, latint, uint)
    ufit  = _func_fit_quadratic(latint, p[0], p[1], p[2])
    jetlat_sh = latint[np.argmax(ufit)]

    # get strength of jet (maximum of u850 at jet latitude)
    jetint_sh = ufit[np.argmax(ufit)] # = ufit.max(

    # delete variables
    del indlat_sh, maxlat_index, maxlat, latint, uint, p, ufit

    # Northern Hemisphere
    indlat_nh = np.squeeze(np.array(np.where((lat>25.0) & (lat<70.0))))
    # find index of maximum value of u within the defined latitude range
    maxlat = np.argmax(u[indlat_nh]) + indlat_nh[0]

    # do a quadratic fit around the maximum, interpolate latitude and u-wind
    latint=np.arange(lat[maxlat-1], lat[maxlat+1], 0.01)
    uint  = np.interp(latint, lat[indlat_nh], u[indlat_nh])
    p, _  = curve_fit(_func_fit_quadratic, latint, uint)
    ufit  = _func_fit_quadratic(latint, p[0], p[1], p[2])
    jetlat_nh = latint[np.argmax(ufit)]

    # get strength of jet (maximum of u850 at jet latitude)
    jetint_nh = ufit[np.argmax(ufit)] # = ufit.max()
   
    return jetlat_sh, jetint_sh, jetlat_nh, jetint_nh