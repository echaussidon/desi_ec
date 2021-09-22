# coding: utf-8
# Author : Edmond Chaussidon (CEA)

# fonction pour calculer les magnitudes / les probas pour la selection des quasars

import sys
import os
import logging
logger = logging.getLogger("TS_utils")

import numpy as np
import healpy as hp

from wrapper import time_measurement

from desitarget.myRF import myRF
from desitarget.cuts import shift_photo_north
from desitarget.geomask import hp_in_box
from pkg_resources import resource_filename

#Y a une fonction dans desiarget pour le faire mais ok on perd pas de temps (et puis c'est la ou je l'ai mis !)
pathToRF = '/global/homes/e/edmondc/Software/desitarget/py/desitarget/data'

# number of variables
nfeatures = 11

## WARNING :
##
## Build to work with pandas DataFrame, read target file with tpcf.read_fits_to_pandas

#------------------------------------------------------------------------------#
# Usefull fonction

def magsExtFromFlux(dataFrame, show=False):
    gflux  = dataFrame['FLUX_G'].values/dataFrame['MW_TRANSMISSION_G'].values
    rflux  = dataFrame['FLUX_R'].values/dataFrame['MW_TRANSMISSION_R'].values
    zflux  = dataFrame['FLUX_Z'].values/dataFrame['MW_TRANSMISSION_Z'].values
    W1flux  = dataFrame['FLUX_W1'].values/dataFrame['MW_TRANSMISSION_W1'].values
    W2flux  = dataFrame['FLUX_W2'].values/dataFrame['MW_TRANSMISSION_W2'].values
    
    W1flux[np.isnan(W1flux)]=0.
    W2flux[np.isnan(W2flux)]=0.
    gflux[np.isnan(gflux)]=0.
    rflux[np.isnan(rflux)]=0.
    zflux[np.isnan(zflux)]=0.
    W1flux[np.isinf(W1flux)]=0.
    W2flux[np.isinf(W2flux)]=0.
    gflux[np.isinf(gflux)]=0.
    rflux[np.isinf(rflux)]=0.
    zflux[np.isinf(zflux)]=0.
  
    is_north = (dataFrame['DEC'].values >= 32) & (dataFrame['RA'].values >= 60) & (dataFrame['RA'].values <= 310)
    logger.info(f'shift photometry for {is_north.sum()} objects')
    if show:
        plt.figure()
        plt.hist(gflux[is_north], bins=100, range=(0, 50), label='not shifted')
    gflux[is_north], rflux[is_north], zflux[is_north] = shift_photo_north(gflux[is_north], rflux[is_north], zflux[is_north])
    if show:
        plt.hist(gflux[is_north], bins=100, range=(0, 50), label='shifted')
        plt.legend()
        plt.show()

    g=np.where(gflux>0,22.5-2.5*np.log10(gflux), 0.)
    r=np.where(rflux>0,22.5-2.5*np.log10(rflux), 0.)
    z=np.where(zflux>0,22.5-2.5*np.log10(zflux), 0.)
    W1=np.where(W1flux>0, 22.5-2.5*np.log10(W1flux), 0.)
    W2=np.where(W2flux>0, 22.5-2.5*np.log10(W2flux), 0.)

    g[np.isnan(g)]=0.
    g[np.isinf(g)]=0.
    r[np.isnan(r)]=0.
    r[np.isinf(r)]=0.
    z[np.isnan(z)]=0.
    z[np.isinf(z)]=0.
    W1[np.isnan(W1)]=0.
    W1[np.isinf(W1)]=0.
    W2[np.isnan(W2)]=0.
    W2[np.isinf(W2)]=0.
    
    return g, r, z, W1, W2


def colors(nbEntries, nfeatures, g, r, z, W1, W2):
    colors = np.zeros((nbEntries,nfeatures))

    colors[:,0] = g-r
    colors[:,1] = r-z
    colors[:,2] = g-z
    colors[:,3] = g-W1
    colors[:,4] = r-W1
    colors[:,5] = z-W1
    colors[:,6] = g-W2
    colors[:,7] = r-W2
    colors[:,8] = z-W2
    colors[:,9] = W1-W2
    colors[:,10] = r

    return colors


@time_measurement
def compute_proba(dataFrame):
    object_g, object_r ,object_z ,object_W1 ,object_W2 = magsExtFromFlux(dataFrame)
    attributes = colors(object_g.size, nfeatures, object_g, object_r, object_z, object_W1, object_W2)

    rf_fileName = pathToRF + f'/rf_model_dr9_final.npz'

    logger.info('Load Random Forest: ')
    logger.info('    * ' + rf_fileName)
    logger.info('Random Forest over: ', len(attributes),' objects\n')
    logger.info('    * start RF calculation...')
    myrf =  myRF(attributes, pathToRF, numberOfTrees=500, version=2)
    myrf.loadForest(rf_fileName)
    proba_rf = myrf.predict_proba()

    return proba_rf 


def give_footprint(dataFrame):
    # Compute in which zone the target is
    # To apply a corresponding threshold
    
    is_north = (dataFrame['DEC'] >= 32.2) &\
               (dataFrame['RA'] >= 60) &\
               (dataFrame['RA'] <= 310)
    
    is_des = (dataFrame['NOBS_G'] > 4) &\
         (dataFrame['NOBS_R'] > 4) &\
         (dataFrame['NOBS_Z'] > 4) &\
         ((dataFrame['RA'] >= 320) | (dataFrame['RA'] <= 100)) &\
         (dataFrame['DEC'] <= 10)
    
    is_south = (dataFrame['DEC'] < 50) & ~is_des & ~is_north
    
    print(is_north.sum() + is_south.sum() + is_des.sum())
    print(is_north.size)
    
    return is_north, is_south, is_des


def build_pixmap(dataFrame, Nside, in_deg=True):

    pixmap = np.zeros(hp.nside2npix(Nside))
    pixels = hp.ang2pix(Nside, dataFrame['RA'][:], dataFrame['DEC'][:], nest=True, lonlat=True)
    pix, counts = np.unique(pixels, return_counts=True)
    pixmap[pix] = counts
    
    pixmap /= hp.nside2pixarea(Nside, degrees=True)
    
    return pixmap


## ADD INFO TO THE DATAFRAME:
def add_mags_to_df(dataFrame):
    object_g, object_r ,object_z ,object_W1 ,object_W2 = magsExtFromFlux(dataFrame)
    dataFrame['g'] = object_g
    dataFrame['r'] = object_r
    dataFrame['z'] = object_z
    dataFrame['W1'] = object_W1
    dataFrame['W2'] = object_W2
    
    
def add_colors_to_df(dataFrame):
    #need to call add_mag_to_df before !
    
    dataFrame['g-r'] = dataFrame['g'] - dataFrame['r']
    dataFrame['r-z'] = dataFrame['r'] - dataFrame['z']
    dataFrame['g-z'] = dataFrame['g'] - dataFrame['z']
    dataFrame['g-W1'] = dataFrame['g'] - dataFrame['W1']
    dataFrame['r-W1'] = dataFrame['r'] - dataFrame['W1']
    dataFrame['z-W1'] = dataFrame['z'] - dataFrame['W1']
    dataFrame['g-W2'] = dataFrame['g'] - dataFrame['W2']
    dataFrame['r-W2'] = dataFrame['z'] - dataFrame['W2']
    dataFrame['z-W2'] = dataFrame['z'] - dataFrame['W2']
    dataFrame['W1-W2'] = dataFrame['W1'] - dataFrame['W2']

    
def add_proba_to_df(dataFrame, filename_proba='oupsi.npy', already_computed=False):
    if already_computed:
        logger.info(f"Load proba from {filename_proba}")
        dataFrame['p_RF'] = np.load(filename_proba)
    else:
        proba_rf = compute_proba(dataFrame)
        dataFrame['p_RF'] = proba_rf
        np.save(filename_proba, proba_rf)
        logger.info(f"Save proba to {filename_proba}")
        

def add_footprint_to_df(dataFrame):
    is_north, is_south, is_des = give_footprint(dataFrame)
    dataFrame['IS_NORTH'] = is_north
    dataFrame['IS_SOUTH'] = is_south
    dataFrame['IS_DES'] = is_des
