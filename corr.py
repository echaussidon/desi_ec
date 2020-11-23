# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import numpy as np
import healpy as hp
import fitsio
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import ascii

#------------------------------------------------------------------------------#
def get_data(Nside, catalog_name, add_ra=0, add_dec=0):
    pixmap = np.zeros(hp.nside2npix(Nside))
    print("[READ File :]", catalog_name)
    catalog = fitsio.FITS(catalog_name)[1]['RA', 'DEC']
    pixels = hp.ang2pix(Nside, catalog['RA'][:] + add_ra, catalog['DEC'][:] + add_dec, nest=True, lonlat=True)
    pix, counts = np.unique(pixels, return_counts=True)
    pixmap[pix] = counts
    return pixmap

#a partir d'un Nside, renvoit les positions de chaque pixel (ra, dec) dans l'ordre NESTED
def get_ra_dec(Nside):
    ra, dec = hp.pix2ang(Nside, np.arange(hp.nside2npix(Nside)), nest=True, lonlat=True)
    return ra, dec

def save_data(Nside, pixmap, ra_list=None, dec_list=None, filename='oups', mean_z=1.6):
    z = np.ones(pixmap.size)*mean_z
    sel = (pixmap != 0) #We remove pixel with nothing inside...
    print('Number of pix selected (non-zeros) in pixmap =', np.sum(sel))

    ascii.write([ra_list[sel], dec_list[sel], z[sel], pixmap[sel]],
                 filename , names=['ra', 'dec', 'z', 'w'],
                 format='no_header', overwrite=True)

def compute_result(filename) :

    data_xi = ascii.read(filename, format='no_header', names=['R','Xi','DD','DR','RD','RR'])

    dd=np.array(data_xi['DD'])
    dr=np.array(data_xi['DR'])+np.array(data_xi['RD'])
    rr=np.array(data_xi['RR'])
    r=np.array(data_xi['R'])
    xi=np.array(data_xi['Xi'])

    edd = np.sqrt(dd)
    edr = np.sqrt(dr)
    err = np.sqrt(rr)
    err_xi = (1+xi)/np.sqrt(dd) #### double check the computation for the error

    err_r = np.zeros(len(r))/rr

    return r, xi, err_r, err_xi

def plot_ang_corr(ax, filename, err_y=True, color=None, linestyle='-', marker='.', markerfacecolor=None, label=None, alpha=1, min_theta=0.05):
    r, xi, err_r, err_xi = compute_result(filename=filename)
    sel = (r>min_theta) & (xi>0.0)
    if err_y==False:
        yerr = None
    else:
        yerr = err_xi[sel]
    ax.errorbar(x=r[sel], y=xi[sel], xerr=None, yerr=yerr, marker=marker, markersize=6, markerfacecolor=markerfacecolor, linestyle=linestyle, color=color, label=label, alpha=alpha)
