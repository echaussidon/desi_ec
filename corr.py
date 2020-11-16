# coding: utf-8

# function for build_data.py in Mocks folder

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
    sel = (pixmap>0) #We remove pixel with nothing inside...
    print('Number of pix selected in pixmap =', np.sum(sel))

    ascii.write([ra_list[sel], dec_list[sel], z[sel], pixmap[sel]],
                 filename , names=['ra', 'dec', 'z', 'w'],
                 format='no_header', overwrite=True)
