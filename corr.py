# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import numpy as np
import healpy as hp
import fitsio
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

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
    print(f'Number of pix selected (non-zeros) in pixmap = {np.sum(sel)}\nsaved in {filename}')

    ascii.write([ra_list[sel], dec_list[sel], z[sel], pixmap[sel]],
                 filename , names=['ra', 'dec', 'z', 'w'],
                 format='no_header', overwrite=True)

def au_dd(filename):
    data_xi = ascii.read(filename, format='no_header', names=['R','Xi','DD','DR','RD','RR'])

    dd=np.array(data_xi['DD'])
    dr=np.array(data_xi['DR'])
    rd=np.array(data_xi['RD'])
    rr=np.array(data_xi['RR'])
    r=np.array(data_xi['R'])

    return r, dd, dr, rd, rr

def compute_result(filename) :

    data_xi = ascii.read(filename, format='no_header', names=['R','Xi','DD','DR','RD','RR'])

    dd=np.array(data_xi['DD'])
    rr=np.array(data_xi['RR'])
    r=np.array(data_xi['R'])
    xi=np.array(data_xi['Xi'])

    err_xi = (1+xi)/np.sqrt(dd) ## au premier ordre c'est bien ca (en negliant les termes en alphabeta, beta^2, gamma^2, gammabeta)
    err_r = np.zeros(len(r))/rr

    return r, xi, err_r, err_xi

def interpolate_ang_corr(r, xi, err_r, err_xi, min_theta=1e-3, max_theta=9.5, nbins=20): #on fait ca avec un binning en log scale
    xi_interp = interpolate.interp1d(r, xi)
    err_r_interp = interpolate.interp1d(r, err_r)
    err_xi_interp = interpolate.interp1d(r, err_xi)

    bins = np.logspace(np.log10(min_theta),np.log10(max_theta), nbins)

    r, xi, err_r, err_xi = bins, xi_interp(bins), err_r_interp(bins), err_xi_interp(bins)

    return r, xi, err_r, err_xi

def plot_ang_corr(ax, filename, err_y=True, color=None, linestyle='-', marker='.', markersize=6, linewidth=None, markerfacecolor=None,
                  label=None, alpha=1, min_theta=0.05, max_theta=10, nbins=None):
    r, xi, err_r, err_xi = compute_result(filename=filename)
    if nbins != None:
        r, xi, err_r, err_xi = interpolate_ang_corr(r, xi, err_r, err_xi, min_theta, max_theta, nbins)

    sel = (r>=min_theta) & (r<=max_theta)& (xi>0.0)
    if err_y==False:
        yerr = None
    else:
        yerr = err_xi[sel]
    ax.errorbar(x=r[sel], y=xi[sel], xerr=None, yerr=yerr, marker=marker, markersize=markersize, markerfacecolor=markerfacecolor, linestyle=linestyle, linewidth=linewidth, color=color, label=label, alpha=alpha)

def reconstruct_ang_corr(file1, file2, split_theta=0.5, min_theta=1e-3, max_theta=9.5, nbins=None):
    #We supposed that file2 goes at smaller theta than file1
    r1, xi1, err_r1, err_xi1 = compute_result(file1)
    r2, xi2, err_r2, err_xi2 = compute_result(file2)

    r = np.concatenate((r2[r2<=split_theta], r1[r1 > split_theta]))
    xi = np.concatenate((xi2[r2<=split_theta], xi1[r1 > split_theta]))
    err_r = np.concatenate((err_r2[r2<=split_theta], err_r1[r1 > split_theta]))
    err_xi = np.concatenate((err_xi2[r2<=split_theta], err_xi1[r1 > split_theta]))

    if nbins != None:
        r, xi, err_r, err_xi = interpolate_ang_corr(r, xi, err_r, err_xi, min_theta, max_theta, nbins)

    sel = (r >= min_theta) & (r<= max_theta)
    r, xi, err_r, err_xi = r[sel], xi[sel], err_r[sel], err_xi[sel]

    return r, xi, err_r, err_xi

def plot_reconstruction_ang_corr(ax, file1, file2, err_y=True, color=None, linestyle='-', marker='.', markersize=6,
                                 linewidth=None, markerfacecolor=None, label=None, alpha=1, min_theta=0.05, max_theta=10, nbins=None):
    #We supposed that file2 goes at smaller theta than file1
    r, xi, err_r, err_xi = reconstruct_ang_corr(file1, file2, min_theta=min_theta, max_theta=max_theta, nbins=nbins)
    sel = (xi>0.0)
    if err_y==False:
        yerr = None
    else:
        yerr = err_xi[sel]
    ax.errorbar(x=r[sel], y=xi[sel], xerr=None, yerr=yerr, marker=marker, markersize=markersize, markerfacecolor=markerfacecolor, linestyle=linestyle, linewidth=linewidth, color=color, label=label, alpha=alpha)

#Ne pas faire de moyenné pondéré car l'erreur dépend de xi .. --> on biase donc notre moyenne :)
def plot_ang_corr_mean(ax, filename, suffixe, range, err_y=True, color=None, marker='o', linestyle=None,
                       markerfacecolor=None, label=None, min_theta=0.01, max_theta=9.5, nbins=None, return_mean=False, plot=True):
    r, xi_temp, err_r, err_xi_temp = compute_result(filename=filename+str(range[0])+suffixe+'.txt')
    #il faut evidement que les correlations aient le meme binning..

    xi_list, err_xi_list = np.zeros((len(range), xi_temp.size)), np.zeros((len(range), err_xi_temp.size))
    xi_list[0], err_xi_list[0] = xi_temp, err_xi_temp

    for i in range[1:]:
        _, xi_temp, _, err_xi_temp = compute_result(filename=filename+str(i)+suffixe+'.txt')
        xi_list[i-1], err_xi_list[i-1] = xi_temp, err_xi_temp

    xi = np.mean(xi_list, axis=0)
    err_xi = np.std(xi_list, axis=0) / np.sqrt(len(range) - 1)

    if nbins != None:
        r, xi, err_r, err_xi = interpolate_ang_corr(r, xi, err_r, err_xi, min_theta, max_theta, nbins)

    if err_y == False:
        err_xi = np.zeros(err_xi.size)

    sel = (r >= min_theta) & (r<= max_theta)
    r, xi, err_r, err_xi = r[sel], xi[sel], err_r[sel], err_xi[sel]

    if plot:
        sel = (xi > 0)
        ax.errorbar(x=r[sel], y=xi[sel], yerr=err_xi[sel], marker=marker, markersize=4, markerfacecolor=markerfacecolor, linestyle=linestyle, capsize=2, color=color, label=label, alpha=1)

    if return_mean:
        return r, xi, err_r, err_xi
