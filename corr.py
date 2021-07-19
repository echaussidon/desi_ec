# coding: utf-8
# Author : Edmond Chaussidon (CEA)
# Fonctions utiles pour le calcul de la fonction de correlation angulaire 2 points

import numpy as np
import healpy as hp
import fitsio
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from astropy.io import ascii
from desitarget.geomask import hp_in_box

#------------------------------------------------------------------------------#
def get_data(Nside, catalog_name, in_deg=False, add_ra=0, add_dec=0):
    pixmap = np.zeros(hp.nside2npix(Nside))
    print(f"[INFO] Build healpix map with Nside={Nside} from {catalog_name}")
    print(f"[INFO] The map is in density/deg^2: {in_deg}")
    catalog = fitsio.FITS(catalog_name)[1]['RA', 'DEC']
    pixels = hp.ang2pix(Nside, catalog['RA'][:] + add_ra, catalog['DEC'][:] + add_dec, nest=True, lonlat=True)
    pix, counts = np.unique(pixels, return_counts=True)
    pixmap[pix] = counts
    if in_deg:
        pixmap /= hp.nside2pixarea(Nside, degrees=True)
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

#------------------------------------------------------------------------------#
#Construction des patchs (sous forme de pixel) pour faire jackknife or subsampling

def find_rabox_from_decline(dec_1, dec_2, footprint, Nside, is_des):
    zone_tmp = np.array(hp_in_box(Nside, [0, 360, dec_1, dec_2]))
    pix_list = zone_tmp[footprint[zone_tmp] == 1]
    ra_list, _ = hp.pix2ang(Nside, pix_list, nest=True, lonlat=True)
    if is_des: ra_list[ra_list>300] = ra_list[ra_list>300] - 360
    ra_min, ra_max = np.floor(np.min(ra_list)), np.ceil(np.max(ra_list))
    if is_des: ra_min += 360
    return ra_min, ra_max

def add_zone(radec_box, footprint, Nside):
    ra1, ra2, dec1, dec2 = radec_box[0], radec_box[1], radec_box[2], radec_box[3]
    if (ra1>300) & (ra2<100):
        zone_tmp = np.array(hp_in_box(Nside, [ra1, 360, dec1, dec2])+ hp_in_box(Nside, [0, ra2, dec1, dec2]))
    else:
        zone_tmp = np.array(hp_in_box(Nside, [ra1, ra2, dec1, dec2]))
    return [zone_tmp[footprint[zone_tmp] == 1]]

def build_patch(dec_min, dec_max, nsplit_dec, nsplit_ra_list, footprint, Nside, is_des, print_info):
    width_dec = (dec_max - dec_min)/nsplit_dec

    patch_list = []

    for i in range(nsplit_dec):
        dec_1, dec_2 = dec_min + i*width_dec, dec_min + (i+1)*width_dec
        if print_info: print("DEC : ", dec_1, dec_2)
        ra_tmp_min, ra_tmp_max = find_rabox_from_decline(dec_1, dec_2, footprint, Nside, is_des)
        if (ra_tmp_min>300):
            width_ra = (ra_tmp_max - ra_tmp_min + 360) / nsplit_ra_list[i]
        else:
            width_ra = (ra_tmp_max - ra_tmp_min) / nsplit_ra_list[i]
        if print_info: print("RA : ", ra_tmp_min, ra_tmp_max, width_ra)

        for j in range(nsplit_ra_list[i]):
            ra_1, ra_2 = (ra_tmp_min + j*width_ra)%360, (ra_tmp_min + (j+1)*width_ra)%360
            if ra_2 == 0:
                ra_2 = 360
            if print_info: print('    ** ', ra_1, ra_2)
            radec_box = [ra_1, ra_2, dec_1, dec_2]
            patch_list += add_zone(radec_box, footprint, Nside)
            if print_info: print('        --> ', len(patch_list[-1]))
        if print_info: print(" ")

    return patch_list

#patch_info_north = {'dec_min':32, 'dec_max':90, 'nsplit_dec':10, 'nsplit_ra_list':[10, 10, 10, 10, 10, 10, 10, 10, 10, 10], 'footprint':north, 'Nside':Nside, 'is_des':False, 'print_info':False}
#patch_info_south = {'dec_min':-18, 'dec_max':34, 'nsplit_dec':10, 'nsplit_ra_list': [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], 'footprint':south, 'Nside':Nside, 'is_des':False, 'print_info':False}
#patch_info_des = {'dec_min':-65, 'dec_max':5, 'nsplit_dec':10, 'nsplit_ra_list':[12, 12, 15, 15, 10, 10, 10, 6, 5, 5], 'footprint':des, 'Nside':Nside, 'is_des':True, 'print_info':False}

#------------------------------------------------------------------------------#
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
    
    print("[WARNING] We interpolate error so it is not true one --> since if bins are smaller, DD will be too")

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


# pour calculer les erreurs sur les mocks à partir de deux fichiers (un petit et l'autre grande échelle)
# si qu'un fichier, plot_ang_corr_mean suffit directement!
def compute_error_from_reconstruction_mocks(filename, suffixe1, suffixe2, range, split_theta=0.5, min_theta=1e-3, max_theta=9.5, nbins=None):
    # We supposed that file2 goes at smaller theta than file1
    r, xi_temp, _, _ = reconstruct_ang_corr(filename+str(range[0])+suffixe1+'.txt', filename+str(range[0])+suffixe2+'.txt', split_theta=split_theta, min_theta=min_theta, max_theta=max_theta, nbins=nbins)
    # il faut evidement que les correlations aient le meme binning..

    xi_list = np.zeros((len(range), xi_temp.size))
    xi_list[0] = xi_temp

    for i in range[1:]:
        _, xi_temp, _, _ = reconstruct_ang_corr(filename+str(i)+suffixe1+'.txt', filename+str(i)+suffixe2+'.txt', split_theta=split_theta, min_theta=min_theta, max_theta=max_theta, nbins=nbins)
        xi_list[i-1] = xi_temp

    xi = np.mean(xi_list, axis=0)
    # on multiplie par sqrt(N/(N-1)) pour avoir une erreur non biasee ! cf. https://www.math.u-bordeaux.fr/~mchabano/Agreg/ProbaAgreg1213-COURS2-Stat1.pdf
    err_xi = np.sqrt(len(range)/(len(range)-1)) * np.std(xi_list, axis=0)

    return r, err_xi


# Ne pas faire de moyenné pondéré car l'erreur dépend de xi .. --> on biase donc notre moyenne :)
def plot_ang_corr_mean(ax, filename, suffixe, range, err_y=True, color=None, marker='o', linestyle=None, markersize=None, linewidth=None, capsize=2,
                       markerfacecolor=None, label=None, min_theta=0.01, max_theta=9.5, nbins=None, return_mean=False, plot=True):
    r, xi_temp, err_r, err_xi_temp = compute_result(filename=filename+str(range[0])+suffixe+'.txt')
    #il faut evidement que les correlations aient le meme binning..

    xi_list, err_xi_list = np.zeros((len(range), xi_temp.size)), np.zeros((len(range), err_xi_temp.size))
    xi_list[0], err_xi_list[0] = xi_temp, err_xi_temp

    for i in range[1:]:
        _, xi_temp, _, err_xi_temp = compute_result(filename=filename+str(i)+suffixe+'.txt')
        xi_list[i-1], err_xi_list[i-1] = xi_temp, err_xi_temp

    xi = np.mean(xi_list, axis=0)
    err_xi = np.std(xi_list, axis=0) / np.sqrt(len(range) - 1)  # en realite c'est sqrt(N/(N-1)) * sigma / sqrt(N)
    # oui c'est l'erreur d'une moyenne cf carnet elle tend forcément vers 0 quand N augmente --> pas la meme chose pour l'erreur que l'on veut grace aux Mocks !

    if nbins != None:
        r, xi, err_r, err_xi = interpolate_ang_corr(r, xi, err_r, err_xi, min_theta, max_theta, nbins)

    if err_y == False:
        err_xi = np.zeros(err_xi.size)

    sel = (r >= min_theta) & (r<= max_theta)
    r, xi, err_r, err_xi = r[sel], xi[sel], err_r[sel], err_xi[sel]

    if plot:
        sel = (xi > 0)
        ax.errorbar(x=r[sel], y=xi[sel], yerr=err_xi[sel], marker=marker, markersize=markersize, linewidth=linewidth, markerfacecolor=markerfacecolor, linestyle=linestyle, capsize=capsize, color=color, label=label, alpha=1)

    if return_mean:
        return r, xi, err_r, err_xi
