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

    # PAS DE PROBLEME car np.where fait quand meme le travail mais n'enleve pas l'erreur ...
    # cf stack overflow
    # invalid value to avoid warning with log estimation --> deal with nan
    with np.errstate(divide='ignore', invalid='ignore'):
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


def build_pixmap(dataFrame, Nside, use_weight=False, in_deg=True):
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


def add_footprint_for_cut_to_df(dataFrame):
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
    is_north, is_south, is_des = give_footprint(dataFrame)
    dataFrame['IS_NORTH'] = is_north
    dataFrame['IS_SOUTH'] = is_south
    dataFrame['IS_DES'] = is_des

#------------------------------------------------------------------------------#
import fitsio
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from desitarget.QA import _prepare_systematics
from desitarget.io import load_pixweight_recarray

from systematics import _load_systematics, systematics_med
from desi_footprint import DR9_footprint

def plot_systematic_from_map(map_list, label_list, savedir='', zone_to_plot=['North', 'South', 'Des'], Nside=256, remove_LMC=False, clear_south=True,
                            pixweight_path='/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/pixweight/main/resolve/dark/pixweight-1-dark.fits',
                            sgr_stream_path='/global/homes/e/edmondc/Systematics/regressor/Sagittarius_Stream/sagittarius_stream_256.npy',
                            ax_lim=0.3, adaptative_binning=False, nobjects_by_bins=2000, show=False, save=True,
                            n_bins=None):

    #load DR9 region
    DR9 = DR9_footprint(Nside, remove_LMC=remove_LMC, clear_south=clear_south)

    # load pixweight file and Sgr. map at correct Nside
    pixmap_tot = load_pixweight_recarray(pixweight_path, nside=Nside)
    sgr_stream_tot = np.load(sgr_stream_path)
    if Nside != 256:
        sgr_stream_tot = hp.ud_grade(sgr_stream_tot, Nside, order_in=True)

    # correct map with the fracarea (for maskbit 1, 12, 13)
    with np.errstate(divide='ignore'): # Ok --> on n'utilise pas les pixels qui n'ont pas été observé, ils sont en-dehors du footprint
        map_list_tot = [mp/pixmap_tot['FRACAREA_12290'] for mp in map_list]

    sysdic = _load_systematics()
    sysnames = list(sysdic.keys())

    for num_fig, key_word in enumerate(zone_to_plot):
        logger.info(f'Work with {key_word}')
        if key_word == 'Global':
            pix_to_keep = DR9.load_footprint()
            key_word_sys = key_word
        elif key_word == 'North':
            pix_to_keep, _, _ = DR9.load_photometry()
            key_word_sys = key_word
        elif key_word == 'South':
            _, pix_to_keep, _ = DR9.load_photometry()
            key_word_sys = key_word
        elif key_word == 'South_ngc':
            _, pix_to_keep, _ = DR9.load_photometry()
            ngc, _ = DR9.load_ngc_sgc()
            pix_to_keep &= ngc
            key_word_sys = 'South'
        elif key_word == 'South_sgc':
            _, pix_to_keep, _ = DR9.load_photometry()
            _, sgc = DR9.load_ngc_sgc()
            pix_to_keep &= sgc
            key_word_sys = 'South'
        elif key_word == 'Des':
            _, _, pix_to_keep = DR9.load_photometry()
            key_word_sys = key_word
        elif key_word == 'South_mid':
            _, pix_to_keep, _ = DR9.load_elg_region()
            key_word_sys = 'South'
        elif key_word == 'South_mid_ngc':
            _, pix_to_keep, _, _ = DR9.load_elg_region(ngc_sgc_split=True)
            key_word_sys = 'South'
        elif key_word == 'South_mid_sgc':
            _, _, pix_to_keep, _ = DR9.load_elg_region(ngc_sgc_split=True)
            key_word_sys = 'South'
        elif key_word == 'South_pole':
            _, _, pix_to_keep = DR9.load_elg_region()
            key_word_sys = 'Des'
        else:
            print("WRONG KEY WORD")
            sys.exit()

        pixmap = pixmap_tot[pix_to_keep]
        sgr_stream = sgr_stream_tot[pix_to_keep]
        map_list = [mp[pix_to_keep] for mp in map_list_tot]
        fracarea = pixmap['FRACAREA_12290']

        #Dessin systematiques traditionnelles
        fig = plt.figure(num_fig, figsize=(16.0,10.0))
        gs = GridSpec(4, 3, figure=fig, left=0.06, right=0.96, bottom=0.08, top=0.96, hspace=0.35 , wspace=0.20)

        num_to_plot = 0
        for i in range(12):
            sysname = sysnames[num_to_plot]
            d, u, plotlabel, nbins = sysdic[sysname][key_word_sys]
            if n_bins is not None:
                nbins=n_bins
            down, up = _prepare_systematics(np.array([d, u]), sysname)

            if sysname == 'STREAM':
                feature = _prepare_systematics(sgr_stream, sysname)
            else:
                feature =  _prepare_systematics(pixmap[sysname], sysname)

            if i not in [9]: #[2, 9]: #on affiche les systematics que l'on veut définies dans _load_systematics()
                ax = fig.add_subplot(gs[i//3, i%3])
                ax.set_xlabel(plotlabel)
                if i in [0, 3, 6]:
                    ax.set_ylabel("Relative QSO density - 1")
                ax.set_ylim([-ax_lim, ax_lim])

                for mp, label in zip(map_list, label_list):
                    bins, binmid, meds, nbr_obj_bins, meds_err = systematics_med(mp, fracarea, feature, sysname, downclip=down, upclip=up, nbins=nbins, adaptative_binning=adaptative_binning, nobjects_by_bins=nobjects_by_bins)
                    ax.errorbar(binmid, meds - 1*np.ones(binmid.size), yerr=meds_err, marker='.', linestyle='-', lw=0.8, label=label)

                ax_hist = ax.twinx()
                ax_hist.set_xlim(ax.get_xlim())
                ax_hist.set_ylim(ax.get_ylim())
                if adaptative_binning:
                    normalisation = nbr_obj_bins/0.1
                else:
                    normalisation = nbr_obj_bins.sum()
                ax_hist.bar(binmid, nbr_obj_bins/normalisation, alpha=0.4, color='dimgray', align='center', width=(bins[1:] - bins[:-1]), label='Fraction of nbr objects by bin \n(after correction) ')
                ax_hist.grid(False)
                ax_hist.set_yticks([])

                num_to_plot += 1

            #if i==2:
             #   ax = fig.add_subplot(gs[i//3, i%3])
             #   ax.axis("off")
             #   ax.text(0.5, 0.5, f"Zone : {key_word}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                #on enleve le STREAM comme feature ok
             #   num_to_plot += 1

            if i==10:
                ax.legend(bbox_to_anchor=(-1.1, 0.8), loc='upper left', borderaxespad=0., frameon=False, ncol=1, fontsize='large')
                ax_hist.legend(bbox_to_anchor=(-1.1, 0.35), loc='upper left', borderaxespad=0., frameon=False, ncol=1, fontsize='large')

        if save:
            plt.savefig(savedir+"{}_systematics_plot.pdf".format(key_word))
        if show:
            plt.show()
        else:
            plt.close()
