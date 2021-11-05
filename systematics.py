## coding: utf-8
## Author : Edmond Chaussidon (CEA)

import logging
logger = logging.getLogger("systematics")

import numpy as np
import warnings

def f(x) : return 22.5 - 2.5*np.log10(5/np.sqrt(x))

def g(y) : return 25*10**(2*(y - 22.5)/2.5)

def _load_systematics_old():
    """
    Loads information for making systematics plots. Copy and adapt from desitarget
    """

    sysdict = {}

    sysdict['STARDENS'] = [150., 4000., 'log10(Stellar Density) per sq. deg.']
    sysdict['EBV'] = [0.001, 0.1, 'E(B-V)']

    sysdict['PSFSIZE_G'] =[0., 3., 'PSF Size in g-band']
    sysdict['PSFSIZE_R'] =[0., 3., 'PSF Size in r-band']
    sysdict['PSFSIZE_Z'] =[0., 3., 'PSF Size in z-band']

    sysdict['PSFDEPTH_G'] =[63., 6300., 'PSF Depth in g-band']
    sysdict['PSFDEPTH_R'] =[25., 2500., 'PSF Depth in r-band']
    sysdict['PSFDEPTH_Z'] =[4., 400., 'PSF Depth in z-band']

    sysdict['PSFDEPTH_W1'] =[0.0, 30.0, 'PSF Depth in W1-band']
    sysdict['PSFDEPTH_W2'] =[0.0, 7.0, 'PSF Depth in W2-band']
    sysdict['SKYMAG_G'] = [23.0, 28.0, 'Sky Mag in g-band']
    sysdict['SKYMAG_R'] = [23.0, 28.0, 'Sky Mag in r-band']
    sysdict['SKYMAG_Z'] = [23.0, 28.0, 'Sky Mag in z-band']

    sysdict['MJD_G'] = [56193, 58500, 'MJD in g-band']
    sysdict['MJD_R'] = [57200, 58500, 'MJD in r-band']
    sysdict['MJD_Z'] = [57300, 58100, 'MJD in z-band']

    sysdict['EXPTIME_G'] = [30.0, 35000, 'Exposure time in g-band']
    sysdict['EXPTIME_R'] = [30.0, 8500, 'Exposure time in r-band']
    sysdict['EXPTIME_Z'] = [60.0, 16000, 'Exposure time in z-band']

    sysdict['FRACAREA_13538'] = [0.01, 1., 'Fraction of pixel area covered for QSOs']


    return sysdict

def _load_systematics():
    sysdict = {}

    sysdict['STARDENS'] = {'North':[150., 4000., 'log10(Stellar Density) per sq. deg.', 35],
                           'South':[150., 4000., 'log10(Stellar Density) per sq. deg.', 35],
                           'Des':[150., 4000., 'log10(Stellar Density) per sq. deg.', 30],
                           'Global':[150., 4000., 'log10(Stellar Density) per sq. deg.', 30]}
    sysdict['EBV'] = {'North':[0.001, 0.1, 'E(B-V)', 35],
                      'South':[0.001, 0.1, 'E(B-V)', 35],
                      'Des':[0.001, 0.09, 'E(B-V)', 35],
                      'Global':[0.001, 0.1, 'E(B-V)', 30]}
    sysdict['STREAM'] = {'North':[0., 1., 'Sgr. Stream', 1],
                      'South':[0.01, 1., 'Sgr. Stream', 20],
                      'Des':[0.01, 0.6, 'Sgr. Stream', 15],
                      'Global':[0.01, 1.5, 'Sgr. Stream', 20]}

    sysdict['PSFSIZE_G'] = {'North':[1.3, 2.6, 'PSF Size in g-band', 35],
                            'South':[1.1, 2.02, 'PSF Size in g-band', 35],
                            'Des':[1.19, 1.7, 'PSF Size in g-band',30],
                            'Global':[0., 3., 'PSF Size in g-band', 30]}
    sysdict['PSFSIZE_R'] ={'North':[1.25, 2.4, 'PSF Size in r-band', 40],
                           'South':[0.95, 1.92, 'PSF Size in r-band', 30],
                           'Des':[1.05, 1.5, 'PSF Size in r-band',30],
                           'Global':[0., 3., 'PSF Size in r-band', 30]}
    sysdict['PSFSIZE_Z'] = {'North':[0.9, 1.78, 'PSF Size in z-band', 35],
                            'South':[0.9, 1.85, 'PSF Size in z-band', 40],
                            'Des':[0.95, 1.4, 'PSF Size in z-band', 30],
                            'Global':[0., 3., 'PSF Size in z-band', 30]}

    sysdict['PSFDEPTH_G'] = {'North':[300., 1600., 'PSF Depth in g-band', 30],
                             'South':[750., 4000., 'PSF Depth in g-band', 35],
                             'Des':[1900., 7000., 'PSF Depth in g-band', 30],
                             'Global':[63., 6300., 'PSF Depth in g-band', 30]}
    sysdict['PSFDEPTH_R'] = {'North':[95., 620., 'PSF Depth in r-band', 30],
                             'South':[260.0, 1600.0, 'PSF Depth in r-band', 30],
                             'Des':[1200., 5523., 'PSF Depth in r-band', 30],
                             'Global':[25., 2500., 'PSF Depth in r-band', 30]}
    sysdict['PSFDEPTH_Z'] ={'North':[60., 275., 'PSF Depth in z-band', 40],
                            'South':[40.0, 360., 'PSF Depth in z-band', 40],
                            'Des':[145., 570., 'PSF Depth in z-band', 30],
                            'Global':[4., 400., 'PSF Depth in z-band', 30]}

    sysdict['PSFDEPTH_W1'] = {'North':[2.7, 12., 'PSF Depth in W1-band', 40],
                              'South':[2.28, 5.5, 'PSF Depth in W1-band', 30],
                              'Des':[2.28, 6.8, 'PSF Depth in W1-band', 30],
                              'Global':[0.0, 30.0, 'PSF Depth in W1-band', 30]}
    sysdict['PSFDEPTH_W2'] = {'North':[0.8, 3.9, 'PSF Depth in W2-band', 40],
                              'South':[0.629, 1.6, 'PSF Depth in W2-band', 30],
                              'Des':[0.62, 2.25, 'PSF Depth in W2-band', 30],
                              'Global':[0.0, 7.0, 'PSF Depth in W2-band', 30]}
    return sysdict

def systematics_med(targets, fracarea, feature, feature_name, downclip=None, upclip=None, nbins=50, use_mean=True, adaptative_binning=False, nobjects_by_bins=1000):
    """
    Return binmid, meds pour pouvoir faire : plt.errorbar(binmid, meds - 1*np.ones(binmid.size), yerr=meds_err, marker='.', linestyle='-', lw=0.9)
    """

    # Selection of pixels with correct fracarea and which respect the up/downclip for the 'feature_name' feature
    sel = (fracarea < 1.1) & (fracarea > 0.9) & (feature >= downclip) & (feature < upclip)
    if not np.any(sel):
        logger.info("Pixel map has no areas (with >90% coverage) with the up/downclip")
        logger.info("Proceeding without clipping systematics for {}".format(feature_name))
        sel = (fracarea < 1.1) & (fracarea > 0.9)
    targets = targets[sel]
    feature = feature[sel]

    if adaptative_binning: #set this option to have variable bin sizes so that each bin contains the same number of pixel (nobjects_by_bins)
        nbr_obj_bins = nobjects_by_bins
        ksort = np.argsort(feature)
        bins=feature[ksort[0::nbr_obj_bins]] #Here, get the bins from the data set (needed to be sorted)
        bins=np.append(bins,feature[ksort[-1]]) #add last point
        nbins = bins.size - 1 #OK
    else: # create bin with fix size (depends on the up/downclip value)
        nbr_obj_bins, bins = np.histogram(feature, nbins)

    # find in which bins belong each feature value
    wbin = np.digitize(feature, bins, right=True)

    if use_mean:
        norm_targets = targets/np.nanmean(targets)
        # I expect to see mean of empty slice (no non NaN value in the considered bin --> return nan value --> ok
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meds = [np.nanmean(norm_targets[wbin == bin]) for bin in range(1, nbins+1)]
    else:
        # build normalized targets : the normalization is done by the median density
        norm_targets = targets/np.nanmedian(targets)
        # digitization of the normalized target density values (first digitized bin is 1 not zero)
        meds = [np.nanmedian(norm_targets[wbin == bin]) for bin in range(1, nbins+1)]


    # error for mean (estimation of the std from sample)
    err_meds = [np.nanstd(norm_targets[wbin == bin]) / np.sqrt((wbin == bin).sum() - 1) if ((wbin == bin).sum() > 1) else 0 for bin in range(1, nbins+1)]

    return bins, (bins[:-1] + bins[1:])/ 2, meds, nbr_obj_bins, err_meds
