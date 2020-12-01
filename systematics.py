import numpy as np

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
                      'Des':[0.001, 0.1, 'E(B-V)', 35],
                      'Global':[0.001, 0.1, 'E(B-V)', 30]}

    sysdict['PSFSIZE_G'] = {'North':[1.3, 2.6, 'PSF Size in g-band', 35],
                            'South':[1.05, 2.1, 'PSF Size in g-band',35],
                            'Des':[1.15, 1.6, 'PSF Size in g-band',20],
                            'Global':[0., 3., 'PSF Size in g-band', 30]}
    sysdict['PSFSIZE_R'] ={'North':[1.25, 2.52, 'PSF Size in r-band',40],
                           'South':[1., 1.9, 'PSF Size in r-band',30],
                           'Des':[1.08, 1.4, 'PSF Size in r-band',15],
                           'Global':[0., 3., 'PSF Size in r-band', 30]}
    sysdict['PSFSIZE_Z'] = {'North':[0.9, 1.8, 'PSF Size in z-band',35],
                            'South':[0.9, 2.0, 'PSF Size in z-band',40],
                            'Des':[0.95, 1.3, 'PSF Size in z-band',20],
                            'Global':[0., 3., 'PSF Size in z-band', 30]}

    sysdict['PSFDEPTH_G'] = {'North':[300., 1600., 'PSF Depth in g-band',30],
                             'South':[500., 3800., 'PSF Depth in g-band',35],
                             'Des':[1500., 5500., 'PSF Depth in g-band',25],
                             'Global':[63., 6300., 'PSF Depth in g-band', 30]}
    sysdict['PSFDEPTH_R'] = {'North':[70.0, 600., 'PSF Depth in r-band',30],
                             'South':[270.0, 1500.0, 'PSF Depth in r-band',25],
                             'Des':[1000., 4000., 'PSF Depth in r-band',20],
                             'Global':[25., 2500., 'PSF Depth in r-band', 30]}
    sysdict['PSFDEPTH_Z'] ={'North':[50., 260., 'PSF Depth in z-band',40],
                            'South':[40.0, 360., 'PSF Depth in z-band',40],
                            'Des':[100., 490.0, 'PSF Depth in z-band',30],
                            'Global':[4., 400., 'PSF Depth in z-band', 30]}

    sysdict['PSFDEPTH_W1'] = {'North':[2.0, 12.5, 'PSF Depth in W1-band',40],
                              'South':[1.9, 4.8, 'PSF Depth in W1-band',20],
                              'Des':[1.9, 4.7, 'PSF Depth in W1-band',20],
                              'Global':[0.0, 30.0, 'PSF Depth in W1-band',30]}
    sysdict['PSFDEPTH_W2'] = {'North':[0.55, 3.2, 'PSF Depth in W2-band',40],
                              'South':[0.48, 1.3, 'PSF Depth in W2-band',20],
                              'Des':[0.48, 1.3, 'PSF Depth in W2-band',20],
                              'Global':[0.0, 7.0, 'PSF Depth in W2-band',30]}
    return sysdict

def qasystematics_binmid_med(targets, pixmap, syscolname, qadir='.', downclip=None, upclip=None, nbins=50, adaptative_binning=True, nobjects_by_bins=1000, fileprefix="sysdens", xlabel=None):
    """
    Return binmid, meds pour pouvoir faire : plt.errorbar(binmid, meds - 1*np.ones(binmid.size), yerr=meds_err, marker='.', linestyle='-', lw=0.9,)
    """
    ii = (pixmap['FRACAREA_13538'] < 1.1) & (pixmap['FRACAREA_13538'] > 0.9) & (pixmap[syscolname] >= downclip) & (pixmap[syscolname] < upclip)

    if np.any(ii):
        pixmapgood = pixmap[ii]
    else:
        print("Pixel map has no areas with >90% coverage for passed up/downclip")
        print("Proceeding without clipping systematics for {}".format(syscolname))
        ii = (pixmap['FRACAREA_13538'] < 1.1) & (pixmap['FRACAREA_13538'] > 0.9)
        pixmapgood = pixmap[ii]

    targets = targets[ii]

    # ADM set up the x-axis as the systematic of interest.
    xx = pixmapgood[syscolname]

    if adaptative_binning:
        nbr_obj_bins = nobjects_by_bins
        ksort = np.argsort(xx)
        bins=xx[ksort[0::nbr_obj_bins]] #Here, get the bins from the data set (needed to be sorted)
        bins=np.append(bins,xx[ksort[-1]]) #add last point
        nbins = bins.size - 1 #OK
        print(nbins)
    else:
        # ADM let np.histogram choose a sensible binning.
        nbr_obj_bins, bins = np.histogram(xx, nbins) #nbins

    # ADM the bin centers rather than the edges.
    binmid = np.nanmean(np.vstack([bins, np.roll(bins, 1)]), axis=0)[1:]

    # ADM set up the y-axis as the deviation of the target density from median density.
    yy = targets/np.nanmedian(targets)

    # ADM determine which bin each systematics value is in.
    wbin = np.digitize(xx, bins)

    # ADM np.digitize closes the end bin whereas np.histogram
    # ADM leaves it open, so shift the end bin value back by one.
    wbin[np.argmax(wbin)] -= 1

    # ADM apply the digitization to the target density values
    # ADM note that the first digitized bin is 1 not zero.
    meds = [np.nanmedian(yy[wbin == bin]) for bin in range(1, nbins+1)]

    # EC add error for mean (estimation of the std from sample)
    err_meds = [np.nanstd(yy[wbin == bin]) / np.sqrt((wbin == bin).sum() - 1) if ((wbin == bin).sum() > 1) else 0 for bin in range(1, nbins+1)]

    return bins, binmid, meds, nbr_obj_bins, err_meds
