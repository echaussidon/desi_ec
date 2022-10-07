import numpy as np
import matplotlib.pyplot as plt
plt.style.use('~/Software/desi_ec/ec_style.mplstyle')

import fitsio
import h5py

from scipy.ndimage.filters import gaussian_filter
from desispec.coaddition import coadd_cameras
from desispec.io import read_spectra

#load redrock template
import redrock.templates
templates = dict()
for filename in redrock.templates.find_templates():
    t = redrock.templates.Template(filename)
    templates[(t.template_type, t.sub_type)] = t

    
# ATTENTION: Il y a eu un changement de nom de fichier... ils n'ont pas été mis à jour dans le daily ! c'est pouruqoi cette fonction ne marche pas avec les premieres tiles de daily.
# pas de probleme car les noms ont été correctement mise à jour à partir de fuji
# PATH_TILES = "/global/cfs/cdirs/desi/spectro/redux/fuji/tiles/cumulative"
# PATH_TILES = "/global/cfs/cdirs/desi/spectro/redux/guadalupe/tiles/cumulative"
PATH_TILES = "/global/cfs/cdirs/desi/spectro/redux/daily/tiles/cumulative"


def get_spectra(spectra_name, targetid):
    spectra = read_spectra(spectra_name)
    spectra = spectra.select(targets=[targetid])

    if 'brz' not in spectra.bands:
        spectra = coadd_cameras(spectra)

    return spectra.wave['brz'], spectra.flux['brz'].T, spectra.ivar['brz'].T


def compute_RR_from_file(zbest_name, redrock_name, index):
    zbest = fitsio.FITS(zbest_name)[1]
    redrock_fit = h5py.File(redrock_name, 'r')
    
    # extract info for the best fit model
    z = zbest['Z'][index]
    spectype = zbest['SPECTYPE'][index].strip()
    subtype = zbest['SUBTYPE'][index].strip()
    fulltype = (spectype, subtype)

    # extract coefficient
    ncoeff = templates[fulltype].flux.shape[0]
    coeff = zbest['COEFF'][index][0:ncoeff]

    # compute best fit model
    flux_fit = templates[fulltype].flux.T.dot(coeff)
    wavelength_fit = templates[fulltype].wave * (1+z)
    
    return wavelength_fit, flux_fit, z, spectype


def compute_RR_from_param(param_fit):
    fulltype = (param_fit['spectype'], param_fit['subtype'])
    ncoeff = templates[fulltype].flux.shape[0]
    coeffs = param_fit['coeffs'][0:ncoeff]

    flux_fit = templates[fulltype].flux.T.dot(coeffs)
    wavelength_fit = templates[fulltype].wave * (1+param_fit['z'])

    return wavelength_fit, flux_fit, param_fit['z'], param_fit['spectype']


lines = {
        'Ha'      : 6562.8,
        'Hb'      : 4862.68,
        'Hg'      : 4340.464,
        'Hd'      : 4101.734,
        'OIII-b'  :  5006.843,
        'OIII-a'  : 4958.911,
        'MgII'    : 2799.49,
        'OII'     : 3728,
        'CIII'    : 1909.,
        'CIV'     : 1549.06,
        'SiIV'    : 1393.76018,
        'LYA'     : 1215.67, 
        'LYB'     : 1025.72}

def plot_lines(ax, lines, z):
    for elem in lines :
        line=(1+z)*lines[elem]
        if line > ax.get_xlim()[0] and line < ax.get_xlim()[1]:
            ax.axvline(line, color="red", linestyle="--",alpha=0.4)
            ax.text((line+60), ax.get_ylim()[1]*0.9, elem.split("-")[0], color="red")


# example param_fit
#param_fit = {'spectype':'QSO', 'subtype':'', 'z':1.6, 'coeffs':np.array([1.0, 1.0, 0., 0., 0., 0., 0., 0., 0., 0.])}

# PATH_TILES = "/global/cfs/cdirs/desi/spectro/redux/fuji/tiles/cumulative"
# PATH_TILES = "/global/cfs/cdirs/desi/spectro/redux/guadalupe/tiles/cumulative"  
# PATH_TILES = "/global/cfs/cdirs/desi/spectro/redux/daily/tiles/cumulative"

def plot_spectrum(tile, night, petal, targetid, path_to_tiles=PATH_TILES, 
                  show_unsmooth=True, show_smooth=True, show_noise=True, show_fit_RR=True, show_info=True,
                  show_line=True, z_line=None, param_fit=None,
                  legend_loc='upper right', 
                  ax=None, show=True, savename=None, gaussian_smoothing_plot=5):

    spectra_name = f'{path_to_tiles}/{tile}/{night}/coadd-{petal}-{tile}-thru{night}.fits'
    zbest_name = f'{path_to_tiles}/{tile}/{night}/redrock-{petal}-{tile}-thru{night}.fits'
    redrock_name = f'{path_to_tiles}/{tile}/{night}/rrdetails-{petal}-{tile}-thru{night}.h5'

    wavelength, flux, ivar_flux = get_spectra(spectra_name, targetid)
    flux_smooth = gaussian_filter(flux, gaussian_smoothing_plot)

    zbest = fitsio.FITS(zbest_name)[1]

    index = np.where(zbest['TARGETID'][:] == targetid)[0][0]

    if ax is None:
        plt.figure(figsize=(10, 3))
        ax = plt.gca()
        
    if show_unsmooth:
        ax.plot(wavelength, flux, lw=1., color='grey', alpha=0.5)
    if show_smooth:
        ax.plot(wavelength, flux_smooth, lw=1., color='black', label=f'Spectrum')
    if show_noise:
        ax.plot(wavelength, np.sqrt(1/ivar_flux), color='darkorange', lw=1., alpha=0.5, label='Noise')
    ax.set_ylim(min(0, np.min(flux_smooth)), np.max(flux_smooth)*1.1)
    ax.set_xlim(3500, 9900)

    if show_fit_RR:
        wavelength_fit, flux_fit, z, spectype = compute_RR_from_file(zbest_name, redrock_name, index)
        ax.plot(wavelength_fit, flux_fit, color='blue', ls='-', lw=1., label=f'{spectype} at z: {z:1.3f}')
        if show_line:
            plot_lines(ax, lines, z)
    
    if not (param_fit is None):
        wavelength_fit, flux_fit, z, spectype = compute_RR_from_param(param_fit)
        ax.plot(wavelength_fit, flux_fit, color='dodgerblue', ls='-', lw=1., label=f'{spectype} at z: {z:1.3f}')
        if show_line:
            plot_lines(ax, lines, z)
            
    if (not show_fit_RR) and (param_fit is None):
        if z_line is None:
            print("ERROR set z_line value")
        else:
            plot_lines(ax, lines, z_line)

    ax.legend(loc=legend_loc, fontsize=12)
    ax.set_xlabel("$\lambda$ [$\AA$]", fontsize=12)
    ax.set_ylabel('Flux [$10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', fontsize=12)
    
    if show_info:
        ax.set_title(f'Tile-Night-Petal: {tile}-{night}-{petal} -- Target ID: {targetid}')
    plt.tight_layout()
    
    if not savename is None:
        plt.savefig(savename)
        if not show:
            plt.close()
            
    if show:
        plt.show()

if __name__ == "__main__":

    tile = 1
    night = 20210406
    petal = 4

    targetid = 39627841599443997

    plot_spectrum(tile, night, petal, targetid, show=False, savename=f'Res/{tile}-{night}-{petal}-{targetid}.pdf')
