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

PATH_TILES = "/global/cfs/cdirs/desi/spectro/redux/daily/tiles/cumulative"

def get_spectra(spectra_name, targetid):
    spectra = read_spectra(spectra_name)
    spectra = spectra.select(targets=[targetid])

    if 'brz' not in spectra.bands:
        spectra = coadd_cameras(spectra)

    return spectra.wave['brz'], spectra.flux['brz'].T, spectra.ivar['brz'].T


def compute_RR_fit(spectype, subtype, coeffs, z):
    fulltype = (spectype, subtype)
    #extract coefficient
    ncoeff = templates[fulltype].flux.shape[0]
    coeffs = coeffs[0:ncoeff]

    flux_fit = templates[fulltype].flux.T.dot(coeffs)
    wavelength_fit = templates[fulltype].wave * (1+z)

    return wavelength_fit, flux_fit, z


def plot_spectrum(tile, night, petal, targetid, spectype=None, subtype=None, z=None, coeffs=None, path_to_tiles=PATH_TILES, ax=None,
                show=True, savename='oups.pdf',
                gaussian_smoothing_plot=5):

    spectra_name = f'{path_to_tiles}/{tile}/{night}/coadd-{petal}-{tile}-thru{night}.fits'
    zbest_name = f'{path_to_tiles}/{tile}/{night}/zbest-{petal}-{tile}-thru{night}.fits'

    wavelength, flux, ivar_flux = get_spectra(spectra_name, targetid)
    flux_smooth = gaussian_filter(flux, gaussian_smoothing_plot)

    zbest = fitsio.FITS(zbest_name)[1]

    index = np.where(zbest['TARGETID'][:] == targetid)[0][0]

    plt.figure(figsize=(14, 4))
    plt.plot(wavelength, flux_smooth, lw=1.5, color='blue', label=f'Target ID: {targetid}')
    plt.plot(wavelength, np.sqrt(1/ivar_flux), color='grey', lw=1.5, alpha=0.5)

    if (spectype != None):
        wavelength_fit, flux_fit, z = compute_RR_fit(spectype, subtype, coeffs, z)
        plt.plot(wavelength_fit, flux_fit, color='orange', lw=1.5, label=f'{spectype} at z: {z:1.3f}')

    plt.ylim(min(0, np.min(flux_smooth)), np.max(flux_smooth)*1.1)

    plt.legend(loc='upper right', fontsize=12)
    plt.xlim(3500, 9900)
    plt.xlabel("$\lambda$ [$\AA$]", fontsize=12)
    plt.ylabel('Flux [$10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', fontsize=12)
    plt.tight_layout()

    if savename != 'oups.pdf':
        plt.savefig(savename)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":

    tile = 1
    night = 20210406
    petal = 4

    targetid = 39627841599443997

    plot_spectra(tile, night, petal, targetid, show=False, savename=f'Res/{tile}-{night}-{petal}-{targetid}.pdf')
