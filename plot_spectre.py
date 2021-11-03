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

def plot_spectrum(tile, night, petal, targetid, path_to_tiles=PATH_TILES, 
                  show_unsmooth=True, show_smooth=True, show_noise=True, show_fit_RR=True, show_info=True,
                  show_line=True, z_line=None, param_fit=None,
                  legend_loc='upper right', 
                  ax=None, show=True, savename=None, gaussian_smoothing_plot=5):

    spectra_name = f'{path_to_tiles}/{tile}/{night}/coadd-{petal}-{tile}-thru{night}.fits'
    zbest_name = f'{path_to_tiles}/{tile}/{night}/zbest-{petal}-{tile}-thru{night}.fits'
    redrock_name = f'{path_to_tiles}/{tile}/{night}/redrock-{petal}-{tile}-thru{night}.h5'

    wavelength, flux, ivar_flux = get_spectra(spectra_name, targetid)
    flux_smooth = gaussian_filter(flux, gaussian_smoothing_plot)

    zbest = fitsio.FITS(zbest_name)[1]

    index = np.where(zbest['TARGETID'][:] == targetid)[0][0]

    plt.figure(figsize=(10, 3))
    if show_unsmooth:
        plt.plot(wavelength, flux, lw=1., color='grey', alpha=0.5)
    if show_smooth:
        plt.plot(wavelength, flux_smooth, lw=1., color='black', label=f'Spectrum')
    if show_noise:
        plt.plot(wavelength, np.sqrt(1/ivar_flux), color='darkorange', lw=1., alpha=0.5, label='Noise')
    plt.ylim(min(0, np.min(flux_smooth)), np.max(flux_smooth)*1.1)
    plt.xlim(3500, 9900)

    if show_fit_RR:
        wavelength_fit, flux_fit, z, spectype = compute_RR_from_file(zbest_name, redrock_name, index)
        plt.plot(wavelength_fit, flux_fit, color='blue', ls='-', lw=1., label=f'{spectype} at z: {z:1.3f}')
        if show_line:
            plot_lines(plt.gca(), lines, z)
    
    if not (param_fit is None):
        wavelength_fit, flux_fit, z, spectype = compute_RR_from_param(param_fit)
        plt.plot(wavelength_fit, flux_fit, color='dodgerblue', ls='-', lw=1., label=f'{spectype} at z: {z:1.3f}')
        if show_line:
            plot_lines(plt.gca(), lines, z)
            
    if (not show_fit_RR) and (param_fit is None):
        if z_line is None:
            print("ERROR set z_line value")
        else:
            plot_lines(plt.gca(), lines, z_line)

    plt.legend(loc=legend_loc, fontsize=12)
    plt.xlabel("$\lambda$ [$\AA$]", fontsize=12)
    plt.ylabel('Flux [$10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', fontsize=12)
    
    if show_info:
        plt.title(f'Tile-Night-Petal: {tile}-{night}-{petal} -- Target ID: {targetid}')
    plt.tight_layout()
    
    if not savename is None:
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


## IDEE A REPREDNRE NOTAMMENT POUR LES LIGNES

# #!/usr/bin/env python
#
#
# import sys
# import argparse
# import matplotlib.pyplot as plt
# import numpy as np
# import fitsio
# from desispec.io import read_spectra
# from desispec.interpolation import resample_flux
# from astropy.table import Table
# import redrock.templates
#
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#                                  description="Display spectra, looping over targets if targetid not set, and optionally show best fit from redrock"
# )
# parser.add_argument('-i','--infile', type = str, default = None, required = True, nargs="*",
#                     help = 'path to spectra file(s)')
# parser.add_argument('-t','--targetid', type = int, default = None, required = False,
#                     help = 'plot specific targetid')
# parser.add_argument('--rebin',type = int, default = None, required = False,
#                     help = 'rebin')
# parser.add_argument('--zbest',type = str, default = None, required = False,
#                     help = 'zbest file')
# parser.add_argument('--spectype',type = str, default = None, required = False,
#                     help = 'spectype to select')
# parser.add_argument('--ylim', type=float, default=None, required=False, nargs=2,
#                     help = 'ymin ymax for plot')
# parser.add_argument('--title', type=str, default=None, required=False,
#                     help = 'plot title')
# parser.add_argument('--rest-frame', action='store_true',
#                     help = 'show rest-frame wavelength')
# parser.add_argument('--errors', action='store_true',
#                     help = 'show error bars')
# parser.add_argument('--only-valid', action='store_true',
#                     help = 'show error bars')
#
#
# args        = parser.parse_args()
#
# if  args.zbest is not None :
#     #- Load redrock templates
#     templates = dict()
#     for filename in redrock.templates.find_templates():
#         tx = redrock.templates.Template(filename)
#         templates[(tx.template_type, tx.sub_type)] = tx
#
# if args.zbest is None and args.rest_frame :
#     args.rest_frame = False
#     print("cannot show rest-frame wavelength without a zbest file")
#
# spectra = []
# for filename in args.infile :
#     spec=read_spectra(filename)
#     if args.only_valid :
#         good=(spec.fibermap["FIBERSTATUS"]==0)
#         spec=spec[good]
#     spectra.append(spec)
#
# targetids=None
# if ( targetids is None ) and ( args.targetid is not None ) :
#     targetids=[args.targetid,]
#
# zbest=None
# if  args.zbest is not None :
#     zbest=Table.read(args.zbest,"ZBEST")
#
# if ( targetids is None ) and ( zbest is not None ) and ( args.spectype is not None ):
#     selection = np.where((zbest["SPECTYPE"]==args.spectype)&(zbest["ZWARN"]==0))[0]
#     targetids=np.unique(spectra[0].fibermap["TARGETID"][selection])
#
# if targetids is None :
#     targetids=np.unique(spectra[0].fibermap["TARGETID"])
#
#
# lines = {
#     'Ha'      : 6562.8,
#     'Hb'       : 4862.68,
#     'Hg'       : 4340.464,
#     'Hd'       : 4101.734,
#     'OIII-b'       :  5006.843,
#     'OIII-a'       : 4958.911,
#     'MgII'    : 2799.49,
#     'OII'         : 3728,
#     'CIII'  : 1909.,
#     'CIV'    : 1549.06,
#     'SiIV'  : 1393.76018,
#     'LYA'         : 1215.67,
#     'LYB'         : 1025.72
# }
#
#
#
# for tid in targetids :
#     line="TARGETID={}".format(tid)
#
#     model_flux=dict()
#     if zbest is not None :
#         j=np.where(zbest["TARGETID"]==tid)[0][0]
#         line += " ZBEST={} SPECTYPE={} ZWARN={}".format(zbest["Z"][j],zbest["SPECTYPE"][j],zbest["ZWARN"][j])
#         zval=zbest["Z"][j]
#         tx = templates[(zbest['SPECTYPE'][j], zbest['SUBTYPE'][j])]
#         for band in spectra[0].bands:
#             model_flux[band] = np.zeros(spectra[0].wave[band].shape)
#             coeff = zbest['COEFF'][j][0:tx.nbasis]
#             model = tx.flux.T.dot(coeff).T
#             mx = resample_flux(spectra[0].wave[band], tx.wave*(1+zbest['Z'][j]), model)
#             k=np.where(spectra[0].fibermap["TARGETID"]==tid)[0][0]
#             model_flux[band] = spectra[0].R[band][k].dot(mx)
#
#     fig=plt.figure(figsize=[10,6])
#     ax = fig.add_subplot(111)
#     print(line)
#     for spec in spectra :
#         jj=np.where(spec.fibermap["TARGETID"]==tid)[0]
#         fiber=spec.fibermap["FIBER"][jj[0]]
#
#         wavescale=1.
#         if args.rest_frame :
#             wavescale = 1./(1+zval)
#
#         for j in jj :
#             for b in spec._bands :
#
#                 i=np.where(spec.ivar[b][j]*(spec.mask[b][j]==0)>1./100.**2)[0]
#                 if i.size<10 : continue
#                 if args.rebin is not None and args.rebin>0:
#                     rwave=np.linspace(spec.wave[b][0],spec.wave[b][-1],spec.wave[b].size//args.rebin)
#                     rflux,rivar = resample_flux(rwave,spec.wave[b],spec.flux[b][j],ivar=spec.ivar[b][j]*(spec.mask[b][j]==0))
#                     plt.plot(wavescale*rwave,rflux)
#                 else :
#                     if args.errors :
#                         plt.errorbar(wavescale*spec.wave[b][i],spec.flux[b][j,i],1./np.sqrt(spec.ivar[b][j,i]))
#                     else :
#                         plt.plot(wavescale*spec.wave[b][i],spec.flux[b][j,i])
#
#                 c=np.polyfit(spec.wave[b][i],spec.flux[b][j,i],3)
#                 pol=np.poly1d(c)(spec.wave[b][i])
#
#         print(spec.fibermap[jj])
#
#     if zbest is not None :
#         for band in spectra[0].bands:
#             plt.plot(wavescale*spectra[0].wave[band],model_flux[band],"-",alpha=0.6)
#             for elem in lines :
#                 line=(1+zval)*lines[elem]
#                 if line>spectra[0].wave[band][0] and line<spectra[0].wave[band][-1] :
#                     plt.axvline(wavescale*line,color="red",linestyle="--",alpha=0.4)
#                     y=np.interp(wavescale*line,wavescale*spectra[0].wave[band],model_flux[band])
#                     plt.text(wavescale*(line+60),y*1.1,elem.split("-")[0],color="red")
#     if args.rest_frame :
#         plt.xlabel("rest-frame wavelength [A]")
#     else :
#         plt.xlabel("wavelength [A]")
#     plt.grid()
#     props = dict(boxstyle='round', facecolor='yellow', alpha=0.2)
#     bla="TID = {}".format(tid)
#     bla+="\nFIBER = {}".format(fiber)
#     if zbest is not None :
#         bla+="\nZ  = {:4.3f}".format(zval)
#     plt.text(0.9,0.9,bla,fontsize=12, bbox=props,transform = ax.transAxes,verticalalignment='top', horizontalalignment='right')
#     if args.ylim is not None:
#         plt.ylim(args.ylim[0], args.ylim[1])
#
#     if args.title is not None:
#         plt.title(args.title)
#
#     plt.tight_layout()
#     plt.show()
#
#
# plt.show()
