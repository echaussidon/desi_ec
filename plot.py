# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import astropy.units as u
from astropy.coordinates import SkyCoord

import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose

#----------------------------------------------------------------------------------------------------#
# Quelques fonctions utiles pour les dessins

def to_tex(string):
    # pour convertir une chaine en chaine de texte affichable par latex dans matplotlib
    # on enleve les _
    string = string.replace('_', ' ')
    return string

def mean_on_healpy_map(Nside, map, depth_neighbours=1): #supposed Nested and map a list of pixel
    def get_all_neighbours(Nside, i, depth_neighbours=1):
        pixel_list = hp.get_all_neighbours(Nside, i, nest=True)
        pixel_tmp = pixel_list
        depth_neighbours -= 1
        while depth_neighbours != 0 :
            pixel_tmp = hp.get_all_neighbours(Nside, pixel_tmp, nest=True)
            pixel_tmp = np.reshape(pixel_tmp, pixel_tmp.size)
            pixel_list = np.append(pixel_list, pixel_tmp)
            depth_neighbours -= 1
        return pixel_list

    print("[WARNING :] We use mean_on_healpy_map")
    mean_map = np.zeros(len(map))
    for i in range(len(map)):
        neighbour_pixels = get_all_neighbours(Nside, i, depth_neighbours)
        mean_map[i] = np.nansum(map[neighbour_pixels], axis=0)/neighbour_pixels.size
    return mean_map

#----------------------------------------------------------------------------------------------------#
# On definit ici les differents plans utiles pour les dessins
# on a copie la class sagittarius de astropy --> cf internet pour la doc ect

class Sagittarius(coord.BaseCoordinateFrame):
    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'Lambda'),
            coord.RepresentationMapping('lat', 'Beta'),
            coord.RepresentationMapping('distance', 'distance')]
        }

SGR_PHI = (180 + 3.75) * u.degree # Euler angles (from Law & Majewski 2010)
SGR_THETA = (90 - 13.46) * u.degree
SGR_PSI = (180 + 14.111534) * u.degree

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(SGR_PHI, "z")
C = rotation_matrix(SGR_THETA, "x")
B = rotation_matrix(SGR_PSI, "z")
A = np.diag([1.,1.,-1.])
SGR_MATRIX = matrix_product(A, B, C, D)

@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Sagittarius)
def galactic_to_sgr():
    """ Compute the transformation matrix from Galactic spherical to
        heliocentric Sgr coordinates.
    """
    return SGR_MATRIX

@frame_transform_graph.transform(coord.StaticMatrixTransform, Sagittarius, coord.Galactic)
def sgr_to_galactic():
    """ Compute the transformation matrix from heliocentric Sgr coordinates to
        spherical Galactic.
    """
    return matrix_transpose(SGR_MATRIX)


# On definit les plans ici :

galactic_plane = SkyCoord(l=np.linspace(0, 2*np.pi, 200)*u.radian, b=np.zeros(200)*u.radian, frame='galactic', distance=1*u.Mpc)
galactic_plane_icrs = galactic_plane.transform_to('icrs')
index_galactic = np.argsort(galactic_plane_icrs.ra.wrap_at(300*u.deg).degree)

ecliptic_plane = SkyCoord(lon=np.linspace(0, 2*np.pi, 200)*u.radian, lat=np.zeros(200)*u.radian, distance=1*u.Mpc, frame='heliocentrictrueecliptic')
ecliptic_plane_icrs = ecliptic_plane.transform_to('icrs')
index_ecliptic = np.argsort(ecliptic_plane_icrs.ra.wrap_at(300*u.deg).degree)

sgr_plane = Sagittarius(Lambda=np.linspace(0, 2*np.pi, 200)*u.radian, Beta=np.zeros(200)*u.radian, distance=1*u.Mpc)
sgr_plane_icrs = sgr_plane.transform_to(coord.ICRS)
index_sgr = np.argsort(sgr_plane_icrs.ra.wrap_at(300*u.deg).degree)


#-----------------------------------------------------------------------------------------------------#

def plot_cart(map, min=None, max=None, title='', label=r'[$\#$ $deg^{-2}$]', savename=None, show=True, galactic_plane=False, ecliptic_plane=False):
    #attention au sens de l'axe en RA ! --> la on le prend normal et on le retourne à la fin :)
    plt.figure(1)
    m = hp.ma(map)
    map_to_plot = hp.cartview(m, nest=True, flip='geo', rot=120, fig=1, return_projected_map=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(11,7))
    map_plotted = plt.imshow(map_to_plot, vmin=min, vmax=max, cmap='jet', origin='lower', extent=[-60, 300, -90, 90])
    if label!=None:
        cb = plt.colorbar(map_plotted, ax=ax, orientation='horizontal', shrink=0.8, aspect=40)
        cb.set_label(label)
    if galactic_plane:
        ax.plot(galactic_plane_icrs.ra.wrap_at(300*u.deg).degree[index_galactic], galactic_plane_icrs.dec.degree[index_galactic], linestyle='-', color='black', label='Galactic plane')
    if ecliptic_plane:
        ax.plot(ecliptic_plane_icrs.ra.wrap_at(300*u.deg).degree[index_ecliptic], ecliptic_plane_icrs.dec.degree[index_ecliptic], linestyle=':', color='slategrey', label='Ecliptic plane')
    if sgr_plane:
        ax.plot(sgr_plane_icrs.ra.wrap_at(300*u.deg).degree[index_sgr], sgr_plane_icrs.dec.degree[index_sgr], linestyle='--', color='navy', label='Sgr. plane')

    ax.set_xlim(-60, 300)
    ax.xaxis.set_ticks(np.arange(-60, 330, 30))
    plt.gca().invert_xaxis()
    ax.set_xlabel('R.A. [deg]')
    ax.set_ylim(-90, 90)
    ax.yaxis.set_ticks(np.arange(-90, 120, 30))
    ax.set_ylabel('Dec. [deg]')

    if galactic_plane or ecliptic_plane:
        ax.legend(loc='lower right')
    if title!='':
        plt.title(title)
    if savename != None:
        plt.savefig(savename)
    if show:
        plt.show()
    else:
        plt.close()

def plot_moll(map, min=None, max=None, title='', label=r'[$\#$ deg$^{-2}$]', savename=None, show=True, galactic_plane=False, ecliptic_plane=False, sgr_plane=False, show_legend=True, rot=120, projection='mollweide', figsize=(11.0, 7.0), xpad=1.25, labelpad=-37, ycb_pos=-0.15):

    #transform healpix map to 2d array
    plt.figure(1)
    m = hp.ma(map)
    map_to_plot = hp.cartview(m, nest=True, rot=rot, flip='geo', fig=1, return_projected_map=True)
    plt.close()

    #build ra, dec meshgrid to plot 2d array
    ra_edge = np.linspace(-180, 180, map_to_plot.shape[1]+1)
    dec_edge = np.linspace(-90, 90, map_to_plot.shape[0]+1)

    ra_edge[ra_edge>180] -=360    # scale conversion to [-180, 180]
    ra_edge=-ra_edge              # reverse the scale: East to the left

    ra_grid, dec_grid = np.meshgrid(ra_edge, dec_edge)

    plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection=projection)
    plt.subplots_adjust(left=0.14, bottom=0.23, right=0.96, top=0.96)

    mesh = plt.pcolormesh(np.radians(ra_grid), np.radians(dec_grid), map_to_plot, vmin=min, vmax=max, cmap='jet', edgecolor='none', lw=0)

    if label!=None:
        ax_cb = inset_axes(ax, width="30%", height="4%", loc='lower left', bbox_to_anchor=(0.346, ycb_pos, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0)
        cb = plt.colorbar(mesh, ax=ax, cax=ax_cb, orientation='horizontal', shrink=0.8, aspect=40)
        cb.outline.set_visible(False)
        cb.set_label(label, x=xpad, labelpad=labelpad)

    if galactic_plane:
        ra, dec = galactic_plane_icrs.ra.degree - rot, galactic_plane_icrs.dec.degree
        ra[ra>180] -=360    # scale conversion to [-180, 180]
        ra=-ra              # reverse the scale: East to the left
        ax.plot(np.radians(ra[index_galactic]), np.radians(dec[index_galactic]), linestyle='-', linewidth=0.8, color='black', label='Galactic plane')
    if ecliptic_plane:
        ra, dec = ecliptic_plane_icrs.ra.degree - rot, ecliptic_plane_icrs.dec.degree
        ra[ra>180] -=360    # scale conversion to [-180, 180]
        ra=-ra              # reverse the scale: East to the left
        ax.plot(np.radians(ra[index_ecliptic]), np.radians(dec[index_ecliptic]), linestyle=':', linewidth=0.8, color='navy', label='Ecliptic plane')
    if sgr_plane:
        ra, dec = sgr_plane_icrs.ra.degree - rot, sgr_plane_icrs.dec.degree
        ra[ra>180] -=360    # scale conversion to [-180, 180]
        ra=-ra              # reverse the scale: East to the left
        ax.plot(np.radians(ra[index_sgr]), np.radians(dec[index_sgr]), linestyle=':', linewidth=0.8, color='navy', label='Sgr. plane')

    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + rot, 360)
    tick_labels = np.array(['{0}°'.format(l) for l in tick_labels])
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel('R.A. [deg]')
    ax.set_ylabel('Dec. [deg]')

    ax.grid(True)

    if show_legend:
        ax.legend(loc='lower right')
    if title!='':
        plt.title(title)
    if savename != None:
        plt.savefig(savename)
    if show:
        plt.show()
    else:
        plt.close()
