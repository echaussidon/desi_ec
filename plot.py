# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

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

galactic_plane = SkyCoord(l=np.linspace(0, 2 * np.pi, 200)*u.radian, b=np.zeros(200)*u.radian, frame='galactic', distance=1 * u.Mpc)
galactic_plane_icrs = galactic_plane.transform_to('icrs')
index_galactic = np.argsort(galactic_plane_icrs.ra.wrap_at(300*u.deg).degree)

ecliptic_plane = SkyCoord(lon=np.linspace(0, 2 * np.pi, 200)*u.radian, lat=np.zeros(200)*u.radian, distance=1 * u.Mpc, frame='heliocentrictrueecliptic')
ecliptic_plane_icrs = ecliptic_plane.transform_to('icrs')
index_ecliptic = np.argsort(ecliptic_plane_icrs.ra.wrap_at(300*u.deg).degree)

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
        ax.plot(ecliptic_plane_icrs.ra.wrap_at(300*u.deg).degree[index_ecliptic], ecliptic_plane_icrs.dec.degree[index_ecliptic], linestyle=':', color='navy', label='Ecliptic plane')
        
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
        
def plot_moll(map, min=None, max=None, title='', label=r'[$\#$ $deg^{-2}$]', savename=None, show=True, galactic_plane=False, ecliptic_plane=False, rot=120):
    
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

    plt.figure(figsize=(11,7))
    ax = plt.subplot(111, projection='mollweide')
    
    mesh = plt.pcolormesh(np.radians(ra_grid), np.radians(dec_grid), map_to_plot, vmin=min, vmax=max, cmap='jet', edgecolor='none', lw=0)   
    if label!=None:
        cb = plt.colorbar(mesh, ax=ax, orientation='horizontal', shrink=0.8, aspect=40)
        cb.set_label(label)
    if galactic_plane:
        ra, dec = galactic_plane_icrs.ra.degree - rot, galactic_plane_icrs.dec.degree
        ra[ra>180] -=360    # scale conversion to [-180, 180]
        ra=-ra              # reverse the scale: East to the left
        ax.plot(np.radians(ra[index_galactic]), np.radians(dec[index_galactic]), linestyle='-', color='black', label='Galactic plane')
    if ecliptic_plane:
        ra, dec = ecliptic_plane_icrs.ra.degree - rot, ecliptic_plane_icrs.dec.degree
        ra[ra>180] -=360    # scale conversion to [-180, 180]
        ra=-ra              # reverse the scale: East to the left
        ax.plot(np.radians(ra[index_ecliptic]), np.radians(dec[index_ecliptic]), linestyle=':', color='navy', label='Ecliptic plane')

    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + rot, 360)
    tick_labels = np.array(['{0}°'.format(l) for l in tick_labels])
    ax.set_xticklabels(tick_labels) 

    ax.set_xlabel('R.A. [deg]')
    ax.set_ylabel('Dec. [deg]')
    
    ax.grid(True)
    
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
