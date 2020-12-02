# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

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

def plot_cart(map, min=None, max=None, title='', label=r'[$\#$ $deg^{-2}$]', savename=None, show=False):

    #attention au sens de l'axe en RA ! --> la on le prend normal et on le retourne à la fin :)
    plt.figure(1)
    m = hp.ma(map)
    map_to_plot = hp.cartview(m, nest=True, flip='geo', rot=120, fig=1, return_projected_map=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(10,8))
    map_plotted = plt.imshow(map_to_plot, vmin=min, vmax=max, cmap='jet', origin='lower', extent=[-60, 300, -90, 90])
    if label!=None:
        cb = plt.colorbar(map_plotted, ax=ax, orientation='horizontal', shrink=0.8, aspect=40)
        cb.set_label(label)
    ax.set_xlim(-60, 300)
    ax.xaxis.set_ticks(np.arange(-60, 330, 30))
    plt.gca().invert_xaxis()
    ax.set_xlabel('R.A. [deg]')
    ax.set_ylim(-90, 90)
    ax.yaxis.set_ticks(np.arange(-90, 120, 30))
    ax.set_ylabel('Dec. [deg]')
    plt.title(title)

    if savename != None:
        plt.savefig(savename)
    if show:
        plt.show()
    else:
        plt.close()
