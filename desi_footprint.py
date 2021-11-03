## coding: utf-8
## Author : Edmond Chaussidon (CEA)
## Load information about DESI footprint and the different
## photometric footprints in function of the different tracer.

## Data/Legacy_Imaging_DR9_footprint_256.fits is built from the fracarea_12290 given by the pixweight
## file generated by desitarget. DES imprint is extracted from an old file of Anand.
## It is built in Target_Selection/Build_footprint_file/

import os

import healpy as hp
import fitsio

from desitarget.geomask import hp_in_box

class DR9_footprint(object):
    """
    Photometric footprint DR9

    Name: North = MzLS, South = DECaLZ (without DES), DES

    WARNING: ISSOUTH is everything with Dec. < 32.275
    """

    def __init__(self, Nside=256, remove_LMC=False, clear_south=False):
        """
        Initialize :class:`DR9_footprint` with Nside=256 map

        Parameters
        ----------
        Nside : int
            Give the resolution of the output masks
        remove_LMC : bool
            Mask out the LMC --> useful for QSO TS
        clear_south : bool
            Mask out disconnected area in the NGC South --> useful to compute ACF

        """
        self.Nside = Nside
        self.remove_LMC = remove_LMC
        self.clear_south = clear_south

        self.data = fitsio.read(os.path.join(os.path.dirname(__file__), 'Data/Legacy_Imaging_DR9_footprint_256.fits'))


    def update_map(self, pixmap, mask_des=False):
        """
        Apply mask / ud_grade the pixmap before to return it

        Parameters
        ----------
        pixmap: pixmap to return with mask at the correct Nside
        mask_des: bool --> to remove pixels around DES when dealing with South footprint

        """
        if self.remove_LMC:
            pixmap[hp_in_box(256, [52, 120, -90, -50], inclusive=True)] = False

        if self.clear_south:
            pixmap[hp_in_box(256, [120, 150, -45, -10], inclusive=True) + hp_in_box(256, [150, 180, -45, -15], inclusive=True) + hp_in_box(256, [210, 240, -20, -12], inclusive=True)] = False

        if mask_des:
            pixmap[hp_in_box(256, [-120, 0, -90, -18.5], inclusive=True) + hp_in_box(256, [0, 120, -90, -17.4], inclusive=True)] = False

        if self.Nside != 256:
            pixmap = hp.ud_grade(pixmap, Nside, order_in='NESTED')

        return pixmap


    def load_ngc_sgc(self):
        """
        Return NGC / SGC mask

        """
        return self.update_map(self.data['NGC']), self.update_map(self.data['SGC'])


    def load_photometry(self, remove_around_des=False):
        """
        Return North / South / DES

        Parameters
        ----------
        remove_around_des: bool
            If True, remove the band in the South around DES, useful for clustering for instance

        """
        return self.update_map(self.data['ISNORTH']), self.update_map(self.data['ISSOUTH'] & ~self.data['ISDES'], mask_des=remove_around_des), self.update_map(self.data['ISDES'])


    def load_elg_region(self, ngc_sgc_split=False):
        """
        Return North / South & DES ( -30 < Dec < 32.275) / DES (Dec. < -30)

        """

        dec, all_south = self.data['DEC'], self.data['ISSOUTH']

        south_mid = all_south.copy()
        south_mid[dec <= -30] = False

        south_pole = all_south.copy()
        south_pole[dec > -30] = False

        if ngc_sgc_split:
            return self.update_map(self.data['ISNORTH']), self.update_map(south_mid & self.data['NGC']), self.update_map(south_mid & self.data['SGC']), self.update_map(south_pole)
        else:
            return self.update_map(self.data['ISNORTH']), self.update_map(south_mid), self.update_map(south_pole)
