## coding: utf-8
## Author : Edmond Chaussidon (CEA)
## Load information about DESI footprint and the different
## photometric footprints in function of the different tracer.

## Data/Legacy_Imaging_DR9_footprint_256.fits is built from the fracarea_12290 given by the pixweight
## file generated by desitarget. DES imprint is extracted from an old file of Anand.
## It is built in Target_Selection/Build_footprint_file/

import logging
logger = logging.getLogger("desi_footprint")
import os

import healpy as hp
import numpy as np
import fitsio

from desitarget.geomask import hp_in_box

class DR9_footprint(object):
    """
    Photometric footprint DR9
    Name: North = MzLS, South = DECaLZ (without DES), DES
    WARNING: ISSOUTH is everything with Dec. < 32.275
    """

    def __init__(self, Nside=256, remove_LMC=False, clear_south=False, mask_around_des=False):
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
        mask_around_des : bool
            Mask the border of the footprint around des which is contained in South --> usefull for systeamtic weights
        """
        self.Nside = Nside
        self.remove_LMC = remove_LMC
        self.clear_south = clear_south
        self.mask_around_des = mask_around_des
        logger.info(f'LOAD DR9 footprint with remove_LMC={self.remove_LMC} and clear_south={self.clear_south}')

        self.data = fitsio.read(os.path.join(os.path.dirname(__file__), '../Data/Legacy_Imaging_DR9_footprint_256.fits'))


    def update_map(self, pixmap):
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

        if self.mask_around_des:
            mask_around_des = np.zeros(hp.nside2npix(256), dtype=bool)
            mask_around_des[hp_in_box(256, [-120, 0, -90, -18.5], inclusive=True) + hp_in_box(256, [0, 120, -90, -17.4], inclusive=True)] = True
            mask_around_des[self.data['ISDES']] = False
            pixmap[mask_around_des] = False

        if self.Nside != 256:
            pixmap = hp.ud_grade(pixmap, self.Nside, order_in='NESTED')

        return pixmap


    def load_footprint(self):
        """
        Return dr9 footprint
        """
        return self.update_map(self.data['ISDR9'])


    def load_ngc_sgc(self):
        """
        Return NGC / SGC mask
        """
        return self.update_map(self.data['ISNGC']), self.update_map(self.data['ISSGC'])


    def load_photometry(self):
        """
        Return North / South / DES
        """
        return self.update_map(self.data['ISNORTH']), self.update_map(self.data['ISSOUTH'] & ~self.data['ISDES']), self.update_map(self.data['ISDES'])


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
            return self.update_map(self.data['ISNORTH']), self.update_map(south_mid & self.data['ISNGC']), self.update_map(south_mid & self.data['ISSGC']), self.update_map(south_pole)
        else:
            return self.update_map(self.data['ISNORTH']), self.update_map(south_mid), self.update_map(south_pole)
