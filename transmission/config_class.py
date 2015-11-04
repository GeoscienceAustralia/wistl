#!/usr/bin/env python
__author__ = 'Sudipta Basak'
import os


class TransmissionConfig(object):
    """
    class to hold all configuration variables.
    """
    def __init__(self):
        self.pdir = os.getcwd()
        self.shape_file_tower = os.path.join(
            self.pdir, 'Shapefile_2015_01/Towers_with_extra_strainers_WGS84.shp')
        self.shape_file_line = os.path.join(
            self.pdir, 'Shapefile_2015_01/Lines_NGCP_with_synthetic_attributes_WGS84.shp')

        self.file_frag = os.path.join(self.pdir, 'input/fragility_GA.csv')
        self.file_cond_pc = os.path.join(self.pdir, 'input/cond_collapse_prob_NGCP.csv')
        self.file_terrain_height = os.path.join(self.pdir, 'input/terrain_height_multiplier.csv')
        self.flag_strainer = ['Strainer', 'dummy'] # consider strainer

        self.file_design_value = os.path.join(self.pdir, 'input/design_value_current.csv')
        #file_topo_value = os.path.join(pdir, 'input/topo_value_scenario_50yr.csv')
        self.file_topo_value = None

        self.dir_wind_timeseries = os.path.join(self.pdir, 'glenda_reduced')
        self.dir_output = os.path.join(self.pdir, 'output_current_glenda')

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # number of simulations for MC
        self.nsims = 20

        # flag for save
        self.flag_save = 1


class TestTransmissionConfig(object):
    """
    class to hold all configuration variables.
    """
    def __init__(self):
        self.pdir = os.getcwd()
        self.shape_file_tower = os.path.join(
            self.pdir, 'Shapefile_2015_01/Towers_with_extra_strainers_WGS84.shp')
        self.shape_file_line = os.path.join(
            self.pdir, 'Shapefile_2015_01/Lines_NGCP_with_synthetic_attributes_WGS84.shp')

        self.file_frag = os.path.join(self.pdir, 'input/fragility_GA.csv')
        self.file_cond_pc = os.path.join(self.pdir, 'input/cond_collapse_prob_NGCP.csv')
        self.file_terrain_height = os.path.join(self.pdir, 'input/terrain_height_multiplier.csv')
        self.flag_strainer = ['Strainer', 'dummy'] # consider strainer

        self.file_design_value = os.path.join(self.pdir, 'input/design_value_current.csv')
        #file_topo_value = os.path.join(pdir, 'input/topo_value_scenario_50yr.csv')
        self.file_topo_value = None

        self.dir_wind_timeseries = os.path.join(self.pdir, 'glenda_reduced')
        self.dir_output = os.path.join(self.pdir, 'transmission', 'tests', 'test_output_current_glenda')

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # number of simulations for MC
        self.nsims = 20

        # flag for save
        self.flag_save = 0