#!/usr/bin/env python
__author__ = 'Sudipta Basak'
import os


class TransmissionConfig(object):
    """
    class to hold all configuration variables.
    Should eventually be read from a config file. Not implemented yet.
    """
    def __init__(self, test=0):
        self.pdir = os.getcwd()
        self.shape_file_tower = os.path.join(
            self.pdir, 'gis_data', 'Towers_with_extra_strainers_WGS84.shp')
        self.shape_file_line = os.path.join(
            self.pdir, 'gis_data', 'Lines_NGCP_with_synthetic_attributes_WGS84.shp')

        self.file_frag = os.path.join(self.pdir, 'input', 'fragility_GA.csv')
        self.file_cond_pc = os.path.join(self.pdir, 'input' ,'cond_collapse_prob_NGCP.csv')
        self.file_terrain_height = os.path.join(self.pdir, 'input', 'terrain_height_multiplier.csv')
        self.flag_strainer = ['Strainer', 'dummy']  # consider strainer

        self.file_design_value = os.path.join(self.pdir, 'input', 'design_value_current.csv')
        #file_topo_value = os.path.join(pdir, 'input/topo_value_scenario_50yr.csv')
        self.file_topo_value = None
        self.dir_wind_timeseries = os.path.join(self.pdir, 'wind_scenario', 'glenda_reduced')

        # flag for test, no need to change
        self.test = test

        if self.test:
            self.flag_save = 0
            self.nsims = 20
            self.dir_output = os.path.join(self.pdir, 'transmission', 'tests', 'test_output_current_glenda')
        else:
            self.flag_save = 0
            self.nsims = 20
            self.dir_output = os.path.join(self.pdir, 'output_current_glenda')

        # parallel or serial computation
        self.parallel = 1

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

    @property
    def test(self):
        return self.__test

    @test.setter
    def test(self, val):
        if val:
            self.__test = 1
        else:
            self.__test = 0

