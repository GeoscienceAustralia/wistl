#!/usr/bin/env python
__author__ = 'Sudipta Basak, Hyeuk Ryu'
import os
import sys
import numpy as np
import pandas as pd
import ConfigParser

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#from scipy.stats import lognorm


class TransmissionConfig(object):
    """
    class to hold all configuration variables.
    """
    def __init__(self, cfg_file=None):

        if not os.path.exists(cfg_file):
            print "Error: configuration file {} not found".format(cfg_file)
            sys.exit()

        conf = ConfigParser.ConfigParser()
        conf.read(cfg_file)
        conf_dic = conf._sections

        # run_type
        self.test = conf.getboolean('run_type', 'flag_test')
        self.parallel = conf.getboolean('run_type', 'flag_parallel')
        self.flag_save = conf.getboolean('run_type', 'flag_save')

        # run_parameters
        self.nsims = conf.getint('run_parameters', 'num_simulations')
        self.flag_adjust_design_by_topo =\
            conf.getboolean('run_parameters', 'adjust_design_by_topography')
        strainer_ = conf.get('run_parameters', 'Strainer')
        self.flag_strainer = [x.strip(' ') for x in strainer_.split(',')]

        # directories
        self.pdir = conf.get('directories', 'project')

        try:
            path_gis_data = conf.get('directories', 'gis_data')
        except ValueError:
            path_gis_data = self.get_path(
                conf.get('directories', 'gis_data', 1), conf_dic)
        try:
            self.dir_wind_timeseries = conf.get('directories', 'wind_scenario')
        except ValueError:
            self.dir_wind_timeseries = self.get_path(
                conf.get('directories', 'wind_scenario', 1), conf_dic)

        try:
            path_input = conf.get('directories', 'input')
        except ValueError:
            path_input = self.get_path(
                conf.get('directories', 'input', 1), conf_dic)

        try:
            self.dir_output = conf.get('directories', 'output')
        except ValueError:
            self.dir_output = self.get_path(
                conf.get('directories', 'output', 1), conf_dic)

        # gis_data
        self.file_shape_tower = os.path.join(path_gis_data,
                                             conf.get('gis_data',
                                                      'shape_tower'))

        self.file_shape_line = os.path.join(path_gis_data,
                                            conf.get('gis_data',
                                                     'shape_line'))

        # wind_scenario
        self.file_name_format = conf.get('wind_scenario', 'file_name_format', 1)

        # input
        self.file_frag = os.path.join(path_input,
                                      conf.get('input', 'fragility'))

        self.file_cond_pc = os.path.join(
            path_input, conf.get('input', 'conditional_collapse_probability'))

        self.file_terrain_height = os.path.join(
            path_input, conf.get('input', 'terrain_height_multiplier'))

        self.file_design_value = os.path.join(path_input,
                                              conf.get('input',
                                                       'design_value'))

        self.file_drag_height_by_type = os.path.join(
            path_input, conf.get('input', 'drag_height_by_type'))

        if self.flag_adjust_design_by_topo:
            self.file_topo_value = os.path.join(
                path_input, conf.get('input', 'topographic_multiplier'))
            self.file_adjust_by_topo = os.path.join(
                path_input, conf.get('input', 'adjust_design_by_topography'))

        # flag for test, no need to change
        # self.test = test

        # if self.test:
        #     self.flag_save = 0
        #     self.nsims = 20
        #     self.dir_output = os.path.join(self.pdir, 'transmission', 'tests',
        #                                    'test_output_current_glenda')
        # else:
        #     self.flag_save = 0
        #     self.nsims = 3000
        #     self.dir_output = os.path.join(self.pdir, 'output_current_glenda')

        # # parallel or serial computation
        # self.parallel = 0

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        #
        self.fragility_curve, self.damage_states, self.no_damage_states =\
            self.read_frag()
        self.cond_pc = self.get_cond_pc()

    @staticmethod
    def get_path(path_, conf_dic):
        '''
        '''
        path_split = path_.split(')/')
        path_key = path_split[0][2:]
        path_full = conf_dic['directories'][path_key]

        return os.path.join(path_full, path_split[1])

    # @property
    # def test(self):
    #     return self.__test

    # @test.setter
    # def test(self, val):
    #     if val:
    #         self.__test = 1
    #     else:
    #         self.__test = 0

    def read_frag(self):
        """
        read collapse fragility parameter values
        """

        data = pd.read_csv(self.file_frag, skipinitialspace=True)
        grouped = (data.groupby(['tower type', 'function'])).groups

        frag = dict()
        for item in grouped.keys():
            (ttype, func) = item
            idx = grouped[item]
            dev0_ = data['dev0'].ix[idx].unique()
            dev1_ = data['dev1'].ix[idx].unique()

            frag.setdefault(ttype, {})[func] = {}
            frag.setdefault(ttype, {}).setdefault(func, {})['dev_angle'] =\
                np.sort(np.union1d(dev0_, dev1_))

            temp = {}
            # dev_angle (0, 5, 15, 30, 360) <= tower_angle 0.0 => index
            # angle is less than 360. if 360 then 0.
            for (j, k) in enumerate(idx):
                idx_dev = int(j/2)+1
                ds_ = data.ix[k]['damage']
                temp.setdefault(idx_dev, {}).setdefault(ds_, {})['idx'] =\
                    data.ix[k]['index']
                temp[idx_dev][ds_]['param0'] = data.ix[k]['param0']
                temp[idx_dev][ds_]['param1'] = data.ix[k]['param1']
                temp[idx_dev][ds_]['cdf'] = data.ix[k]['cdf']

            frag[ttype][func].update(temp)

        ds_list = [(x, frag[ttype][func][1][x]['idx'])
                   for x in frag[ttype][func][1].keys()]
        ds_list.sort(key=lambda tup: tup[1])  # sort by ids
        nds = len(ds_list)

        return frag, ds_list, nds

    def get_cond_pc(self):
        """
        read condition collapse probability defined by tower function
        """

        data = pd.read_csv(self.file_cond_pc, skipinitialspace=1)

        cond_pc = dict()
        for line in data.iterrows():
            func, cls_str, thr, pb, n0, n1 = [
                line[1][x] for x in ['FunctionType', 'class', 'threshold',
                                     'probability', 'start', 'end']]
            list_ = range(int(n0), int(n1)+1)
            cond_pc.setdefault(func, {})['threshold'] = thr
            cond_pc[func].setdefault(cls_str, {}).setdefault('prob', {})[
                tuple(list_)] = float(pb)

        for func in cond_pc.keys():
            cls_str = cond_pc[func].keys()
            cls_str.remove('threshold')
            for cls in cls_str:
                max_no_adj_towers = np.max(np.abs([
                    j for k in cond_pc[func][cls]['prob'].keys() for j in k]))
                cond_pc[func][cls]['max_adj'] = max_no_adj_towers

        return cond_pc
