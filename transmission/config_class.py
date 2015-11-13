#!/usr/bin/env python
__author__ = 'Sudipta Basak'
import os
import numpy as np
import pandas as pd
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#from scipy.stats import lognorm


class TransmissionConfig(object):
    """
    class to hold all configuration variables.
    Should eventually be read from a config file. Not implemented yet.
    """
    def __init__(self, test=0):
        self.pdir = os.getcwd()
        self.file_shape_tower = os.path.join(
            self.pdir,
            'gis_data',
            'Towers_with_extra_strainers_WGS84.shp')
        self.file_shape_line = os.path.join(
            self.pdir,
            'gis_data',
            'Lines_NGCP_with_synthetic_attributes_WGS84.shp')
        self.file_frag = os.path.join(self.pdir, 'input', 'fragility_GA.csv')
        self.file_cond_pc = os.path.join(self.pdir, 'input',
                                         'cond_collapse_prob_NGCP.csv')
        self.file_terrain_height = os.path.join(self.pdir, 'input',
                                                'terrain_height_multiplier.csv')
        self.flag_strainer = ['Strainer', 'dummy']  # consider strainer

        self.file_design_value = os.path.join(self.pdir, 'input',
                                              'design_value_current.csv')
        #file_topo_value = os.path.join(pdir,
        #                                'input/topo_value_scenario_50yr.csv')
        self.file_topo_value = None
        #self.file_adjust_by_topo = os.path.join(self.pdir, 'input',
        #                                        'adjust_by_topo.txt')
        self.file_adjust_by_topo = None
        self.dir_wind_timeseries = os.path.join(self.pdir, 'wind_scenario',
                                                'glenda_reduced')

        # flag for test, no need to change
        self.test = test

        if self.test:
            self.flag_save = 0
            self.nsims = 20
            self.dir_output = os.path.join(self.pdir, 'transmission', 'tests',
                                           'test_output_current_glenda')
        else:
            self.flag_save = 0
            self.nsims = 3000
            self.dir_output = os.path.join(self.pdir, 'output_current_glenda')

        # parallel or serial computation
        self.parallel = 0

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        self.fragility_curve, self.damage_states, self.no_damage_states =\
            self.read_frag()
        self.cond_pc = self.get_cond_pc()

    @property
    def test(self):
        return self.__test

    @test.setter
    def test(self, val):
        if val:
            self.__test = 1
        else:
            self.__test = 0

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
