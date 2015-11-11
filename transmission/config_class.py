#!/usr/bin/env python
__author__ = 'Sudipta Basak'
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm


class TransmissionConfig(object):
    """
    class to hold all configuration variables.
    Should eventually be read from a config file. Not implemented yet.
    """
    def __init__(self, test=0):
        self.pdir = os.getcwd()
        self.shape_file_tower = os.path.join(self.pdir, 'gis_data', 'Towers_with_extra_strainers_WGS84.shp')
        self.shape_file_line = os.path.join(self.pdir, 'gis_data', 'Lines_NGCP_with_synthetic_attributes_WGS84.shp')

        self.file_frag = os.path.join(self.pdir, 'input', 'fragility_GA.csv')
        self.file_cond_pc = os.path.join(self.pdir, 'input', 'cond_collapse_prob_NGCP.csv')
        self.file_terrain_height = os.path.join(self.pdir, 'input', 'terrain_height_multiplier.csv')
        self.flag_strainer = ['Strainer', 'dummy']  # consider strainer

        self.file_design_value = os.path.join(self.pdir, 'input', 'design_value_current.csv')
        #file_topo_value = os.path.join(pdir,
        #                                'input/topo_value_scenario_50yr.csv')
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
            self.nsims = 200
            self.dir_output = os.path.join(self.pdir, 'output_current_glenda')

        # parallel or serial computation
        self.parallel = 0
        self.parallel_time_level = 1

        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        self.fragility_curve, self.damage_states, self.no_damage_states = self.read_frag()
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

    def read_frag(self, flag_plot=None):
        """read collapse fragility parameter values
        >>> txt = '''ConstType, damage, index, function, param0, param1
        ... Unknown, minor, 1, lognorm, 0.85, 0.05'''
        ... Unknown, collpase, 2, lognorm, 1.02, 0.05'''
        >>> data = read_collapse_frag(StringIO(txt))
        >>> data'''
        {'Unknown': {'collapse': {'function': 'lognorm',
           'param0': 1.02,
           'param1': 0.05},
          'minor': {'function': 'lognorm', 'param0': 0.85, 'param1': 0.05}}}
        """

        data = pd.read_csv(self.file_frag, skipinitialspace=1)

        frag = {}
        for ttype in data['tower type'].unique():
            for func in data['function'].unique():
                idx = (data['tower type'] == ttype) & (data['function'] == func)
                dev0_ = data['dev0'].ix[idx].unique()
                dev1_ = data['dev1'].ix[idx].unique()
                dev_ = np.sort(np.union1d(dev0_, dev1_))

                frag.setdefault(ttype, {}).setdefault(func, {})['dev_angle'] = dev_

                for j, val in enumerate(dev0_):

                    idx2 = np.where(idx & (data['dev0'] == val))[0]

                    for k in idx2:
                        ds_ = data.ix[k]['damage']
                        idx_ = data.ix[k]['index']
                        cdf_ = data.ix[k]['cdf']
                        param0_ = data.ix[k]['param0']
                        param1_ = data.ix[k]['param1']

                        # dev_angle (0, 5, 15, 30, 360) <= tower_angle 0.0 => index
                        # angle is less than 360. if 360 then 0.
                        frag[ttype][func].setdefault(j+1, {}).setdefault(ds_, {})['idx'] = idx_
                        frag[ttype][func][j+1][ds_]['param0'] = param0_
                        frag[ttype][func][j+1][ds_]['param1'] = param1_
                        frag[ttype][func][j+1][ds_]['cdf'] = cdf_

        ds_list = [(x, frag[ttype][func][1][x]['idx']) for x in frag[ttype][func][1].keys()]
        ds_list.sort(key=lambda tup: tup[1])  # sort by ids

        nds = len(ds_list)

        if flag_plot:
            x = np.arange(0.5, 1.5, 0.01)
            line_style = {'minor': '--', 'collapse': '-'}

            for ttype in frag.keys():
                for func in frag[ttype].keys():
                    plt.figure()

                    for idx in frag[ttype][func].keys():
                        try:
                            for ds in frag[ttype][func][idx].keys():
                                med = frag[ttype][func][idx][ds]['param0']
                                sig = frag[ttype][func][idx][ds]['param1']
                                y = lognorm.cdf(x, sig, scale=med)
                                plt.plot(x,y, line_style[ds])
                        except AttributeError:
                            print "no"

                    plt.legend(['collapse', 'minor'], 2)
                    plt.xlabel('Ratio of wind speed to adjusted design wind speed')
                    plt.ylabel('Probability of exceedance')
                    plt.title(ttype+':'+func)
                    plt.yticks(np.arange(0, 1.1, 0.1))
                    plt.grid(1)
                    plt.savefig(ttype + '_' + func + '.png')

        return frag, ds_list, nds

    def get_cond_pc(self):
        """read condition collapse probability defined by tower function

        >>> txt = '''FunctionType, # of collapse, probability, start, end
        ... suspension, 1, 0.075, 0, 1
        ... suspension, 1, 0.075, -1, 0
        ... suspension, 2, 0.35, -1, 1
        ... suspension, 3, 0.025, -1, 2
        ... suspension, 3, 0.025, -2, 1
        ... suspension, 4, 0.10, -2, 2
        ... strainer, 1, 0.075, 0, 1
        ... strainer, 1, 0.075, -1, 0
        ... strainer, 2, 0.35, -1, 1
        ... strainer, 3, 0.025, -1, 2
        ... strainer, 3, 0.025, -2, 1
        ... strainer, 4, 0.10, -2, 2'''
        ... strainer, 5, 0.10, -2, 2'''
        ... strainer, 5, 0.10, -2, 2'''
        ... strainer, 5, 0.10, -2, 2'''
        >>> cond_pc = read_cond_prob(StringIO(txt))
        >>> cond_pc'''
        {'strainer': {'max_adj': 2,
          (-2, -1, 0, 1): 0.025,
          (-2, -1, 0, 1, 2): 0.1,
          (-1, 0): 0.075,
          (-1, 0, 1): 0.35,
          (-1, 0, 1, 2): 0.025,
          (0, 1): 0.075},
         'suspension': {'max_adj': 2,
          (-2, -1, 0, 1): 0.025,
          (-2, -1, 0, 1, 2): 0.1,
          (-1, 0): 0.075,
          (-1, 0, 1): 0.35,
          (-1, 0, 1, 2): 0.025,
          (0, 1): 0.075}}
        """

        data = pd.read_csv(self.file_cond_pc, skipinitialspace=1)
        cond_pc = {}
        for line in data.iterrows():
            func, cls_str, thr, pb, n0, n1 = [line[1][x] for x in
                                    ['FunctionType', 'class', 'threshold', 'probability', 'start', 'end']]
            list_ = range(int(n0), int(n1)+1)
            cond_pc.setdefault(func, {})['threshold'] = thr
            cond_pc[func].setdefault(cls_str, {}).setdefault('prob', {})[tuple(list_)] = float(pb)

        for func in cond_pc.keys():
            cls_str = cond_pc[func].keys()
            cls_str.remove('threshold')
            for cls in cls_str:
                max_no_adj_towers = np.max(np.abs([j for k in cond_pc[func][cls]['prob'].keys()
                                for j in k]))
                cond_pc[func][cls]['max_adj'] = max_no_adj_towers

        return cond_pc
