#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu, Sudipta Basak'

import os
import sys
import numpy as np
import pandas as pd
import ConfigParser


class TransmissionConfig(object):
    """
    class to hold all configuration variables.
    """
    def __init__(self, cfg_file=None):

        if not os.path.exists(cfg_file):
            print('Error: configuration file {} not found'.format(cfg_file))
            sys.exit(1)

        conf = ConfigParser.ConfigParser()
        conf.optionxform = str
        conf.read(cfg_file)

        # run_type
        self.test = conf.getboolean('run_type', 'test')
        self.parallel = conf.getboolean('run_type', 'parallel')
        self.save = conf.getboolean('run_type', 'save')
        self.figure = conf.getboolean('run_type', 'figure')
        self.random_seed = conf.getboolean('run_type', 'random_seed')

        # random seed
        if self.random_seed:
            self.seed = dict()
            rnd_events = conf.get('random_seed', 'events').split(',')
            rnd_lines = conf.get('random_seed', 'lines').split(',')

            for rnd_event in rnd_events:
                (event_key, event_val) = self.split_str(rnd_event, ':')
                for rnd_line in rnd_lines:
                    (line_key, line_val) = self.split_str(rnd_line, ':')
                    self.seed.setdefault(event_key, {})[line_key] = \
                        event_val + line_val

        # run_parameters
        self.nsims = conf.getint('run_parameters', 'num_simulations')
        self.analytical = conf.getboolean('run_parameters', 'run_analytical')
        self.simulation = conf.getboolean('run_parameters', 'run_simulation')
        self.skip_non_cascading_collapse = conf.getboolean(
            'run_parameters', 'skip_non_cascading_collapse')
        self.adjust_design_by_topo = conf.getboolean(
            'run_parameters', 'adjust_design_by_topography')
        strainer_ = conf.get('run_parameters', 'Strainer')
        self.strainer = [x.strip() for x in strainer_.split(',')]

        if conf.getboolean('run_parameters', 'parallel_line_interaction'):
            self.parallel_line = dict():
            for line in conf.options('parallel_line_interaction'):
                self.parallel_line_interaction[line] = [x.strip() for x in 
                    conf.get('parallel_line_interaction', line).split(',')]

        # directories
        self.path_gis_data = self.get_path(conf.get('directories', 'gis_data'), 
                                           cfg_file)

        self.path_wind_scenario = [self.get_path(x.strip(), cfg_file) 
            for x in conf.get('directories', 'wind_scenario').split(',')]

        path_input = self.get_path(conf.get('directories', 'input'), cfg_file)

        self.path_output = self.get_path(conf.get('directories', 'output'), 
                                         cfg_file)

        # gis_data
        self.file_shape_tower = os.path.join(self.path_gis_data,
                                             conf.get('gis_data',
                                                      'shape_tower'))

        self.file_shape_line = os.path.join(self.path_gis_data,
                                            conf.get('gis_data',
                                                     'shape_line'))

        # wind_scenario
        self.file_name_format = conf.get('wind_scenario',
                                         'file_name_format', 1)

        # input
        self.file_design_value = os.path.join(path_input,
                                              conf.get('input',
                                                       'design_value'))

        self.file_fragility = os.path.join(path_input,
                                           conf.get('input', 'fragility'))

        self.file_cond_collapse_prob = os.path.join(
            path_input, conf.get('input', 'conditional_collapse_probability'))

        self.file_terrain_multiplier = os.path.join(
            path_input, conf.get('input', 'terrain_height_multiplier'))

        self.file_drag_height_by_type = os.path.join(
            path_input, conf.get('input', 'drag_height_by_type'))

        # read information
        self.design_value = self.read_design_value()

        self.sel_lines = self.design_value.keys()

        self.fragility_curve, self.damage_states, self.no_damage_states =\
            self.read_fragility()

        self.cond_collapse_prob = self.read_cond_collapse_prob()

        self.terrain_multiplier = self.read_ASNZS_terrain_multiplier()

        self.drag_height = self.read_drag_height_by_type()

        if self.adjust_design_by_topo:

            self.file_topo_multiplier = os.path.join(
                path_input, conf.get('input', 'topographic_multiplier'))

            self.file_design_adjustment_factor_by_topo = os.path.join(
                path_input, conf.get(
                    'input', 'design_adjustment_factor_by_topography'))

            self.topo_multiplier = self.read_topographic_multiplier()

            self.design_adjustment_factor_by_topo = \
                self.read_design_adjustment_factor_by_topography_mutliplier()

    @staticmethod
    def get_path(path_, file_):
        '''
        return absolute path of path_ which is relative to location of file_
        '''
        path_file_ = os.path.join(os.path.abspath(file_), os.pardir)
        return os.path.abspath(os.path.join(path_file_, path_))

    @staticmethod
    def split_str(str_, str_split):
        ''' split string with split_str and return tuple of str and integer
        '''
        list_ = str_.split(str_split)
        return list_[0].strip(), int(list_[1])

    # @property
    # def test(self):
    #     return self.__test

    # @test.setter
    # def test(self, val):
    #     if val:
    #         self.__test = 1
    #     else:
    #         self.__test = 0

    def read_design_value(self):
        """read design values by line
        """
        data = pd.read_csv(self.file_design_value,
                           skipinitialspace=True, skiprows=1,
                           names=['lineroute', 'speed', 'span', 'cat',
                                  'level'], index_col=0)
        return data.transpose().to_dict()

    def read_fragility(self):
        """
        read collapse fragility parameter values
        """

        data = pd.read_csv(self.file_fragility, skipinitialspace=True)
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

    def read_cond_collapse_prob(self):
        """
        read condition collapse probability defined by tower function
        """

        data = pd.read_csv(self.file_cond_collapse_prob, skipinitialspace=1)

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

    def read_ASNZS_terrain_multiplier(self):
        """
        read terrain multiplier (ASNZ 1170.2:2011 Table 4.1)
        """
        data = pd.read_csv(self.file_terrain_multiplier,
                           skipinitialspace=True,
                           skiprows=1,
                           names=['height', 'tc1', 'tc2', 'tc3', 'tc4'])
        return data.to_dict('list')

    def read_drag_height_by_type(self):
        """read typical drag height by tower type
        :param file_: input file
        :type file_: string

        :returns: drag height
        :rtype: dict

        """
        # design speed adjustment factor
        temp = pd.read_csv(self.file_drag_height_by_type,
                           names=['type', 'value'], skiprows=1, index_col=0)
        return temp['value'].to_dict()

    def read_topographic_multiplier(self):
        """read topograhpic multipler value from the input file
        :param file_: input file
        :type file_: string
        :returns: topography value at each site
        :rtype: dict
        """
        names_str = ['Name', '', '', '', '', '', '', '', '', 'Mh', 'Mhopp']
        data = pd.read_csv(self.file_topo_multiplier,
                           usecols=[0, 9, 10],
                           skiprows=1,
                           names=names_str)
        data['topo'] = data[['Mh', 'Mhopp']].max(axis=1)
        return data.set_index('Name').to_dict()['topo']

    def read_design_adjustment_factor_by_topography_mutliplier(self):
        """read design wind speed adjustment based on topographic multiplier
        :param file_: input file
        :type file_: string

        :returns: design speed adjustment factor
        :rtype: dict

        """

        # design speed adjustment factor
        temp = pd.read_csv(self.file_design_adjustment_factor_by_topo,
                           skiprows=2)
        data = temp.set_index('key').to_dict()['value']

        # threshold
        temp = pd.read_csv(self.file_design_adjustment_factor_by_topo,
                           skiprows=1)
        data['threshold'] = np.array([float(x) for x in temp.columns])

        assert len(data['threshold']) == len(data.keys()) - 2
        return data
