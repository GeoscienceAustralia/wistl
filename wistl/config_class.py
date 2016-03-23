#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu, Sudipta Basak'

import os
import sys
import numpy as np
import pandas as pd
import ConfigParser

WISTL = os.environ['WISTL']

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
        self.strainer = [x.strip() for x in conf.get('run_parameters',
                         'Strainer').split(',')]

        if conf.getboolean('run_parameters', 'parallel_line_interaction'):
            self.parallel_line = dict()
            for line in conf.options('parallel_line_interaction'):
                self.parallel_line[line] = [
                    x.strip() for x in conf.get('parallel_line_interaction',
                                                line).split(',')]

        # directories
        self.path_gis_data = self.get_path(conf.get('directories', 'gis_data'),
                                           cfg_file)

        self.wind_scenarios_path = self.get_path(
            conf.get('directories', 'wind_scenario_main'),
            cfg_file)

        self.path_wind_scenario = [
            self.get_path(x.strip(), cfg_file) for x in conf.get(
                'directories', 'wind_scenario').split(',')]

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
        file_name_format = conf.get('wind_scenario', 'file_name_format', 1)
        self.file_head = file_name_format.split('%')[0]
        self.file_tail = file_name_format.split(')')[-1]

        # input
        self.file_design_value = os.path.join(path_input,
                                              conf.get('input', 'design_value'))

        self.file_fragility_metadata = os.path.join(
            path_input, conf.get('input', 'fragility_metadata'))

        self.file_cond_collapse_prob_metadata = os.path.join(
            path_input, conf.get('input',
                                 'conditional_collapse_probability_metadata'))

        self.file_terrain_multiplier = os.path.join(
            path_input, conf.get('input', 'terrain_height_multiplier'))

        self.file_drag_height_by_type = os.path.join(
            path_input, conf.get('input', 'drag_height_by_type'))

        # read information
        self.design_value = self.read_design_value()

        self.sel_lines = self.design_value.keys()

        self.fragility_metadata, self.fragility, self.damage_states, \
            self.no_damage_states = self.read_fragility()

        self.cond_collapse_prob_metadata, self.cond_collapse_prob = \
            self.read_cond_collapse_prob()

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

        if conf.getboolean('run_parameters', 'parallel_line_interaction'):
            self.file_line_interaction = os.path.join(
                path_input, conf.get('input', 'line_interaction'))

            #self.prob_line_interaction = self.read_parallel_interaction_prob()

    @staticmethod
    def get_path(path_, file_):
        """
        return absolute path of path_ which is relative to location of file_
        """
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
        if not os.path.exists(self.file_design_value):
            print('Error: file_design_value {} not found'.format(
                self.file_design_value))
            sys.exit(1)

        data = pd.read_csv(self.file_design_value,
                           skipinitialspace=True, skiprows=1,
                           names=['lineroute', 'speed', 'span', 'cat',
                                  'level'], index_col=0)
        return data.transpose().to_dict()

    def read_fragility(self):
        """
        read collapse fragility parameter values
        """
        if not os.path.exists(self.file_fragility_metadata):
            print('Error: file_fragility_metadata {} not found'.format(
                self.file_fragility_metadata))
            sys.exit(1)

        metadata = ConfigParser.ConfigParser()
        metadata.read(self.file_fragility_metadata)
        metadata = metadata._sections

        for item, value in metadata['main'].iteritems():
            if ',' in value:
                metadata['main'][item] = [x.strip() for x in value.split(',')]

        # remove main
        meta_data = metadata['main'].copy()
        metadata.pop('main')
        meta_data.update(metadata)

        meta_data['file'] = self.get_path(meta_data['file'],
                                          self.file_fragility_metadata)

        try:
            data = pd.read_csv(meta_data['file'], skipinitialspace=True)
        except IOError:
            print('Error: file_fragility {} not found'.format(meta_data['file']))
            sys.exit(1)

        return meta_data, data, meta_data['limit_states'], len(
            meta_data['limit_states'])

    def read_cond_collapse_prob(self):
        """
        read condition collapse probability defined by tower function
        """
        if not os.path.exists(self.file_cond_collapse_prob_metadata):
            print('Error: file_cond_collapse_prob_metadata {} not found'.format(
                self.file_cond_collapse_prob_metadata))
            sys.exit(1)

        metadata = ConfigParser.ConfigParser()
        metadata.read(self.file_cond_collapse_prob_metadata)
        metadata = metadata._sections

        for item, value in metadata['main'].iteritems():
            if ',' in value:
                metadata['main'][item] = [x.strip() for x in value.split(',')]

        # remove main
        meta_data = metadata['main'].copy()
        metadata.pop('main')
        meta_data.update(metadata)

        cond_pc = dict()
        for item in meta_data['list']:
            file_ = self.get_path(meta_data[item]['file'],
                                  self.file_cond_collapse_prob_metadata)
            meta_data[item]['file'] = file_
            df_tmp = pd.read_csv(file_, skipinitialspace=1)
            df_tmp['start'] = df_tmp['start'].astype(np.int64)
            df_tmp['end'] = df_tmp['end'].astype(np.int64)
            df_tmp['list'] = df_tmp.apply(lambda x: tuple(range(x['start'],
                                          x['end'] + 1)), axis=1)
            cond_pc[item] = df_tmp.loc[df_tmp[meta_data['by']] == item]
            meta_data[item]['max_adj'] = cond_pc[item]['end'].max()

        return meta_data, cond_pc

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
        adic = ConfigParser.ConfigParser()
        adic.read(self.file_design_adjustment_factor_by_topo)

        data = dict()
        for key, value in adic.items('main'):
            try:
                data[int(key)] = float(value)
            except ValueError:
                data[key] = np.array([float(x) for x in value.split(',')])

        assert len(data['threshold']) == len(data.keys()) - 2
        return data
