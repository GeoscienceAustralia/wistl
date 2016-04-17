#!/usr/bin/env python
from __future__ import print_function

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
            msg = 'Error: file {} not found'.format(cfg_file)
            sys.exit(msg)

        conf = ConfigParser.ConfigParser()
        conf.optionxform = str
        conf.read(cfg_file)

        path_cfg_file = os.path.dirname(os.path.realpath(cfg_file))

        # run_type
        key = 'run_type'
        self.parallel = conf.getboolean(key, 'parallel')
        self.save = conf.getboolean(key, 'save')
        self.figure = conf.getboolean(key, 'figure')
        self.random_seed = conf.getboolean(key, 'random_seed')

        # random seed
        self.seed = None
        if self.random_seed:
            self.set_random_seed(conf)

        # run_parameters
        key = 'run_parameters'
        self.no_sims = conf.getint(key, 'number_simulations')
        self.analytical = conf.getboolean(key, 'run_analytical')
        self.simulation = conf.getboolean(key, 'run_simulation')
        self.skip_non_cascading_collapse = conf.getboolean(
            key, 'skip_non_cascading_collapse')
        self.adjust_design_by_topography = conf.getboolean(
            key, 'adjust_design_by_topography')

        self.strainer = []
        for x in conf.get(key, 'Strainer').split(','):
            self.strainer.append(x.strip())

        self.selected_lines = []
        for x in conf.get(key, 'selected_lines').split(','):
            self.selected_lines.append(x.strip())

        self.rtol = float(conf.get(key, 'relative_tolerance'))
        self.atol = float(conf.get(key, 'absolute_tolerance'))


        # directories
        key = 'directories'
        self.path_gis_data = os.path.join(path_cfg_file,
                                          conf.get(key, 'gis_data'))

        self.path_wind_scenario_base = os.path.join(
            path_cfg_file, conf.get(key, 'wind_scenario'))

        self.path_input = os.path.join(path_cfg_file,
                                       conf.get(key, 'input'))

        self.path_output = os.path.join(path_cfg_file,
                                        conf.get(key, 'output'))

        # gis_data
        self.file_shape_tower = os.path.join(self.path_gis_data,
                                             conf.get('gis_data',
                                                      'shape_tower'))

        self.file_shape_line = os.path.join(self.path_gis_data,
                                            conf.get('gis_data',
                                                     'shape_line'))

        # wind_scenario
        self.scale = dict()
        for event_id in conf.options('wind_scenario'):
            self.scale[event_id] = []
            for x in conf.get('wind_scenario', event_id).split(','):
                try:
                    self.scale[event_id].append(float(x))
                except ValueError:
                    pass

        # format
        wind_file_name_format = conf.get('format', 'wind_scenario', 1)
        self.wind_file_head = wind_file_name_format.split('%')[0]
        self.wind_file_tail = wind_file_name_format.split(')')[-1]
        self.event_id_scale_str = conf.get('format', 'event_id_scale')

        # input
        self.file_design_value = os.path.join(self.path_input,
                                              conf.get('input', 'design_value'))
        self.file_fragility_metadata = os.path.join(
            self.path_input, conf.get('input', 'fragility_metadata'))
        self.file_cond_collapse_prob_metadata = os.path.join(
            self.path_input, conf.get(
                'input', 'conditional_collapse_probability_metadata'))
        self.file_terrain_multiplier = os.path.join(
            self.path_input, conf.get('input', 'terrain_height_multiplier'))
        self.file_drag_height_by_type = os.path.join(
            self.path_input, conf.get('input', 'drag_height_by_type'))

        self.file_topo_multiplier = None
        self.file_design_adjustment_factor_by_topo = None
        self.topo_multiplier = None
        self.design_adjustment_factor_by_topo = None
        if self.adjust_design_by_topography:
            self.set_adjust_design_by_topography(conf)

        self.file_line_interaction_metadata = None
        self.line_interaction = None
        self.prob_line_interaction_metadata = None
        self.prob_line_interaction = None
        if conf.getboolean('run_parameters', 'line_interaction'):
            self.set_line_interaction(conf)

        # read information
        self.design_value = self.read_design_value()

        self.fragility_metadata, self.fragility, self.damage_states, \
            self.no_damage_states = self.read_fragility()

        self.cond_collapse_prob_metadata, self.cond_collapse_prob = \
            self.read_cond_collapse_prob()

        self.terrain_multiplier = self.read_terrain_multiplier()

        self.drag_height = self.read_drag_height_by_type()

    def set_random_seed(self, conf):
        """
        read random seed info
        :param conf:
        :return:
        """
        self.seed = dict()

        rnd_events = conf.get('random_seed', 'events').split(',')
        rnd_lines = conf.get('random_seed', 'lines').split(',')

        for rnd_event in rnd_events:
            (event_key, event_val) = split_str(rnd_event, ':')

            for rnd_line in rnd_lines:
                (line_key, line_val) = split_str(rnd_line, ':')
                self.seed.setdefault(event_key, {})[line_key] = \
                    event_val + line_val

    def set_adjust_design_by_topography(self, conf):

        self.file_topo_multiplier = os.path.join(
            self.path_input, conf.get('input', 'topographic_multiplier'))

        self.file_design_adjustment_factor_by_topo = os.path.join(
            self.path_input, conf.get(
                'input', 'design_adjustment_factor_by_topography'))

        self.topo_multiplier = self.read_topographic_multiplier()

        self.design_adjustment_factor_by_topo = \
            self.read_design_adjustment_factor_by_topography_mutliplier()

    def set_line_interaction(self, conf):

        self.line_interaction = dict()
        for line in conf.options('line_interaction'):
            self.line_interaction[line] = [
                x.strip() for x in conf.get('line_interaction',
                                            line).split(',')]

        self.file_line_interaction_metadata = os.path.join(
            self.path_input, conf.get('input', 'line_interaction_metadata'))

        self.prob_line_interaction_metadata, self.prob_line_interaction = \
            self.read_prob_line_interaction()

    # @staticmethod
    # def get_path(path_, file_):
    #     """
    #     :param path_: path
    #     :param file_: file
    #     :return: absolute path of path_ which is relative to location of file_
    #     """
    #     path_file_ = os.path.join(os.path.abspath(file_), os.pardir)
    #     return os.path.abspath(os.path.join(path_file_, path_))

    def read_design_value(self):
        """read design values by line
        """
        try:
            data = pd.read_csv(self.file_design_value,
                               skipinitialspace=True,
                               index_col=0)
            data.columns = ['speed', 'span', 'cat', 'level']
            return data.transpose().to_dict()

        except IOError:
            msg = 'Error: file {} not found'.format(self.file_design_value)
            raise IOError(msg)

    def read_fragility(self):
        """
        read collapse fragility parameter values
        """
        if not os.path.exists(self.file_fragility_metadata):
            msg = 'Error: file {} not found'.format(
                self.file_fragility_metadata)
            sys.exit(msg)

        metadata = ConfigParser.ConfigParser()
        metadata.read(self.file_fragility_metadata)
        metadata = metadata._sections

        path_metadata = os.path.dirname(
            os.path.realpath(self.file_cond_collapse_prob_metadata))

        for item, value in metadata['main'].iteritems():
            if ',' in value:
                metadata['main'][item] = [x.strip() for x in value.split(',')]

        # remove main
        meta_data = metadata['main'].copy()
        metadata.pop('main')
        meta_data.update(metadata)

        meta_data['file'] = os.path.join(path_metadata, meta_data['file'])

        try:
            data = pd.read_csv(meta_data['file'], skipinitialspace=True)
        except IOError:
            msg = 'Error: file {} not found'.format(meta_data['file'])
            raise IOError(msg)

        return meta_data, data, meta_data['limit_states'], len(
            meta_data['limit_states'])

    def read_cond_collapse_prob(self):
        """
        read condition collapse probability defined by tower function
        """
        if not os.path.exists(self.file_cond_collapse_prob_metadata):
            msg = 'Error: file {} not found'.format(
                self.file_cond_collapse_prob_metadata)
            sys.exit(msg)

        metadata = ConfigParser.ConfigParser()
        metadata.read(self.file_cond_collapse_prob_metadata)
        metadata = metadata._sections

        path_metadata = os.path.dirname(
            os.path.realpath(self.file_cond_collapse_prob_metadata))

        for item, value in metadata['main'].iteritems():
            if ',' in value:
                metadata['main'][item] = [x.strip() for x in value.split(',')]

        # remove main
        meta_data = metadata['main'].copy()
        metadata.pop('main')
        meta_data.update(metadata)

        cond_pc = dict()
        for item in meta_data['list']:
            file_ = os.path.join(path_metadata, meta_data[item]['file'])
            meta_data[item]['file'] = file_
            df_tmp = pd.read_csv(file_, skipinitialspace=1)
            df_tmp['start'] = df_tmp['start'].astype(np.int64)
            df_tmp['end'] = df_tmp['end'].astype(np.int64)
            df_tmp['list'] = df_tmp.apply(lambda x_: tuple(range(x_['start'],
                                          x_['end'] + 1)), axis=1)
            cond_pc[item] = df_tmp.loc[df_tmp[meta_data['by']] == item]
            meta_data[item]['max_adj'] = cond_pc[item]['end'].max()

        return meta_data, cond_pc

    def read_prob_line_interaction(self):
        """
        read conditional parallel line interaction probability
        """
        if not os.path.exists(self.file_line_interaction_metadata):
            msg = 'Error: file {} not found'.format(
                self.file_line_interaction_metadata)
            sys.exit(msg)

        path_metadata = os.path.dirname(
            os.path.realpath(self.file_line_interaction_metadata))

        metadata = ConfigParser.ConfigParser()
        metadata.read(self.file_line_interaction_metadata)

        meta_data = dict()
        for key, value in metadata.items('main'):
                meta_data[key] = value

        file_ = os.path.join(path_metadata, meta_data['file'])
        cond_pc = pd.read_csv(file_, skipinitialspace=1)

        return meta_data, cond_pc

    def read_terrain_multiplier(self):
        """
        read terrain multiplier (AS/NZS 1170.2:2011 Table 4.1)
        """
        try:
            data = pd.read_csv(self.file_terrain_multiplier,
                               skipinitialspace=True)
            data.columns = ['height', 'tc1', 'tc2', 'tc3', 'tc4']
        except IOError:
            msg = 'Error: file {} not found'.format(
                self.file_terrain_multiplier)
            raise IOError(msg)
        return data.to_dict('list')

    def read_drag_height_by_type(self):
        """read typical drag height by tower type
        :returns: drag height
        :rtype: dict

        """
        # design speed adjustment factor
        try:
            data = pd.read_csv(self.file_drag_height_by_type,
                               skipinitialspace=True,
                               index_col=0)
            data.columns = ['value']
        except IOError:
            msg = 'Error: file {} not found'.format(
                self.file_drag_height_by_type)
            raise IOError(msg)

        return data['value'].to_dict()

    def read_topographic_multiplier(self):
        """read topographic multiplier value from the input file
        :returns: topography value at each site
        :rtype: dict
        """
        try:
            data = pd.read_csv(self.file_topo_multiplier,
                               usecols=[0, 9, 10])
            data.columns = ['Name', 'Mh', 'Mhopp']
            data['topo'] = data[['Mh', 'Mhopp']].max(axis=1)
        except IOError:
            msg = 'Error: file {} not found'.format(self.file_topo_multiplier)
            raise IOError(msg)

        return data.set_index('Name').to_dict()['topo']

    def read_design_adjustment_factor_by_topography_mutliplier(self):
        """read design wind speed adjustment based on topographic multiplier
        :returns: design speed adjustment factor
        :rtype: dict

        """
        if not os.path.exists(self.file_design_adjustment_factor_by_topo):
            msg = 'Error: file {} not found'.format(
                self.file_design_adjustment_factor_by_topo)
            sys.exit(msg)

        dic_ = ConfigParser.ConfigParser()
        dic_.read(self.file_design_adjustment_factor_by_topo)

        data = dict()
        for key, value in dic_.items('main'):
            try:
                data[int(key)] = float(value)
            except ValueError:
                data[key] = np.array([float(x) for x in value.split(',')])

        assert len(data['threshold']) == len(data.keys()) - 2
        return data


def split_str(str_, str_split):
    """
    split string with split_str
    :param str_: string
    :param str_split: split
    :return: tuple of str and integer or float
    """
    list_ = str_.split(str_split)
    try:
        return list_[0].strip(), int(list_[1])
    except ValueError:
        try:
            return list_[0].strip(), float(list_[1])
        except ValueError:
            return list_[0].strip(), list_[1].strip()
