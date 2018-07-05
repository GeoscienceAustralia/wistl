#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import logging
import numpy as np
import pandas as pd
import ConfigParser
import shapefile
from shapely import geometry
from geopy.distance import great_circle

from wistl.constants import K_FACTOR, NO_CIRCUIT


class Config(object):
    """
    class to hold all configuration variables.
    """
    option_keys = ['run_parallel', 'save_output', 'save_figure',
                   'use_random_seed', 'run_analytical', 'run_simulation',
                   'skip_non_cascading_collapse', 'adjust_design_by_topography',
                   'apply_line_interaction']

    directory_keys = ['gis_data', 'wind_scenario_base', 'input', 'output']

    input_keys = ['design_value', 'fragility_metadata', 'drag_height_by_type',
                  'cond_collapse_prob_metadata',
                  'terrain_multiplier', 'topographic_multiplier',
                  'design_adjustment_factor_by_topography',
                  'line_interaction_metadata']

    def __init__(self, file_cfg=None, logger=None):

        self.file_cfg = file_cfg
        self.logger = logger or logging.getLogger(__name__)

        self.options = {}

        self.seed = {}
        self.no_sims = None
        self.strainer = []
        self.selected_lines = []
        self.abs_tol = None
        self.rel_tol = None

        self.path_gis_data = None
        self.path_input = None
        self.path_output = None
        self.path_wind_scenario_base = None

        self.file_shape_line = None
        self.file_shape_tower = None
        self.file_design_value = None
        self.file_fragility_metadata = None
        self.file_cond_collapse_prob_metadata = None
        self.file_terrain_multiplier = None
        self.file_drag_height_by_type = None
        # if only adjust_design_by_topography
        self.file_topographic_multiplier = None
        self.file_design_adjustment_factor_by_topography = None
        self.file_line_interaction_metadata = None

        self.event_id = []  # list of tuples of event_name and scale
        self.wind_file_head = None
        self.wind_file_tail = None
        self.event_id_format = None
        self.line_interaction = {}

        self._topographic_multiplier = None
        self._design_value_by_line = None
        self._terrain_multiplier = None
        self._drag_height_by_type = None
        self._design_adjustment_factor_by_topography = None

        self._prob_line_interaction = None
        self._prob_line_interaction_metadata = None

        self._fragility_metadata = None
        self._fragility = None
        self._damage_states = None
        self._no_damage_states = None

        self._cond_collapse_prob_metadata = None
        self._cond_collapse_prob = None

        self._towers = None
        self._lines = None
        self._no_towers_by_line = None

        if not os.path.isfile(file_cfg):
            msg = '{} not found'.format(file_cfg)
            sys.exit(msg)
        else:
            self.path_cfg_file = os.sep.join(os.path.abspath(file_cfg).split(os.sep)[:-1])
            self.read_config()
            self.process_config()

    @property
    def drag_height_by_type(self):
        """read typical drag height by tower type
        :returns: drag height
        :rtype: dict

        """
        if self._drag_height_by_type is None:
            try:
                data = pd.read_csv(self.file_drag_height_by_type,
                                   skipinitialspace=True,
                                   index_col=0,
                                   names=['value'],
                                   skiprows=1)
            except IOError:
                msg = '{} not found'.format(self.file_drag_height_by_type)
                self.logger.critical(msg)
            else:
                self._drag_height_by_type = data['value'].to_dict()

        return self._drag_height_by_type

    @property
    def topographic_multiplier(self):
        """read topographic multiplier value from the input file
        :returns: topography value at each site
        :rtype: dict
        """
        if self._topographic_multiplier is None and self.options['adjust_design_by_topography']:
            try:
                data = pd.read_csv(self.file_topographic_multiplier,
                                   usecols=[0, 9, 10])
                data.columns = ['Name', 'Mh', 'Mhopp']
                data['topo'] = data[['Mh', 'Mhopp']].max(axis=1)
            except IOError:
                msg = '{} not found'.format(self.file_topo_multiplier)
                self.logger.error(msg)
            else:
                self._topographic_multiplier = data.set_index('Name').to_dict()['topo']

        return self._topographic_multiplier

    @property
    def design_adjustment_factor_by_topography(self):
        """read design wind speed adjustment based on topographic multiplier
        :returns: design speed adjustment factor
        :rtype: dict

        """
        if self._design_adjustment_factor_by_topography is None and self.options['adjust_design_by_topography']:

            if os.path.exists(self.file_design_adjustment_factor_by_topography):

                dic = ConfigParser.ConfigParser()
                dic.read(self.file_design_adjustment_factor_by_topography)

                data = {}
                for key, value in dic.items('main'):
                    try:
                        data[int(key)] = float(value)
                    except ValueError:
                        data[key] = np.array(
                            [float(x) for x in value.split(',')])

                assert len(data['threshold']) == len(data.keys()) - 2

                self._design_adjustment_factor_by_topography = data
                return self._design_adjustment_factor_by_topography

            else:
                msg = '{} not found'.format(
                    self.file_design_adjustment_factor_by_topography)
                self.logger.error(msg)

        return self._design_adjustment_factor_by_topography

    @property
    def terrain_multiplier(self):
        """
        read terrain multiplier (AS/NZS 1170.2:2011 Table 4.1)
        """
        if self._terrain_multiplier is None:
            try:
                data = pd.read_csv(self.file_terrain_multiplier,
                                   skipinitialspace=True,
                                   names=['height', 'tc1', 'tc2', 'tc3', 'tc4'],
                                   skiprows=1)
            except IOError:
                msg = '{} not found'.format(self.file_terrain_multiplier)
                self.logger.critical(msg)
            else:
                self._terrain_multiplier = data.to_dict('list')

        return self._terrain_multiplier

    @property
    def design_value_by_line(self):
        """read design values by line
        """
        if self._design_value_by_line is None:
            try:
                data = pd.read_csv(self.file_design_value,
                                   skipinitialspace=True,
                                   index_col=0,
                                   names=['speed', 'span', 'cat', 'level'],
                                   skiprows=1)
                self._design_value_by_line = data.transpose().to_dict()

            except IOError:
                msg = '{} not found'.format(self.file_design_value)
                self.logger.critical(msg)

        return self._design_value_by_line

    @property
    def fragility_metadata(self):
        """
        read collapse fragility parameter values
        """
        if self._fragility_metadata is None:

            metadata = ConfigParser.ConfigParser()
            ok = metadata.read(self.file_fragility_metadata)

            if ok:
                self._fragility_metadata = {}
                for item, value in metadata.items('main'):
                    if ',' in value:
                        self._fragility_metadata[item] = [x.strip() for x in value.split(',')]
                    else:
                        self._fragility_metadata[item] = value

                left = metadata.sections()
                left.remove('main')
                for name in left:
                    self._fragility_metadata[name] = dict(metadata.items(name))

            else:
                msg = '{} not found'.format(self.file_fragility_metadata)
                self.logger.critical(msg)

        return self._fragility_metadata

    @property
    def fragility(self):

        if self._fragility is None:

            path_metadata = os.path.dirname(
                os.path.realpath(self.file_fragility_metadata))
            _file = os.path.join(path_metadata, self.fragility_metadata['file'])
            try:
                self._fragility = pd.read_csv(_file, skipinitialspace=True)
            except IOError:
                self.logger.critical('{} not found'.format(_file))

        return self._fragility

    @property
    def damage_states(self):
        if self._damage_states is None:
            self._damage_states = self.fragility_metadata['limit_states']
        return self._damage_states

    @property
    def no_damage_states(self):
        if self._no_damage_states is None:
            self._no_damage_states = len(self.fragility_metadata['limit_states'])
        return self._no_damage_states

    @property
    def cond_collapse_prob(self):

        if self._cond_collapse_prob is None:

            metadata = ConfigParser.ConfigParser()
            ok = metadata.read(self.file_cond_collapse_prob_metadata)

            if ok:
                path_metadata = os.path.dirname(
                    os.path.realpath(self.file_cond_collapse_prob_metadata))

                kinds = [x.strip() for x in dict(metadata.items('main'))['list'].split(',')]
                self._cond_collapse_prob = {}
                for item in kinds:
                    _file = os.path.join(path_metadata,
                                         dict(metadata.items(item))['file'])
                    df = pd.read_csv(_file, skipinitialspace=1)
                    # df['start'] = df['start'].astype(np.int64)
                    # df['end'] = df['end'].astype(np.int64)
                    df['list'] = df.apply(lambda row:
                                          tuple(range(row['start'], row['end']+1)), axis=1)
                    self._cond_collapse_prob[item] = df
            else:
                msg = '{} not found'.format(self.file_cond_collapse_prob_metadata)
                self.logger.critical(msg)

        return self._cond_collapse_prob

    @property
    def cond_collapse_prob_metadata(self):
        """
        read condition collapse probability defined by tower function
        """
        if self._cond_collapse_prob_metadata is None:

            if not os.path.exists(self.file_cond_collapse_prob_metadata):
                msg = '{} not found'.format(self.file_cond_collapse_prob_metadata)
                self.logger.critical(msg)
            else:
                metadata = ConfigParser.ConfigParser()
                metadata.read(self.file_cond_collapse_prob_metadata)

                self._cond_collapse_prob_metadata = {}
                for item, value in metadata.items('main'):
                    if ',' in value:
                        self._cond_collapse_prob_metadata[item] = [x.strip() for x in value.split(',')]
                    else:
                        self._cond_collapse_prob_metadata[item] = value

                left = metadata.sections()
                left.remove('main')
                for item in left:
                    self._cond_collapse_prob_metadata[item] = dict(metadata.items(item))

        return self._cond_collapse_prob_metadata

    @property
    def prob_line_interaction_metadata(self):
        """
        read conditional parallel line interaction probability
        """
        if self.options['apply_line_interaction'] and self._prob_line_interaction_metadata is None:

            if not os.path.exists(self.file_line_interaction_metadata):
                msg = '{} not found'.format(self.file_line_interaction_metadata)
                self.logger.critical(msg)
            else:
                metadata = ConfigParser.ConfigParser()
                metadata.read(self.file_line_interaction_metadata)

                self._prob_line_interaction_metadata = {}
                for key, value in metadata.items('main'):
                    self._prob_line_interaction_metadata[key] = value

        return self._prob_line_interaction_metadata

    @property
    def prob_line_interaction(self):

        if self.options['apply_line_interaction'] and self._prob_line_interaction is None:

            path_metadata = os.path.dirname(
                os.path.realpath(self.file_line_interaction_metadata))
            _file = os.path.join(path_metadata,
                                 self.prob_line_interaction_metadata['file'])
            try:
                self._prob_line_interaction = pd.read_csv(_file, skipinitialspace=1)
            except IOError:
                msg = '{} not found'.format(self.file_line_interaction_metadata)
                self.logger.critical(msg)

        return self._prob_line_interaction

    @property
    def no_towers_by_line(self):

        if self._no_towers_by_line is None:

            self._no_towers_by_line = {k: v['no_towers']
                                       for k, v in self.lines.items()}

        return self._no_towers_by_line

    @property
    def towers(self):

        if self._towers is None:

            df = read_shape_file(self.file_shape_tower)

            # only selected lines
            df = df.loc[df.LineRoute.isin(self.selected_lines)]

            # coord, coord_lat_lon, point
            df = df.merge(df['Shapes'].apply(assign_shapely_point),
                          left_index=True, right_index=True)

            # design_span, design_level, design_speed, terrain_cat
            df = df.merge(df.apply(self.assign_design_values, axis=1),
                          left_index=True, right_index=True)

            # frag_scale, frag_arg, frag_func
            df = df.merge(df.apply(self.assign_fragility_parameters, axis=1),
                          left_index=True, right_index=True)
            df['file_wind_base_name'] = df['Name'].apply(
                lambda x: self.wind_file_head + x + self.wind_file_tail)

            df['height_z'] = df['Function'].apply(lambda x: self.drag_height_by_type[x])

            df['factor_10_to_z'] = df.apply(self.factor_10_to_z, axis=1)

            self._towers = df

        return self._towers

    @property
    def lines(self):

        if self._lines is None:

            df = read_shape_file(self.file_shape_line)

            # only selected lines
            df = df.loc[df.LineRoute.isin(self.selected_lines)]

            # add coord, coord_lat_lon, line_string
            df = df.merge(df['Shapes'].apply(assign_shapely_line),
                          left_index=True, right_index=True)

            df['name_output'] = df['LineRoute'].apply(
                lambda x: '_'.join(x for x in x.split() if x.isalnum()))

            df['no_towers'] = df['coord'].apply(lambda x: len(x))

            df['actual_span'] = df['coord_lat_lon'].apply(
                calculate_distance_between_towers)

            self._lines = df.set_index('LineRoute').to_dict('index')

        return self._lines

    def read_config(self):

        conf = ConfigParser.ConfigParser()
        conf.optionxform = str
        conf.read(self.file_cfg)

        # run_type
        for item in self.option_keys:
            try:
                self.options[item] = conf.getboolean('options', item)
            except ConfigParser.NoOptionError:
                msg = '{} not set in {}'.format(item, self.file_cfg)
                self.logger.critical(msg)

        # random seed
        self.read_random_seed(conf)

        # run_parameters
        key = 'run_parameters'
        self.no_sims = conf.getint(key, 'no_simulations')
        for x in conf.get(key, 'strainer').split(','):
            self.strainer.append(x.strip())
        for x in conf.get(key, 'selected_lines').split(','):
            self.selected_lines.append(x.strip())
        self.rel_tol = conf.getfloat(key, 'relative_tolerance')
        self.abs_tol = conf.getfloat(key, 'absolute_tolerance')

        # directories
        key = 'directories'
        for item in self.directory_keys:
            setattr(self, 'path_{}'.format(item),
                    os.path.join(self.path_cfg_file, conf.get(key, item)))

        # gis_data
        key = 'gis_data'
        for item in ['shape_tower', 'shape_line']:
            setattr(self, 'file_{}'.format(item),
                    os.path.join(self.path_gis_data, conf.get(key, item)))

        # wind_scenario
        for event_name in conf.options('wind_scenario'):
            for x in conf.get('wind_scenario', event_name).split(','):
                try:
                    self.event_id.append((event_name, float(x)))
                except ValueError:
                    pass

        # format
        file_format = conf.get('format', 'wind_scenario', 1)
        self.wind_file_head = file_format.split('%')[0]
        self.wind_file_tail = file_format.split(')')[-1]
        self.event_id_format = conf.get('format', 'event_id')

        # input
        key = 'input'
        for item in self.input_keys:
            try:
                setattr(self, 'file_{}'.format(item),
                        os.path.join(self.path_input, conf.get(key, item)))
            except ConfigParser.NoOptionError:
                self.logger.warning('{} is not set'.format(item))

        # line_interaction
        self.read_line_interaction(conf)

    def process_config(self):

        # max_adj
        for item in self.cond_collapse_prob_metadata['list']:
            self._cond_collapse_prob_metadata[item]['max_adj'] = \
                self.cond_collapse_prob[item]['end'].max()

        # id2name, ids, names
        for line_name, line in self.lines.items():
            line.update(self.sort_by_location(line=line, line_name=line_name))

        # actual_span, collapse_capacity
        self._towers = self.towers.merge(
            self.towers.apply(self.assign_collapse_capacity, axis=1),
            left_index=True, right_index=True)

        # cond_pc, max_no_adj_towers
        self._towers = self.towers.merge(
            self.towers.apply(self.assign_cond_collapse_prob, axis=1),
            left_index=True, right_index=True)

        # id_local, id_adj
        self._towers = self.towers.merge(
            self.towers.apply(self.assign_id_adj_towers, axis=1),
            left_index=True, right_index=True)

        # cond_pc_adj, cond_pc_adj_mc_rel_idx, cond_pc_adj_mc_cum_prob
        self._towers = self.towers.merge(
            self.towers.apply(assign_cond_pc_adj, axis=1),
            left_index=True, right_index=True)

    def sort_by_location(self, line_name, line):

        selected_towers = self.towers.loc[self.towers.LineRoute == line_name]

        id_by_line = selected_towers.index.tolist()
        name_by_line = selected_towers.Name.tolist()

        idx_sorted = []
        for _, tower in selected_towers.iterrows():
            temp = np.linalg.norm(line['coord'] - tower.coord, axis=1)
            id_closest = np.argmin(temp)
            ok = abs(temp[id_closest]) < 1.0e-4
            if not ok:
                msg = 'Can not locate {tower:} in {line:}'.format(
                    tower=tower.Name, line=line_name)
                self.logger.error(msg)
            idx_sorted.append(id_closest)

        return {'id2name': selected_towers['Name'].to_dict(),
                'ids': [id_by_line[x] for x in idx_sorted],
                'names': [name_by_line[x] for x in idx_sorted]}

    def assign_collapse_capacity(self, tower):
        """ calculate adjusted collapse wind speed for a tower
        Vc = Vd(h=z)/sqrt(u)
        where u = 1-k(1-Sw/Sd)
        Sw: actual wind span
        Sd: design wind span (defined by line route)
        k: 0.33 for a single, 0.5 for double circuit
        :rtype: float
        """

        selected_line = self.lines[tower.LineRoute]
        idx = selected_line['names'].index(tower.Name)
        actual_span = selected_line['actual_span'][idx]

        # calculate utilization factor
        # 1 in case sw/sd > 1
        u_factor = 1.0 - K_FACTOR[NO_CIRCUIT] * (1 - actual_span / tower.design_span)
        u_factor = min(1.0, u_factor)

        return pd.Series({'actual_span': actual_span,
                          'collapse_capacity': tower.design_speed / np.sqrt(u_factor)})

    def read_random_seed(self, conf):
        """
        read random seed info
        :param conf:
        :return:
        """

        if self.options['use_random_seed']:

            rnd_events = conf.get('random_seed', 'events').split(',')
            rnd_lines = conf.get('random_seed', 'lines').split(',')

            for rnd_event in rnd_events:
                (event_key, event_val) = split_str(rnd_event, ':')

                for rnd_line in rnd_lines:
                    (line_key, line_val) = split_str(rnd_line, ':')
                    self.seed.setdefault(event_key, {})[line_key] = \
                        event_val + line_val

    def read_line_interaction(self, conf):

        if self.options['apply_line_interaction']:

            selected_lines = self.selected_lines[:]
            for line in conf.options('line_interaction'):
                self.line_interaction[line] = [
                    x.strip() for x in conf.get('line_interaction', line).split(',')]
                try:
                    selected_lines.remove(line)
                except ValueError:
                    self.logger.error('{} is excluded in the simulation'.format(line))

            # check completeness
            if selected_lines:
                msg = 'No line interaction info provided for {}'.format(selected_lines)
                self.logger.error(msg)

    def assign_cond_collapse_prob(self, row):
        """ get dict of conditional collapse probabilities
        :return: cond_pc, max_no_adj_towers
        """
        kind = row[self.cond_collapse_prob_metadata['by']]
        df_prob = self.cond_collapse_prob[kind]
        att = self.cond_collapse_prob_metadata[kind]['by']
        att_type = self.cond_collapse_prob_metadata[kind]['type']

        tf = None
        if att_type == 'string':
            tf = df_prob[att] == row[att]
        elif att_type == 'numeric':
            tf = (df_prob[att + '_lower'] <= row[att]) & (df_prob[att + '_upper'] > row[att])

        return pd.Series({'cond_pc': dict(zip(df_prob.loc[tf, 'list'],
                                              df_prob.loc[tf, 'probability'])),
                          'max_no_adj_towers': self.cond_collapse_prob_metadata[kind]['max_adj']})

    def factor_10_to_z(self, row):
        """
        compute Mz,cat(h=z) / Mz,cat(h=10)/
        row: pandas series of tower
        """
        tc_str = 'tc' + str(row.terrain_cat)  # Terrain
        try:
            mzcat_z = np.interp(row.height_z,
                                self.terrain_multiplier['height'],
                                self.terrain_multiplier[tc_str])
        except KeyError:
            msg = '{} is undefined in {}'.format(tc_str, self.file_terrain_multiplier)
            self.logger.critical(msg)
        else:
            idx_10 = self.terrain_multiplier['height'].index(10)
            mzcat_10 = self.terrain_multiplier[tc_str][idx_10]
            return mzcat_z / mzcat_10

    def assign_design_values(self, row):
        """
        create pandas series of design level related values
        row: pandas series of line
        :return: pandas series of design_span, design_level, design_speed, and
                 terrain cat
        """

        # def adjust_design_speed(self):
        #     """ determine design speed """
        design_value = self.design_value_by_line[row['LineRoute']]
        design_speed = design_value['speed']

        if self.options['adjust_design_by_topography']:
            idx = (self.topographic_multiplier[row['Name']] >=
                   self.design_adjustment_factor_by_topography['threshold']).sum()
            design_speed *= self.design_adjustment_factor_by_topography[idx]

        return pd.Series({'design_span': design_value['span'],
                          'design_level': design_value['level'],
                          'design_speed': design_speed,
                          'terrain_cat': design_value['cat']})

    def assign_fragility_parameters(self, row):
        """
        assign fragility parameters by Type, Function, Dev Angle
        :param row: pandas series of tower
        :return: pandas series of frag_func, frag_scale, frag_arg
        """

        tf_array = np.ones((self.fragility.shape[0],), dtype=bool)
        for att, att_type in zip(self.fragility_metadata['by'],
                                 self.fragility_metadata['type']):
            if att_type == 'string':
                tf_array *= self.fragility[att] == row[att]
            elif att_type == 'numeric':
                tf_array *= (self.fragility[att + '_lower'] <= row[att]) & \
                            (self.fragility[att + '_upper'] > row[att])

        params = pd.Series({'frag_scale': dict(), 'frag_arg': dict(),
                            'frag_func': None})
        for ds in self.damage_states:
            try:
                idx = self.fragility[tf_array &
                                    (self.fragility['limit_states'] == ds)].index[0]
            except IndexError:
                msg = 'No matching fragility params for {}'.format(row.Name)
                self.logger.error(msg)
            else:
                fn_form = self.fragility.loc[idx, self.fragility_metadata['function']]
                params['frag_func'] = fn_form
                params['frag_scale'][ds] = self.fragility.loc[
                    idx, self.fragility_metadata[fn_form]['scale']]
                params['frag_arg'][ds] = self.fragility.loc[
                    idx, self.fragility_metadata[fn_form]['arg']]

        return params

    def assign_id_adj_towers(self, row):
        """
        assign id of adjacent towers which can be affected by collapse
        :param row: pandas series of tower
        :return: idl: local id, idg: global id, id_adj: adjacent ids
        """

        names = self.lines[row.LineRoute]['names']
        tower_id = names.index(row.Name)  # tower id in the line
        max_no_towers = self.no_towers_by_line[row.LineRoute]

        list_left = create_list_idx(tower_id, row.max_no_adj_towers,
                                    max_no_towers, -1)
        list_right = create_list_idx(tower_id, row.max_no_adj_towers,
                                     max_no_towers, 1)
        id_adj = list_left[::-1] + [tower_id] + list_right

        # assign -1 to strainer tower
        for i, tid in enumerate(id_adj):
            if tid >= 0:
                flag_strainer = self.towers.loc[
                    self.towers.Name == names[tid], 'Function'].isin(self.strainer).iloc[0]
                if flag_strainer:
                    id_adj[i] = -1

        return pd.Series({'id_adj': id_adj, 'id_line': tower_id})


def assign_cond_pc_adj(tower):
    """
    calculate conditional collapse probability of jth tower given ith tower
    P(j|i)

    :param tower: pandas series of tower
    :return:
    """
    # FIXME !! VERY HARD TO UNDERSTAND

    # rel_idx_strainer
    idx_m1 = np.array([i for i in range(len(tower.id_adj))
                       if tower.id_adj[i] == -1]) - tower.max_no_adj_towers

    try:
        max_neg = np.max(idx_m1[idx_m1 < 0]) + 1
    except ValueError:
        max_neg = -1 * tower.max_no_adj_towers

    try:
        min_pos = np.min(idx_m1[idx_m1 > 0])
    except ValueError:
        min_pos = tower.max_no_adj_towers + 1

    bound_ = set(range(max_neg, min_pos))

    cond_prob = {}
    for item in tower.cond_pc.keys():
        w = list(set(item).intersection(bound_))
        w.sort()
        w = tuple(w)
        if w in cond_prob:
            cond_prob[w] += tower.cond_pc[item]
        else:
            cond_prob[w] = tower.cond_pc[item]

    if (0,) in cond_prob:
        cond_prob.pop((0,))

    # sort by cond. prob
    rel_idx = sorted(cond_prob, key=cond_prob.get)
    prob = map(lambda v: cond_prob[v], rel_idx)

    cum_prob = np.cumsum(np.array(prob))

    # sum by node
    cond_pc_adj = dict()
    for key_ in cond_prob:
        for i in key_:
            try:
                cond_pc_adj[i] += cond_prob[key_]
            except KeyError:
                cond_pc_adj[i] = cond_prob[key_]

    if 0 in cond_pc_adj:
        cond_pc_adj.pop(0)

    return pd.Series({'cond_pc_adj': cond_pc_adj,
                      'cond_pc_adj_mc_rel_idx': rel_idx,
                      'cond_pc_adj_mc_cum_prob': cum_prob})


def create_list_idx(idx, no_towers, max_no_towers, flag_direction):
    """
    create list of adjacent towers in each direction (flag=+/-1)

    :param idx: tower index
    :param no_towers: no of towers on either side
    :param max_no_towers: max. no of towers
    :param flag_direction: +1/-l
    :return:
    """

    list_id = []
    for i in range(no_towers):
        idx += flag_direction
        if idx < 0 or idx > max_no_towers - 1:
            list_id.append(-1)
        else:
            list_id.append(idx)
    return list_id


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


def read_shape_file(file_shape):
    """
    read shape file and return data frame
    :param file_shape: Esri shape file
    :return data_frame: pandas.DataFrame
    """

    try:
        sf = shapefile.Reader(file_shape)
    except shapefile.ShapefileException:
        msg = '{} is not a valid shapefile'.format(file_shape)
        raise shapefile.ShapefileException(msg)
    else:
        shapes = sf.shapes()
        records = sf.records()
        fields = [x[0] for x in sf.fields[1:]]
        fields_type = [x[1] for x in sf.fields[1:]]

    dic_type = {'C': object, 'F': np.float64, 'N': np.int64}
    df = pd.DataFrame(records, columns=fields)

    for name, _type in zip(df.columns, fields_type):
        if df[name].dtype != dic_type[_type]:
            df[name] = df[name].astype(dic_type[_type])

    if 'Shapes' in fields:
        raise KeyError('Shapes is already in the fields')
    else:
        df['Shapes'] = shapes

    return df


def assign_shapely_point(shape):
    """
    create pandas series of coord, coord_lat_lon, and Point
    :param shape: Shapefile instance
    :return: pandas series of coord, coord_lat_lon, and Point
    """
    coord = np.array(shape.points[0])
    return pd.Series({'coord': coord,
                      'coord_lat_lon': coord[::-1],
                      'point': geometry.Point(coord)})


def assign_shapely_line(shape):
    """
    create pandas series of coord, coord_lat_lon, and line_string
    :param shape: Shapefile instance
    :return: pandas series of coord, coord_lat_lon, and line_string
    """
    coord = shape.points
    return pd.Series({'coord': np.array(coord),
                      'coord_lat_lon': np.array(coord)[:, ::-1].tolist(),
                      'line_string': geometry.LineString(coord)})


def calculate_distance_between_towers(coord_lat_lon):
    """
    calculate actual span between the towers
    :param coord_lat_lon: list of coord in lat, lon
    :return: array of actual span between towers
    """
    coord_lat_lon = np.stack(coord_lat_lon)
    dist_forward = np.zeros(len(coord_lat_lon) - 1)
    for i, (pt0, pt1) in enumerate(zip(coord_lat_lon[0:-1], coord_lat_lon[1:])):
        dist_forward[i] = great_circle(pt0, pt1).meters

    actual_span = 0.5 * (dist_forward[0:-1] + dist_forward[1:])
    actual_span = np.insert(actual_span, 0, [0.5 * dist_forward[0]])
    actual_span = np.append(actual_span, [0.5 * dist_forward[-1]])
    return actual_span


# TODO
def set_line_interaction(self):
    """maybe moved to config ??"""

    for line_name, line in self.lines.items():

        for tower in line.towers.itervalues():
            id_on_target_line = dict()

            for target_line in self.cfg.line_interaction[line_name]:

                line_string = self.lines[target_line].line_string
                line_coord = self.lines[target_line].coord

                closest_pt_on_line = line_string.interpolate(
                    line_string.project(tower.point))

                closest_pt_coord = np.array(closest_pt_on_line.coords)[0, :]

                closest_pt_lat_lon = closest_pt_coord[::-1]

                # compute distance
                dist_from_line = geopy.distance.great_circle(
                    tower.coord_lat_lon, closest_pt_lat_lon).meters

                if dist_from_line < tower.height:

                    id_on_target_line[target_line] = {
                        'id': find_id_nearest_pt(closest_pt_coord,
                                                 line_coord),
                        'vector': unit_vector(closest_pt_coord -
                                              tower.coord)}

            if id_on_target_line:
                tower.id_on_target_line = id_on_target_line


def get_cond_prob_line_interaction(self):
    """ get dict of conditional collapse probabilities
    :return: cond_prob_line
    """

    att = self.cfg.prob_line_interaction_metadata['by']
    att_type = self.cfg.prob_line_interaction_metadata['type']
    df_prob = self.cfg.prob_line_interaction

    tf_array = None
    if att_type == 'string':
        tf_array = df_prob == getattr(self, att)
    elif att_type == 'numeric':
        tf_array = (df_prob[att + '_lower'] <= self.ps_tower[att]) & \
                   (df_prob[att + '_upper'] > self.ps_tower[att])

    self.cond_pc_line_mc['no_collapse'] = \
        df_prob.loc[tf_array, 'no_collapse'].tolist()
    self.cond_pc_line_mc['cum_prob'] = np.cumsum(
        df_prob.loc[tf_array, 'probability']).tolist()


def find_id_nearest_pt(pt_coord, line_coord):
    """
    find the index of line_coord matching point coord
    :param pt_coord: (2,)
    :param line_coord: (#,2)
    :return: index of the nearest in the line_coord
    """
    assert pt_coord.shape == (2,)
    assert line_coord.shape[1] == 2
    diff = np.linalg.norm(line_coord - pt_coord, axis=1)
    return np.argmin(diff)


def unit_vector(vector):
    """
    create unit vector
    :param vector: tuple(x, y)
    :return: unit vector

    """
    return vector / np.linalg.norm(vector)
