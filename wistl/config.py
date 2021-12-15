import os
import bisect
import sys
import logging
import math
import yaml
import numpy as np
import pandas as pd
import configparser
import shapefile
from collections import namedtuple, defaultdict
from shapely import geometry
from geopy.distance import geodesic
from scipy import stats

from wistl.constants import K_FACTOR, NO_CIRCUIT, FIELDS_TOWER, params_event


OPTIONS = ['run_parallel', 'save_output', 'save_figure',
           'run_analytical', 'run_simulation', 'use_random_seed',
           'run_simulation_wo_cascading', 'adjust_design_by_topography',
           'apply_line_interaction', 'use_collapse_capacity']
DIRECTORIES = ['gis_data', 'wind_event_base', 'input', 'output']
GIS_DATA = ['shape_tower', 'shape_line']
FORMAT = ['wind_file', 'event_id']
INPUT_FILES = ['fragility_metadata', 'cond_prob_metadata',
               'terrain_multiplier', 'topographic_multiplier',
               'design_adjustment_factor_by_topography',
               'cond_prob_interaction_metadata']
SHAPEFILE_TYPE = {'C': object, 'F': np.float64, 'N': np.int64}


Event = namedtuple('Event', params_event)

# scenario -> damage scenario
# event -> wind event


class Config(object):
    """
    class to hold all configuration variables.
    """

    def __init__(self, file_cfg=None, logger=None):

        self.file_cfg = os.path.abspath(file_cfg)
        self.logger = logger or logging.getLogger(__name__)

        self.options = {}

        self.no_sims = None
        self.strainer = []
        self.selected_lines = []
        self.atol = None
        self.rtol = None
        self.dmg_threshold = None
        self.events = []  # list of tuples of event_name and scale
        self.line_interaction = {}

        self._topographic_multiplier = None
        self._design_value_by_line = None
        self._terrain_multiplier = None
        self._drag_height_by_type = None
        self._design_adjustment_factor_by_topography = None

        # cond prob wrt line_interaction
        self._cond_prob_interaction = None
        self._cond_prob_interaction_metadata = None

        self._fragility_metadata = None
        self._fragility = None   # pandas.DataFrame
        self._damage_states = None
        self._no_damage_states = None
        self._non_collapse = None

        self._cond_prob_metadata = None
        self._cond_prob = None  # dict of pd.DataFrame

        if not os.path.isfile(file_cfg):
            self.logger.error(f'{file_cfg} not found')
        else:
            self.read_config()

    #def __getstate__(self):
    #    d = self.__dict__.copy()
    #    if 'logger' in d:
    #        d['logger'] = d['logger'].name
    #    return d

    #def __setstate__(self, d):
    #    if 'logger' in d:
    #        d['logger'] = logging.getLogger(d['logger'])
    #    self.__dict__.update(d)

    @property
    def drag_height_by_type(self):
        """read typical drag height by tower type
        :returns: drag height
        :rtype: dict

        """
        if self._drag_height_by_type is None:
            if os.path.exists(self.file_drag_height_by_type):
                self._drag_height_by_type = h_drag_height_by_type(self.file_drag_height_by_type)
            else:
                msg = f'{self.file_drag_height_by_type} not found'
                self.logger.critical(msg)

        return self._drag_height_by_type

    @property
    def topographic_multiplier(self):
        """read topographic multiplier value from the input file
        :returns: topography value at each site
        :rtype: dict
        """
        if self._topographic_multiplier is None and self.options['adjust_design_by_topography']:
            if os.path.exists(self.file_topographic_multiplier):
                self._topographic_multiplier = h_topographic_multiplier(
                    self.file_topographic_multiplier)
            else:
                msg = f'{self.file_topographic_multiplier} not found'
                self.logger.critical(msg)

        return self._topographic_multiplier

    @property
    def design_adjustment_factor_by_topography(self):
        """read design wind speed adjustment based on topographic multiplier
        :returns: design speed adjustment factor
        :rtype: dict

        """
        if self._design_adjustment_factor_by_topography is None and self.options['adjust_design_by_topography']:
            if os.path.exists(self.file_design_adjustment_factor_by_topography):
                self._design_adjustment_factor_by_topography = \
                    h_design_adjustment_factor_by_topography(
                        self.file_design_adjustment_factor_by_topography)
            else:
                msg = f'{self.file_design_adjustment_factor_by_topography} not found'
                self.logger.critical(msg)

        return self._design_adjustment_factor_by_topography

    @property
    def terrain_multiplier(self):
        """
        read terrain multiplier (AS/NZS 1170.2:2011 Table 4.1)
        """
        if self._terrain_multiplier is None:
            if os.path.exists(self.file_terrain_multiplier):
                self._terrain_multiplier = h_terrain_multiplier(
                    self.file_terrain_multiplier)
            else:
                msg = f'{self.file_terrain_multiplier} not found'
                self.logger.critical(msg)

        return self._terrain_multiplier

    @property
    def design_value_by_line(self):
        """read design values by line
        """
        if self._design_value_by_line is None:
            if os.path.exists(self.file_design_value_by_line):
                self._design_value_by_line = h_design_value_by_line(self.file_design_value_by_line)
            else:
                msg = f'{self.file_design_value_by_line} not found'
                self.logger.critical(msg)

        return self._design_value_by_line

    @property
    def fragility_metadata(self):
        """
        read collapse fragility parameter values
        """
        if self._fragility_metadata is None:
            try:
                with open(self.file_fragility_metadata, 'r') as ymlfile:
                    tmp = yaml.load(ymlfile, Loader=yaml.FullLoader)
            except IOError:
                msg = f'{self.file_fragility_metadata} not found'
                self.logger.critical(msg)
            else:
                self._fragility_metadata = nested_dic(tmp)
        return self._fragility_metadata

    @property
    def fragility(self):
        if self._fragility is None:
            path_metadata = os.path.dirname(
                os.path.realpath(self.file_fragility_metadata))
            _file = os.path.join(path_metadata, self.fragility_metadata['main']['file'])

            if os.path.exists(_file):
                self._fragility = h_fragility(_file)
            else:
                self.logger.critical(f'{_file} not found')

        return self._fragility

    @property
    def damage_states(self):
        if self._damage_states is None:
            self._damage_states = self.fragility_metadata['main']['limit_states']
        return self._damage_states

    @property
    def no_damage_states(self):
        if self._no_damage_states is None:
            self._no_damage_states = len(self.damage_states)
        return self._no_damage_states

    @property
    def non_collapse(self):
        if self._non_collapse is None:
            self._non_collapse = self.damage_states[:]
            self._non_collapse.remove('collapse')
        return self._non_collapse

    @property
    def cond_prob_metadata(self):
        """
        read condition collapse probability defined by tower function
        """
        if self._cond_prob_metadata is None:

            if os.path.exists(self.file_cond_prob_metadata):
                self._cond_prob_metadata = read_yml_file(
                    self.file_cond_prob_metadata)
            else:
                msg = f'{self.file_cond_prob_metadata} not found'
                self.logger.critical(msg)
        return self._cond_prob_metadata

    @property
    def cond_prob(self):

        if self._cond_prob is None:

            _file = os.path.join(self.cond_prob_metadata['path'],
                                 self.cond_prob_metadata['file'])

            self._cond_prob = h_cond_prob(_file)

        return self._cond_prob

    @property
    def cond_prob_interaction_metadata(self):
        """
        read conditional line interaction probability
        """
        if self.options['apply_line_interaction'] and self._cond_prob_interaction_metadata is None:

            if not os.path.exists(self.file_cond_prob_interaction_metadata):
                msg = f'{self.file_cond_prob_interaction_metadata} not found'
                self.logger.critical(msg)
            else:
                self._cond_prob_interaction_metadata = read_yml_file(
                        self.file_cond_prob_interaction_metadata)

        return self._cond_prob_interaction_metadata

    @property
    def cond_prob_interaction(self):

        if self._cond_prob_interaction is None and self.cond_prob_interaction_metadata:

            _file = os.path.join(self.cond_prob_interaction_metadata['path'],
                                 self.cond_prob_interaction_metadata['file'])
            with open(_file, 'r') as ymlfile:
                self._cond_prob_interaction = yaml.load(ymlfile, Loader=yaml.FullLoader)

        return self._cond_prob_interaction

    def read_config(self):

        conf = configparser.ConfigParser()
        conf.optionxform = str
        conf.read(self.file_cfg)

        self.path_cfg = os.sep.join(self.file_cfg.split(os.sep)[:-1])

        # options 
        for item in OPTIONS:
            try:
                self.options[item] = conf.getboolean('options', item)
            except configparser.NoOptionError:
                msg = f'{item} not set in [options] in {self.file_cfg}'
                self.logger.critical(msg)

        # set FIELDS_TOWER 
        if self.options['use_collapse_capacity'] and 'collapse_capacity' not in FIELDS_TOWER:
            FIELDS_TOWER.append('collapse_capacity')

        # run_parameters
        key = 'run_parameters'
        self.no_sims = conf.getint(key, 'no_simulations')
        for x in conf.get(key, 'strainer').split(','):
            self.strainer.append(x.strip())
        for x in conf.get(key, 'selected_lines').split(','):
            self.selected_lines.append(x.strip())
        self.atol = conf.getfloat(key, 'atol')
        self.rtol = conf.getfloat(key, 'rtol')
        self.dmg_threshold = conf.getfloat(key, 'dmg_threshold')

        # directories: gis_data, wind_event_base, input, output
        key = 'directories'
        for item in DIRECTORIES:
            try:
                setattr(self, f'path_{item}',
                        os.path.join(self.path_cfg, conf.get(key, item)))
            except configparser.NoOptionError:
                msg = f'{item} not set in [{key}] in {self.file_cfg}'
                self.logger.critical(msg)

            else:
                if not os.path.exists(getattr(self, f'path_{item}')):
                    if item == 'output':
                        os.makedirs(self.path_output)
                        self.logger.info(f'{self.path_output} is created')
                    else:
                        self.logger.error(f'Invalid path for {item}')

        # gis_data
        key = 'gis_data'
        for item in GIS_DATA:
            try:
                tmp = [os.path.join(self.path_gis_data, x.strip())
                       for x in conf.get(key, item).split(',')]
                setattr(self, f'file_{item}', tmp)
            except configparser.NoOptionError:
                msg = f'{item} not set in [{key}] in {self.file_cfg}'
                self.logger.critical(msg)

        # format
        for item in FORMAT:
            key = f'{item}_format'
            try:
                setattr(self, key, conf.get('format', item))
            except configparser.NoOptionError:
                msg = f'{item} not set in [format] in {self.file_cfg}'
                self.logger.critical(msg)

        # wind_event
        self.read_wind_event(conf)

        # input
        key = 'input_files'
        for item in INPUT_FILES:
            try:
                setattr(self, f'file_{item}',
                        os.path.join(self.path_input, conf.get(key, item)))
            except configparser.NoOptionError:
                if item == 'cond_prob_interaction_metadata':
                    if self.options['apply_line_interaction']:
                        self.logger.warning(f'{item} is not seg in [{key}] in {self.file_cfg}')
                elif 'topograph' in item:
                    if self.options['adjust_design_by_topography']:
                        self.logger.warning(f'{item} is not set in [{key}] in {self.file_cfg}')
                else:
                    self.logger.warning(f'{item} is not set in [{key}] in {self.file_cfg}')

        # line_interaction
        self.read_line_interaction(conf)


    def read_wind_event(self, conf):

        # read random_seed
        seeds = []
        if self.options['use_random_seed']:
            try:
                for event_name in conf.options('random_seed'):
                    for x in conf.get('random_seed', event_name).split(','):
                        try:
                            seeds.append(int(x))
                        except ValueError:
                            self.logger.error('Invalid random_seed')
            except configparser.NoOptionError:
                self.logger.critical(f'[random_seed] is not set in {self.file_cfg}')

        # read wind_event
        k = -1
        for i, event_name in enumerate(conf.options('wind_event')):
            for x in conf.get('wind_event', event_name).split(','):
                k += 1
                if seeds:
                    seed = seeds[k]
                else:
                    seed = k

                try:
                    _path = os.path.join(self.path_wind_event_base,
                                         event_name)
                    _id = self.event_id_format.format(
                        event_name=event_name, scale=float(x))
                    self.events.append(Event(_id,
                                             _path,
                                             event_name,
                                             float(x),
                                             seed))
                except ValueError:
                    msg = 'Invalid wind_event input'
                    self.logger.error(msg)

        self.no_events = len(self.events)

    def read_line_interaction(self, conf):

        if self.options['apply_line_interaction']:

            selected_lines = self.selected_lines[:]
            for line in conf.options('line_interaction'):
                self.line_interaction[line] = [
                    x.strip() for x in conf.get('line_interaction', line).split(',')]
                try:
                    selected_lines.remove(line)
                except ValueError:
                    msg = f'{line} is excluded in the simulation'
                    self.logger.error(msg)

            # check completeness
            if selected_lines:
                msg = f'missing line_interactation for {selected_lines}'
                self.logger.error(msg)

    #def assign_design_values(self, row):
    #    """
    #    create pandas series of design level related values
    #    row: pandas series of tower
    #    :return: pandas series of design_span, design_level, design_speed, and
    #             terrain cat
    #    """

    #    try:
    #        design_value = self.design_value_by_line[row['lineroute']]
    #    except KeyError:
    #        msg = f"{row['lineroute']} is undefined in {self.file_design_value}"
    #        self.logger.critical(msg)
    #    else:
    #        design_speed = design_value['design_speed']

    #        if self.options['adjust_design_by_topography']:
    #            idx = (self.topographic_multiplier[row['name']] >=
    #                   self.design_adjustment_factor_by_topography['threshold']).sum()
    #            design_speed *= self.design_adjustment_factor_by_topography[idx]

    #        return pd.Series({'design_span': design_value['design_span'],
    #                          'design_level': design_value['design_level'],
    #                          'design_speed': design_speed,
    #                          'terrain_cat': design_value['terrain_cat']})

def _set_towers_by_line(cfg):

    df = pd.DataFrame(None)
    for _file in cfg.file_shape_tower:
        if '.shp' in _file:
            df = df.append(read_shape_file(_file))
        else:
            tmp = pd.read_csv(_file, skipinitialspace=True, usecols=FIELDS_TOWER)
            df = df.append(tmp)

    df.set_index('name', inplace=True, drop=False)

    # set dtype of lineroute chr
    df['lineroute'] = df['lineroute'].astype(str)

    # only selected lines
    df = df.loc[df['lineroute'].isin(cfg.selected_lines)]

    # coord, coord_lat_lon, point
    df = df.merge(df.apply(assign_shapely_point, axis=1),
                  left_index=True, right_index=True)

    # design_span, design_level, design_speed, terrain_cat
    #df = df.merge(df.apply(self.assign_design_values, axis=1),
    #              left_index=True, right_index=True)

    # frag_dic
    df = df.merge(df.apply(assign_fragility_parameters, args=(cfg,), axis=1),
                  left_index=True, right_index=True)

    df['file_wind_base_name'] = df['name'].apply(
        lambda x: cfg.wind_file_format.format(tower_name=x))

    #df['height_z'] = df['function'].apply(lambda x: self.drag_height_by_type[x])

    df['ratio_z_to_10'] = df.apply(ratio_z_to_10, args=(cfg,), axis=1)

    towers_by_line = {}
    for name, grp in df.groupby('lineroute'):
        towers_by_line[name] = grp.to_dict('index')

    return towers_by_line


def _set_lines(cfg):

    df = pd.DataFrame(None)
    for _file in cfg.file_shape_line:
        df = df.append(read_shape_file(_file))
    df.set_index('lineroute', inplace=True, drop=False)

    # only selected lines
    df = df.loc[df['lineroute'].isin(cfg.selected_lines)].copy()

    # add coord, coord_lat_lon, line_string
    df = df.merge(df['shapes'].apply(assign_shapely_line),
                  left_index=True, right_index=True)

    df['name_output'] = df['lineroute'].apply(
        lambda x: '_'.join(x for x in x.split() if x.isalnum()))

    df['no_towers'] = df['coord'].apply(lambda x: len(x))

    df['actual_span'] = df['coord_lat_lon'].apply(
        calculate_distance_between_towers)

    df['seed'] = np.arange(len(df))

    lines = df.set_index('lineroute').to_dict('index')

    return lines


def set_towers_and_lines(cfg):

    towers_by_line = _set_towers_by_line(cfg)

    lines = _set_lines(cfg)

    # add no_towers_by_line
    #cfg.no_towers_by_line = {k: len(v) for k, v in towers_by_line.items()}

    # id2name, ids, names
    for line_name, line in lines.items():
        line.update(sort_by_location(towers_by_line=towers_by_line, line=line))

    # actual_span, collapse_capacity
    for line_name, grp in towers_by_line.items():

        for tower_id, tower in grp.items():

            # actual_span, collapse_capacity
            if not cfg.options['use_collapse_capacity']:
                tower.update(assign_collapse_capacity(tower=tower, lines=lines))

            # cond_pc, max_no_adj_towers
            tower.update(assign_cond_pc(tower=tower, cfg=cfg))

            # idl, id_adj
            tower.update(assign_id_adj_towers(tower=tower,
                                              towers_by_line=towers_by_line,
                                              lines=lines,
                                              cfg=cfg))

            # cond_pc_adj, cond_pc_adj_sim_idx, cond_pc_adj_sim_prob
            tower.update(assign_cond_pc_adj(tower=tower))

    return towers_by_line, lines


#def no_towers_by_line(lines):
#
#    return {k: v['no_towers'] for k, v in lines.items()}


def assign_fragility_parameters(tower, cfg):
    """
    assign fragility parameters by Type, Function, Dev Angle
    :param tower: pandas series of tower
    :return: pandas series of frag_func, frag_scale, frag_arg
    """
    return pd.Series({'frag_dic':
        get_value_given_conditions(cfg.fragility_metadata['fragility'],
                                   cfg.fragility, tower)})


def assign_id_adj_towers(tower, towers_by_line, lines, cfg):
    """
    assign id of adjacent towers which can be affected by collapse
    :param tower: pandas series of tower
    :return: idl: local id, idg: global id, id_adj: adjacent ids
    """

    names = lines[tower['lineroute']]['names']
    idl = names.index(tower['name'])  # tower id in the line
    max_no_towers = lines[tower['lineroute']]['no_towers']

    list_left = create_list_idx(idl, tower['max_no_adj_towers'],
                                max_no_towers, -1)
    list_right = create_list_idx(idl, tower['max_no_adj_towers'],
                                 max_no_towers, 1)
    id_adj = list_left[::-1] + [idl] + list_right

    # assign -1 to strainer tower
    for i, idx in enumerate(id_adj):
        if idx >= 0:
            global_id = lines[tower['lineroute']]['name2id'][names[idx]]
            _tower = towers_by_line[tower['lineroute']][global_id]
            flag_strainer = _tower['function'] in cfg.strainer

            if flag_strainer:
                id_adj[i] = -1

    return {'id_adj': id_adj, 'idl': idl}


def assign_cond_pc(tower, cfg):
    """ get dict of conditional collapse probabilities
    :return: cond_pc, max_no_adj_towers
    """

    cond_pc = get_value_given_conditions(cfg.cond_prob_metadata['probability'],
                                         cfg.cond_prob, tower)

    max_no_adj_towers = sorted(cond_pc.keys(), key=lambda x: len(x))[-1][-1]
    return {'cond_pc': cond_pc, 'max_no_adj_towers': max_no_adj_towers}


def ratio_z_to_10(tower, cfg):
    """
    compute Mz,cat(h=z) / Mz,cat(h=10)/
    tower: pandas series of tower
    """
    tc_str = 'tc' + str(tower['terrain_cat'])  # Terrain
    try:
        mzcat_z = np.interp(tower['height_z'],
                            cfg.terrain_multiplier['height'],
                            cfg.terrain_multiplier[tc_str])
    except KeyError:
        msg = f'{tc_str} is undefined in {self.file_terrain_multiplier}'
        logger.critical(msg)
    else:
        idx_10 = cfg.terrain_multiplier['height'].index(10)
        mzcat_10 = cfg.terrain_multiplier[tc_str][idx_10]
        return mzcat_z / mzcat_10


def assign_collapse_capacity(tower, lines):
    """ calculate adjusted collapse wind speed for a tower
    Vc = Vd(h=z)/sqrt(u)
    where u = 1-k(1-Sw/Sd)
    Sw: actual wind span
    Sd: design wind span (defined by tower)
    k: 0.33 for a single, 0.5 for double circuit
    :rtype: float
    """

    selected_line = lines[tower['lineroute']]
    idx = selected_line['names'].index(tower['name'])
    actual_span = selected_line['actual_span'][idx]

    # calculate utilization factor
    # 1 in case sw/sd > 1
    u_factor = 1.0 - K_FACTOR[NO_CIRCUIT] * (1.0 - actual_span / tower['design_span'])
    u_factor = min(1.0, u_factor)

    return {'actual_span': actual_span,
            'u_factor': u_factor,
            'collapse_capacity': tower['design_speed'] / math.sqrt(u_factor)}


def sort_by_location(line, towers_by_line):

    id_by_line = []
    name_by_line = []
    idx_sorted = []

    for tower_id, tower in towers_by_line[line['linename']].items():

        id_by_line.append(tower_id)
        name_by_line.append(tower['name'])

        temp = np.linalg.norm(line['coord'] - tower['coord'], axis=1)
        id_closest = np.argmin(temp)
        ok = abs(temp[id_closest]) < 1.0e-4
        if not ok:
            msg = f"Can not locate tower:{tower['name']} in line:{line['linename']}"
            logger.error(msg)
        idx_sorted.append(id_closest)

    return {'id2name': {k: v for k, v in zip(id_by_line, name_by_line)},
            'ids': [x for _, x in sorted(zip(idx_sorted, id_by_line))],
            'names': [x for _, x in sorted(zip(idx_sorted, name_by_line))],
            'name2id': {k: v for k, v in zip(name_by_line, id_by_line)}}


def assign_cond_pc_adj(tower):
    """
    calculate conditional collapse probability of jth tower given ith tower
    P(j|i)

    :param tower: pandas series of tower
    :return:
    """

    # list of relative index of -1 (center is zero)
    idx_m1 = np.array([i for i in range(len(tower['id_adj']))
                       if tower['id_adj'][i] == -1]) - tower['max_no_adj_towers']

    # range of index to be ignored
    try:
        max_neg = np.max(idx_m1[idx_m1 < 0]) + 1
    except ValueError:
        max_neg = -1 * tower['max_no_adj_towers']

    try:
        min_pos = np.min(idx_m1[idx_m1 > 0])
    except ValueError:
        min_pos = tower['max_no_adj_towers'] + 1

    bound = set(range(max_neg, min_pos))

    # option 1: include all patterns
    cond_prob = defaultdict(float)
    for key, value in tower['cond_pc'].items():
        rel_idx = sorted(set(key).intersection(bound))
        abs_idx = tuple(tower['id_adj'][j + tower['max_no_adj_towers']]
                        for j in rel_idx if j != 0)
        cond_prob[abs_idx] += value

    if () in cond_prob:
        cond_prob.pop(())

    # sort by cond. prob
    abs_idx = sorted(cond_prob, key=cond_prob.get)
    prob = list(map(lambda v: cond_prob[v], abs_idx))
    cum_prob = np.cumsum(np.array(prob))

    # sum of cond. prob by each tower
    cond_pc_adj = defaultdict(float)
    for key, value in zip(abs_idx, prob):
        for idx in key:
            cond_pc_adj[idx] += value

    return {'cond_pc_adj': cond_pc_adj,
            'cond_pc_adj_sim_idx': abs_idx,
            'cond_pc_adj_sim_prob': cum_prob}


def create_list_idx(idx, no_towers, max_no_towers, flag_direction):
    """
    create list of adjacent towers in each direction (flag=+/-1)

    :param idx: tower index (starting from 0)
    :param no_towers: no of towers on either side (depending tower type)
    :param max_no_towers: no of towers in the line
    :param flag_direction: +1/-l
    :return:
    """

    if flag_direction > 0:
        list_id = range(idx + 1, no_towers + idx + 1)
    else:
        list_id = range(idx - 1, idx - no_towers - 1, -1)

    list_id = [-1 if (x < 0) or (x > max_no_towers - 1)
               else x for x in list_id]

    return list_id


def split_str(string, str_split):
    """
    split string with split_str
    :param string: string
    :param str_split: split
    :return: tuple of str and integer or float
    """
    _list = string.split(str_split)
    try:
        return _list[0].strip(), int(_list[1])
    except ValueError:
        try:
            return _list[0].strip(), float(_list[1])
        except ValueError:
            return _list[0].strip(), _list[1].strip()


def read_shape_file(file_shape):
    """
    read shape file and return data frame
    :param file_shape: Esri shape file
    :return data_frame: pandas.DataFrame
    """
    try:
        with shapefile.Reader(file_shape) as sf:
            shapes = sf.shapes()
            records = [x[:] for x in sf.records()]
            fields = [x[0].lower() for x in sf.fields[1:]]
            fields_type = [x[1] for x in sf.fields[1:]]
    except shapefile.ShapefileException:
        msg = f'{file_shape} is not a valid shapefile'
        raise shapefile.ShapefileException(msg)
    else:
        df = pd.DataFrame(records, columns=fields)

        # assert dic_type of lan, long should be float rather than integer
        if 'latitude' in fields:
            for x in ['latitude', 'longitude']:
                fields_type[fields.index(x)] = 'F'

        for name, _type in zip(df.columns, fields_type):
            if df[name].dtype != SHAPEFILE_TYPE[_type]:
                df[name] = df[name].astype(SHAPEFILE_TYPE[_type])

        if 'shapes' in fields:
            raise KeyError('shapes is already in the fields')
        else:
            df['shapes'] = shapes

    return df


def assign_shapely_point(row):
    """
    create pandas series of coord, coord_lat_lon, and Point
    :param shape: Shapefile instance
    :return: pandas series of coord, coord_lat_lon, and Point
    """
    coord = [row['longitude'], row['latitude']]
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
        dist_forward[i] = geodesic(pt0, pt1).meters

    actual_span = 0.5 * (dist_forward[0:-1] + dist_forward[1:])
    actual_span = np.insert(actual_span, 0, [0.5 * dist_forward[0]])
    actual_span = np.append(actual_span, [0.5 * dist_forward[-1]])
    return actual_span


def find_id_nearest_pt(pt_coord, line_coord):
    """
    find the index of line_coord matching point coord
    :param pt_coord: (2,)
    :param line_coord: (#,2)
    :return: index of the nearest in the line_coord
    """

    logger = logging.getLogger(__name__)

    if not isinstance(pt_coord, np.ndarray):
        pt_coord = np.array(pt_coord)
    try:
        assert pt_coord.shape == (2,)
    except AssertionError:
        logger.error(f'Invalid pt_coord: {pt_coord}')

    if not isinstance(line_coord, np.ndarray):
        line_coord = np.array(line_coord)
    try:
        assert line_coord.shape[1] == 2
    except AssertionError:
        logger.error(f'Invalid line_coord: {line_coord}')

    diff = np.linalg.norm(line_coord - pt_coord, axis=1)
    return np.argmin(diff)


def unit_vector(vector):
    """
    create unit vector
    :param vector: tuple(x, y)
    :return: unit vector

    """
    return vector / np.linalg.norm(vector)


def unit_vector_by_bearing(angle_deg):
    """
    return unit vector given bearing
    :param angle_deg: 0-360
    :return: unit vector given bearing in degree
    """
    angle_rad = np.deg2rad(angle_deg)
    return np.array([-1.0 * np.sin(angle_rad), -1.0 * np.cos(angle_rad)])


def angle_between_unit_vectors(v1, v2):
    """
    compute angle between two unit vectors
    :param v1: vector 1
    :param v2: vector 2
    :return: the angle in degree between vectors 'v1' and 'v2'

    """
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

def h_fragility(_file):
    with open(_file, 'r') as ymlfile:
        out = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return nested_dic(out)

def nested_dic(d):
    assert isinstance(d, dict)
    for k, v in d.items():
        if isinstance(v, dict):
            nested_dic(v)
        else:
            # clean up value
            if (k != 'file') & (',' in v):
                d[k] = [x.strip() for x in v.split(',')]
    return d


def h_design_adjustment_factor_by_topography(_file):

    dic = configparser.ConfigParser()
    dic.read(_file)

    data = {}
    for key, value in dic.items('main'):
        try:
            data[int(key)] = float(value)
        except ValueError:
            data[key] = np.array(
                [float(x) for x in value.split(',')])

    assert len(data['threshold']) == len(data.keys()) - 2

    return data

def h_terrain_multiplier(_file):
    data = pd.read_csv(_file,
                       skipinitialspace=True,
                       names=['height', 'tc1', 'tc2', 'tc3', 'tc4'],
                       skiprows=1)
    return data.to_dict('list')

#def h_design_value_by_line(_file):
#    data = pd.read_csv(_file, index_col=0, skipinitialspace=True)
#    return data.transpose().to_dict()

def h_topographic_multiplier(_file):
    data = pd.read_csv(_file, index_col=0)
    return data.max(axis=1).to_dict()

#def h_drag_height_by_type(_file):
#    data = pd.read_csv(_file,
#                       skipinitialspace=True,
#                       index_col=0,
#                       names=['value'],
#                       skiprows=1)
#    return data['value'].to_dict()

def read_yml_file(_file):

    with open(_file, 'r') as ymlfile:
        out = yaml.load(ymlfile, Loader=yaml.FullLoader)

    out['path'] = os.path.dirname(os.path.realpath(_file))

    return out


def h_cond_prob(_file):

    def h_dic(d):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                h_dic(v)
            else:
                # clean up value
                if ',' in k:
                    tmp = [int(x.strip()) for x in k.split(',')]
                    key = tuple(range(tmp[0], tmp[1]+1))
                    d[key] = d.pop(k)

    with open(_file, 'r') as ymlfile:
        out = yaml.load(ymlfile, Loader=yaml.FullLoader)
    h_dic(out)

    return out


def get_value_given_conditions(metadata, prob, tower):
    """ find prob/fragility meeting condition given metadata
    """
    logger = logging.getLogger(__name__)

    def h_dic(metadata, prob, tower):

        nonlocal attr
        nonlocal result

        for k, v in metadata.items():

            if k in tower:
                key = tower[k]

                if isinstance(v[key], dict):
                    h_dic(v[key], prob[key], tower)

                else:
                    attr = v[key]
                    result = prob[key]
            else:
                h_dic(v, prob, tower)

    attr = None
    result = None

    h_dic(metadata, prob, tower)

    idx = tower[attr]
    if not isinstance(idx, str):
        sorted_keys = sorted(result.keys())
        if idx <= sorted_keys[-1]:
            loc = min(bisect.bisect_right(sorted_keys, idx),
                  len(sorted_keys) - 1)
        else:
            loc = 0  # take the first one 
            logger.critical(f'unable to assign cond_pc for tower {tower["name"]}')
        idx = sorted_keys[loc]
    return result[idx]


# FIXME
def assign_target_line(self):

    for line_name, line in self.lines.items():

        line['target_no_towers'] = {x: self.lines[x]['no_towers']
                                    for x in self.line_interaction[line_name]}

        for _, tower in self.towers_by_line[line_name].items():

            # cond_pc_interaction, cond_pc_interaction_prob
            tower.update(self.assign_cond_pc_interaction(tower=tower))

            target_line = {}

            for target in self.line_interaction[line_name]:

                line_string = self.lines[target]['line_string']
                line_coord = self.lines[target]['coord']

                closest_pt_on_line = line_string.interpolate(
                    line_string.project(tower['point']))

                closest_pt_coord = np.array(closest_pt_on_line.coords)[0, :]

                closest_pt_lat_lon = closest_pt_coord[::-1]

                # compute distance
                dist_from_line = geodesic(
                    tower['coord_lat_lon'], closest_pt_lat_lon).meters

                if dist_from_line < tower['height']:

                    target_line[target] = {
                        'id': find_id_nearest_pt(closest_pt_coord,
                                                 line_coord),
                        'vector': unit_vector(closest_pt_coord -
                                              tower['coord'])}

            tower['target_line'] = target_line


# FIXME
def assign_cond_pc_interaction(self, tower):
    """ get dict of conditional probabilities due to line interaction
    :return: cond_pc, max_no_adj_towers
    """
    try:
        cond_pc = get_value_given_conditions(
                self.cond_prob_interaction_metadata['probability'],
                self.cond_prob_interaction, tower)
    except TypeError:
        return {'cond_pc_interaction_no': None,
                'cond_pc_interaction_cprob': None,
                'cond_pc_interaction_prob': None}
    else:
        tmp = [(k, v) for k, v in cond_pc.items()]
        return {'cond_pc_interaction_no': [x[0] for x in tmp],
                'cond_pc_interaction_cprob': np.cumsum([x[1] for x in tmp]),
                'cond_pc_interaction_prob': {x[0]:x[1] for x in tmp}}



#FIXME
def read_cond_prob_interaction_metadata(cfg):
    """
    read conditional line interaction probability
    """
    if options['apply_line_interaction'] and cond_prob_interaction_metadata is None:

        if not os.path.exists(self.file_cond_prob_interaction_metadata):
            msg = f'{self.file_cond_prob_interaction_metadata} not found'
            self.logger.critical(msg)
        else:
            self._cond_prob_interaction_metadata = read_yml_file(
                    self.file_cond_prob_interaction_metadata)

    return self._cond_prob_interaction_metadata

