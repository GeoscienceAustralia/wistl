#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import logging
import scipy.stats as stats

from wistl.constants import ATOL, RTOL


class Tower(object):
    """
    class Tower
    Tower class represent an individual tower.
    """

    registered = ['axisaz',   # azimuth of strong axis
                  #'barangay',
                  #'comment',
                  #'constcost',
                  #'consttype',
                  # 'devangle',  # deviation angle
                  'function',
                  #'ht_source',
                  # 'height',
                  # 'latitude',
                  'lineroute',
                  #'locsource',
                  'longitude',
                  #'mun',
                  #'number',
                  'name',
                  #'operator',
                  #'owner',
                  #'point_x',
                  #'point_y',
                  #'psgc',
                  #'shapes',
                  'type',
                  #'yrbuilt',
                  'actual_span',
                  'collapse_capacity',
                  # 'cond_pc',
                  'cond_pc_adj',
                  'cond_pc_adj_sim_idx',
                  'cond_pc_adj_sim_prob',
                  'coord',
                  'coord_lat_lon',
                  'design_level',
                  'design_span',
                  'design_speed',
                  'ratio_z_to_10',
                  'file_wind_base_name',
                  'no_sims',
                  'damage_states',
                  'rnd_state',
                  'scale',
                  'frag_arg',
                  'frag_func',
                  'frag_scale',
                  'id_adj',
                  'idl',
                  'max_no_adj_towers',
                  'height_z',
                  'point',
                  'terrain_cat',
                  'path_event']

    def __init__(self, idn=None, logger=None, **kwargs):
        """
        :param cfg: instance of config clas
        :param ps_tower: panda series containing tower details
        """

        self.idn = idn  # id within network
        self.logger = logger or logging.getLogger(__name__)

        self.no_sims = None  # o
        self.rnd_state = None  # o

        # self.DevAngle = None
        # self.Function = None  # Suspension, Terminal, Strainer
        self.height = None
        self.latitude = None
        self.lineroute = None
        self.longitude = None
        self.name = None
        self.type = None
        self.function = None
        self.actual_span = None
        # self.cond_pc = None
        self.cond_pc_adj_sim_idx = None
        self.cond_pc_adj_sim_prob = None
        self.coord = None
        self.coord_lat_lon = None
        self.design_level = None  # design level
        self.design_span = None  # design wind span
        self.design_speed = None
        self.damage_states = None

        self.axisaz = None  # azimuth of strong axis relative to North (deg)
        self.ratio_z_to_10 = None
        self.cond_pc_adj = None  # dict
        self.collapse_capacity = None
        self.file_wind_base_name = None
        self.frag_arg = None
        self.frag_func = None
        self.frag_scale = None
        self.path_event = None
        self.idl = None  # local id (starting from 0 for each line)
        self.id_adj = None
        self.max_no_adj_towers = None

        self.height_z = None
        self.point = None
        self.terrain_cat = None  # Terrain Category
        self.scale = None

        for key, value in kwargs.items():
            if key in self.registered:
                setattr(self, key, value)

        # self._damage_states = None
        # print('{}'.format(self.id_adj))
        # computed by functions in transmission_line class
        # self.id_sides = None  # (left, right) ~ assign_id_both_sides
        # self.id_adj = None  # (23,24,0,25,26) ~ update_id_adj_towers

        # initialised and later populated
        self._file_wind = None
        self._wind = None
        self._no_time = None

        # analytical method
        self._dmg = None
        self._collapse_adj = None

        # simulation method: determine_damage_isolation_sim,
        self._dmg_state_sim = None
        self._dmg_sim = None
        self._dmg_id_sim = None
        self._collapse_adj_sim = None
        # self.damage_adjacent_sim = dict.fromkeys(['id_adj', 'id_time', 'id_sim'])
        # self.damage_adjacent_sim = pd.DataFrame(None, columns=['id_adj',
        #                                                       'id_time',
        #                                                       'id_sim'],
        #                                        dtype=np.int64)
        # self.damage_interaction_sim = pd.DataFrame(None, columns=['no_collapse',
        #                                                          'id_time',
        #                                                          'id_sim'],
        #                                           dtype=np.int64)

        # line interaction
        # self._id_on_target_line = dict()
        # self.cond_pc_line_sim = {key: None for key in ['no_collapse',
        #                                               'cum_prob']}
        # self._sim_parallel_line = None

    # @property
    # def cond_pc_line(self):
    #     return self._cond_pc_line

    # @property
    # def damage_states(self):
    #     if self._damage_states is None:
    #         self._damage_states = self.frag_arg.keys()
    #     return self._damage_states

    def __repr__(self):
        return 'Tower(name={}, function={}, idl={}, idn={})'.format(
            self.name, self.function, self.idl, self.idn)

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)


    @property
    def no_time(self):
        if self._no_time is None:
            self._no_time = len(self.wind.index)
        return self._no_time

    @property
    def file_wind(self):
        if self._file_wind is None:
            try:
                self._file_wind = os.path.join(
                    self.path_event, self.file_wind_base_name)
            except AttributeError:
                self.logger.error('Invalid path_event {}'.format(self.path_event))
            else:
                try:
                    assert os.path.exists(self._file_wind)
                except AssertionError:
                    self.logger.error(
                        'Invalid file_wind {}'.format(self._file_wind))

        return self._file_wind

    # @property
    # def id_on_target_line(self):
    #     return self._id_on_target_line
    #
    # @property
    # def mc_parallel_line(self):
    #     return self._sim_parallel_line
    #
    # @id_on_target_line.setter
    # def id_on_target_line(self, id_on_target_line):
    #     self._id_on_target_line = id_on_target_line
    #     self.get_cond_prob_line_interaction()

    @property
    def wind(self):
        """
        set the wind given a file_wind
        """
        if self._wind is None:

            try:
                self._wind = pd.read_csv(self.file_wind,
                                         parse_dates=[0],
                                         index_col=['Time'],
                                         usecols=['Time', 'Speed', 'Bearing'])
            except IOError:
                msg = 'Invalid file_wind {}'.format(self.file_wind)
                self.logger.critical(msg)
            else:
                self._wind['Speed'] *= self.scale
                self._wind['directional_speed'] = self._wind.apply(
                    self.compute_directional_wind_speed, axis=1)
                self._wind['ratio'] = self._wind['directional_speed'] / self.collapse_capacity

        return self._wind

    @property
    def dmg(self):
        """
        compute probability of damage of tower in isolation (Pc)
        """
        if self._dmg is None:
            self._dmg = pd.DataFrame(None, columns=self.damage_states)
            for ds, arg in self.frag_arg.items():
                value = getattr(stats, self.frag_func).cdf(
                    self.wind['ratio'], arg, scale=self.frag_scale[ds])
                self._dmg[ds] = pd.Series(value, index=self.wind.index, name=ds)
        return self._dmg

    @property
    def collapse_adj(self):
        """
        used only for analytical approach
        calculate collapse probability of jth tower due to pull by the tower
        Pc(j,i) = P(j|i)*Pc(i)
        """
        # only applicable for tower collapse
        if self._collapse_adj is None:

            self._collapse_adj = {}

            for key, value in self.cond_pc_adj.items():

                self._collapse_adj[key] = self.dmg['collapse'].values * value

        return self._collapse_adj

    @property
    def dmg_state_sim(self):
        """
        determine if adjacent tower collapses or not due to pull by the tower
        j_time: time index (array)
        idx: multiprocessing thread id

        """

        if self._dmg_state_sim is None:

            # 1. determine damage state of tower due to wind
            rv = self.rnd_state.uniform(size=(self.no_sims, self.no_time))

            # ds_wind.shape == (no_sims, no_time)
            # PD not PE = 0(non), 1, 2 (collapse)
            self._dmg_state_sim = np.array(
                [rv[:, :, np.newaxis] < self.dmg.values])[0].sum(axis=2)
        return self._dmg_state_sim

    @property
    def dmg_sim(self):

        if self._dmg_sim is None:

            self._dmg_sim = {}

            for ids, ds in enumerate(self.damage_states, 1):

                self._dmg_sim[ds] = (self.dmg_state_sim >= ids).sum(axis=0) / self.no_sims

                # check whether MC simulation is close to analytical
                idx_not_close, = np.where(~np.isclose(self._dmg_sim[ds],
                                                      self.dmg[ds],
                                                      atol=ATOL,
                                                      rtol=RTOL))

                for idx in idx_not_close:
                    msg = 'PE of {}: simulation {:.3f} vs. analytical {:.3f}'
                    self.logger.warning(
                        msg.format(ds, self._dmg_sim[idx], self.dmg[ds].iloc[idx]))

        return self._dmg_sim

    @property
    def dmg_id_sim(self):
        """
        determine if adjacent tower collapses or not due to pull by the tower
        j_time: time index (array)
        idx: multiprocessing thread id

        """

        if self._dmg_id_sim is None:

            self._dmg_id_sim = {}

            for ids, ds in enumerate(self.damage_states, 1):

                id_sim, id_time = np.where(self.dmg_state_sim == ids)

                self._dmg_id_sim[ds] = pd.DataFrame(
                    np.vstack((id_sim, id_time)).T, columns=['id_sim', 'id_time'])

        return self._dmg_id_sim

    @property
    def collapse_adj_sim(self):
        """

        :param seed: seed is None if no seed number is provided
        :return:
        """

        if self._collapse_adj_sim is None and self.cond_pc_adj_sim_idx:

            self._collapse_adj_sim = self.dmg_id_sim['collapse'].copy()

            # generate regardless of time index
            rv = self.rnd_state.uniform(size=len(self._collapse_adj_sim['id_sim']))

            self._collapse_adj_sim['id_adj'] = (
                rv[:, np.newaxis] >= self.cond_pc_adj_sim_prob).sum(axis=1)

            # remove case with no adjacent tower collapse
            self._collapse_adj_sim.loc[
                self._collapse_adj_sim['id_adj'] == len(self.cond_pc_adj_sim_prob),
                'id_adj'] = None
            self._collapse_adj_sim = self._collapse_adj_sim.loc[
                self._collapse_adj_sim['id_adj'].notnull()]

            # replace index with tower id
            self._collapse_adj_sim['id_adj'] = self._collapse_adj_sim['id_adj'].apply(
                lambda x: self.cond_pc_adj_sim_idx[int(x)])

        return self._collapse_adj_sim

    # def determine_damage_by_interaction_at_tower_level(self, seed=None):
    #     """
    #     determine damage to tower in target line
    #     :param seed: seed is None if no seed number is provided
    #     :return:
    #     """
    #
    #     if self.cond_pc_line_sim['cum_prob']:
    #
    #         rnd_state = np.random.RandomState(seed + 50)  # replication
    #
    #         # generate regardless of time index
    #         no_sim_collapse = len(
    #             self.damage_isolation_sim['collapse']['id_time'])
    #         rv = rnd_state.uniform(size=no_sim_collapse)
    #
    #         list_idx_cond = map(lambda rv_: sum(
    #             rv_ >= self.cond_pc_line_sim['cum_prob']), rv)
    #         idx_no_adjacent_collapse = len(self.cond_pc_line_sim['cum_prob'])
    #
    #         # list of idx of adjacent towers in collapse
    #         ps_ = pd.Series([self.cond_pc_line_sim['no_collapse'][x]
    #                          if x < idx_no_adjacent_collapse
    #                          else None for x in list_idx_cond],
    #                         index=self.damage_isolation_sim['collapse'].index,
    #                         name='no_collapse')
    #
    #         adj_sim = pd.concat([self.damage_isolation_sim['collapse'], ps_],
    #                            axis=1)
    #
    #         # remove case with no adjacent tower collapse
    #         adj_sim_null_removed = adj_sim.loc[pd.notnull(ps_)].copy()
    #         if len(adj_sim_null_removed.index):
    #             adj_sim_null_removed['no_collapse'] = \
    #                 adj_sim_null_removed['no_collapse'].astype(np.int64)
    #             self.damage_interaction_sim = adj_sim_null_removed

    def compute_directional_wind_speed(self, row):

        return self.ratio_z_to_10 * row.Speed

    def compute_directional_wind_speed_needs_to_be_fixed(self, row):
        """

        :param row: pandas Series of wind
        :return:

            ------------
           |            |
           |            |
           |            |
           |            |
            ------------

        FIXME!!!
        """

        # angle of conductor relative to NS
        t0 = np.deg2rad(self.axisaz) - np.pi / 2.0

        # angle between wind direction and tower conductor
        phi = np.abs(np.deg2rad(row.Bearing) - t0)

        # angle within normal direction
        tf = (phi <= np.pi / 4) | (phi > np.pi / 4 * 7) | \
             ((phi > np.pi / 4 * 3) & (phi <= np.pi / 4 * 5))

        _cos = abs(np.cos(np.pi / 4.0 - phi))
        _sin = abs(np.sin(np.pi / 4.0 - phi))

        adjustment = row.Speed * np.max([_cos, _sin])

        return self.ratio_z_to_10 * np.where(tf, adjustment, row.Speed)
