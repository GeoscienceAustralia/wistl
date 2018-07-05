#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import pandas as pd
import os
import logging
import scipy.stats as stats


class Tower(object):
    """
    class Tower
    Tower class represent an individual tower.
    """
    registered = ['AxisAz',
                  #'Barangay',
                  #'Comment',
                  #'ConstCost',
                  #'ConstType',
                  # 'DevAngle',
                  'Function',
                  #'HT_Source',
                  # 'Height',
                  # 'Latitude',
                  # 'LineRoute',
                  #'LocSource',
                  'Longitude',
                  #'Mun',
                  #'NUMBER',
                  'Name',
                  #'Operator',
                  #'Owner',
                  #'POINT_X',
                  #'POINT_Y',
                  #'PSGC',
                  #'Shapes',
                  'Type',
                  #'YrBuilt',
                  # 'actual_span',
                  'collapse_capacity',
                  'cond_pc',
                  'coord',
                  'coord_lat_lon',
                  'design_level',
                  'design_span',
                  'design_speed',
                  'factor_10_to_z',
                  'file_wind_base_name',

                  'cond_pc_adj',
                  'frag_arg',
                  'frag_func',
                  'frag_scale',
                  'id_adj',
                  'id_line',
                  'max_no_adj_towers',

                  'height_z',
                  'point',
                  'terrain_cat',
                  'path_event']

    def __init__(self, tower_id=None, logger=None, **kwargs):
        """
        :param cfg: instance of config clas
        :param ps_tower: panda series containing tower details
        """

        self.id = tower_id  # retrieved from shapefile
        self.logger = logger or logging.getLogger(__name__)

        self.no_sims = None
        self.rnd_state = None

        # self.DevAngle = None
        # self.Function = None  # Suspension, Terminal, Strainer
        self.Height = None
        self.Latitude = None
        self.LineRoute = None
        self.Longitude = None
        self.Name = None
        self.Type = None
        self.actual_span = None
        self.cond_pc = None
        self.cond_pc_adj_mc_rel_idx = None
        self.cond_pc_adj_mc_cum_prob = None
        self.coord = None
        self.coord_lat_lon = None
        self.design_level = None  # design level
        self.design_span = None  # design wind span
        self.design_speed = None

        self.AxisAz = None  # azimuth of strong axis relative to North (deg)
        self.factor_10_to_z = None
        self.cond_pc_adj = None  # dict
        self.collapse_capacity = None
        self.file_wind_base_name = None
        self.frag_arg = None
        self.frag_func = None
        self.frag_scale = None
        self.path_event = None
        self.id_line = None  # local id (starting from 0 for each line)
        self.id_adj = None
        self.max_no_adj_towers = None

        self.height_z = None
        self.point = None
        self.terrain_cat = None  # Terrain Category
        self.scale = None

        for key, value in kwargs.items():
            if key in self.registered:
                setattr(self, key, value)

        self._damage_states = None
        # print('{}'.format(self.id_adj))
        # computed by functions in transmission_line class
        # self.id_sides = None  # (left, right) ~ assign_id_both_sides
        # self.id_adj = None  # (23,24,0,25,26) ~ update_id_adj_towers

        # computed by calculate_cond_pc_adj

        # initialised and later populated
        self._file_wind = None
        self._wind = None

        # analytical method
        self._damage_prob_isolation = None
        self._damage_prob_adjacent = None

        # simulation method: determine_damage_isolation_mc,
        # self.damage_isolation_mc = None
        # self.damage_adjacent_mc = dict.fromkeys(['id_adj', 'id_time', 'id_sim'])
        # self.damage_adjacent_mc = pd.DataFrame(None, columns=['id_adj',
        #                                                       'id_time',
        #                                                       'id_sim'],
        #                                        dtype=np.int64)
        # self.damage_interaction_mc = pd.DataFrame(None, columns=['no_collapse',
        #                                                          'id_time',
        #                                                          'id_sim'],
        #                                           dtype=np.int64)

        # line interaction
        # self._id_on_target_line = dict()
        # self.cond_pc_line_mc = {key: None for key in ['no_collapse',
        #                                               'cum_prob']}
        # self._mc_parallel_line = None

    # @property
    # def cond_pc_line(self):
    #     return self._cond_pc_line

    @property
    def damage_states(self):
        if self._damage_states is None:
            self._damage_states = self.frag_arg.keys()
        return self._damage_states

    @property
    def file_wind(self):
        if self._file_wind is None:
            self._file_wind = os.path.join(self.path_event, self.file_wind_base_name)
        return self._file_wind

    # @property
    # def id_on_target_line(self):
    #     return self._id_on_target_line
    #
    # @property
    # def mc_parallel_line(self):
    #     return self._mc_parallel_line
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
                                         index_col=[0],
                                         usecols=[0, 3, 6])
            except IOError:
                msg = 'file {} does not exist'.format(self.file_wind)
                self.logger.critical(msg)
            else:
                self._wind['directional_speed'] = self._wind.apply(
                    self.compute_directional_wind_speed, axis=1)
                self._wind['ratio'] = self._wind['directional_speed'] / self.collapse_capacity

        return self._wind

    @property
    def damage_prob_isolation(self):
        """
        compute probability of damage of tower in isolation
        """
        if self._damage_prob_isolation is None:
            self._damage_prob_isolation = pd.DataFrame(None, columns=self.damage_states)
            for ds, arg in self.frag_arg.items():
                value = getattr(stats, self.frag_func).cdf(
                    self.wind.ratio,
                    arg, scale=self.frag_scale[ds])
                self._damage_prob_isolation[ds] = pd.Series(value,
                                                            index=self.wind.index,
                                                            name=ds)
        return self._damage_prob_isolation

    @property
    def damage_prob_adjacent(self):
        """
        only for analytical approach
        calculate collapse probability of jth tower due to pull by the tower
        Pc(j,i) = P(j|i)*Pc(i)
        """
        # only applicable for tower collapse
        if self._damage_prob_adjacent is None:
            self._damage_prob_adjacent = {}
            for rel_idx in self.cond_pc_adj.keys():
                abs_idx = self.id_adj[rel_idx + self.max_no_adj_towers]
                self._damage_prob_adjacent[abs_idx] = \
                    self.damage_prob_isolation.collapse.values * \
                    self.cond_pc_adj[rel_idx]

        return self._damage_prob_adjacent

    def compute_directional_wind_speed(self, row):
        """

        :param row: pandas Series of wind
        :return:
        """

        # angle of conductor relative to NS
        t0 = np.deg2rad(self.AxisAz) - np.pi / 2.0

        # angle between wind direction and tower conductor
        phi = np.abs(np.deg2rad(row.Bearing) - t0)
        tf = (phi <= np.pi / 4) | (phi > np.pi / 4 * 7) | \
             ((phi > np.pi / 4 * 3) & (phi <= np.pi / 4 * 5))

        _cos = abs(np.cos(np.pi / 4.0 - phi))
        _sin = abs(np.sin(np.pi / 4.0 - phi))

        adjustment = row.Speed * np.max(_cos, _sin)

        return self.factor_10_to_z * np.where(tf, adjustment, row.Speed)


    # def determine_damage_isolation_mc(self):
    #     """
    #     2. determine if adjacent tower collapses or not due to pull by the tower
    #     j_time: time index (array)
    #     idx: multiprocessing thread id
    #
    #     :param rv: random variable
    #     :return: update self.damage_isolation_mc and self.damage_adjacent_mc
    #     """
    #
    #     rv = self.rnd_state.uniform(size=(self.no_sims, len(self.wind.index)))
    #     # 1. determine damage state of tower due to wind
    #
    #     val = np.array([rv < self.damage_prob_isolation[ds].values
    #                     for ds in
    #                     self.cfg.damage_states])  # (nds, no_sims, no_time)
    #
    #     # ds_wind.shape == (no_sims, no_time)
    #     # PD not PE = 0(non), 1, 2 (collapse)
    #     ds_wind = np.sum(val, axis=0)
    #
    #     # mc_wind = {ds: {'id_sim': None, 'id_time': None}
    #     #            for ds in self.cfg.damage_states}
    #     # for ids, ds in enumerate(self.cfg.damage_states, 1):
    #     #     mc_wind[ds]['id_sim'], mc_wind[ds]['id_time'] = \
    #     #         np.where(ds_wind == ids)
    #
    #     mc_wind = dict()
    #     for ids, ds in enumerate(self.cfg.damage_states, 1):
    #         id_sim, id_time = np.where(ds_wind == ids)
    #
    #         mc_wind[ds] = pd.DataFrame(np.vstack((id_sim, id_time)).T,
    #                                    columns=['id_sim', 'id_time'])
    #
    #         # check whether MC simulation is close to analytical
    #         prob_damage_isolation_mc = pd.Series(
    #             np.sum(ds_wind[:, ] >= ids, axis=0) / float(rv.shape[0]),
    #             index=self.prob_damage_isolation.index)
    #
    #         idx_not_close, = np.where(~np.isclose(prob_damage_isolation_mc,
    #                                               self.prob_damage_isolation[
    #                                                   ds],
    #                                               atol=self.cfg.atol,
    #                                               rtol=self.cfg.rtol))
    #
    #         for idx in idx_not_close:
    #             print('prob. {} from simulation {:.3f}, analytical: {:.3f}'.format(ds, prob_damage_isolation_mc.ix[idx],
    #                        self.prob_damage_isolation[ds].ix[idx]))
    #
    #     self.damage_isolation_mc = mc_wind
    #
    # def determine_damage_adjacent_mc(self, seed=None):
    #     """
    #
    #     :param seed: seed is None if no seed number is provided
    #     :return:
    #     """
    #
    #     if self.cond_pc_adj_mc['rel_idx']:
    #
    #         if self.cfg.random_seed:
    #             rnd_state = np.random.RandomState(seed + 100)  # replication
    #         else:
    #             rnd_state = np.random.RandomState()
    #
    #         # generate regardless of time index
    #         no_sim_collapse = len(
    #             self.damage_isolation_mc['collapse']['id_time'])
    #         rv = rnd_state.uniform(size=no_sim_collapse)
    #
    #         list_idx_cond = map(lambda rv_: sum(
    #             rv_ >= self.cond_pc_adj_mc['cum_prob']), rv)
    #         idx_no_adjacent_collapse = len(self.cond_pc_adj_mc['cum_prob'])
    #
    #         # list of idx of adjacent towers in collapse
    #         abs_idx_list = []
    #         for rel_idx in self.cond_pc_adj_mc['rel_idx']:
    #             abs_idx = tuple((self.id_adj[j + self.max_no_adj_towers]
    #                              for j in rel_idx if j != 0))
    #             abs_idx_list.append(abs_idx)
    #
    #         ps_ = pd.Series([abs_idx_list[x] if x < idx_no_adjacent_collapse
    #                          else None for x in list_idx_cond],
    #                         index=self.damage_isolation_mc['collapse'].index,
    #                         name='id_adj')
    #
    #         adj_mc = pd.concat([self.damage_isolation_mc['collapse'], ps_],
    #                            axis=1)
    #
    #         # remove case with no adjacent tower collapse
    #         self.damage_adjacent_mc = adj_mc.loc[pd.notnull(ps_)]

    # def determine_damage_by_interaction_at_tower_level(self, seed=None):
    #     """
    #     determine damage to tower in target line
    #     :param seed: seed is None if no seed number is provided
    #     :return:
    #     """
    #
    #     if self.cond_pc_line_mc['cum_prob']:
    #
    #         rnd_state = np.random.RandomState(seed + 50)  # replication
    #
    #         # generate regardless of time index
    #         no_sim_collapse = len(
    #             self.damage_isolation_mc['collapse']['id_time'])
    #         rv = rnd_state.uniform(size=no_sim_collapse)
    #
    #         list_idx_cond = map(lambda rv_: sum(
    #             rv_ >= self.cond_pc_line_mc['cum_prob']), rv)
    #         idx_no_adjacent_collapse = len(self.cond_pc_line_mc['cum_prob'])
    #
    #         # list of idx of adjacent towers in collapse
    #         ps_ = pd.Series([self.cond_pc_line_mc['no_collapse'][x]
    #                          if x < idx_no_adjacent_collapse
    #                          else None for x in list_idx_cond],
    #                         index=self.damage_isolation_mc['collapse'].index,
    #                         name='no_collapse')
    #
    #         adj_mc = pd.concat([self.damage_isolation_mc['collapse'], ps_],
    #                            axis=1)
    #
    #         # remove case with no adjacent tower collapse
    #         adj_mc_null_removed = adj_mc.loc[pd.notnull(ps_)].copy()
    #         if len(adj_mc_null_removed.index):
    #             adj_mc_null_removed['no_collapse'] = \
    #                 adj_mc_null_removed['no_collapse'].astype(np.int64)
    #             self.damage_interaction_mc = adj_mc_null_removed



