#!/usr/bin/env python
from __future__ import print_function, division

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
                  # 'lineroute',
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
                  # 'actual_span',
                  'collapse_capacity',
                  # 'cond_pc',
                  'cond_pc_adj',
                  'cond_pc_adj_mc_idx',
                  'cond_pc_adj_mc_prob',
                  'coord',
                  'coord_lat_lon',
                  'design_level',
                  'design_span',
                  'design_speed',
                  'ratio_z_to_10',
                  'file_wind_base_name',
                  'no_sims',
                  'damage_states',
                  'event_name',
                  'rnd_state',
                  'scale',
                  'path_event',
                  'frag_arg',
                  'frag_func',
                  'frag_scale',
                  'id_adj',
                  'lid',
                  'max_no_adj_towers',
                  'height_z',
                  'point',
                  'terrain_cat',
                  'path_event']

    def __init__(self, nid=None, logger=None, **kwargs):
        """
        :param cfg: instance of config clas
        :param ps_tower: panda series containing tower details
        """

        self.nid = nid  # id within network
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
        self.cond_pc_adj_mc_idx = None
        self.cond_pc_adj_mc_prob = None
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
        self.lid = None  # local id (starting from 0 for each line)
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
        self._damage_prob = None
        self._collapse_prob_adj = None

        # simulation method: determine_damage_isolation_mc,
        self._damage_prob_mc = None
        self._collapse_prob_adj_mc = None
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

    # @property
    # def damage_states(self):
    #     if self._damage_states is None:
    #         self._damage_states = self.frag_arg.keys()
    #     return self._damage_states

    def __repr__(self):
        return 'Tower(name={}, function={}, lid={}, nid={})'.format(
            self.name, self.function, self.lid, self.nid)

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
    def damage_prob(self):
        """
        compute probability of damage of tower in isolation
        """
        if self._damage_prob is None:

            self._damage_prob = pd.DataFrame(None, columns=self.damage_states)

            for ds, arg in self.frag_arg.items():

                value = getattr(stats, self.frag_func).cdf(
                    self.wind['ratio'], arg, scale=self.frag_scale[ds])

                self._damage_prob[ds] = pd.Series(value,
                                                  index=self.wind.index,
                                                  name=ds)
        return self._damage_prob

    @property
    def collapse_prob_adj(self):
        """
        only for analytical approach
        calculate collapse probability of jth tower due to pull by the tower
        Pc(j,i) = P(j|i)*Pc(i)
        """
        # only applicable for tower collapse
        if self._collapse_prob_adj is None:

            self._collapse_prob_adj = {}

            for key, value in self.cond_pc_adj.items():

                self._collapse_prob_adj[key] = \
                    self.damage_prob['collapse'].values * value

        return self._collapse_prob_adj

    @property
    def damage_prob_mc(self):
        """
        2. determine if adjacent tower collapses or not due to pull by the tower
        j_time: time index (array)
        idx: multiprocessing thread id

        :param rv: random variable
        :return: update self.damage_isolation_mc and self.damage_adjacent_mc
        """

        if self._damage_prob_mc is None:

            self._damage_prob_mc = {}

            # 1. determine damage state of tower due to wind
            rv = self.rnd_state.uniform(size=(self.no_sims, self.no_time))

            # val = np.array([rv < self.damage_prob[ds].values
            #                 for ds in self.damage_states])  # (nds, no_sims, no_time)
            # ds_wind = np.sum(val, axis=0)

            # ds_wind.shape == (no_sims, no_time)
            # PD not PE = 0(non), 1, 2 (collapse)
            ds_wind = np.array([rv[:, :, np.newaxis] <
                                self.damage_prob.values])[0].sum(axis=2)

            # mc_wind = {ds: {'id_sim': None, 'id_time': None}
            #            for ds in self.cfg.damage_states}
            # for ids, ds in enumerate(self.cfg.damage_states, 1):
            #     mc_wind[ds]['id_sim'], mc_wind[ds]['id_time'] = \
            #         np.where(ds_wind == ids)

            for ids, ds in enumerate(self.damage_states, 1):

                id_sim, id_time = np.where(ds_wind == ids)

                self._damage_prob_mc[ds] = pd.DataFrame(
                    np.vstack((id_sim, id_time)).T,
                    columns=['id_sim', 'id_time'])

                # check whether MC simulation is close to analytical
                # prob_damage_mc = (ds_wind >= ids).sum(axis=0) / self.no_sims

                # idx_not_close, = np.where(~np.isclose(prob_damage_mc,
                #                                       self.damage_prob[ds],
                #                                       atol=ATOL,
                #                                       rtol=RTOL))
                #
                # for idx in idx_not_close:
                #     msg = 'PE of {}: simulation {:.3f} vs. analytical {:.3f}'
                #     self.logger.debug(msg.format(ds, prob_damage_mc[idx],
                #                                    self.damage_prob[ds].iloc[idx]))

        return self._damage_prob_mc

    @property
    def collapse_prob_adj_mc(self):
        """

        :param seed: seed is None if no seed number is provided
        :return:
        """

        if self._collapse_prob_adj_mc is None and self.cond_pc_adj_mc_idx:

            # msg = 'P({}|{}) at {}: simulation {:.3f} vs. analytical {:.3f}'

            df = self.damage_prob_mc['collapse'].copy()

            # generate regardless of time index
            rv = self.rnd_state.uniform(size=len(df['id_sim']))

            # list_idx_cond = map(lambda x: sum(x >= self.cond_pc_adj_mc_prob), rv)
            df['id_adj'] = (rv[:, np.newaxis] >=
                            self.cond_pc_adj_mc_prob).sum(axis=1)

            # remove case with no adjacent tower collapse
            df.loc[df['id_adj'] == len(self.cond_pc_adj_mc_prob), 'id_adj'] = None
            df = df.loc[df['id_adj'].notnull()]

            # replace index with tower id
            df['id_adj'] = df['id_adj'].apply(lambda x:
                                              self.cond_pc_adj_mc_idx[int(x)])

            self._collapse_prob_adj_mc = df

            # check simulation results
            # id_adj_removed = [x for x in self.id_adj if x >= 0]
            # id_adj_removed.remove(self.lid)
            #
            # for id_time, grouped in df.groupby('id_time'):
            #
            #     for lid in id_adj_removed:
            #
            #         prob = grouped['id_adj'].apply(lambda x: lid in x).sum() / self.no_sims
            #
            #         if not np.isclose(prob, self.collapse_prob_adj[lid][id_time],
            #                           atol=ATOL, rtol=RTOL):
            #
            #             self.logger.debug(msg.format(lid, self.name, id_time,
            #                                          prob,
            #                                          self.collapse_prob_adj[lid][id_time]))

        return self._collapse_prob_adj_mc

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
