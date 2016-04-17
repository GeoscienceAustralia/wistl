#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import numpy as np
import pandas as pd
import scipy.stats as stats


class Tower(object):
    """
    class Tower
    Tower class represent an individual wistl tower.
    """

    def __init__(self, conf, ps_tower):
        """
        :param conf: instance of config class
        :param ps_tower: panda series containing tower details
        """

        self.conf = conf
        self.ps_tower = ps_tower
        self.id = ps_tower.name  # Series.name

        self.coord = ps_tower.coord  # (1, 2)
        self.point = ps_tower.point  # (2,)
        self.coord_lat_lon = ps_tower.coord_lat_lon  # (1, 2)

        self.name = ps_tower.Name
        self.type = ps_tower.Type  # Lattice Tower or Steel Pole
        self.function = ps_tower.Function  # Suspension, Terminal, Strainer
        self.line_route = ps_tower.LineRoute
        self.no_circuit = 2  # double circuit (default value)

        self.design_span = ps_tower.design_span  # design wind span
        self.terrain_cat = ps_tower.terrain_cat  # Terrain Category
        self.design_level = ps_tower.design_level  # design level

        # azimuth of strong axis relative to North (deg)
        self.strong_axis = ps_tower.AxisAz
        self.dev_angle = ps_tower.DevAngle  # deviation angle
        self.height = ps_tower.Height
        self.height_z = conf.drag_height[self.function]
        self.actual_span = ps_tower.actual_span
        self.file_wind_base_name = self.ps_tower.file_wind_base_name

        self.design_speed = ps_tower.design_speed
        if self.conf.adjust_design_by_topography:
            self.adjust_design_speed()

        self.collapse_capacity = self.compute_collapse_capacity()
        self.convert_factor = self.convert_10_to_z()

        self.cond_pc, self.max_no_adj_towers = self.get_cond_collapse_prob()

        # computed by functions in transmission_line class
        self.id_sides = None  # (left, right) ~ assign_id_both_sides
        self.id_adj = None  # (23,24,0,25,26) ~ update_id_adj_towers

        # computed by calculate_cond_pc_adj
        self.cond_pc_adj = None  # dict
        self.cond_pc_adj_mc = {'rel_idx': [None], 'cum_prob': [None]}

        # initialised and later populated
        self._event_tuple = None
        self.file_wind = None
        self.scale = None
        self.wind = None
        self.time_index = None

        # analytical method
        self.prob_damage_isolation = None  # compute_damage_prob_isolation
        self.prob_damage_adjacent = None  # dict (no_time,) <- compute_pc_adj

        # simulation method
        self.damage_isolation_mc = dict()  # (no_sims, no_time)
        self.damage_adjacent_mc = dict()  #

        # line interaction
        self._id_on_target_line = dict()
        self._cond_parallel_line = None
        self._mc_parallel_line = None

    @property
    def event_tuple(self):
        return self._event_tuple

    @property
    def id_on_target_line(self):
        return self._id_on_target_line

    @property
    def cond_parallel_line(self):
        return self._cond_parallel_line

    @property
    def mc_parallel_line(self):
        return self._mc_parallel_line

    @event_tuple.setter
    def event_tuple(self, value):
        try:
            file_wind_, scale_ = value
        except ValueError:
            raise ValueError("Pass an iterable with two items")
        else:
            self.file_wind = file_wind_
            self.scale = scale_
            self._set_wind()
            self.compute_damage_prob_isolation()

    @id_on_target_line.setter
    def id_on_target_line(self, id_on_target_line):
        self._id_on_target_line = id_on_target_line
        self._get_cond_prob_line_interaction()

    def _set_wind(self):
        """
        set the wind, time_index variables given a file_wind
        """
        try:
            data = self.scale * pd.read_csv(self.file_wind,
                                            header=0,
                                            parse_dates=[0],
                                            index_col=[0],
                                            usecols=[0, 3, 6],
                                            names=['', '', '', 'speed', '', '',
                                                   'bearing', ''])
        except IOError:
            msg = 'file {} does not exist'.format(self.file_wind)
            raise IOError(msg)

        speed = data['speed'].values
        bearing = np.deg2rad(data['bearing'].values)

        # angle of conductor relative to NS
        t0 = np.deg2rad(self.strong_axis) - np.pi / 2.0

        data['dir_speed'] = pd.Series(
            self.convert_factor * self.compute_directional_wind_speed(
                speed, bearing, t0), index=data.index)
        data['ratio'] = data['dir_speed']/self.collapse_capacity

        self.wind = data
        self.time_index = data.index

    @staticmethod
    def compute_directional_wind_speed(speed, bearing, t0):

        # angle between wind direction and tower conductor
        phi = np.abs(bearing - t0)
        tf = (phi <= np.pi / 4) | (phi > np.pi / 4 * 7) | \
             ((phi > np.pi / 4 * 3) & (phi <= np.pi / 4 * 5))

        cos_ = abs(np.cos(np.pi / 4.0 - phi))
        sin_ = abs(np.sin(np.pi / 4.0 - phi))

        adj = speed * np.max(np.vstack((cos_, sin_)), axis=0)
        return np.where(tf, adj, speed)  # adj if true, otherwise speed

    def compute_damage_prob_isolation(self):
        """
        compute probability of damage of tower in isolation
        """
        if self.file_wind:
            prob_damage = dict()
            for ds in self.conf.damage_states:
                value = getattr(stats, self.ps_tower.frag_func).cdf(
                    self.wind.ratio,
                    self.ps_tower.frag_arg[ds],
                    scale=self.ps_tower.frag_scale[ds])
                prob_damage[ds] = pd.Series(value,
                                            index=self.wind.index,
                                            name=ds)
            self.prob_damage_isolation = pd.DataFrame.from_dict(prob_damage)

            # self.prob_damage_isolation = pd.DataFrame(
            #     np.zeros(shape=(len(self.time_index),
            #              self.conf.no_damage_states)),
            #     columns=self.conf.damage_states,
            #     index=self.wind.index)

    def compute_damage_prob_adjacent(self):
        """
        only for analytical approach
        calculate collapse probability of jth tower due to pull by the tower
        Pc(j,i) = P(j|i)*Pc(i)
        """
        # only applicable for tower collapse

        pc_adj = {}
        for rel_idx in self.cond_pc_adj.keys():
            abs_idx = self.id_adj[rel_idx + self.max_no_adj_towers]
            pc_adj[abs_idx] = (self.prob_damage_isolation.collapse.values *
                               self.cond_pc_adj[rel_idx])

        self.prob_damage_adjacent = pc_adj

    def determine_damage_isolation_mc(self, rv):
        """
        2. determine if adjacent tower collapses or not due to pull by the tower
        j_time: time index (array)
        idx: multiprocessing thread id

        :param rv: random variable
        :param seed: seed is None if no seed number is provided
        :return: update self.damage_isolation_mc and self.damage_adjacent_mc
        """
        # 1. determine damage state of tower due to wind
        val = np.array([rv < self.prob_damage_isolation[ds].values
                        for ds in
                        self.conf.damage_states])  # (nds, no_sims, no_time)

        # ds_wind.shape == (no_sims, no_time)
        # PD not PE = 0(non), 1, 2 (collapse)
        ds_wind = np.sum(val, axis=0)

        # mc_wind = {ds: {'id_sim': None, 'id_time': None}
        #            for ds in self.conf.damage_states}
        # for ids, ds in enumerate(self.conf.damage_states, 1):
        #     mc_wind[ds]['id_sim'], mc_wind[ds]['id_time'] = \
        #         np.where(ds_wind == ids)

        mc_wind = dict()
        for ids, ds in enumerate(self.conf.damage_states, 1):
            id_sim, id_time = np.where(ds_wind == ids)

            mc_wind[ds] = pd.DataFrame(np.vstack((id_sim, id_time)).T,
                                       columns=['id_sim', 'id_time'])

            # check whether MC simulation is close to analytical
            prob_damage_isolation_mc = pd.Series(
                np.sum(ds_wind[:, ] >= ids, axis=0) / float(self.conf.no_sims),
                index=self.prob_damage_isolation.index)

            idx_not_close, = np.where(~np.isclose(prob_damage_isolation_mc,
                                      self.prob_damage_isolation[ds],
                                      atol=self.conf.atol,
                                      rtol=self.conf.rtol))

            for idx in idx_not_close:
                print('{}:{}'.format(prob_damage_isolation_mc.ix[idx],
                                     self.prob_damage_isolation[ds].ix[idx]))

        self.damage_isolation_mc = mc_wind

    def determine_damage_adjacent_mc(self, seed=None):

        if self.cond_pc_adj_mc['rel_idx']:

            if self.conf.random_seed:
                rnd_state = np.random.RandomState(seed + 100)  # replication
            else:
                rnd_state = np.random.RandomState()

            # generate regardless of time index
            no_sim_collapse = len(
                self.damage_isolation_mc['collapse']['id_time'])
            rv = rnd_state.uniform(size=no_sim_collapse)

            list_idx_cond = map(lambda rv_: sum(
                rv_ >= self.cond_pc_adj_mc['cum_prob']), rv)
            idx_no_adjacent_collapse = len(self.cond_pc_adj_mc['cum_prob'])

            # list of idx of adjacent towers in collapse
            abs_idx_list = []
            for rel_idx in self.cond_pc_adj_mc['rel_idx']:
                abs_idx = tuple((self.id_adj[j + self.max_no_adj_towers]
                                 for j in rel_idx if j != 0))
                abs_idx_list.append(abs_idx)

            ps_ = pd.Series([abs_idx_list[x] if x < idx_no_adjacent_collapse
                                else None for x in list_idx_cond],
                            index=self.damage_isolation_mc['collapse'].index,
                            name='id_adj')

            adj_mc = pd.concat([self.damage_isolation_mc['collapse'], ps_],
                               axis=1)
            adj_mc = adj_mc[pd.notnull(ps_)]

            for (id_time_, id_adj_), grouped in adj_mc.groupby(['id_time',
                                                                'id_adj']):
                self.damage_adjacent_mc.setdefault(id_time_, {})[id_adj_] = \
                    grouped.id_sim.tolist()

    # def compute_mc_adj(self, rv, seed=None):
    #     """
    #     2. determine if adjacent tower collapses or not due to pull by the tower
    #     jtime: time index (array)
    #     idx: multiprocessing thread id
    #     """
    #     if self.file_wind:
    #         # 1. determine damage state of tower due to wind
    #         val = np.array([rv < self.prob_damage_isolation[ds].values
    #                        for ds in self.conf.damage_states])  # (nds, nsims, ntime)
    #
    #         ds_wind = np.sum(val, axis=0)  # (nsims, ntime) 0(non), 1, 2 (collapse)
    #
    #         mc_wind = dict()
    #         for ids, ds in enumerate(self.conf.damage_states):
    #             mc_wind.setdefault(ds, {})['id_sim'], mc_wind.setdefault(ds, {})['id_time'] = np.where(ds_wind == (ids + 1))
    #
    #         # for collapse
    #         unq_itime = np.unique(mc_wind['collapse']['id_time'])
    #         nprob = len(self.cond_pc_adj_mc['cum_prob'])  #
    #
    #         mc_adj = dict()  # impact on adjacent towers
    #
    #         if self.conf.random_seed:
    #             prng = np.random.RandomState(seed + 100)  # replication
    #         else:
    #             prng = np.random.RandomState()
    #
    #         if nprob > 0:
    #             for jtime in unq_itime:
    #                 jdx = np.where(mc_wind['collapse']['id_time'] == jtime)[0]
    #                 idx_sim = mc_wind['collapse']['id_sim'][jdx]  # index of simulation
    #
    #                 rv = prng.uniform(size=len(idx_sim))
    #
    #                 list_idx_cond = map(lambda rv_: sum(
    #                     rv_ >= self.cond_pc_adj_mc['cum_prob']), rv)
    #
    #                 # ignore simulation where none of adjacent tower collapses
    #                 unq_list_idx_cond = set(list_idx_cond) - set([nprob])
    #
    #                 for idx_cond in unq_list_idx_cond:
    #
    #                     # list of idx of adjacent towers in collapse
    #                     rel_idx = self.cond_pc_adj_mc['rel_idx'][idx_cond]
    #
    #                     # convert relative to absolute fid
    #                     #abs_idx = [self.id_adj[j + self.max_no_adj_towers]
    #                     #           for j in rel_idx]
    #                     abs_idx = [self.id_adj[j + self.max_no_adj_towers]
    #                                for j in rel_idx if j != 0]
    #
    #                     # filter simulation
    #                     isim = [i for i, x in enumerate(list_idx_cond)
    #                             if x == idx_cond]
    #                     mc_adj.setdefault(jtime, {})[tuple(abs_idx)] = idx_sim[isim]
    #
    #         self.damage_isolation_mc['collapse'] = pd.DataFrame(mc_wind['collapse'])
    #         self.damage_isolation_mc['minor'] = pd.DataFrame(mc_wind['minor'])
    #         self.damage_adjacent_mc = mc_adj

    def get_cond_collapse_prob(self):
        """ get dict of conditional collapse probabilities
        :return: cond_pc, max_no_adj_towers
        """

        # FIXME: move to transmission_line
        att_value = self.ps_tower[self.conf.cond_collapse_prob_metadata['by']]
        df_prob = self.conf.cond_collapse_prob[att_value]

        att = self.conf.cond_collapse_prob_metadata[att_value]['by']
        att_type = self.conf.cond_collapse_prob_metadata[att_value]['type']
        tf_array = None
        if att_type == 'string':
            tf_array = df_prob[att] == getattr(self, att)
        elif att_type == 'numeric':
            tf_array = (df_prob[att + '_lower'] <= self.ps_tower[att]) & \
                       (df_prob[att + '_upper'] > self.ps_tower[att])

        # change to dictionary
        cond_pc = dict(zip(df_prob.loc[tf_array, 'list'],
                           df_prob.loc[tf_array, 'probability']))
        max_no_adj_towers = \
            self.conf.cond_collapse_prob_metadata[att_value]['max_adj']

        return cond_pc, max_no_adj_towers

    def _get_cond_prob_line_interaction(self):
        """ get dict of conditional collapse probabilities
        :return: cond_prob_line
        """

        # FIXME: move to transmission_line
        att = self.conf.prob_line_interaction_metadata['by']
        att_type = self.conf.prob_line_interaction_metadata['type']
        df_prob = self.conf.prob_line_interaction

        tf_array = None
        if att_type == 'string':
            tf_array = df_prob == getattr(self, att)
        elif att_type == 'numeric':
            tf_array = (df_prob[att + '_lower'] <= self.ps_tower[att]) & \
                       (df_prob[att + '_upper'] > self.ps_tower[att])

        # change to dictionary
        return dict(zip(df_prob.loc[tf_array, 'no_collapse'],
                        df_prob.loc[tf_array, 'probability']))

    def adjust_design_speed(self):
        """ determine design speed """
        id_topo = sum(self.conf.topo_multiplier[self.name] >=
                      self.conf.design_adjustment_factor_by_topo['threshold'])
        self.design_speed *= self.conf.design_adjustment_factor_by_topo[id_topo]

    def compute_collapse_capacity(self):
        """ calculate adjusted collapse wind speed for a tower
        Vc = Vd(h=z)/sqrt(u)
        where u = 1-k(1-Sw/Sd)
        Sw: actual wind span
        Sd: design wind span (defined by line route)
        k: 0.33 for a single, 0.5 for double circuit
        :rtype: float
        """

        # k: 0.33 for a single, 0.5 for double circuit
        k_factor = {1: 0.33, 2: 0.5}  # hard-coded

        # calculate utilization factor
        # 1 in case sw/sd > 1
        u = min(1.0, 1.0 - k_factor[self.no_circuit] *
                (1.0 - self.actual_span / self.design_span))
        return self.design_speed / np.sqrt(u)

    def convert_10_to_z(self):
        """
        Mz,cat(h=10)/Mz,cat(h=z)
        tc: terrain category
        asset is a Tower class instance.

        :return:
        """
        terrain_cat = 'tc' + str(self.terrain_cat)
        try:
            terrain_multiplier_z = np.interp(
                self.height_z,
                self.conf.terrain_multiplier['height'],
                self.conf.terrain_multiplier[terrain_cat])
        except KeyError:
            msg = '{} is undefined in {}'.format(
                terrain_cat, self.conf.file_terrain_multiplier)
            raise KeyError(msg)

        id10 = self.conf.terrain_multiplier['height'].index(10)
        terrain_multiplier_10 = self.conf.terrain_multiplier[terrain_cat][id10]
        return terrain_multiplier_z / terrain_multiplier_10

    def calculate_cond_pc_adj(self):
        """
        calculate conditional collapse probability of jth tower given ith tower
        P(j|i)
        """

        # rel_idx_strainer
        idx_m1 = np.array([i for i in range(len(self.id_adj))
                           if self.id_adj[i] == -1]) - self.max_no_adj_towers

        try:
            max_neg = np.max(idx_m1[idx_m1 < 0]) + 1
        except ValueError:
            max_neg = -1 * self.max_no_adj_towers

        try:
            min_pos = np.min(idx_m1[idx_m1 > 0])
        except ValueError:
            min_pos = self.max_no_adj_towers + 1

        bound_ = set(range(max_neg, min_pos))

        cond_prob = {}
        for item in self.cond_pc.keys():
            w = list(set(item).intersection(bound_))
            w.sort()
            w = tuple(w)
            if w in cond_prob:
                cond_prob[w] += self.cond_pc[item]
            else:
                cond_prob[w] = self.cond_pc[item]

        if (0,) in cond_prob:
            cond_prob.pop((0,))

        # sort by cond. prob
        rel_idx = sorted(cond_prob, key=cond_prob.get)
        prob = map(lambda v: cond_prob[v], rel_idx)

        cum_prob = np.cumsum(np.array(prob))

        self.cond_pc_adj_mc['rel_idx'] = rel_idx
        self.cond_pc_adj_mc['cum_prob'] = cum_prob

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

        self.cond_pc_adj = cond_pc_adj
