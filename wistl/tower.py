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

    def __init__(self, conf, df_tower):
        """
        :param conf: instance of config class
        :param df_tower: data frame containing tower details
        """

        self.conf = conf
        self.df_tower = df_tower

        self.id = df_tower['id']
        self.name = df_tower['Name']
        # self.ttype = df_tower['Type']  # Lattice Tower or Steel Pole
        self.function = df_tower['Function']  # e.g., suspension, terminal, strainer
        self.line_route = df_tower['LineRoute']  # string
        self.no_circuit = 2  # double circuit (default value)
        self.design_span = self.conf.design_value[self.line_route]['span']  # design wind span
        self.terrain_cat = self.conf.design_value[self.line_route]['cat']  # Terrain Cateogry
        self.design_level = self.conf.design_value[self.line_route]['level']  # design level
        self.strong_axis = df_tower['AxisAz']  # azimuth of strong axis relative to North (deg)
        self.dev_angle = df_tower['DevAngle']  # deviation angle
        self.height = df_tower['Height']
        self.height_z = conf.drag_height[self.function]
        self.actual_span = df_tower['actual_span']  # actual wind span on eith side

        self.design_speed = self.assign_design_speed  # design wind speed
        self.collapse_capacity = self.compute_collapse_capacity
        self.file_wind_base_name = self.get_wind_file
        self.convert_factor = self.convert_10_to_z()

        self.cond_pc, self.max_no_adj_towers = self.get_cond_collapse_prob()

        # computed by functions in transmission_line class
        self.id_sides = None  # (left, right) ~ assign_id_both_sides
        self.id_adj = None  # (23,24,0,25,26) ~ update_id_adj_towers, update_id_adj_towers

        # computed by calculate_cond_pc_adj
        self.cond_pc_adj = None  # dict
        self.cond_pc_adj_mc = {'rel_idx': [None], 'cum_prob': [None]}

        # moved from damage_tower.py
        self._file_wind = None
        self._wind = None  # self.read_wind_timeseries()
        self._time_index = None  # self.wind.index

        # analytical method
        self._pc_wind = None  # compute_pc_wind
        self._pc_adj = None  # dict (ntime,) <- compute_pc_adj

        # simulation method
        self._mc_wind = None  # dict(nsims, ntime)
        self._mc_adj = None  # dict <- compute_mc_adj

    @property
    def wind(self):
        return self._wind

    def _set_wind(self):
        """
        set the wind, time_index variables given a file_wind
        """
        try:
            data = pd.read_csv(self._file_wind, header=0, parse_dates=[0],
                               index_col=[0], usecols=[0, 3, 6],
                               names=['', '', '', 'speed', '', '',
                                      'bearing', ''])
        except IOError:
            msg = 'file {} does not exist'.format(self._file_wind)
            print(msg)
            raise IOError(msg)

        speed = data['speed'].values
        bearing = np.deg2rad(data['bearing'].values)  # degree

        # angle of conductor relative to NS
        t0 = np.deg2rad(self.strong_axis) - np.pi/2.0

        data['dir_speed'] = pd.Series(
            self.convert_factor * self.compute_directional_wind_speed(
                speed, bearing, t0), index=data.index)
        self._wind = data
        self._time_index = data.index

    @property
    def time_index(self):
        return self._time_index

    @property
    def file_wind(self):
        return self._file_wind

    @file_wind.setter
    def file_wind(self, file_wind):
        self._file_wind = file_wind
        self._set_wind()

    @staticmethod
    def compute_directional_wind_speed(speed, bearing, t0):

        # angle between wind direction and tower conductor
        phi = np.abs(bearing - t0)
        tf = (phi <= np.pi / 4) | (phi > np.pi / 4 * 7) | \
             ((phi > np.pi / 4 * 3) & (phi <= np.pi / 4 * 5))

        cos_ = abs(np.cos(np.pi / 4.0 - phi))
        sin_ = abs(np.sin(np.pi / 4.0 - phi))

        adj = speed*np.max(np.vstack((cos_, sin_)), axis=0)
        return np.where(tf, adj, speed)  # adj if true, otherwise speed

    def compute_pc_wind(self):
        """
        compute probability of damage due to wind
        """
        if self._file_wind:
            pc_wind = np.zeros(shape=(len(self.time_index),
                                      self.conf.no_damage_states))
            vratio = self.wind.dir_speed.values / self.collapse_capacity

            tf_array = np.ones((self.conf.fragility.shape[0],), dtype=bool)
            for att, att_type in zip(self.conf.fragility_metadata['by'],
                                     self.conf.fragility_metadata['type']):
                if att_type == 'string':
                    tf_array *= self.conf.fragility[att] == self.df_tower[att]
                elif att_type == 'numeric':
                    tf_array *= (self.conf.fragility[att + '_lower'] <=
                                 self.df_tower[att]) & \
                                (self.conf.fragility[att + '_upper'] >
                                 self.df_tower[att])

            for ids, limit_state in enumerate(self.conf.damage_states):

                idx = tf_array & (self.conf.fragility['limit_states'] ==
                                  limit_state)
                fn_form = self.conf.fragility.loc[
                    idx, self.conf.fragility_metadata['function']].values[0]
                arg_ = self.conf.fragility.loc[
                    idx, self.conf.fragility_metadata[fn_form]['arg']]
                scale_ = self.conf.fragility.loc[
                    idx, self.conf.fragility_metadata[fn_form]['scale']]

                temp = getattr(stats, fn_form).cdf(vratio, arg_, scale=scale_)
                pc_wind[:, ids] = temp

            self._pc_wind = pd.DataFrame(pc_wind,
                                         columns=self.conf.damage_states,
                                         index=self.wind.index)

    @property
    def pc_wind(self):
        return self._pc_wind

    def compute_pc_adj(self):  # only for analytical approach
        """
        only for analytical approach
        calculate collapse probability of jth tower due to pull by the tower
        Pc(j,i) = P(j|i)*Pc(i)
        """
        # only applicable for tower collapse

        pc_adj = {}
        for rel_idx in self.cond_pc_adj.keys():
            abs_idx = self.id_adj[rel_idx + self.max_no_adj_towers]
            pc_adj[abs_idx] = (self.pc_wind.collapse.values *
                               self.cond_pc_adj[rel_idx])

        self._pc_adj = pc_adj

    @property
    def pc_adj(self):
        return self._pc_adj

    def compute_mc_adj(self, rv, seed=None):
        """
        2. determine if adjacent tower collapses or not due to pull by the tower
        jtime: time index (array)
        idx: multiprocessing thread id
        """
        if self._file_wind:
            # 1. determine damage state of tower due to wind
            val = np.array([rv < self.pc_wind[ds].values
                           for ds in self.conf.damage_states])  # (nds, nsims, ntime)

            ds_wind = np.sum(val, axis=0)  # (nsims, ntime) 0(non), 1, 2 (collapse)

            mc_wind = dict()
            for ids, ds in enumerate(self.conf.damage_states):
                mc_wind.setdefault(ds, {})['isim'], mc_wind.setdefault(ds, {})['itime'] = np.where(ds_wind == (ids + 1))

            # for collapse
            unq_itime = np.unique(mc_wind['collapse']['itime'])
            nprob = len(self.cond_pc_adj_mc['cum_prob'])  #

            mc_adj = dict()  # impact on adjacent towers

            if self.conf.random_seed:
                prng = np.random.RandomState(seed + 100)  # replication
            else:
                prng = np.random.RandomState()

            if nprob > 0:
                for jtime in unq_itime:
                    jdx = np.where(mc_wind['collapse']['itime'] == jtime)[0]
                    idx_sim = mc_wind['collapse']['isim'][jdx]  # index of simulation

                    rv = prng.uniform(size=len(idx_sim))

                    list_idx_cond = map(lambda rv_: sum(
                        rv_ >= self.cond_pc_adj_mc['cum_prob']), rv)

                    # ignore simulation where none of adjacent tower collapses
                    unq_list_idx_cond = set(list_idx_cond) - set([nprob])

                    for idx_cond in unq_list_idx_cond:

                        # list of idx of adjacent towers in collapse
                        rel_idx = self.cond_pc_adj_mc['rel_idx'][idx_cond]

                        # convert relative to absolute fid
                        abs_idx = [self.id_adj[j + self.max_no_adj_towers]
                                   for j in rel_idx]

                        # filter simulation
                        isim = [i for i, x in enumerate(list_idx_cond)
                                if x == idx_cond]
                        mc_adj.setdefault(jtime, {})[tuple(abs_idx)] = idx_sim[isim]

            self._mc_wind = mc_wind
            self._mc_adj = mc_adj

    @property
    def mc_wind(self):
        return self._mc_wind

    @property
    def mc_adj(self):
        return self._mc_adj

    def get_cond_collapse_prob(self):
        """ get dict of conditional collapse probabilities
        :return: cond_pc, max_no_adj_towers
        """

        att_value = self.df_tower[self.conf.cond_collapse_prob_metadata['by']]
        df_prob = self.conf.cond_collapse_prob[att_value]

        att = self.conf.cond_collapse_prob_metadata[att_value]['by']
        att_type = self.conf.cond_collapse_prob_metadata[att_value]['type']
        if att_type == 'string':
            tf_array = df_prob[att] == getattr(self, att)
        elif att_type == 'numeric':
            tf_array = (df_prob[att + '_lower'] <= self.df_tower[att]) & \
                       (df_prob[att + '_upper'] > self.df_tower[att])

        # change to dictionary
        cond_pc = dict(zip(df_prob.loc[tf_array, 'list'],
                           df_prob.loc[tf_array, 'probability']))
        max_no_adj_towers = \
            self.conf.cond_collapse_prob_metadata[att_value]['max_adj']

        return cond_pc, max_no_adj_towers

    @property
    def get_wind_file(self):
        """
        :return: string
        """
        return self.conf.file_head + self.name + self.conf.file_tail

    @property
    def assign_design_speed(self):
        """ determine design speed """
        self.design_speed = self.conf.design_value[self.line_route]['speed']
        if self.conf.adjust_design_by_topo:
            id_topo = np.sum(
                self.conf.topo_multiplier[self.name] >=
                self.conf.design_adjustment_factor_by_topo['threshold'])
            self.design_speed *= self.conf.design_adjustment_factor_by_topo[id_topo]
        return self.design_speed

    @property
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
        tc: terrain category (defined by line route)
        asset is a Tower class instance.
        """

        tc_str = 'tc' + str(self.terrain_cat)  # Terrain
        try:
            mzcat_z = np.interp(self.height_z,
                                self.conf.terrain_multiplier['height'],
                                self.conf.terrain_multiplier[tc_str])
        except KeyError:
            msg = '{} is undefined in {}'.format(
                tc_str, self.conf.file_terrain_multiplier)
            print(msg)
            raise KeyError(msg)

        idx_10 = self.conf.terrain_multiplier['height'].index(10)
        mzcat_10 = self.conf.terrain_multiplier[tc_str][idx_10]
        return mzcat_z/mzcat_10

    def calculate_cond_pc_adj(self):
        """
        calculate conditional collapse probability of jth tower given ith tower
        P(j|i)
        """

        idx_m1 = np.array([i for i in range(len(self.id_adj))
            if self.id_adj[i] == -1]) - self.max_no_adj_towers  # rel_index

        try:
            max_neg = np.max(idx_m1[idx_m1 < 0]) + 1
        except ValueError:
            max_neg = - self.max_no_adj_towers

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
