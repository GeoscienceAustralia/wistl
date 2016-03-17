#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import scipy.stats as stats
import pandas as pd

from wistl.tower import Tower


class DamageTower(object):

    """
    class DamageTower
    Inputs:
    tower: instance of tower class
    vel_file: velocity file containing velocity time history
    at this tower location.
    """

    def __init__(self, conf, tower, file_wind):
        # super(DamageTower, self).__init__(conf=conf, df_tower=tower.df_tower)
        self._parent = tower
        self.file_wind = file_wind
        self.wind = self.read_wind_timeseries()
        self.time_index = self.wind.index

        # analytical method
        self.pc_wind = None  # compute_pc_wind
        self.pc_adj = None  # dict (ntime,) <- compute_pc_adj

        # simulation method
        self.mc_wind = None  # dict(nsims, ntime)
        self.mc_adj = None  # dict <- compute_mc_adj

    def __getattr__(self, attr_name):
        return getattr(self._parent, attr_name)

    def read_wind_timeseries(self):
        # Time,Longitude,Latitude,Speed,UU,VV,Bearing,Pressure
        try:
            data = pd.read_csv(self.file_wind, header=0, parse_dates=[0],
                               index_col=[0], usecols=[0, 3, 6],
                               names=['', '', '', 'speed', '', '', 'bearing',
                                      ''])
        except IOError:
            print('file {} does not exist'.format(self.file_wind))

        speed = data['speed'].values
        bearing = np.deg2rad(data['bearing'].values)  # degree

        # angle of conductor relative to NS
        t0 = np.deg2rad(self.strong_axis) - np.pi/2.0

        data['dir_speed'] = pd.Series(
            self.convert_factor * self.compute_directional_wind_speed(
                speed, bearing, t0), index=data.index)
        return data

    @staticmethod
    def compute_directional_wind_speed(speed, bearing, t0):

        # angle between wind direction and tower conductor
        phi = np.abs(bearing - t0)
        tf = (phi <= np.pi/4) | (phi > np.pi/4*7) | ((phi > np.pi/4*3) &
             (phi <= np.pi/4*5))

        cos_ = abs(np.cos(np.pi/4.0-phi))
        sin_ = abs(np.sin(np.pi/4.0-phi))

        adj = speed*np.max(np.vstack((cos_, sin_)), axis=0)
        return np.where(tf, adj, speed)  # adj if true, otherwise speed

    def compute_pc_wind(self):
        """
        compute probability of damage due to wind
        """
        pc_wind = np.zeros(shape=(len(self.time_index),
                                  self.conf.no_damage_states))
        vratio = self.wind.dir_speed.values/self.collapse_capacity

        tf_array = np.ones((self.conf.fragility.shape[0],), dtype=bool)
        for att, att_type in zip(self.conf.fragility_metadata['by'],
                                 self.conf.fragility_metadata['type']):
            if att_type == 'string':
                tf_array *= self.conf.fragility[att] == self.df_tower[att]
            elif att_type == 'numeric':
                tf_array *= (self.conf.fragility[att + '_lower'] <= self.df_tower[att]) & \
                            (self.conf.fragility[att + '_upper'] > self.df_tower[att])

        for ids, limit_state in enumerate(self.conf.damage_states):

            idx = tf_array & (self.conf.fragility['limit_states'] == limit_state)
            fn_form = self.conf.fragility.loc[idx, self.conf.fragility_metadata['function']].values[0]
            arg_ = self.conf.fragility.loc[idx, self.conf.fragility_metadata[fn_form]['arg']]
            scale_ = self.conf.fragility.loc[idx, self.conf.fragility_metadata[fn_form]['scale']]

            temp = getattr(stats, fn_form).cdf(vratio, arg_, scale=scale_)
            pc_wind[:, ids] = temp

        self.pc_wind = pd.DataFrame(
            pc_wind,
            columns=self.conf.damage_states,
            index=self.wind.index)

    def compute_pc_adj(self):  # only for analytical approach
        """
        calculate collapse probability of jth tower due to pull by the tower
        Pc(j,i) = P(j|i)*Pc(i)
        """
        # only applicable for tower collapse

        pc_adj = {}
        for rel_idx in self.cond_pc_adj.keys():
            abs_idx = self.id_adj[rel_idx + self.max_no_adj_towers]
            pc_adj[abs_idx] = (self.pc_wind.collapse.values *
                               self.cond_pc_adj[rel_idx])

        self.pc_adj = pc_adj

    def compute_mc_adj(self, rv, seed=None):
        """
        2. determine if adjacent tower collapses or not due to pull by the tower
        jtime: time index (array)
        idx: multiprocessing thread id
        """

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

        self.mc_wind = mc_wind
        self.mc_adj = mc_adj

if __name__ == '__main__':
    from config_class import TransmissionConfig
    conf = TransmissionConfig()
    from wistl.transmission_network import TransmissionNetwork
    network = TransmissionNetwork(conf)
    tower, sel_lines, fid_by_line, id2name, lon, lat =\
        network.read_tower_gis_information(conf)

