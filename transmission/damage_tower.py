#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from scipy.stats import lognorm
import pandas as pd


class DamageTower(object):

    """
    class DamageTower
    Inputs:
    tower: instance of tower class
    vel_file: velocity file containing velocity time history
    at this tower location.
    """

    def __init__(self, tower, vel_file):
        self.tower = tower
        self.vel_file = vel_file
        self.wind = self.read_wind_timeseries()
        self.idx_time = self.wind.index

        self.pc_wind = None
        self.pc_adj = None  # dict (ntime,) <- cal_pc_adj_towers
        self.mc_wind = None  # dict(nsims, ntime)
        self.mc_adj = None  # dict

    def read_wind_timeseries(self):
        # Time,Longitude,Latitude,Speed,UU,VV,Bearing,Pressure
        try:
            data = pd.read_csv(self.vel_file, header=0, parse_dates=[0],
                               index_col=[0], usecols=[0, 3, 6],
                               names=['', '', '', 'speed', '', '', 'bearing',
                                      ''])
        except IOError:
            print('file {} does not exist'.format(self.vel_file))

        speed = data['speed'].values
        bearing = np.deg2rad(data['bearing'].values)  # degree

        # angle of conductor relative to NS
        t0 = np.deg2rad(self.tower.strong_axis) - np.pi/2.0
        convert_factor = self.convert_10_to_z()
        data['dir_speed'] = pd.Series(
            convert_factor * self.compute_directional_wind_speed(
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

    def convert_10_to_z(self):
        """
        Mz,cat(h=10)/Mz,cat(h=z)
        tc: terrain category (defined by line route)
        asset is a Tower class instance.
        """

        tc_str = 'tc' + str(self.tower.terrain_cat)  # Terrain
        try:
            mzcat_z = np.interp(self.tower.height_z,
                                self.tower.conf.terrain_multiplier['height'],
                                self.tower.conf.terrain_multiplier[tc_str])
        except KeyError:
            print('{} is not defined'.format(tc_str))
#            return {'error': "{} is not defined".format(tc_str)}  # these errors should be handled properly

        idx_10 = self.tower.conf.terrain_multiplier['height'].index(10)
        mzcat_10 = self.tower.conf.terrain_multiplier[tc_str][idx_10]
        return mzcat_z/mzcat_10

    def compute_pc_wind(self):
        """
        compute probability of damage due to wind
        - asset: instance of Tower object
        - frag: dictionary by asset.const_type
        - ntime:
        - damage_states: [('collapse', 2), ('minor', 1)]
        - nds:
        """
        pc_wind = np.zeros(shape=(len(self.idx_time),
                                  self.tower.conf.no_damage_states))
        vratio = self.wind.dir_speed.values/self.tower.collapse_capacity

        try:
            fragx = self.tower.conf.fragility_curve[self.tower.ttype][self.tower.funct]
            idf = np.sum(fragx['dev_angle'] <= self.tower.dev_angle)

            for (ds, ids) in self.tower.conf.damage_states:  # damage state
                med = fragx[idf][ds]['param0']
                sig = fragx[idf][ds]['param1']

                temp = lognorm.cdf(vratio, sig, scale=med)
                pc_wind[:, ids-1] = temp  # 2->1

        except KeyError:
                print('fragility is not defined for {}'.format(
                      self.tower.const_type))

        self.pc_wind = pd.DataFrame(
            pc_wind,
            columns=[x[0] for x in self.tower.conf.damage_states],
            index=self.wind.index)

    def compute_pc_adj(self):  # only for analytical approach
        """
        calculate collapse probability of jth tower due to pull by the tower
        Pc(j,i) = P(j|i)*Pc(i)
        """
        # only applicable for tower collapse

        pc_adj = {}
        for rel_idx in self.tower.cond_pc_adj.keys():
            abs_idx = self.tower.id_adj[rel_idx +
                                          self.tower.max_no_adj_towers]
            pc_adj[abs_idx] = (self.pc_wind.collapse.values *
                               self.tower.cond_pc_adj[rel_idx])

        self.pc_adj = pc_adj

    def compute_mc_adj(self, asset, damage_states, rv, idx):
        """
        2. determine if adjacent tower collapses or not due to pull by the tower
        jtime: time index (array)
        idx: multiprocessing thread id
        """

        # 1. determine damage state of tower due to wind
        val = np.array([rv < self.pc_wind[ds[0]].values
                       for ds in damage_states])  # (nds, nsims, ntime)

        ds_wind = np.sum(val, axis=0)  # (nsims, ntime) 0(non), 1, 2 (collapse)

        mc_wind = dict()
        for ds, ids in damage_states:
            mc_wind.setdefault(ds, {})['isim'],\
            mc_wind.setdefault(ds, {})['itime'] = np.where(ds_wind == ids)

        # for collapse
        unq_itime = np.unique(mc_wind['collapse']['itime'])
        nprob = len(asset.cond_pc_adj_mc['cum_prob'])  #

        mc_adj = {}  # impact on adjacent towers

        if idx:
            prng = np.random.RandomState(idx)
        else:
            prng = np.random.RandomState()

        if nprob > 0:
            for jtime in unq_itime:
                jdx = np.where(mc_wind['collapse']['itime'] == jtime)[0]
                idx_sim = mc_wind['collapse']['isim'][jdx]  # index of simulation
                nsims = len(idx_sim)
                rv = prng.uniform(size=nsims)

                list_idx_cond = map(lambda rv_: sum(
                    rv_ >= asset.cond_pc_adj_mc['cum_prob']), rv)

                # ignore simulation where none of adjacent tower collapses
                unq_list_idx_cond = set(list_idx_cond) - set([nprob])

                for idx_cond in unq_list_idx_cond:

                    # list of idx of adjacent towers in collapse
                    rel_idx = asset.cond_pc_adj_mc['rel_idx'][idx_cond]

                    # convert relative to absolute fid
                    abs_idx = [asset.adj_list[j + asset.max_no_adj_towers]
                               for j in rel_idx]

                    # filter simulation
                    isim = [i for i, x in enumerate(list_idx_cond)
                            if x == idx_cond]
                    mc_adj.setdefault(jtime, {})[tuple(abs_idx)] = idx_sim[isim]

        self.mc_wind = mc_wind
        self.mc_adj = mc_adj
        return

# if __name__ == '__main__':
#     from config_class import TransmissionConfig
#     conf = TransmissionConfig()
#     from read import TransmissionNetwork
#     network = TransmissionNetwork(conf)
#     tower, sel_lines, fid_by_line, id2name, lon, lat =\
#         network.read_tower_gis_information(conf)

