#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import os
import copy
import numpy as np
import pandas as pd
import warnings
from scipy.stats import itemfreq

from damage_tower import DamageTower
from transmission_line import TransmissionLine

# ignore NaturalNameWarning
warnings.filterwarnings("ignore", lineno=100, module='tables')


class DamageLine(TransmissionLine):
    """ class for a collectin of damage to line """

    def __init__(self, line, path_wind):

        line_ = copy.deepcopy(line)  # avoid any change to line

        self._parent = line_  # instance of TransmissionLine class
        self.event_id = path_wind.split('/')[-1]

        for key, tower in line_.towers.iteritems():

            file_wind = os.path.join(path_wind, tower.file_wind)
            self.towers[key] = DamageTower(tower, file_wind)

            # compute pc_wind and pc_adj
            self.towers[key].compute_pc_wind()
            self.towers[key].compute_pc_adj()

        # assuming same time index for each tower in the same network
        self.time_index = self.towers[key].time_index

        # analytical method
        self.damage_prob_analytical = dict()

        # simulation method
        self.damage_prob_simulation = dict()
        self.prob_no_damage = None
        self.est_no_damage = None

        # non cascading collapse
        self.damage_prob_simulation_non_cascading = dict()
        self.prob_no_damage_non_cascading = None
        self.est_no_damage_non_cascading = None

    def __getattr__(self, attr_name):
        return getattr(self._parent, attr_name)

    def write_hdf5(self, file_str, val):

        h5file = os.path.join(self.conf.path_output,
                              self.event_id,
                              '{}_{}.h5'.format(file_str, self.name_output))
        hdf = pd.HDFStore(h5file)

        for ds, _ in self.conf.damage_states:
            hdf.put(ds, val[ds], format='table', data_columns=True)
        hdf.close()
        print('{} is created'.format(h5file))

    def compute_damage_probability_analytical(self):
        """
        calculate damage probability of towers analytically

        Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
        whre Pd: collapse due to direct wind
        Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))

        pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
        """

        pc_adj_agg = np.zeros((self.no_towers,
                               self.no_towers,
                               len(self.time_index)))

        # prob of collapse
        for irow, name in enumerate(self.name_by_line):
            for j in self.towers[name].pc_adj.keys():
                jcol = self.id_by_line.index(j)
                pc_adj_agg[irow, jcol, :] = self.towers[name].pc_adj[j]
            pc_adj_agg[irow, irow, :] = self.towers[name].pc_wind.collapse.values

        pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)  # (ntower, ntime)

        self.damage_prob_analytical['collapse'] = pd.DataFrame(
            pc_collapse.T,
            columns=self.name_by_line,
            index=self.time_index)

        # prob of non-collapse damage
        cds_list = [x for x, _ in self.conf.damage_states]  # only string
        cds_list.remove('collapse')  # non-collapse

        for ds in cds_list:
            temp = np.zeros_like(pc_collapse)
            for irow, name in enumerate(self.name_by_line):
                val = (self.towers[name].pc_wind[ds].values
                       - self.towers[name].pc_wind.collapse.values
                       + pc_collapse[irow, :])
                val = np.where(val > 1.0, 1.0, val)
                temp[irow, :] = val

            self.damage_prob_analytical[ds] = pd.DataFrame(
                temp.T,
                columns=self.name_by_line,
                index=self.time_index)

        if self.conf.save:
            self.write_hdf5(file_str='damage_prob_analytical',
                            val=self.damage_prob_analytical)

    def compute_damage_probability_simulation(self):

        tf_ds = np.zeros((self.no_towers,
                          self.conf.nsims,
                          len(self.time_index)), dtype=bool)

        # collapse by adjacent towers
        for name in self.name_by_line:

            for j in self.towers[name].mc_adj.keys():  # time

                for k in self.towers[name].mc_adj[j].keys():  # fid

                    isim = self.towers[name].mc_adj[j][k]

                    for l in k:  # each fid

                        tf_ds[self.id_by_line.index(l), isim, j] = True

        cds_list = self.conf.damage_states[:]  # to avoid effect
        cds_list.reverse()  # [(collapse, 2), (minor, 1)]

        tf_sim = dict()

        # append damage stae by direct wind
        for ds, _ in cds_list:

            for irow, name in enumerate(self.name_by_line):

                for j, k in zip(self.towers[name].mc_wind[ds]['isim'],
                                self.towers[name].mc_wind[ds]['itime']):

                    tf_ds[irow, j, k] = True

            self.damage_prob_simulation[ds] = pd.DataFrame(
                np.sum(tf_ds, axis=1).T/float(self.conf.nsims),
                columns=self.name_by_line,
                index=self.time_index)

            tf_sim[ds] = np.copy(tf_ds)  # why copy??

        self.est_no_damage, self.prob_no_damage = \
            self.compute_damage_stats(tf_sim)

        if self.conf.save:

            self.write_hdf5(file_str='damage_prob_simulation',
                            val=self.damage_prob_simulation)

            self.write_hdf5(file_str='est_no_damage_simulation',
                            val=self.est_no_damage)

            self.write_hdf5(file_str='prob_no_damage_simulation',
                            val=self.prob_no_damage)

    def compute_damage_probability_simulation_non_cascading(self):

        tf_sim_non_cascading = dict()

        for ds, _ in self.conf.damage_states:

            tf_ds = np.zeros((self.no_towers,
                              self.conf.nsims,
                              len(self.time_index)), dtype=bool)

            for irow, name in enumerate(self.name_by_line):

                isim = self.towers[name].mc_wind[ds]['isim']
                itime = self.towers[name].mc_wind[ds]['itime']

                tf_ds[irow, isim, itime] = True

            self.damage_prob_simulation_non_cascading[ds] = pd.DataFrame(
                np.sum(tf_ds, axis=1).T/float(self.conf.nsims),
                columns=self.name_by_line,
                index=self.time_index)

            tf_sim_non_cascading[ds] = np.copy(tf_ds)

        self.est_no_damage_non_cascading, \
            self.prob_no_damage_non_cascading = \
            self.compute_damage_stats(tf_sim_non_cascading)

        if self.conf.save:

            self.write_hdf5(file_str='damage_prob_simulation_non_cascading',
                            val=self.damage_prob_simulation_non_cascading)

            self.write_hdf5(file_str='est_no_damage_simulation_non_cascading',
                            val=self.est_no_damage_non_cascading)

            self.write_hdf5(file_str='prob_no_damage_simulation_non_cascading',
                            val=self.prob_no_damage_non_cascading)

    def compute_damage_stats(self, tf_sim):
        """
        compute mean and std of no. of ds
        tf_collapse_sim.shape = (ntowers, nsim, ntime)
        """

        est_damage_tower = dict()
        prob_damage_tower = dict()

        ntime = len(self.time_index)
        ntowers = self.no_towers
        nsims = self.conf.nsims

        for ds in tf_sim:

            assert tf_sim[ds].shape == (ntowers, nsims, ntime)

            # mean and standard deviation
            x_ = np.array(range(ntowers + 1))[:, np.newaxis]  # (ntowers, 1)
            x2_ = np.power(x_, 2.0)

            no_ds_across_towers = np.sum(tf_sim[ds], axis=0)  # (nsims, ntime)
            no_freq = np.zeros(shape=(ntime, ntowers + 1))  # (ntime, ntowers)

            for i in range(ntime):
                val = itemfreq(no_ds_across_towers[:, i])  # (value, freq)
                no_freq[i, [int(x) for x in val[:, 0]]] = val[:, 1]

            prob = no_freq / float(nsims)  # (ntime, ntowers)

            exp_ntower = np.dot(prob, x_)
            std_ntower = np.sqrt(np.dot(prob, x2_) - np.power(exp_ntower, 2))

            est_damage_tower[ds] = pd.DataFrame(
                np.hstack((exp_ntower, std_ntower)),
                columns=['mean', 'std'],
                index=self.time_index)

            prob_damage_tower[ds] = pd.DataFrame(
                prob,
                columns=[str(x) for x in range(ntowers+1)],
                index=self.time_index)

        return est_damage_tower, prob_damage_tower
