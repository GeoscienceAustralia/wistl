#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import os
import numpy as np
import pandas as pd


from damage_tower import DamageTower


class DamageLine(object):
    """ class for a collectin of damage to line """

    def __init__(self, line, path_wind):
        self.line = line

        self.towers = dict()
        for key, tower in self.line.towers.iteritems():

            if key in self.towers:
                raise KeyError('{} is already assigned'.format(key))

            vel_file = os.path.join(path_wind, tower.file_wind)
            self.towers[key] = DamageTower(tower, vel_file)

            # compute pc_wind and pc_adj
            self.towers[key].compute_pc_wind()
            self.towers[key].compute_pc_adj()

        # assuming same time index for each tower in the same network
        self.idx_time = self.towers[key].idx_time

    def compute_damage_probability_analytical(self):
        """
        calculate damage probability of towers analytically

        Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
        whre Pd: collapse due to direct wind
        Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))

        pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
        """

        df_prob = dict()

        pc_adj_agg = np.zeros((self.line.no_towers,
                               self.line_no_towers,
                               len(self.idx_time)))

        # prob of collapse
        for irow, name in enumerate(self.line.name_by_line):
            for j in self.towers[name].pc_adj.keys():
                jcol = self.line.id_by_line.index(j)
                pc_adj_agg[irow, jcol, :] = self.towers[name].pc_adj[j]
            pc_adj_agg[irow, irow, :] = self.towers[name].pc_wind.collapse.values

        pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)  # (ntower, ntime)

        df_prob['collapse'] = pd.DataFrame(pc_collapse.T,
                                           columns=self.line.name_by_line,
                                           index=self.idx_time)

        # prob of non-collapse damage
        cds_list = [x for x, _ in self.line.conf.damage_states]  # only string
        cds_list.remove('collapse')  # non-collapse

        for ds in cds_list:
            temp = np.zeros_like(pc_collapse)
            for irow, name in enumerate(self.line.name_by_line):
                val = (self.towers[name].pc_wind[ds].values
                       - self.towers[name].pc_wind.collapse.values
                       + pc_collapse[irow, :])
                val = np.where(val > 1.0, 1.0, val)
                temp[irow, :] = val
            df_prob[ds] = pd.DataFrame(temp.T,
                                       columns=self.line.name_by_line,
                                       index=self.idx_time)

        return df_prob

    def compute_damage_probability_simulation(self):

        prob_sim = dict()
        tf_sim = dict()

        tf_ds = np.zeros((self.line.no_towers, self.line.conf.nsims,
                         len(self.idx_time)), dtype=bool)

        # collapse by adjacent towers
        for name in self.line.name_by_line:

            for j in self.towers[name].mc_adj.keys():  # time

                for k in self.towers[name].mc_adj[j].keys():  # fid

                    isim = self.towers[name].mc_adj[j][k]

                    for l in k:  # each fid

                        tf_ds[self.line.id_by_line.index(l), isim, j] = True

        cds_list = self.line.conf.damage_states[:]  # to avoid effect
        cds_list.reverse()  # [(collapse, 2), (minor, 1)]

        # append damage stae by direct wind
        for ds, _ in cds_list:
            for irow, name in enumerate(self.line.name_by_line):
                for (j1, k1) in zip(self.towers[name].mc_wind[ds]['isim'],
                                    self.towers[name].mc_wind[ds]['itime']):

                    tf_ds[irow, j1, k1] = True

            temp = np.sum(tf_ds, axis=1)/float(self.line.conf.nsims)

            prob_sim[ds] = pd.DataFrame(temp.T,
                                        columns=self.line.name_by_line,
                                        index=self.idx_time)

            tf_sim[ds] = np.copy(tf_ds)  # why copy??

        return tf_sim, prob_sim







