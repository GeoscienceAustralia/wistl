#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import os
import numpy as np
import pandas as pd

from transmission_network import TransmissionNetwork
from damage_tower import DamageTower


def create_event_set(conf):
    """ a collection of wind events
    """

    # read GIS information
    network = TransmissionNetwork(conf)

    events = dict()
    for path_wind in conf.path_wind_timeseries:
        event_id = path_wind.split('/')[-1]
        if event_id in events:
            raise KeyError('{} is already assigned'.format(event_id))
        events[event_id] = DamageNetwork(network, path_wind)
    return events


class DamageNetwork(object):
    """ class for a collection of damage to lines
    """

    def __init__(self, network, path_wind):
        self.network = network
        self.path_wind = path_wind

        self.lines = dict()
        for key, line in self.network.lines.iteritems():
            if key in self.lines:
                raise KeyError('{} is already assigned'.format(key))
            self.lines[key] = DamageLine(line, self.path_wind)

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

    def compute_collapse_of_towers_analytical(self):
        """
        calculate collapse of towers analytically

        Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
        whre Pd: collapse due to direct wind
        Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))

        pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
        """

        idx_time = self.towers[self.line.name_by_line[0]].wind.index
        ntime = len(idx_time)
        ntower = self.line.no_towers
        cds_list = [x[0] for x in self.line.conf.damage_states]  # only string
        cds_list.remove('collapse')  # non-collapse

        pc_adj_agg = np.zeros((ntower, ntower, ntime))

        for irow, name in enumerate(self.line.name_by_line):
            for j in self.towers[name].pc_adj.keys():
                jcol = self.line.id_by_line.index(j)
                pc_adj_agg[irow, jcol, :] = self.towers[name].pc_adj[j]
            pc_adj_agg[irow, irow, :] = self.towers[name].pc_wind.collapse.values

        pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)  # (ntower, ntime)

        # prob of non-collapse damage
        df_prob = {}
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
                                       index=idx_time)

        df_prob['collapse'] = pd.DataFrame(pc_collapse.T,
                                           columns=self.line.name_by_line,
                                           index=idx_time)

        return df_prob
