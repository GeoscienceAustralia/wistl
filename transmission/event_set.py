#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import os
import numpy as np
import pandas as pd

from transmission_network import TransmissionNetwork


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

        # assuming same time index for each tower in the same network
        self.idx_time = self.lines[key].idx_time




    def mc_simulation_over_line(self, idx_time):

        ntime = len(idx_time)
        damage_states = conf.damage_states

        if conf.test:
            print('we are in test, Loop {}'.format(id))
            prng = np.random.RandomState(id)
        else:
            print('MC sim, Loop: {}'.format(id))
            prng = np.random.RandomState()
            id = None  # required for true random inside cal_mc_adj

        rv = prng.uniform(size=(conf.nsims, network.))  # perfect correlation within a single line

        for i in fid_by_line[line]:
            event[id2name[i]].cal_mc_adj(tower[id2name[i]], damage_states, rv, id)

        # compute estimated number and probability of towers without considering
        # cascading effect
        est_ntower_nc, prob_ntower_nc = cal_exp_std_no_cascading(fid_by_line[line],
                                                                 event,
                                                                 id2name,
                                                                 damage_states,
                                                                 conf.nsims,
                                                                 idx_time,
                                                                 ntime)

        # compute collapse of tower considering cascading effect
        tf_sim, prob_sim = cal_collapse_of_towers_mc(fid_by_line[line],
                                                     event,
                                                     id2name,
                                                     damage_states,
                                                     conf.nsims,
                                                     idx_time,
                                                     ntime)
        est_ntower, prob_ntower = cal_exp_std(tf_sim, idx_time)
        if conf.flag_save:
            line_ = line.replace(' - ', '_')
            for (ds, _) in damage_states:
                npy_file = os.path.join(conf.dir_output,
                                        'tf_line_mc_{}_{}.npy'.format(ds, line_))
                np.save(npy_file, tf_sim[ds])

                csv_file = os.path.join(conf.dir_output,
                                        'pc_line_mc_{}_{}.csv'.format(ds, line_))
                prob_sim[ds].to_csv(csv_file)

                csv_file = os.path.join(conf.dir_output,
                                        'est_ntower_{}_{}.csv'.format(ds, line_))
                est_ntower[ds].to_csv(csv_file)

                npy_file = os.path.join(conf.dir_output,
                                        'prob_ntower_{}_{}.npy'.format(ds, line_))
                np.save(npy_file, prob_ntower[ds])

                csv_file = os.path.join(conf.dir_output,
                                        'est_ntower_nc_{}_{}.csv'.format(ds, line_))
                est_ntower_nc[ds].to_csv(csv_file)

                npy_file = os.path.join(conf.dir_output,
                                        'prob_ntower_nc_{}_{}.npy'.format(ds,
                                                                          line_))
                np.save(npy_file, prob_ntower_nc[ds])
        print('loop {} finished'.format(id))
        return tf_sim, prob_sim, est_ntower, prob_ntower, est_ntower_nc,\
            prob_ntower_nc




