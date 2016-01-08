#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import os
import copy
import numpy as np
import parmap
#import pandas as pd

from transmission_network import TransmissionNetwork
from damage_line import DamageLine


def create_event_set(conf):
    """ a collection of wind events
    """

    network = TransmissionNetwork(conf)
    print(network.lines['Calaca - Amadeo'].towers)
    events = dict()

    for path_wind in conf.path_wind_scenario:

        event_id = path_wind.split('/')[-1]

        if event_id in events:
            raise KeyError('{} is already assigned'.format(event_id))

        path_output_scenario = os.path.join(conf.path_output, event_id)
        if not os.path.exists(path_output_scenario):
            os.makedirs(path_output_scenario)

        events[event_id] = DamageNetwork(network, path_wind)

        print(network.lines['Calaca - Amadeo'].towers)

    return events


class DamageNetwork(TransmissionNetwork):
    """ class for a collection of damage to lines
    """

    def __init__(self, network, path_wind):

        network_ = copy.deepcopy(network)  # avoid any change to network

        self._parent = network_  # instance of TransmissionNetwork class
        self.path_wind = path_wind
        self.event_id = path_wind.split('/')[-1]
        #self.damage_lines = dict()

        for key, line in network_.lines.iteritems():
            self.lines[key] = DamageLine(line, self.path_wind)

        # assuming same time index for each tower in the same network
        self.time_index = self.lines[key].time_index

    def __getattr__(self, attr_name):
        return getattr(self._parent, attr_name)

    def mc_simulation(self):

        if self.conf.parallel:
            print('parallel MC run on.......')
            parmap.map(self.mc_loop_over_line, self.lines.values())

        else:
            print('serial MC run on.......')
            for _, line in self.lines.iteritems():
                self.mc_loop_over_line(line)

    def mc_loop_over_line(self, damage_line):

        if self.conf.random_seed:
            try:
                seed = self.conf.seed[self.event_id][damage_line.name]
            except KeyError:
                print('{}:{} is undefined. Check the config file'.format(
                    self.event_id, damage_line.name))
        else:
            seed = None

        prng = np.random.RandomState(seed)

        # perfect correlation within a single line
        rv = prng.uniform(size=(self.conf.nsims,
                                len(damage_line.time_index)))

        for _, tower in damage_line.towers.iteritems():
            tower.compute_mc_adj(rv, seed)

        #damage_line.est_damage_tower, damage_line.prob_damage_tower = \
        damage_line.compute_damage_probability_simulation()

        if not self.conf.skip_non_cascading_collapse:
            damage_line.compute_damage_probability_simulation_non_cascading()

    #return est_damage_tower, prob_damage_tower

        # if self.conf.save:
        #     line_ = line.replace(' - ', '_')
        #     for (ds, _) in damage_states:
        #         npy_file = os.path.join(conf.dir_output,
        #                                 'tf_line_mc_{}_{}.npy'.format(ds, line_))
        #         np.save(npy_file, tf_sim[ds])

        #         csv_file = os.path.join(conf.dir_output,
        #                                 'pc_line_mc_{}_{}.csv'.format(ds, line_))
        #         prob_sim[ds].to_csv(csv_file)

        #         csv_file = os.path.join(conf.dir_output,
        #                                 'est_ntower_{}_{}.csv'.format(ds, line_))
        #         est_ntower[ds].to_csv(csv_file)

        #         npy_file = os.path.join(conf.dir_output,
        #                                 'prob_ntower_{}_{}.npy'.format(ds, line_))
        #         np.save(npy_file, prob_ntower[ds])

        #         csv_file = os.path.join(conf.dir_output,
        #                                 'est_ntower_nc_{}_{}.csv'.format(ds, line_))
        #         est_ntower_nc[ds].to_csv(csv_file)

        #         npy_file = os.path.join(conf.dir_output,
        #                                 'prob_ntower_nc_{}_{}.npy'.format(ds,
        #                                                                   line_))
        #         np.save(npy_file, prob_ntower_nc[ds])
        # print('loop {} finished'.format(id))
        # return tf_sim, prob_sim, est_ntower, prob_ntower, est_ntower_nc,\
        #     prob_ntower_nc


