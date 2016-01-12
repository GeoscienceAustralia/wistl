#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import os
import copy
import numpy as np
#import pandas as pd

from transmission_network import TransmissionNetwork
from damage_line import DamageLine


def create_event_set(conf):
    """ a collection of wind events
    """

    network = TransmissionNetwork(conf)
    #print(network.lines['Calaca - Amadeo'].towers)
    events = dict()

    for path_wind in conf.path_wind_scenario:

        event_id = path_wind.split('/')[-1]

        if event_id in events:
            raise KeyError('{} is already assigned'.format(event_id))

        path_output_scenario = os.path.join(conf.path_output, event_id)
        if not os.path.exists(path_output_scenario):
            os.makedirs(path_output_scenario)

        events[event_id] = DamageNetwork(network, path_wind)

        #print(network.lines['Calaca - Amadeo'].towers)

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


def mc_loop_over_line(damage_line):

    event_id = damage_line.event_id
    line_name = damage_line.name
    if damage_line.conf.random_seed:
        try:
            seed = damage_line.conf.seed[event_id][line_name]
        except KeyError:
            print('{}:{} is undefined. Check the config file'.format(
                event_id, line_name))
    else:
        seed = None

    prng = np.random.RandomState(seed)

    # perfect correlation within a single line
    rv = prng.uniform(size=(damage_line.conf.nsims,
                            len(damage_line.time_index)))

    for _, tower in damage_line.towers.iteritems():
        tower.compute_mc_adj(rv, seed)

    damage_line.compute_damage_probability_simulation()

    if not damage_line.conf.skip_non_cascading_collapse:
        damage_line.compute_damage_probability_simulation_non_cascading()
