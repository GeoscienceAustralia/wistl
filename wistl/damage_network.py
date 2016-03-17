#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import os
import numpy as np

from wistl.transmission_network import TransmissionNetwork
from wistl.damage_line import DamageLine


def create_damaged_network(conf):
    """
    dict of damaged network states
    """

    # network = TransmissionNetwork(conf)
    #print(network.lines['Calaca - Amadeo'].towers)
    damaged_networks = dict()
    for path_wind in conf.path_wind_scenario:
        event_id = path_wind.split('/')[-1]

        if event_id in damaged_networks:
            raise KeyError('{} is already assigned'.format(event_id))

        path_output_scenario = os.path.join(conf.path_output, event_id)
        if not os.path.exists(path_output_scenario):
            os.makedirs(path_output_scenario)

        damaged_networks[event_id] = DamageNetwork(conf, event_id)

        #print(network.lines['Calaca - Amadeo'].towers)

    return damaged_networks


class DamageNetwork(TransmissionNetwork):
    """ class for a collection of damage to lines
    """

    def __init__(self, conf, event_id):

        self.event_id = event_id
        super(DamageNetwork, self).__init__(conf)

        # line is a TransmissionLine instance
        for key, line in self.lines.iteritems():
            self.lines[key] = DamageLine(line, event_id)

        # assuming same time index for each tower in the same network
        self.time_index = self.lines[key].time_index


def mc_loop_over_line(damage_line):

    event_id = damage_line.event_id
    line_name = damage_line.name

    if damage_line.conf.random_seed:
        try:
            seed = damage_line.conf.seed[event_id][line_name]
        except KeyError:
            msg = '{}:{} is undefined. Check the config file'.format(
                event_id, line_name)
            print(msg)
            raise KeyError(msg)
    else:
        seed = None

    prng = np.random.RandomState(seed)

    # perfect correlation within a single line
    rv = prng.uniform(size=(damage_line.conf.nsims,
                            len(damage_line.time_index)))

    for tower in damage_line.towers.itervalues():
        tower.compute_mc_adj(rv, seed)

    damage_line.compute_damage_probability_simulation()

    if not damage_line.conf.skip_non_cascading_collapse:
        damage_line.compute_damage_probability_simulation_non_cascading()
