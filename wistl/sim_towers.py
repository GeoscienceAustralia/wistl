#!/usr/bin/env python
from __future__ import print_function

"""
WISTL: Wind Impact Simulation on Transmission Lines
"""

import time
import sys
import parmap

from wistl.transmission_network import create_damaged_network, \
    mc_loop_over_line, compute_damage_probability_simulation_line_interaction


def sim_towers(cfg):
    """
    main function
    :param cfg: an instance of TransmissionConfig
    :return:
    """

    damaged_networks = create_damaged_network(cfg)

    tic = time.time()

    if cfg.analytical:
        print('Computing damage probability using analytical method')

        for event_key, network in damaged_networks.iteritems():
            # print(' event: {}'.format(event_key))
            for line_key, line in network.lines.iteritems():
                line.compute_damage_probability_analytical()

        print('Analytical method took {} seconds'.format(time.time() - tic))

    tic = time.time()

    damaged_lines = None
    if cfg.simulation:
        print('Computing damage probability using simulation method')

        if cfg.parallel:
            print('parallel MC run on.......')

            list_line = []
            for network in damaged_networks.itervalues():
                for line in network.lines.itervalues():
                    list_line.append(line)

            damaged_lines = parmap.map(mc_loop_over_line, list_line)

            collection_by_event_and_line = dict()
            for line in damaged_lines:
                collection_by_event_and_line.setdefault(
                    line.event_id_scale, {})[line.name] = line

        else:
            print('serial MC run on.......')

            collection_by_event_and_line = dict()

            for event_key, network in damaged_networks.iteritems():
                # print(' event: {}'.format(event_key))
                for line in network.lines.itervalues():
                    collection_by_event_and_line.setdefault(
                        line.event_id_scale, {})[line.name] = \
                        mc_loop_over_line(line)

        if cfg.line_interaction:

            for collection_by_event in collection_by_event_and_line.itervalues():
                compute_damage_probability_simulation_line_interaction(collection_by_event)

        print('MC simulation took {} seconds'.format(time.time() - tic))

    return damaged_networks, damaged_lines

if __name__ == '__main__':

    args = sys.argv[1:]

    if not args:
        print('python sim_towers.py <config-file>')
        sys.exit(1)

    from wistl.config import TransmissionConfig
    conf = TransmissionConfig(cfg_file=args[0])
    sim_towers(conf)
