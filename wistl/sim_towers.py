#!/usr/bin/env python
from __future__ import print_function

"""
WISTL: Wind Impact Simulation on Transmission Lines
"""

import time
import sys
import parmap

from wistl.transmission_network import (
    create_transmission_network_under_wind_event,
    compute_damage_probability_line_interaction_per_network)
from wistl.transmission_line import compute_damage_probability_per_line


def sim_towers(cfg):
    """
    main function
    :param cfg: an instance of TransmissionConfig
    :return: damaged_networks: list of damaged networks, dictionaries of
                               damaged lines
    """

    tic = time.time()
    damaged_networks = []

    if cfg.parallel:
        print('parallel MC run on.......')

        # create transmission network with wind event
        networks = parmap.map(
            create_transmission_network_under_wind_event,
            cfg.event_id_scale, cfg)

        lines = []
        for network in networks:
            for line in network.lines.itervalues():
                lines.append(line)

        # compute damage probability for each pair of line and wind event
        damaged_lines = parmap.map(compute_damage_probability_per_line, lines)

        nested_dic = dict()
        for line in damaged_lines:
            nested_dic.setdefault(line.event_id_scale, {})[line.name] = line

        if cfg.line_interaction:
            damaged_networks = parmap.map(
                compute_damage_probability_line_interaction_per_network,
                [network for network in nested_dic.itervalues()])
        else:
            damaged_networks = [network for network in nested_dic.itervalues()]

    else:
        print('serial MC run on.......')

        # create transmission network with wind event
        for event_id_scale in cfg.event_id_scale:

            network = create_transmission_network_under_wind_event(
                event_id_scale, cfg)

            for line in network.lines.itervalues():
                compute_damage_probability_per_line(line)

            network_dic = {line.name: line for
                           line in network.lines.itervalues()}

            if cfg.line_interaction:
                network_dic = \
                    compute_damage_probability_line_interaction_per_network(
                        network_dic)

            damaged_networks.append(network_dic)

    print('MC simulation took {} seconds'.format(time.time() - tic))

    return damaged_networks

if __name__ == '__main__':

    args = sys.argv[1:]

    if not args:
        print('python sim_towers.py <config-file>')
        sys.exit(1)

    from wistl.config import TransmissionConfig
    conf = TransmissionConfig(cfg_file=args[0])
    sim_towers(conf)
