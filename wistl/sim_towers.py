#!/usr/bin/env python
from __future__ import print_function

"""
WISTL: Wind Impact Simulation on Transmission Lines
"""

import time
import sys
import os
import parmap

from wistl.damage_network import create_event_set, mc_loop_over_line


def sim_towers(cfg):

    if not os.path.exists(cfg.path_output):
        os.makedirs(cfg.path_output)

    events = create_event_set(cfg)

    tic = time.time()

    if cfg.analytical:
        print('Computing damage probability using analytical method')

        for event_key, network in events.iteritems():
            print(' event: {}'.format(event_key))

            for line_key, line in network.lines.iteritems():
                line.compute_damage_probability_analytical()

        print('Analytical method took {} seconds'.format(time.time() - tic))

    tic = time.time()

    if cfg.simulation:
        print('Computing damage probability using simulation method')

        if cfg.parallel:
            print('parallel MC run on.......')
            _list = [line for network in events.itervalues()
                     for line in network.lines.itervalues()]
            parmap.map(mc_loop_over_line, _list)
        else:
            print('serial MC run on.......')
            for event_key, network in events.iteritems():
                print(' event: {}'.format(event_key))
                for line in network.lines.itervalues():
                    mc_loop_over_line(line)

        print('MC simulation took {} seconds'.format(time.time() - tic))

    return events

if __name__ == '__main__':

    args = sys.argv[1:]

    if not args:
        print('python sim_towers.py <config-file>')
        sys.exit(1)

    from wistl.config_class import TransmissionConfig
    conf = TransmissionConfig(cfg_file=args[0])
    scenario_events = sim_towers(conf)
