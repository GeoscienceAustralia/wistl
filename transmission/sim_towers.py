#!/usr/bin/env python
from __future__ import print_function

"""
simulation of collapse of transmission towers
based on empricially derived conditional probability

required input
    -. tower geometry whether rectangle or squre (v) - all of them are square
    -. design wind speed (requiring new field in input) v
    -. tower type and associated collapse fragility v
    -. conditional probability by tower type (suspension and strainer) v
    -. idenfity strainer tower v

Todo:
    -. creating module (v)
    -. visualisation (arcGIS?) (v)
    -. postprocessing of mc results (v)
    -. think about how to sample random numbers (spatial correlation) (v)
    -. adding additional damage state (v)
    -. adj_list by function type (v)
    -. update cond adj each simulation/time
    -. assigning cond (different)
    -. with our without cascading effect (mc simulation) - priority
"""

import time
import sys
import os
import parmap

from damage_network import create_event_set, mc_loop_over_line


def sim_towers(conf):

    if conf.test:
        print("==============>testing")

    if not os.path.exists(conf.path_output):
        os.makedirs(conf.path_output)

    events = create_event_set(conf)

    tic = time.time()

    if conf.analytical:
        print('Computing damage probability using analytical method')

        for event_key, network in events.iteritems():
            print(' event: {}'.format(event_key))

            for line_key, line in network.lines.iteritems():
                line.compute_damage_probability_analytical()

        print('Analytical method took {} seconds'.format(time.time() - tic))
        tic = time.time()

    if conf.simulation:
        print('Computing damage probability using simulation method')

        if conf.parallel:
            list_ = [line for network in events.itervalues()
                     for line in network.lines.itervalues()]
            no_cpus = parmap.multiprocessing.cpu_count()

            if no_cpus >= len(list_):
                print('parallel MC run on.......')
                parmap.map(mc_loop_over_line, list_)

                print('MC simulation took {} seconds'.format(time.time() - tic))
                sys.exit(0)

            else:
                print('Not enough cpus for parallel processing')

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

    from config_class import TransmissionConfig
    conf = TransmissionConfig(cfg_file=args[0])
    events = sim_towers(conf)
