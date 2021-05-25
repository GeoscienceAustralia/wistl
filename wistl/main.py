#!/usr/bin/env python

"""
WISTL: Wind Impact Simulation on Transmission Lines
"""

import os
import sys
import time
import collections
import logging.config
import pandas as pd
from optparse import OptionParser
import dask
from dask.distributed import Client
import multiprocessing
import itertools
import functools

from wistl.config import Config, set_towers_and_lines
from wistl.version import VERSION_DESC
from wistl.scenario import Scenario
from wistl.constants import params_event, params_tower, params_line
#from wistl.line import compute_damage_per_line
from wistl.tower import compute_tower_damage
from itertools import repeat

# create a list of towers 
Tower = collections.namedtuple('Tower', params_tower)
Line = collections.namedtuple('Line', params_line)

logger = logging.getLogger(__name__)


def combine_by_line(x0, x1):
    """
    x0: (DF, str)
    x1: (DF, str)
    """
    df0, str0 = x0
    df1, str1 = x1
    xc = df0.merge(df1,
                  how='outer',
                  left_on='Time',
                  right_on='Time',
                  suffixes=(str0, str1))

    return (xc, '')


def run_simulation_alt(cfg, client_ip=None):

    logger.info('parallel MC run on.......')

    #cfg.no_sims = 100000

    dic_towers, dic_lines = set_towers_and_lines(cfg)

    # using default dict or named tuple
    towers = collections.defaultdict(list)
    lines = collections.defaultdict(list)
    for line_name, items in dic_towers.items():
        lines[line_name] = Line(**dic_lines[line_name])
        for tower_name, item in items.items():
            towers[tower_name] = Tower(**item)

    client = Client(client_ip)

    results = []
    #result = collections.defaultdict(dict)
    for event in cfg.events:
        for _, tower in towers.items():

            result = client.submit(compute_tower_damage, event, lines, tower, cfg)
            results.append(result)

    results = client.gather(results)

    client.close()

    # aggregate by event and line
    damage_prob = {}
    for event in cfg.events:
        damage_prob[event.id] = {}
        for line_name in dic_lines.keys():
            dump_str, dump_df = zip(*[(x['tower'], x['dmg']) for x in results if (x['event']==event.id) and (x['line']==line_name)])
            try:
                damage_prob[event.id][line_name] = functools.reduce(combine_by_line, zip(dump_df, dump_str))[0]
            except KeyError:
                pass

    #with multiprocessing.Pool() as pool:
    #    results = pool.starmap(compute_dmg, itertools.product(cfg.events, towers.values()))
    #print(len(results))
    return results

def run_simulation(cfg, client_ip=None):
    """
    main function
    :param cfg: an instance of TransmissionConfig
    """

    tic = time.time()

    logger = logging.getLogger(__name__)

    logger.info('parallel MC run on.......')

    # create a list of towers 
    Tower = namedtuple('Tower', params_tower)
    Line = namedtuple('Line', params_line)

    # using default dict or named tuple
    towers = defaultdict(list)
    lines = defaultdict(list)
    for line_name, items in cfg.towers_by_line.items():
        lines[line_name] = Line(**cfg.lines[line_name])
        for tower_name, item in items.items():
            towers[tower_name] = Tower(**item)

    # first compute analytical 
    if cfg.options['run_parallel']:

        logger.info('parallel MC run on.......')
        client = Client(client_ip)

        lines = []
        for event in cfg.events:

            scenario = Scenario(event=event, cfg=cfg)

            for _, line in scenario.lines.items():

                line = client.submit(line.compute_damage_per_line, cfg=cfg)
                lines.append(line)

        client.gather(lines)
        client.close()

    else:

        logger.info('serial MC run on.......')

        # create transmission network with wind event

        lines = []

        for event in cfg.events:

            scenario = Scenario(event=event, cfg=cfg)

            damage_prob_max = pd.DataFrame(None, columns=cfg.damage_states)

            for _, line in scenario.lines.items():

                _ = line.compute_damage_per_line(cfg=cfg)

                df = pd.DataFrame(None, columns=cfg.damage_states)

                for ds in cfg.damage_states:
                    try:
                        tmp = line.damage_prob[ds].max(axis=0)
                    except KeyError:
                        pass
                    else:
                        df[ds] = tmp

                damage_prob_max = damage_prob_max.append(df)

            if not damage_prob_max.empty:
                damage_prob_max.index.name = 'name'
                damage_prob_max.to_csv(scenario.file_output)
                logger.info(f'{scenario.file_output} is saved')

            if cfg.line_interaction:
                _ = scenario.compute_damage_probability_line_interaction()

    logger.info(f'MC simulation took {time.time() - tic} seconds')

    return lines

def set_logger(path_cfg, logging_level=None):
    """debug, info, warning, error, critical"""

    config_dic = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "[%(levelname)s] %(name)s: %(message)s"
            }
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },

        },

        "loggers": {
        },

        "root": {
            "level": "INFO",
            "handlers": ["console"]
        }
    }

    if logging_level:

        try:
            level = getattr(logging, logging_level.upper())
        except (AttributeError, TypeError):
            logging_level = 'DEBUG'
            level = 'DEBUG'
        finally:
            path_output = os.path.join(path_cfg, 'output')
            file_log = os.path.join(path_output, f'{logging_level}.log')

            if not os.path.exists(path_output):
                os.makedirs(path_output)

            added_file_handler = {"added_file_handler": {
                                  "class": "logging.handlers.RotatingFileHandler",
                                  "level": level,
                                  "formatter": "simple",
                                  "filename": file_log,
                                  "encoding": "utf8",
                                  "mode": "w"}
                            }
            config_dic['handlers'].update(added_file_handler)
            config_dic['root']['handlers'].append('added_file_handler')
            config_dic['root']['level'] = "DEBUG"

    logging.config.dictConfig(config_dic)


def process_commandline():
    usage = '%prog -c <config_file> [-i <client_ip>] [-v <logging_level>]'
    parser = OptionParser(usage=usage, version=VERSION_DESC)
    parser.add_option("-c", "--config",
                      dest="config_file",
                      help="read configuration from FILE",
                      metavar="FILE")
    parser.add_option("-i", "--ip",
                      dest="client_ip",
                      help="set client ip address for dask cluster",
                      metavar="ip_address")
    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      default=None,
                      metavar="logging_level",
                      help="set logging level")
    return parser


def main():

    parser = process_commandline()

    (options, args) = parser.parse_args()

    if options.config_file:
        if not os.path.isfile(options.config_file):
            sys.exit(f'{options.config_file} not found')
        else:
            path_cfg = os.path.dirname(os.path.realpath(options.config_file))
            set_logger(path_cfg, options.verbose)
            conf = Config(file_cfg=options.config_file)
            #run_simulation(cfg=conf, client_ip=options.client_ip)
            start = time.time()
            results = run_simulation_alt(cfg=conf, client_ip=options.client_ip)
            #results = demo_dask(cfg=conf, client_ip=options.client_ip)
            #print(results)
            print(f'Elapsed time: {time.time() - start}')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
