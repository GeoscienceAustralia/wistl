#!/usr/bin/env python
from __future__ import print_function

"""
WISTL: Wind Impact Simulation on Transmission Lines
"""

import time
import parmap
from optparse import OptionParser
import os
import logging.config

from wistl.config import Config
from wistl.network import (
    create_transmission_network_under_wind_event,
    compute_damage_probability_line_interaction_per_network)
from wistl.line import compute_damage_probability_per_line
from wistl.version import VERSION_DESC


def run_simulation(cfg):
    """
    main function
    :param cfg: an instance of TransmissionConfig
    :return: damaged_networks: list of damaged networks, dictionaries of
                               damaged lines
    """

    tic = time.time()
    damage_by_event = {}

    logger = logging.getLogger(__name__)

    if cfg.options['run_parallel']:

        logger.info('parallel MC run on.......')

        # create transmission network with wind event
        networks = parmap.map(
            create_transmission_network_under_wind_event, cfg.events, cfg)

        lines = []
        for network in networks:
            for _, line in network.lines.itms():
                lines.append(line)

        # compute damage probability for each pair of line and wind event
        lines = parmap.map(compute_damage_probability_per_line, lines, cfg)

        nested_dic = {}
        for line in lines:
            nested_dic.setdefault(line.event_id, {})[line.name] = line

        # if cfg.line_interaction:
        #     damaged_networks = parmap.map(
        #         compute_damage_probability_line_interaction_per_network,
        #         [network for network in nested_dic.itervalues()])
        # else:
        damage_by_event = [network for _, network in nested_dic.items()]

    else:

        logger.info('serial MC run on.......')

        # create transmission network with wind event
        for event in cfg.events:

            network = create_transmission_network_under_wind_event(
                event, cfg)

            for _, line in network.lines.items():
                compute_damage_probability_per_line(line=line, cfg=cfg)

            network = {line.name: line for _, line in network.lines.items()}

            # if cfg.line_interaction:
            #     network_dic = \
            #         compute_damage_probability_line_interaction_per_network(
            #             network_dic)

            damage_by_event[event] = network

    logger.info('MC simulation took {} seconds'.format(time.time() - tic))

    return damage_by_event


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
            file_log = os.path.join(path_cfg, 'output', '{}.log'.format(logging_level))
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
    usage = '%prog -c <config_file> [-v <logging_level>]'
    parser = OptionParser(usage=usage, version=VERSION_DESC)
    parser.add_option("-c", "--config",
                      dest="config_file",
                      help="read configuration from FILE",
                      metavar="FILE")
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
        path_cfg = os.path.dirname(os.path.realpath(options.config_file))
        set_logger(path_cfg, options.verbose)

        conf = Config(file_cfg=options.config_file)
        _ = run_simulation(conf)
    else:
        print('Error: Must provide a config file to run')
        parser.print_help()

if __name__ == '__main__':
    main()
