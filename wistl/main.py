#!/usr/bin/env python

"""
WISTL: Wind Impact Simulation on Transmission Lines
"""

import os
import sys
import time
import logging.config
from optparse import OptionParser
import dask
from dask.distributed import Client

from wistl.config import Config
from wistl.version import VERSION_DESC
from wistl.scenario import Scenario
from wistl.line import compute_damage_per_line


def run_simulation(cfg, client_ip=None):
    """
    main function
    :param cfg: an instance of TransmissionConfig
    """

    tic = time.time()

    logger = logging.getLogger(__name__)

    if cfg.options['run_parallel']:

        logger.info('parallel MC run on.......')
        client = Client(client_ip)

        lines = []
        for event in cfg.events:

            scenario = Scenario(event=event, cfg=cfg)

            for _, line in scenario.lines.items():

                line = client.submit(compute_damage_per_line, line=line, cfg=cfg)
                lines.append(line)

        client.gather(lines)
        client.close()

    else:

        logger.info('serial MC run on.......')

        # create transmission network with wind event

        lines = []

        for event in cfg.events:

            scenario = Scenario(event=event, cfg=cfg)

            for _, line in scenario.lines.items():

                lines.append(line)

                _ = compute_damage_per_line(line=line, cfg=cfg)

            # if cfg.line_interaction:
            #     network_dic = \
            #         compute_damage_probability_line_interaction_per_network(
            #             network_dic)

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
    usage = '%prog -c <config_file> [-v <logging_level>]'
    parser = OptionParser(usage=usage, version=VERSION_DESC)
    parser.add_option("-c", "--config",
                      dest="config_file",
                      help="read configuration from FILE",
                      metavar="FILE")
    parser.add_option("-i", "--ip",
                      dest="client_ip",
                      help="set client ip address",
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
        if not os.path.isfile(options.config_file):
            sys.exit(f'{options.config_file} not found')
        else:
            path_cfg = os.path.dirname(os.path.realpath(options.config_file))
            set_logger(path_cfg, options.verbose)
            conf = Config(file_cfg=options.config_file)
            run_simulation(cfg=conf, client_ip=options.client_ip)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
