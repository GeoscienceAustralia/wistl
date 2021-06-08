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
from dask.distributed import Client

from wistl.config import Config, set_towers_and_lines
from wistl.version import VERSION_DESC
from wistl.constants import params_tower, params_line
from wistl.line import compute_damage_by_line
from wistl.tower import compute_damage_by_tower


# define Tower and Line class
Tower = collections.namedtuple('Tower', params_tower)
Line = collections.namedtuple('Line', params_line)


def run_simulation(cfg, client_ip=None):
    """
    main function
    :cfg: an instance of Config
    :
    """

    tic = time.time()

    logger = logging.getLogger(__name__)

    towers, lines = create_towers_and_lines(cfg)

    if cfg.options['run_parallel']:

        logger.info('parallel MC run on....')
        client = Client(client_ip)

        out = []
        for event in cfg.events:
            results = []
            # compute damage by tower and event
            for _, tower in towers.items():

                result = client.submit(compute_damage_by_tower, tower, event, lines, cfg)
                results.append(result)

            # compute damage by line and event
            for _, line in lines.items():

                result = client.submit(compute_damage_by_line, line, results, event, cfg)
                out.append(result)

        _ = client.gather(out)

        client.close()

        logger.info(f'Elapsed time: {time.time()-tic:.3f}')
    else:

        logger.info('serial MC run on....')

        for event in cfg.events:
            results = []
            for _, tower in towers.items():

                result = compute_damage_by_tower(tower, event, lines, cfg)
                results.append(result)

        # compute damage by line and event
            for _, line in lines.items():

                _ = compute_damage_by_line(line, results, event, cfg)

        logger.info(f'Elapsed time: {time.time()-tic:.3f}')


def create_towers_and_lines(cfg):

    dic_towers, dic_lines = set_towers_and_lines(cfg)

    # using default dict or named tuple
    towers, lines = {}, {}
    for line_name, items in dic_towers.items():
        lines[line_name] = Line(**dic_lines[line_name])
        for tower_name, item in items.items():
            towers[tower_name] = Tower(**item)

    return towers, lines


def compute_damage_probability_line_interaction(self):
    """
    compute damage probability due to line interaction
    :param lines: a dictionary of lines
    :return: lines: a dictionary of lines
    """
    for line_name, line in self.lines.items():

        dic_tf_ds = {}
        tf_ds = np.zeros((line.no_towers,
                          line.no_sims,
                          self.no_time), dtype=bool)

        for trigger, target in self.cfg.line_interaction.items():

            if line_name in target:

                try:
                    id_tower, id_sim, id_time = zip(
                        *self.lines[trigger].dmg_idx_interaction[line_name])
                except ValueError:
                    self.logger.info(f'no interaction applied: from {trigger} to {line_name}')
                else:
                    dt = line.dmg_time_idx[0] - self.dmg_time_idx[0]
                    tf_ds[id_tower, id_sim, id_time + dt] = True
                    self.logger.info(f'interaction applied: from {trigger} to {line_name}')

        # append damage state by line itself
        # due to either direct wind and adjacent towers
        # also need to override non-collapse damage states

        if tf_ds.sum():
            print(f'tf_ds is not empty for {line.name}')
        else:
            print(f'tf_ds is empty for {line.name}')

        # append damage state by either direct wind or adjacent towers
        for ds in self.cfg.damage_states[::-1]:

            line.damage_prob_interaction = {}

            try:
                id_tower, id_sim, id_time = line.dmg_idx[ds]
            except ValueError:
                self.logger.info(f'no damage {ds} for {line_name}')
            else:
                dt = line.dmg_time_idx[0] - self.dmg_time_idx[0]
                tf_ds[id_tower, id_sim, id_time + dt] = True

                line.damage_prob_interaction[ds] = pd.DataFrame(tf_ds.sum(axis=1).T / line.no_sims,
                    columns=line.names, index=self.time)

                dic_tf_ds[ds] = np.copy(tf_ds)

            # check whether collapse induced by line interaction
            #tf_ds_itself[id_tower, id_sim, id_time] = True

            #collapse_by_interaction = np.logical_xor(tf_sim['collapse'],
            #                                         tf_ds_itself)

            #if np.any(collapse_by_interaction):
            #    print(f'{line_name} is affected by line interaction')


        # compute mean and std of no. of towers for each of damage states
        try:
            line.no_damage_interaction, line.prob_no_damage_interaction = line.compute_stats(dic_tf_ds)
        except KeyError:
            print(dic_tf_ds)


def run_simulation_old(cfg, client_ip=None):
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
            run_simulation(cfg=conf, client_ip=options.client_ip)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
