"""
WISTL: Wind Impact Simulation on Transmission Lines
"""


import os
import sys
import time
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from dask.distributed import Client

from wistl.config import Config, set_towers_and_lines
from wistl.version import VERSION_DESC
from wistl.line import Line, compute_dmg_by_line
from wistl.tower import Tower, compute_dmg_by_tower



def run_simulation(cfg, client_ip=None):
    """
    main function
    :cfg: an instance of Config
    :
    """

    tic = time.time()

    towers, lines = create_towers_and_lines(cfg)

    if cfg.options['run_parallel']:

        from distributed.worker import logger

        logger.info('parallel MC run on....')
        client = Client(client_ip)

        out = []
        for event in cfg.events:
            results = []
            # compute damage by tower and event
            for _, tower in towers.items():

                result = client.submit(compute_dmg_by_tower, tower, event, lines[tower.lineroute], cfg)
                results.append(result)

            # compute damage by line and event
            for _, line in lines.items():

                result = client.submit(compute_dmg_by_line, line, results, event, cfg)
                out.append(result)

        _ = client.gather(out)

        client.close()

        logger.info(f'Elapsed time: {time.time()-tic:.3f}')
    else:

        #import logging

        #logger = logging.getLogger(__name__)
        print('serial MC run on....')

        for event in cfg.events:
            results = [compute_dmg_by_tower(tower, event, lines[tower.lineroute], cfg)
                       for _, tower in towers.items()]

        # compute damage by line and event
            for _, line in lines.items():

                _ = compute_dmg_by_line(line, results, event, cfg)

        print(f'Elapsed time: {time.time()-tic:.3f}')


def create_towers_and_lines(cfg):

    dic_towers, dic_lines = set_towers_and_lines(cfg)

    # using default dict or named tuple
    towers, lines = {}, {}
    for line_name, items in dic_towers.items():
        lines[line_name] = Line(**dic_lines[line_name])
        for tower_name, item in items.items():
            towers[tower_name] = Tower(**item)

    return towers, lines


def compute_dmg_probability_line_interaction(self):
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
        for ds in self.cfg.dmg_states[::-1]:

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


def process_commandline():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config",
                        dest="config_file",
                        help="read configuration from FILE",
                        metavar="FILE")
    parser.add_argument("-i", "--ip",
                        dest="client_ip",
                        help="set client ip address for dask cluster",
                        metavar="ip_address")
    return parser


def main():

    parser = process_commandline()

    args = parser.parse_args()

    if args.config_file:
        if not os.path.isfile(args.config_file):
            sys.exit(f'{args.config_file} not found')
        else:
            path_cfg = os.path.dirname(os.path.realpath(args.config_file))
            conf = Config(file_cfg=args.config_file)
            run_simulation(cfg=conf, client_ip=args.client_ip)
            # move output.log to output directory
            old = os.path.join(os.getcwd(), 'output.log')
            new = os.path.join(path_cfg, 'output', 'output.log')
            Path(old).rename(new)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
