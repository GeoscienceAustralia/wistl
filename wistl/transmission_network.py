from __future__ import print_function, division

import os
import logging
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from wistl.transmission_line import TransmissionLine


def create_transmission_network_under_wind_event(event_id, cfg):
    """ create dict of transmission network
    :param event_id: tuple of event_name and scale
    :param cfg: instance of config class
    :return: an instance of TransmissionNetwork
    """
    event_name, scale = event_id
    network = TransmissionNetwork(cfg=cfg, event_name=event_name, scale=scale)

    return network


class TransmissionNetwork(object):
    """ class for a collection of transmission lines"""

    def __init__(self, cfg=None, event_name=None, scale=None, logger=None):

        self.cfg = cfg
        self.event_name = event_name
        self.scale = scale
        self.logger = logger or logging.getLogger(__name__)

        # attributes
        self._event_id = None
        self._path_event = None
        self._path_output = None
        self._lines = None

        # self.set_line_interaction()

            # if cfg.figure:
            #     self.plot_line_interaction()

    @property
    def lines(self):
        if self._lines is None:

            self._lines = {}

            for line_name, towers in self.cfg.towers.groupby('LineRoute'):

                dic_line = self.cfg.lines[line_name].copy()

                dic_line.update({'no_sims': self.cfg.no_sims,
                                 'damage_states': self.cfg.damage_states,
                                 'event_id': self.event_id,
                                 'path_event': self.path_event})

                line = TransmissionLine(name=line_name, **dic_line)
                line.towers = towers.to_dict('index')

                self._lines[line_name] = line

        return self._lines

    @property
    def path_event(self):
        if self._path_event is None:
            self._path_event = os.path.join(self.cfg.path_wind_scenario_base,
                                            self.event_name)
        return self._path_event

    @property
    def event_id(self):
        if self._event_id is None:
            self._event_id = self.cfg.event_id_format.format(
                event_name=self.event_name, scale=self.scale)
        return self._event_id

    @property
    def path_output(self):
        if self._path_output is None:
            self._path_output = os.path.join(self.cfg.path_output, self.event_id)

            if not os.path.exists(self._path_output) and self.cfg.options['save_output']:
                os.makedirs(self._path_output)
                self.logger.info('{} is created'.format(self._path_output))

        return self._path_output

    # def plot_line_interaction(self):
    #
    #     for line_name, line in self.lines.items():
    #
    #         plt.figure()
    #         plt.plot(line.coord[:, 0],
    #                  line.coord[:, 1], '-', label=line_name)
    #
    #         for target_line in self.cfg.line_interaction[line_name]:
    #
    #             plt.plot(self.lines[target_line].coord[:, 0],
    #                      self.lines[target_line].coord[:, 1],
    #                      '--', label=target_line)
    #
    #             for tower in line.towers.itervalues():
    #                 try:
    #                     id_pt = tower.id_on_target_line[target_line]['id']
    #                 except KeyError:
    #                     plt.plot(tower.coord[0], tower.coord[1], 'ko')
    #                 else:
    #                     target_tower_name = self.lines[
    #                         target_line].name_by_line[id_pt]
    #                     target_tower = self.lines[target_line].towers[
    #                         target_tower_name]
    #
    #                     plt.plot([tower.coord[0], target_tower.coord[0]],
    #                              [tower.coord[1], target_tower.coord[1]],
    #                              'ro-',
    #                              label='{}->{}'.format(tower.name,
    #                                                    target_tower_name))
    #
    #         plt.title(line_name)
    #         plt.legend(loc=0)
    #         plt.xlabel('Longitude')
    #         plt.ylabel('Latitude')
    #         png_file = os.path.join(self.cfg.path_output,
    #                                 'line_interaction_{}.png'.format(line_name))
    #
    #         if not os.path.exists(self.cfg.path_output):
    #             os.makedirs(self.cfg.path_output)
    #
    #         plt.savefig(png_file)
    #         print('{} is created'.format(png_file))
    #         plt.close()


def compute_damage_probability_line_interaction_per_network(network, cfg):
    """
    compute damage probability due to line interaction
    :param network: a dictionary of lines
    :return: network: a dictionary of lines
    """

    for line_name, line in network.items():

        tf_ds = np.zeros((line['no_towers'],
                          cfg.no_sims,
                          len(line.time_index)), dtype=bool)

        tf_ds_itself = np.zeros((line.no_towers,
                                 cfg.no_sims,
                                 len(line.time_index)), dtype=bool)

        for trigger_line, target_lines in cfg.line_interaction.iteritems():

            if line_name in target_lines:

                try:
                    pd_id = np.vstack(
                        network[trigger_line].damage_index_line_interaction[
                            line_name])
                except ValueError:
                    print('no interaction applied: from {} to {}'.format(
                        trigger_line, line_name))
                else:
                    print('interaction applied: from {} to {}'.format(
                        trigger_line, line_name))
                    # id_tower = pd_id['id_tower'].values
                    # id_sim = pd_id['id_sim'].values
                    # id_time = pd_id['id_time'].values

                    # try:
                    tf_ds[pd_id[:, 0], pd_id[:, 1], pd_id[:, 2]] = True
                    # except IndexError:
                    #     print('{}:{}:{}'.format(pd_id.head(),
                    #                             pd_id.dtypes,
                    #                             'why???'))
                    #     print('trigger:{}, {}, {}'.format(trigger_line,
                    #                                       line_name,
                    #                                       line.event_id_scale))

        # append damage state by line itself
        # due to either direct wind and adjacent towers
        # also need to override non-collapse damage states

        cds_list = cfg.damage_states[:]  # to avoid effect
        cds_list.reverse()  # [collapse, minor]

        tf_sim = dict()

        # append damage state by either direct wind or adjacent towers
        for ds in cds_list:

            tf_ds[line.damage_index_simulation[ds]['id_tower'],
                  line.damage_index_simulation[ds]['id_sim'],
                  line.damage_index_simulation[ds]['id_time']] = True

            tf_sim[ds] = np.copy(tf_ds)

            line.damage_prob_line_interaction[ds] = \
                pd.DataFrame(np.sum(tf_ds, axis=1).T / float(cfg.no_sims),
                             columns=line.name_by_line,
                             index=line.time_index)

        # check whether collapse induced by line interaction
        tf_ds_itself[line.damage_index_simulation['collapse']['id_tower'],
                     line.damage_index_simulation['collapse']['id_sim'],
                     line.damage_index_simulation['collapse']['id_time']] = True

        collapse_by_interaction = np.logical_xor(tf_sim['collapse'],
                                                 tf_ds_itself)

        if np.any(collapse_by_interaction):
            print('{} is affected by line interaction'.format(line_name))

        # compute mean and std of no. of towers for each of damage states
        (line.est_no_damage_line_interaction,
            line.prob_no_damage_line_interaction) = \
            line.compute_damage_stats(tf_sim)

        if cfg.options['save_output']:
            line.write_hdf5(file_str='damage_prob_line_interaction',
                            value=line.damage_prob_line_interaction)

            line.write_hdf5(file_str='est_no_damage_line_interaction',
                            value=line.est_no_damage_line_interaction)

            line.write_hdf5(file_str='prob_no_damage_line_interaction',
                            value=line.prob_no_damage_line_interaction)

    return network
