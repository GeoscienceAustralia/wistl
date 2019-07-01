
import os
import logging
import pandas as pd
import numpy as np

from wistl.line import Line


def create_scenario(event, cfg):
    """ create dict of transmission line
    :param event: tuple of event_name, scale, and seed
    :param cfg: instance of config class
    :return: list of instance of Line class
    """
    scenario = Scenario(cfg=cfg, event=event)

    return [x for _, x in scenario.lines.items()]


class Scenario(object):
    """ class for a scenario"""

    def __init__(self, cfg=None, event=None, logger=None):

        self.cfg = cfg
        self.name = event[0]
        self.scale = event[1]
        self.seed = event[2]
        self.logger = logger or logging.getLogger(__name__)

        # attributes
        self._id = None
        self._path_event = None
        self._path_output = None
        self._lines = None
        self._no_lines = None

        # self.set_line_interaction()

    def __repr__(self):
        return 'Scenario(id={}, no_lines={})'.format(self.id, self.no_lines)

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'logger' in d:
            d['logger'] = d['logger'].name
        return d

    def __setstate__(self, d):
        if 'logger' in d:
            d['logger'] = logging.getLogger(d['logger'])
        self.__dict__.update(d)

    @property
    def id(self):
        if self._id is None:
            self._id = self.cfg.event_id_format.format(
                event_name=self.name, scale=self.scale)
        return self._id

    @property
    def lines(self):
        if self._lines is None:

            self._lines = {}

            for i, name in enumerate(self.cfg.lines):

                dic = self.cfg.lines[name].copy()

                dic.update({'no_sims': self.cfg.no_sims,
                            'damage_states': self.cfg.damage_states,
                            'non_collapse': self.cfg.non_collapse,
                            'event_name': self.name,
                            'scale': self.scale,
                            'event_id': self.id,
                            'rnd_state': np.random.RandomState(seed=self.seed + i),
                            'path_event': self.path_event,
                            'dic_towers': self.cfg.towers_by_line[name]})

                self._lines[name] = Line(name=name, **dic)

        return self._lines

    @property
    def no_lines(self):
        if self._no_lines is None:
            self._no_lines = len(self.lines)
        return self._no_lines

    @property
    def path_event(self):
        if self._path_event is None:
            self._path_event = os.path.join(self.cfg.path_wind_scenario_base,
                                            self.name)
        return self._path_event

    @property
    def path_output(self):
        if self._path_output is None:
            event_scale = self.cfg.event_id_format.format(
                event_name=self.name, scale=self.scale)
            self._path_output = os.path.join(self.cfg.path_output, event_scale)

            if not os.path.exists(self._path_output) and self.cfg.options['save_output']:
                os.makedirs(self._path_output)
                self.logger.info('{} is created'.format(self._path_output))

        return self._path_output


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

        for trigger_line, target_lines in cfg.line_interaction.items():

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

    return network
