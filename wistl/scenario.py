import os
import logging
import pandas as pd
import numpy as np
import dask.array as da

from wistl.line import Line


class Scenario(object):
    """ class for a scenario"""

    def __init__(self, cfg=None, event=None, logger=None):

        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        try:
            assert len(event) == 3
        except AssertionError:
            self.logger.critical('event should be (name, scale, seed)')
        else:
            self.name = event[0]
            self.scale = event[1]
            self.seed = event[2]

        # attributes
        self._id = None
        self._path_event = None
        self._path_output = None
        self._list_lines = None
        self._lines = None
        self._no_lines = None
        self._time_idx = None
        self._time = None
        self._no_time = None
        # self.set_line_interaction()

    def __repr__(self):
        return f'Scenario(id={self.id}, no_lines={self.no_lines})'

    #def __getstate__(self):
    #    d = self.__dict__.copy()
    #    if 'logger' in d:
    #        d['logger'] = d['logger'].name
    #    return d

    #def __setstate__(self, d):
    #    if 'logger' in d:
    #        d['logger'] = logging.getLogger(d['logger'])
    #    self.__dict__.update(d)

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
                dic.update({'name': name,
                            'no_sims': self.cfg.no_sims,
                            'damage_states': self.cfg.damage_states,
                            'non_collapse': self.cfg.non_collapse,
                            'event_id': self.id,
                            'scale': self.scale,
                            'rnd_state': np.random.RandomState(seed=self.seed + i),
                            'rtol': self.cfg.rtol,
                            'atol': self.cfg.atol,
                            'pm_threshold': self.cfg.pm_threshold,
                            'path_event': self.path_event,
                            'dic_towers': self.cfg.towers_by_line[name]})

                self._lines[name] = Line(**dic)

        return self._lines

    # TODO: why list_lines need?
    #@property
    #def list_lines(self):
    #    if self._list_lines is None:
    #        self._list_lines = [x for _, x in self.lines.items()]
    #    return self._list_lines

    @property
    def time_idx(self):
        if self._time_idx is None:
            tmp = []
            for _, value in self.lines.items():
                tmp.append(value.time_idx)
            id0 = list(map(min, zip(*tmp)))[0]
            id1 = list(map(max, zip(*tmp)))[1]
            self._time_idx = (id0, id1)
        return self._time_idx

    @property
    def time(self):
        if self._time is None:
            try:
                key = [*self.lines][0]
                self._time = self.lines[key].towers[0].wind.index[
                        self.time_idx[0]:self.time_idx[1]]
            except AttributeError:
                self.logger.error('Can not set time')
        return self._time

    @property
    def no_time(self):
        if self._no_time is None:
            self._no_time = len(self.time)
        return self._no_time

    @property
    def no_lines(self):
        if self._no_lines is None:
            self._no_lines = len(self.lines)
        return self._no_lines

    @property
    def path_event(self):
        if self._path_event is None:
            self._path_event = os.path.join(self.cfg.path_wind_event_base, self.name)
        return self._path_event

    @property
    def path_output(self):
        if self._path_output is None:
            self._path_output = os.path.join(self.cfg.path_output, self.id)

            if not os.path.exists(self._path_output) and self.cfg.options['save_output']:
                os.makedirs(self._path_output)
                self.logger.info(f'{self._path_output} is created')

        return self._path_output


def compute_damage_probability_line_interaction(lines, cfg):
    """
    compute damage probability due to line interaction
    :param lines: a dictionary of lines
    :return: lines: a dictionary of lines
    """

    for line_name, line in lines.items():

        tf_ds = np.zeros((line.no_towers,
                          line.no_sims,
                          line.time_index), dtype=bool)

        tf_ds_itself = np.zeros((line.no_towers,
                                 line.no_sims,
                                 line.time_index), dtype=bool)

        for trigger, target in cfg.line_interaction.items():

            if line_name in target:

                try:
                    pd_id = np.vstack(
                        lines[trigger].damage_index_line_interaction[
                            line_name])
                except ValueError:
                    print(f'no interaction applied: from {trigger_line} to {line_name}')
                else:
                    print(f'interaction applied: from {trigger_line} to {line_name}')
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
            print(f'{line_name} is affected by line interaction')

        # compute mean and std of no. of towers for each of damage states
        (line.est_no_damage_line_interaction,
            line.prob_no_damage_line_interaction) = \
            line.compute_damage_stats(tf_sim)

    return lines
