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
        self._dmg_lines = None
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
                dic = h_line(scenario=self, name=name, seed=self.seed + i)
                self._lines[name] = Line(**dic)

        return self._lines

    @property
    def dmg_lines(self):
        if self._dmg_lines is None:
            self._dmg_lines = [k for k,v in self.lines.items() if v.no_time]
        return self._dmg_lines

    @property
    def time_idx(self):
        if self._time_idx is None:
            tmp = [self.lines[k].time_idx for k in self.dmg_lines]
            try:
                id0 = list(map(min, zip(*tmp)))[0]
            except IndexError:
                self.logger.debug(f'Scenario:{self.id} sustains no damage')
            else:
                id1 = list(map(max, zip(*tmp)))[1]
                self._time_idx = (id0, id1)
        return self._time_idx

    @property
    def time(self):
        if self._time is None and self.dmg_lines:
            try:
                key = [*self.lines][0]
                self._time = self.lines[key].towers[0].wind.index[
                        self.time_idx[0]:self.time_idx[1]]
            except AttributeError:
                self.logger.error('Can not set time')
        return self._time

    @property
    def no_time(self):
        if self._no_time is None and self.dmg_lines:
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


    def compute_damage_probability_line_interaction(self):
        """
        compute damage probability due to line interaction
        :param lines: a dictionary of lines
        :return: lines: a dictionary of lines
        """
        for line_name, line in self.lines.items():

            tf_ds = np.zeros((line.no_towers,
                              line.no_sims,
                              self.no_time), dtype=bool)

            tf_ds_itself = np.zeros((line.no_towers,
                                     line.no_sims,
                                     self.no_time), dtype=bool)

            for trigger, target in cfg.line_interaction.items():

                if line_name in target:

                    try:
                        id_tower, id_sim, id_time = zip(*lines[trigger].dmg_idx_line[line_name])
                        #pd_id = np.vstack(
                        #    lines[trigger].dmg_idx_line[line_name])
                    except ValueError:
                        print(f'no interaction applied: from {trigger_line} to {line_name}')
                    else:
                        print(f'interaction applied: from {trigger_line} to {line_name}')
                        tf_ds[id_tower, id_sim, id_time] = True

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

def h_line(scenario, name, seed):

    dic = scenario.cfg.lines[name].copy()
    dic.update({'name': name,
                'no_sims': scenario.cfg.no_sims,
                'damage_states': scenario.cfg.damage_states,
                'non_collapse': scenario.cfg.non_collapse,
                'event_id': scenario.id,
                'line_interaction': scenario.cfg.line_interaction.get(name),
                'scale': scenario.scale,
                'rnd_state': np.random.RandomState(seed=seed),
                'rtol': scenario.cfg.rtol,
                'atol': scenario.cfg.atol,
                'dmg_threshold': scenario.cfg.dmg_threshold,
                'path_event': scenario.path_event,
                'path_output': scenario.path_output,
                'dic_towers': scenario.cfg.towers_by_line[name]})

    return dic

