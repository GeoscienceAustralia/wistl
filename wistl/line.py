#!/usr/bin/env python

import copy
import time
import dask
import itertools
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import h5py
import logging

from wistl.tower import Tower
from wistl.config import create_list_idx, unit_vector_by_bearing, angle_between_unit_vectors


class Line(object):
    """ class for a collection of towers """

    registered = ['name',
                  'coord',
                  'coord_lat_lon',
                  'dic_towers',
                  'id2name',
                  'ids',
                  'line_string',
                  'name_output',
                  'names',
                  'no_towers',
                  'no_sims',
                  'damage_states',
                  'non_collapse',
                  'scale',
                  'event_id',
                  'rnd_state',
                  'rtol',
                  'atol',
                  'dmg_threshold',
                  'line_interaction',
                  'path_event',
                  'path_output']

    def __init__(self, logger=None, **kwargs):

        self.logger = logger or logging.getLogger(__name__)

        self.name = None
        self.coord = None
        self.coord_lat_lon = None
        self.dic_towers = None
        self.id2name = None
        self.ids = None
        self.event_id = None
        self.line_string = None
        self.name_output = None
        self.names = None
        self.no_towers = None
        self.no_sims = None
        self.damage_states = None
        self.non_collapse = None
        self.path_event = None
        self.path_output = None
        self.scale = None

        for key, value in kwargs.items():
            if key in self.registered:
                setattr(self, key, value)

        self._towers = None
        self._dmg_towers = None

        # analytical method
        self.damage_prob = None

        # simulation method
        self.damage_prob_sim = None
        self.no_damage = None
        self.prob_no_damage = None

        # non cascading collapse
        self.damage_prob_sim_no_cascading = None
        self.no_damage_no_cascading = None
        self.prob_no_damage_no_cascading = None

        # line interaction
        self._dmg_idx_line = None
        self.damage_prob_sim_line = None
        self.prob_no_damage_line = None
        self.no_damage_line = None

        self._time = None
        self._time_idx = None
        self._no_time = None


        self._file_output = None

    def __repr__(self):
        return f'Line(name={self.name}, no_towers={self.no_towers}, event_id={self.event_id})'

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
    def towers(self):

        if self._towers is None:

            self._towers = {}

            for key in self.dic_towers.keys():
                dic = h_tower(line=self, idn=key)
                self._towers[dic['idl']] = Tower(**dic)

        return self._towers

    @property
    def file_output(self):
        if self._file_output is None:
            self._file_output = os.path.join(self.path_output, f'{self.event_id}_{self.name}.h5')
        return self._file_output

    @property
    def time(self):
        if self._time is None and self.dmg_towers:
            try:
                self._time = self.towers[0].wind.index[
                        self.time_idx[0]:self.time_idx[1]]
            except AttributeError:
                self.logger.error(
                        f'Can not retrieve index of {self.towers[0].name}{self.time_idx}')
        return self._time

    @property
    def time_idx(self):
        if self._time_idx is None:
            # get min of dmg_time_idx[0], max of dmg_time_idx[1]
            tmp = [self.towers[k].dmg_time_idx for k in self.dmg_towers]
            try:
                id0 = list(map(min, zip(*tmp)))[0]
            except IndexError:
                self.logger.debug(f'Line:{self.name} sustains no damage')
            else:
                id1 = list(map(max, zip(*tmp)))[1]
                self._time_idx = (id0, id1)
        return self._time_idx

    @property
    def dmg_towers(self):
        if self._dmg_towers is None:
            self._dmg_towers = [k for k, v in self.towers.items() if v.dmg_time_idx]
        return self._dmg_towers

    @property
    def no_time(self):
        if self._no_time is None and self.dmg_towers:
            self._no_time = len(self.time)
        return self._no_time

    @property
    def dmg_idx_line(self):
        """
        compute damage probability due to parallel line interaction using MC
        simulation
        :param seed:
        :return:
        """
        if self._dmg_idx_line is None:

            self._dmg_idx_line = defaultdict(list)

            for _, tower in self.towers.items():

                # determine damage state by line interaction
                # collapse_line_sim['id_sim', 'id_time', 'no_collapse']

                for id_time, grp in tower.collapse_line_sim.groupby('id_time'):

                    wind_vector = unit_vector_by_bearing(
                        tower.wind['Bearing'][id_time])

                    angle = {}
                    for line_name, value in tower.target_line.items():
                        angle[line_name] = angle_between_unit_vectors(
                            wind_vector, value['vector'])

                    target_line = min(angle, key=angle.get)

                    if angle[target_line] < 180:

                        target_tower_id = tower.target_line[target_line]['id']
                        #max_no_towers = self.cfg.no_towers_by_line[target_line]

                        for no_collapse, subgrp in grp.groupby('no_collapse'):

                            no_towers = int(no_collapse / 2)

                            right = create_list_idx(target_tower_id, no_towers, self.no_towers, 1)
                            left = create_list_idx(target_tower_id, no_towers, self.no_towers, -1)

                            id_towers = [x for x in right + [target_tower_id] + left if x >= 0]

                            list_id = list(itertools.product(id_towers, subgrp['id_sim'], [id_time]))

                            self._dmg_idx_line[target_line] += list_id

                    else:
                        msg = f'tower:{tower.name}, angle: {angle[target_line]}, wind:{wind_vector}'
                        self.logger.warning(msg)
        return self._dmg_idx_line

    def compute_damage_prob(self):
        """
        calculate damage probability of towers analytically
        Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
        where Pd: collapse due to direct wind
        Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))
        pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
        """
        self.damage_prob = {}

        if self.no_time:
            pc_adj_agg = np.zeros((self.no_towers,
                                   self.no_towers,
                                   self.no_time))

            # prob of collapse
            for k in self.dmg_towers:

                # reset time index to line
                idt0, idt1 = self.towers[k].dmg_time_idx
                idt0 -= self.time_idx[0]
                idt1 -= self.time_idx[0]

                for idl, prob in self.towers[k].collapse_adj.items():
                    pc_adj_agg[self.towers[k].idl, idl, idt0:idt1] = prob

                pc_adj_agg[self.towers[k].idl, self.towers[k].idl,
                           idt0:idt1] = self.towers[k].dmg['collapse']

            # pc_collapse.shape == (no_tower, no_time)
            pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)

            self.damage_prob['collapse'] = pd.DataFrame(
                pc_collapse.T,
                columns=self.names,
                index=self.time)

            # non_collapse state
            for ds in self.non_collapse:

                temp = np.zeros_like(pc_collapse)

                for k, tower in self.towers.items():

                    if tower.dmg_time_idx:

                        idt0, idt1 = tower.dmg_time_idx
                        idt0 -= self.time_idx[0]
                        idt1 -= self.time_idx[0]

                        # P(DS>ds) - P(collapse directly) + P(collapse induced)
                        try:
                            value = tower.dmg[ds].values \
                                - tower.dmg['collapse'].values \
                                + pc_collapse[tower.idl, idt0:idt1]
                        except ValueError:
                            self.logger.critical(f'shape: {tower.dmg[ds].values.shape} '
                                                f'shape: {tower.dmg["collapse"].values.shape} '
                                                f'shape: {pc_collapse.shape} '
                                                f'idt0, idt1: {idt0}, {idt1}')
                        else:
                            temp[tower.idl, idt0:idt1] = np.where(value > 1.0, [1.0], value)

                    else:
                        temp[tower.idl, :] = pc_collapse[tower.idl, :]

                self.damage_prob[ds] = pd.DataFrame(
                    temp.T,
                    columns=self.names,
                    index=self.time)

        return self.damage_prob

    def compute_damage_prob_sim_given_sim(self, id_sim):
        #tic = time.time()
        # perfect correlation within a single line

        results = {ds: None for ds in self.damage_states}
        # list of (time, isim, tower_id) where tower_id can be a tuple
        tf_ds = np.zeros((self.no_towers, self.no_time), dtype=bool)

        # collapse by adjacent towers
        for _, tower in self.towers.items():
            _df = tower.collapse_adj_sim.loc[tower.collapse_adj_sim['id_sim']==id_sim]
            for id_adj, grouped in _df.groupby('id_adj'):
                for idl in id_adj:
                    tf_ds[idl, grouped['id_time'].values] = True

        # append damage state by direct wind
        for ds in self.damage_states[::-1]:  # collapse first
            for _, tower in self.towers.items():
                _df = tower.dmg_state_sim[ds].loc[
                        tower.dmg_state_sim[ds]['id_sim']==id_sim]
                tf_ds[tower.idl, _df['id_time'].values] = True

            idx = np.where(tf_ds)
            results[ds] = [(id_time, id_sim, id_tower) for id_time, id_tower in zip(*idx)]

        return results

    def compute_damage_prob_sim(self):

        # perfect correlation within a single line
        self.damage_prob_sim = {}

        if self.no_time:
            #idx_by_ds = {ds: None for ds in self.damage_states}
            dic_tf_ds = {ds: None for ds in self.damage_states}
            tf_ds = np.zeros((self.no_towers,
                              self.no_sims,
                              self.no_time), dtype=bool)

            # collapse by adjacent towers
            for k in self.dmg_towers:
                for id_adj, grp in self.towers[k].collapse_adj_sim.groupby('id_adj'):
                    # reset time index relative to line
                    for idl in id_adj:
                        tf_ds[idl, grp['id_sim'], grp['id_time'] - self.time_idx[0]] = True

            # append damage state by direct wind
            for ds in self.damage_states[::-1]:  # collapse first

                for k in self.dmg_towers:
                    tf_ds[self.towers[k].idl,
                          self.towers[k].dmg_state_sim[ds]['id_sim'],
                          self.towers[k].dmg_state_sim[ds]['id_time'] - self.time_idx[0]] = True

                # PE(DS)
                self.damage_prob_sim[ds] = pd.DataFrame(tf_ds.sum(axis=1).T / self.no_sims,
                    columns=self.names, index=self.time)

                dic_tf_ds[ds] = tf_ds.copy()
                #idx = np.where(tf_ds)
                #idx_by_ds[ds] = [(id_time, id_sim, id_tower) for
                #                 id_tower, id_sim, id_time in zip(*idx)]

            # compute mean, std of no. of damaged towers
            self.no_damage, self.prob_no_damage = self.compute_stats(dic_tf_ds)

            # checking against analytical value
            for name in self.names:
                try:
                    np.testing.assert_allclose(self.damage_prob['collapse'][name].values,
                        self.damage_prob_sim['collapse'][name].values, atol=self.atol, rtol=self.rtol)
                except AssertionError:
                    self.logger.warning(f'Simulation results of {name}:collapse are not close to the analytical')
        return self.damage_prob_sim

    def compute_damage_prob_sim_no_cascading(self):

        self.damage_prob_sim_no_cascading = {}

        if self.no_time:
            dic_tf_ds = {ds: None for ds in self.damage_states}

            tf_ds = np.zeros((self.no_towers,
                              self.no_sims,
                              self.no_time), dtype=bool)

            for ds in self.damage_states[::-1]:  # collapse first

                for k in self.dmg_towers:

                    tf_ds[self.towers[k].idl,
                          self.towers[k].dmg_state_sim[ds]['id_sim'],
                          self.towers[k].dmg_state_sim[ds]['id_time'] - self.time_idx[0]] = True

                # PE(DS)
                self.damage_prob_sim_no_cascading[ds] = pd.DataFrame(
                    tf_ds.sum(axis=1).T / self.no_sims,
                    columns=self.names,
                    index=self.time)

                dic_tf_ds[ds] = tf_ds.copy()

            self.no_damage_no_cascading, self.prob_no_damage_no_cascading = \
                self.compute_stats(dic_tf_ds)

            # checking against analytical value
            for k in self.dmg_towers:
                idt0, idt1 = self.towers[k].dmg_time_idx
                name = self.towers[k].name
                idt0_ = idt0 - self.time_idx[0]
                idt1_ = idt1 - self.time_idx[0]
                try:
                    np.testing.assert_allclose(self.towers[k].dmg[ds].values,
                            self.damage_prob_sim_no_cascading[ds].iloc[idt0_:idt1_][name].values, atol=self.atol, rtol=self.rtol)
                except AssertionError:
                    self.logger.warning(f'Simulation results of {name}:{ds} are not close to the analytical')
        return self.damage_prob_sim_no_cascading

    def compute_stats(self, dic_tf_ds):
        """
        compute mean and std of no. of ds
        tf_collapse_sim.shape = (no_towers, no_sim, no_time)
        :param tf_sim:
        :return:
        """
        #tic = time.time()
        no_damage = {}
        prob_no_damage = {}
        columns = [str(x) for x in range(self.no_towers + 1)]

        # (no_towers, 1)
        x_tower = np.array(range(self.no_towers + 1))[:, np.newaxis]
        x2_tower = x_tower ** 2.0

        tf_ds = np.zeros((self.no_towers,
                          self.no_sims,
                          self.no_time), dtype=bool)

        # from collapse and minor
        for ds in self.damage_states[::-1]:

            tf_ds = np.logical_xor(dic_tf_ds[ds], tf_ds)

            # mean and standard deviation
            # no_ds_across_towers.shape == (no_sims, no_time)
            no_ds_across_towers = tf_ds.sum(axis=0)
            prob = np.zeros(shape=(self.no_time, self.no_towers + 1))

            for i in range(self.no_time):
                value, freq = np.unique(no_ds_across_towers[:, i], return_counts=True)  # (value, freq)
                prob[i, [int(x) for x in value]] = freq

            prob /= self.no_sims  # (no_time, no_towers)

            _exp = np.dot(prob, x_tower)
            _std = np.sqrt(np.dot(prob, x2_tower) - _exp ** 2)

            no_damage[ds] = pd.DataFrame(np.hstack((_exp, _std)),
                columns=['mean', 'std'], index=self.time)

            prob_no_damage[ds] = pd.DataFrame(prob, columns=columns, index=self.time)

        #print(f'stat: {time.time() - tic}')
        return no_damage, prob_no_damage

    def compute_stats1(self, idx_by_ds):
        """
        compute mean and std of no. of ds
        idx_by_ds = dict of list of tuples(id_time, id_sim, id_tower)
        :return:
        """

        tic = time.time()
        exp_no_tower = {}
        prob_no_tower = {}
        columns = [str(x) for x in range(self.no_towers + 1)]

        # (no_towers, 1)
        x_tower = np.array(range(self.no_towers + 1))[:, np.newaxis]
        x2_tower = x_tower ** 2.0

        # from collapse and minor
        prev = set()
        for ds in self.damage_states[::-1]:

            prob = np.zeros((self.no_time, self.no_towers + 1))
            # mean and standard deviation
            # no_ds_across_towers.shape == (no_sims, no_time)
            tmp = list(set(idx_by_ds[ds]).difference(prev))
            df = pd.DataFrame(tmp, columns=['id_time','id_sim','id_tower'])
            prev = set(idx_by_ds[ds])
            for id_time, grp in df.groupby('id_time'):
                freq = grp.groupby('id_sim').agg({'id_tower': len})['id_tower'].value_counts()
                freq[0] = self.no_sims - freq.sum()
                prob[id_time][freq.index] = freq / self.no_sims

            _mean = np.dot(prob, x_tower)
            _std = np.sqrt(np.dot(prob, x2_tower) - _mean ** 2)

            exp_no_tower[ds] = pd.DataFrame(np.hstack((_mean, _std)),
                columns=['mean', 'std'], index=self.time)
            prob_no_tower[ds] = pd.DataFrame(prob, columns=columns, index=self.time)
        print(f'stat1: {time.time() - tic}')
        return exp_no_tower, prob_no_tower


    def write_hdf5(self):

        items = ['damage_prob', 'damage_prob_sim', 'damage_prob_sim_no_cascading',
                 'no_damage', 'no_damage_no_cascading',
                 'prob_no_damage', 'prob_no_damage_no_cascading']

        columns_by_item = {'damage_prob': self.names,
                           'damage_prob_sim': self.names,
                           'damage_prob_sim_no_cascading': self.names,
                           'no_damage': ['mean', 'std'],
                           'no_damage_no_cascading': ['mean', 'std'],
                           'prob_no_damage': range(self.no_towers + 1),
                           'prob_no_damage_no_cascading': range(self.no_towers + 1)
                           }

        if self.no_time:

            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                self.logger.info(f'{self.path_output} is created')

            with h5py.File(self.file_output, 'w') as hf:

                for item in items:

                    group = hf.create_group(item)

                    for ds in self.damage_states:

                        value = getattr(self, item)[ds]
                        data = group.create_dataset(ds, data=value)

                        # metadata
                        data.attrs['nrow'], data.attrs['ncol'] = value.shape
                        data.attrs['time_start'] = str(self.time[0])
                        try:
                            data.attrs['time_freq'] = str(self.time[1]-self.time[0])
                        except IndexError:
                            data.attrs['time_freq'] = str(self.time[0])
                        data.attrs['time_period'] = self.time.shape[0]

                        if columns_by_item[item]:
                            data.attrs['columns'] = ','.join('{}'.format(x) for x in columns_by_item[item])
            self.logger.info(f'{self.file_output} is saved')
        else:
            self.logger.info(f'No output file for Line:{self.name}')

def compute_damage_per_line(line, cfg):
    """
    mc simulation over transmission line
    :param line: instance of transmission line
           cfg: instance of config
    :return: None but update attributes of
    """

    logger = logging.getLogger(__name__)

    logger.info(f'computing damage of {line.name} for {line.event_id}')

    # compute damage probability analytically
    line.compute_damage_prob()

    # perfect correlation within a single line
    line.compute_damage_prob_sim()

    if not cfg.options['skip_no_cascading_collapse']:
        line.compute_damage_prob_sim_no_cascading()

    # compare simulation against analytical
    #for ds in cfg.damage_states:
    #    idx_not_close = np.where(~np.isclose(line.damage_prob_sim[ds].values,
    #                             line.damage_prob[ds].values,
    #                             atol=ATOL,
    #                             rtol=RTOL))
    #    for idc in idx_not_close[1]:
    #        logger.warning(f'Simulation not CLOSE {ds}:{line.towers[idc].name}')

    # save
    if cfg.options['save_output']:
        line.write_hdf5()

    return line

def h_tower(line, idn):

    dic = line.dic_towers[idn].copy()
    dic.update({'idn': idn,
                'no_sims': line.no_sims,
                'damage_states': line.damage_states,
                'scale': line.scale,
                'rtol': line.rtol,
                'atol': line.atol,
                'dmg_threshold': line.dmg_threshold,
                'rnd_state': line.rnd_state,
                'path_event': line.path_event})

    return dic

