#!/usr/bin/env python

import numpy as np
import pandas as pd
import time
import dask
import dask.dataframe as dd
import multiprocessing
import os
import logging
import scipy.stats as stats
import bisect

from wistl.config import unit_vector_by_bearing, angle_between_unit_vectors

class Tower(object):
    """
    class Tower
    Tower class represent an individual tower.
    """

    registered = ['axisaz',   # azimuth of strong axis
                 #'constcost',
                  #'consttype',
                  # 'devangle',  # deviation angle
                  'function',
                  #'ht_source',
                  # 'height',
                  # 'latitude',
                  'lineroute',
                  #'locsource',
                  'longitude',
                  #'mun',
                  #'number',
                  'name',
                  #'shapes',
                  'type',
                  #'yrbuilt',
                  'actual_span',
                  'collapse_capacity',
                  # 'cond_pc',
                  'cond_pc_adj',
                  'cond_pc_adj_sim_idx',
                  'cond_pc_adj_sim_prob',
                  'cond_pc_interaction_no',
                  'cond_pc_interaction_prob',
                  'cond_pc_interaction_cprob',
                  'coord',
                  'coord_lat_lon',
                  'design_level',
                  'design_span',
                  'design_speed',
                  'ratio_z_to_10',
                  'file_wind_base_name',
                  'no_sims',
                  'damage_states',
                  'rnd_state',
                  'scale',
                  'frag_dic',
                  'id_adj',
                  'idl',
                  'idn',
                  'max_no_adj_towers',
                  'height_z',
                  'atol',
                  'rtol',
                  'dmg_threshold',
                  'target_line',
                  #'point',
                  'terrain_cat',
                  'path_event']

    def __init__(self, logger=None, **kwargs):
        """
        :param cfg: instance of config clas
        :param ps_tower: panda series containing tower details
        """

        self.logger = logger or logging.getLogger(__name__)

        self.no_sims = None  # o
        self.rnd_state = None  # o

        # self.DevAngle = None
        # self.Function = None  # Suspension, Terminal, Strainer
        self.height = None
        self.latitude = None
        self.lineroute = None
        self.longitude = None
        self.name = None
        self.type = None
        self.function = None
        self.actual_span = None
        # self.cond_pc = None
        self.cond_pc_adj_sim_idx = None
        self.cond_pc_adj_sim_prob = None
        self.coord = None
        self.coord_lat_lon = None
        self.design_level = None  # design level
        self.design_span = None  # design wind span
        self.design_speed = None
        self.damage_states = None

        self.axisaz = None  # azimuth of strong axis relative to North (deg)
        self.ratio_z_to_10 = None
        self.cond_pc_adj = None  # dict
        self.collapse_capacity = None
        self.file_wind_base_name = None
        self.frag_dic = None
        self.path_event = None
        self.idl = None  # tower id within line (starting from 0 for each line)
        self.idn = None  # tower id within network 
        self.id_adj = None
        self.max_no_adj_towers = None

        self.height_z = None
        # self.point = None
        self.terrain_cat = None  # Terrain Category
        self.scale = None

        # line interaction
        self.cond_pc_interaction_no = None
        self.cond_pc_interaction_prob = None
        self.cond_pc_interaction_cprob = None
        self.target_line = None

        for key, value in kwargs.items():
            if key in self.registered:
                setattr(self, key, value)

        # self._damage_states = None
        # print('{}'.format(self.id_adj))
        # computed by functions in transmission_line class
        # self.id_sides = None  # (left, right) ~ assign_id_both_sides
        # self.id_adj = None  # (23,24,0,25,26) ~ update_id_adj_towers

        # initialised and later populated
        self._file_wind = None
        self._wind = None
        self._no_time = None
        self._sorted_frag_dic_keys = None

        # analytical method
        # dmg, dmg_time_idx, collapse_adj
        self._dmg = None
        self._dmg1 = None
        self._dmg_time_idx = None
        self._collapse_adj = None

        # simulation method: determine_damage_isolation_sim,
        # dmg_state_sim -> dmg_sim (just for validation against dmg)
        # dmg_state_sim -> collapse_adj_sim
        self._dmg_state_sim = None
        self._dmg_sim = None
        self._collapse_adj_sim = None
        self._collapse_interaction = None

        # line interaction
        # self.cond_pc_line_sim = {key: None for key in ['no_collapse',
        #                                               'cum_prob']}
        # self._sim_parallel_line = None

    # @property
    # def cond_pc_line(self):
    #     return self._cond_pc_line

    # @property
    # def damage_states(self):
    #     if self._damage_states is None:
    #         self._damage_states = self.frag_arg.keys()
    #     return self._damage_states

    def __repr__(self):
        return f'Tower(name={self.name}, function={self.function}, idl={self.idl}, idn={self.idn})'

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
    def sorted_frag_dic_keys(self):
        if self._sorted_frag_dic_keys is None:
            self._sorted_frag_dic_keys = sorted(self.frag_dic.keys())
        return self._sorted_frag_dic_keys

    @property
    def no_time(self):
        if self._no_time is None and not self.dmg.empty:
            self._no_time = len(self.dmg.index)
        return self._no_time

    @property
    def dmg_time_idx(self):
        """
        return starting and edning index of dmg relative to wind time index
        """
        if self._dmg_time_idx is None and not self.dmg.empty:
            idt = self.wind.index.intersection(self.dmg.index)
            idt0 = self.wind.index.get_loc(idt[0])
            idt1 = self.wind.index.get_loc(idt[-1]) + 1
            self._dmg_time_idx = (idt0, idt1)
        return self._dmg_time_idx

    @property
    def file_wind(self):
        if self._file_wind is None:
            try:
                assert os.path.exists(self.path_event)
            except AssertionError:
                self.logger.error(f'Invalid path_event {self.path_event}')
            else:
                self._file_wind = os.path.join(
                    self.path_event, self.file_wind_base_name)
                try:
                    assert os.path.exists(self._file_wind)
                except AssertionError:
                    self.logger.error(
                        f'Invalid file_wind {self._file_wind}')

        return self._file_wind

    # @property
    # def id_on_target_line(self):
    #     return self._id_on_target_line
    #
    # @property
    # def mc_parallel_line(self):
    #     return self._sim_parallel_line
    #
    # @id_on_target_line.setter
    # def id_on_target_line(self, id_on_target_line):
    #     self._id_on_target_line = id_on_target_line
    #     self.get_cond_prob_line_interaction()

    @property
    def wind(self):
        """
        set the wind given a file_wind
        """
        if self._wind is None:

            try:
                tmp = pd.read_csv(self.file_wind,
                                  parse_dates=[0],
                                  index_col=['Time'],
                                  usecols=['Time', 'Speed', 'Bearing'])
            except IOError:
                msg = f'Invalid file_wind {self.file_wind}'
                self.logger.critical(msg)
            else:
                self._wind = tmp.loc[tmp.isnull().sum(axis=1) == 0].copy()
                self._wind['Speed'] *= self.scale * self.ratio_z_to_10
                self._wind['ratio'] = self._wind['Speed'] / self.collapse_capacity

        return self._wind

    @property
    def dmg(self):
        """
        compute probability exceedance of damage of tower in isolation (Pc)
        Note: dmg index is not identical to wind index
        """
        if self._dmg is None:

            df = self.wind.apply(self.compute_damage_using_directional_vulnerability, axis=1)

            # apply thresholds
            valid = np.where(df['minor'] > self.dmg_threshold)[0]
            try:
                idt0 = min(valid)

            except ValueError:
                self.logger.info(f'{self.name} sustains no damage')
                self._dmg = pd.DataFrame()
            else:
                idt1 = max(valid) + 1
                self._dmg = df.iloc[idt0:idt1]
                self._dmg.index = self.wind.index[idt0:idt1]

        return self._dmg

    @property
    def dmg1(self):
        """
        compute probability exceedance of damage of tower in isolation (Pc)
        Note: dmg index is not identical to wind index
        """
        if self._dmg1 is None:

            wind = dd.from_pandas(self.wind, npartitions=4*multiprocessing.cpu_count())
            #df = self.wind1.apply(self.compute_damage_using_directional_vulnerability, axis=1)
            df = wind.map_partitions(lambda x: x.apply(self.compute_damage_using_directional_vulnerability, axis=1)).compute(scheduler='processes')
            # apply thresholds
            valid = np.where(df['minor'] > self.dmg_threshold)[0]

            try:
                idt0 = min(valid)

            except ValueError:
                self.logger.debug(f'tower:{self.name} sustains no damage1')

            else:
                idt1 = max(valid) + 1
                self._dmg1 = df.iloc[idt0:idt1]
                self._dmg1.index = self.wind.index[idt0:idt1]

        return self._dmg1

    @property
    def dmg_state_sim(self):
        """
        determine damage state of tower in isolation by simulation
        # PD not PE = 0(non), 1, 2 (collapse)
        TODO: changed to for loop by no_sims and using delayed?
        """

        if self._dmg_state_sim is None and not self.dmg.empty:

            self._dmg_state_sim = {}

            # 1. determine damage state of tower due to wind
            rv = self.rnd_state.uniform(size=(self.no_sims, self.no_time))

            # ds_wind.shape == (no_sims, no_time)
            # PD not PE = 0(non), 1, 2 (collapse)
            # dmg_state_sim.shape == (no_sims, no_time)
            _array = (rv[:, :, np.newaxis] < self.dmg.values).sum(axis=2)

            for ids, ds in enumerate(self.damage_states, 1):

                # Note that id_time always starts from 0, irrespective of dmg_time_idx 
                id_sim, id_time = np.where(_array == ids)

                # convert to wind time index for aggregation at line level
                #id_time += self.dmg_time_idx[0]

                self._dmg_state_sim[ds] = pd.DataFrame(
                    np.vstack((id_sim, id_time)).T, columns=['id_sim', 'id_time'])

        return self._dmg_state_sim

    #def compute_dmg_state_sim(self):
    #    """
    #    determine if adjacent tower collapses or not due to pull by the tower
    #    j_time: time index (array)
    #    idx: multiprocessing thread id

    #    # PD not PE = 0(non), 1, 2 (collapse)
    #    TODO: changed to for loop by no_sims and using delayed?
    #    """
    #    dmg_state_sim = {}
    #    # 1. determine damage state of tower due to wind
    #    rv = self.rnd_state.uniform(size=(self.no_time))

    #    # ds_wind.shape == (no_sims, no_time)
    #    # PD not PE = 0(non), 1, 2 (collapse)
    #    # dmg_state_sim.shape == (no_time)
    #    _array = (rv[:, np.newaxis] < self.dmg.values).sum(axis=1)

    #    for ids, ds in enumerate(self.damage_states, 1):

    #        id_time = np.where(_array == ids)[0]

    #        # convert to wind time index for aggregation at line level
    #        id_time += self.dmg_time_idx[0]

    #        dmg_state_sim[ds] = id_time

    #    return dmg_state_sim

    @property
    def dmg_sim(self):
        """
        calls self.dmg_state_sim and compare against dmg
        PE not PD 1, 2 (collapse)
        """

        if self._dmg_sim is None and not self.dmg.empty:

            self._dmg_sim = {}

            pb = np.zeros(self.dmg['collapse'].shape)

            for ds in self.damage_states[::-1]:  # collapse first

                _array = self.dmg_state_sim[ds].groupby('id_time').agg(len)['id_sim'].values
                self._dmg_sim[ds] = pb + _array / self.no_sims
                pb = _array / self.no_sims

                # check whether MC simulation is close to analytical
                idx_not_close, = np.where(~np.isclose(self._dmg_sim[ds],
                                                      self.dmg[ds],
                                                      atol=self.atol,
                                                      rtol=self.rtol))
                if len(idx_not_close):
                    idx = idx_not_close[0]
                    self.logger.warning(f'PE of {ds}: {self._dmg_sim[ds][idx]:.3f} vs {self.dmg[ds].iloc[idx]:.3f}')
        return self._dmg_sim

    @property
    def collapse_adj(self):
        """
        used only for analytical approach
        calculate collapse probability of jth tower due to pull by the tower
        Pc(j,i) = P(j|i)*Pc(i)
        """
        # only applicable for tower collapse
        if self._collapse_adj is None and not self.dmg.empty:

            self._collapse_adj = {}

            for key, value in self.cond_pc_adj.items():

                self._collapse_adj[key] = self.dmg['collapse'].values * value

        return self._collapse_adj

    @property
    def collapse_adj_sim(self):
        """
        : calls self.dmg_state_sim
        :param seed: seed is None if no seed number is provided
        :return:
        """

        if self._collapse_adj_sim is None and self.cond_pc_adj_sim_idx and not self.dmg_state_sim['collapse'].empty:

            df = self.dmg_state_sim['collapse'].copy()

            # generate regardless of time index
            rv = self.rnd_state.uniform(
                size=len(self.dmg_state_sim['collapse']['id_sim']))

            df['id_adj'] = (rv[:, np.newaxis] >= self.cond_pc_adj_sim_prob).sum(axis=1)

            # remove case with no adjacent tower collapse
            # copy to avoid warning 
            df = df[df['id_adj'] < len(self.cond_pc_adj_sim_prob)].copy()

            # replace index with tower id
            df['id_adj'] = df['id_adj'].apply(lambda x: self.cond_pc_adj_sim_idx[x])

            # check against collapse_adj
            tmp = df.groupby(['id_time', 'id_adj']).apply(len).reset_index()

            for idl in self.cond_pc_adj.keys():

                prob = tmp.loc[tmp['id_adj'].apply(lambda x: idl in x)].groupby('id_time').sum() / self.no_sims
                try:
                    np.testing.assert_allclose(prob[0], self.collapse_adj[idl], atol=self.atol, rtol=self.rtol)
                except AssertionError:
                    self.logger.warning(
                        f'Pc({idl}|{self.name}): '
                        f'simulation {prob[0].values} vs. '
                        f'analytical {self.collapse_adj[idl]}')

            self._collapse_adj_sim = df.copy()

        return self._collapse_adj_sim

    #def compute_collapse_adj_sim(self, dmg_id_sim):
    #    """
    #    : calls self.dmg_state_sim
    #    :param seed: seed is None if no seed number is provided
    #    :return:
    #    """

    #    collapse_adj_sim = pd.DataFrame(None, columns=['id_time', 'id_adj'])
    #    collapse_adj_sim['id_time'] = dmg_id_sim['collapse']

    #    rv = self.rnd_state.uniform(size=len(dmg_id_sim['collapse']))

    #    collapse_adj_sim['id_adj'] = (
    #        rv[:, np.newaxis] >= self.cond_pc_adj_sim_prob).sum(axis=1)

    #    # remove case with no adjacent tower collapse
    #    collapse_adj_sim.loc[
    #        collapse_adj_sim['id_adj'] == len(self.cond_pc_adj_sim_prob),
    #        'id_adj'] = None
    #    collapse_adj_sim = collapse_adj_sim.loc[collapse_adj_sim['id_adj'].notnull()]

    #    # replace index with tower id
    #    collapse_adj_sim['id_adj'] = collapse_adj_sim['id_adj'].apply(
    #        lambda x: self.cond_pc_adj_sim_idx[int(x)])

    #    # check whether MC simulation is close to analytical
    #    #id_adj_removed = [x for x in self.id_adj if x >= 0]
    #    #if self.idl in id_adj_removed:
    #    #    id_adj_removed.remove(self.idl)

    #    return collapse_adj_sim


    @property
    def collapse_interaction(self):
        """
        determine damage to tower in target line
        :param seed: seed is None if no seed number is provided
        :return:
        """

        if self._collapse_interaction is None and not self.dmg.empty and self.cond_pc_interaction_no:

            self._collapse_interaction = self.dmg_state_sim['collapse'].copy()

            # generate regardless of time index
            rv = self.rnd_state.uniform(size=len(self.dmg_state_sim['collapse']['id_sim']))

            self._collapse_interaction['no_collapse'] = (
                rv[:, np.newaxis] >= self.cond_pc_interaction_cprob).sum(axis=1)

            # remove case with no tower collapse
            self._collapse_interaction = self._collapse_interaction.loc[
                self._collapse_interaction['no_collapse'] < len(self.cond_pc_interaction_cprob)].copy()

            # replace index with no_collapse
            self._collapse_interaction['no_collapse'] = self._collapse_interaction['no_collapse'].apply(lambda x: self.cond_pc_interaction_no[x])

            # check simulation vs. 
            df = self._collapse_interaction.groupby(['id_time', 'no_collapse']).apply(len).reset_index()
            for _, row in df.iterrows():
                analytical = self.dmg['collapse'].iloc[row['id_time']] * self.cond_pc_interaction_prob[row['no_collapse']]
                simulation = row[0] / self.no_sims
                try:
                    np.testing.assert_allclose(analytical, simulation, atol=self.atol, rtol=self.rtol)
                except AssertionError:
                    self.logger.warning(
                            f'Pc_interaction({self.name}): '
                            f'simulation {simulation} vs. '
                            f'analytical {analytical}')

        return self._collapse_interaction

    def init(self):
        """
        init properties
        """

        self._no_time = None
        self._dmg_time_idx = None
        self._file_wind = None
        self._wind = None
        self._dmg = None
        self._dmg_state_sim = None
        self._dmg_sim = None
        self._collapse_adj = None
        self._collapse_adj_sim = None
        self._collapse_interaction = None

        self.logger.debug(f'{self.name} is initialized')

    def get_directional_vulnerability(self, bearing):
        """

        :param row: pandas Series of wind
        :return:
                 | North
            ------------
           |     |      |
           |     |______| strong axis
           |            |
           |            |
            ------------

        """

        if len(self.sorted_frag_dic_keys) > 1:

            try:
                angle = angle_between_two(bearing, self.axisaz)
            except AssertionError:
                self.logger.error(f'Something wrong in bearing: {bearing}, axisaz: {self.axisaz} of {self.name}')
            else:
                # choose fragility given angle
                loc = min(bisect.bisect_right(self.sorted_frag_dic_keys, angle),
                      len(self.sorted_frag_dic_keys) - 1)
        else:
            loc = 0

        return self.sorted_frag_dic_keys[loc]

    def compute_damage_using_directional_vulnerability(self, row):
        """
        :param row: pandas Series of wind

        """

        key = self.get_directional_vulnerability(row['Bearing'])

        dmg = {}
        for ds, (fn, param1, param2) in self.frag_dic[key].items():
            value = getattr(stats, fn).cdf(row['ratio'], float(param2), scale=float(param1))
            dmg[ds] = np.nan_to_num(value, 0.0)

        return pd.Series(dmg)


def angle_between_two(deg1, deg2):
    """
    :param: deg1: angle 1 (0, 360)
            deg2: angle 2 (0, 360)
    """
    assert (deg1 >= 0) and (deg1 <= 360)
    assert (deg2 >= 0) and (deg2 <= 360)

    # angle between wind and tower strong axis (normal1)
    v1 = unit_vector_by_bearing(deg1)
    v2 = unit_vector_by_bearing(deg1 + 180)

    u = unit_vector_by_bearing(deg2)

    angle = min(angle_between_unit_vectors(u, v1),
                angle_between_unit_vectors(u, v2))

    return angle

