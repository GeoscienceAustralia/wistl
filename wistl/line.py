
import copy
import time
import dask
import os
import pandas as pd
import numpy as np
import h5py
import logging

from wistl.tower import Tower
from wistl.constants import ATOL, RTOL


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
                  'path_event']

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
        self.scale = None

        for key, value in kwargs.items():
            if key in self.registered:
                setattr(self, key, value)

        self._towers = None

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

        # parallel line interaction
        # if self.cfg.line_interaction:
        #     self.damage_index_line_interaction = {target_line: [] for
        #         target_line in self.cfg.line_interaction[self.name]}
        # else:
        #     self.damage_index_line_interaction = None
        # self.damage_prob_line_interaction = {}
        # self.prob_no_damage_line_interaction = None
        # self.est_no_damage_line_interaction = None

        self._time = None
        self._time_idx = None
        self._no_time = None

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

            for key, value in self.dic_towers.items():

                value.update({'idn': key,
                              'no_sims': self.no_sims,
                              'damage_states': self.damage_states,
                              'scale': self.scale,
                              'rnd_state': self.rnd_state,
                              'path_event': self.path_event})

                self._towers[value['idl']] = Tower(**value)

        return self._towers

    @property
    def time(self):
        if self._time is None:
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
            tmp = []
            for _, value in self.towers.items():
                tmp.append(value.dmg_time_idx)
            id0 = list(map(min, zip(*tmp)))[0]
            id1 = list(map(max, zip(*tmp)))[1] + 1
            self._time_idx = (id0, id1)
        return self._time_idx

    @property
    def no_time(self):
        if self._no_time is None:
            self._no_time = len(self.time)
        return self._no_time

    def compute_damage_prob(self):
        """
        calculate damage probability of towers analytically
        Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
        where Pd: collapse due to direct wind
        Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))
        pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
        """
        self.damage_prob = {}
        pc_adj_agg = np.zeros((self.no_towers,
                               self.no_towers,
                               self.no_time))

        # prob of collapse
        for _, tower in self.towers.items():

            #idt = self.time.intersection(tower.dmg.index)
            #idt0 = self.time.get_loc(idt[0])
            #idt1 = self.time.get_loc(idt[-1]) + 1
            idt0, idt1 = tower.dmg_time_idx
            idt0 -= self.time_idx[0]
            idt1 -= self.time_idx[0]
            for idl, prob in tower.collapse_adj.items():
                pc_adj_agg[tower.idl, idl, idt0:idt1] = prob

            pc_adj_agg[tower.idl, tower.idl, idt0:idt1] = tower.dmg['collapse']

        # pc_collapse.shape == (no_tower, no_time)
        pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)

        self.damage_prob['collapse'] = pd.DataFrame(
            pc_collapse.T,
            columns=self.names,
            index=self.time)

        # non_collapse state
        for ds in self.non_collapse:
            temp = np.zeros_like(pc_collapse)
            for _, tower in self.towers.items():

                idt0, idt1 = tower.dmg_time_idx

                # P(DS>ds) - P(collapse directly) + P(collapse induced)
                try:
                    value = tower.dmg[ds].values \
                        - tower.dmg['collapse'].values \
                        + pc_collapse[tower.idl, idt0:idt1]
                except ValueError:
                    print(tower.dmg[ds].values.shape)
                    print(idt0, idt1)
                else:
                    temp[tower.idl, idt0:idt1] = np.where(value > 1.0, [1.0], value)

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
        #idx_by_ds = {ds: None for ds in self.damage_states}
        dic_tf_ds = {ds: None for ds in self.damage_states}
        tf_ds = np.zeros((self.no_towers,
                          self.no_sims,
                          self.no_time), dtype=bool)

        # collapse by adjacent towers
        for _, tower in self.towers.items():
            for id_adj, grp in tower.collapse_adj_sim.groupby('id_adj'):
                for idl in id_adj:
                    tf_ds[idl, grp['id_sim'], grp['id_time']] = True

        # append damage state by direct wind
        for ds in self.damage_states[::-1]:  # collapse first

            for _, tower in self.towers.items():
                tf_ds[tower.idl,
                      tower.dmg_state_sim[ds]['id_sim'],
                      tower.dmg_state_sim[ds]['id_time']] = True

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
                    self.damage_prob_sim['collapse'][name].values, atol=ATOL, rtol=RTOL)
            except AssertionError:
                self.logger.warning(f'Simulation results of {name}:collapse are not close to the analytical')

        #return idx_by_ds

    def compute_damage_prob_sim_no_cascading(self):

        self.damage_prob_sim_no_cascading = {}

        dic_tf_ds = {ds: None for ds in self.damage_states}

        tf_ds = np.zeros((self.no_towers,
                          self.no_sims,
                          self.no_time), dtype=bool)

        for ds in self.damage_states[::-1]:  # collapse first

            for _, tower in self.towers.items():

                tf_ds[tower.idl,
                      tower.dmg_state_sim[ds]['id_sim'],
                      tower.dmg_state_sim[ds]['id_time']] = True

            # PE(DS)
            self.damage_prob_sim_no_cascading[ds] = pd.DataFrame(
                tf_ds.sum(axis=1).T / self.no_sims,
                columns=self.names,
                index=self.time)

            dic_tf_ds[ds] = tf_ds.copy()

        self.no_damage_no_cascading, self.prob_no_damage_no_cascading = \
            self.compute_stats(dic_tf_ds)

        # checking against analytical value
        for _id, name in enumerate(self.names):
            idt0, idt1 = self.towers[_id].dmg_time_idx
            try:
                np.testing.assert_allclose(self.towers[_id].dmg[ds].values,
                        self.damage_prob_sim_no_cascading[ds].iloc[idt0:idt1][name].values, atol=ATOL, rtol=RTOL)
            except AssertionError:
                self.logger.warning(f'Simulation results of {name}:{ds} are not close to the analytical')


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


    def write_hdf5(self, output_file):

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

        with h5py.File(output_file, 'w') as hf:

            for item in items:

                group = hf.create_group(item)

                for ds in self.damage_states:

                    value = getattr(self, item)[ds]
                    data = group.create_dataset(ds, data=value)

                    # metadata
                    data.attrs['nrow'], data.attrs['ncol'] = value.shape
                    data.attrs['time_start'] = str(self.time[0])
                    data.attrs['time_freq'] = str(self.time[1]-self.time[0])
                    data.attrs['time_period'] = self.time.shape[0]

                    if columns_by_item[item]:
                        data.attrs['columns'] = ','.join('{}'.format(x) for x in columns_by_item[item])


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
        output_file = os.path.join(cfg.path_output,
                                   f'{line.event_id}_{line.name}.h5')
        #print(f"max: {line.name} - {line.damage_prob_sim['minor'].max()}")
        line.write_hdf5(output_file=output_file)
        logger.info(f'{output_file} is saved')

    #
    # try:
    #     line.cfg.line_interaction[line_name]
    # except (TypeError, KeyError):
    #     pass
    # else:
    #     line.determine_damage_by_interaction_at_line_level(seed)

    return line

'''
    def determine_damage_by_interaction_at_line_level(self, seed=None):
        """
        compute damage probability due to parallel line interaction using MC
        simulation
        :param seed:
        :return:
        """

        # for target_line in self.cfg.line_interaction[self.name]:
        #
        #     damage_index_line_interaction[target_line] = \

        # print('trigger line: {}'.format(self.name))
        # for target_line in self.cfg.line_interaction[self.name]:
        #
        #     damage_index_line_interaction[target_line] = \
        #         pd.DataFrame(columns=['id_tower', 'id_sim', 'id_time'],
        #                      dtype=np.int64)

            # self.tf_array_by_line[target_line] = np.zeros((
            #     self.cfg.no_towers_by_line[target_line],
            #     self.cfg.no_sims,
            #     len(self.time)), dtype=bool)

        #print('target in: {}'.format(self.damage_index_line_interaction[target_line].dtypes))

        for _, tower in self.towers.items():

            # determine damage state by line interaction
            # damage_interaction_sim['id_sim', 'id_time', 'no_collapse']
            tower.determine_damage_by_interaction_at_tower_level(seed)

            for id_time, grouped \
                    in tower.damage_interaction_sim.groupby('id_time'):

                wind_vector = unit_vector_by_bearing(
                    tower.wind['Bearing'][id_time])

                angle = {}
                for line_name, value in tower.id_on_target_line.items():
                    angle[line_name] = angle_between_unit_vectors(
                        wind_vector, value['vector'])

                target_line = min(angle, key=angle.get)

                if angle[target_line] < 180:

                    target_tower_id = tower.id_on_target_line[target_line]['id']
                    max_no_towers = self.cfg.no_towers_by_line[target_line]

                    for no_collapse, subgroup in grouped.groupby('no_collapse'):

                        no_towers_on_either_side = int(no_collapse / 2)

                        list_one_direction = tower.create_list_idx(
                            target_tower_id, no_towers_on_either_side,
                            max_no_towers, +1)

                        list_another_direction = tower.create_list_idx(
                            target_tower_id, no_towers_on_either_side,
                            max_no_towers, -1)

                        list_towers = [x for x in list_one_direction
                                       + [target_tower_id]
                                       + list_another_direction if x >= 0]

                        list_id = list(itertools.product(list_towers,
                                                    subgroup['id_sim'].values,
                                                    [id_time]))

                        # try:
                        #     df_id = pd.DataFrame(list_id, columns=['id_tower',
                        #                                        'id_sim',
                        #                                        'id_time'],
                        #                          dtype=np.int64)
                        # except IndexError:
                        #     print('IndexError')

                        if len(list_id) > 0:

                            self.damage_index_line_interaction[
                                target_line].append(list_id)

                        # for item in ):
                        #     self.damage_index_line_interaction[target_line][
                        #         item[0], item[1], id_time] = True
                else:
                    msg = 'tower:{}, angle: {}, wind:{}'.format(
                        tower.name, angle[target_line], wind_vector)
                    print(msg)
'''





