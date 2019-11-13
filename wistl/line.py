
import copy
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
        self.rnd_state = None

        for key, value in kwargs.items():
            if key in self.registered:
                setattr(self, key, value)

        self._towers = None

        # analytical method
        self._damage_prob = None

        # simulation method
        self._damage_prob_sim = None
        self.est_no_damage = None
        self.prob_no_damage = None

        # non cascading collapse
        self.damage_prob_sim_no_cascading = None
        self.est_no_damage_no_cascading = None
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

        self._time_index = None
        self._no_time = None

    def __repr__(self):
        return f'Line(name={self.name}, no_towers={self.no_towers}, event_id={self.event_id})'

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
    def time_index(self):
        if self._time_index is None:
            try:
                self._time_index = self.towers[0].wind.index
            except AttributeError:
                self.logger.error(f'Can not retrieve time_index of tower {self.towers[0].name}')
        return self._time_index

    @property
    def no_time(self):
        if self._no_time is None:
            self._no_time = len(self.time_index)
        return self._no_time

    @property
    def damage_prob(self):
        """
        calculate damage probability of towers analytically
        Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
        where Pd: collapse due to direct wind
        Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))
        pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
        """
        if self._damage_prob is None:

            self._damage_prob = {}

            pc_adj_agg = np.zeros((self.no_towers,
                                   self.no_towers,
                                   self.no_time))

            # prob of collapse
            for _, tower in self.towers.items():

                for idl, prob in tower.collapse_adj.items():
                    pc_adj_agg[tower.idl, idl, :] = prob

                pc_adj_agg[tower.idl, tower.idl, :] = tower.dmg['collapse'].values

            # pc_collapse.shape == (no_tower, no_time)
            pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)

            self._damage_prob['collapse'] = pd.DataFrame(
                pc_collapse.T,
                columns=self.names,
                index=self.time_index)

            # non_collapse state
            for ds in self.non_collapse:
                temp = np.zeros_like(pc_collapse)
                for _, tower in self.towers.items():
                    # P(DS>ds) - P(collapse directly) + P(collapse induced)
                    value = tower.dmg[ds].values \
                            - tower.dmg['collapse'].values \
                            + pc_collapse[tower.idl, :]
                    temp[tower.idl, :] = np.where(value > 1.0, [1.0], value)

                self._damage_prob[ds] = pd.DataFrame(
                    temp.T,
                    columns=self.names,
                    index=self.time_index)

        return self._damage_prob

    def compute_damage_prob_sim(self):

        # perfect correlation within a single line
        self.damage_prob_sim = {}

        tf_sim = {ds: None for ds in self.damage_states}

        tf_ds = np.zeros((self.no_towers,
                          self.no_sims,
                          self.no_time), dtype=bool)

        # collapse by adjacent towers
        for _, tower in self.towers.items():
            for id_adj, grouped in tower.collapse_adj_sim.groupby('id_adj'):
                for idl in id_adj:
                    tf_ds[idl,
                          grouped['id_sim'].values,
                          grouped['id_time'].values] = True

        # append damage state by direct wind
        for ds in self.damage_states[::-1]:  # collapse first

            for _, tower in self.towers.items():
                tf_ds[tower.idl,
                      tower.dmg_id_sim[ds]['id_sim'].values,
                      tower.dmg_id_sim[ds]['id_time'].values] = True

            # PE(DS)
            self.damage_prob_sim[ds] = pd.DataFrame(
                tf_ds.sum(axis=1).T / self.no_sims,
                columns=self.names,
                index=self.time_index)

            tf_sim[ds] = copy.deepcopy(tf_ds)

        # checking against analytical value
        for name in self.names:
            try:
                np.testing.assert_allclose(self.damage_prob['collapse'][name].values,
                    self.damage_prob_sim['collapse'][name].values, atol=ATOL, rtol=RTOL)
            except AssertionError:
                self.logger.warning(f'Simulation not close to analytical value for {name}')

        self.est_no_damage, self.prob_no_damage = self.compute_stats(tf_sim)

    def compute_damage_prob_sim_no_cascading(self):

        self.damage_prob_sim_no_cascading = {}

        tf_sim = {ds: None for ds in self.damage_states}

        tf_ds = np.zeros((self.no_towers,
                          self.no_sims,
                          self.no_time), dtype=bool)

        for ds in self.damage_states[::-1]:  # collapse first

            for _, tower in self.towers.items():

                tf_ds[tower.idl,
                      tower.dmg_id_sim[ds]['id_sim'].values,
                      tower.dmg_id_sim[ds]['id_time'].values] = True

            # PE(DS)
            self.damage_prob_sim_no_cascading[ds] = pd.DataFrame(
                tf_ds.sum(axis=1).T / self.no_sims,
                columns=self.names,
                index=self.time_index)

            tf_sim[ds] = tf_ds.copy()

        # checking against analytical value
        for _id, name in enumerate(self.names):
            try:
                np.testing.assert_allclose(self.towers[_id].dmg[ds].values,
                    self.damage_prob_sim_no_cascading[ds][name].values, atol=ATOL, rtol=RTOL)
            except AssertionError:
                self.logger.warning(f'Simulation not close to analytical value for {name}')

        self.est_no_damage_no_cascading, self.prob_no_damage_no_cascading = \
            self.compute_stats(tf_sim)

    def compute_stats(self, tf_sim):
        """
        compute mean and std of no. of ds
        tf_collapse_sim.shape = (no_towers, no_sim, no_time)
        :param tf_sim:
        :return:
        """

        est_no_tower = {}
        prob_no_tower = {}

        # (no_towers, 1)
        x_tower = np.array(range(self.no_towers + 1))[:, np.newaxis]
        x2_tower = x_tower ** 2.0

        tf_ds = np.zeros((self.no_towers,
                          self.no_sims,
                          self.no_time), dtype=bool)

        # from collapse and minor
        for ds in self.damage_states[::-1]:

            tf_ds = np.logical_xor(tf_sim[ds], tf_ds)

            # mean and standard deviation
            # no_ds_across_towers.shape == (no_sims, no_time)
            no_ds_across_towers = tf_ds.sum(axis=0)
            freq = np.zeros(shape=(self.no_time, self.no_towers + 1))

            for i in range(self.no_time):
                _value, _freq = np.unique(no_ds_across_towers[:, i], return_counts=True)  # (value, freq)
                freq[i, [int(x) for x in _value]] = _freq

            prob = freq / self.no_sims  # (no_time, no_towers)

            exp_no_tower = np.dot(prob, x_tower)
            std_no_tower = np.sqrt(np.dot(prob, x2_tower) - exp_no_tower ** 2)

            est_no_tower[ds] = pd.DataFrame(
                np.hstack((exp_no_tower, std_no_tower)),
                columns=['mean', 'std'],
                index=self.time_index)

            prob_no_tower[ds] = pd.DataFrame(
                prob,
                columns=[str(x) for x in range(self.no_towers + 1)],
                index=self.time_index)

        return est_no_tower, prob_no_tower

    def write_hdf5(self, output_file):

        items = ['damage_prob', 'damage_prob_sim', 'damage_prob_sim_no_cascading',
                 'est_no_damage', 'est_no_damage_no_cascading',
                 'prob_no_damage', 'prob_no_damage_no_cascading']

        columns_by_item = {'damage_prob': self.names,
                           'damage_prob_sim': self.names,
                           'damage_prob_sim_no_cascading': self.names,
                           'est_no_damage': ['mean', 'std'],
                           'est_no_damage_no_cascading': ['mean', 'std'],
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
                    data.attrs['time_start'] = str(self.time_index[0])
                    data.attrs['time_freq'] = str(self.time_index[1]-self.time_index[0])
                    data.attrs['time_period'] = self.time_index.shape[0]

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
    # line.compute_damage_prob()

    # perfect correlation within a single line
    line.compute_damage_prob_sim()

    if not cfg.options['skip_no_cascading_collapse']:
        line.compute_damage_prob_sim_no_cascading()

    # compare simulation against analytical
    for ds in cfg.damage_states:

        idx = np.where(~np.isclose(line.damage_prob_sim[ds],
                                   line.damage_prob[ds],
                                   atol=ATOL,
                                   rtol=RTOL))
        logger.debug(
                f'{ds}: {len(idx[0]):d} difference out of {line.no_towers*line.no_time:d}')

    # save
    if cfg.options['save_output']:
        output_file = os.path.join(cfg.path_output,
                                   f'{line.event_id}_{line.name}.h5')
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
            #     len(self.time_index)), dtype=bool)

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





