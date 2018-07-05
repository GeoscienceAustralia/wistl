from __future__ import print_function

import os
import copy
import warnings
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

from scipy.stats import itemfreq

from wistl.tower import Tower

# ignore NaturalNameWarning
warnings.filterwarnings("ignore", lineno=100, module='tables')


class TransmissionLine(object):
    """ class for a collection of towers """

    registered = ['coord',
                  'coord_lat_lon',
                  'id2name',
                  'ids',
                  'line_string',
                  'name_output',
                  'names',
                  'no_towers',
                  'flag_save',
                  'no_sims',
                  'damage_states',
                  'event_id',
                  'path_event']

    def __init__(self, name=None, logger=None, **kwargs):

        self.name = name
        self.logger = logger or logging.getLogger(__name__)

        self.coord = None
        self.coord_lat_lon = None
        self.id2name = None
        self.ids = None
        self.line_string = None
        self.name_output = None
        self.names = None
        self.no_towers = None
        self.flag_save = None
        self.no_sims = None
        self.damage_states = None
        self.path_event = None
        self.event_id = None

        for key, value in kwargs.items():
            if key in self.registered:
                setattr(self, key, value)

        self._towers = None

        # analytical method
        self._damage_prob_analytical = None

        # simulation method
        self.damage_prob_simulation = {}
        self.damage_index_simulation = {}
        self.prob_no_damage_simulation = None
        self.est_no_damage_simulation = None

        # non cascading collapse
        self.damage_prob_simulation_non_cascading = {}
        self.prob_no_damage_simulation_non_cascading = None
        self.est_no_damage_simulation_non_cascading = None

        # parallel line interaction
        # if self.cfg.line_interaction:
        #     self.damage_index_line_interaction = {target_line: [] for
        #         target_line in self.cfg.line_interaction[self.name]}
        # else:
        #     self.damage_index_line_interaction = None
        self.damage_prob_line_interaction = {}
        self.prob_no_damage_line_interaction = None
        self.est_no_damage_line_interaction = None

        self._time_index = None

    @property
    def towers(self):
        return self._towers

    @towers.setter
    def towers(self, dic):
        assert isinstance(dic, dict)

        self._towers = {}

        for key, value in dic.items():

            value.update({'path_event': self.path_event})

            tower = Tower(tower_id=key, **value)
            self._towers[value['id_line']] = tower

    @property
    def time_index(self):
        if self._time_index is None:
            self._time_index = self.towers[0].wind.index
        return self._time_index

    # def plot_tower_line(self):
    #
    #     coord_towers = []
    #     for i in range(self.no_towers):
    #         coord_towers.append(self.df_towers.loc[i, 'coord'])
    #     coord_towers = np.array(coord_towers)
    #
    #     plt.figure()
    #     plt.plot(self.coord[:, 0],
    #              self.coord[:, 1], 'ro',
    #              coord_towers[:, 0],
    #              coord_towers[:, 1], 'b-')
    #     plt.title(self.name)
    #
    #     png_file = os.path.join(self.cfg.path_output,
    #                             'line_{}.png'.format(self.name))
    #
    #     if not os.path.exists(self.cfg.path_output):
    #         os.makedirs(self.cfg.path_output)
    #
    #     plt.savefig(png_file)
    #     print('{} is created'.format(png_file))
    #     plt.close()

    def compute_damage_prob_analytical(self):
        """
        calculate damage probability of towers analytically
        Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
        where Pd: collapse due to direct wind
        Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))
        pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
        """

        self._damage_prob_analytical = {}

        pc_adj_agg = np.zeros((self.no_towers,
                               self.no_towers,
                               self.time_index.size))

        # prob of collapse
        for _, tower in self.towers.items():

            for key, value in tower.damage_prob_adjacent.items():
                pc_adj_agg[tower.id_line, key, :] = value

            pc_adj_agg[tower.id_line, tower.id_line, :] = \
                tower.damage_prob_isolation['collapse'].values

        # pc_collapse.shape == (no_tower, no_time)
        pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)

        self._damage_prob_analytical['collapse'] = pd.DataFrame(
            pc_collapse.T,
            columns=self.names,
            index=self.time_index)

        # prob of non-collapse damage
        cds_list = copy.deepcopy(self.damage_states)
        cds_list.remove('collapse')  # remove the last item 'collapse'

        for ds in cds_list:
            temp = np.zeros_like(pc_collapse)
            for _, tower in self.towers.items():
                value = tower.damage_prob_isolation[ds].values \
                        - tower.damage_prob_isolation['collapse'].values \
                        + pc_collapse[tower.id_line, :]
                temp[tower.id_line, :] = np.where(value > 1.0, [1.0], value)

            self._damage_prob_analytical[ds] = pd.DataFrame(
                temp.T,
                columns=self.names,
                index=self.time_index)

    """
    def compute_damage_probability_simulation(self, seed):

        # perfect correlation within a single line
        rnd_state = np.random.RandomState(seed)
        rv = rnd_state.uniform(size=(self.cfg.no_sims, len(self.time_index)))

        tf_ds = np.zeros((self.no_towers,
                          self.cfg.no_sims,
                          len(self.time_index)), dtype=bool)

        # collapse by adjacent towers
        for tower in self.towers.itervalues():

            # tower.compute_mc_adj(rv, seed)
            tower.determine_damage_isolation_mc(rv)
            tower.determine_damage_adjacent_mc(seed)

            for id_adj, grouped in tower.damage_adjacent_mc.groupby('id_adj'):
                for i in id_adj:
                    tf_ds[i, grouped['id_sim'].values, grouped['id_time'].values] = True

        cds_list = self.cfg.damage_states[:]  # to avoid effect
        cds_list.reverse()  # [collapse, minor]

        tf_sim = {}

        # append damage state by direct wind
        for ds in cds_list:
            for tower in self.towers.itervalues():

                tf_ds[tower.id,
                      tower.damage_isolation_mc[ds]['id_sim'].values,
                      tower.damage_isolation_mc[ds]['id_time'].values] = True

            id_tower, id_sim, id_time = np.where(tf_ds)

            self.damage_index_simulation[ds] = pd.DataFrame(
                np.vstack((id_tower, id_sim, id_time)).T,
                columns=['id_tower', 'id_sim', 'id_time'])

            self.damage_prob_simulation[ds] = pd.DataFrame(
                np.sum(tf_ds, axis=1).T / float(self.cfg.no_sims),
                columns=self.names,
                index=self.time_index)

            tf_sim[ds] = np.copy(tf_ds)  #

        # save tf_sim[ds]
        self.est_no_damage_simulation, self.prob_no_damage_simulation = \
            self.compute_damage_stats(tf_sim)


    def compute_damage_probability_simulation_non_cascading(self):

        tf_sim_non_cascading = {}

        for ds in self.cfg.damage_states:

            tf_ds = np.zeros((self.no_towers,
                              self.cfg.no_sims,
                              len(self.time_index)), dtype=bool)

            for i_row, name in enumerate(self.names):

                id_sim = self.towers[name].damage_isolation_mc[ds]['id_sim'].values
                id_time = self.towers[name].damage_isolation_mc[ds]['id_time'].values

                tf_ds[i_row, id_sim, id_time] = True

            self.damage_prob_simulation_non_cascading[ds] = pd.DataFrame(
                np.sum(tf_ds, axis=1).T / float(self.cfg.no_sims),
                columns=self.names,
                index=self.time_index)

            tf_sim_non_cascading[ds] = np.copy(tf_ds)

        self.est_no_damage_simulation_non_cascading, \
            self.prob_no_damage_simulation_non_cascading = \
            self.compute_damage_stats(tf_sim_non_cascading)
    """
    '''
    def compute_damage_stats(self, tf_sim):
        """
        compute mean and std of no. of ds
        tf_collapse_sim.shape = (no_towers, no_sim, no_time)
        :param tf_sim:
        :return:
        """

        est_damage_tower = {}
        prob_damage_tower = {}

        no_time = len(self.time_index)
        no_towers = self.no_towers
        no_sims = self.cfg.no_sims

        for ds in tf_sim:

            assert tf_sim[ds].shape == (no_towers, no_sims, no_time)

            # mean and standard deviation
            x_ = np.array(range(no_towers + 1))[:, np.newaxis]  # (no_towers, 1)
            x2_ = x_ ** 2.0

            # no_ds_across_towers.shape == (no_sims, no_time)
            no_ds_across_towers = np.sum(tf_sim[ds], axis=0)
            no_freq = np.zeros(shape=(no_time, no_towers + 1))

            for i in range(no_time):
                val = itemfreq(no_ds_across_towers[:, i])  # (value, freq)
                no_freq[i, [int(x) for x in val[:, 0]]] = val[:, 1]

            prob = no_freq / float(no_sims)  # (no_time, no_towers)

            exp_no_tower = np.dot(prob, x_)
            std_no_tower = np.sqrt(np.dot(prob, x2_) - exp_no_tower ** 2)

            est_damage_tower[ds] = pd.DataFrame(
                np.hstack((exp_no_tower, std_no_tower)),
                columns=['mean', 'std'],
                index=self.time_index)

            prob_damage_tower[ds] = pd.DataFrame(
                prob,
                columns=[str(x) for x in range(no_towers + 1)],
                index=self.time_index)

        return est_damage_tower, prob_damage_tower

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

        for tower in self.towers.itervalues():

            # determine damage state by line interaction
            # damage_interaction_mc['id_sim', 'id_time', 'no_collapse']
            tower.determine_damage_by_interaction_at_tower_level(seed)

            for id_time, grouped \
                    in tower.damage_interaction_mc.groupby('id_time'):

                wind_vector = unit_vector_by_bearing(
                    tower.wind['Bearing'][id_time])

                angle = {}
                for line_name, value in tower.id_on_target_line.iteritems():
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

def unit_vector_by_bearing(angle_deg):
    """
    return unit vector given bearing
    :param angle_deg: 0-360
    :return: unit vector given bearing in degree
    """
    angle_rad = np.deg2rad(angle_deg)
    return np.array([ -1.0 * np.sin(angle_rad), -1.0 * np.cos(angle_rad)])


def angle_between_unit_vectors(v1, v2):
    """
    compute angle between two unit vectors
    :param v1: vector 1
    :param v2: vector 2
    :return: the angle in degree between vectors 'v1' and 'v2'

            >>> angle_between_unit_vectors((1, 0), (0, 1))
            90.0
            >>> angle_between_unit_vectors((1, 0), (1, 0))
            0.0
            >>> angle_between_unit_vectors((1, 0), (-1, 0))
            180.0

    """
    #     v1_u = unit_vector(v1)
    #     v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def compute_damage_probability_per_line(line, cfg):
    """
    mc simulation over transmission line
    :param line: instance of transmission line
           cfg:
    :return: None but update attributes of
    """

    # logger = logging.getLogger(__name__)

    # event_id = line.event_id
    # line_name = line.name
    #
    # if line.cfg.random_seed:
    #     try:
    #         seed = line.cfg.seed[event_id][line_name]
    #     except KeyError:
    #         msg = '{}:{} is undefined. Check the config file'.format(
    #             event_id, line_name)
    #         raise KeyError(msg)
    # else:
    #     seed = None

    # compute damage probability analytically
    if cfg.options['run_analytical']:
        line.compute_damage_prob_analytical()

    # # perfect correlation within a single line
    # line.compute_damage_probability_simulation(seed)
    #
    # if not line.cfg.skip_non_cascading_collapse:
    #     line.compute_damage_probability_simulation_non_cascading()
    #
    # try:
    #     line.cfg.line_interaction[line_name]
    # except (TypeError, KeyError):
    #     pass
    # else:
    #     line.determine_damage_by_interaction_at_line_level(seed)

    return line




