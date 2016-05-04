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

from scipy.stats import itemfreq

from wistl.tower import Tower

# ignore NaturalNameWarning
warnings.filterwarnings("ignore", lineno=100, module='tables')


class TransmissionLine(object):
    """ class for a collection of towers """

    def __init__(self, cfg, df_towers, ps_line):

        self.cfg = cfg
        self.ps_line = ps_line
        self.df_towers = df_towers

        self.line_string = self.ps_line.line_string
        self.coord = np.stack(self.ps_line.coord)
        self.coord_lat_lon = np.stack(self.ps_line.coord_lat_lon)
        self.name = ps_line.LineRoute
        self.name_output = ps_line.name_output
        self.no_towers = len(self.df_towers)

        self._event_tuple = None

        self.event_id_scale = None
        self.event_id = None
        self.scale = None

        self.time_index = None
        self.path_event = None

        # analytical method
        self.damage_prob_analytical = dict()

        # simulation method
        self.damage_prob_simulation = dict()
        self.damage_index_simulation = dict()
        self.prob_no_damage_simulation = None
        self.est_no_damage_simulation = None

        # non cascading collapse
        self.damage_prob_simulation_non_cascading = dict()
        self.prob_no_damage_simulation_non_cascading = None
        self.est_no_damage_simulation_non_cascading = None

        # parallel line interaction
        if self.cfg.line_interaction:
            self.damage_index_line_interaction = {target_line: [] for
                target_line in self.cfg.line_interaction[self.name]}
        else:
            self.damage_index_line_interaction = None
        self.damage_prob_line_interaction = dict()
        self.prob_no_damage_line_interaction = None
        self.est_no_damage_line_interaction = None

        # reset index by tower location according to line shapefile
        self.df_towers.index = self.sort_by_location()

        if self.cfg.figure:
            self.plot_tower_line()

        self.id_by_line = range(self.no_towers)
        self.id2name = self.df_towers['Name'].to_dict()
        self.name_by_line = [self.id2name[i] for i in range(self.no_towers)]
        self.df_towers.loc[:, 'actual_span'] = ps_line['actual_span']

        self.towers = dict()
        for id_, ps_tower in self.df_towers.iterrows():
            self.towers[ps_tower.Name] = Tower(cfg=self.cfg, ps_tower=ps_tower)

        for tower in self.towers.itervalues():
            tower.id_adj = self.update_id_adj_by_filtering_strainer(tower)
            tower.calculate_cond_pc_adj()

    @property
    def event_tuple(self):
        return self._event_tuple

    @event_tuple.setter
    def event_tuple(self, value):
        try:
            event_id, scale = value
        except ValueError:
            msg = "Pass a tuple of event_id and scale"
            raise ValueError(msg)
        else:
            self.event_id = event_id
            self.scale = scale
            self.path_event = os.path.join(self.cfg.path_wind_scenario_base,
                                           event_id)
            self.event_id_scale = self.cfg.event_id_scale_str.format(
                event_id=event_id, scale=scale)
            self.set_damage_tower()

    def set_damage_tower(self):

        tower = None
        for tower in self.towers.itervalues():
            file_wind = os.path.join(self.path_event, tower.file_wind_base_name)
            # set wind file, and compute damage prob. of tower in isolation
            tower.event_tuple = (file_wind, self.scale)

        try:
            # assuming same time index for each tower in the same network
            self.time_index = tower.time_index
        except AttributeError:
            msg = 'No transmission tower created'
            raise AttributeError(msg)

    def sort_by_location(self):
        """ sort towers by location
        find the index of tower according to line
        """

        idx_sorted = []
        for coord in self.df_towers.coord:
            diff = self.coord - np.ones((self.no_towers, 1)) * np.stack(coord)
            temp = np.linalg.norm(diff, axis=1)
            idx = np.argmin(temp)
            tf = abs(temp[idx]) < 1.0e-4
            if not tf:
                msg = 'Can not locate the tower in {}'.format(self.name)
                raise ValueError(msg)
            idx_sorted.append(idx)

        return idx_sorted

    def plot_tower_line(self):

        coord_towers = []
        for i in range(self.no_towers):
            coord_towers.append(self.df_towers.loc[i, 'coord'])
        coord_towers = np.array(coord_towers)

        plt.figure()
        plt.plot(self.coord[:, 0],
                 self.coord[:, 1], 'ro',
                 coord_towers[:, 0],
                 coord_towers[:, 1], 'b-')
        plt.title(self.name)

        png_file = os.path.join(self.cfg.path_output,
                                'line_{}.png'.format(self.name))

        if not os.path.exists(self.cfg.path_output):
            os.makedirs(self.cfg.path_output)

        plt.savefig(png_file)
        print('{} is created'.format(png_file))
        plt.close()

    # def assign_id_adj_towers(self, idx):
    #     """
    #     assign id of adjacent towers which can be affected by collapse
    #     :param idx: index of tower
    #     :return:
    #     """
    #
    #     max_no_adj_towers = self.towers[self.id2name[idx]].max_no_adj_towers
    #
    #     list_left = self.create_list_idx(idx, max_no_adj_towers, -1)
    #     list_right = self.create_list_idx(idx, max_no_adj_towers, 1)
    #
    #     return list_left[::-1] + [idx] + list_right
    #
    # def create_list_idx(self, idx, no_towers, flag_direction):
    #     """
    #     create list of adjacent towers in each direction (flag=+/-1)
    #
    #     :param idx:
    #     :param no_towers:
    #     :param flag_direction:
    #     :return:
    #     """
    #     list_id = []
    #     for i in range(no_towers):
    #         idx += flag_direction
    #         if idx < 0 or idx > self.no_towers - 1:
    #             list_id.append(-1)
    #         else:
    #             list_id.append(idx)
    #     return list_id

    # def calculate_distance_between_towers(self):
    #     """ calculate actual span between the towers """
    #     dist_forward = np.zeros(len(self.coord_lat_lon) - 1)
    #     for i, (pt0, pt1) in enumerate(zip(self.coord_lat_lon[0:-1],
    #                                    self.coord_lat_lon[1:])):
    #         dist_forward[i] = great_circle(pt0, pt1).meters
    #
    #     actual_span = 0.5 * (dist_forward[0:-1] + dist_forward[1:])
    #     actual_span = np.insert(actual_span, 0, [0.5 * dist_forward[0]])
    #     actual_span = np.append(actual_span, [0.5 * dist_forward[-1]])
    #     return pd.Series(actual_span, name='actual_span')

    def update_id_adj_by_filtering_strainer(self, tower):
        """
        replace id of strain tower with -1
        :param tower:
        :return:
        """
        id_adj = tower.id_adj
        for i, tid in enumerate(id_adj):
            try:
                name_ = self.id2name[tid]
            except KeyError:
                pass
            else:
                if self.towers[name_].function in self.cfg.strainer:
                    id_adj[i] = -1
        return id_adj

    def write_hdf5(self, file_str, value):

        h5file = os.path.join(self.cfg.path_output,
                              self.event_id_scale,
                              '{}_{}.h5'.format(file_str, self.name_output))
        hdf = pd.HDFStore(h5file)

        for ds in self.cfg.damage_states:
            try:
                hdf.put(ds, value[ds], format='table', data_columns=True)
            except TypeError:
                msg = '{0} of {1} is empty'.format(ds, h5file)
                print(msg)
        hdf.close()
        print('{} is created'.format(h5file))

    def compute_damage_probability_analytical(self):
        """
        calculate damage probability of towers analytically
        Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
        where Pd: collapse due to direct wind
        Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))
        pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
        """
        pc_adj_agg = np.zeros((self.no_towers,
                               self.no_towers,
                               len(self.time_index)))

        # prob of collapse
        for tower in self.towers.itervalues():
            tower.compute_damage_prob_adjacent()  # FIXME

            for key, value in tower.prob_damage_adjacent.iteritems():
                pc_adj_agg[tower.id, key, :] = value

            pc_adj_agg[tower.id, tower.id, :] = \
                tower.prob_damage_isolation.collapse.values

        # pc_collapse.shape == (no_tower, no_time)
        pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)

        self.damage_prob_analytical['collapse'] = pd.DataFrame(
            pc_collapse.T,
            columns=self.name_by_line,
            index=self.time_index)

        # prob of non-collapse damage
        cds_list = copy.deepcopy(self.cfg.damage_states)
        cds_list.pop()  # remove the last item 'collapse'

        for ds in cds_list:
            temp = np.zeros_like(pc_collapse)
            for tower in self.towers.itervalues():
                value = tower.prob_damage_isolation[ds].values \
                        - tower.prob_damage_isolation.collapse.values \
                        + pc_collapse[tower.id, :]
                temp[tower.id, :] = np.where(value > 1.0, [1.0], value)

            self.damage_prob_analytical[ds] = pd.DataFrame(
                temp.T,
                columns=self.name_by_line,
                index=self.time_index)

        if self.cfg.save:
            self.write_hdf5(file_str='damage_prob_analytical',
                            value=self.damage_prob_analytical)

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

        tf_sim = dict()

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
                columns=self.name_by_line,
                index=self.time_index)

            tf_sim[ds] = np.copy(tf_ds)  #

        # save tf_sim[ds]
        self.est_no_damage_simulation, self.prob_no_damage_simulation = \
            self.compute_damage_stats(tf_sim)


        if self.cfg.save:
            self.write_hdf5(file_str='damage_prob_simulation',
                            value=self.damage_prob_simulation)

            self.write_hdf5(file_str='est_no_damage_simulation',
                            value=self.est_no_damage_simulation)

            self.write_hdf5(file_str='prob_no_damage_simulation',
                            value=self.prob_no_damage_simulation)

    def compute_damage_probability_simulation_non_cascading(self):

        tf_sim_non_cascading = dict()

        for ds in self.cfg.damage_states:

            tf_ds = np.zeros((self.no_towers,
                              self.cfg.no_sims,
                              len(self.time_index)), dtype=bool)

            for i_row, name in enumerate(self.name_by_line):

                id_sim = self.towers[name].damage_isolation_mc[ds]['id_sim'].values
                id_time = self.towers[name].damage_isolation_mc[ds]['id_time'].values

                tf_ds[i_row, id_sim, id_time] = True

            self.damage_prob_simulation_non_cascading[ds] = pd.DataFrame(
                np.sum(tf_ds, axis=1).T / float(self.cfg.no_sims),
                columns=self.name_by_line,
                index=self.time_index)

            tf_sim_non_cascading[ds] = np.copy(tf_ds)

        self.est_no_damage_simulation_non_cascading, \
            self.prob_no_damage_simulation_non_cascading = \
            self.compute_damage_stats(tf_sim_non_cascading)

        if self.cfg.save:

            self.write_hdf5(file_str='damage_prob_simulation_non_cascading',
                            value=self.damage_prob_simulation_non_cascading)

            self.write_hdf5(file_str='est_no_damage_simulation_non_cascading',
                            value=self.est_no_damage_simulation_non_cascading)

            self.write_hdf5(file_str='prob_no_damage_simulation_non_cascading',
                            value=self.prob_no_damage_simulation_non_cascading)

    def compute_damage_stats(self, tf_sim):
        """
        compute mean and std of no. of ds
        tf_collapse_sim.shape = (no_towers, no_sim, no_time)
        :param tf_sim:
        :return:
        """

        est_damage_tower = dict()
        prob_damage_tower = dict()

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

                angle = dict()
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


def compute_damage_probability_per_line(damage_line):
    """
    mc simulation over transmission line
    :param damage_line: instance of transmission line
    :return: None but update attributes of
    """

    event_id = damage_line.event_id
    line_name = damage_line.name

    if damage_line.cfg.random_seed:
        try:
            seed = damage_line.cfg.seed[event_id][line_name]
        except KeyError:
            msg = '{}:{} is undefined. Check the config file'.format(
                event_id, line_name)
            raise KeyError(msg)
    else:
        seed = None

    # compute damage probability analytically
    if damage_line.cfg.analytical:
        damage_line.compute_damage_probability_analytical()

    # perfect correlation within a single line
    damage_line.compute_damage_probability_simulation(seed)

    if not damage_line.cfg.skip_non_cascading_collapse:
        damage_line.compute_damage_probability_simulation_non_cascading()

    try:
        damage_line.cfg.line_interaction[line_name]
    except (TypeError, KeyError):
        pass
    else:
        damage_line.determine_damage_by_interaction_at_line_level(seed)

    return damage_line




