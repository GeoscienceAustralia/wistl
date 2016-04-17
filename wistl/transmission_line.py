from __future__ import print_function

import os
import copy
import warnings
import pandas as pd
import numpy as np

from geopy.distance import great_circle
from scipy.stats import itemfreq

from wistl.tower import Tower

# ignore NaturalNameWarning
warnings.filterwarnings("ignore", lineno=100, module='tables')


class TransmissionLine(object):
    """ class for a collection of towers """

    def __init__(self, conf, df_towers, ps_line):

        self.conf = conf
        self.df_towers = df_towers
        self.ps_line = ps_line

        self.line_string = self.ps_line.line_string
        self.coord = np.stack(self.ps_line.coord)
        self.coord_lat_lon = np.stack(self.ps_line.coord_lat_lon)
        self.name = ps_line.LineRoute
        self.name_output = ps_line.name_output
        self.no_towers = len(self.df_towers)

        # reset index by tower location according to line shapefile
        self.df_towers.index = self.sort_by_location()

        if self.conf.figure:
            self.plot_tower_line()

        self.id_by_line = range(self.no_towers)
        self.id2name = self.df_towers['Name'].to_dict()
        self.name_by_line = [self.df_towers.loc[i, 'Name'] for i in
                             range(self.no_towers)]

        self.df_towers = self.df_towers.join(
            self.calculate_distance_between_towers())

        self.towers = dict()
        for id_, ps_tower in self.df_towers.iterrows():
            name_ = ps_tower.Name
            self.towers[name_] = Tower(conf=self.conf, ps_tower=ps_tower)
            self.towers[name_].id_sides = self.assign_id_both_sides(id_)
            self.towers[name_].id_adj = self.assign_id_adj_towers(id_)

        for tower in self.towers.itervalues():
            tower.id_adj = \
                self.update_id_adj_by_filtering_strainer(tower)
            tower.calculate_cond_pc_adj()

        # moved from damage_line.py
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
        self.prob_no_damage_simulation = None
        self.est_no_damage_simulation = None

        # non cascading collapse
        self.damage_prob_simulation_non_cascading = dict()
        self.prob_no_damage_simulation_non_cascading = None
        self.est_no_damage_simulation_non_cascading = None

        # parallel line interaction
        self.damage_prob_simulation_line_interaction = dict()
        self.prob_no_damage_simulation_line_interaction = None
        self.est_no_damage_simulation_line_interaction = None

    @property
    def event_tuple(self):
        return self._event_tuple

    @event_tuple.setter
    def event_tuple(self, value):
        try:
            event_id_, scale_ = value
        except ValueError:
            raise ValueError("Pass an iterable with two items")
        else:
            self.event_id = event_id_
            self.scale = scale_
            self.path_event = os.path.join(self.conf.path_wind_scenario_base,
                                           event_id_)
            self.event_id_scale = '{}_s{:.1f}'.format(event_id_, scale_)
            self._set_damage_tower()

    def _set_damage_tower(self):

        tower = None
        for tower in self.towers.itervalues():
            file_wind = os.path.join(self.path_event,
                                     tower.file_wind_base_name)

            # set wind file, and compute damage prob. of tower in isolation
            tower.event_tuple = (file_wind, self.scale)
            # self.towers[key].compute_damage_prob_isolation()

        self.time_index = tower.time_index

    def sort_by_location(self):
        """ sort towers by location
        find the index of tower according to line
        """

        idx_sorted = []
        for item in self.df_towers.coord:
            diff = self.coord - np.ones((self.no_towers, 1)) * np.stack(item)
            temp = np.linalg.norm(diff, axis=1)
            idx = np.argmin(temp)
            tf = abs(temp[idx]) < 1.0e-4
            if not tf:
                msg = 'Can not locate the tower in {}'.format(self.name)
                raise ValueError(msg)
            idx_sorted.append(idx)

        return idx_sorted

    def plot_tower_line(self):

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

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
        png_file = os.path.join(self.conf.path_output,
                                'line_{}.png'.format(self.name))
        plt.savefig(png_file)
        plt.close()

    def assign_id_both_sides(self, idx):
        """
        assign id of towers on both sides
        :param idx: index of tower
        :return:
        """
        if idx == self.no_towers - 1:
            return self.no_towers - 2, -1
        else:
            return idx - 1, idx + 1

    def assign_id_adj_towers(self, idx):
        """
        assign id of adjacent towers which can be affected by collapse
        :param idx: index of tower
        :return:
        """

        max_no_adj_towers = self.towers[self.id2name[idx]].max_no_adj_towers

        list_left = self.create_list_idx(idx, max_no_adj_towers, -1)
        list_right = self.create_list_idx(idx, max_no_adj_towers, 1)

        return list_left[::-1] + [idx] + list_right

    def create_list_idx(self, idx, no_towers, flag_direction):
        """
        create list of adjacent towers in each direction (flag=+/-1)

        :param idx:
        :param no_towers:
        :param flag_direction:
        :return:
        """
        list_id = []
        for i in range(no_towers):
            idx += flag_direction
            if idx < 0 or idx > self.no_towers - 1:
                list_id.append(-1)
            else:
                list_id.append(idx)
        return list_id

    def calculate_distance_between_towers(self):
        """ calculate actual span between the towers """
        dist_forward = np.zeros(len(self.coord_lat_lon) - 1)
        for i, (pt0, pt1) in enumerate(zip(self.coord_lat_lon[0:-1],
                                       self.coord_lat_lon[1:])):
            dist_forward[i] = great_circle(pt0, pt1).meters

        actual_span = 0.5 * (dist_forward[0:-1] + dist_forward[1:])
        actual_span = np.insert(actual_span, 0, [0.5 * dist_forward[0]])
        actual_span = np.append(actual_span, [0.5 * dist_forward[-1]])
        return pd.DataFrame(actual_span, columns=['actual_span'])

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
                if self.towers[name_].function in self.conf.strainer:
                    id_adj[i] = -1
        return id_adj

    def write_hdf5(self, file_str, val):

        h5file = os.path.join(self.conf.path_output,
                              self.event_id_scale,
                              '{}_{}.h5'.format(file_str, self.name_output))
        hdf = pd.HDFStore(h5file)

        for ds in self.conf.damage_states:
            hdf.put(ds, val[ds], format='table', data_columns=True)
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

        for tower in self.towers.itervalues():
            tower.compute_damage_prob_adjacent()  # FIXME

        pc_adj_agg = np.zeros((self.no_towers,
                               self.no_towers,
                               len(self.time_index)))

        # prob of collapse
        for i, name in zip(self.id_by_line, self.name_by_line):
            for j in self.towers[name].prob_damage_adjacent.keys():
                pc_adj_agg[i, j, :] = self.towers[name].prob_damage_adjacent[j]
            pc_adj_agg[i, i, :] = self.towers[name].prob_damage_isolation.collapse.values

        # pc_collapse.shape == (no_tower, no_time)
        pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)

        self.damage_prob_analytical['collapse'] = pd.DataFrame(
            pc_collapse.T,
            columns=self.name_by_line,
            index=self.time_index)

        # prob of non-collapse damage
        cds_list = copy.deepcopy(self.conf.damage_states)
        cds_list.pop()  # remove the last item 'collapse'

        for ds in cds_list:
            temp = np.zeros_like(pc_collapse)
            for i, name in zip(self.id_by_line, self.name_by_line):
                val = (self.towers[name].prob_damage_isolation[ds].values -
                       self.towers[name].prob_damage_isolation.collapse.values +
                       pc_collapse[i, :])
                val = np.where(val > 1.0, [1.0], val)
                temp[i, :] = val

            self.damage_prob_analytical[ds] = pd.DataFrame(
                temp.T,
                columns=self.name_by_line,
                index=self.time_index)

        if self.conf.save:
            self.write_hdf5(file_str='damage_prob_analytical',
                            val=self.damage_prob_analytical)

    # def compute_damage_probability_simulation(self, seed):
    #
    #     # perfect correlation within a single line
    #     rnd_state = np.random.RandomState(seed)
    #
    #     rv = rnd_state.uniform(size=(self.conf.no_sims, len(self.time_index)))
    #
    #     tf_ds = np.zeros((self.no_towers,
    #                       self.conf.no_sims,
    #                       len(self.time_index)), dtype=bool)
    #
    #     # collapse by adjacent towers
    #     for tower in self.towers.itervalues():
    #
    #         tower.compute_mc_adj(rv, seed)
    #         # tower.determine_damage_isolation_mc(rv)
    #         # tower.determine_damage_adjacent_mc(seed)
    #
    #         for id_time, item in tower.damage_adjacent_mc.iteritems():
    #             for id_adj, id_sim in item.iteritems():
    #                 for i in id_adj:
    #                     tf_ds[i, id_sim, id_time] = True
    #
    #     cds_list = self.conf.damage_states[:]  # to avoid effect
    #     cds_list.reverse()  # [collapse, minor]
    #
    #     tf_sim = dict()
    #
    #     # append damage state by direct wind
    #     for ds in cds_list:
    #         for i_row, name in enumerate(self.name_by_line):
    #
    #             for j, k in zip(self.towers[name].damage_isolation_mc[ds]['id_sim'],
    #                             self.towers[name].damage_isolation_mc[ds]['id_time']):
    #
    #                 tf_ds[i_row, j, k] = True
    #
    #         self.damage_prob_simulation[ds] = pd.DataFrame(
    #             np.sum(tf_ds, axis=1).T / float(self.conf.no_sims),
    #             columns=self.name_by_line,
    #             index=self.time_index)
    #
    #         tf_sim[ds] = np.copy(tf_ds)  #
    #
    #     # line_interaction
    #     # if self.conf.line_interaction:
    #
    #     self.est_no_damage_simulation, self.prob_no_damage_simulation = \
    #         self.compute_damage_stats(tf_sim)
    #
    #     if self.conf.save:
    #
    #         self.write_hdf5(file_str='damage_prob_simulation',
    #                         val=self.damage_prob_simulation)
    #
    #         self.write_hdf5(file_str='est_no_damage_simulation',
    #                         val=self.est_no_damage_simulation)
    #
    #         self.write_hdf5(file_str='prob_no_damage_simulation',
    #                         val=self.prob_no_damage_simulation)

    def compute_damage_probability_simulation(self, seed):

        # perfect correlation within a single line
        rnd_state = np.random.RandomState(seed)
        rv = rnd_state.uniform(size=(self.conf.no_sims, len(self.time_index)))

        tf_ds = np.zeros((self.no_towers,
                          self.conf.no_sims,
                          len(self.time_index)), dtype=bool)

        # collapse by adjacent towers
        for tower in self.towers.itervalues():

            # tower.compute_mc_adj(rv, seed)
            tower.determine_damage_isolation_mc(rv)
            tower.determine_damage_adjacent_mc(seed)

            for id_time, item in tower.damage_adjacent_mc.iteritems():
                for id_adj, id_sim in item.iteritems():
                    for i in id_adj:
                        try:
                            tf_ds[i, id_sim, id_time] = True
                        except ValueError:
                            msg = 'id:{},\nid_sim:{}\n,id_time:{}'.\
                                format(i, id_sim, id_time)
                            raise ValueError(msg)

        cds_list = self.conf.damage_states[:]  # to avoid effect
        cds_list.reverse()  # [collapse, minor]

        tf_sim = dict()
        # append damage state by direct wind
        for ds in cds_list:
            for tower in self.towers.itervalues():
                for id_time, grouped in \
                        tower.damage_isolation_mc[ds].groupby('id_time'):

                    tf_ds[tower.id, grouped.id_sim.tolist(), id_time] = True

            self.damage_prob_simulation[ds] = pd.DataFrame(
                np.sum(tf_ds, axis=1).T / float(self.conf.no_sims),
                columns=self.name_by_line,
                index=self.time_index)

            tf_sim[ds] = np.copy(tf_ds)  #

        self.est_no_damage_simulation, self.prob_no_damage_simulation = \
            self.compute_damage_stats(tf_sim)

        if self.conf.save:
            self.write_hdf5(file_str='damage_prob_simulation',
                            val=self.damage_prob_simulation)

            self.write_hdf5(file_str='est_no_damage_simulation',
                            val=self.est_no_damage_simulation)

            self.write_hdf5(file_str='prob_no_damage_simulation',
                            val=self.prob_no_damage_simulation)

    def compute_damage_probability_simulation_non_cascading(self):

        tf_sim_non_cascading = dict()

        for ds in self.conf.damage_states:

            tf_ds = np.zeros((self.no_towers,
                              self.conf.no_sims,
                              len(self.time_index)), dtype=bool)

            for i_row, name in enumerate(self.name_by_line):

                id_sim = self.towers[name].damage_isolation_mc[ds]['id_sim']
                id_time = self.towers[name].damage_isolation_mc[ds]['id_time']

                tf_ds[i_row, id_sim, id_time] = True

            self.damage_prob_simulation_non_cascading[ds] = pd.DataFrame(
                np.sum(tf_ds, axis=1).T / float(self.conf.no_sims),
                columns=self.name_by_line,
                index=self.time_index)

            tf_sim_non_cascading[ds] = np.copy(tf_ds)

        self.est_no_damage_simulation_non_cascading, \
            self.prob_no_damage_simulation_non_cascading = \
            self.compute_damage_stats(tf_sim_non_cascading)

        if self.conf.save:

            self.write_hdf5(file_str='damage_prob_simulation_non_cascading',
                            val=self.damage_prob_simulation_non_cascading)

            self.write_hdf5(file_str='est_no_damage_simulation_non_cascading',
                            val=self.est_no_damage_simulation_non_cascading)

            self.write_hdf5(file_str='prob_no_damage_simulation_non_cascading',
                            val=self.prob_no_damage_simulation_non_cascading)

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
        no_sims = self.conf.no_sims

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

    '''
    def compute_damage_probability_simulation_line_interaction(self):

        for tower in self.towers.iteritems():

            # if

            for target_line,
                target_tower_id in tower.id_on_target_line.iteritems():

                #

        for target_line in self.conf.line_interaction[self.name]:

            tf_ds = np.zeros((target_line.no_towers,
                              self.conf.no_sims,
                              len(self.time_index)), dtype=bool)

    '''
