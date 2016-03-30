from __future__ import print_function

import os
import copy
import pandas as pd
import numpy as np
import warnings

from geopy.distance import great_circle
from scipy.stats import itemfreq

from wistl.tower import Tower

# ignore NaturalNameWarning
warnings.filterwarnings("ignore", lineno=100, module='tables')


class TransmissionLine(object):
    """ class for a collection of towers """

    def __init__(self, conf, df_towers, df_line):

        self.conf = conf
        self.df_towers = df_towers
        self.df_line = df_line
        self.name = df_line['LineRoute']

        name_ = self.name.split()
        self.name_output = '_'.join(x for x in name_ if x.isalnum())

        self.no_towers = len(self.df_towers)

        self.coord_line = np.array(self.df_line['Shapes'].points)  # lon, lat
        actual_span = self.calculate_distance_between_towers()

        self.id2name = dict()
        self.coord_towers = np.zeros((self.no_towers, 2))
        id_by_line_unsorted = []
        name_by_line_unsorted = []

        for i, (key, val) in enumerate(df_towers.iterrows()):
            self.id2name[key] = val['Name']
            self.coord_towers[i, :] = val['Shapes'].points[0]  # [Lon, Lat]
            id_by_line_unsorted.append(key)
            name_by_line_unsorted.append(val['Name'])
        self.sort_idx = self.sort_by_location()
        self.id_by_line = [id_by_line_unsorted[x] for x in self.sort_idx]
        self.name_by_line = [name_by_line_unsorted[x] for x in self.sort_idx]

        self.towers = dict()
        for i, tid in enumerate(self.id_by_line):
            val = df_towers.loc[tid, :].copy()
            name = val['Name']
            val['id'] = tid
            val['actual_span'] = actual_span[i]
            self.towers[name] = Tower(conf=self.conf, df_tower=val)
            self.towers[name].id_sides = self.assign_id_both_sides(i)
            self.towers[name].id_adj = self.assign_id_adj_towers(i)

        for key, val in self.towers.iteritems():
            self.towers[key].id_adj = self.update_id_adj_towers(val)
            self.towers[key].calculate_cond_pc_adj()

        # moved from damage_line.py
        self._event_id = None
        self._time_index = None
        self._path_event = None

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

    @property
    def time_index(self):
        return self._time_index

    @property
    def event_id(self):
        return self._event_id

    @property
    def path_event(self):
        return self._path_event

    @path_event.setter
    def path_event(self, path_event):
        self._path_event = path_event

    @event_id.setter
    def event_id(self, event_id):
        self._event_id = event_id
        self._set_damage_tower()

    def _set_damage_tower(self):

        key = None
        for key, tower in self.towers.iteritems():
            file_wind = os.path.join(self._path_event,
                                     tower.file_wind_base_name)

            # set wind file, which also sets wind and time_index
            self.towers[key].file_wind = file_wind

            # compute pc_wind and pc_adj
            self.towers[key].compute_pc_wind()
            self.towers[key].compute_pc_adj()

        self._time_index = self.towers[key].time_index

    def sort_by_location(self):
        """ sort towers by location"""

        idx_sorted = []
        for item in self.coord_line:
            diff = (self.coord_towers - np.ones((self.no_towers, 1)) *
                    item[np.newaxis, :])
            temp = np.linalg.norm(diff, axis=1)
            idx = np.argmin(temp)
            tf = abs(temp[idx]) < 1.0e-4
            if not tf:
                msg = 'Can not locate the tower in {}'.format(self.name)
                raise ValueError(msg)
            idx_sorted.append(idx)

        if self.conf.figure:

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(self.coord_line[:, 0],
                     self.coord_line[:, 1], 'ro-',
                     self.coord_towers[idx_sorted, 0],
                     self.coord_towers[idx_sorted, 1], 'b-')
            plt.title(self.name)
            png_file = os.path.join(self.conf.path_output,
                                    'line_{}.png'.format(self.name))
            plt.savefig(png_file)
            plt.close()

        return idx_sorted

    def assign_id_both_sides(self, idx):
        """
        assign id of towers on both sides
        :param idx: index of tower
        :return:
        """
        if idx == 0:
            return -1, self.id_by_line[idx + 1]
        elif idx == self.no_towers - 1:
            return self.id_by_line[idx - 1], -1
        else:
            return self.id_by_line[idx - 1], self.id_by_line[idx + 1]

    def assign_id_adj_towers(self, idx):
        """
        assign id of adjacent towers which can be affected by collapse
        :param idx: index of tower
        :return:
        """

        tid = self.id_by_line[idx]
        max_no_adj_towers = self.towers[self.id2name[tid]].max_no_adj_towers

        list_left = self.create_list_idx(idx, max_no_adj_towers, -1)
        list_right = self.create_list_idx(idx, max_no_adj_towers, 1)

        return list_left[::-1] + [tid] + list_right

    def create_list_idx(self, idx, no_towers, flag_direction):
        """
        create list of adjacent towers in each direction (flag=+/-1)

        :param idx:
        :param no_towers:
        :param flag_direction:
        :return:
        """
        list_tid = []
        for i in range(no_towers):
            idx += flag_direction
            if idx < 0 or idx > self.no_towers - 1:
                list_tid.append(-1)
            else:
                list_tid.append(self.id_by_line[idx])
        return list_tid

    def calculate_distance_between_towers(self):
        """ calculate actual span between the towers """
        dist_forward = np.zeros(len(self.coord_line) - 1)
        for i, (pt0, pt1) in enumerate(zip(self.coord_line[0:-1],
                                       self.coord_line[1:])):
            dist_forward[i] = great_circle(pt0[::-1], pt1[::-1]).meters

        actual_span = 0.5 * (dist_forward[0:-1] + dist_forward[1:])
        actual_span = np.insert(actual_span, 0, [0.5 * dist_forward[0]])
        actual_span = np.append(actual_span, [0.5 * dist_forward[-1]])

        return actual_span

    def update_id_adj_towers(self, tower):
        """
        replace id of strain tower with -1
        :param tower:
        :return:
        """
        id_adj = tower.id_adj
        for i, tid in enumerate(id_adj):
            if tid >= 0:
                function_ = self.towers[self.id2name[tid]].function
                if function_ in self.conf.strainer:
                    id_adj[i] = -1
        return id_adj

    # Moved from damage_line.py
    def write_hdf5(self, file_str, val):

        h5file = os.path.join(self.conf.path_output,
                              self.event_id,
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

        pc_adj_agg = np.zeros((self.no_towers,
                               self.no_towers,
                               len(self.time_index)))

        # prob of collapse
        for i_row, name in enumerate(self.name_by_line):
            for j in self.towers[name].pc_adj.keys():
                j_col = self.id_by_line.index(j)
                pc_adj_agg[i_row, j_col, :] = self.towers[name].pc_adj[j]
            pc_adj_agg[i_row, i_row, :] = \
                self.towers[name].pc_wind.collapse.values

        # pc_collapse.shape == (no_tower, no_time)
        pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)

        self.damage_prob_analytical['collapse'] = pd.DataFrame(
            pc_collapse.T,
            columns=self.name_by_line,
            index=self.time_index)

        # prob of non-collapse damage
        cds_list = copy.deepcopy(self.conf.damage_states)
        cds_list.remove('collapse')  # non-collapse

        for ds in cds_list:
            temp = np.zeros_like(pc_collapse)
            for i_row, name in enumerate(self.name_by_line):
                val = (self.towers[name].pc_wind[ds].values -
                       self.towers[name].pc_wind.collapse.values +
                       pc_collapse[i_row, :])
                val = np.where(val > 1.0, [1.0], val)
                temp[i_row, :] = val

            self.damage_prob_analytical[ds] = pd.DataFrame(
                temp.T,
                columns=self.name_by_line,
                index=self.time_index)

        if self.conf.save:
            self.write_hdf5(file_str='damage_prob_analytical',
                            val=self.damage_prob_analytical)

    def compute_damage_probability_simulation(self):

        tf_ds = np.zeros((self.no_towers,
                          self.conf.no_sims,
                          len(self.time_index)), dtype=bool)

        # collapse by adjacent towers
        for name in self.name_by_line:

            for j in self.towers[name].mc_adj.keys():  # time

                for k in self.towers[name].mc_adj[j].keys():  # fid

                    id_sim = self.towers[name].mc_adj[j][k]

                    for l in k:  # each fid

                        tf_ds[self.id_by_line.index(l), id_sim, j] = True

        cds_list = self.conf.damage_states[:]  # to avoid effect
        cds_list.reverse()  # [collapse, minor]

        tf_sim = dict()

        # append damage state by direct wind
        for ds in cds_list:

            for i_row, name in enumerate(self.name_by_line):

                for j, k in zip(self.towers[name].mc_wind[ds]['id_sim'],
                                self.towers[name].mc_wind[ds]['id_time']):

                    tf_ds[i_row, j, k] = True

            self.damage_prob_simulation[ds] = pd.DataFrame(
                np.sum(tf_ds, axis=1).T / float(self.conf.no_sims),
                columns=self.name_by_line,
                index=self.time_index)

            tf_sim[ds] = np.copy(tf_ds)  # why copy??

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

                id_sim = self.towers[name].mc_wind[ds]['id_sim']
                id_time = self.towers[name].mc_wind[ds]['id_time']

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
