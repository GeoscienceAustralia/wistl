#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import numpy as np
from tower import Tower
from geopy.distance import great_circle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TransmissionLine(object):
    """ class for a collection of towers """

    def __init__(self, conf, df_towers, df_line):

        self.conf = conf
        self.df_towers = df_towers
        self.df_line = df_line
        self.no_towers = len(self.df_towers)

        self.coord_line = np.array(self.df_line['Shapes'].values[0].points)
        actual_span = self.calculate_distance_between_towers()

        self.id2name = dict()
        self.coord_towers = np.zeros((self.no_towers, 2))
        id_by_line_unsorted = []
        #name_by_line_unsorted = []
        for i, (key, item) in enumerate(df_towers.iterrows()):
            self.id2name[key] = item['Name']
            self.coord_towers[i, :] = item['Shapes'].points[0]  # [Lon, Lat]
            id_by_line_unsorted.append(key)
            #name_by_line_unsorted.append(item['Name'])

        self.sort_idx = self.sort_by_location()  # update id_by_line
        self.id_by_line = [id_by_line_unsorted[x] for x in self.sort_idx]
        #self.name_by_line = [name_by_line_unsorted[x] for x in ]
        self.towers = dict()
        for i, tid in enumerate(self.id_by_line):
            item = df_towers.loc[tid, :].copy()
            item['id'] = tid
            item['actual_span'] = actual_span[i]
            self.towers[item['Name']] = self.add_tower(item)
            self.towers[item['Name']].id_sides = self.assign_id_both_sides(i)
            self.towers[item['Name']].id_adj = self.assign_id_adj_towers(i)

    def add_tower(self, df_tower):
        """ add tower to a transmission line """
        name = df_tower['Name']
        if name in self.towers:
            raise KeyError('{} is already assigned'.format(name))
        self.towers[name] = Tower(self.conf, df_tower)

        return self.towers[name]

    def sort_by_location(self):
        """ sort towers by location"""

        idx_sorted = []
        for item in self.coord_line:
            diff = (self.coord_towers - np.ones((self.no_towers, 1)) *
                    item[np.newaxis, :])
            temp = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
            idx = np.argmin(temp)
            tf = np.allclose(temp[idx], 0.0, 1.0e-4)
            if not tf:
                raise ValueError('Can not locate the tower in {}'.
                                 format(self.df_line['LineRoute'].values[0]))
            idx_sorted.append(idx)

        if self.conf.flag_figure:
            plt.figure()
            plt.plot(self.coord_line[:, 0],
                     self.coord_line[:, 1], 'ro-',
                     self.coord_towers[idx_sorted, 0],
                     self.coord_towers[idx_sorted, 1], 'b-')
            plt.title(self.df_line['LineRoute'].values[0])

        return idx_sorted

    def assign_id_both_sides(self, idx):
        """ assign id of towers on both sides"""

        if idx == 0:
            return (-1, self.id_by_line[idx + 1])
        elif idx == self.no_towers - 1:
            return (self.id_by_line[idx - 1], -1)
        else:
            return (self.id_by_line[idx - 1], self.id_by_line[idx + 1])

    def assign_id_adj_towers(self, idx):
        """ assign id of adjacent towers which can be influenced by collapse
        """

        tid = self.id_by_line[idx]
        max_no_adj_towers = self.towers[self.id2name[tid]].max_no_adj_towers

        list_left = self.create_list_idx(idx, max_no_adj_towers, -1)
        list_right = self.create_list_idx(idx, max_no_adj_towers, 1)

        return list_left[::-1] + [tid] + list_right

    def create_list_idx(self, idx, nsteps, flag_direction):
        """ create list of adjacent towers in each direction (flag=+/-1)
        """
        list_tid = []
        for i in range(nsteps):
            idx += flag_direction
            if idx < 0 or idx > self.no_towers-1:
                list_tid.append(-1)
            else:
                list_tid.append(self.id_by_line[idx])
        return list_tid

    def mod_list_idx(self, list_idx):
        """
        replace id of strain tower with -1
        """
        for i, tid in enumerate(list_idx):
            if tid >= 0:
                funct_ = self.towers[self.id2name[tid]].funct
                if funct_ in self.conf.flag_strainer:
                    list_idx[i] = -1
        return list_idx

        if self.conf.flag_strainer:
            list_left = mod_list_idx(list_left)
            list_right = mod_list_idx(list_right)


    def calculate_distance_between_towers(self):

        dist_forward = []
        for i in range(self.no_towers - 1):
            pt0 = (self.coord_line[i, 1], self.coord_line[i, 0])
            pt1 = (self.coord_line[i+1, 1], self.coord_line[i+1, 0])
            dist_forward.append(great_circle(pt0, pt1).meters)

        actual_span = []
        for i in range(self.no_towers):
            if i == 0:
                actual_span.append(0.5 * dist_forward[i])
            elif i == self.no_towers - 1:
                actual_span.append(0.5 * dist_forward[i-1])
            else:
                actual_span.append(0.5 * (dist_forward[i-1]+dist_forward[i]))

        return actual_span