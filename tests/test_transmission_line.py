# coding=utf-8
"""
Transmission Line Class Test Cases
"""

from __future__ import print_function

__author__ = 'Hyeuk Ryu'
__date__ = '8/3/2016'

import unittest
import os
import numpy as np
import copy
import pandas as pd

from wistl.config_class import TransmissionConfig
from wistl.transmission_line import TransmissionLine
from test_config_class import assertDeepAlmostEqual
from wistl.transmission_network import read_shape_file, populate_df_lines, \
    populate_df_towers

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class TestTransmissionLine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.conf = TransmissionConfig(os.path.join(BASE_DIR, 'test.cfg'))

        cls.conf.no_sims = 10000
        cls.all_towers = read_shape_file(cls.conf.file_shape_tower)
        cls.all_towers.loc[1, 'Function'] = 'Suspension'
        populate_df_towers(cls.all_towers, cls.conf)

        cls.all_lines = read_shape_file(cls.conf.file_shape_line)
        populate_df_lines(cls.all_lines)

        # Calaca - Amadeo
        df_line = cls.all_lines.loc[0, :]
        tf = cls.all_towers['LineRoute'] == df_line['LineRoute']
        df_towers = cls.all_towers.loc[tf, :]
        cls.line0 = TransmissionLine(cls.conf, df_towers, df_line)

        # a bit of change
        df_towers = copy.deepcopy(cls.all_towers.loc[[3, 4, 5, 0, 2, 1], :])
        df_towers.index = [0, 3, 2, 1, 4, 5]
        df_towers.loc[1, 'Function'] = 'Terminal'
        df_towers.loc[[5, 4, 3, 2], 'Function'] = 'Suspension'
        df_towers.loc[0, 'Function'] = 'Strainer'
        cls.line1 = TransmissionLine(cls.conf, df_towers, df_line)
        # original index is ignored, and then re-assigned based on the line
        # print('{}'.format(df_towers.Function))

    def test_calculate_distance_between_towers(self):

        expected = np.array([[132.99806926573766,
                              276.41651744,
                              288.30116294,
                              333.17949993,
                              330.8356423,
                              142.53885713642302]]).T

        np.testing.assert_almost_equal(
            expected, self.line0.calculate_distance_between_towers())

    def test_run(self):

        # Calaca - Santa Rosa
        id_line = 1
        ps_line = self.all_lines.loc[id_line, :]
        tf = self.all_towers['LineRoute'] == ps_line['LineRoute']
        df_towers = self.all_towers.loc[tf, :]
        df_towers.index = [3, 5, 4, 0, 1, 2]
        line = TransmissionLine(self.conf, df_towers, ps_line)

        expected = dict()
        expected['id_by_line'] = range(6)
        expected['name_by_line'] = ['CB-001', 'CB-002', 'CB-003', 'CB-004',
                                    'CB-005', 'CB-006']
        expected['id2name'] = dict(zip(expected['id_by_line'],
                                       expected['name_by_line']))

        for key, value in expected.iteritems():
            self.assertEqual(value, getattr(line, key))

    def test_sort_by_location(self):

        # Calaca - Amadeo
        self.assertEqual([3, 4, 5, 0, 2, 1], self.line1.sort_by_location())
        self.assertEqual([0, 1, 2, 3, 4, 5], self.line1.id_by_line)
        self.assertEqual(['AC-099',
                          'AC-100',
                          'AC-101',
                          'AC-102',
                          'AC-103',
                          'AC-104'], self.line1.name_by_line)

    def test_assign_id_both_sides(self):

        # Calaca - Amadeo
        expected = {0: (-1, 1), 1: (0, 2), 2: (1, 3),
                    3: (2, 4), 4: (3, 5), 5: (4, -1)}
        for i in range(5):
            outcome = self.line0.assign_id_both_sides(i)
            self.assertEqual(expected[i], outcome)

        for i in range(5):
            outcome = self.line1.assign_id_both_sides(i)
            self.assertEqual(expected[i], outcome)

    def test_assign_id_adj_towers(self):

        # Calaca - Amadeo
        id_line = 0
        df_line = self.all_lines.loc[id_line, :]
        tf = self.all_towers['LineRoute'] == df_line['LineRoute']
        df_towers = copy.deepcopy(self.all_towers.loc[tf, :])
        df_towers.loc[:, 'Function'] = 'Suspension'
        line = TransmissionLine(self.conf, df_towers, df_line)

        expected = {0: [-1, -1, 0, 1, 2],
                    1: [-1, 0, 1, 2, 3],
                    2: [0, 1, 2, 3, 4],
                    3: [1, 2, 3, 4, 5],
                    4: [2, 3, 4, 5, -1],
                    5: [3, 4, 5, -1, -1]}
        for i in line.id_by_line:
            outcome = line.assign_id_adj_towers(i)
            # max_no = line.towers[line.id2name[tid]].max_no_adj_towers
            self.assertEqual(expected[i], outcome)

        # # a bit of change
        expected = {0: [-1, -1, 0, 1, 2],
                    1: [-1, 0, 1, 2, 3],
                    2: [0, 1, 2, 3, 4],
                    3: [-1, -1, -1, 0, 1, 2, 3, 4, 5, -1, -1, -1, -1],
                    4: [2, 3, 4, 5, -1],
                    5: [3, 4, 5, -1, -1]}

        for i in self.line1.id_by_line:
            outcome = self.line1.assign_id_adj_towers(i)
            # print('{}:{}'.format(outcome, expected[i]))
            self.assertEqual(expected[i], outcome)

    def test_create_list_idx(self):

        self.assertEqual(self.line0.create_list_idx(0, 4, +1),
                         [1, 2, 3, 4])
        self.assertEqual(self.line0.create_list_idx(2, 4, -1),
                         [1, 0, -1, -1])

    def test_update_id_adj_by_filtering_strainer(self):

        # tower[3] is strainer
        expected = {0: [-1, -1, 0, 1, 2],
                    1: [-1, 0, 1, 2, -1],
                    2: [0, 1, 2, -1, 4],
                    3: [-1, -1, -1, 0, 1, 2, -1, 4, 5, -1, -1, -1, -1],
                    4: [2, -1, 4, 5, -1],
                    5: [-1, 4, 5, -1, -1]}

        for tower in self.line1.towers.itervalues():
            # print('{}:{}:{}'.format(tower.id,
            #                         tower.id_adj,
            #                         expected[tower.id]))
            self.assertEqual(tower.id_adj, expected[tower.id])

    def test_calculate_cond_pc_adj(self):

        # FIXME: Need to explain the methodology

        """

        - AC-099 : Terminal
        [-1, -1, 1, 5, 4]

        probability          list
          0.075             (0, 1)  (0, 1)
          0.075            (-1, 0)  x
          0.350         (-1, 0, 1)  (0, 1)
          0.025     (-2, -1, 0, 1)  (0, 1)
          0.025      (-1, 0, 1, 2)  (0, 1, 2)
          0.100  (-2, -1, 0, 1, 2)  (0, 1, 2)

          rel_idx: (0, 1, 2), (0,1),
          cum_prob: 0.125, 0.45+0.125

        - AC-100: Suspension
        [-1, 1, 5, 4, -1]
        probability          list
          0.075             (0, 1)  (0, 1)
          0.075            (-1, 0)  (-1, 0)
          0.350         (-1, 0, 1)  (-1, 0, 1)
          0.025     (-2, -1, 0, 1)  (-1, 0, 1)
          0.025      (-1, 0, 1, 2)  (-1, 0, 1)
          0.100  (-2, -1, 0, 1, 2)  (-1, 0, 1)

          rel_idx: (0, 1), (-1, 0), (-1, 0, 1)
          cum_prob: 0.075, 0.075+0.075, 0.5+0.15

        - AC-101: Suspension
        [1, 5, 4, -1, 3]
        probability          list
          0.075             (0, 1)  x
          0.075            (-1, 0)  (-1, 0)
          0.350         (-1, 0, 1)  (-1, 0)
          0.025     (-2, -1, 0, 1)  (-2, -1, 0)
          0.025      (-1, 0, 1, 2)  (-1, 0)
          0.100  (-2, -1, 0, 1, 2)  (-2, -1, 0)

          rel_idx: (-2, -1, 1), (-1, 0)
          cum_prob: 0.125, 0.125+0.45

        - AC-102: Strainer
        [-1, -1, -1, 1, 5, 4, (-1), 3, 2, -1, -1, -1, -1]
        probability                                      list
           0.05                                     (-1, 0, 1) (-1,0,1)
           0.08                              (-2, -1, 0, 1, 2) (-2,-1,0,1,2)
           0.10                       (-3, -2, -1, 0, 1, 2, 3) (-3,-2,-1,0,1,2)
           0.08                (-4, -3, -2, -1, 0, 1, 2, 3, 4) (-3,-2,-1,0,1,2)
           0.05         (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5) (-3,-2,-1,0,1,2)
           0.62  (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6) (-3,-2,-1,0,1,2)

          rel_idx: (-1, 0, 1), (-2, -1, 0, 1, 2), (-3, -2, -1, 0, 1, 2)
          cum_prob: 0.05, 0.08+0.05, 0.85+0.13

        - AC-103: Suspension
        [4, -1, (3), 2, -1]
        probability          list
          0.075             (0, 1)  (0, 1)
          0.075            (-1, 0)  x
          0.350         (-1, 0, 1)  (0, 1)
          0.025     (-2, -1, 0, 1)  (0, 1)
          0.025      (-1, 0, 1, 2)  (0, 1)
          0.100  (-2, -1, 0, 1, 2)  (0, 1)

          rel_idx: (0, 1)
          cum_prob: 0.575

        - AC-104: Suspension
        [-1, 3, (2), -1, -1
        probability          list
          0.075             (0, 1)  x
          0.075            (-1, 0)  (-1, 0)
          0.350         (-1, 0, 1)  (-1, 0)
          0.025     (-2, -1, 0, 1)  (-1, 0)
          0.025      (-1, 0, 1, 2)  (-1, 0)
          0.100  (-2, -1, 0, 1, 2)  (-1, 0)

          rel_idx: (-1, 0)
          cum_prob: 0.575

        """

        expected = {'AC-099': {1: 0.575, 2: 0.125},
                    'AC-100': {-1: 0.075 + 0.5, 1: 0.075 + 0.5},
                    'AC-101': {-2: 0.125, -1: 0.125 + 0.45},
                    'AC-102': {-3: 0.85, -2: 0.08 + 0.85,
                               -1: 0.05 + 0.08 + 0.85,
                               1: 0.05 + 0.08 + 0.85,
                               2: 0.08 + 0.85},
                    'AC-103': {1: 0.575},
                    'AC-104': {-1: 0.575}}

        expected_cond_pc_adj_mc = {
            'AC-099': {'rel_idx': [(0, 1, 2), (0, 1)],
                       'cum_prob': np.array([0.125, 0.45 + 0.125])},
            'AC-100': {'rel_idx': [(0, 1), (-1, 0), (-1, 0, 1)],
                       'cum_prob': np.array([
                           0.075, 0.075 + 0.075, 0.5 + 0.15])},
            'AC-101': {'rel_idx': [(-2, -1, 0), (-1, 0)],
                       'cum_prob': np.array([0.125, 0.125 + 0.45])},
            'AC-102': {'rel_idx': [(-1, 0, 1), (-2, -1, 0, 1, 2),
                                   (-3, -2, -1, 0, 1, 2)],
                       'cum_prob': np.array([0.05, 0.08 + 0.05, 0.85 + 0.13])},
            'AC-103': {'rel_idx': [(0, 1)],
                       'cum_prob': np.array([0.575])},
            'AC-104': {'rel_idx': [(-1, 0)],
                       'cum_prob': np.array([0.575])}}

        for name, val in self.line1.towers.iteritems():
            assertDeepAlmostEqual(self, val.cond_pc_adj, expected[name])
            assertDeepAlmostEqual(self, val.cond_pc_adj_mc,
                                  expected_cond_pc_adj_mc[name])
            # print("{}:{}".format(name, val.cond_pc_adj))
            # print("{}:{}:{}".format(name, val.cond_pc_adj_mc,
            # expected_cond_pc_adj_mc[name]))

    """
    def test_compute_damage_probability_simulation_alt(self):

        event_id = 'test2'
        scale = 2.5
        self.line1.event_tuple = (event_id, scale)

        seed = self.conf.seed[event_id][self.line1.name]
        rnd_state = np.random.RandomState(seed)

        rv = rnd_state.uniform(size=(self.conf.no_sims,
                                     len(self.line1.time_index)))

        # tf_ds = pd.Panel(np.zeros((self.conf.no_sims,
        #                            len(self.line1.time_index),
        #                            self.line1.no_towers), dtype=bool),
        #                  items=range(self.conf.no_sims),
        #                  major_axis=self.line1.time_index,
        #                  minor_axis=self.line1.name_by_line)

        tf_ds = np.zeros((self.line1.no_towers,
                          self.conf.no_sims,
                          len(self.line1.time_index)), dtype=bool)

        for name, tower in self.line1.towers.iteritems():
            tower.determine_damage_isolation_mc(rv)
            tower.determine_damage_adjacent_mc(seed)

            # print('{}'.format(name))
            # print('{}, {}'.format(tower.prob_damage_isolation['collapse'].max(),
            #                   tower.prob_damage_isolation['minor'].max()))
            # print ('{}'.format(tower.damage_mc['collapse'].head()))
            # print ('{}'.format(tower.damage_mc['minor'].head()))

            valid_ = tower.damage_mc['collapse'][
                tower.damage_mc['collapse'].id_adj.notnull()]

            # for key, grouped in tower.damage_mc['collapse'].groupby('id_time'):
            #
            #     np.testing.assert_almost_equal(len(grouped)/float(self.conf.no_sims),
            #                                    tower.prob_damage_isolation.ix[key, 'collapse'],
            #                                    decimal=1)

            # for key, grouped in tower.damage_mc['collapse'].groupby('id_time'):

            for _, item in valid_.iterrows():
                # print('{}'.format(item))
                # print('{}:{}:{}'.format(item['id_sim'],
                #                        item['id_time'],
                #                        item['id_adj']))
                for idx in item['id_adj']:
                    tf_ds[idx, item['id_sim'], item['id_time']] = True

    """
    def test_compute_damage_probability_simulation(self):

        event_id = 'test2'
        scale = 2.5
        self.line1.event_tuple = (event_id, scale)

        seed = self.conf.seed[event_id][self.line1.name]
        rnd_state = np.random.RandomState(seed)

        rv = rnd_state.uniform(size=(self.conf.no_sims,
                                     len(self.line1.time_index)))

        tf_ds = np.zeros((self.line1.no_towers,
                          self.conf.no_sims,
                          len(self.line1.time_index)), dtype=bool)

        # for name, tower in self.line1.towers.iteritems():

        # collapse by adjacent towers
        for name, tower in self.line1.towers.iteritems():

            # tower.compute_mc_adj(rv, seed)
            tower.determine_damage_isolation_mc(rv)
            tower.determine_damage_adjacent_mc(seed)

            print('{}'.format(tower.damage_adjacent_mc))
            for j in tower.damage_adjacent_mc.keys():  # time

                for k in tower.damage_adjacent_mc[j].keys():  # fid

                    id_sim = tower.damage_adjacent_mc[j][k]

                    for l in k:  # each fid

                        try:
                            tf_ds[l, id_sim, j] = True
                        except ValueError:
                            print ('FIXME')
                            print('{}:{}:{}:{}'.format(tower.name,
                                                   tower.damage_adjacent_mc,
                                                   k, tower.cond_pc_adj_mc))
                            pass


if __name__ == '__main__':
    unittest.main()
