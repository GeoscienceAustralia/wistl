#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
import pandas as pd
import os
import StringIO
import numpy as np
import copy

from collections import OrderedDict
from transmission.config_class import TransmissionConfig
from transmission.transmission_line import TransmissionLine
from test_config_class import assertDeepAlmostEqual
from transmission.transmission_network import read_shape_file

class TestTransmissionLine(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        path_ = '/'.join(__file__.split('/')[:-1])
        self.conf = TransmissionConfig(os.path.join(path_, 'test.cfg'))

        self.all_towers = read_shape_file(self.conf.file_shape_tower)
        self.all_lines = read_shape_file(self.conf.file_shape_line)

        # Calaca - Amadeo
        id_line = 0
        df_line = self.all_lines.loc[id_line, :]
        tf = self.all_towers['LineRoute']==df_line['LineRoute']
        df_towers = self.all_towers.loc[tf, :]
        self.line0 = TransmissionLine(self.conf, df_towers, df_line)

        # a bit of change
        df_towers = copy.deepcopy(self.all_towers.loc[[3, 4, 5, 0, 2, 1], :])
        df_towers.index = [0, 3, 2, 1, 4, 5]
        df_towers.loc[1, 'Function'] = 'Terminal'
        df_towers.loc[[5, 4, 3, 2], 'Function'] = 'Suspension'
        df_towers.loc[0, 'Function'] = 'Strainer'
        self.line1 = TransmissionLine(self.conf, df_towers, df_line)

    def test_calculate_distance_between_towers(self):

        expected = np.array([132.99806926573766,
                             276.41651744,
                             288.30116294,
                             333.17949993,
                             330.8356423,
                             142.53885713642302])

        np.testing.assert_almost_equal(expected,
            self.line0.calculate_distance_between_towers())

    def test_run(self):

        # Calaca - Santa Rosa
        id_line = 1
        df_line = self.all_lines.loc[id_line, :]
        tf = self.all_towers['LineRoute']==df_line['LineRoute']
        df_towers = self.all_towers.loc[tf, :]
        line = TransmissionLine(self.conf, df_towers, df_line)

        expected = dict()
        expected['sort_idx'] = range(6)
        expected['id_by_line'] = range(6, 12)
        expected['name_by_line'] = ['CB-001', 'CB-002', 'CB-003', 'CB-004',
                                    'CB-005', 'CB-006']

        for key, value in expected.iteritems():
            self.assertEqual(value, getattr(line, key))

    def test_sort_by_location(self):

        # Calaca - Amadeo
        self.assertEqual([3, 5, 4, 0, 1, 2], self.line1.sort_by_location())
        self.assertEqual([1, 5, 4, 0, 3, 2], self.line1.id_by_line)
        self.assertEqual(['AC-099', 'AC-100', 'AC-101', 'AC-102',
                          'AC-103', 'AC-104'], self.line1.name_by_line)

    def test_assign_id_both_sides(self):

        # Calaca - Amadeo
        expected = dict({0: (-1, 1), 1: (0, 2), 2: (1, 3),
                         3: (2, 4), 4: (3, 5), 5: (4, -1)})
        for i in range(5):
            outcome = self.line0.assign_id_both_sides(i)
            self.assertEqual(expected[i], outcome)

        # a bit of change
        expected = dict({3: (4, 3), 0: (-1, 5), 5: (3, -1),
                         4: (0, 2), 2: (5, 0), 1: (1, 4)})
        for i in range(5):
            outcome = self.line1.assign_id_both_sides(i)
            self.assertEqual(expected[i], outcome)

    def test_assign_id_adj_towers(self):

        # Calaca - Amadeo
        id_line = 0
        df_line = self.all_lines.loc[id_line, :]
        tf = self.all_towers['LineRoute']==df_line['LineRoute']
        df_towers = copy.deepcopy(self.all_towers.loc[tf, :])
        df_towers.loc[:, 'Function'] = 'Suspension'
        line = TransmissionLine(self.conf, df_towers, df_line)

        expected = dict({0: [-1, -1, 0, 1, 2],
                         1: [-1,  0, 1, 2, 3],
                         2: [0, 1, 2, 3, 4],
                         3: [1, 2, 3, 4, 5],
                         4: [2, 3, 4, 5, -1],
                         5: [3, 4, 5, -1, -1]})
        for i in range(5):
            tid = line.id_by_line[i]
            outcome = line.assign_id_adj_towers(i)
            max_no = line.towers[line.id2name[tid]].max_no_adj_towers
            self.assertEqual(expected[i], outcome)

        # a bit of change
        # id_by_line: [ 1, 5, 4, 0, 3, 2]
        expected = dict({1: [-1, -1, 1, 5, 4],
                         5: [-1, 1, 5, 4, 0],
                         4: [1, 5, 4, 0, 3],
                         0: [-1, -1, -1, 1, 5, 4, 0, 3, 2, -1, -1, -1, -1],
                         3: [4, 0, 3, 2, -1],
                         2: [0, 3, 2, -1, -1]})
        for i, tid in enumerate(self.line1.id_by_line):
            outcome = self.line1.assign_id_adj_towers(i)
            self.assertEqual(expected[tid], outcome)


    def test_create_list_idx(self):

        self.assertEqual(self.line0.create_list_idx(0, 4, +1),
                         [1, 2, 3, 4])
        self.assertEqual(self.line0.create_list_idx(2, 4, -1),
                         [1, 0, -1, -1])

    def test_update_id_adj_towers(self):

        # id_by_line: [ 1, 5, 4, 0, 3, 2]
        expected = dict({1: [-1, -1, 1, 5, 4],
                         5: [-1, 1, 5, 4, -1],
                         4: [1, 5, 4, -1, 3],
                         0: [-1, -1, -1, 1, 5, 4, -1, 3, 2, -1, -1, -1, -1],
                         3: [4, -1, 3, 2, -1],
                         2: [-1, 3, 2, -1, -1]})

        for key, val in self.line1.towers.iteritems():
            tid = val.id
            self.assertEqual(val.id_adj, expected[tid])

    def test_calculate_cond_pc_adj(self):

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
           0.05                                     (-1, 0, 1) (-1, 0, 1)
           0.08                              (-2, -1, 0, 1, 2) (-2, -1, 0, 1, 2)
           0.10                       (-3, -2, -1, 0, 1, 2, 3) (-3, -2, -1, 0, 1, 2)
           0.08                (-4, -3, -2, -1, 0, 1, 2, 3, 4) (-3, -2, -1, 0, 1, 2)
           0.05         (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5) (-3, -2, -1, 0, 1, 2)
           0.62  (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6) (-3, -2, -1, 0, 1, 2)

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

        expected = dict({'AC-099': {1: 0.575, 2: 0.125},
                         'AC-100': {-1: 0.075+0.5, 1:0.075+0.5},
                         'AC-101': {-2: 0.125, -1: 0.125+0.45},
                         'AC-102': {-3:0.85, -2:0.08+0.85, -1:0.05+0.08+0.85, 1:0.05+0.08+0.85, 2:0.08+0.85},
                         'AC-103': {1: 0.575},
                         'AC-104': {-1: 0.575}})

        expected_cond_pc_adj_mc = dict({
            'AC-099': {'rel_idx': [(0, 1, 2), (0,1)],
                        'cum_prob': np.array([0.125, 0.45+0.125])},
            'AC-100': {'rel_idx': [(0, 1), (-1, 0), (-1, 0, 1)],
                       'cum_prob': np.array([0.075, 0.075+0.075, 0.5+0.15])},
            'AC-101': {'rel_idx': [(-2, -1, 0), (-1, 0)],
                       'cum_prob': np.array([0.125, 0.125+0.45])},
            'AC-102': {'rel_idx': [(-1, 0, 1), (-2, -1, 0, 1, 2), (-3, -2, -1, 0, 1, 2)],
                       'cum_prob': np.array([0.05, 0.08+0.05, 0.85+0.13])},
            'AC-103': {'rel_idx': [(0, 1)],
                       'cum_prob': np.array([0.575])},
            'AC-104': {'rel_idx': [(-1, 0)],
                       'cum_prob': np.array([0.575])}})

        for name, val in self.line1.towers.iteritems():
            assertDeepAlmostEqual(self, val.cond_pc_adj, expected[name])
            assertDeepAlmostEqual(self, val.cond_pc_adj_mc,
                                  expected_cond_pc_adj_mc[name])
            #print("{}:{}".format(name, val.cond_pc_adj))
            #print("{}:{}:{}".format(name, val.cond_pc_adj_mc, expected_cond_pc_adj_mc[name]))


if __name__ == '__main__':
    unittest.main()
