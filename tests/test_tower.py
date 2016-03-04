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
from transmission.tower import Tower
from test_config_class import assertDeepAlmostEqual


class TestTransmissionConfig(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        path_ = '/'.join(__file__.split('/')[:-1])
        self.conf = TransmissionConfig(os.path.join(path_, 'test.cfg'))

        self.stower = pd.Series({'id': 1,
                                 'actual_span': 10.0,
                                 'AxisAz': 134,
                                 'ConstType': 'Unknown',
                                 'DevAngle': 0,
                                 'Function': 'Suspension',
                                 'Height': 17.837622726399999,
                                 'Latitude': 13.938321649600001,
                                 'LineRoute': 'Calaca - Amadeo',
                                 'Longitude': 120.804464197,
                                 'Name': 'AC-099',
                                 'Type': 'Lattice Tower'})

        self.tower = Tower(self.conf, self.stower)

    def test_get_cond_collapse_prob(self):

        list_function = ['Suspension']*2 + ['Terminal']*2
        list_value = [20.0, 50.0]*2

        for funct_, value_ in zip(list_function, list_value):

            stower = copy.deepcopy(self.stower)
            conf = copy.deepcopy(self.conf)

            stower['Function'] = funct_
            stower['Height'] = value_
            tower = Tower(conf, stower)

            if tower.cond_pc:
                tmp = conf.cond_collapse_prob[funct_]
                expected = tmp.set_index('list').to_dict()['probability']
                assertDeepAlmostEqual(self, tower.cond_pc, expected)

        list_function = ['Strainer']*2
        list_value = ['low', 'high']

        for funct_, value_ in zip(list_function, list_value):

            stower = copy.deepcopy(self.stower)
            stower['Function'] = funct_
            conf = copy.deepcopy(self.conf)

            conf.design_value[self.stower['LineRoute']]['level'] = value_
            tower = Tower(conf, stower)

            if tower.cond_pc:
                tmp = conf.cond_collapse_prob[funct_]
                expected = tmp.loc[tmp['design_level'] == value_, :].set_index('list').to_dict()['probability']
                assertDeepAlmostEqual(self, tower.cond_pc, expected)

    def test_get_wind_file(self):

        expected = 'ts.{}.csv'.format(self.stower['Name'])
        self.assertEqual(self.tower.file_wind, expected)

    def test_assign_design_speed(self):

        expected = 75.0 * 1.2
        conf = copy.deepcopy(self.conf)
        conf.adjust_design_by_topo = True
        conf.topo_multiplier[self.stower['Name']] = 1.15
        tower = Tower(conf, self.stower)
        self.assertEqual(tower.design_speed, expected)

        expected = 75.0
        conf.adjust_design_by_topo = False
        tower = Tower(self.conf, self.stower)
        self.assertEqual(tower.design_speed, expected)

    def test_compute_collapse_capacity(self):
        pass

    def test_convert_10_to_z(self):

        cond_collapse_prob_metadata = StringIO.StringIO("""\
[main]
by: Function
list: Suspension, Terminal, Strainer

[Suspension]
by: Height
type: numeric
max_adj: 2
file: ./cond_collapse_prob_suspension_terminal.csv

[Terminal]
by: Height
type: numeric
max_adj: 2
file: ./cond_collapse_prob_suspension_terminal.csv

[Strainer]
by: design_level
type: string
max_adj: 2
file: ./cond_collapse_prob_strainer.csv""")

        cond_collapse_prob_suspension_terminal = StringIO.StringIO("""\
Function, Height_lower, Height_upper, no_collapse, probability, start, end
Suspension, 0, 40, 1, 0.075, 0, 1
Suspension, 0, 40, 1, 0.075, -1, 0
Suspension, 0, 40, 2, 0.35, -1, 1
Suspension, 0, 40, 3, 0.025, -1, 2
Suspension, 0, 40, 3, 0.025, -2, 1
Suspension, 0, 40, 4, 0.10, -2, 2
Terminal, 0, 40, 1, 0.075, 0, 1
Terminal, 0, 40, 1, 0.075, -1, 0
Terminal, 0, 40, 2, 0.35, -1, 1
Terminal, 0, 40, 3, 0.025, -1, 2
Terminal, 0, 40, 3, 0.025, -2, 1
Terminal, 0, 40, 4, 0.10, -2, 2""")

        cond_collapse_prob_strainer = StringIO.StringIO("""\
Function, design_level, no_collapse, probability, start, end
Strainer, low, 2, 0.05, -1, 1
Strainer, low, 4, 0.08, -2, 2
Strainer, low, 6, 0.10, -3, 3
Strainer, low, 8, 0.08, -4, 4
Strainer, low, 10, 0.05, -5, 5
Strainer, low, 12, 0.62, -6, 6
Strainer, high, 2, 0.15, -1, 1
Strainer, high, 4, 0.22, -2, 2
Strainer, high, 6, 0.22, -3, 3
Strainer, high, 8, 0.15, -4, 4
Strainer, high, 10, 0.12, -5, 5
Strainer, high, 12, 0.08, -6, 6""")

        expected_metadata = OrderedDict([('__name__', 'main'),
            ('by', 'Function'),
            ('list', ['Suspension', 'Terminal', 'Strainer']),
            ('Suspension', OrderedDict([('__name__', 'Suspension'),
                ('by', 'Height'),
                ('type', 'numeric'),
                ('max_adj', 2),
                ('file', './cond_collapse_prob_suspension_terminal.csv')])),
            ('Terminal', OrderedDict([('__name__', 'Terminal'),
                ('by', 'Height'),
                ('type', 'numeric'),
                ('max_adj', 2),
                ('file', './cond_collapse_prob_suspension_terminal.csv')])),
            ('Strainer', OrderedDict([('__name__', 'Strainer'),
                ('by', 'design_level'),
                ('type', 'string'),
                ('max_adj', 2),
                ('file', './cond_collapse_prob_strainer.csv')]))])

        with open(self.conf.file_cond_collapse_prob_metadata, 'r') as file1:
            for line1, line2 in zip(file1, cond_collapse_prob_metadata):
                self.assertEqual(line1, line2)

        with open(self.conf.cond_collapse_prob_metadata['Strainer']['file'], 'r') as file1:
            for line1, line2 in zip(file1, cond_collapse_prob_strainer):
                self.assertEqual(line1, line2)

        path_input = os.path.join(os.path.abspath(
            self.conf.file_fragility_metadata), '../')

        for item in expected_metadata['list']:
            expected_metadata[item]['file'] = os.path.abspath(
                os.path.join(path_input, expected_metadata[item]['file']))

        assertDeepAlmostEqual(self, expected_metadata,
                              self.conf.cond_collapse_prob_metadata)

        expected_cond_pc = dict()
        for item in expected_metadata['list']:
            df_tmp = pd.read_csv(expected_metadata[item]['file'], skipinitialspace=1)
            df_tmp['start'] = df_tmp['start'].astype(np.int64)
            df_tmp['end'] = df_tmp['end'].astype(np.int64)
            df_tmp['list'] = df_tmp.apply(lambda x: tuple(range(x['start'], x['end'] + 1)), axis=1)
            expected_cond_pc[item] = df_tmp.loc[df_tmp[expected_metadata['by']] == item]

            pd.util.testing.assert_frame_equal(
                self.conf.cond_collapse_prob[item], expected_cond_pc[item])


    def test_calculate_cond_pc_adj(self):

        terrain_multiplier = StringIO.StringIO("""\
height(m), terrain category 1, terrain category 2, terrain category 3, terrain category 4
3,   0.99,  0.91,  0.83,  0.75
5,   1.05,  0.91,  0.83,  0.75
10,  1.12,  1.00,  0.83,  0.75
15,  1.16,  1.05,  0.89,  0.75
20,  1.19,  1.08,  0.94,  0.75
30,  1.22,  1.12,  1.00,  0.80
40,  1.24,  1.16,  1.04,  0.85
50,  1.25,  1.18,  1.07,  0.90
75,  1.27,  1.22,  1.12,  0.98
100, 1.29,  1.24,  1.16,  1.03
150, 1.31,  1.27,  1.21,  1.11
200, 1.32,  1.29,  1.24,  1.16""")

        with open(self.conf.file_terrain_multiplier, 'r') as file1:
            for line1, line2 in zip(file1, terrain_multiplier):
                self.assertEqual(line1, line2)

        terrain_multiplier.seek(0)
        expected = pd.read_csv(terrain_multiplier, skipinitialspace=True)
        expected.columns = ['height', 'tc1', 'tc2', 'tc3', 'tc4']

        assertDeepAlmostEqual(self, expected.to_dict('list'),
                              self.conf.terrain_multiplier)


    def test_read_drag_height_by_type(self):
        drag_height = StringIO.StringIO("""\
# typical drag height by tower type
Suspension,15.4
Strainer,12.2
Terminal,12.2""")

        with open(self.conf.file_drag_height_by_type, 'r') as file1:
            for line1, line2 in zip(file1, drag_height):
                self.assertEqual(line1, line2)

        drag_height.seek(0)
        expected = pd.read_csv(drag_height, skipinitialspace=True, index_col=0)
        expected.columns = ['value']

        assertDeepAlmostEqual(self, expected['value'].to_dict(),
                              self.conf.drag_height)


    def test_read_topographic_multiplier(self):

        topo_multiplier = StringIO.StringIO("""\
Station,Time,Longitude,Latitude,Speed,UU,VV,Bearing,Pressure,Mh,Mhopp
AC-099,2014-07-16 00:30,120.80446, 13.93832, 55.49, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000
AC-100,2014-07-16 00:30,120.80279, 13.93657, 55.07, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000
AC-101,2014-07-16 00:30,120.80041, 13.93543, 55.20, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000
AC-102,2014-07-16 00:20,120.79799, 13.93429, 57.51, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
AC-103,2014-07-16 00:20,120.79462, 13.93344, 56.63, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
AC-104,2014-07-16 00:20,120.79201, 13.93304, 54.48, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
CB-001,2014-07-16 00:20,120.79210, 13.93267, 54.46, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
CB-002,2014-07-16 00:20,120.79493, 13.93333, 57.53, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
CB-003,2014-07-16 00:20,120.79793, 13.93403, 57.42, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
CB-004,2014-07-16 00:30,120.80082, 13.93542, 55.48, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000
CB-005,2014-07-16 00:30,120.80287, 13.93640, 55.18, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000
CB-006,2014-07-16 00:30,120.80513, 13.93877, 54.56, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000""")

        with open(self.conf.file_topo_multiplier, 'r') as file1:
            for line1, line2 in zip(file1, topo_multiplier):
                self.assertEqual(line1, line2)

        topo_multiplier.seek(0)
        expected = pd.read_csv(topo_multiplier, skipinitialspace=True)
        expected['topo'] = expected[['Mh', 'Mhopp']].max(axis=1)

        # assertDeepAlmostEqual(self, expected.set_index('Station').to_dict()['topo'],
        #                       self.conf.topo_multiplier)

    def test_read_design_adjustment_factor_by_topography_mutliplier(self):

        design_adj_factor = StringIO.StringIO("""\
[main]
threshold: 1.05, 1.1, 1.2, 1.3, 1.45
# key = np.sum(x > threshold), value
0:1.0
1:1.1
2:1.2
3:1.3
4:1.45
5:1.6""")

        expected = {0: 1.0,
                    1: 1.1,
                    2: 1.2,
                    3: 1.3,
                    4: 1.45,
                    5: 1.60,
                    'threshold': np.array([1.05, 1.1, 1.2, 1.3, 1.45])}

        with open(self.conf.file_design_adjustment_factor_by_topo, 'r') as file1:
            for line1, line2 in zip(file1, design_adj_factor):
                self.assertEqual(line1, line2)

        assertDeepAlmostEqual(self, expected,
                              self.conf.design_adjustment_factor_by_topo)

    def test_run(self):

        expected = {'test1': {'Calaca - Amadeo': 11,
                              'Calaca - Santa Rosa': 22},
                    'test2': {'Calaca - Amadeo': 12,
                              'Calaca - Santa Rosa': 23}}

        assertDeepAlmostEqual(self, expected, self.conf.seed)

if __name__ == '__main__':
    unittest.main()
