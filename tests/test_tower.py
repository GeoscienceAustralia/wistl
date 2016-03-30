#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
import pandas as pd
import os
import numpy as np
import copy

from wistl.config_class import TransmissionConfig
from wistl.tower import Tower
from test_config_class import assertDeepAlmostEqual

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class TestTower(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.conf = TransmissionConfig(os.path.join(BASE_DIR, 'test.cfg'))

        cls.stower = pd.Series({'id': 1,
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

        cls.tower = Tower(cls.conf, cls.stower)

    def test_get_cond_collapse_prob(self):

        list_function = ['Suspension'] * 2 + ['Terminal'] * 2
        list_value = [20.0, 50.0] * 2

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

        list_function = ['Strainer'] * 2
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
        self.assertEqual(self.tower.file_wind_base_name, expected)

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

        stower = copy.deepcopy(self.stower)
        conf = copy.deepcopy(self.conf)

        conf.design_value[stower['LineRoute']]['span'] = 20.0
        stower['actual_span'] = 10.0
        tower = Tower(conf, stower)

        self.assertEqual(tower.collapse_capacity,
                         tower.design_speed/np.sqrt(0.75))

        # design wind span < actual span
        conf.design_value[stower['LineRoute']]['span'] = 20.0
        stower['actual_span'] = 25.0
        tower = Tower(conf, stower)

        self.assertEqual(tower.collapse_capacity,
                         tower.design_speed/1.0)


    def test_convert_10_to_z(self):

        # terrain category
        # ASNZS 1170.2:2011
        category = [1, 2, 3, 4]
        mzcat10 = [1.12, 1.0, 0.83, 0.75]

        # suspension
        height_z = 15.4   # Suspension
        for cat_, mzcat10_ in zip(category, mzcat10):
            conf = copy.deepcopy(self.conf)
            conf.design_value[self.stower['LineRoute']]['cat'] = cat_
            expected = np.interp(height_z, conf.terrain_multiplier['height'],
                                 conf.terrain_multiplier['tc'+str(cat_)])/mzcat10_
            tower = Tower(conf, self.stower)
            self.assertEqual(tower.convert_factor, expected)

        height_z = 12.2  # Strainer, Terminal
        for cat_, mzcat10_ in zip(category, mzcat10):
            conf = copy.deepcopy(self.conf)
            stower = copy.deepcopy(self.stower)
            stower['Function'] = 'Strainer'
            conf.design_value[stower['LineRoute']]['cat'] = cat_
            expected = np.interp(height_z, conf.terrain_multiplier['height'],
                                 conf.terrain_multiplier['tc'+str(cat_)])/mzcat10_
            tower = Tower(conf, stower)
            self.assertEqual(tower.convert_factor, expected)

    def test_calculate_cond_pc_adj(self):
        """included in test_transmission_line.py"""

if __name__ == '__main__':
    unittest.main()
