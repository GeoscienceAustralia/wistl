#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
#import numpy as np
import pandas as pd
import os
WISTL = os.environ['WISTL']

from wistl.config_class import TransmissionConfig
from wistl.sim_towers import sim_towers
#from pandas.util.testing import assert_frame_equal


class TestTransmission(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.conf = TransmissionConfig(os.path.join(WISTL, 'tests', 'test.cfg'))
        cls.damaged_networks = sim_towers(cls.conf)

    @classmethod
    def h5file_full(cls, damage_line, str_head):

        h5file = os.path.join(
            cls.conf.path_output,
            damage_line.event_id,
            '{}_{}.h5'.format(str_head, damage_line.name_output))

        return h5file

    def test_transmission_analytical(self):

        if self.conf.analytical:
            for network in self.damaged_networks.itervalues():
                for line in network.lines.itervalues():

                    self.check_file_consistency_analytical(line)

    def test_transmission_simulation(self):

        if self.conf.simulation:
            for network in self.damaged_networks.itervalues():
                for line in network.lines.itervalues():

                    self.check_file_consistency_simulation(line)

    def test_transmission_simulation_non_cascading(self):

        if not self.conf.skip_non_cascading_collapse:
            for network in self.damaged_networks.itervalues():
                for line in network.lines.itervalues():

                    self.check_file_consistency_simulation_non_cascading(line)

    def test_transmission_analytical_vs_simulation_non_cascading(self):

        if not self.conf.skip_non_cascading_collapse and self.conf.analytical:
            for network in self.damaged_networks.itervalues():
                for line in network.lines.itervalues():

                    self.compare_anlytical_vs_simulation_non_cascading(line)

    def check_file_consistency_analytical(self, damage_line):

        h5file = self.h5file_full(damage_line, 'damage_prob_analytical')

        for ds in self.conf.damage_states:
            df_value = pd.read_hdf(h5file, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.damage_prob_analytical[ds])

    def check_file_consistency_simulation(self, damage_line):

        h5file_damage = self.h5file_full(damage_line, 'damage_prob_simulation')
        h5file_est = self.h5file_full(damage_line, 'est_no_damage_simulation')
        h5file_prob = self.h5file_full(damage_line, 'prob_no_damage_simulation')

        for ds in self.conf.damage_states:

            df_value = pd.read_hdf(h5file_damage, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.damage_prob_simulation[ds])

            df_value = pd.read_hdf(h5file_est, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.est_no_damage_simulation[ds])

            df_value = pd.read_hdf(h5file_prob, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.prob_no_damage_simulation[ds])

    def check_file_consistency_simulation_non_cascading(self, damage_line):

        h5file1 = os.path.join(
            self.conf.path_output,
            damage_line.event_id,
            'damage_prob_simulation_non_cascading_{}.h5'.format(damage_line.name_output))

        h5file2 = os.path.join(
            self.conf.path_output,
            damage_line.event_id,
            'est_no_damage_simulation_non_cascading_{}.h5'.format(damage_line.name_output))

        h5file3 = os.path.join(
            self.conf.path_output,
            damage_line.event_id,
            'prob_no_damage_simulation_non_cascading_{}.h5'.format(damage_line.name_output))

        for ds in self.conf.damage_states:

            df_value = pd.read_hdf(h5file1, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.damage_prob_simulation_non_cascading[ds])

            df_value = pd.read_hdf(h5file2, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.est_no_damage_simulation_non_cascading[ds])

            df_value = pd.read_hdf(h5file3, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.prob_no_damage_simulation_non_cascading[ds])

    def compare_anlytical_vs_simulation_non_cascading(self, damage_line):

        for ds in self.conf.damage_states:

            pd.util.testing.assert_frame_equal(
                damage_line.damage_prob_analytical[ds],
                damage_line.damage_prob_simulation_non_cascading[ds])


# class TestTransmissionConfig(unittest.TestCase):

#     def test_whether_config_is_test(self):

#         path_ = '/'.join(__file__.split('/')[:-1])
#         conf = TransmissionConfig(os.path.join(path_, 'test.cfg'))

#         self.assertEqual(conf.test, 1)
#         self.assertEqual(conf.save, 0)
#         self.assertEqual(conf.nsims, 20)

if __name__ == '__main__':
    unittest.main()
