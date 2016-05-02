#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
import pandas as pd
import numpy as np
import os

from wistl.config import TransmissionConfig
from wistl.sim_towers import sim_towers

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class TestTransmission(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = TransmissionConfig(os.path.join(BASE_DIR, 'test.cfg'))
        cls.cfg.save = False

        cls.damaged_networks = sim_towers(cls.cfg)
        # if cls.cfg.parallel:
        #     cls.damaged_networks, cls.damaged_lines = \
        #         sim_towers(cls.cfg)
        # else:
        #     cls.damaged_networks, _ = sim_towers(cls.cfg)

    @classmethod
    def h5file_full(cls, damage_line, str_head):

        h5file = os.path.join(
            cls.cfg.path_output,
            damage_line.event_id_scale,
            '{}_{}.h5'.format(str_head, damage_line.name_output))

        return h5file

    def test_transmission_analytical(self):

        if self.cfg.analytical:
            for network in self.damaged_networks:
                for line in network.itervalues():
                    self.check_file_consistency_analytical(line)

    def test_transmission_simulation(self):

        if self.cfg.simulation:

            for network in self.damaged_networks:
                for line in network.itervalues():
                    self.check_file_consistency_simulation(line)

            # if self.cfg.parallel:
            #     for line in self.damaged_lines:
            #         self.check_file_consistency_simulation(line)
            # else:
            #     for network in self.damaged_networks:
            #         for line in network.lines.itervalues():
            #             self.check_file_consistency_simulation(line)

    def test_transmission_simulation_non_cascading(self):

        if not self.cfg.skip_non_cascading_collapse:

            for network in self.damaged_networks:
                for line in network.itervalues():
                    self.check_file_consistency_simulation_non_cascading(line)

            # if self.cfg.parallel:
            #     for line in self.damaged_lines:
            #         self.check_file_consistency_simulation_non_cascading(line)
            # else:
            #     for network in self.damaged_networks:
            #         for line in network.lines.itervalues():
            #
            #             self.check_file_consistency_simulation_non_cascading(line)

    def test_transmission_analytical_vs_simulation_only_isolation(self):

        for network in self.damaged_networks:
            for line in network.itervalues():
                for tower in line.towers.itervalues():
                    self.compare_analytical_vs_simulation_for_collapse(tower)

        # if self.cfg.parallel:
        #     for line in self.damaged_lines:
        #         for tower in line.towers.itervalues():
        #             self.compare_analytical_vs_simulation_for_collapse(tower)
        # else:
        #     for network in self.damaged_networks:
        #         for line in network.lines.itervalues():
        #             for tower in line.towers.itervalues():
        #                 self.compare_analytical_vs_simulation_for_collapse(tower)

    def check_file_consistency_analytical(self, damage_line):

        h5file = self.h5file_full(damage_line, 'damage_prob_analytical')

        for ds in self.cfg.damage_states:
            df_value = pd.read_hdf(h5file, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.damage_prob_analytical[ds])

    def check_file_consistency_simulation(self, damage_line):

        h5file_damage = self.h5file_full(damage_line, 'damage_prob_simulation')
        h5file_est = self.h5file_full(damage_line, 'est_no_damage_simulation')
        h5file_prob = self.h5file_full(damage_line, 'prob_no_damage_simulation')

        for ds in self.cfg.damage_states:

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
            self.cfg.path_output,
            damage_line.event_id_scale,
            'damage_prob_simulation_non_cascading_{}.h5'.format(damage_line.name_output))

        h5file2 = os.path.join(
            self.cfg.path_output,
            damage_line.event_id_scale,
            'est_no_damage_simulation_non_cascading_{}.h5'.format(damage_line.name_output))

        h5file3 = os.path.join(
            self.cfg.path_output,
            damage_line.event_id_scale,
            'prob_no_damage_simulation_non_cascading_{}.h5'.format(damage_line.name_output))

        for ds in self.cfg.damage_states:

            df_value = pd.read_hdf(h5file1, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.damage_prob_simulation_non_cascading[ds])

            df_value = pd.read_hdf(h5file2, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.est_no_damage_simulation_non_cascading[ds])

            df_value = pd.read_hdf(h5file3, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.prob_no_damage_simulation_non_cascading[ds])

    def compare_analytical_vs_simulation_for_collapse(self, tower):

        for key, grouped in tower.damage_isolation_mc['collapse'].groupby('id_time'):
            np.testing.assert_almost_equal(
                len(grouped)/float(self.cfg.no_sims),
                tower.prob_damage_isolation.ix[key, 'collapse'],
                decimal=1)

# class TestTransmissionConfig(unittest.TestCase):

#     def test_whether_config_is_test(self):

#         path_ = '/'.join(__file__.split('/')[:-1])
#         cfg = TransmissionConfig(os.path.join(path_, 'test.cfg'))

#         self.assertEqual(cfg.test, 1)
#         self.assertEqual(cfg.save, 0)
#         self.assertEqual(cfg.nsims, 20)

if __name__ == '__main__':
    unittest.main()
