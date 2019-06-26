#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
import pandas as pd
import os

from wistl.config import Config
from wistl.main import run_simulation

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

"""
class TestTransmission(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(os.path.join(BASE_DIR,
                                                  'test_line_interaction.cfg'))
        cls.cfg.save = False
        cls.cfg.figure = False
        cls.damaged_networks = run_simulation(cls.cfg)

    @classmethod
    def h5file_full(cls, damage_line, str_head):

        h5file = os.path.join(damage_line.cfg.path_output,
                              damage_line.event_id_scale,
                              '{}_{}.h5'.format(str_head,
                                                damage_line.name_output))

        return h5file

    def test_transmission_simulation_interaction(self):

        if self.cfg.simulation:

            for network in self.damaged_networks:
                for _, line in network.items():

                    self.check_file_consistency_simulation(line)

    def check_file_consistency_simulation(self, damage_line):

        h5file_damage = self.h5file_full(damage_line,
                                         'damage_prob_line_interaction')
        h5file_est = self.h5file_full(damage_line,
                                      'est_no_damage_line_interaction')
        h5file_prob = self.h5file_full(damage_line,
                                       'prob_no_damage_line_interaction')

        for ds in self.cfg.damage_states:

            df_value = pd.read_hdf(h5file_damage, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.damage_prob_line_interaction[ds])

            df_value = pd.read_hdf(h5file_est, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.est_no_damage_line_interaction[ds])

            df_value = pd.read_hdf(h5file_prob, ds)
            pd.util.testing.assert_frame_equal(
                df_value, damage_line.prob_no_damage_line_interaction[ds])
"""

if __name__ == '__main__':
    unittest.main()
