__author__ = 'sudipta'

import unittest
import numpy as np
import pandas as pd
import os, sys

from transmission.config_class import TransmissionConfig
from transmission.sim_towers_v13_2 import sim_towers
from transmission.read import distance
from transmission.read import TransmissionNetwork
from transmission.event import Event


class TestTransmission(unittest.TestCase):

    def test_transmission(self):
        conf = TransmissionConfig(test=1)
        dir_output = conf.dir_output

        # read GIS information
        network = TransmissionNetwork(conf)
        tower, sel_lines, fid_by_line, fid2name, lon, lat = network.read_tower_gis_information(conf)

        try:
            tf_sim_all, prob_sim_all, est_ntower_all, prob_ntower_all, \
                                    est_ntower_nc_all, prob_ntower_nc_all, sel_lines = sim_towers(conf)
        except ValueError:
            self.assertEquals(True, False, 'Something went wrong in function {}'.format(
                network.read_tower_gis_information.__name__))
            return

        for line in sel_lines:
            for (ds, _) in conf.ds_list:
                try:
                    self.check_file_consistency(dir_output, ds, est_ntower_all, est_ntower_nc_all, line,
                                                prob_ntower_all, prob_ntower_nc_all, prob_sim_all, tf_sim_all)
                except IOError:
                    conf.flag_save = 1  # if the test files don't exist, e.g., when run for the fist time
                    tf_sim_all, prob_sim_all, est_ntower_all, prob_ntower_all, \
                                    est_ntower_nc_all, prob_ntower_nc_all, sel_lines = sim_towers(conf)
                    self.check_file_consistency(dir_output, ds, est_ntower_all, est_ntower_nc_all, line,
                                                prob_ntower_all, prob_ntower_nc_all, prob_sim_all, tf_sim_all)

    def check_file_consistency(self, dir_output, ds, est_ntower_all, est_ntower_nc_all, line, prob_ntower_all,
                               prob_ntower_nc_all, prob_sim_all, tf_sim_all):
        npy_file = dir_output + "/tf_line_mc_" + ds + '_' + line.replace(' - ', '_') + ".npy"
        tf_sim_test = np.load(npy_file)
        np.testing.assert_array_equal(tf_sim_test, tf_sim_all[line][ds])
        csv_file = dir_output + "/pc_line_mc_" + ds + '_' + line.replace(' - ', '_') + ".csv"
        prob_sim_test = pd.read_csv(csv_file, names=prob_sim_all[line][ds].columns, header=False)  # dataframe
        np.testing.assert_array_almost_equal(prob_sim_test.as_matrix(), prob_sim_all[line][ds].as_matrix())
        csv_file = dir_output + "/est_ntower_" + ds + '_' + line.replace(' - ', '_') + ".csv"
        est_ntower_test = pd.read_csv(csv_file, names=est_ntower_all[line][ds].columns, header=False)
        np.testing.assert_array_almost_equal(est_ntower_test.as_matrix(), est_ntower_all[line][ds].as_matrix())
        npy_file = dir_output + "/prob_ntower_" + ds + '_' + line.replace(' - ', '_') + ".npy"
        prob_ntower_test = np.load(npy_file)
        self.assertEqual(np.array_equal(prob_ntower_test, prob_ntower_all[line][ds]), 1)
        csv_file = dir_output + "/est_ntower_nc_" + ds + '_' + line.replace(' - ', '_') + ".csv"
        est_ntower_nc_test = pd.read_csv(csv_file, names=est_ntower_all[line][ds].columns, header=False)
        np.testing.assert_array_almost_equal(est_ntower_nc_test.as_matrix(),
                                             est_ntower_nc_all[line][ds].as_matrix())
        npy_file = dir_output + "/prob_ntower_nc_" + ds + '_' + line.replace(' - ', '_') + ".npy"
        prob_ntower_nc_test = np.load(npy_file)
        self.assertEqual(np.array_equal(prob_ntower_nc_test, prob_ntower_nc_all[line][ds]), 1)


class TestTransmissionConfig(unittest.TestCase):
    def test_whether_config_is_test(self):
        conf = TransmissionConfig(test=1)
        self.assertEqual(conf.test, 1)
        self.assertEqual(conf.flag_save, 0)
        self.assertEqual(conf.nsims, 20)

        conf1 = TransmissionConfig(test=0)
        self.assertEqual(conf1.test, 0)


class TestReadDotPy(unittest.TestCase):
    '''
    Tests Hyeuk's distance function with that of geopy.distance.great_circle
    Hyeuk's function is not used in the code anymore. Instead the geopy.distance.great_circle is used.
    '''
    def test_distance(self):
        from geopy.distance import great_circle
        newport_ri = (41.49008, -71.312796)
        cleveland_oh = (41.499498, -81.695391)
        self.assertAlmostEqual(distance(newport_ri, cleveland_oh), great_circle(newport_ri, cleveland_oh).kilometers,
                               places=0)

if __name__ == '__main__':
    unittest.main()
