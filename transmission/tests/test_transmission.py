__author__ = 'sudipta, Hyeuk Ryu'

import unittest
import numpy as np
import pandas as pd
from os.path import join
from transmission.config_class import TransmissionConfig
from transmission.sim_towers import sim_towers
from transmission.read import read_tower_GIS_information, read_velocity_profile, read_frag
from transmission.tower import Tower


class TestTransmission(unittest.TestCase):

    def test_transmision(self):
        conf = TransmissionConfig(test=1)
        shape_file_tower = conf.shape_file_tower
        shape_file_line = conf.shape_file_line
        file_frag = conf.file_frag
        file_design_value = conf.file_design_value
        file_topo_value= conf.file_topo_value
        dir_output = conf.dir_output
        (tower, sel_lines, fid_by_line, fid2name, lon, lat) = \
            (read_tower_GIS_information(shape_file_tower, shape_file_line,
                                        file_design_value, file_topo_value))

        (frag, ds_list, nds) = read_frag(file_frag)

        (tf_sim_all, prob_sim_all, est_ntower_all, prob_ntower_all,
            est_ntower_nc_all, prob_ntower_nc_all) = sim_towers(conf)

        for line in sel_lines:
            for (ds, _) in ds_list:
                npy_file = dir_output + "/tf_line_mc_" + ds + '_' + line.replace(' - ','_') + ".npy"
                tf_sim_test = np.load(npy_file)
                np.testing.assert_array_equal(tf_sim_test, tf_sim_all[line][ds])
                print "**"

                csv_file = dir_output + "/pc_line_mc_" + ds + '_' + line.replace(' - ','_') + ".csv"
                prob_sim_test = pd.read_csv(csv_file, names=prob_sim_all[line][ds].columns, header=False)  # dataframe
                np.testing.assert_array_almost_equal(prob_sim_test.as_matrix(), prob_sim_all[line][ds].as_matrix())

                csv_file = dir_output + "/est_ntower_" + ds + '_' + line.replace(' - ','_') + ".csv"
                est_ntower_test = pd.read_csv(csv_file, names=est_ntower_all[line][ds].columns, header=False)
                np.testing.assert_array_almost_equal(est_ntower_test.as_matrix(), est_ntower_all[line][ds].as_matrix())

                npy_file = dir_output + "/prob_ntower_" + ds + '_' + line.replace(' - ','_') + ".npy"
                prob_ntower_test = np.load(npy_file)
                self.assertEqual(np.array_equal(prob_ntower_test, prob_ntower_all[line][ds]), 1)

                csv_file = dir_output + "/est_ntower_nc_" + ds + '_' + line.replace(' - ','_') + ".csv"
                est_ntower_nc_test = pd.read_csv(csv_file, names=est_ntower_all[line][ds].columns, header=False)
                np.testing.assert_array_almost_equal(est_ntower_nc_test.as_matrix(), est_ntower_nc_all[line][ds].as_matrix())

                npy_file = dir_output + "/prob_ntower_nc_" + ds + '_' + line.replace(' - ','_') + ".npy"
                prob_ntower_nc_test = np.load(npy_file)
                self.assertEqual(np.array_equal(prob_ntower_nc_test, prob_ntower_nc_all[line][ds]), 1)

    def test_something_else(self):
        self.assertEqual(True, True)


class TestTransmissionConfig(unittest.TestCase):
    def test_whether_config_is_test(self):
        conf = TransmissionConfig(test=1)
        self.assertEqual(conf.test, 1)
        self.assertEqual(conf.flag_save, 0)
        self.assertEqual(conf.nsims, 20)

        conf1 = TransmissionConfig(test=0)
        self.assertEqual(conf1.test, 0)



if __name__ == '__main__':
    unittest.main()
