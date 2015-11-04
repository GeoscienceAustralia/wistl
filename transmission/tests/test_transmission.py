__author__ = 'sudipta'

import unittest
import numpy as np
import pandas as pd
from transmission.config_class import TestTransmissionConfig
from transmission.sim_towers_v13_2 import main
from transmission.read import read_tower_GIS_information, read_velocity_profile, read_frag
from transmission.class_Tower import Tower


class TestTransmission(unittest.TestCase):
    def test_something(self):
        conf = TestTransmissionConfig()

        shape_file_tower = conf.shape_file_tower
        shape_file_line = conf.shape_file_line
        file_frag = conf.file_frag
        file_design_value = conf.file_design_value
        file_topo_value= conf.file_topo_value
        dir_output = conf.dir_output
        (tower, sel_lines, fid_by_line, fid2name, lon, lat) = \
            (read_tower_GIS_information(Tower, shape_file_tower, shape_file_line, file_design_value, file_topo_value))

        (frag, ds_list, nds) = read_frag(file_frag)

        tf_sim_all, prob_sim_all, est_ntower_all, prob_ntower_all, est_ntower_nc_all, prob_ntower_nc_all = main(conf)

        for line in sel_lines:
            for (ds, _) in ds_list:
                npy_file = dir_output + "/tf_line_mc_" + ds + '_' + line.replace(' - ','_') + ".npy"
                tf_sim_test = np.load(npy_file)
                self.assertEqual(np.array_equal(tf_sim_test, tf_sim_all[line][ds]), 1)

                csv_file = dir_output + "/pc_line_mc_" + ds + '_' + line.replace(' - ','_') + ".csv"
                prob_sim_test = pd.read_csv(csv_file)
                self.assertEqual(prob_sim_test.equals(prob_sim_all[line][ds]), 1)

                csv_file = dir_output + "/est_ntower_" + ds + '_' + line.replace(' - ','_') + ".csv"
                est_ntower_test = pd.read_csv(csv_file)
                self.assertEqual(est_ntower_test.equals(est_ntower_all[line][ds]), 1)

                npy_file = dir_output + "/prob_ntower_" + ds + '_' + line.replace(' - ','_') + ".npy"
                prob_ntower_test = np.load(npy_file)
                self.assertEqual(np.array_equal(prob_ntower_test, prob_ntower_all[line][ds]), 1)

                csv_file = dir_output + "/est_ntower_nc_" + ds + '_' + line.replace(' - ','_') + ".csv"
                est_ntower_nc_test = pd.read_csv(csv_file)
                self.assertEqual(est_ntower_nc_test.equals(est_ntower_nc_all[line][ds]), 1)

                npy_file = dir_output + "/prob_ntower_nc_" + ds + '_' + line.replace(' - ','_') + ".npy"
                prob_ntower_nc_test = np.load(npy_file)
                self.assertEqual(np.array_equal(prob_ntower_nc_test, prob_ntower_nc_all[line][ds]), 1)


if __name__ == '__main__':
    unittest.main()
