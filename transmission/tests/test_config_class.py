
__author__ = 'sudipta'

import unittest
import numpy as np
import pandas as pd
import os
import inspect

from transmission.config_class import TransmissionConfig
#from transmission.sim_towers import sim_towers
#from transmission.read import distance
#from transmission.read import TransmissionNetwork

class TestTransmissionConfig(unittest.TestCase):

    def setUp(self):

        path_ = '/'.join(__file__.split('/')[:-1])
        self.conf = TransmissionConfig(os.path.join(path_, 'test.cfg'))


#         self.input_path = os.path.join(self.proj_path, 'data/input')
#         self.test_data_path = os.path.join(self.proj_path, 'data/test_data')
        self.adjust_by_topo_file = "adjust_by_topo.txt"
#         self.drag_height_by_type_file = "drag_height_by_type.csv"
#         self.topo_value_file = "test_topo_value.csv"
#         self.terrain_height_file = "test_terrain_height.csv"
#         self.design_value_file = "test_design_value.csv"
#         self.frag_file = "test_fragility.csv"
#         self.cond_prob_file = "test_cond_collapse_prob.csv"

    def test_config(self):

        # run_type
        self.assertEqual(self.conf.test, True)
        self.assertEqual(self.conf.parallel, True)
        self.assertEqual(self.conf.save, False)
        self.assertEqual(self.conf.figure, True)

        # run_parameters
        self.assertEqual(self.conf.nsims, 20)
        self.assertEquale(self.conf.analytical, True)
        self.assertEqual(self.conf.simulation, True)
        self.assertEqual(self.conf.skip_non_cascading_collapse, False)
        self.assertEqual(self.conf.adjust_design_by_topo, True)
        self.assertEquale(self.conf.strainer, ['Strainer', 'dummy'])

        # directories
        self.assertEqual(self.conf.path_wind_scenario, )

[directories]
project: /Users/hyeuk/Projects/mctlda
gis_data: %(project)/gis_data
wind_scenario: %(project)/wind_scenario/glenda_reduced
#               %(project)/wind_scenario/scenario50yr,
#               %(project)/wind_scenario/scenario100yr,
#               %(project)/wind_scenario/scenario200yr
input: %(project)/input
output: %(project)/test_output_current_glenda

# shape files are located in the gis_data directory
[gis_data]
shape_tower: Towers_with_extra_strainers_WGS84.shp
shape_line: Lines_NGCP_with_synthetic_attributes_WGS84.shp

# wind scenario files are located in the wind_scenario directory
[wind_scenario]
file_name_format: ts.%(tower_name).csv

# The files below are located in the input directory.
[input]
design_value: design_value_current.csv
fragility: fragility_GA.csv
conditional_collapse_probability: cond_collapse_prob_NGCP.csv
terrain_height_multiplier: terrain_height_multiplier.csv
drag_height_by_type: drag_height_by_type.csv
# only required if adjust_design_by_topography is True
topographic_multiplier: topo_value_scenario_50yr.csv
design_adjustment_factor_by_topography: adjust_design_by_topo.txt






    def test_read_drag_height_by_type(self):
        expected = {
            'Strainer': 12.2,
            'Suspension': 15.4,
            'Terminal': 12.2}
        for item in expected.keys():
            np.testing.assert_allclose(self.conf.drag_height[item],
                                       expected[item], atol=1.0e-4)

#     def test_read_adjust_by_topo(self):
#         expected = {
#             'threshold': np.array([1.05, 1.1, 1.2, 1.3, 1.45]),
#             0: 1.0,
#             1: 1.1,
#             2: 1.2,
#             3: 1.3,
#             4: 1.45,
#             5: 1.6}
#         result = read_adjust_by_topo(os.path.join(self.input_path,
#                                      self.adjust_by_topo_file))

#         for item in expected.keys():
#             try:
#                 message = 'Expecting %s, but it returns %s' % (
#                     expected[item], result[item])
#                 self.assertEqual(expected[item], result[item], message)
#             except ValueError:
#                 np.testing.assert_allclose(result[item],
#                                            expected[item], atol=1.0e-4)

#     def test_read_topo_value(self):
#         expected = {
#             'AC-001': 1.00832,
#             'AC-002': 1.12596,
#             'BM-102': 1.00000
#             }
#         result = read_topo_value(os.path.join(self.test_data_path,
#                                               self.topo_value_file))
#         message = 'Expecting %s, but it returns %s' % (
#             expected, result)
#         self.assertEqual(expected, result, message)

#     def test_read_terrain_height_multiplier(self):
#         expected = {
#             'height': [3, 200],
#             'tc1': [0.99, 1.32],
#             'tc2': [0.91, 1.29],
#             'tc3': [0.83, 1.24],
#             'tc4': [0.75, 1.16]
#             }
#         result = read_terrain_height_multiplier(os.path.join(
#             self.test_data_path, self.terrain_height_file))
#         message = 'Expecting %s, but it returns %s' % (
#             expected, result)
#         self.assertEqual(expected, result, message)

#     def test_read_design_value(self):
#         expected = {
#             'Batangas - Makban': {
#                 'cat': 2,
#                 'level': 'low',
#                 'span': 400.0,
#                 'speed': 75.0
#                 },
#             'Calaca - Amadeo': {
#                 'cat': 2,
#                 'level': 'low',
#                 'span': 400.0,
#                 'speed': 75.0
#                 }
#             }
#         (_, result) = read_design_value(os.path.join(
#             self.test_data_path, self.design_value_file))
#         message = 'Expecting %s, but it returns %s' % (
#             expected, result)
#         self.assertEqual(expected, result, message)

#     def test_read_frag(self):
#         expected_frag = collections.OrderedDict([(
#             'Steel Pole', {
#                 'Terminal': {
#                     1: {
#                         'collapse': {
#                             'cdf': 'lognorm',
#                             'param1': 0.05,
#                             'idx': 2,
#                             'param0': 4.02
#                             },
#                         'minor': {
#                             'cdf': 'lognorm',
#                             'param1': 0.05,
#                             'idx': 1,
#                             'param0': 3.85
#                             }
#                         }
#                     }
#                 }
#         ), (
#             'Lattice Tower', {
#                 'Suspension': {
#                     1: {
#                         'collapse': {
#                             'cdf': 'lognorm',
#                             'param1': 0.03,
#                             'idx': 2,
#                             'param0': 1.05
#                             },
#                         'minor': {
#                             'cdf': 'lognorm',
#                             'param1': 0.02,
#                             'idx': 1,
#                             'param0': 1.02
#                             }
#                         }
#                     }
#                 }
#             )])
#         expected_ds_list = [('minor', 1), ('collapse', 2)]
#         expected_dev_angle = np.array([0, 360])

#         (result_frag, result_ds_list, _) = read_frag(os.path.join(
#             self.test_data_path, self.frag_file))

#         # problem with numpy value
#         np.testing.assert_allclose(
#             result_frag['Steel Pole']['Terminal']['dev_angle'],
#             expected_dev_angle, atol=1.0e-4)
#         np.testing.assert_allclose(
#             result_frag['Lattice Tower']['Suspension']['dev_angle'],
#             expected_dev_angle, atol=1.0e-4)

#         result_frag['Steel Pole']['Terminal'].pop('dev_angle', None)
#         result_frag['Lattice Tower']['Suspension'].pop('dev_angle', None)

#         message = 'Expecting %s, but it returns %s' % (
#             expected_frag, result_frag)
#         self.assertDictEqual(expected_frag, result_frag, message)
#         message = 'Expecting %s, but it returns %s' % (
#             expected_ds_list, result_ds_list)
#         self.assertEqual(expected_ds_list, result_ds_list, message)

#     def test_read_cond_prob(self):
#         expected = {
#             'Strain': {
#                 'low': {
#                     'max_adj': 6,
#                     'prob': {
#                         (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6): 0.62,
#                         (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5): 0.05,
#                         (-4, -3, -2, -1, 0, 1, 2, 3, 4): 0.08,
#                         (-3, -2, -1, 0, 1, 2, 3): 0.1,
#                         (-2, -1, 0, 1, 2): 0.08,
#                         (-1, 0, 1): 0.05
#                         }
#                     },
#                 'threshold': 'containment'
#             },
#             'Suspension': {
#                 'lower': {
#                     'max_adj': 2,
#                     'prob': {
#                         (-2, -1, 0, 1): 0.025,
#                         (-2, -1, 0, 1, 2): 0.1,
#                         (-1, 0): 0.075,
#                         (-1, 0, 1): 0.35,
#                         (-1, 0, 1, 2): 0.025,
#                         (0, 1): 0.075
#                         }
#                     },
#                 'threshold': '40'
#                 }
#             }
#         result = read_cond_prob(os.path.join(
#             self.test_data_path, self.cond_prob_file))
#         message = 'Expecting %s, but it returns %s' % (
#             expected, result)
#         self.assertDictEqual(expected, result, message)

#     def test_compute_distance(self):
#         # http://www.movable-type.co.uk/scripts/latlong.html
#         origin = (38.898556, -77.037852)  # lat, lon
#         destination = (38.897147, -77.043934)  # lat, lon
#         expected = 0.5492
#         result = compute_distance(origin, destination)
#         message = 'Expecting %s, but it returns %s' % (
#             expected, result)
#         self.assertAlmostEqual(expected, result, places=3, msg=message)

#     # def test_dir_wind_speed(self):
#     #     expected =
#     #     (shapes, records, fields) = dir_wind_speed(self.shape_file)

#     def test_convert_10_to_z(self):
#         terrain_height = {
#             'height': [3, 10, 200],
#             'tc1': [0.99, 1.12, 1.32],
#             'tc2': [0.91, 1.0, 1.29],
#             'tc3': [0.83, 0.83, 1.24],
#             'tc4': [0.75, 0.75, 1.16]
#             }
#         expected = np.array([0.967, 0.974, 1.0, 1.0])
#         result = []
#         for i in range(1, 5):
#             asset = Tower(0, 'Lattice Tower', 'Strain', 'line_route', 0.0, 0.0,
#                           0.0, i, 0.0, 0.0, 30.0, 8.0)
#             result.append(convert_10_to_z(asset, terrain_height))
#         result = np.array(result)
#         np.testing.assert_allclose(expected, result, atol=1.0e-3)

#     def test_read_velocity_profile(self):
#         tower = {'8DZ-001': Tower(0, 'Lattice Tower', 'Strain', 'line_route',
#                                   0.0, 0.0, 0.0, 1, 0.0, 0.0, 30.0, 8.0)}
#         result = read_velocity_profile(
#             Event,
#             self.test_data_path,
#             tower,
#             os.path.join(self.test_data_path, self.terrain_height_file))

#         print result['8DZ-001'].wind

#         #print result.convert_factor

#         #print(result[name].wind)



#     def test_config_is_test(self):
#         conf = TransmissionConfig(cfg_file='./transmission/tests/test.cfg')
#         self.assertEqual(conf.test, 1)
#         #self.assertEqual(conf.flag_save, 0)
#         self.assertEqual(conf.nsims, 20)

#         #conf1 = TransmissionConfig(test=0)
#         #self.assertEqual(conf1.test, 0)


# # class TestReadDotPy(unittest.TestCase):
# #     '''
# #     Tests Hyeuk's distance function with that of geopy.distance.great_circle
# #     Hyeuk's function is not used in the code anymore. Instead the geopy.distance.great_circle is used.
# #     '''
# #     def test_distance(self):
# #         from geopy.distance import great_circle
# #         newport_ri = (41.49008, -71.312796)
# #         cleveland_oh = (41.499498, -81.695391)
# #         self.assertAlmostEqual(distance(newport_ri, cleveland_oh), great_circle(newport_ri, cleveland_oh).kilometers,
# #                                places=0)

if __name__ == '__main__':
    unittest.main()
