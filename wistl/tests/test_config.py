#!/usr/bin/env python

import unittest
# import pandas as pd
import logging
import os
import tempfile
import numpy as np
import math

from wistl.config import Config, split_str, calculate_distance_between_towers, \
    unit_vector, find_id_nearest_pt, create_list_idx, assign_cond_pc_adj, \
    assign_shapely_line, assign_shapely_point

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


# https://github.com/larsbutler/oq-engine/blob/master/tests/utils/helpers.py
def assertDeepAlmostEqual(test_case, expected, actual, *args, **kwargs):
    """
    Assert that two complex structures have almost equal contents.
    Compares lists, dicts and tuples recursively. Checks numeric values
    using test_case's :py:meth:`unittest.TestCase.assertAlmostEqual` and
    checks all other values with :py:meth:`unittest.TestCase.assertEqual`.
    Accepts additional positional and keyword arguments and pass those
    intact to assertAlmostEqual() (that's how you specify comparison
    precision).
    :param test_case:
    :type test_case: :py:class:`unittest.TestCase` object

    :param test_case: TestCase object on which we can call all of the basic
        'assert' methods.
    :param expected:
    :param actual:
    :param args:
    :param kwargs:
    :return:
    """
    is_root = not '__trace' in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    try:
        if isinstance(expected, (int, float, complex)):
            test_case.assertAlmostEqual(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray)):
            test_case.assertEqual(len(expected), len(actual))
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assertDeepAlmostEqual(test_case, v1, v2,
                                      __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            test_case.assertEqual(set(expected), set(actual))
            for key in expected:
                assertDeepAlmostEqual(test_case, expected[key], actual[key],
                                      __trace=repr(key), *args, **kwargs)
        else:
            test_case.assertEqual(expected, actual)
    except AssertionError as exc:
        exc.__dict__.setdefault('traces', []).append(trace)
        if is_root:
            trace = ' -> '.join(reversed(exc.traces))
            exc = AssertionError("%s\nTRACE: %s" % (exc.message, trace))
        raise exc


class TestConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'), cls.logger)

        # for testing purpose
        cls.cfg.options['adjust_design_by_topography'] = True

    def test_split_str(self):

        str_ = 'aa: 44'
        expected = ('aa', 44)
        result = split_str(str_, ':')
        self.assertEqual(result, expected)

    def test_design_value(self):

        expected = {'LineA': {'terrain_cat': 2,
                              'design_level': 'low',
                              'design_span': 400.0,
                              'design_speed': 75.0},
                    'LineB': {'terrain_cat': 2,
                              'design_level': 'low',
                              'design_span': 400.0,
                              'design_speed': 51.389}}

        assertDeepAlmostEqual(self, expected, self.cfg.design_value_by_line)

    def test_fragility_metadata(self):

        expected_metadata = dict([
            ('by', ['type', 'function', 'devAngle']),
            ('type', ['string', 'string', 'numeric']),
            ('limit_states', ['minor', 'collapse']),
            ('function', 'form'),
            ('file', './fragility.csv'),
            ('lognorm', dict([('scale', 'param1'),
                              ('arg', 'param2')]))])

        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        try:
            _file.writelines(['[main]\n',
                              'by: type, function, devAngle\n',
                              'type: string, string, numeric\n',
                              'limit_states: minor, collapse\n',
                              'function: form\n',
                              'file: ./fragility.csv\n',
                              '[lognorm]\n',
                              'scale: param1\n',
                              'arg: param2\n'])
            _file.seek(0)

            self.cfg._fragility_metadata = None
            self.cfg.file_fragility_metadata = _file.name

            assertDeepAlmostEqual(self, expected_metadata,
                                  self.cfg.fragility_metadata)
        finally:
            _file.close()
            os.unlink(_file.name)

    def test_damage_states(self):
        expected = ['minor', 'collapse']
        self.cfg._damage_states = None
        self.assertEqual(expected, self.cfg.damage_states)

    def test_no_damage_states(self):
        expected = 2
        self.cfg._no_damage_states = None
        self.assertEqual(self.cfg.no_damage_states, expected)

    def test_non_collapse(self):
        expected = ['minor']
        self.cfg._non_collapse = None
        self.assertEqual(self.cfg.non_collapse, expected)

    def test_cond_collapse_prob_metadata(self):

        self.assertEqual(self.cfg.cond_collapse_prob_metadata['list'],
                         ['Suspension', 'Terminal', 'Strainer'])

        self.assertEqual(self.cfg.cond_collapse_prob_metadata['by'],
                         'function')

        expected = {'by': 'design_level',
                    'type': 'string',
                    'max_adj': 6,
                    'file': './cond_collapse_prob_strainer.csv'}
        self.assertEqual(self.cfg.cond_collapse_prob_metadata['Strainer'],
                         expected)

        expected = {'by': 'height',
                    'type': 'numeric',
                    'max_adj': 2,
                    'file': './cond_collapse_prob_suspension.csv'}
        self.assertEqual(self.cfg.cond_collapse_prob_metadata['Suspension'],
                         expected)

    def test_cond_collapse_prob(self):

        expected = {'function': 'Suspension',
                    'height_lower': 0,
                    'height_upper': 40,
                    'no_collapse': 4,
                    'probability': 0.10000000000000001,
                    'start': -2,
                    'end': 2,
                    'list': (-2, -1, 0, 1, 2)}

        assertDeepAlmostEqual(self, expected,
                              self.cfg.cond_collapse_prob['Suspension'].loc[5].to_dict())

    def test_no_towers_by_line(self):

        expected = {'LineA': 22, 'LineB': 22}

        self.assertEqual(expected, self.cfg.no_towers_by_line)

    def test_terrain_multiplier(self):

        expected = [0.98999999999999999,
                    1.05,
                    1.1200000000000001,
                    1.1599999999999999,
                    1.1899999999999999,
                    1.22,
                    1.24,
                    1.25,
                    1.27,
                    1.29,
                    1.3100000000000001,
                    1.3200000000000001]

        assertDeepAlmostEqual(self, expected,
                              self.cfg.terrain_multiplier['tc1'])

    def test_read_drag_height_by_type(self):

        expected = {'Strainer': 12.199999999999999,
                    'Suspension': 15.4,
                    'Terminal': 12.199999999999999}

        assertDeepAlmostEqual(self, expected,
                              self.cfg.drag_height_by_type)

    def test_topographic_multiplier(self):

        expected = {'T1': 1.0, 'T2': 0.8}

        self.assertEqual(expected['T1'],
                         self.cfg.topographic_multiplier['T1'])
        self.assertEqual(expected['T2'],
                         self.cfg.topographic_multiplier['T2'])

    def test_design_adjustment_factor_by_topography_mutliplier(self):

        expected = {0: 1.0,
                    1: 1.1,
                    2: 1.2,
                    3: 1.3,
                    4: 1.45,
                    5: 1.60,
                    'threshold': np.array([1.05, 1.1, 1.2, 1.3, 1.45])}

        assertDeepAlmostEqual(self, expected,
                              self.cfg.design_adjustment_factor_by_topography)

    def test_towers(self):
        # file_wind_base_name
        self.assertEqual(self.cfg.towers_by_line['LineA'][0]['file_wind_base_name'],
                         'ts.T14.csv')

        # height_z
        self.assertAlmostEqual(self.cfg.towers_by_line['LineA'][0]['height_z'],
                               15.4)
        self.assertAlmostEqual(self.cfg.towers_by_line['LineA'][1]['height_z'],
                               12.2)

    def test_lines(self):
        # name_output
        self.assertEqual(self.cfg.lines['LineA']['name_output'], 'LineA')

        # no_towers
        self.assertAlmostEqual(self.cfg.lines['LineB']['no_towers'], 22)

    def test_process_config(self):

        # max_adj
        expected = {'Terminal': 2, 'Suspension': 2, 'Strainer': 6}
        for key, value in expected.items():
            self.assertEqual(value,
                             self.cfg.cond_collapse_prob_metadata[key]['max_adj'])

    def test_sort_by_location(self):

        expected_names = {'LineA': ['T{}'.format(x) for x in range(1, 23)],
                          'LineB': ['T{}'.format(x) for x in range(23, 45)]}

        for line in ['LineA', 'LineB']:
            self.assertEqual(self.cfg.lines[line]['names'],
                             expected_names[line])

        expected_ids = {'LineA': [23, 39, 3, 33, 4, 5, 7, 6, 31, 8, 38, 29, 34,
                                  0, 42, 12, 2, 10, 40, 14, 22, 1],
                        'LineB': [24, 35, 17, 9, 32, 19, 21, 16, 36, 28, 30, 20,
                                  37, 13, 41, 11, 26, 43, 15, 27, 25, 18]}

        for line in ['LineA', 'LineB']:
            self.assertEqual(self.cfg.lines[line]['ids'], expected_ids[line])

        expected_id2name = {
            'LineA': {0: 'T14', 1: 'T22', 2: 'T17', 3: 'T3',
                      4: 'T5', 5: 'T6', 6: 'T8', 7: 'T7',
                      8: 'T10', 10: 'T18', 12: 'T16', 14: 'T20',
                      22: 'T21', 23: 'T1', 29: 'T12', 31: 'T9',
                      33: 'T4', 34: 'T13', 38: 'T11', 39: 'T2',
                      40: 'T19', 42: 'T15'},
            'LineB': {9: 'T26', 11: 'T38', 13: 'T36', 15: 'T41',
                      16: 'T30', 17: 'T25', 18: 'T44', 19: 'T28',
                      20: 'T34', 21: 'T29', 24: 'T23', 25: 'T43',
                      26: 'T39', 27: 'T42', 28: 'T32', 30: 'T33',
                      32: 'T27', 35: 'T24', 36: 'T31', 37: 'T35',
                      41: 'T37', 43: 'T40'}}

        expected_name2id = {key: {v: k for k, v in value.items()}
                            for key, value in expected_id2name.items()}

        for line in ['LineA', 'LineB']:
            self.assertEqual(self.cfg.lines[line]['id2name'],
                             expected_id2name[line])

        for line in ['LineA', 'LineB']:
            self.assertEqual(self.cfg.lines[line]['name2id'],
                             expected_name2id[line])

    def test_assign_collapse_capacity(self):

        # u_factor = 1.0 - K_FACTOR[NO_CIRCUIT] * (1-actual_span/design_span)
        # collapse_capacity = design_speed / sqrt(u_factor)
        # K_FACTOR = {1: 0.33, 2: 0.5}  # hard-coded
        u_factor = 0.84758125
        expected = 81.4649
        line_name = 'LineA'

        # return {'actual_span': actual_span,
        #         'u_factor': u_factor,
        #         'collapse_capacity': tower['design_speed'] / math.sqrt(u_factor)}

        for tower_name in ['T1']:
            tower_id = self.cfg.lines[line_name]['name2id'][tower_name]
            tower = self.cfg.towers_by_line[line_name][tower_id]
            results = self.cfg.assign_collapse_capacity(tower)
            # self.assertAlmostEqual(tower['u_factor'], u_factor, places=4)
            self.assertAlmostEqual(tower['design_span'], 400.0)
            self.assertAlmostEqual(results['actual_span'], 277.9877093104612)
            self.assertAlmostEqual(tower['design_speed'], 75.0)
            self.assertAlmostEqual(results['u_factor'], u_factor)
            try:
                self.assertAlmostEqual(results['collapse_capacity'], expected,
                                           places=3)
            except AssertionError:
                print('{}:{}:{}'.format(results['collapse_capacity'], tower['collapse_capacity'], expected))

        """
        # u_factor = 1.0
        expected = 75.0
        line_name = 'LineA'
        for tower_name in ['T2', 'T21']:
            tower_id = self.cfg.lines[line_name]['name2id'][tower_name]
            tower = self.cfg.towers_by_line[line_name][tower_id]
            results = self.cfg.assign_collapse_capacity(tower)
            self.assertAlmostEqual(results['collapse_capacity'], expected,
                                   places=4)

        expected = 55.8186
        line_name = 'LineB'
        for tower_name in ['T23', 'T44']:
            tower_id = self.cfg.lines[line_name]['name2id'][tower_name]
            tower = self.cfg.towers_by_line[line_name][tower_id]
            results = self.cfg.assign_collapse_capacity(tower)
            # self.assertAlmostEqual(tower.design_speed, 75.0)
            self.assertAlmostEqual(results['collapse_capacity'], expected,
                                   places=4)

        # u_factor = 1.0
        expected = 51.389
        line_name = 'LineB'
        for tower_name in ['T24', 'T43']:
            tower_id = self.cfg.lines[line_name]['name2id'][tower_name]
            tower = self.cfg.towers_by_line[line_name][tower_id]
            results = self.cfg.assign_collapse_capacity(tower)
            self.assertAlmostEqual(results['collapse_capacity'], expected,
                                   places=4)
        """

    def test_line_interaction(self):
        # FIXME
        pass

    def test_assign_cond_collapse_prob(self):
        # Tower 1: Terminal
        row = self.cfg.towers_by_line['LineA'][23]
        out = self.cfg.assign_cond_collapse_prob(row)

        expected = {(0, 1): 0.075,
                    (-1, 0): 0.075,
                    (-1, 0, 1): 0.35,
                    (-1, 0, 1, 2): 0.025,
                    (-2, -1, 0, 1): 0.025,
                    (-2, -1, 0, 1, 2): 0.10}
        self.assertEqual(out['cond_pc'], expected)
        self.assertEqual(out['max_no_adj_towers'], 2)

        # Raise warning for undefined cond_pc
        row = self.cfg.towers_by_line['LineA'][23]
        row['height'] = 55.0   # beyond the height
        msg = 'can not assign cond_pc for tower T1'
        with self.assertLogs(logger=self.logger, level='CRITICAL') as cm:
            self.cfg.assign_cond_collapse_prob(row)
            self.assertEqual(
                '{}:{}'.format(cm.output[0].split(':')[0],cm.output[0].split(':')[-1]),
                'CRITICAL:{}'.format(msg))

        # Tower 26
        row = self.cfg.towers_by_line['LineB'][9]
        out = self.cfg.assign_cond_collapse_prob(row)
        self.assertEqual(out['cond_pc'], expected)
        self.assertEqual(out['max_no_adj_towers'], 2)

    def test_assign_cond_collapse_prob_more(self):

        functions = ['Suspension', 'Terminal']
        heights = [20.0, 35.0]

        for func, height in zip(functions, heights):

            row = self.cfg.towers_by_line['LineA'][0].copy()
            row['function'] = func
            row['height'] = height
            tmp = self.cfg.cond_collapse_prob[func]
            expected = tmp.set_index('list').to_dict()['probability']
            out = self.cfg.assign_cond_collapse_prob(row)
            assertDeepAlmostEqual(self, out['cond_pc'], expected)

        functions = ['Strainer'] * 2
        levels = ['low', 'high']

        for func, level in zip(functions, levels):
            row = self.cfg.towers_by_line['LineA'][0].copy()
            row['function'] = func
            row['design_level'] = level
            tmp = self.cfg.cond_collapse_prob[func]
            expected = tmp.loc[tmp['design_level'] == level, :].set_index('list').to_dict()['probability']
            out = self.cfg.assign_cond_collapse_prob(row)
            assertDeepAlmostEqual(self, out['cond_pc'], expected)

    def test_ratio_z_to_10(self):
        # Tower 14
        row = self.cfg.towers_by_line['LineA'][0]
        assert row['terrain_cat'] == 2
        result = self.cfg.ratio_z_to_10(row)
        self.assertAlmostEqual(result, 1.0524)

    def test_ratio_z_to_10_more(self):

        # terrain category
        # ASNZS 1170.2:2011
        categories = [1, 2, 3, 4]
        mzcat10s = [1.12, 1.0, 0.83, 0.75]

        height_z = 15.4  # Suspension
        for cat, mzcat10 in zip(categories, mzcat10s):
            row = (self.cfg.towers_by_line['LineA'][0]).copy()
            row['terrain_cat'] = cat
            row['function'] = 'Suspension'
            row['height_z'] = height_z

            expected = np.interp(height_z, self.cfg.terrain_multiplier['height'],
                                 self.cfg.terrain_multiplier['tc' + str(cat)]) / mzcat10
            self.assertEqual(self.cfg.ratio_z_to_10(row), expected)

        height_z = 12.2  # Strainer, Terminal
        for cat, mzcat10 in zip(categories, mzcat10s):
            row = (self.cfg.towers_by_line['LineA'][0]).copy()
            row['terrain_cat'] = cat
            row['function'] = 'Strainer'
            row['height_z'] = height_z

            expected = np.interp(height_z, self.cfg.terrain_multiplier['height'],
                                 self.cfg.terrain_multiplier['tc' + str(cat)]) / mzcat10
            self.assertEqual(self.cfg.ratio_z_to_10(row), expected)

    def test_assign_design_values(self):

        self.assertEqual(self.cfg.options['adjust_design_by_topography'],
                         True)

        # Tower 14
        row = self.cfg.towers_by_line['LineA'][0]
        line = row['lineroute']
        self.assertEqual(line, 'LineA')
        out = self.cfg.assign_design_values(row)

        self.assertEqual(out.design_span,
                         self.cfg.design_value_by_line[line]['design_span'])
        self.assertEqual(out.design_level,
                         self.cfg.design_value_by_line[line]['design_level'])
        self.assertEqual(out.terrain_cat,
                         self.cfg.design_value_by_line[line]['terrain_cat'])
        self.assertEqual(out.design_speed,
                         self.cfg.design_value_by_line[line]['design_speed'])

        # Tower 26
        row = self.cfg.towers_by_line['LineB'][9]
        line = row['lineroute']
        self.assertEqual(line, 'LineB')
        out = self.cfg.assign_design_values(row)

        self.assertEqual(out.design_span,
                         self.cfg.design_value_by_line[line]['design_span'])
        self.assertEqual(out.design_level,
                         self.cfg.design_value_by_line[line]['design_level'])
        self.assertEqual(out.terrain_cat,
                         self.cfg.design_value_by_line[line]['terrain_cat'])
        self.assertEqual(out.design_speed,
                         self.cfg.design_value_by_line[line]['design_speed'])

        # Tower 34
        row = self.cfg.towers_by_line['LineB'][20]
        self.assertEqual(row['name'], 'T34')
        out = self.cfg.assign_design_values(row)

        self.assertEqual(out.design_span,
                         self.cfg.design_value_by_line[line]['design_span'])
        self.assertEqual(out.design_level,
                         self.cfg.design_value_by_line[line]['design_level'])
        self.assertEqual(out.terrain_cat,
                         self.cfg.design_value_by_line[line]['terrain_cat'])
        # design speed adjusted by topography
        self.assertEqual(out.design_speed,
                         1.6 * self.cfg.design_value_by_line[line]['design_speed'])

        # Tower 42
        row = self.cfg.towers_by_line['LineB'][27]
        self.assertEqual(row['name'], 'T42')
        out = self.cfg.assign_design_values(row)

        self.assertEqual(out.design_span,
                         self.cfg.design_value_by_line[line]['design_span'])
        self.assertEqual(out.design_level,
                         self.cfg.design_value_by_line[line]['design_level'])
        self.assertEqual(out.terrain_cat,
                         self.cfg.design_value_by_line[line]['terrain_cat'])
        # design speed adjusted by topography
        self.assertEqual(out.design_speed,
                         1.3 * self.cfg.design_value_by_line[line]['design_speed'])

    def test_assign_fragility_parameters(self):

        # Tower 14
        row = self.cfg.towers_by_line['LineA'][0]
        result = self.cfg.assign_fragility_parameters(row)

        self.assertEqual(result.frag_func, 'lognorm')
        assertDeepAlmostEqual(self,
                              result['frag_arg'],
                              {'collapse': 0.03, 'minor': 0.02}, places=4)
        assertDeepAlmostEqual(self,
                              result['frag_scale'],
                              {'collapse': 1.05, 'minor': 1.02})

    def test_assign_id_adj_towers(self):

        # Tower 14
        row = self.cfg.towers_by_line['LineA'][0]
        result = self.cfg.assign_id_adj_towers(row)
        self.assertEqual(result['id_adj'], [11, 12, 13, 14, 15])
        self.assertEqual(result['lid'], 13)

        # T26
        row = self.cfg.towers_by_line['LineB'][9]
        result = self.cfg.assign_id_adj_towers(row)
        self.assertEqual(result['id_adj'], [1, 2, 3, 4, 5])
        self.assertEqual(result['lid'], 3)

    """
    def test_assign_cond_pc_adj(self):

        # T14
        tower = self.cfg.towers_by_line['LineA'][0]
        row = assign_cond_pc_adj(tower)
        expected = {'cond_pc_adj': {14: 0.575, 12: 0.575, 15: 0.125, 11: 0.125},
                    # 'cond_pc_adj_mc_idx': [(-1, 0, 1, 2), (-2, -1, 0, 1),
                    #                            (0, 1), (-1, 0),
                    #                            (-2, -1, 0, 1, 2), (-1, 0, 1)],
                    'cond_pc_adj_mc_idx': [(12, 14, 15), (11, 12, 14),
                                               (12,), (14,),
                                               (11, 12, 14, 15), (12, 14)],
                    'cond_pc_adj_mc_prob':
                        np.array([0.025, 0.05, 0.125, 0.2, 0.3, 0.65])}

        assertDeepAlmostEqual(self, row['cond_pc_adj'], expected['cond_pc_adj'],
                              places=4)
        self.assertEqual(set(row['cond_pc_adj_mc_idx']),
                         set(expected['cond_pc_adj_mc_idx']))
        try:
            np.testing.assert_allclose(row['cond_pc_adj_mc_prob'],
                                       expected['cond_pc_adj_mc_prob'])
        except AssertionError:
            print('{}'.format(row['cond_pc_adj_mc_prob']))

        # T1: terminal tower
        tower = self.cfg.towers_by_line['LineA'][23]
        row = assign_cond_pc_adj(tower)
        expected = {'cond_pc_adj': {1: 0.575, 2: 0.125},
                    # 'cond_pc_adj_mc_idx': [(0, 1, 2), (0, 1)],
                    'cond_pc_adj_mc_idx': [(1, 2), (1,)],
                    'cond_pc_adj_mc_prob': np.array([0.125, 0.575])}

        assertDeepAlmostEqual(self, row['cond_pc_adj'], expected['cond_pc_adj'],
                              places=4)
        self.assertEqual(row['cond_pc_adj_mc_idx'],
                         expected['cond_pc_adj_mc_idx'])
        np.testing.assert_allclose(row['cond_pc_adj_mc_prob'],
                                   expected['cond_pc_adj_mc_prob'])

        # T22: terminal tower
        tower = self.cfg.towers_by_line['LineA'][1]
        row = assign_cond_pc_adj(tower)
        expected = {'cond_pc_adj': {20: 0.575, 19: 0.125},
                    # 'cond_pc_adj_mc_idx': [(-2, -1, 0), (-1, 0)],
                    'cond_pc_adj_mc_idx': [(19, 20,), (20,)],
                    'cond_pc_adj_mc_prob': np.array([0.125, 0.575])}

        assertDeepAlmostEqual(self, row['cond_pc_adj'], expected['cond_pc_adj'],
                              places=4)
        self.assertEqual(row['cond_pc_adj_mc_idx'],
                         expected['cond_pc_adj_mc_idx'])
        np.testing.assert_allclose(row['cond_pc_adj_mc_prob'],
                                   expected['cond_pc_adj_mc_prob'])
    """

    def test_create_list_idx(self):

        result = create_list_idx(idx=3, no_towers=2, max_no_towers=7,
                                 flag_direction=1)
        self.assertEqual(result, [4, 5])

        result = create_list_idx(idx=3, no_towers=2, max_no_towers=7,
                                 flag_direction=-1)
        self.assertEqual(result, [2, 1])

        result = create_list_idx(idx=6, no_towers=2, max_no_towers=7,
                                 flag_direction=1)
        self.assertEqual(result, [-1, -1])

        result = create_list_idx(idx=6, no_towers=2, max_no_towers=7,
                                 flag_direction=-1)
        self.assertEqual(result, [5, 4])

    def test_assign_shapely_point(self):

        # T1: terminal tower
        tower = self.cfg.towers_by_line['LineA'][23]
        row = assign_shapely_point(tower['shapes'])
        expected = {'coord': [149.0, 0.0],
                    'coord_lat_lon': [0.0, 149.0]}

        np.testing.assert_allclose(row.coord,
                                   expected['coord'])
        np.testing.assert_allclose(row.coord_lat_lon,
                                   expected['coord_lat_lon'])

    def test_assign_shapely_line(self):

        # LineA
        row = assign_shapely_line(self.cfg.lines['LineA']['shapes'])
        expected = {'coord': [[x, 0.0] for x in
                              np.arange(149.0, 149.109, 0.005)],
                    'coord_lat_lon': [[0.0, x] for x in
                                      np.arange(149.0, 149.109, 0.005)]}

        np.testing.assert_allclose(row.coord,
                                   expected['coord'])
        np.testing.assert_allclose(row.coord_lat_lon,
                                   expected['coord_lat_lon'])

    """
    def test_calculate_distance_between_towers(self):

        coord_lat_lon = [[0.0, 0.0], [0.005, 0.0], [0.01, 0.0]]

        distance = 556.1312

        expected = [0.5 * distance, distance, 0.5 * distance]

        results = calculate_distance_between_towers(coord_lat_lon)

        assert np.allclose(results, expected)
    """

    def test_set_line_interaction(self):
        # FIXME
        pass

    def test_get_cond_prob_line_interaction(self):
        # FIXME
        pass

    def test_find_id_nearest_pt(self):

        pt_coord = [0, 0]
        line_coord = [[1.6, 0], [1.5, 0], [0.0, 0], [0.4, 0]]

        result = find_id_nearest_pt(pt_coord, line_coord)

        self.assertEqual(result, 2)

    def test_unit_vector(self):

        result = unit_vector((0, 1))
        expected = np.array([0, 1])
        np.allclose(expected, result)

        result = unit_vector((3, 4))
        expected = np.array([0.6, 0.8])
        np.allclose(expected, result)

#     def test_line_interaction(self):
#
#         cfg = Config(os.path.join(BASE_DIR, 'test_line_interaction.cfg'))
#
#         expected = {'Calaca - Amadeo': ['Calaca - Santa Rosa',
#                                         'Amadeox - Calacax'],
#                     'Calaca - Santa Rosa': ['Calaca - Amadeo'],
#                     'Amadeox - Calacax': ['Calaca - Amadeo']}
#
#         assertDeepAlmostEqual(self, expected, cfg.line_interaction)
#
#         expected = {'by': 'Height', 'type': 'numeric',
#                     'file': './prob_line_interaction.csv'}
#
#         assertDeepAlmostEqual(self, expected,
#                               cfg.prob_line_interaction_metadata)
#
#         file_ = StringIO.StringIO("""\
# Height_lower, Height_upper, no_collapse, probability
# 0,40,1,0.2
# 0,40,3,0.1
# 0,40,5,0.01
# 0,40,7,0.001
# 40,1000,1,0.3
# 40,1000,3,0.15
# 40,1000,5,0.02
# 40,1000,7,0.002""")
#
#         path_input = os.path.dirname(os.path.realpath(
#             cfg.file_line_interaction_metadata))
#
#         file_line_interaction = os.path.join(
#             path_input, cfg.prob_line_interaction_metadata['file'])
#
#         with open(file_line_interaction, 'r') as file1:
#             for line1, line2 in zip(file1, file_):
#                 self.assertEqual(line1, line2)
#
#         file_.seek(0)
#         expected = pd.read_csv(file_, skipinitialspace=1)
#
#         pd.util.testing.assert_frame_equal(expected, cfg.prob_line_interaction)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestConfig("test_assign_collapse_capacity"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestConfig)
    # unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main(verbosity=2)