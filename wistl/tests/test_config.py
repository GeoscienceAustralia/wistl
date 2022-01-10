#!/usr/bin/env python

import unittest
import logging
import os
import tempfile
import numpy as np
import shapefile
import pandas as pd
from unittest.mock import patch

from wistl.config import *
from geopy.distance import geodesic

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
            exc = AssertionError(f'{str(exc)}\nTRACE: {trace}')
        raise exc


def create_shapefiles():

    # fields
    fields_pt = [['Type', 'C'],
                  ['Name', 'C'],
                  ['Latitude', 'N', 5],
                  ['Longitude', 'N', 5],
                  ['Comment', 'C'],
                  ['Function', 'C'],
                  ['Shape', 'C'],
                  ['DevAngle', 'N', 2],
                  ['AxisAz', 'N', 2],
                  ['ConstCost', 'N', 2],
                  ['Height', 'N', 2],
                  ['YrBuilt', 'N', 0],
                  ['LocSource', 'C'],
                  ['LineRoute', 'C']]
     # records
    #file_records_pt = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    #file_records_ln = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    file_records_pt = os.path.join(BASE_DIR, 'gis_data/data_for_points.csv')
    file_records_ln = os.path.join(BASE_DIR, 'gis_data/data_for_lines.csv')

    # read file_records_pt
#    file_records_pt.writelines([
#        'Type,Name,Latitude,Longitude,Comment,Function,Shape,DevAngle,AsixAz,ConstCost,Height,YrBuilt,LocSource,LineRoute\n',
#        'Lattice Tower,T1,0.01,149.0,Test,Terminal,Rectangle,0,134,0.00E+00,1.78E+01,1980,Fake,LineA\n',
#        'Lattice Tower,T2,0.01,149.01,Test,Suspension,Rectangle,0,134,0.00E+00,1.78E+01,1980,Fake,LineA\n',
#        'Lattice Tower,T3,0.01,149.02,Test,Terminal,Rectangle,0,134,0.00E+00,1.78E+01,1980,Fake,LineB\n',
#        'Lattice Tower,T4,0.01,149.03,Test,Suspension,Rectangle,0,134,0.00E+00,1.78E+01,1980,Fake,LineB\n'])
#    file_records_pt.seek(0)

    # shapefile_pt
    #shapefile_pt = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    #shapefile_pt = open('test_points.shp', 'w')
    _file = os.path.join(BASE_DIR, 'test_points.shp')
    if os.path.exists(_file):
        os.remove(_file)
    w_pt = shapefile.Writer(_file)
    for item in fields_pt:
        if item[1] == 'N':
            w_pt.field(item[0], item[1], decimal=item[2])
        else:
            w_pt.field(item[0], item[1])

    #records_pt = pd.read_csv(file_records_pt.name)
    records_pt = pd.read_csv(file_records_pt)
    _index = np.random.choice(records_pt.index, size=len(records_pt), replace=False)

    for _id in _index:
        item = records_pt.loc[_id]
        w_pt.record(*item)
        w_pt.point(item.Longitude, item.Latitude)

    w_pt.close()

    # fields
    fields_ln = [['LineName', 'C'],
                   ['Type', 'C'],
                   ['Owner', 'C'],
                   ['Operator', 'C'],
                   ['LineRoute', 'C'],
                   ['Capacity', 'N'],
                   ['TypeConduc', 'C'],
                   ['NumCircuit', 'N'],
                   ['Current', 'C'],
                   ['YrBuilt', 'N']]
#    # read file_records_ln
#    file_records_ln.writelines([
#        'LineName,Type,Owner,Operator,LineRoute,Capacity,TypeConduc,NumCircuit,Current,YrBuilt\n',
#        'LineA,HV Transmission Line,A,B,LineA,230,Unknown,2,Unknown,1980\n',
#        'LineB,HV Transmission Line,C,D,LineB,230,Unknown,2,Unknown,1985\n',
#        ])
#    file_records_ln.seek(0)
#
    # shapefile_lines
    #shapefile_ln = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    #w_ln = shapefile.Writer(shapefile_ln.name)
    _file = os.path.join(BASE_DIR, 'test_lines.shp')
    if os.path.exists(_file):
        os.remove(_file)
    w_ln = shapefile.Writer(_file)
    for item in fields_ln:
        w_ln.field(item[0], item[1])

    records_ln = pd.read_csv(file_records_ln)
    for _, row in records_ln.iterrows():
        w_ln.record(*row)
        coords = records_pt.loc[records_pt.LineRoute==row.LineRoute, ['Longitude', 'Latitude']].values.tolist()
        w_ln.line([coords])

    w_ln.close()


class TestLogger(unittest.TestCase):

    def test_cfg_not_found(self):

        # Raise error for undefined 
        with self.assertLogs('wistl.config', level='INFO') as cm:
            file_cfg = 'dummy.cfg'
            cfg = Config(file_cfg)
        msg = f'{file_cfg} not found'
        self.assertIn(f'ERROR:wistl.config:{msg}', cm.output)

    def test_file_not_found(self):

        for item in INPUT_FILES:
            # Raise error for undefined 
            with self.assertLogs('wistl.config', level='INFO') as cm:
                file_cfg = os.path.join(BASE_DIR, 'test_interaction.cfg')
                cfg = Config(file_cfg)
                cfg.options['adjust_design_by_topography'] = True
                setattr(cfg, f'_{item}', None)
                setattr(cfg, f'file_{item}', 'dummy.file')
                getattr(cfg, item)
            msg = f'dummy.file not found'
            self.assertIn(f'CRITICAL:wistl.config:{msg}', cm.output)

    def test_logger_fragility(self):
        with self.assertLogs('wistl.config', level='CRITICAL') as cm:
            file_cfg = os.path.join(BASE_DIR, 'test.cfg')
            cfg = Config(file_cfg)
            cfg._fragility = None
            cfg.fragility_metadata['main']['file'] = 'dummy.file'
            cfg.fragility
            path_metadata = os.path.dirname(os.path.realpath(cfg.file_fragility_metadata))
        msg = f'{path_metadata}/dummy.file not found'
        self.assertIn(f'CRITICAL:wistl.config:{msg}', cm.output)


class TestConfig1(unittest.TestCase):

    def test_nested_dic(self):

        _input = {'main': {'limit_states': 'minor, collapse',
                           'file': './test_fragility.yml'},
                  'fragility': {'function': {'Terminal': 'shape, angle',
                      'Suspension': 'shape, angle', 'Strainer': 'devangle, angle'}},
                  }
        expected = {'main': {'limit_states': ['minor', 'collapse'],
                             'file': './test_fragility.yml'},
                    'fragility': {'function':
                            {'Terminal': ['shape', 'angle'],
                             'Suspension': ['shape', 'angle'],
                             'Strainer': ['devangle', 'angle']}},
                    }

        output = nested_dic(_input)
        assertDeepAlmostEqual(self, expected, output)

    def test_h_design_adjustment_factor_by_topograhpy(self):

        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        _file.writelines(['[main]\n',
                          'threshold: 1.05, 1.1, 1.2, 1.3, 1.45\n',
                          '0:1.0\n',
                          '1:1.1\n',
                          '2:1.2\n',
                          '3:1.3\n',
                          '4:1.45\n',
                          '5:1.6\n',
                          ])
        _file.seek(0)

        expected = {0: 1.0,
                    1: 1.1,
                    2: 1.2,
                    3: 1.3,
                    4: 1.45,
                    5: 1.60,
                    'threshold': np.array([1.05, 1.1, 1.2, 1.3, 1.45])}

        output = h_design_adjustment_factor_by_topography(_file.name)
        assertDeepAlmostEqual(self, expected, output)

        _file.close()
        os.unlink(_file.name)

    def test_h_terrain_multiplier(self):
        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        _file.writelines([
            'height(m), terrain category 1, terrain category 2, terrain category 3, terrain category 4\n',
            '3,   0.99,  0.91,  0.83,  0.75\n',
            '5,   1.05,  0.91,  0.83,  0.75\n',
            '10,  1.12,  1.00,  0.83,  0.75\n',
            '15,  1.16,  1.05,  0.89,  0.75\n',
            '20,  1.19,  1.08,  0.94,  0.75\n',
            '30,  1.22,  1.12,  1.00,  0.80\n',
            '40,  1.24,  1.16,  1.04,  0.85\n',
            '50,  1.25,  1.18,  1.07,  0.90\n',
            '75,  1.27,  1.22,  1.12,  0.98\n',
            '100, 1.29,  1.24,  1.16,  1.03\n',
            '150, 1.31,  1.27,  1.21,  1.11\n',
            '200, 1.32,  1.29,  1.24,  1.16\n',
                          ])
        _file.seek(0)

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
        expected_keys = ['height', 'tc1', 'tc2', 'tc3', 'tc4']

        output = h_terrain_multiplier(_file.name)

        self.assertEqual(list(output.keys()), expected_keys)
        assertDeepAlmostEqual(self, expected,
                              output['tc1'])

        _file.close()
        os.unlink(_file.name)

    #def test_h_design_value_by_line(self):
    #    _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    #    _file.writelines([
    #        'lineroute, design_speed, design_span, terrain_cat, design_level\n',
    #        'LineA, 75.0, 400.0, 2, low\n',
    #        'LineB, 51.389, 400.0, 2, low\n',
    #                      ])
    #    _file.seek(0)

    #    expected = {'LineA': {'design_speed': 75.0,
    #                          'design_span': 400.0,
    #                          'terrain_cat': 2,
    #                          'design_level': 'low'},
    #                'LineB': {'design_speed': 51.389,
    #                          'design_span': 400.0,
    #                          'terrain_cat': 2,
    #                          'design_level': 'low'},
    #                }

    #    output = h_design_value_by_line(_file.name)

    #    assertDeepAlmostEqual(self, expected, output)

    #    _file.close()
    #    os.unlink(_file.name)

    def test_h_topographic_multiplier(self):
        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        _file.writelines([
            'Station,Mh,Mhopp\n',
            'T1,1,1\n',
            'T2,0.8,0.6\n',
            'T3,1,1\n',
            'T4,1,1\n',
                          ])
        _file.seek(0)

        expected = {'T1': 1.0,
                    'T2': 0.8,
                    'T3': 1.0,
                    'T4': 1.0}

        output = h_topographic_multiplier(_file.name)

        assertDeepAlmostEqual(self, expected, output)

        _file.close()
        os.unlink(_file.name)

    #def test_h_drag_height_by_type(self):
    #    _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
    #    _file.writelines([
    #        '# typical drag height by tower type\n',
    #        'Suspension,15.4\n',
    #        'Strainer,12.2\n',
    #        'Terminal,12.2\n',
    #                      ])
    #    _file.seek(0)

    #    expected = {'Suspension': 15.4,
    #                'Strainer': 12.2,
    #                'Terminal': 12.2}

    #    output = h_drag_height_by_type(_file.name)

    #    assertDeepAlmostEqual(self, expected, output)

    #    _file.close()
    #    os.unlink(_file.name)

    def test_read_yml_file(self):
        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        _file.writelines([
            '---\n',
            'file: ./test_cond_collapse_prob.yml\n',
            'probability:\n',
            '    type:\n',
            '        Lattice Tower:\n',
            '            function:\n',
            '                Suspension: height\n',
            '                Terminal: height\n',
            '                Strainer: design_level\n',
            ])
        _file.seek(0)

        expected = {'file': './test_cond_collapse_prob.yml',
                    'probability': {'type': {'Lattice Tower':
                        {'function': {'Suspension': 'height',
                                      'Terminal': 'height',
                                      'Strainer': 'design_level'}}}},
                    'path': os.path.dirname(os.path.realpath(_file.name))
                    }

        output = read_yml_file(_file.name)

        assertDeepAlmostEqual(self, expected, output)

        _file.close()
        os.unlink(_file.name)

    def test_h_cond_prob(self):

        expected = {'Lattice Tower': {
            'Suspension': {40: { (0,1): 0.075, (-1,0): 0.075, (-2, -1, 0, 1, 2): 0.02}},
            'Strainer': {'low': { (-1, 0, 1): 0.05, (-2,-1,0,1,2): 0.08}},
            }}

        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        _file.writelines([
            'Lattice Tower:\n',
            '    Suspension:\n',
            '        40:\n',
            '            0,1: 0.075\n',
            '            -1,0: 0.075\n',
            '            -2,2: 0.02\n',
            '    Strainer:\n',
            '        low:\n',
            '            -1,1: 0.05\n',
            '            -2,2: 0.08\n',
            ])

        _file.seek(0)

        output = h_cond_prob(_file.name)
        assertDeepAlmostEqual(self, expected, output)
        _file.close()
        os.unlink(_file.name)

    def test_split_str(self):

        str_ = 'aa: 44'
        expected = ('aa', 44)
        result = split_str(str_, ':')
        self.assertEqual(result, expected)

        str_ = 'bb: 0.0'
        expected = ('bb', 0.0)
        result = split_str(str_, ':')
        self.assertEqual(result, expected)

        str_ = 'cc: dd'
        expected = ('cc', 'dd')
        result = split_str(str_, ':')
        self.assertEqual(result, expected)


class TestConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.basicConfig(level=logging.INFO)
        #cls.logger = logging.getLogger(__name__)
        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'))
        cls.cfg.towers_by_line, cls.cfg.lines = set_towers_and_lines(cls.cfg)
        # for testing purpose
        #cls.cfg.options['adjust_design_by_topography'] = True


    #def test_design_value(self):

    #    expected = {'LineA': {'terrain_cat': 2,
    #                          'design_level': 'low',
    #                          'design_span': 400.0,
    #                          'design_speed': 75.0},
    #                'LineB': {'terrain_cat': 2,
    #                          'design_level': 'low',
    #                          'design_span': 400.0,
    #                          'design_speed': 51.389},
    #                'LineC': {'terrain_cat': 2,
    #                          'design_level': 'low',
    #                          'design_span': 400.0,
    #                          'design_speed': 51.389}}

    #    assertDeepAlmostEqual(self, expected, self.cfg.design_value_by_line)

    def test_fragility_metadata(self):
        expected = {'main': {'limit_states': ['minor', 'collapse'],
                             'file': './test_fragility.yml'},
                    'fragility': {'function':
                            {'Terminal': ['shape', 'angle'],
                             'Suspension': ['shape', 'angle'],
                             'Strainer': ['devangle', 'angle']}},
                    }


        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        _file.writelines([
            '---\n',
            'main:\n',
            '    limit_states: minor, collapse\n',
            '    file: ./test_fragility.yml\n',
            '\n',
            '#function:\n',
            '#    lognorm: \n',
            '#        param1: scale\n',
            '#        param2: arg\n',
            'fragility: \n',
            '    function: \n',
            '        Terminal: shape, angle \n',
            '        Suspension: shape, angle\n',
            '        Strainer: devangle, angle\n',
            ])
        _file.seek(0)

        self.cfg._fragility_metadata = None
        self.cfg.file_fragility_metadata = _file.name
        assertDeepAlmostEqual(self, expected,
                              self.cfg.fragility_metadata)
        _file.close()
        os.unlink(_file.name)

    def test_h_fragility(self):
        expected = {'Lattice Tower': {
            'Suspension': {'rectangle': { 11.5: {'minor': ['lognorm', '1.02', '0.02'],
                'collapse': ['lognorm', '1.05', '0.02']}}},
            'Strainer': {5.0: { 180.0: {'minor': ['lognorm', '1.143', '0.032'],
                'collapse': ['lognorm', '1.18', '0.04']}}},
            }}

        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        _file.writelines([
            '---\n',
            'Lattice Tower:\n',
            '    Suspension:\n',
            '        rectangle: # assume starting from 0\n',
            '            11.5: # normal1 \n',
            '                minor: lognorm,1.02,0.02\n',
            '                collapse: lognorm,1.05,0.02\n',
            '    Strainer:\n',
            '        5: # dev_angle\n',
            '            180: # all\n',
            '                minor: lognorm,1.143,0.032\n',
            '                collapse: lognorm,1.18,0.04\n',
            ])

        _file.seek(0)

        output = h_fragility(_file.name)
        assertDeepAlmostEqual(self, expected, output)
        _file.close()
        os.unlink(_file.name)

    def test_damage_states(self):
        expected = ['minor', 'collapse']
        self.cfg._damage_states = None
        self.assertEqual(expected, self.cfg.dmg_states)

    def test_no_damage_states(self):
        expected = 2
        self.cfg._no_dmg_states = None
        self.assertEqual(self.cfg.no_dmg_states, expected)

    def test_non_collapse(self):
        expected = ['minor']
        self.cfg._non_collapse = None
        self.assertEqual(self.cfg.non_collapse, expected)

    def test_no_towers_by_line(self):

        expected = {'LineA': 22, 'LineB': 22}
        for key, value in expected.items():

            self.assertEqual(value, len(self.cfg.towers_by_line[key]))

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

    #def test_read_drag_height_by_type(self):

    #    expected = {'Strainer': 12.199999999999999,
    #                'Suspension': 15.4,
    #                'Terminal': 12.199999999999999}

    #    assertDeepAlmostEqual(self, expected,
    #                          self.cfg.drag_height_by_type)

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

        try:
            self.assertTrue(self.cfg.options['adjust_design_by_topography'])
        except AssertionError:
            self.cfg.options['adjust_design_by_topography'] = True

        assertDeepAlmostEqual(self, expected,
                              self.cfg.design_adjustment_factor_by_topography)

    def test_towers(self):
        # file_wind_base_name
        self.assertEqual(self.cfg.towers_by_line['LineA']['T1']['file_wind_base_name'],
                         'ts.T1.csv')

        # height_z
        # terminal
        self.assertAlmostEqual(self.cfg.towers_by_line['LineA']['T1']['height_z'],
                               12.2)
        # suspension
        self.assertAlmostEqual(self.cfg.towers_by_line['LineA']['T11']['height_z'],
                               15.4)
        # strainer
        self.assertAlmostEqual(self.cfg.towers_by_line['LineA']['T9']['height_z'],
                               12.2)

    def test_lines(self):

        # name_output
        self.assertEqual(self.cfg.lines['LineA']['name_output'], 'LineA')

        # no_towers
        self.assertAlmostEqual(self.cfg.lines['LineB']['no_towers'], 22)

    def test_process_config(self):
        pass
        ## max_adj
        #expected = {'Terminal': 2, 'Suspension': 2, 'Strainer': 6}
        #for key, value in expected.items():
        #    self.assertEqual(value,
        #                     self.cfg.cond_collapse_prob_metadata[key]['max_adj'])

    def test_sort_by_location(self):

        expected_names = {'LineA': ['T{}'.format(x) for x in range(1, 23)],
                          'LineB': ['T{}'.format(x) for x in range(23, 45)]}

        for line in ['LineA', 'LineB']:
            self.assertEqual(self.cfg.lines[line]['names'],
                             expected_names[line])

        expected_id2name = {
            'LineA': {2: 'T17', 6: 'T19', 8: 'T20', 19: 'T22', 20: 'T7', 23: 'T9', 24: 'T8', 29: 'T18', 32: 'T13', 33: 'T3', 39: 'T14', 40: 'T15', 43: 'T6', 44: 'T2', 46: 'T11', 48: 'T10', 50: 'T12', 56: 'T21', 57: 'T1', 62: 'T16', 63: 'T5', 64: 'T4'},
            'LineB': {1: 'T44', 4: 'T37', 5: 'T39', 10: 'T24', 11: 'T36', 12: 'T27', 13: 'T38', 14: 'T28', 21: 'T25', 22: 'T40', 25: 'T41', 26: 'T42', 27: 'T32', 34: 'T43', 41: 'T23', 47: 'T30', 49: 'T33', 54: 'T29', 55: 'T34', 58: 'T26', 59: 'T31', 65: 'T35'}
            }

        expected_ids_old = {
            'LineA': [57, 44, 33, 64, 63, 43, 20, 24, 23, 48, 46, 50, 32, 39, 40, 62, 2, 29, 6, 8, 56, 19]
,
            'LineB': [41, 10, 21, 58, 12, 14, 54, 47, 59, 27, 49, 55, 65, 11, 4, 13, 5, 22, 25, 26, 34, 1]
             }

        expected_ids = {x: [expected_id2name[x][k] for k in expected_ids_old[x]] for x in ['LineA', 'LineB']}

        #for line in ['LineA', 'LineB']:
        #    self.assertEqual(self.cfg.lines[line]['ids'], expected_ids[line])

        #expected_name2id = {key: {v: k for k, v in value.items()}
        #                    for key, value in expected_id2name.items()}

        #for line in ['LineA', 'LineB']:
        #    self.assertEqual(self.cfg.lines[line]['id2name'],
        #                     expected_id2name[line])

        #for line in ['LineA', 'LineB']:
        #    self.assertEqual(self.cfg.lines[line]['name2id'],
        #                     expected_name2id[line])

    def test_assign_collapse_capacity(self):
        # TODO
        # u_factor = 1.0 - K_FACTOR[NO_CIRCUIT] * (1-actual_span/design_span)
        # collapse_capacity = design_speed / sqrt(u_factor)
        # K_FACTOR = {1: 0.33, 2: 0.5}  # hard-coded

        u_factor = 0.84787
        expected = 81.4509
        line_name = 'LineA'
        for tower_name in ['T1', 'T22']:
            #tower_id = self.cfg.lines[line_name]['name2id'][tower_name]
            tower = self.cfg.towers_by_line[line_name][tower_name]
            results = assign_collapse_capacity(tower, self.cfg.lines)
            self.assertAlmostEqual(results['actual_span'], 278.2987, places=3)
            self.assertAlmostEqual(tower['design_span'], 400.0, places=3)
            self.assertAlmostEqual(results['u_factor'], u_factor, places=3)
            self.assertAlmostEqual(tower['design_speed'], 75.0, places=3)
            self.assertAlmostEqual(results['collapse_capacity'], expected, places=3)

        u_factor = 1.0
        expected = 75.0
        line_name = 'LineA'
        for tower_name in ['T2', 'T21']:
            #tower_id = self.cfg.lines[line_name]['name2id'][tower_name]
            tower = self.cfg.towers_by_line[line_name][tower_name]
            results = assign_collapse_capacity(tower, self.cfg.lines)
            self.assertAlmostEqual(results['actual_span'], 556.59745, places=3)
            self.assertAlmostEqual(tower['design_span'], 400.0, places=3)
            self.assertAlmostEqual(results['u_factor'], u_factor, places=3)
            self.assertAlmostEqual(tower['design_speed'], 75.0, places=3)
            self.assertAlmostEqual(results['collapse_capacity'], expected,
                                   places=4)

        expected = 55.80905
        u_factor = 0.84787
        line_name = 'LineB'
        for tower_name in ['T23', 'T44']:
            #tower_id = self.cfg.lines[line_name]['name2id'][tower_name]
            tower = self.cfg.towers_by_line[line_name][tower_name]
            results = assign_collapse_capacity(tower, self.cfg.lines)
            self.assertAlmostEqual(results['actual_span'], 278.2987, places=3)
            self.assertAlmostEqual(tower['design_span'], 400.0, places=3)
            self.assertAlmostEqual(results['u_factor'], u_factor, places=3)
            self.assertAlmostEqual(tower['design_speed'], 51.3889, places=3)
            self.assertAlmostEqual(results['collapse_capacity'], expected,
                                   places=4)

        # u_factor = 1.0
        expected = 51.389
        line_name = 'LineB'
        for tower_name in ['T24', 'T43']:
            #tower_id = self.cfg.lines[line_name]['name2id'][tower_name]
            tower = self.cfg.towers_by_line[line_name][tower_name]
            results = assign_collapse_capacity(tower, self.cfg.lines)
            self.assertAlmostEqual(results['collapse_capacity'], expected,
                                   places=4)

    def test_assign_cond_pc_logging(self):

        # Raise warning for undefined cond_pc
        with self.assertLogs('wistl.config', level='INFO') as cm:
            row = self.cfg.towers_by_line['LineB']['T27']
            row['height'] = 55.0   # beyond the height
            assign_cond_pc(row, self.cfg)
        msg = f'unable to assign cond_pc for tower {row["name"]}'
        self.assertIn(f'CRITICAL:wistl.config:{msg}', cm.output)

    def test_assign_cond_pc(self):

        #logger = logging.getLogger(__file__)
        # Tower 1: Terminal
        row = self.cfg.towers_by_line['LineA']['T1']
        out = assign_cond_pc(row, self.cfg)

        expected = {(0, 1): 0.075,
                    (-1, 0): 0.075,
                    (-1, 0, 1): 0.35,
                    (-1, 0, 1, 2): 0.025,
                    (-2, -1, 0, 1): 0.025,
                    (-2, -1, 0, 1, 2): 0.1}
        self.assertEqual(out['cond_pc'], expected)
        self.assertEqual(out['max_no_adj_towers'], 2)

        # Tower 26
        row = self.cfg.towers_by_line['LineB']['T26']
        out = assign_cond_pc(row, self.cfg)
        self.assertEqual(out['cond_pc'], expected)
        self.assertEqual(out['max_no_adj_towers'], 2)

    def test_assign_cond_pc_more(self):

        functions = ['Suspension', 'Terminal']
        heights = [20.0, 35.0]

        expected = {(0, 1): 0.075,
                    (-1, 0): 0.075,
                    (-1, 0, 1): 0.35,
                    (-1, 0, 1, 2): 0.025,
                    (-2, -1, 0, 1): 0.025,
                    (-2, -1, 0, 1, 2): 0.1}
        for func, height in zip(functions, heights):

            row = self.cfg.towers_by_line['LineA']['T12'].copy()
            row['type'] = 'Lattice Tower'
            row['function'] = func
            row['height'] = height
            out = assign_cond_pc(row, self.cfg)
            assertDeepAlmostEqual(self, out['cond_pc'], expected)

        functions = ['Strainer'] * 2
        levels = ['low', 'high']
        expected = {'low': {(-1, 0, 1): 0.05,
                    (-2, -1, 0, 1, 2): 0.08,
                    (-3, -2, -1, 0, 1, 2, 3): 0.10,
                    (-4, -3, -2, -1, 0, 1, 2, 3, 4): 0.08,
                    (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5): 0.05,
                    (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6): 0.62},
                    'high': {(-1, 0, 1): 0.15,
                    (-2, -1, 0, 1, 2): 0.22,
                    (-3, -2, -1, 0, 1, 2, 3): 0.22,
                    (-4, -3, -2, -1, 0, 1, 2, 3, 4): 0.15,
                    (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5): 0.12,
                    (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6): 0.08},
                    }


        for func, level in zip(functions, levels):
            row = self.cfg.towers_by_line['LineA']['T12'].copy()
            row['type'] = 'Lattice Tower'
            row['function'] = func
            row['design_level'] = level
            out = assign_cond_pc(row, self.cfg)
            assertDeepAlmostEqual(self, out['cond_pc'], expected[level])

    def test_ratio_z_to_10(self):
        # Tower 14
        row = self.cfg.towers_by_line['LineA']['T14']
        assert row['terrain_cat'] == 2
        result = ratio_z_to_10(row, self.cfg)
        self.assertAlmostEqual(result, 1.0524)

    def test_ratio_z_to_10_more(self):

        # terrain category
        # ASNZS 1170.2:2011
        categories = [1, 2, 3, 4]
        mzcat10s = [1.12, 1.0, 0.83, 0.75]

        height_z = 15.4  # Suspension
        for cat, mzcat10 in zip(categories, mzcat10s):
            row = (self.cfg.towers_by_line['LineA']['T14']).copy()
            row['terrain_cat'] = cat
            row['function'] = 'Suspension'
            row['height_z'] = height_z

            expected = np.interp(height_z, self.cfg.terrain_multiplier['height'],
                                 self.cfg.terrain_multiplier['tc' + str(cat)]) / mzcat10
            self.assertEqual(ratio_z_to_10(row, self.cfg), expected)

        height_z = 12.2  # Strainer, Terminal
        for cat, mzcat10 in zip(categories, mzcat10s):
            row = (self.cfg.towers_by_line['LineA']['T1']).copy()
            row['terrain_cat'] = cat
            row['function'] = 'Strainer'
            row['height_z'] = height_z

            expected = np.interp(height_z, self.cfg.terrain_multiplier['height'],
                                 self.cfg.terrain_multiplier['tc' + str(cat)]) / mzcat10
            self.assertEqual(ratio_z_to_10(row, self.cfg), expected)

    #def test_assign_design_values(self):

    #    self.assertEqual(self.cfg.options['adjust_design_by_topography'],
    #                     True)

    #    # Tower 14
    #    row = self.cfg.towers_by_line['LineA'][39]
    #    line = row['lineroute']
    #    self.assertEqual(line, 'LineA')
    #    out = self.cfg.assign_design_values(row)

    #    self.assertEqual(out.design_span,
    #                     self.cfg.design_value_by_line[line]['design_span'])
    #    self.assertEqual(out.design_level,
    #                     self.cfg.design_value_by_line[line]['design_level'])
    #    self.assertEqual(out.terrain_cat,
    #                     self.cfg.design_value_by_line[line]['terrain_cat'])
    #    self.assertEqual(out.design_speed,
    #                     self.cfg.design_value_by_line[line]['design_speed'])

    #    # Tower 26
    #    row = self.cfg.towers_by_line['LineB'][58]
    #    line = row['lineroute']
    #    self.assertEqual(line, 'LineB')
    #    out = self.cfg.assign_design_values(row)

    #    self.assertEqual(out.design_span,
    #                     self.cfg.design_value_by_line[line]['design_span'])
    #    self.assertEqual(out.design_level,
    #                     self.cfg.design_value_by_line[line]['design_level'])
    #    self.assertEqual(out.terrain_cat,
    #                     self.cfg.design_value_by_line[line]['terrain_cat'])
    #    self.assertEqual(out.design_speed,
    #                     self.cfg.design_value_by_line[line]['design_speed'])

    #    # Tower 34
    #    row = self.cfg.towers_by_line['LineB'][55]
    #    self.assertEqual(row['name'], 'T34')
    #    out = self.cfg.assign_design_values(row)

    #    self.assertEqual(out.design_span,
    #                     self.cfg.design_value_by_line[line]['design_span'])
    #    self.assertEqual(out.design_level,
    #                     self.cfg.design_value_by_line[line]['design_level'])
    #    self.assertEqual(out.terrain_cat,
    #                     self.cfg.design_value_by_line[line]['terrain_cat'])
    #    # design speed adjusted by topography
    #    self.assertEqual(out.design_speed,
    #                     1.6 * self.cfg.design_value_by_line[line]['design_speed'])

    #    # Tower 42
    #    row = self.cfg.towers_by_line['LineB'][26]
    #    self.assertEqual(row['name'], 'T42')
    #    out = self.cfg.assign_design_values(row)

    #    self.assertEqual(out.design_span,
    #                     self.cfg.design_value_by_line[line]['design_span'])
    #    self.assertEqual(out.design_level,
    #                     self.cfg.design_value_by_line[line]['design_level'])
    #    self.assertEqual(out.terrain_cat,
    #                     self.cfg.design_value_by_line[line]['terrain_cat'])
    #    # design speed adjusted by topography
    #    self.assertEqual(out.design_speed,
    #                     1.3 * self.cfg.design_value_by_line[line]['design_speed'])

   # def test_assign_fragility_parameters(self):

   #     # Tower 14
   #     row = self.cfg.towers_by_line['LineA'][0]
   #     result = self.cfg.assign_fragility_parameters(row)

   #     self.assertEqual(result.frag_func, 'lognorm')
   #     assertDeepAlmostEqual(self,
   #                           result['frag_arg'],
   #                           {'collapse': 0.03, 'minor': 0.02}, places=4)
   #     assertDeepAlmostEqual(self,
   #                           result['frag_scale'],
   #                           {'collapse': 1.05, 'minor': 1.02})

    def test_assign_id_adj_towers(self):

        # Tower 14
        row = self.cfg.towers_by_line['LineA']['T14']
        self.assertEqual(row['name'], 'T14')
        result = assign_id_adj_towers(row, self.cfg.towers_by_line, self.cfg.lines, self.cfg)
        self.assertEqual(result['id_adj'], [11, 12, 13, 14, 15])
        self.assertEqual(result['idl'], 13)

        # T26
        row = self.cfg.towers_by_line['LineB']['T26']
        result = assign_id_adj_towers(row, self.cfg.towers_by_line, self.cfg.lines, self.cfg)
        self.assertEqual(result['id_adj'], [1, 2, 3, 4, 5])
        self.assertEqual(result['idl'], 3)

        # T32
        row = self.cfg.towers_by_line['LineB']['T32']
        result = assign_id_adj_towers(row, self.cfg.towers_by_line, self.cfg.lines, self.cfg)
        self.assertEqual(result['id_adj'],
                         [3, 4, 5, 6, 7, 8, -1, 10, 11, 12, 13, 14, 15])
        self.assertEqual(result['idl'], 9)

    def test_assign_cond_pc_adj(self):

        # T14: suspension tower
        tower = self.cfg.towers_by_line['LineA']['T14']
        self.assertEqual(tower['name'], 'T14')
        self.assertEqual(tower['idl'], 13)
        row = assign_cond_pc_adj(tower)
        expected = {'cond_pc_adj': {14: 0.575, 12: 0.575, 15: 0.125, 11: 0.125},
                    'cond_pc_adj_sim_idx': [(12, 14, 15), (11, 12, 14),
                                               (12,), (14,),
                                               (11, 12, 14, 15), (12, 14)],
                    'cond_pc_adj_sim_prob':
                        np.array([0.025, 0.05, 0.125, 0.2, 0.3, 0.65])}

        assertDeepAlmostEqual(self, row['cond_pc_adj'], expected['cond_pc_adj'],
                              places=4)
        self.assertEqual(set(row['cond_pc_adj_sim_idx']),
                         set(expected['cond_pc_adj_sim_idx']))
        try:
            np.testing.assert_allclose(row['cond_pc_adj_sim_prob'],
                                       expected['cond_pc_adj_sim_prob'])
        except AssertionError:
            print('{}'.format(row['cond_pc_adj_sim_prob']))

        # T1: terminal tower
        tower = self.cfg.towers_by_line['LineA']['T1']
        row = assign_cond_pc_adj(tower)
        self.assertEqual(tower['name'], 'T1')
        self.assertEqual(tower['idl'], 0)
        expected = {'cond_pc_adj': {1: 0.575, 2: 0.125},
                    'cond_pc_adj_sim_idx': [(1, 2), (1,)],
                    'cond_pc_adj_sim_prob': np.array([0.125, 0.575])}

        assertDeepAlmostEqual(self, row['cond_pc_adj'], expected['cond_pc_adj'],
                              places=4)
        self.assertEqual(row['cond_pc_adj_sim_idx'],
                         expected['cond_pc_adj_sim_idx'])
        np.testing.assert_allclose(row['cond_pc_adj_sim_prob'],
                                   expected['cond_pc_adj_sim_prob'])

        # T22: terminal tower
        tower = self.cfg.towers_by_line['LineA']['T22']
        self.assertEqual(tower['name'], 'T22')
        self.assertEqual(tower['idl'], 21)
        row = assign_cond_pc_adj(tower)
        expected = {'cond_pc_adj': {20: 0.575, 19: 0.125},
                    'cond_pc_adj_sim_idx': [(19, 20,), (20,)],
                    'cond_pc_adj_sim_prob': np.array([0.125, 0.575])}

        assertDeepAlmostEqual(self, row['cond_pc_adj'], expected['cond_pc_adj'],
                              places=4)
        self.assertEqual(row['cond_pc_adj_sim_idx'],
                         expected['cond_pc_adj_sim_idx'])
        np.testing.assert_allclose(row['cond_pc_adj_sim_prob'],
                                   expected['cond_pc_adj_sim_prob'])

        # T9: strainer tower
        tower = self.cfg.towers_by_line['LineA']['T9']
        self.assertEqual(tower['name'], 'T9')
        self.assertEqual(tower['idl'], 8)
        row = assign_cond_pc_adj(tower)
        expected = {'cond_pc_adj': {7: 0.98, 9: 0.98,
            6: 0.93, 10: 0.93,
            5: 0.85, 11: 0.85,
            4: 0.75, 12: 0.75,
            3: 0.67, 13: 0.67,
            2: 0.62, 14: 0.62},
            'cond_pc_adj_sim_idx': [(7, 9,), (3,4,5,6,7,9,10,11,12,13,),(6,7,9,10,), (4,5,6,7,9,10,11,12,),(5,6,7,9,10,11,),(2,3,4,5,6,7,9,10,11,12,13,14,)],
            'cond_pc_adj_sim_prob': np.array([0.05, 0.10, 0.18, 0.26, 0.36, 0.98])}

        assertDeepAlmostEqual(self, row['cond_pc_adj'], expected['cond_pc_adj'],
                              places=4)
        self.assertEqual(row['cond_pc_adj_sim_idx'],
                         expected['cond_pc_adj_sim_idx'])
        np.testing.assert_allclose(row['cond_pc_adj_sim_prob'],
                                   expected['cond_pc_adj_sim_prob'])

        # T1: suspension tower neighboring strainer (9)
        tower = self.cfg.towers_by_line['LineB']['T33']
        row = assign_cond_pc_adj(tower)
        self.assertEqual(tower['name'], 'T33')
        self.assertEqual(tower['idl'], 10)
        expected = {'cond_pc_adj': {12: 0.125, 11:0.575},
                    'cond_pc_adj_sim_idx': [(11, 12), (11,)],
                    'cond_pc_adj_sim_prob': np.array([0.125, 0.575])}

        assertDeepAlmostEqual(self, row['cond_pc_adj'], expected['cond_pc_adj'],
                              places=4)
        self.assertEqual(row['cond_pc_adj_sim_idx'],
                         expected['cond_pc_adj_sim_idx'])
        np.testing.assert_allclose(row['cond_pc_adj_sim_prob'],
                                   expected['cond_pc_adj_sim_prob'])

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
        tower = self.cfg.towers_by_line['LineA']['T1']
        row = assign_shapely_point(tower)
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

    def test_distance(self):

        # lat, long
        pt1 = (52.2296756, 21.0122287)
        pt2 = (52.406374, 16.9251681)

        expected = 279352.901604
        result = geodesic(pt1, pt2).meters
        self.assertAlmostEqual(expected, result, places=4)

    def test_calculate_distance_between_towers(self):

        coord_lat_lon = [[0.0, 0.0], [0.005, 0.0], [0.01, 0.0]]

        distance = 552.87138

        dist1 = geodesic(coord_lat_lon[0], coord_lat_lon[1]).meters
        dist2 = geodesic(coord_lat_lon[1], coord_lat_lon[2]).meters

        self.assertAlmostEqual(dist1, distance, places=4)
        self.assertAlmostEqual(dist2, distance, places=4)

        expected = [0.5 * distance, distance, 0.5 * distance]

        results = calculate_distance_between_towers(coord_lat_lon)

        try:
            assert np.allclose(results, expected)
        except AssertionError:
            print('{} is expected but {}'.format(expected, results))

    def test_read_wind_scenario(self):

        self.assertEqual(self.cfg.events[0].id, 'test1_s3.0')
        self.assertEqual(self.cfg.events[1].id, 'test2_s2.5')
        self.assertEqual(self.cfg.events[2].id, 'test2_s3.5')

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


class TestConfig2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        cls.cfg = Config(os.path.join(BASE_DIR, 'test1.cfg'))
        cls.cfg.towers_by_line, cls.cfg.lines = set_towers_and_lines(cls.cfg)

    def test_read_wind_scenario(self):

        self.assertEqual(self.cfg.events[0].id, 'test2_s1.4')
        self.assertEqual(self.cfg.events[0].name, 'test2')
        self.assertEqual(self.cfg.events[0].scale, 1.4)
        self.assertEqual(self.cfg.events[0].seed, 1)
        #self.assertEqual(self.cfg.events[1], ('test2', 2.5, 1))
        #self.assertEqual(self.cfg.events[2], ('test2', 3.5, 2))

class TestConfig3(unittest.TestCase):
    """
    testing line interaction
    """

    @classmethod
    def setUpClass(cls):

        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test_interaction.cfg'))
        cls.cfg.towers_by_line, cls.cfg.lines = set_towers_and_lines(cls.cfg)

        # for testing purpose
        #cls.cfg.options['adjust_design_by_topography'] = True

    def test_read_yml_file2(self):
        # test read_yml_file against cond_prob_interaction
        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        _file.writelines([
            '---\n',
            'file: ./cond_prob_interaction.yml\n',
            'probability:\n',
            '    type:\n',
            '        Lattice Tower:\n',
            '            function:\n',
            '                Suspension: height\n',
            '                Terminal: height\n',
            '                Strainer: height\n',
            ])
        _file.seek(0)

        expected = {'file': './cond_prob_interaction.yml',
                    'probability': {'type': {'Lattice Tower':
                        {'function': {'Suspension': 'height',
                                      'Terminal': 'height',
                                      'Strainer': 'height'}}}},
                    'path': os.path.dirname(os.path.realpath(_file.name))
                    }

        output = read_yml_file(_file.name)

        assertDeepAlmostEqual(self, expected, output)

        _file.close()
        os.unlink(_file.name)

    @unittest.skip("FIXME")
    def test_cond_prob_interaction(self):

        expected = {'Lattice Tower': {
            'Suspension': {40: { 1: 0.2, 3: 0.1, 5: 0.01, 7: 0.001}},
            }}

        _file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        _file.writelines([
            'Lattice Tower:\n',
            '    Suspension:\n',
            '        40:\n',
            '            1: 0.2\n',
            '            3: 0.1\n',
            '            5: 0.01\n',
            '            7: 0.001\n',
            ])

        _file.seek(0)
        with open(_file.name, 'r') as ymlfile:
            output = yaml.load(ymlfile, Loader=yaml.FullLoader)
        assertDeepAlmostEqual(self, expected, output)
        _file.close()
        os.unlink(_file.name)

    def test_read_line_interaction(self):

        self.assertEqual(self.cfg.line_interaction['LineB'], ['LineA'])
        self.assertEqual(self.cfg.line_interaction['LineA'], ['LineB', 'LineC'])
        self.assertEqual(self.cfg.line_interaction['LineC'], ['LineA'])

    @unittest.skip("FIXME")
    def test_assign_cond_pc_interaction(self):

        # Tower 1: Terminal
        row = self.cfg.towers_by_line['LineA']['T1']
        row.pop('cond_pc_interaction_no')
        row.pop('cond_pc_interaction_prob')
        row.pop('cond_pc_interaction_cprob')
        out = self.cfg.assign_cond_pc_interaction(row)

        self.assertTrue(row['height'] < 40)
        self.assertEqual(out['cond_pc_interaction_no'], [1, 3, 5, 7])
        np.testing.assert_allclose(out['cond_pc_interaction_cprob'], np.array([0.2, 0.3, 0.31, 0.311]))
        self.assertDictEqual(out['cond_pc_interaction_prob'], {1:0.2, 3:0.1, 5:0.01, 7:0.001})

    @unittest.skip("FIXME")
    def test_assign_cond_pc_interaction_more(self):

        functions = ['Suspension', 'Terminal', 'Strainer']
        heights = [35.0] * 3
        expected = {'no': [1, 3, 5, 7], 'prob': [0.2, 0.3, 0.31, 0.311]}

        for func, height in zip(functions, heights):

            row = self.cfg.towers_by_line['LineA']['T12'].copy()
            row['type'] = 'Lattice Tower'
            row['function'] = func
            row['height'] = height
            row.pop('cond_pc_interaction_no')
            row.pop('cond_pc_interaction_prob')
            row.pop('cond_pc_interaction_cprob')
            out = self.cfg.assign_cond_pc_interaction(row)
            self.assertEqual(out['cond_pc_interaction_no'], expected['no'])
            np.testing.assert_allclose(out['cond_pc_interaction_cprob'], expected['prob'])
            self.assertDictEqual(out['cond_pc_interaction_prob'], {1:0.2, 3:0.1, 5:0.01, 7:0.001})

        functions = ['Suspension', 'Terminal', 'Strainer']
        heights = [75.0] *3
        expected = {'no': [1, 3, 5, 7], 'prob': [0.3, 0.45, 0.47, 0.472]}

        for func, height in zip(functions, heights):

            row = self.cfg.towers_by_line['LineA']['T12'].copy()
            row['type'] = 'Lattice Tower'
            row['function'] = func
            row['height'] = height
            row.pop('cond_pc_interaction_no')
            row.pop('cond_pc_interaction_prob')
            row.pop('cond_pc_interaction_cprob')
            out = self.cfg.assign_cond_pc_interaction(row)
            self.assertEqual(out['cond_pc_interaction_no'], expected['no'])
            np.testing.assert_allclose(out['cond_pc_interaction_cprob'], expected['prob'])
            self.assertDictEqual(out['cond_pc_interaction_prob'], {1:0.3, 3:0.15, 5:0.02, 7:0.002})

    @unittest.skip("FIXME")
    def test_assign_target_line(self):

        """
        LineC: 30(T45), 15(T46)
        LineA: 57(T1), 44(T2)
        LineB: 41(T23), 10(T24)
        """

        # LineB -> LineA
        for idl, idn in enumerate(self.cfg.lines['LineB']['ids']):
            t = self.cfg.towers_by_line['LineB'][idn]
            assertDeepAlmostEqual(self, t['target_line'],
                                  {'LineA': {'id': idl, 'vector': np.array([0.0, 1.0])}})

        # LineC -> LineA
        for idl, idn in enumerate(self.cfg.lines['LineC']['ids']):
            t = self.cfg.towers_by_line['LineC'][idn]
            assertDeepAlmostEqual(self, t['target_line'],
                                  {'LineA': {'id': idl, 'vector': np.array([0.0, -1.0])}})

        # LineA -> LineB and LineA
        for idl, idn in enumerate(self.cfg.lines['LineA']['ids']):
            t = self.cfg.towers_by_line['LineA'][idn]
            assertDeepAlmostEqual(self, t['target_line'],
                                  {'LineB': {'id': idl, 'vector': np.array([0.0, -1.0])},
                                   'LineC': {'id': idl, 'vector': np.array([0.0, 1.0])}})

if __name__ == '__main__':
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestConfig)
    # suite.addTest(TestConfig("test_assign_collapse_capacity"))
    #runner = unittest.TextTestRunner(verbosity=2)
    #runner.run(suite)
    unittest.main(verbosity=2)
