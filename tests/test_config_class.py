#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
import pandas as pd
import os
import StringIO
import numpy as np

from collections import OrderedDict
from wistl.config_class import TransmissionConfig

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
    :param test_case: TestCase object on which we can call all of the basic
        'assert' methods.
    :type test_case: :py:class:`unittest.TestCase` object
    """
    is_root = not '__trace' in kwargs
    trace = kwargs.pop('__trace', 'ROOT')
    try:
        if isinstance(expected, (int, float, long, complex)):
            test_case.assertAlmostEqual(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray)):
            test_case.assertEqual(len(expected), len(actual))
            for index in xrange(len(expected)):
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


class TestTransmissionConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.conf = TransmissionConfig(os.path.join(BASE_DIR, 'test.cfg'))

    def test_get_path(self):

        file_ = __file__
        path_ = '../tests/output'
        path_only = '/'.join(__file__.split('/')[:-1])
        expected = os.path.abspath(os.path.join(path_only, path_))
        result = self.conf.get_path(path_, file_)
        self.assertEqual(result, expected)

    def test_split_str(self):

        str_ = 'aa: 44'
        expected = ('aa', 44)
        result = self.conf.split_str(str_, ':')
        self.assertEqual(result, expected)

    def test_read_design_value(self):
        design_value = StringIO.StringIO("""\
lineroute, design wind speed, design wind span, terrain category, design level
Calaca - Amadeo, 75.0, 400.0, 2, low
Calaca - Santa Rosa, 51.389, 400.0, 2, low""")

        with open(self.conf.file_design_value, 'r') as file1:
            for line1, line2 in zip(file1, design_value):
                self.assertEqual(line1, line2)

        expected = {'Calaca - Amadeo': {'cat': 2,
                                        'level': 'low',
                                        'span': 400.0,
                                        'speed': 75.0},
                    'Calaca - Santa Rosa': {'cat': 2,
                                            'level': 'low',
                                            'span': 400.0,
                                            'speed': 51.389}}

        assertDeepAlmostEqual(self, expected, self.conf.design_value)

    def test_read_fragility(self):

        fragility_metadata = StringIO.StringIO("""\
[main]
by: Type, Function, DevAngle
type: string, string, numeric
limit_states: minor, collapse
function: form
file: ./fragility.csv

[lognorm]
scale: param1
arg: param2""")

        fragility = StringIO.StringIO("""\
Type,Function,DevAngle_lower,DevAngle_upper,limit_states,form,param1,param2
Lattice Tower,Suspension,0,360,minor,lognorm,1.02,0.02
Lattice Tower,Suspension,0,360,collapse,lognorm,1.05,0.03
Lattice Tower,Terminal,0,360,minor,lognorm,1.02,0.02
Lattice Tower,Terminal,0,360,collapse,lognorm,1.05,0.03
Lattice Tower,Strainer,0,5,minor,lognorm,1.143,0.032
Lattice Tower,Strainer,0,5,collapse,lognorm,1.18,0.04
Lattice Tower,Strainer,5,15,minor,lognorm,1.173,0.032
Lattice Tower,Strainer,5,15,collapse,lognorm,1.21,0.04
Lattice Tower,Strainer,15,30,minor,lognorm,1.208,0.032
Lattice Tower,Strainer,15,30,collapse,lognorm,1.245,0.04
Lattice Tower,Strainer,30,360,minor,lognorm,1.243,0.032
Lattice Tower,Strainer,30,360,collapse,lognorm,1.28,0.04
Steel Pole,Suspension,0,360,minor,lognorm,3.85,0.05
Steel Pole,Suspension,0,360,collapse,lognorm,4.02,0.05
Steel Pole,Terminal,0,360,minor,lognorm,3.85,0.05
Steel Pole,Terminal,0,360,collapse,lognorm,4.02,0.05
Steel Pole,Strainer,0,5,minor,lognorm,3.85,0.05
Steel Pole,Strainer,0,5,collapse,lognorm,4.02,0.05
Steel Pole,Strainer,5,15,minor,lognorm,3.95,0.05
Steel Pole,Strainer,5,15,collapse,lognorm,4.12,0.05
Steel Pole,Strainer,15,30,minor,lognorm,3.05,0.05
Steel Pole,Strainer,15,30,collapse,lognorm,4.22,0.05
Steel Pole,Strainer,30,360,minor,lognorm,3.15,0.05
Steel Pole,Strainer,30,360,collapse,lognorm,4.32,0.05""")

        expected_metadata = OrderedDict([('__name__', 'main'),
            ('by', ['Type', 'Function', 'DevAngle']),
            ('type', ['string', 'string', 'numeric']),
            ('limit_states', ['minor', 'collapse']),
            ('function', 'form'),
            ('file', './fragility.csv'),
            ('lognorm', OrderedDict([('__name__', 'lognorm'),
            ('scale', 'param1'), ('arg', 'param2')]))])

        expected_metadata['file'] = os.path.abspath(
            os.path.join(os.path.abspath(
                self.conf.file_fragility_metadata),
                '../',
                expected_metadata['file']))

        with open(self.conf.file_fragility_metadata, 'r') as file1:
            for line1, line2 in zip(file1, fragility_metadata):
                self.assertEqual(line1, line2)

        with open(self.conf.fragility_metadata['file'], 'r') as file1:
            for line1, line2 in zip(file1, fragility):
                self.assertEqual(line1, line2)

        assertDeepAlmostEqual(self, expected_metadata,
                              self.conf.fragility_metadata)

        fragility.seek(0)
        expected_fragility = pd.read_csv(fragility)
        pd.util.testing.assert_frame_equal(
            self.conf.fragility, expected_fragility)

    def test_read_cond_collapse_prob(self):

        cond_collapse_prob_metadata = StringIO.StringIO("""\
[main]
by: Function
list: Suspension, Terminal, Strainer

[Suspension]
by: Height
type: numeric
file: ./cond_collapse_prob_suspension_terminal.csv

[Terminal]
by: Height
type: numeric
file: ./cond_collapse_prob_suspension_terminal.csv

[Strainer]
by: design_level
type: string
file: ./cond_collapse_prob_strainer.csv""")

        cond_collapse_prob_suspension_terminal = StringIO.StringIO("""\
Function, Height_lower, Height_upper, no_collapse, probability, start, end
Suspension, 0, 40, 1, 0.075, 0, 1
Suspension, 0, 40, 1, 0.075, -1, 0
Suspension, 0, 40, 2, 0.35, -1, 1
Suspension, 0, 40, 3, 0.025, -1, 2
Suspension, 0, 40, 3, 0.025, -2, 1
Suspension, 0, 40, 4, 0.10, -2, 2
Terminal, 0, 40, 1, 0.075, 0, 1
Terminal, 0, 40, 1, 0.075, -1, 0
Terminal, 0, 40, 2, 0.35, -1, 1
Terminal, 0, 40, 3, 0.025, -1, 2
Terminal, 0, 40, 3, 0.025, -2, 1
Terminal, 0, 40, 4, 0.10, -2, 2""")

        cond_collapse_prob_strainer = StringIO.StringIO("""\
Function, design_level, no_collapse, probability, start, end
Strainer, low, 2, 0.05, -1, 1
Strainer, low, 4, 0.08, -2, 2
Strainer, low, 6, 0.10, -3, 3
Strainer, low, 8, 0.08, -4, 4
Strainer, low, 10, 0.05, -5, 5
Strainer, low, 12, 0.62, -6, 6
Strainer, high, 2, 0.15, -1, 1
Strainer, high, 4, 0.22, -2, 2
Strainer, high, 6, 0.22, -3, 3
Strainer, high, 8, 0.15, -4, 4
Strainer, high, 10, 0.12, -5, 5
Strainer, high, 12, 0.08, -6, 6""")

        expected_metadata = OrderedDict([('__name__', 'main'),
            ('by', 'Function'),
            ('list', ['Suspension', 'Terminal', 'Strainer']),
            ('Suspension', OrderedDict([('__name__', 'Suspension'),
                ('by', 'Height'),
                ('type', 'numeric'),
                ('max_adj', 2),
                ('file', './cond_collapse_prob_suspension_terminal.csv')])),
            ('Terminal', OrderedDict([('__name__', 'Terminal'),
                ('by', 'Height'),
                ('type', 'numeric'),
                ('max_adj', 2),
                ('file', './cond_collapse_prob_suspension_terminal.csv')])),
            ('Strainer', OrderedDict([('__name__', 'Strainer'),
                ('by', 'design_level'),
                ('type', 'string'),
                ('max_adj', 6),
                ('file', './cond_collapse_prob_strainer.csv')]))])

        with open(self.conf.file_cond_collapse_prob_metadata, 'r') as file1:
            for line1, line2 in zip(file1, cond_collapse_prob_metadata):
                self.assertEqual(line1, line2)

        with open(self.conf.cond_collapse_prob_metadata['Strainer']['file'], 'r') as file1:
            for line1, line2 in zip(file1, cond_collapse_prob_strainer):
                self.assertEqual(line1, line2)

        with open(self.conf.cond_collapse_prob_metadata['Suspension']['file'], 'r') as file1:
            for line1, line2 in zip(file1, cond_collapse_prob_suspension_terminal):
                self.assertEqual(line1, line2)

        path_input = os.path.join(os.path.abspath(
            self.conf.file_fragility_metadata), '../')

        for item in expected_metadata['list']:
            expected_metadata[item]['file'] = os.path.abspath(
                os.path.join(path_input, expected_metadata[item]['file']))

        assertDeepAlmostEqual(self, expected_metadata,
                              self.conf.cond_collapse_prob_metadata)

        expected_cond_pc = dict()
        for item in expected_metadata['list']:
            df_tmp = pd.read_csv(expected_metadata[item]['file'], skipinitialspace=1)
            df_tmp['start'] = df_tmp['start'].astype(np.int64)
            df_tmp['end'] = df_tmp['end'].astype(np.int64)
            df_tmp['list'] = df_tmp.apply(lambda x: tuple(range(x['start'], x['end'] + 1)), axis=1)
            expected_cond_pc[item] = df_tmp.loc[df_tmp[expected_metadata['by']] == item]

            pd.util.testing.assert_frame_equal(
                self.conf.cond_collapse_prob[item], expected_cond_pc[item])

    def test_read_ASNZS_terrain_multiplier(self):

        terrain_multiplier = StringIO.StringIO("""\
height(m), terrain category 1, terrain category 2, terrain category 3, terrain category 4
3,   0.99,  0.91,  0.83,  0.75
5,   1.05,  0.91,  0.83,  0.75
10,  1.12,  1.00,  0.83,  0.75
15,  1.16,  1.05,  0.89,  0.75
20,  1.19,  1.08,  0.94,  0.75
30,  1.22,  1.12,  1.00,  0.80
40,  1.24,  1.16,  1.04,  0.85
50,  1.25,  1.18,  1.07,  0.90
75,  1.27,  1.22,  1.12,  0.98
100, 1.29,  1.24,  1.16,  1.03
150, 1.31,  1.27,  1.21,  1.11
200, 1.32,  1.29,  1.24,  1.16""")

        with open(self.conf.file_terrain_multiplier, 'r') as file1:
            for line1, line2 in zip(file1, terrain_multiplier):
                self.assertEqual(line1, line2)

        terrain_multiplier.seek(0)
        expected = pd.read_csv(terrain_multiplier, skipinitialspace=True)
        expected.columns = ['height', 'tc1', 'tc2', 'tc3', 'tc4']

        assertDeepAlmostEqual(self, expected.to_dict('list'),
                              self.conf.terrain_multiplier)

    def test_read_drag_height_by_type(self):
        drag_height = StringIO.StringIO("""\
# typical drag height by tower type
Suspension,15.4
Strainer,12.2
Terminal,12.2""")

        with open(self.conf.file_drag_height_by_type, 'r') as file1:
            for line1, line2 in zip(file1, drag_height):
                self.assertEqual(line1, line2)

        drag_height.seek(0)
        expected = pd.read_csv(drag_height, skipinitialspace=True, index_col=0)
        expected.columns = ['value']

        assertDeepAlmostEqual(self, expected['value'].to_dict(),
                              self.conf.drag_height)

    def test_read_topographic_multiplier(self):

        topo_multiplier = StringIO.StringIO("""\
Station,Time,Longitude,Latitude,Speed,UU,VV,Bearing,Pressure,Mh,Mhopp
AC-099,2014-07-16 00:30,120.80446, 13.93832, 55.49, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000
AC-100,2014-07-16 00:30,120.80279, 13.93657, 55.07, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000
AC-101,2014-07-16 00:30,120.80041, 13.93543, 55.20, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000
AC-102,2014-07-16 00:20,120.79799, 13.93429, 57.51, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
AC-103,2014-07-16 00:20,120.79462, 13.93344, 56.63, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
AC-104,2014-07-16 00:20,120.79201, 13.93304, 54.48, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
CB-001,2014-07-16 00:20,120.79210, 13.93267, 54.46, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
CB-002,2014-07-16 00:20,120.79493, 13.93333, 57.53, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
CB-003,2014-07-16 00:20,120.79793, 13.93403, 57.42, 44.17, 44.43,224.83,97374.38,  1.00000,  1.00000
CB-004,2014-07-16 00:30,120.80082, 13.93542, 55.48, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000
CB-005,2014-07-16 00:30,120.80287, 13.93640, 55.18, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000
CB-006,2014-07-16 00:30,120.80513, 13.93877, 54.56, 45.59, 43.98,226.03,97664.55,  1.00000,  1.00000""")

        with open(self.conf.file_topo_multiplier, 'r') as file1:
            for line1, line2 in zip(file1, topo_multiplier):
                self.assertEqual(line1, line2)

        topo_multiplier.seek(0)
        expected = pd.read_csv(topo_multiplier, skipinitialspace=True)
        expected['topo'] = expected[['Mh', 'Mhopp']].max(axis=1)

        assertDeepAlmostEqual(self, expected.set_index('Station').to_dict()['topo'],
                              self.conf.topo_multiplier)

    def test_read_design_adjustment_factor_by_topography_mutliplier(self):

        design_adj_factor = StringIO.StringIO("""\
[main]
threshold: 1.05, 1.1, 1.2, 1.3, 1.45
# key = np.sum(x > threshold), value
0:1.0
1:1.1
2:1.2
3:1.3
4:1.45
5:1.6""")

        expected = {0: 1.0,
                    1: 1.1,
                    2: 1.2,
                    3: 1.3,
                    4: 1.45,
                    5: 1.60,
                    'threshold': np.array([1.05, 1.1, 1.2, 1.3, 1.45])}

        with open(self.conf.file_design_adjustment_factor_by_topo, 'r') as file1:
            for line1, line2 in zip(file1, design_adj_factor):
                self.assertEqual(line1, line2)

        assertDeepAlmostEqual(self, expected,
                              self.conf.design_adjustment_factor_by_topo)

    def test_run(self):

        expected = {'test1': {'Calaca - Amadeo': 11,
                              'Calaca - Santa Rosa': 22},
                    'test2': {'Calaca - Amadeo': 12,
                              'Calaca - Santa Rosa': 23}}

        assertDeepAlmostEqual(self, expected, self.conf.seed)

if __name__ == '__main__':
    unittest.main()
