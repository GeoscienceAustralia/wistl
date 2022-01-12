#!/usr/bin/env python

__author__ = 'Hyeuk Ryu'

import unittest
import logging
import copy
import pandas as pd
import os
import tempfile
import numpy as np
from scipy import stats

from wistl.config import Config, Event
#from wistl.constants import RTOL, ATOL
from wistl.tower import *
from wistl.line import Line
from wistl.tests.test_config import assertDeepAlmostEqual

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

RTOL = 0.05
ATOL = 0.001
PM_THRESHOLD = 1.0e-3

def create_wind_given_bearing(bearing, ratio):

    if isinstance(bearing, list):
        assert len(bearing) == len(ratio)
        df = pd.DataFrame(np.array([ratio, bearing]).T, columns=['ratio', 'Bearing'])
        nperiods = len(bearing)
    else:
        df = pd.DataFrame([[ratio, bearing], [ratio, bearing]], columns=['ratio', 'Bearing'])
        nperiods = 2

    df['Time'] = pd.date_range(start='01/01/2011', periods=nperiods, freq='D')
    df.set_index('Time', inplace=True)

    return df


class TestTower1(unittest.TestCase):
    # suspension tower
    @classmethod
    def setUpClass(cls):

        cls.logger = logging.getLogger(__name__)

        frag_dic = {11.5: {'minor': ['lognorm', '1.02', '0.02'],
                           'collapse': ['lognorm', '1.05', '0.02']},
                    28.75: {'minor': ['lognorm', '1.0', '0.02'],
                            'collapse': ['lognorm', '1.02', '0.02']},
                    41.25: {'minor': ['lognorm', '1.04', '0.02'],
                            'collapse': ['lognorm', '1.07', '0.02']},
                    90: {'minor': ['lognorm', '-1.05', '0.02'],
                         'collapse': ['lognorm', '-1.05', '0.02']},
                    }

        cond_pc = {
            (0, 1): 0.075,
            (-1, 0): 0.075,
            (-1, 0, 1): 0.35,
            (-1, 0, 1, 2): 0.025,
            (-2, -1, 0, 1): 0.025,
            (-2, -1, 0, 1, 2): 0.1}

        cond_pc_adj = {
            12: 0.575,
            14: 0.575,
            15: 0.125,
            11: 0.125}

        cond_pc_adj_sim_idx = [(12, 14, 15), (11, 12, 14), (14,), (12,), (11, 12, 14, 15), (12, 14)]

        cond_pc_adj_sim_prob = np.array([0.025, 0.05 , 0.125, 0.2  , 0.3  , 0.65 ])

        cls.tower_dic = {
            'type': 'Lattice Tower',
            'name': 'T14',
            'latitude': 0.0,
            'longitude': 149.0,
            #'comment': 'Test',
            'function': 'Suspension',
            'devangle': 0,
            'axisaz': 134,
            #'constcost': 0.0,
            'height': 17.0,
            #'yrbuilt': 1980,
            #'locsource': 'Fake',
            'lineroute': 'LineA',
            #'shapes': <shapefile.Shape object at 0x7ff06908ec50>,
            'coord': np.array([149.065,   0.   ]),
            'coord_lat_lon': np.array([  0.   , 149.065]),
            #'point': <shapely.geometry.point.Point object at 0x7ff06908e320>,
            'design_span': 400.0,
            'design_level': 'low',
            'design_speed': 75.0,
            'terrain_cat': 2,
            'file_wind_base_name': 'ts.T14.csv',
            'height_z': 15.4,
            'ratio_z_to_10': 1.0524,
            'actual_span': 556.5974539658616,
            'u_factor': 1.0,
            'collapse_capacity': 75.0,
            'cond_pc': cond_pc,
            'max_no_adj_towers': 2,
            'id_adj': [11, 12, 13, 14, 15],
            'idl': 13,
            #idn': 0,
            'cond_pc_adj': cond_pc_adj,
            'cond_pc_adj_sim_idx': cond_pc_adj_sim_idx,
            'cond_pc_adj_sim_prob': cond_pc_adj_sim_prob,
            #'no_sims': 1000,
            #'damage_states': ['minor', 'collapse'],
            #'non_collapse': ['minor'],
            #'rnd_state': np.random.RandomState(1),
            #'event_id': 0,
            #'scale': 1.0,
            'frag_dic': frag_dic,
            #'rtol': RTOL,
            #'atol': ATOL,
            #'dmg_threshold': PM_THRESHOLD,
            #'path_event': os.path.join(BASE_DIR, 'wind_event/test1'),
            'shape': None,
            'point': None,
            }

        cls.tower = Tower(**cls.tower_dic)

        cls.event = Event(**{'id': 'test1_s1.0',
                           'path_wind_event': os.path.join(BASE_DIR, 'wind_event/test1'),
                           'name': 'test1',
                           'scale': 1.0,
                           'seed': 1,
                           })

        cls.line_dic = {
            'linename': 'LineA',
            'type': 'HV Transmission Line',
            'capacity': 230,
             'numcircuit': 2,
             'shapes': None,
             'coord': np.array([[149.   ,   0.   ], [149.105,   0.   ]]),
             'coord_lat_lon': [[0.0, 149.0],[0.0, 149.105]],
             'line_string': None,
             'name_output': 'LineA',
             'no_towers': 22,
             'actual_span': np.array([278.29872698, 556.59745397]),
             'seed': 0,
             'names': ['T14', 'T22'],
             }

        cls.line = Line(**cls.line_dic)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'))

        cls.wind = pd.DataFrame([[0.1, 130.0], [0.9605, 130.0], [0.1, 130.0], [0.97, 130.0]], columns=['ratio', 'Bearing'])
        cls.wind['Time'] = pd.date_range(start='01/01/2011', end='01/04/2011', freq='D')
        cls.wind.set_index('Time', inplace=True)

    def test_compute_dmg_by_tower(self):

        result = compute_dmg_by_tower(self.tower, self.event, self.line, self.cfg)
        self.assertListEqual([*result], ['event', 'line', 'tower', 'dmg', 'collapse_adj', 'dmg_state_sim',  'collapse_adj_sim'])
        self.assertEqual(result['event'], 'test1_s1.0')
        self.assertEqual(result['line'], 'LineA')
        self.assertEqual(result['tower'], 'T14')
        self.assertTrue(result['dmg'].empty)
        #self.assertTrue(result['collapse_adj'].empty)

    def test_logger_file_wind1(self):
        event_dummy = Event(**{'id': 'test1_s1.0',
                            'path_wind_event': 'dummy_path',
                            'name': 'test1',
                            'scale': 1.0,
                            'seed': 1,
                            })

        with self.assertLogs('distributed.worker', level='INFO') as cm:
            read_wind(self.tower, event_dummy)
        msg = f'Invalid file_wind {event_dummy.path_wind_event}/{self.tower.file_wind_base_name}'
        self.assertIn(f'{msg}', cm.output[0])

    def test_logger_file_wind2(self):

        dic1 = self.tower_dic.copy()
        dic1.update({'file_wind_base_name' : 'dummy'})
        tower1 = Tower(**dic1)

        with self.assertLogs('distributed.worker', level='INFO') as cm:
            read_wind(tower1, self.event)
        msg = f'Invalid file_wind {self.event.path_wind_event}/dummy'
        self.assertIn(f'{msg}', cm.output[0])

    def test_logger_dmg_sim(self):

        with self.assertLogs('distributed.worker', level='INFO') as cm:

            dic1 = self.tower_dic.copy()
            tower = Tower(**dic1)
            self.cfg.no_sims = 1
            # 1. determine damage state of tower due to wind
            wind = create_wind_given_bearing(130.0, 1.0712)  # 1.05*np.exp(0.02)

            dmg = set_dmg(tower, wind, self.cfg)

            dmg_state_sim = set_dmg_state_sim(dmg, self.cfg, np.random.RandomState(1))

            check_sim_accuracy(tower, dmg_state_sim, dmg, self.event, self.cfg)

        msg = f'PE(collapse of {tower.name}|{self.event.id}'
        self.assertIn(msg, cm.output[0])

    def test_logger_collapse_adj_sim(self):

        with self.assertLogs('distributed.worker', level='INFO') as cm:

            rnd_state = np.random.RandomState(1)

            self.cfg.no_sims = 1

            wind = create_wind_given_bearing(130.0, 1.0712)  # 1.05*np.exp(0.02)

            dmg = set_dmg(self.tower, wind, self.cfg)
            collapse_adj = set_collapse_adj(self.tower, dmg)

            dmg_state_sim = set_dmg_state_sim(dmg, self.cfg, rnd_state)
            collapse_adj_sim = set_collapse_adj_sim(self.tower, dmg_state_sim, collapse_adj, rnd_state, self.cfg)
            _ = check_against_collapse_adj(collapse_adj_sim, collapse_adj, self.tower, self.cfg)

        msg = 'Pc'
        self.assertIn(msg, cm.output[0])

    @unittest.skip("NOT YET")
    def test_logger_collapse_interaction(self):

        with self.assertLogs('distributed.worker', level='INFO') as cm:
            tower_dic = self.tower_dic.copy()
            tower_dic.update({'axisaz': 91, 'no_sims': 50, 'cond_pc_interaction_no': [1, 3, 5, 7],
                              'cond_pc_interaction_cprob': [0.2, 0.3, 0.31, 0.311],
                              'cond_pc_interaction_prob': {1:0.2, 3:0.1, 5:0.01, 7:0.001}})
            tower = Tower(**tower_dic)
            tower.init()
            prob_dic = {1: 0.2, 3: 0.1, 5: 0.01, 7: 0.001}
            tower._wind = create_wind_given_bearing(130.0, 1.072)
            tower.collapse_interaction

        msg = f'WARNING:wistl.tower:Pc_interaction({tower.name})'
        self.assertEqual(msg, ':'.join(cm.output[0].split(':')[:3]))

    def test_sorted_frag_dic_keys(self):

        self.assertEqual(sorted(self.tower.frag_dic.keys()), [11.5, 28.75, 41.25, 90.0])

    def test_file_wind(self):

        assert self.tower.name == 'T14'
        self.assertEqual(self.event.path_wind_event, os.path.join(BASE_DIR, 'wind_event/test1'))
        self.assertEqual(self.tower.file_wind_base_name, 'ts.T14.csv')
        #expected = os.path.join(BASE_DIR, 'wind_event/test1', 'ts.T14.csv')
        #self.assertEqual(self.tower.file_wind, expected)

    def test_angle_between_two(self):

        deg1 = [  0.,   30.,  60.,
                 90.,  120., 150.,
                180.,  210., 240.,
                270.,  300., 330.]

        d2 = 0
        expected = [0, 30, 60,
                   90, 60, 30,
                    0, 30, 60,
                   90, 60, 30]

        for d1, e in zip(deg1, expected):
            result = angle_between_two(d1, d2)
            self.assertAlmostEqual(result, e)

        d2 = 90
        expected = [90, 60, 30,
                    0, 30, 60,
                    90, 60, 30,
                    0, 30, 60]

        for d1, e in zip(deg1, expected):
            result = angle_between_two(d1, d2)
            self.assertAlmostEqual(result, e)

        d2 = 180
        expected = [0, 30, 60,
                   90, 60, 30,
                    0, 30, 60,
                   90, 60, 30]

        for d1, e in zip(deg1, expected):
            result = angle_between_two(d1, d2)
            self.assertAlmostEqual(result, e)

        d2 = 270
        expected = [90, 60, 30,
                   0, 30, 60,
                   90, 60, 30,
                   0, 30, 60]

        for d1, e in zip(deg1, expected):
            result = angle_between_two(d1, d2)
            self.assertAlmostEqual(result, e)


    def test_get_directional_vulnerability1(self):
        # thresholds: 11.5, 28.75, 41.25, 90.0 

        tower_dic = self.tower_dic.copy()
        tower_dic.update({'axisaz': 90})

        bearings = [10.0, 45.0, 60.0, 70.0, 80.0,
                    170.0, 135.0, 120.0, 110.0, 100.0,
                    190.0, 225.0, 240.0, 250.0, 260.0,
                    350.0, 315.0, 300.0, 290.0, 280.0]
        expected = [90.0, 90.0, 41.25, 28.75, 11.5] * 4

        for bearing, value in zip(bearings, expected):

            tower = Tower(**tower_dic)
            result = get_directional_vulnerability(tower, bearing)
            try:
                self.assertAlmostEqual(result, value)
            except AssertionError:
                print(f'Wrong: bearing:{bearing}, axisaz: {tower_dic["axisaz"]}, result:{result}, expected: {value}')

    def test_get_directional_vulnerability2(self):
        # thresholds: 11.5, 28.75, 41.25, 90.0 

        tower_dic = self.tower_dic.copy()
        tower_dic.update({'axisaz': 0})

        bearings = [  0.,   15.,  30.,  45.,  60.,  75.,
                     90.,  105., 120., 135., 150., 165.,
                     180., 195., 210., 225., 240., 255.,
                     270., 285., 300., 315., 330., 345.]
        expected = [11.5, 28.75, 41.25, 90.0, 90.0, 90.0,
                    90.0, 90.0, 90.0, 90.0, 41.25, 28.75] * 2

        for bearing, value in zip(bearings, expected):

            tower = Tower(**tower_dic)
            result = get_directional_vulnerability(tower, bearing)
            try:
                self.assertAlmostEqual(result, value)
            except AssertionError:
                print(f'Wrong: bearing:{bearing}, axisaz: {tower_dic["axisaz"]}, result:{result}, expected: {value}')

    def test_read_wind(self):

        wind = read_wind(self.tower, self.event)
        file_wind = os.path.join(self.event.path_wind_event, self.tower.file_wind_base_name)
        self.assertTrue(os.path.exists(file_wind))
        self.assertAlmostEqual(self.event.scale, 1.0)
        self.assertAlmostEqual(self.tower.ratio_z_to_10, 1.0524)
        self.assertAlmostEqual(self.tower.collapse_capacity, 75.0)
        self.assertAlmostEqual(wind.loc['2014-07-13 09:00', 'Speed'], 1.0524*3.600014855085)
        self.assertAlmostEqual(wind.loc['2014-07-13 09:05', 'ratio'], 1.0524*3.60644345304/75.0)


    def test_compute_dmg_using_directional_vulnerability(self):

        self.assertAlmostEqual(self.tower.collapse_capacity, 75.0)
        self.assertAlmostEqual(self.tower.axisaz, 134.0)

        bearing_ratio_angle = [(130.0, 1.02, 11.5), (45.0, 1.02, 90.0),
                               (110.0, 1.04, 28.75), (100.0, 1.0, 41.25)]

        for bearing, ratio, angle in bearing_ratio_angle:
            wind = create_wind_given_bearing(bearing, ratio)
            key = get_directional_vulnerability(self.tower, bearing)
            self.assertEqual(key, angle)

            df = wind.apply(compute_dmg_using_directional_vulnerability, args=(self.tower,), axis=1)
            for ds in self.cfg.dmg_states:
                v1 = float(self.tower.frag_dic[key][ds][1])
                v2 = float(self.tower.frag_dic[key][ds][2])
                self.assertAlmostEqual(df.loc['01/01/2011', ds],
                                       np.nan_to_num(stats.lognorm.cdf(ratio, v2, scale=v1)))

    def test_set_dmg(self):

        self.assertAlmostEqual(self.tower.collapse_capacity, 75.0)
        self.assertAlmostEqual(self.tower.axisaz, 134.0)

        bearing_ratio_threshold = [(130.0, 1.02, 0.0005, 2),
                                   (45.0, 1.02, 0.02, 0),
                                   (110.0, 1.04, 0.01, 3),
                                   (100.0, 1.0, 0.0005, 2)]

        for bearing, ratio, threshold, expected in bearing_ratio_threshold:

            wind = create_wind_given_bearing(bearing, ratio)
            key = get_directional_vulnerability(self.tower, bearing)
            v1 = float(self.tower.frag_dic[key]['minor'][1])
            v2 = float(self.tower.frag_dic[key]['minor'][2])
            new_ratio = np.nan_to_num(stats.lognorm.ppf(threshold, v2 , scale=v1))
            new = pd.DataFrame([[new_ratio, bearing]],
                               columns=wind.columns,
                               index=[np.datetime64("2011-01-03")])
            wind = pd.concat([wind, new])
            df = set_dmg(self.tower, wind, self.cfg)

            self.assertEqual(df.shape[0], expected)

    def test_set_collapse_adj(self):

        cond_pc_adj = {11: 0.125, 12: 0.575, 14: 0.575, 15: 0.125}
        assertDeepAlmostEqual(self, dict(self.tower.cond_pc_adj), cond_pc_adj)

        wind = create_wind_given_bearing(130, 1.0712)  # 1.05*np.exp(0.02)

        dmg = set_dmg(self.tower, wind, self.cfg)

        collapse_adj = set_collapse_adj(self.tower, dmg)

        self.assertEqual(self.tower.id_adj, [11, 12, 13, 14, 15])

        for id_abs, value in cond_pc_adj.items():

            self.assertAlmostEqual(collapse_adj[id_abs][0],
                                   0.842 * value, places=2)

    def test_check_sim_accuracy(self):

        # 1. determine damage state of tower due to wind
        wind = create_wind_given_bearing(130.0, 1.0712)  # 1.05*np.exp(0.02)

        #self.cfg.no_sims = 10000
        dmg = set_dmg(self.tower, wind, self.cfg)

        self.assertAlmostEqual(dmg['collapse'].values[0],
                               stats.lognorm.cdf(1.0712, 0.02, scale=1.05), places=2)
        self.assertAlmostEqual(dmg['minor'].values[0],
                               stats.lognorm.cdf(1.0712, 0.02, scale=1.02), places=2)

        dmg_state_sim = set_dmg_state_sim(dmg, self.cfg, np.random.RandomState(1))

        dmg_sim = check_sim_accuracy(self.tower, dmg_state_sim, dmg, self.event, self.cfg)

        pd.testing.assert_frame_equal(dmg_sim, dmg, rtol=self.cfg.rtol, atol=self.cfg.atol)

    def test_dmg_state_sim_old(self):

        wind = create_wind_given_bearing(130.0, 1.0712)  # 1.05*np.exp(0.02)
        dmg = set_dmg(self.tower, wind, self.cfg)
        rv = stats.uniform.rvs(size=(self.cfg.no_sims, dmg.shape[0]))

        a = np.array([rv < dmg[ds].values
                      for ds in self.cfg.dmg_states]).sum(axis=0)

        b = (rv[:, :, np.newaxis] < dmg.values).sum(axis=2)

        np.testing.assert_array_equal(a, b)

    def test_dmg_threshold(self):

        dmg = set_dmg(self.tower, self.wind, self.cfg)

        # checking index
        pd.testing.assert_index_equal(dmg.index, self.wind.index[1:3+1])

    def test_dmg_idxmax(self):

        dmg = set_dmg(self.tower, self.wind, self.cfg)

        # checking index
        for ds in self.cfg.dmg_states:
            self.assertEqual(dmg[ds].idxmax(), self.wind.index[3])

    def test_compare_dmg_with_dmg_sim(self):

        dmg = set_dmg(self.tower, self.wind, self.cfg)

        dmg_state_sim = set_dmg_state_sim(dmg, self.cfg, np.random.RandomState(1))

        # dmg_isolated vs. dmg_sim
        prob_sim = 0
        for ds in self.cfg.dmg_states[::-1]:
            no = len(dmg_state_sim[ds]['id_sim'][dmg_state_sim[ds]['id_time'] == 1])
            prob_sim += no / self.cfg.no_sims
            isclose = np.isclose(prob_sim,
                                 dmg[ds].values[1],
                                 rtol=RTOL, atol=ATOL)
            if not isclose:
                self.logger.warning(f'PE of {ds}: '
                                    f'simulation {prob_sim:.3f} vs. '
                                    f'analytical {self.tower.dmg[ds].values[1]:.3f}')

    def test_dmg_state_sim(self):

        rv = np.array([[0, 0], [1, 1], [0.5, 0.9]])   # no_sims, no_time
        dmg = pd.DataFrame(np.array([[0.9928, 0.8412],
                                     [0.9928, 0.8412]]), columns=['minor', 'collapse'])
        _array = (rv[:, :, np.newaxis] < dmg.values).sum(axis=2)

        np.testing.assert_equal(_array, np.array([[2, 2], [0, 0], [2, 1]]))

        dmg_state_sim = {}
        for ids, ds in enumerate(self.cfg.dmg_states, 1):
            id_sim, id_time = np.where(_array == ids)
            dmg_state_sim[ds] = pd.DataFrame(np.vstack((id_sim, id_time)).T, columns=['id_sim', 'id_time'])

        np.testing.assert_equal(dmg_state_sim['minor']['id_sim'].values, np.array([2]))
        np.testing.assert_equal(dmg_state_sim['minor']['id_time'].values, np.array([1]))

        np.testing.assert_equal(dmg_state_sim['collapse']['id_sim'].values, np.array([0, 0, 2]))
        np.testing.assert_equal(dmg_state_sim['collapse']['id_time'].values, np.array([0, 1, 0]))

    def test_dmg_state_sim_threshold(self):

        rv = np.array([[0.9, 0.5, 0],
                      [0.1, 0.5, 0.9],
                      [0.5, 0.9, 0.7]])   # no_sims, no_time
        dmg = pd.DataFrame(np.array([[0.9928, 0.8412],
                                     [0.0, 0.0],
                                     [0.9928, 0.8412]]), columns=['minor', 'collapse'])
        _array = (rv[:, :, np.newaxis] < dmg.values).sum(axis=2)

        np.testing.assert_equal(_array, np.array([[1, 0, 2], [2, 0, 1], [2, 0, 2]]))  # no_sims, no_time

        dmg_state_sim = {}
        for ids, ds in enumerate(self.cfg.dmg_states, 1):
            id_sim, id_time = np.where(_array == ids)
            dmg_state_sim[ds] = pd.DataFrame(np.vstack((id_sim, id_time)).T, columns=['id_sim', 'id_time'])

        np.testing.assert_equal(dmg_state_sim['minor']['id_sim'].values, np.array([0, 1]))
        np.testing.assert_equal(dmg_state_sim['minor']['id_time'].values, np.array([0, 2]))

        np.testing.assert_equal(dmg_state_sim['collapse']['id_sim'].values, np.array([0, 1, 2, 2]))
        np.testing.assert_equal(dmg_state_sim['collapse']['id_time'].values, np.array([2, 0, 0, 2]))

    def test_collapse_adj_sim(self):

        rnd_state = np.random.RandomState(1)
        self.cfg.no_sims = 10000
        # tower14 (idl: 13,
        tower_dic = self.tower_dic.copy()
        tower_dic.update({'axisaz': 90})

        tower = Tower(**tower_dic)
        wind = create_wind_given_bearing([130, 130, 120, 130],[0.0712, 1.0712, 1.0712, 0.0712])  # 1.05*np.exp(0.02)
        dmg = set_dmg(tower, wind, self.cfg)
        collapse_adj = set_collapse_adj(tower, dmg)
        dmg_state_sim = set_dmg_state_sim(dmg, self.cfg, rnd_state)
        collapse_adj_sim = set_collapse_adj_sim(tower, dmg_state_sim, collapse_adj, rnd_state, self.cfg)
        df = collapse_adj_sim.groupby(['id_time','id_adj']).apply(len).reset_index()
        for idl in tower.cond_pc_adj.keys():
            x = df.loc[df['id_adj'].apply(lambda x: idl in x)].groupby('id_time').sum()/self.cfg.no_sims
            np.testing.assert_allclose(x[0].values, collapse_adj[idl], atol=ATOL, rtol=RTOL)

    @unittest.skip("NOT YET")
    def test_collapse_interaction(self):

        tower_dic = self.tower_dic.copy()
        tower_dic.update({'axisaz': 91, 'no_sims': 5000, 'cond_pc_interaction_no': [1, 3, 5, 7],
                          'cond_pc_interaction_cprob': [0.2, 0.3, 0.31, 0.311],
                          'cond_pc_interaction_prob': {1:0.2, 3:0.1, 5:0.01, 7:0.001}})
        tower = Tower(**tower_dic)
        wind = create_wind_given_bearing(130.0, 1.072)

        #tower.collapse_interaction
        x = tower.collapse_interaction.groupby(['id_time', 'no_collapse']).apply(len).reset_index()
        for _, row in x.iterrows():
            expected = tower.dmg['collapse'].iloc[row['id_time']] * tower.cond_pc_interaction_prob[row['no_collapse']]
            result = row[0] / tower.no_sims
            np.testing.assert_allclose(expected, result, atol=ATOL, rtol=RTOL)




class TestTower2(unittest.TestCase):
    # strainer 
    @classmethod
    def setUpClass(cls):
        # 23, T9
        cls.logger = logging.getLogger(__name__)

        frag_dic = {180: {'minor': ['lognorm','1.143','0.032'],
                          'collapse': ['lognorm','1.18','0.04']}
                   }

        cond_pc = {
            (-1,0,1): 0.05,
            (-2,-1,0,1,2): 0.08,
            (-3,-2,-1,0,1,2,3): 0.10,
            (-4,-3,-2,-1,0,1,2,3,4): 0.08,
            (-5,-4,-3,-2,-1,0,1,2,3,4,5): 0.05,
            (-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6): 0.62,
            }

        cond_pc_adj = {
            2: 0.62,
            3: 0.67,
            4: 0.75,
            5: 0.85,
            6: 0.93,
            7: 0.98,
            9: 0.98,
            10: 0.93,
            11: 0.85,
            12: 0.75,
            13: 0.67,
            14: 0.62}

        cond_pc_adj_sim_idx = [
            (7, 9),
            (3, 4, 5, 6, 7, 9, 10, 11, 12, 13),
            (6, 7, 9, 10),
            (4, 5, 6, 7, 9, 10, 11, 12),
            (5, 6, 7, 9, 10, 11),
            (2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14)]

        cond_pc_adj_sim_prob = np.array([0.05, 0.10 , 0.18, 0.26, 0.36, 0.98])

        cls.tower_dic = {
            'type': 'Lattice Tower',
            'name': 'T9',
            'latitude': 0.0,
            'longitude': 149.0,
            #'comment': 'Test',
            'function': 'Suspension',
            'devangle': 0,
            'axisaz': 134,
            #'constcost': 0.0,
            'height': 17.0,
            #'yrbuilt': 1980,
            #'locsource': 'Fake',
            'lineroute': 'LineA',
            #'shapes': <shapefile.Shape object at 0x7ff06908ec50>,
            'coord': np.array([149.065,   0.   ]),
            'coord_lat_lon': np.array([  0.   , 149.065]),
            #'point': <shapely.geometry.point.Point object at 0x7ff06908e320>,
            'design_span': 400.0,
            'design_level': 'low',
            'design_speed': 75.0,
            'terrain_cat': 2,
            'file_wind_base_name': 'ts.T9.csv',
            'height_z': 15.4,
            'ratio_z_to_10': 1.0524,
            'actual_span': 556.5974539658616,
            'u_factor': 1.0,
            'collapse_capacity': 75.0,
            'cond_pc': cond_pc,
            'max_no_adj_towers': 6,
            'id_adj': [2, 3, 4, 5, 6, 7, -1, 9, 10, 11, 12, 13, 14],
            'idl': 8,
            #'idn': 0,
            'cond_pc_adj': cond_pc_adj,
            'cond_pc_adj_sim_idx': cond_pc_adj_sim_idx,
            'cond_pc_adj_sim_prob': cond_pc_adj_sim_prob,
            #'no_sims': 10000,
            #'damage_states': ['minor', 'collapse'],
            #'non_collapse': ['minor'],
            #'rnd_state': np.random.RandomState(1),
            #'event_id': 0,
            #'rtol': RTOL,
            #'atol': ATOL,
            #'dmg_threshold': PM_THRESHOLD,
            #'scale': 1.0,
            'frag_dic': frag_dic,
            #'path_event': os.path.join(BASE_DIR, 'wind_event/test1'),
            'shape': None,
            'point': None,
            }

        cls.tower = Tower(**cls.tower_dic)

        cls.event = Event(**{'id': 'test1_s1.0',
                           'path_wind_event': os.path.join(BASE_DIR, 'wind_event/test1'),
                           'name': 'test1',
                           'scale': 1.0,
                           'seed': 1,
                           })

        cls.line_dic = {
            'linename': 'LineA',
            'type': 'HV Transmission Line',
            'capacity': 230,
             'numcircuit': 2,
             'shapes': None,
             'coord': np.array([[149.   ,   0.   ], [149.105,   0.   ]]),
             'coord_lat_lon': [[0.0, 149.0],[0.0, 149.105]],
             'line_string': None,
             'name_output': 'LineA',
             'no_towers': 22,
             'actual_span': np.array([278.29872698, 556.59745397]),
             'seed': 0,
             'names': ['T14', 'T22'],
             }

        cls.line = Line(**cls.line_dic)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'))

        cls.wind = pd.DataFrame([[0.1, 110.0], [0.9605, 110.0], [0.1, 110.0], [1.04, 110.0]], columns=['ratio', 'Bearing'])
        cls.wind['Time'] = pd.date_range(start='01/01/2011', end='01/04/2011', freq='D')
        cls.wind.set_index('Time', inplace=True)


    def test_compute_dmg_by_tower(self):

        result = compute_dmg_by_tower(self.tower, self.event, self.line, self.cfg)
        self.assertListEqual([*result], ['event', 'line', 'tower', 'dmg', 'collapse_adj', 'dmg_state_sim',  'collapse_adj_sim'])
        self.assertEqual(result['event'], 'test1_s1.0')
        self.assertEqual(result['line'], 'LineA')
        self.assertEqual(result['tower'], 'T9')
        #self.assertTrue(result['dmg'].empty)
        #self
    def test_sorted_frag_dic_keys(self):

        self.assertEqual(sorted(self.tower.frag_dic.keys()), [180.0])

    def test_file_wind(self):

        assert self.tower.name == 'T9'
        self.assertEqual(self.event.path_wind_event, os.path.join(BASE_DIR, 'wind_event/test1'))
        self.assertEqual(self.tower.file_wind_base_name, 'ts.T9.csv')

    def test_get_directional_vulnerability1(self):
        # thresholds: 11.5, 28.75, 41.25, 90.0 

        tower_dic = self.tower_dic.copy()
        tower_dic.update({'axisaz': 90})

        bearings = [10.0, 45.0, 60.0, 70.0, 80.0,
                    170.0, 135.0, 120.0, 110.0, 100.0,
                    190.0, 225.0, 240.0, 250.0, 260.0,
                    350.0, 315.0, 300.0, 290.0, 280.0]
        expected = [180] * 20

        for bearing, value in zip(bearings, expected):

            tower = Tower(**tower_dic)
            result = get_directional_vulnerability(tower, bearing)
            try:
                self.assertAlmostEqual(result, value)
            except AssertionError:
                print(f'Wrong: bearing:{bearing}, axisaz: {tower_dic["axisaz"]}, result:{result}, expected: {value}')

    def test_set_dmg(self):

        self.assertAlmostEqual(self.tower.collapse_capacity, 75.0)
        self.assertAlmostEqual(self.tower.axisaz, 134.0)

        bearing, ratio = 130.0, 1.02
        wind = create_wind_given_bearing(bearing, ratio)
        key = get_directional_vulnerability(self.tower, bearing)
        self.assertEqual(key, 180)
        df = set_dmg(self.tower, wind, self.cfg)
        self.assertTrue(df.empty)

        bearing, ratio = 110.0, 1.04
        wind = create_wind_given_bearing(bearing, ratio)
        key = get_directional_vulnerability(self.tower, bearing)
        self.assertEqual(key, 180)
        df = set_dmg(self.tower, wind, self.cfg)
        self.assertAlmostEqual(df.loc['01/01/2011', 'minor'],
                         stats.lognorm.cdf(1.04, 0.032, scale=1.143))
        self.assertAlmostEqual(df.loc['01/01/2011', 'collapse'],
                         stats.lognorm.cdf(1.04, 0.04, scale=1.18))

    def test_set_collapse_adj(self):
        cond_pc_adj = {
            2: 0.62,
            3: 0.67,
            4: 0.75,
            5: 0.85,
            6: 0.93,
            7: 0.98,
            9: 0.98,
            10: 0.93,
            11: 0.85,
            12: 0.75,
            13: 0.67,
            14: 0.62}

        assertDeepAlmostEqual(self, dict(self.tower.cond_pc_adj), cond_pc_adj)

        wind = create_wind_given_bearing(130, 1.22816)  # 1.18*np.exp(0.04)

        dmg = set_dmg(self.tower, wind, self.cfg)

        collapse_adj = set_collapse_adj(self.tower, dmg)

        self.assertEqual(self.tower.id_adj, [2, 3, 4, 5, 6, 7, -1, 9, 10, 11, 12, 13, 14])

        for id_abs, value in cond_pc_adj.items():

            self.assertAlmostEqual(collapse_adj[id_abs][0],
                                   0.842 * value, places=2)

    def test_check_sim_accuracy(self):

        # 1. determine damage state of tower due to wind
        wind = create_wind_given_bearing(130.0, 1.22816)  # 1.18*np.exp(0.04)

        dmg = set_dmg(self.tower, wind, self.cfg)

        self.assertAlmostEqual(dmg['collapse'].values[0],
                               stats.lognorm.cdf(1.22816, 0.04, scale=1.18), places=2)
        self.assertAlmostEqual(dmg['minor'].values[0],
                               stats.lognorm.cdf(1.22816, 0.032, scale=1.143), places=2)

        dmg_state_sim = set_dmg_state_sim(dmg, self.cfg, np.random.RandomState(1))

        dmg_sim = check_sim_accuracy(self.tower, dmg_state_sim, dmg, self.event, self.cfg)

        #self.assertAlmostEqual(dmg_sim['collapse'][0], self.tower.dmg['collapse'].values[0], places=2)

        pd.testing.assert_frame_equal(dmg_sim, dmg, rtol=self.cfg.rtol, atol=self.cfg.atol)

    def test_dmg_state_sim_old(self):

        wind = create_wind_given_bearing(130.0, 1.22816)  # 1.18*np.exp(0.04)
        dmg = set_dmg(self.tower, wind, self.cfg)
        rv = stats.uniform.rvs(size=(self.cfg.no_sims, dmg.shape[0]))

        a = np.array([rv < dmg[ds].values
                      for ds in self.cfg.dmg_states]).sum(axis=0)

        b = (rv[:, :, np.newaxis] < dmg.values).sum(axis=2)

        np.testing.assert_array_equal(a, b)

    def test_compare_dmg_with_dmg_sim(self):

        self.cfg.no_sims = 10000
        dmg = set_dmg(self.tower, self.wind, self.cfg)

        dmg_state_sim = set_dmg_state_sim(dmg, self.cfg, np.random.RandomState(1))
        # dmg_isolated vs. dmg_sim
        prob_sim = 0
        for ds in self.cfg.dmg_states[::-1]:
            no = len(dmg_state_sim[ds]['id_sim'][dmg_state_sim[ds]['id_time'] == 0])
            prob_sim += no / self.cfg.no_sims
            isclose = np.isclose(prob_sim,
                                 dmg[ds].values[0],
                                 rtol=self.cfg.rtol, atol=self.cfg.atol)
            if not isclose:
                self.logger.warning(f'PE of {ds}: '
                                    f'simulation {prob_sim:.3f} vs. '
                                    f'analytical {dmg[ds].values[0]:.3f}')

    def test_dmg_state_sim(self):

        wind = create_wind_given_bearing(130.0, 1.22816)  # 1.18*np.exp(0.04)

        dmg = set_dmg(self.tower, wind, self.cfg)

        np.testing.assert_allclose(dmg,
                                   np.array([[0.987637, 0.841361],
                                             [0.987637, 0.841361]]),
                                   rtol=1.e-4)  # minor, collapse
        #self.tower._dmg_state_sim = (rv[:, :, np.newaxis] < self.tower.dmg.values).sum(axis=2)

        #np.testing.assert_equal(self.tower._dmg_state_sim, np.array([[2, 2], [0, 0], [2, 1]]))
        dmg_state_sim = set_dmg_state_sim(dmg, self.cfg, np.random.RandomState(1))

    def test_dmg_state_sim2(self):

        rv = np.array([[0, 0], [1, 1], [0.5, 0.9]])   # no_sims, no_time
        dmg = pd.DataFrame(np.array([[0.987637, 0.841361],
                                    [0.987637, 0.841361]]), columns=['minor', 'collapse'])
        _array = (rv[:, :, np.newaxis] < dmg.values).sum(axis=2)

        np.testing.assert_equal(_array, np.array([[2, 2], [0, 0], [2, 1]]))

        dmg_state_sim = {}
        for ids, ds in enumerate(self.cfg.dmg_states, 1):
            id_sim, id_time = np.where(_array == ids)
            dmg_state_sim[ds] = pd.DataFrame(np.vstack((id_sim, id_time)).T, columns=['id_sim', 'id_time'])

        np.testing.assert_equal(dmg_state_sim['minor']['id_sim'].values, np.array([2]))
        np.testing.assert_equal(dmg_state_sim['minor']['id_time'].values, np.array([1]))

        np.testing.assert_equal(dmg_state_sim['collapse']['id_sim'].values, np.array([0, 0, 2]))
        np.testing.assert_equal(dmg_state_sim['collapse']['id_time'].values, np.array([0, 1, 0]))


    def test_collapse_adj_sim(self):

        rnd_state = np.random.RandomState(1)
        # tower14 (idl: 13,
        tower_dic = self.tower_dic.copy()
        tower_dic.update({'axisaz': 90,
                          #'no_sims': 5000,
                          'function': 'strainer'})

        tower = Tower(**tower_dic)
        wind = create_wind_given_bearing(130.0, 1.2282)  # 1.18*np.exp(0.04)
        dmg = set_dmg(tower, wind, self.cfg)
        collapse_adj = set_collapse_adj(tower, dmg)

        dmg_state_sim = set_dmg_state_sim(dmg, self.cfg, rnd_state)
        collapse_adj_sim = set_collapse_adj_sim(tower, dmg_state_sim, collapse_adj, rnd_state, self.cfg)
        df = collapse_adj_sim.groupby(['id_time','id_adj']).apply(len).reset_index()
        for idl in tower.cond_pc_adj.keys():
            x = df.loc[df['id_adj'].apply(lambda x: idl in x)].groupby('id_time').sum()/self.cfg.no_sims
            np.testing.assert_allclose(x[0].values, collapse_adj[idl], atol=ATOL, rtol=RTOL)


    def test_unit_vector_by_bearing(self):

        result = unit_vector_by_bearing(0)
        expected = np.array([1, 0])
        np.allclose(expected, result)

        result = unit_vector_by_bearing(45.0)
        expected = np.array([0.7071, 0.7071])
        np.allclose(expected, result)

    def test_angle_between_unit_vector(self):

        result = angle_between_unit_vectors((1, 0), (0, 1))
        expected = 90.0
        np.allclose(expected, result)

        result = angle_between_unit_vectors((1, 0), (1, 0))
        expected = 0.0
        np.allclose(expected, result)

        result = angle_between_unit_vectors((1, 0), (-1, 0))
        expected = 180.0
        np.allclose(expected, result)

        result = angle_between_unit_vectors((0.7071, 0.7071), (0, 1))
        expected = 45.0
        np.allclose(expected, result)


class TestTower3(unittest.TestCase):
    # suspension tower neighboring strainer
    @classmethod
    def setUpClass(cls):

        cls.logger = logging.getLogger(__name__)

        frag_dic = {11.5: {'minor': ['lognorm', '1.02', '0.02'],
                           'collapse': ['lognorm', '1.05', '0.02']},
                    28.75: {'minor': ['lognorm', '1.0', '0.02'],
                            'collapse': ['lognorm', '1.02', '0.02']},
                    41.25: {'minor': ['lognorm', '1.04', '0.02'],
                            'collapse': ['lognorm', '1.07', '0.02']},
                    90: {'minor': ['lognorm', '-1.05', '0.02'],
                         'collapse': ['lognorm', '-1.05', '0.02']},
                    }

        cond_pc = {
            (0, 1): 0.075,
            (-1, 0): 0.075,
            (-1, 0, 1): 0.35,
            (-1, 0, 1, 2): 0.025,
            (-2, -1, 0, 1): 0.025,
            (-2, -1, 0, 1, 2): 0.1}

        cond_pc_adj = {
            11: 0.575,
            12: 0.125}

        cond_pc_adj_sim_idx = [(11, 12,), (11, )]

        cond_pc_adj_sim_prob = np.array([0.125, 0.575 ])

        cls.tower_dic = {
            'type': 'Lattice Tower',
            'name': 'T33',
            'latitude': 0.0,
            'longitude': 149.0,
            #'comment': 'Test',
            'function': 'Suspension',
            'devangle': 0,
            'axisaz': 134,
            #'constcost': 0.0,
            'height': 17.0,
            #'yrbuilt': 1980,
            #'locsource': 'Fake',
            'lineroute': 'LineB',
            #'shapes': <shapefile.Shape object at 0x7ff06908ec50>,
            'coord': np.array([149.065,   0.   ]),
            'coord_lat_lon': np.array([  0.   , 149.065]),
            #'point': <shapely.geometry.point.Point object at 0x7ff06908e320>,
            'design_span': 400.0,
            'design_level': 'low',
            'design_speed': 75.0,
            'terrain_cat': 2,
            'file_wind_base_name': 'ts.T33.csv',
            'height_z': 15.4,
            'ratio_z_to_10': 1.0524,
            'actual_span': 556.5974539658616,
            'u_factor': 1.0,
            'collapse_capacity': 75.0,
            'cond_pc': cond_pc,
            'max_no_adj_towers': 2,
            'id_adj': [8, -1, 10, 11, 12],
            'idl': 10,
            'cond_pc_adj': cond_pc_adj,
            'cond_pc_adj_sim_idx': cond_pc_adj_sim_idx,
            'cond_pc_adj_sim_prob': cond_pc_adj_sim_prob,
            'frag_dic': frag_dic,
            'shape': None,
            'point': None,
            }

        cls.tower = Tower(**cls.tower_dic)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'))

    def test_collapse_adj_sim(self):

        # tower14 (idl: 13,
        rnd_state = np.random.RandomState(1)
        tower_dic = self.tower_dic.copy()
        tower_dic.update({'axisaz': 90,
                          #'no_sims': 5000
                          })

        self.cfg.no_sims = 10000
        tower = Tower(**tower_dic)
        wind = create_wind_given_bearing([130, 130, 130, 130],[0.0712, 1.0712, 1.0712, 0.0712])  # 1.05*np.exp(0.02)
        dmg = set_dmg(tower, wind, self.cfg)
        collapse_adj = set_collapse_adj(tower, dmg)
        dmg_state_sim = set_dmg_state_sim(dmg, self.cfg, rnd_state)
        collapse_adj_sim = set_collapse_adj_sim(tower, dmg_state_sim, collapse_adj, rnd_state, self.cfg)
        df= collapse_adj_sim.groupby(['id_time','id_adj']).apply(len).reset_index()
        for idl in tower.cond_pc_adj.keys():
            x = df.loc[df['id_adj'].apply(lambda x: idl in x)].groupby('id_time').sum()/self.cfg.no_sims
            np.testing.assert_allclose(x[0].values, collapse_adj[idl], atol=self.cfg.atol, rtol=self.cfg.rtol)


if __name__ == '__main__':
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestTower3)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main(verbosity=2)
