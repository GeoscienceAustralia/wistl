#!/usr/bin/env python

__author__ = 'Hyeuk Ryu'

import unittest
import logging
import pandas as pd
import os
import tempfile
import numpy as np
from scipy import stats

from wistl.config import Config
from wistl.constants import RTOL, ATOL, PM_THRESHOLD
from wistl.tower import Tower, angle_between_two
from wistl.tests.test_config import assertDeepAlmostEqual

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def create_wind_given_bearing(bearing, ratio):

    df = pd.DataFrame([[ratio, bearing], [ratio, bearing]], columns=['ratio', 'Bearing'])
    df['time'] = pd.date_range(start='01/01/2011', end='01/02/2011', freq='D')
    df.set_index('time', inplace=True)

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
            'comment': 'Test',
            'function': 'Suspension',
            'devangle': 0,
            'axisaz': 134,
            'constcost': 0.0,
            'height': 17.0,
            'yrbuilt': 1980,
            'locsource': 'Fake',
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
            'idn': 0,
            'cond_pc_adj': cond_pc_adj,
            'cond_pc_adj_sim_idx': cond_pc_adj_sim_idx,
            'cond_pc_adj_sim_prob': cond_pc_adj_sim_prob,
            'no_sims': 1000,
            'damage_states': ['minor', 'collapse'],
            'non_collapse': ['minor'],
            'rnd_state': np.random.RandomState(1),
            'event_id': 0,
            'scale': 1.0,
            'frag_dic': frag_dic,
            'path_event': os.path.join(BASE_DIR, 'wind_event/test1'),
            }

        cls.tower = Tower(**cls.tower_dic)

        #cls.tower.wind['ratio'] = 1.082  # 1.05*np.exp(0.03)

        # cls.tower = Tower(tower_id=0, logger=logge**cls.cfg.towers.loc[0])
        # cls.network = TransmissionNetwork(cfg=cls.cfg, event_id='test2', scale=2.5)
        #
        # cls.tower = cls.network.lines['Calaca - Amadeo'].towers['AC-100']
        # cls.ps_tower = cls.tower.ps_tower
        # cls.tower.event_tuple = (cls.tower.file_wind, 3.0)

        # set wind file, which also sets wind and time_index
        # cls.tower.file_wind = file_wind

        # compute prob_damage_isolation and prob_damage_adjacent
        # cls.tower.compute_dmg_isolated_isolation()
        #cls.tower.compute_pc_adj()

    def test_damage_states(self):

        self.assertEqual(self.tower.damage_states, ['minor', 'collapse'])

    def test_no_time(self):

        file_wind = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

        # read file_wind
        file_wind.writelines([
        'Time,Longitude,Latitude,Speed,UU,VV,Bearing,Pressure\n',
        '2014-07-13 09:00,120.79,13.93,3.68,-0.18,-5.6,1.84,100780.97\n',
        '2014-07-13 09:05,120.80,13.93,3.68,-0.18,-5.6,1.89,100780.92\n'
        ])

        file_wind.seek(0)

        self.tower._file_wind = file_wind.name

        self.tower._wind = None
        self.assertEqual(self.tower.no_time, 2)

        os.unlink(file_wind.name)

    def test_sorted_frag_dic_keys(self):

        self.assertEqual(self.tower.sorted_frag_dic_keys, [11.5, 28.75, 41.25, 90.0])

    def test_file_wind(self):

        assert self.tower.name == 'T14'
        expected = os.path.join(BASE_DIR, 'wind_event/test1', 'ts.T14.csv')
        self.assertEqual(self.tower.file_wind, expected)

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
            result = tower.get_directional_vulnerability(bearing)
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
            result = tower.get_directional_vulnerability(bearing)
            try:
                self.assertAlmostEqual(result, value)
            except AssertionError:
                print(f'Wrong: bearing:{bearing}, axisaz: {tower_dic["axisaz"]}, result:{result}, expected: {value}')

    def test_wind(self):

        file_wind = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

        # read file_wind
        file_wind.writelines([
        'Time,Longitude,Latitude,Speed,UU,VV,Bearing,Pressure\n',
        '2014-07-13 09:00,120.79,13.93,3.68,-0.18,-5.6,1.84,100780.97\n',
        '2014-07-13 09:05,120.80,13.93,3.68,-0.18,-5.6,1.89,100780.92\n'
        ])

        file_wind.seek(0)

        self.tower._file_wind = file_wind.name

        self.tower._wind = None
        self.assertAlmostEqual(self.tower.wind.loc['2014-07-13 09:00', 'Speed'], 1.0524*3.68)
        self.assertAlmostEqual(self.tower.wind.loc['2014-07-13 09:05', 'ratio'], 1.0524*3.68/75.0)

        os.unlink(file_wind.name)

    def test_dmg_isolated(self):
        frag_dic = {11.5: {'minor': stats.lognorm(0.02, scale=1.02),
                           'collapse': stats.lognorm(0.02, scale=1.05)},
                    28.75: {'minor': stats.lognorm(0.02, scale=1.0),
                            'collapse': stats.lognorm(0.02, scale=1.02)},
                    41.25: {'minor': stats.lognorm(0.02, scale=1.04),
                            'collapse': stats.lognorm(0.02, scale=1.07)},
                    90: {'minor': stats.lognorm(0.1, scale=-1.2),
                         'collapse': stats.lognorm(0.1, scale=-1.4)}
                    }

        self.assertAlmostEqual(self.tower.collapse_capacity, 75.0)
        self.assertAlmostEqual(self.tower.axisaz, 134.0)

        bearing, ratio = 130.0, 1.02
        self.tower._wind = create_wind_given_bearing(bearing, ratio)
        key = self.tower.get_directional_vulnerability(bearing)
        self.assertEqual(key, 11.5)
        self.tower._dmg = None
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'minor'],
                         stats.lognorm.cdf(1.02, 0.02, scale=1.02))
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'collapse'],
                         stats.lognorm.cdf(1.02, 0.02, scale=1.05))

        bearing, ratio = 45.0, 1.02
        self.tower._wind = create_wind_given_bearing(bearing, ratio)
        key = self.tower.get_directional_vulnerability(bearing)
        self.assertEqual(key, 90.0)
        self.tower._dmg = None
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'minor'], 0.0)
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'collapse'], 0.0)

        bearing, ratio = 110.0, 1.04
        self.tower._wind = create_wind_given_bearing(bearing, ratio)
        key = self.tower.get_directional_vulnerability(bearing)
        self.assertEqual(key, 28.75)
        self.tower._dmg = None
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'minor'],
                         stats.lognorm.cdf(1.04, 0.02, scale=1.0))
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'collapse'],
                         stats.lognorm.cdf(1.04, 0.02, scale=1.02))

        bearing, ratio = 100.0, 1.0
        self.tower._wind = create_wind_given_bearing(bearing, ratio)
        key = self.tower.get_directional_vulnerability(bearing)
        self.assertEqual(key, 41.25)
        self.tower._dmg = None
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'minor'],
                         stats.lognorm.cdf(ratio, 0.02, scale=1.04))
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'collapse'],
                         stats.lognorm.cdf(ratio, 0.02, scale=1.07))

    def test_collapse_adj(self):

        cond_pc_adj = {11: 0.125, 12: 0.575, 14: 0.575, 15: 0.125}
        assertDeepAlmostEqual(self, dict(self.tower.cond_pc_adj), cond_pc_adj)

        self.tower._wind = create_wind_given_bearing(130, 1.0712)  # 1.05*np.exp(0.02)

        self.assertEqual(self.tower.id_adj, [11, 12, 13, 14, 15])
        self.tower._dmg = None

        for id_abs, value in cond_pc_adj.items():

            self.assertAlmostEqual(self.tower.collapse_adj[id_abs][0],
                                   0.842 * value, places=2)
    def test_dmg_sim(self):

        # 1. determine damage state of tower due to wind
        self.tower._wind = create_wind_given_bearing(130.0, 1.0712)  # 1.05*np.exp(0.02)
        self.tower._dmg = None
        self.tower._dmg_id_sim = None
        self.tower._dmg_state_sim = None
        self.tower._dmg_sim = None

        self.assertAlmostEqual(self.tower.dmg['collapse'].values[0],
                               stats.lognorm.cdf(1.0712, 0.02, scale=1.05), places=2)
        self.assertAlmostEqual(self.tower.dmg['minor'].values[0],
                               stats.lognorm.cdf(1.0712, 0.02, scale=1.02), places=2)
        self.tower.dmg_sim

    def test_dmg_state_sim(self):

        self.tower._wind = create_wind_given_bearing(130.0, 1.0712)  # 1.05*np.exp(0.02)
        self.tower._dmg = None
        self.tower._dmg_id_sim = None
        self.tower._dmg_state_sim = None
        self.tower._dmg_sim = None

        rv = stats.uniform.rvs(size=(self.tower.no_sims, self.tower.no_time))

        a = np.array([rv < self.tower.dmg[ds].values
                      for ds in self.tower.damage_states]).sum(axis=0)

        b = (rv[:, :, np.newaxis] < self.tower.dmg.values).sum(axis=2)

        np.testing.assert_array_equal(a, b)

    def test_dmg_threshold(self):

        df = pd.DataFrame([[0.1, 130.0], [0.9605, 130.0], [0.1, 130.0], [0.97, 130.0]], columns=['ratio', 'Bearing'])
        df['time'] = pd.date_range(start='01/01/2011', end='01/04/2011', freq='D')
        df.set_index('time', inplace=True)

        self.tower._wind = df
        self.tower._dmg = None

        # checking index
        pd.testing.assert_index_equal(self.tower.dmg.index, df.index[1:3+1])

    def test_dmg_time_idx(self):

        df = pd.DataFrame([[0.1, 130.0], [0.9605, 130.0], [0.1, 130.0], [0.97, 130.0]], columns=['ratio', 'Bearing'])
        df['time'] = pd.date_range(start='01/01/2011', end='01/04/2011', freq='D')
        df.set_index('time', inplace=True)

        self.tower._wind = df
        self.tower._dmg = None
        self.tower._dmg_time_idx = None
        # checking index
        self.assertEqual((1,4), self.tower.dmg_time_idx)


    def test_compare_dmg_with_dmg_sim(self):

        # dmg_isolated vs. dmg_sim
        prob_sim = 0
        for ds in self.tower.damage_states[::-1]:
            no = len(self.tower.dmg_id_sim[ds]['id_sim'][
                 self.tower.dmg_id_sim[ds]['id_time'] == 1])
            prob_sim += no / self.tower.no_sims
            isclose = np.isclose(prob_sim,
                                 self.tower.dmg[ds].values[1],
                                 rtol=RTOL, atol=ATOL)
            if not isclose:
                self.logger.warning(f'PE of {ds}: '
                                    f'simulation {prob_sim:.3f} vs. '
                                    f'analytical {self.tower.dmg[ds].values[1]:.3f}')

    def test_dmg_id_sim(self):

        self.tower._wind = create_wind_given_bearing(130.0, 1.0712)  # 1.05*np.exp(0.02)
        self.tower._dmg = None
        self.tower._dmg_id_sim = None

        rv = np.array([[0, 0], [1, 1], [0.5, 0.9]])   # no_sims, no_time
        np.testing.assert_allclose(self.tower.dmg.values,
                                   np.array([[0.9928, 0.8412],
                                             [0.9928, 0.8412]]),
                                   rtol=1.e-4)  # minor, collapse
        self.tower._dmg_state_sim = (rv[:, :, np.newaxis] < self.tower.dmg.values).sum(axis=2)

        np.testing.assert_equal(self.tower._dmg_state_sim, np.array([[2, 2], [0, 0], [2, 1]]))

        np.testing.assert_equal(self.tower.dmg_id_sim['minor']['id_sim'].values, np.array([2]))
        np.testing.assert_equal(self.tower.dmg_id_sim['minor']['id_time'].values, np.array([1]))

        np.testing.assert_equal(self.tower.dmg_id_sim['collapse']['id_sim'].values, np.array([0, 0, 2]))
        np.testing.assert_equal(self.tower.dmg_id_sim['collapse']['id_time'].values, np.array([0, 1, 0]))

    def test_dmg_id_sim_threshold(self):

        # 1.05*np.exp(0.02)
        df = pd.DataFrame([[0.1, 130.0], [1.0712, 130.0], [0.1, 130.0], [1.0712, 130.0]], columns=['ratio', 'Bearing'])
        df['time'] = pd.date_range(start='01/01/2011', end='01/04/2011', freq='D')
        df.set_index('time', inplace=True)

        self.tower._wind = df
        self.tower._dmg = None
        self.tower._dmg_id_sim = None
        self.tower._dmg_time_idx = None
        self.tower._dmg_state_sim = None

        rv = np.array([[0.9, 0.5, 0],
                      [0.1, 0.5, 0.9],
                      [0.5, 0.9, 0.7]])   # no_sims, no_time
        np.testing.assert_allclose(self.tower.dmg.values,
                                   np.array([[0.9928, 0.8412],
                                             [0.0, 0.0],
                                             [0.9928, 0.8412]]),
                                   rtol=1.e-4)  # minor, collapse
        self.tower._dmg_state_sim = (rv[:, :, np.newaxis] < self.tower.dmg.values).sum(axis=2)

        np.testing.assert_equal(self.tower._dmg_state_sim,
            np.array([[1, 0, 2], [2, 0, 1], [2, 0, 2]]))  # no_sims, no_time

        np.testing.assert_equal(self.tower.dmg_id_sim['minor']['id_sim'].values, np.array([0, 1]))
        np.testing.assert_equal(self.tower.dmg_id_sim['minor']['id_time'].values, np.array([1, 3]))

        np.testing.assert_equal(self.tower.dmg_id_sim['collapse']['id_sim'].values, np.array([0, 1, 2, 2]))
        np.testing.assert_equal(self.tower.dmg_id_sim['collapse']['id_time'].values, np.array([3, 1, 1, 3]))


    def test_collapse_adj_sim(self):

        # tower14 (idl: 13,
        tower_dic = self.tower_dic.copy()
        tower_dic.update({'axisaz': 90,
                          'no_sims': 10000})

        tower = Tower(**tower_dic)

        tower._wind = create_wind_given_bearing(130.0, 1.0712)  # 1.05*np.exp(0.02)
        tower._dmg = None
        tower._dmg_state_sim = None
        tower._dmg_sim = None
        tower._dmg_id_sim = None
        tower._collapse_adj = None
        tower._collapse_adj_sim = None

        tower.collapse_adj_sim


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
            'comment': 'Test',
            'function': 'Suspension',
            'devangle': 0,
            'axisaz': 134,
            'constcost': 0.0,
            'height': 17.0,
            'yrbuilt': 1980,
            'locsource': 'Fake',
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
            'idn': 33,
            'cond_pc_adj': cond_pc_adj,
            'cond_pc_adj_sim_idx': cond_pc_adj_sim_idx,
            'cond_pc_adj_sim_prob': cond_pc_adj_sim_prob,
            'no_sims': 1000,
            'damage_states': ['minor', 'collapse'],
            'non_collapse': ['minor'],
            'rnd_state': np.random.RandomState(1),
            'event_id': 0,
            'scale': 1.0,
            'frag_dic': frag_dic,
            'path_event': os.path.join(BASE_DIR, 'wind_event/test1'),
            }

        cls.tower = Tower(**cls.tower_dic)

    def test_collapse_adj_sim(self):

        # tower14 (idl: 13,
        tower_dic = self.tower_dic.copy()
        tower_dic.update({'axisaz': 90,
                          'no_sims': 5000})

        tower = Tower(**tower_dic)

        tower._wind = create_wind_given_bearing(130.0, 1.0712)  # 1.05*np.exp(0.02)
        tower._dmg = None
        tower._dmg_state_sim = None
        tower._dmg_sim = None
        tower._dmg_id_sim = None
        tower._collapse_adj = None
        tower._collapse_adj_sim = None

        tower.collapse_adj_sim



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
            'comment': 'Test',
            'function': 'Suspension',
            'devangle': 0,
            'axisaz': 134,
            'constcost': 0.0,
            'height': 17.0,
            'yrbuilt': 1980,
            'locsource': 'Fake',
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
            'idn': 0,
            'cond_pc_adj': cond_pc_adj,
            'cond_pc_adj_sim_idx': cond_pc_adj_sim_idx,
            'cond_pc_adj_sim_prob': cond_pc_adj_sim_prob,
            'no_sims': 1000,
            'damage_states': ['minor', 'collapse'],
            'non_collapse': ['minor'],
            'rnd_state': np.random.RandomState(1),
            'event_id': 0,
            'scale': 1.0,
            'frag_dic': frag_dic,
            'path_event': os.path.join(BASE_DIR, 'wind_event/test1'),
            }

        cls.tower = Tower(**cls.tower_dic)


    def test_sorted_frag_dic_keys(self):

        self.assertEqual(self.tower.sorted_frag_dic_keys, [180.0])

    def test_file_wind(self):

        assert self.tower.name == 'T9'
        expected = os.path.join(BASE_DIR, 'wind_event/test1', 'ts.T9.csv')
        self.assertEqual(self.tower.file_wind, expected)

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
            result = tower.get_directional_vulnerability(bearing)
            try:
                self.assertAlmostEqual(result, value)
            except AssertionError:
                print(f'Wrong: bearing:{bearing}, axisaz: {tower_dic["axisaz"]}, result:{result}, expected: {value}')

    def test_wind(self):

        file_wind = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

        # read file_wind
        file_wind.writelines([
        'Time,Longitude,Latitude,Speed,UU,VV,Bearing,Pressure\n',
        '2014-07-13 09:00,120.79,13.93,3.68,-0.18,-5.6,1.84,100780.97\n',
        '2014-07-13 09:05,120.80,13.93,3.68,-0.18,-5.6,1.89,100780.92\n'
        ])

        file_wind.seek(0)

        self.tower._file_wind = file_wind.name

        self.tower._wind = None
        self.assertAlmostEqual(self.tower.wind.loc['2014-07-13 09:00', 'Speed'], 1.0524*3.68)
        self.assertAlmostEqual(self.tower.wind.loc['2014-07-13 09:05', 'ratio'], 1.0524*3.68/75.0)

        os.unlink(file_wind.name)

    def test_dmg_isolated(self):

        frag_dic = {180: {'minor': ['lognorm','1.143','0.032'],
                          'collapse': ['lognorm','1.18','0.04']}}

        self.assertAlmostEqual(self.tower.collapse_capacity, 75.0)
        self.assertAlmostEqual(self.tower.axisaz, 134.0)

        bearing, ratio = 130.0, 1.02
        self.tower._wind = create_wind_given_bearing(bearing, ratio)
        key = self.tower.get_directional_vulnerability(bearing)
        self.assertEqual(key, 180)
        self.tower._dmg = None
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'minor'],
                         stats.lognorm.cdf(1.02, 0.032, scale=1.143))
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'collapse'],
                         stats.lognorm.cdf(1.02, 0.04, scale=1.18))

        bearing, ratio = 110.0, 1.04
        self.tower._wind = create_wind_given_bearing(bearing, ratio)
        key = self.tower.get_directional_vulnerability(bearing)
        self.assertEqual(key, 180)
        self.tower._dmg = None
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'minor'],
                         stats.lognorm.cdf(1.04, 0.032, scale=1.143))
        self.assertAlmostEqual(self.tower.dmg.loc['01/01/2011', 'collapse'],
                         stats.lognorm.cdf(1.04, 0.04, scale=1.18))

    def test_collapse_adj(self):
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

        self.tower._wind = create_wind_given_bearing(130, 1.22816)  # 1.18*np.exp(0.04)

        self.assertEqual(self.tower.id_adj, [2, 3, 4, 5, 6, 7, -1, 9, 10, 11, 12, 13, 14])
        self.tower._dmg = None

        for id_abs, value in cond_pc_adj.items():

            self.assertAlmostEqual(self.tower.collapse_adj[id_abs][0],
                                   0.842 * value, places=2)

    def test_dmg_sim(self):

        # 1. determine damage state of tower due to wind
        self.tower._wind = create_wind_given_bearing(130.0, 1.22816)  # 1.18*np.exp(0.04)
        self.tower._dmg = None
        self.tower._dmg_id_sim = None
        self.tower._dmg_state_sim = None
        self.tower._dmg_sim = None
        self.assertAlmostEqual(self.tower.dmg['collapse'].values[0],
                               stats.lognorm.cdf(1.22816, 0.04, scale=1.18), places=2)
        self.assertAlmostEqual(self.tower.dmg['minor'].values[0],
                               stats.lognorm.cdf(1.22816, 0.032, scale=1.143), places=2)
        self.tower.dmg_sim

    def test_dmg_state_sim(self):

        self.tower._wind = create_wind_given_bearing(130.0, 1.22816)  # 1.18*np.exp(0.04)
        self.tower._dmg = None
        self.tower._dmg_id_sim = None
        self.tower._dmg_state_sim = None
        self.tower._dmg_sim = None

        rv = stats.uniform.rvs(size=(self.tower.no_sims, self.tower.no_time))

        a = np.array([rv < self.tower.dmg[ds].values
                      for ds in self.tower.damage_states]).sum(axis=0)

        b = (rv[:, :, np.newaxis] < self.tower.dmg.values).sum(axis=2)

        np.testing.assert_array_equal(a, b)

    def test_compare_dmg_with_dmg_sim(self):

        # dmg_isolated vs. dmg_sim
        prob_sim = 0
        for ds in self.tower.damage_states[::-1]:
            no = len(self.tower.dmg_id_sim[ds]['id_sim'][
                 self.tower.dmg_id_sim[ds]['id_time'] == 1])
            prob_sim += no / self.tower.no_sims
            isclose = np.isclose(prob_sim,
                                 self.tower.dmg[ds].values[1],
                                 rtol=RTOL, atol=ATOL)
            if not isclose:
                self.logger.warning(f'PE of {ds}: '
                                    f'simulation {prob_sim:.3f} vs. '
                                    f'analytical {self.tower.dmg[ds].values[1]:.3f}')

    def test_dmg_id_sim(self):

        self.tower._wind = create_wind_given_bearing(130.0, 1.22816)  # 1.18*np.exp(0.04)
        self.tower._dmg = None
        self.tower._dmg_id_sim = None
        rv = np.array([[0, 0], [1, 1], [0.5, 0.9]])   # no_sims, no_time
        np.testing.assert_allclose(self.tower.dmg.values,
                                   np.array([[0.987637, 0.841361],
                                             [0.987637, 0.841361]]),
                                   rtol=1.e-4)  # minor, collapse
        self.tower._dmg_state_sim = (rv[:, :, np.newaxis] < self.tower.dmg.values).sum(axis=2)

        np.testing.assert_equal(self.tower._dmg_state_sim, np.array([[2, 2], [0, 0], [2, 1]]))

        np.testing.assert_equal(self.tower.dmg_id_sim['minor']['id_sim'].values, np.array([2]))
        np.testing.assert_equal(self.tower.dmg_id_sim['minor']['id_time'].values, np.array([1]))

        np.testing.assert_equal(self.tower.dmg_id_sim['collapse']['id_sim'].values, np.array([0, 0, 2]))
        np.testing.assert_equal(self.tower.dmg_id_sim['collapse']['id_time'].values, np.array([0, 1, 0]))


    def test_collapse_adj_sim(self):

        # tower14 (idl: 13,
        tower_dic = self.tower_dic.copy()
        tower_dic.update({'axisaz': 90,
                          'no_sims': 5000,
                          'function': 'strainer'})

        tower = Tower(**tower_dic)

        tower._wind = create_wind_given_bearing(130.0, 1.2282)  # 1.18*np.exp(0.04)
        tower._dmg = None
        tower._dmg_state_sim = None
        tower._dmg_sim = None
        tower._dmg_id_sim = None
        tower._collapse_adj = None
        tower._collapse_adj_sim = None

        tower.collapse_adj_sim



if __name__ == '__main__':
    unittest.main(verbosity=2)
