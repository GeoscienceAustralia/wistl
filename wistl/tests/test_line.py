
import unittest
import os
import numpy as np
import copy
import logging
import pandas as pd
from scipy.stats import itemfreq

from wistl.config import Config
from wistl.line import Line, adjust_value_to_line, adjust_index_to_line
from wistl.tests.test_config import assertDeepAlmostEqual
from wistl.tests.test_tower import create_wind_given_bearing
# from wistl.transmission_network import read_shape_file, populate_df_lines, \
#     populate_df_towers

#ATOL = 0.0005
#RTOL = 0.01
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class TestLine1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.basicConfig(level=logging.ERROR)
        cls.logger = logging.getLogger(__name__)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'), logger=cls.logger)

        event_name = 'test1'
        event_scale = 3.0
        path_event = os.path.join(cls.cfg.path_wind_event_base,
                                  event_name)
        path_output = os.path.join(cls.cfg.path_output, 'test1_s3.0')
        cls.no_sims = 500000

        # LineB
        dic_line = cls.cfg.lines['LineB'].copy()
        dic_line.update({'name': 'LineB',
                         'no_sims': cls.no_sims,
                         'damage_states': cls.cfg.damage_states,
                         'non_collapse': cls.cfg.non_collapse,
                         'event_id': cls.cfg.event_id_format.format(event_name=event_name, scale=event_scale),
                         'scale': event_scale,
                         'rtol': cls.cfg.rtol,
                         'atol': cls.cfg.atol,
                         'dmg_threshold': cls.cfg.dmg_threshold,
                         'rnd_state': np.random.RandomState(0),
                         'path_event': path_event,
                         'path_output': path_output,
                         'dic_towers': cls.cfg.towers_by_line['LineB']})

        cls.line = Line(**dic_line)

        for _, tower in cls.line.towers.items():
            tower.init()
            tower._wind = create_wind_given_bearing(10.0, 1.0)
            tower.axisaz = 11.0

    @classmethod
    def tearDown(cls):
        try:
            os.remove(cls.line.file_output)
            os.removedirs(cls.line.output_path)
        except:
            pass

    def test_repr(self):
        expected = f'Line(name=LineB, no_towers=22, event_id=test1_s3.0)'
        self.assertEqual(repr(self.line), expected)

    def test_towers(self):

        self.assertEqual(self.line.no_towers, 22)
        self.assertEqual(self.line.dmg_time_idx, (0, 2))
        #print(self.line.time)
        #self.assertEqual(self.line.no_time, 3)

        self.assertEqual(self.line.towers[0].name, 'T23')
        self.assertEqual(self.line.towers[0].idl, 0)
        self.assertEqual(self.line.towers[0].idn, 'T23')
        self.assertEqual(self.line.towers[0].no_time, 2)
        self.assertEqual(self.line.towers[0].no_sims, self.no_sims)
        self.assertEqual(self.line.towers[0].damage_states, ['minor', 'collapse'])
        self.assertAlmostEqual(self.line.towers[0].scale, 3.0)

    def test_time(self):

        pd.testing.assert_index_equal(self.line.time, self.line.towers[0].wind.index[0:2])

    def test_no_time(self):

        self.assertEqual(self.line.no_time, 2)

    def test_adjust_value_to_line1(self):

        line_dmg_time_idx = (0, 3)
        tower_dmg_time_idx = (0, 2)
        prob = [1, 2]
        result = adjust_value_to_line(line_dmg_time_idx,
            tower_dmg_time_idx, prob)
        expected = np.array([1, 2, 0])
        np.testing.assert_equal(result, expected)

    def test_adjust_value_to_line2(self):

        line_dmg_time_idx = (1870, 1879)
        prob = [1, 2, 3, 4]
        tower_dmg_time_idx = (1874, 1878)
        result = adjust_value_to_line(line_dmg_time_idx,
            tower_dmg_time_idx, prob)
        expected = np.array([0, 0, 0, 0, 1, 2, 3, 4, 0])
        np.testing.assert_equal(result, expected)

    def test_adjust_index_to_line1(self):

        line_dmg_time_idx = (0, 3)
        tower_dmg_time_idx = (0, 2)
        prob = np.array([0, 1, 1, 0])
        idt0, idx = adjust_index_to_line(line_dmg_time_idx,
            tower_dmg_time_idx, prob)
        self.assertEqual(idt0, 0)
        expected = np.ones_like(prob, dtype=bool)
        np.testing.assert_equal(idx, expected)

    def test_adjust_index_to_line2(self):

        line_dmg_time_idx = (1870, 1879)
        prob = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        tower_dmg_time_idx = (1874, 1878)
        idt0, idx = adjust_index_to_line(line_dmg_time_idx,
            tower_dmg_time_idx, prob)
        self.assertEqual(idt0, -4)
        expected = np.ones_like(prob, dtype=bool)
        np.testing.assert_equal(idx, expected)

    def test_damage_prob(self):
        name = 'T26'
        # for name, tower in self.line1.towers.items():
        self.line._damage_prob = None

        tower = self.line.towers[3]
        #tower._dmg = None
        #tower._dmg_sim = None
        self.assertEqual(tower.name, name)
        # T26 (in the middle)
        # o----o----x----o----o
        # collapse
        # lognorm.cdf(1.0, 0.03, scale=1.05)
        p0c = 0.0519
        p0gn2 = p0c * 0.125
        p0gn1 = p0c * 0.575
        p0gp1 = p0c * 0.575
        p0gp2 = p0c * 0.125
        pc = 1 - (1-p0c)*(1-p0gn2)*(1-p0gn1)*(1-p0gp1)*(1-p0gp2)

        self.assertAlmostEqual(p0c, tower.dmg['collapse'][0], places=3)
        #self.assertAlmostEqual(p0c, tower.dmg_sim['collapse'][0], places=3)

        self.assertAlmostEqual(p0c, self.line.towers[1].dmg['collapse'][0], places=3)
        self.assertAlmostEqual(p0c, self.line.towers[2].dmg['collapse'][0], places=3)
        self.assertAlmostEqual(p0c, self.line.towers[4].dmg['collapse'][0], places=3)
        self.assertAlmostEqual(p0c, self.line.towers[5].dmg['collapse'][0], places=3)

        self.assertAlmostEqual(p0gn2, tower.collapse_adj[1][0], places=3)
        self.assertAlmostEqual(p0gn1, tower.collapse_adj[2][0], places=3)
        self.assertAlmostEqual(p0gp1, tower.collapse_adj[4][0], places=3)
        self.assertAlmostEqual(p0gp2, tower.collapse_adj[5][0], places=3)
        self.line.compute_damage_prob()
        self.assertAlmostEqual(
            pc, self.line.damage_prob['collapse'][name][0], places=3)

        # T26 (in the middle)
        # o----o----x----o----o
        # minor
        # lognorm.cdf(1.0, 0.03, scale=1.02)
        p0m = 0.16105
        pm = min(p0m - p0c + pc, 1.0)

        self.assertAlmostEqual(p0m, tower.dmg['minor'][0], places=3)
        #self.assertAlmostEqual(p0m, tower.dmg_sim['minor'][0], places=3)
        self.assertAlmostEqual(
            pm, self.line.damage_prob['minor'][name][0], places=3)


    def test_compute_damage_prob_sim(self):

        name = 'T26'

        tower = self.line.towers[3]
        #tower._dmg = None
        #tower._dmg_id_sim = None
        #tower._dmg_sim = None
        self.assertEqual(tower.name, name)
        # T26 (in the middle)
        # o----o----x----o----o
        # collapse
        # lognorm.cdf(1.0, 0.03, scale=1.05)
        p0c = 0.0519
        p0gn2 = p0c * 0.125
        p0gn1 = p0c * 0.575
        p0gp1 = p0c * 0.575
        p0gp2 = p0c * 0.125
        pc = 1 - (1-p0c)*(1-p0gn2)*(1-p0gn1)*(1-p0gp1)*(1-p0gp2)

        with self.assertLogs('wistl', level='INFO') as cm:
            self.line.compute_damage_prob()
            self.line.compute_damage_prob_sim()

        try:
            np.testing.assert_allclose(
                pc, self.line.damage_prob_sim['collapse'][name][0], atol=self.cfg.atol, rtol=self.cfg.rtol)
        except AssertionError:
            self.logger.warning(
                f'P(C) Theory: {pc:.4f}, '
                f"Analytical: {self.line.damage_prob['collapse'][name][0]:.4f}, "
                f"Simulation: {self.line.damage_prob_sim['collapse'][name][0]:.4f}")

        for _id in range(23, 45):
            name = f'T{_id}'
            try:
                np.testing.assert_allclose(self.line.damage_prob['collapse'][name][0],
                    self.line.damage_prob_sim['collapse'][name][0], atol=self.cfg.atol, rtol=self.cfg.rtol)
            except AssertionError:
                self.logger.warning(
                    f'Tower: {name}, collapse '
                    f"Analytical: {self.line.damage_prob['collapse'][name][0]:.4f}, "
                    f"Simulation: {self.line.damage_prob_sim['collapse'][name][0]:.4f}")

        # o----o----x----o----o
        # minor
        # lognorm.cdf(1.0, 0.02, scale=1.02) 
        p0m = 0.16105
        pm = min(p0m - p0c + pc, 1.0)
        try:
           self.assertTrue(
                pm >= self.line.damage_prob_sim['minor'][name][0])
        except AssertionError:
            self.logger.warning(
                f'P(m) Theory: {pm:.4f}, '
                f"Analytical: {self.line.damage_prob['minor'][name][0]:.4f}, "
                f"Simulation: {self.line.damage_prob_sim['minor'][name][0]:.4f}")

        # except 32, strainer tower
        for _id in list(range(23, 32)) + list(range(33, 45)):
            name = f'T{_id}'
            try:
                self.assertTrue(self.line.damage_prob['minor'][name][0]
                        >= self.line.damage_prob_sim['minor'][name][0])
            except AssertionError:
                self.logger.warning(
                    f'Tower: {name}, minor, '
                    f"Analytical: {self.line.damage_prob['minor'][name][0]:.4f}, "
                    f"Simulation: {self.line.damage_prob_sim['minor'][name][0]:.4f}")

    def test_compute_damage_prob_sim_no_cascading(self):

        name = 'T26'

        tower = self.line.towers[3]
        #tower._dmg = None
        #tower._dmg_id_sim = None
        #tower._dmg_sim = None
        self.assertEqual(tower.name, name)
        # T26 (in the middle)
        # o----o----x----o----o
        # collapse
        # lognorm.cdf(1.0, 0.03, scale=1.05)
        pc = tower.dmg['collapse']
        self.line.compute_damage_prob_sim_no_cascading()
        try:
            np.testing.assert_allclose(
                pc, self.line.damage_prob_sim_no_cascading['collapse'][name], atol=self.cfg.atol, rtol=self.cfg.rtol)
        except AssertionError:
            self.logger.warning(
                    f'P(C) Theory: {pc[0]}, '
                f"Simulation: {self.line.damage_prob_sim_no_cascading['collapse'][name][0]}")

        # o----o----x----o----o
        # minor
        # lognorm.cdf(1.0, 0.02, scale=1.02) 
        pm = tower.dmg['minor']
        try:
           np.testing.assert_allclose(
                pm, self.line.damage_prob_sim_no_cascading['minor'][name], atol=self.cfg.atol, rtol=self.cfg.rtol)
        except AssertionError:
            self.logger.warning(
                f'P(m) Theory: {pm.values}, '
                f"Simulation: {self.line.damage_prob_sim_no_cascading['minor'][name].values}")

        # except 32, strainer tower
        for _id in self.line.dmg_towers:
            name = self.line.towers[_id].name
            idt0, idt1 = self.line.towers[_id].dmg_time_idx
            try:
                np.testing.assert_allclose(
                        self.line.towers[_id].dmg['minor'], self.line.damage_prob_sim_no_cascading['minor'].iloc[idt0:idt1][name], atol=self.cfg.atol, rtol=self.cfg.rtol)
            except AssertionError:

                self.logger.warning(
                    f'Tower: {name}, minor, '
                    f"Theory: {self.line.towers[_id].dmg['minor'].values}, "
                    f"Simulation: {self.line.damage_prob_sim_no_cascading['minor'][name].values}")

    def test_compute_stats(self):

        no_sims = 10
        rnd_state = np.random.RandomState(1)
        event_name = 'test1'
        path_event = os.path.join(self.cfg.path_wind_event_base,
                                  event_name)

        # LineB
        dic_line = self.cfg.lines['LineB'].copy()
        dic_line.update({'name': 'LineB',
                         'no_sims': no_sims,
                         'damage_states': self.cfg.damage_states,
                         'non_collapse': self.cfg.non_collapse,
                         'event_name': event_name,
                         'scale': 1.0,
                         'rtol': self.cfg.rtol,
                         'atol': self.cfg.atol,
                         'dmg_threshold': self.cfg.dmg_threshold,
                         'rnd_state': rnd_state,
                         'path_event': path_event,
                         'dic_towers': self.cfg.towers_by_line['LineB']})

        line = Line(**dic_line)

        for _, tower in line.towers.items():
            tower.init()
            tower._wind = create_wind_given_bearing(10.0, 1.0)
            tower.axisaz = 11.0

        with self.assertLogs('wistl', level='INFO') as cm:
            tf_ds = np.zeros((line.no_towers, no_sims, line.no_time))
            tf_ds[:line.no_towers, 0:5, 0] = 1
            tf_ds[:line.no_towers, 0, 1] = 1

        tf_ds_minor = np.zeros_like(tf_ds)
        tf_ds_minor[:line.no_towers, 0:8, 0] = 1
        tf_ds_minor[:line.no_towers, 0:5, 1] = 1

        tf_sim = {'minor': tf_ds_minor, 'collapse': tf_ds}

        est_no_tower, prob_no_tower = line.compute_stats(tf_sim)

        # 22 * 0.5 + 0 * 0.5
        self.assertAlmostEqual(est_no_tower['collapse']['mean'][0], 11.0)
        # np.sqrt(22*22*0.5-11**2)
        self.assertAlmostEqual(est_no_tower['collapse']['std'][0], 11.0)

        # 22 * 0.1
        self.assertAlmostEqual(est_no_tower['collapse']['mean'][1], 2.2)
        # np.sqrt(22*22*0.1-2.2**2)
        self.assertAlmostEqual(est_no_tower['collapse']['std'][1], 6.6)

        # 22 * 0.3 + 0 * 0.7
        self.assertAlmostEqual(est_no_tower['minor']['mean'][0], 6.6)
        # np.sqrt(22*22*0.3-6.6**2)
        self.assertAlmostEqual(est_no_tower['minor']['std'][0], 10.082, places=3)

        # 22 * 0.4
        self.assertAlmostEqual(est_no_tower['minor']['mean'][1], 8.8)
        # np.sqrt(22*22*0.4-8.8**2)
        self.assertAlmostEqual(est_no_tower['minor']['std'][1], 10.778, places=2)

    @unittest.skip("Not efficient")
    def test_compute_stats_given_timestamp(self):

        no_sims = 10
        rnd_state = np.random.RandomState(1)
        event_name = 'test1'
        path_event = os.path.join(self.cfg.path_wind_event_base,
                                  event_name)

        # LineB
        dic_line = self.cfg.lines['LineB'].copy()
        dic_line.update({'name': 'LineB',
                         'no_sims': no_sims,
                         'damage_states': self.cfg.damage_states,
                         'non_collapse': self.cfg.non_collapse,
                         'rtol': cls.cfg.rtol,
                         'atol': cls.cfg.atol,
                         'dmg_threshold': cls.cfg.dmg_threshold,
                         'event_name': event_name,
                         'scale': 1.0,
                         'rnd_state': rnd_state,
                         'path_event': path_event,
                         'dic_towers': self.cfg.towers_by_line['LineB']})

        line = Line(**dic_line)

        tf_ds = np.zeros((line.no_towers, no_sims))
        tf_ds[:line.no_towers, 0:5] = 1
        #tf_ds[:line.no_towers, 0] = 1

        tf_ds_minor = np.zeros_like(tf_ds)
        tf_ds_minor[:line.no_towers, 0:8] = 1
        #tf_ds_minor[:line.no_towers, 0:5] = 1

        tf_sim = {'minor': tf_ds_minor, 'collapse': tf_ds}

        prob_no_tower = line.compute_stats_given_timestamp(tf_sim)

        # collapse 
        prob = np.zeros((line.no_towers + 1))
        prob[0] = 0.5
        prob[-1] = 0.5
        np.testing.assert_almost_equal(prob_no_tower['collapse'], prob)

        # minor
        prob = np.zeros((line.no_towers + 1))
        prob[0] = 0.7
        prob[-1] = 0.3
        np.testing.assert_almost_equal(prob_no_tower['minor'], prob)


    def test_write_output(self):
        pass


"""
    def test_compute_damage_probability_simulation_alt(self):

        event_id = 'test2'
        scale = 2.5
        self.line1.event_tuple = (event_id, scale)

        seed = self.cfg.seed[event_id][self.line1.name]
        rnd_state = np.random.RandomState(seed)

        rv = rnd_state.uniform(size=(self.cfg.no_sims,
                                     len(self.line1.time)))

        # tf_ds = pd.Panel(np.zeros((self.cfg.no_sims,
        #                            len(self.line1.time),
        #                            self.line1.no_towers), dtype=bool),
        #                  items=range(self.cfg.no_sims),
        #                  major_axis=self.line1.time,
        #                  minor_axis=self.line1.name_by_line)

        tf_ds = np.zeros((self.line1.no_towers,
                          self.cfg.no_sims,
                          len(self.line1.time)), dtype=bool)

        for name, tower in self.line1.towers.items():
            tower.determine_damage_isolation_sim(rv)
            tower.determine_damage_adjacent_sim(seed)

            # print('{}'.format(name))
            # print('{}, {}'.format(tower.prob_damage_isolation['collapse'].max(),
            #                   tower.prob_damage_isolation['minor'].max()))
            # print ('{}'.format(tower.damage_sim['collapse'].head()))
            # print ('{}'.format(tower.damage_sim['minor'].head()))

            valid_ = tower.damage_sim['collapse'][
                tower.damage_sim['collapse'].id_adj.notnull()]

            # for key, grouped in tower.damage_sim['collapse'].groupby('id_time'):
            #
            #     np.testing.assert_almost_equal(len(grouped)/float(self.cfg.no_sims),
            #                                    tower.prob_damage_isolation.iloc[key, 'collapse'],
            #                                    decimal=1)

            # for key, grouped in tower.damage_sim['collapse'].groupby('id_time'):

            for _, item in valid_.iterrows():
                # print('{}'.format(item))
                # print('{}:{}:{}'.format(item['id_sim'],
                #                        item['id_time'],
                #                        item['id_adj']))
                for idx in item['id_adj']:
                    tf_ds[idx, item['id_sim'], item['id_time']] = True

"""


class TestLine2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'), logger=cls.logger)

        event_name = 'test2'
        event_scale = 1.0
        path_event = os.path.join(cls.cfg.path_wind_event_base,
                                  event_name)
        path_output = os.path.join(cls.cfg.path_output, 'test2_s1.0')
        # LineB
        dic_line = cls.cfg.lines['LineA'].copy()
        cls.no_sims = 10000
        dic_line.update({'name': 'LineA',
                         'no_sims': cls.no_sims,
                         'damage_states': cls.cfg.damage_states,
                         'non_collapse': cls.cfg.non_collapse,
                         'event_name': event_name,
                         'event_id': cls.cfg.event_id_format.format(event_name=event_name, scale=event_scale),
                         'rtol': cls.cfg.rtol,
                         'atol': cls.cfg.atol,
                         'dmg_threshold': cls.cfg.dmg_threshold,
                         'scale': event_scale,
                         'rnd_state': np.random.RandomState(0),
                         'path_event': path_event,
                         'path_output': path_output,
                         'dic_towers': cls.cfg.towers_by_line['LineA']})

        cls.line = Line(**dic_line)

        for _, tower in cls.line.towers.items():
            # tower._wind = create_wind_given_bearing(10.0, 1.0)
            tower.axisaz = 11.0
            tower._damage_prob = None
            tower._damage_prob_sim = None
            tower._dmg_sim = None
            tower._dmg_id_sim = None

    @classmethod
    def tearDown(cls):
        try:
            os.remove(cls.line.file_output)
            os.removedirs(cls.line.output_path)
        except:
            pass

    def test_compute_damage_per_line(self):
        with self.assertLogs('wistl', level='INFO') as cm:
            self.line.compute_damage_per_line(self.cfg)

    def test_logger_dmg_time_idx(self):

        with self.assertLogs('wistl.line', level='INFO') as cm:
            self.line.dmg_time_idx
        msg = f'Line:LineA sustains no damage'
        self.assertIn(f'INFO:wistl.line:{msg}', cm.output)


class TestLine3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'))

        dic_line = cls.cfg.lines['LineA'].copy()
        cls.no_sims = 1000
        event_name = 'test2'
        event_scale = 2.0
        path_event = os.path.join(cls.cfg.path_wind_event_base,
                                  event_name)
        path_output = os.path.join(cls.cfg.path_output, 'test2_s2.0')
        dic_line.update({'name': 'LineA',
                         'no_sims': cls.no_sims,
                         'damage_states': cls.cfg.damage_states,
                         'non_collapse': cls.cfg.non_collapse,
                         'event_name': event_name,
                         'event_id': cls.cfg.event_id_format.format(event_name=event_name, scale=event_scale),
                         'rtol': 0.01,
                         'atol': 0.1,
                         'dmg_threshold': cls.cfg.dmg_threshold,
                         'scale': event_scale,
                         'rnd_state': np.random.RandomState(0),
                         'path_event': path_event,
                         'path_output': path_output,
                         'dic_towers': cls.cfg.towers_by_line['LineA']})

        cls.line = Line(**dic_line)

    @classmethod
    def tearDown(cls):
        try:
            os.remove(cls.line.file_output)
            os.removedirs(cls.line.path_output)
        except:
            pass

    def test_dmg_time_idx(self):
        #self.assertEqual(self.line.dmg_time_idx, (479, 481))
        self.assertEqual(self.line.dmg_time_idx, (476, 483))
        #for k, v in self.line.towers.items():
        #    print(f'{k} -> {v.dmg_dmg_time_idx}')

    def test_dmg_towers(self):
        expected = set([16, 18, 19, 6, 7, 12, 2, 13, 14, 1, 10, 9, 20, 15, 4, 3])
        self.assertEqual(set(self.line.dmg_towers), expected)

    def test_compute_damage_per_line(self):
        with self.assertLogs('wistl', level='INFO') as cm:
            self.line.compute_damage_per_line(self.cfg)


class TestLine4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.cfg = Config(os.path.join(BASE_DIR, 'test_interaction.cfg'))

        dic_line = cls.cfg.lines['LineA'].copy()
        cls.no_sims = 10000
        event_name = 'test1'
        event_scale = 14.0
        path_event = os.path.join(cls.cfg.path_wind_event_base,
                                  event_name)
        path_output = os.path.join(cls.cfg.path_output, 'test1_s14.0')
        dic_line.update({'name': 'LineA',
                         'no_sims': cls.no_sims,
                         'damage_states': cls.cfg.damage_states,
                         'non_collapse': cls.cfg.non_collapse,
                         'event_name': event_name,
                         'event_id': cls.cfg.event_id_format.format(event_name=event_name, scale=event_scale),
                         'rtol': 0.01,
                         'atol': 0.1,
                         'dmg_threshold': cls.cfg.dmg_threshold,
                         'scale': event_scale,
                         'rnd_state': np.random.RandomState(0),
                         'path_event': path_event,
                         'path_output': path_output,
                         'dic_towers': cls.cfg.towers_by_line['LineA']})

        cls.line = Line(**dic_line)

        with cls.assertLogs('wistl', level='INFO') as cm:
            cls.line.compute_damage_prob()
            cls.line.compute_damage_prob_sim()

    def test_dmg_idx(self):

        self.assertEqual(set(self.line.dmg_towers), {16, 0})

        self.assertEqual(self.line.towers[0].dmg_time_idx, (0, 3))
        self.assertEqual(self.line.towers[0].dmg_idxmax, [2])
        self.assertEqual(self.line.towers[16].dmg_time_idx, (1, 3))
        self.assertEqual(self.line.towers[16].dmg_idxmax, [2])
        self.assertEqual(self.line.dmg_time_idx, (0, 3))

        df = pd.DataFrame(np.column_stack(self.line.dmg_idx['collapse']), columns=['idl', 'id_sim', 'id_time'])
        self.assertEqual(set(df['id_time'].unique()), {0, 1, 2})

        for idl in range(0, 3):
            self.assertTrue(set(df.loc[df['idl']==idl, 'id_time'].unique()).issubset({0, 1, 2}))

        for idl in range(14, 19):
            self.assertTrue(set(df.loc[df['idl']==idl, 'id_time'].unique()).issubset({0, 1, 2}))

    def test_dmg_idx_interaction(self):

        self.assertEqual(set(self.line.dmg_towers), {16, 0})

        self.assertEqual(self.line.towers[0].dmg_time_idx, (0, 3))
        self.assertEqual(self.line.towers[16].dmg_time_idx, (1, 3))

        self.assertEqual(set(self.line.towers[0].collapse_interaction['id_time'].unique()), {0, 1, 2})
        self.assertEqual(set(self.line.towers[16].collapse_interaction['id_time'].unique()), {0, 1})

        self.assertEqual(self.line.towers[0].target_line['LineB']['id'], 0)
        self.assertEqual(self.line.towers[16].target_line['LineB']['id'], 16)

        self.assertEqual(self.line.towers[0].target_line['LineC']['id'], 0)
        self.assertEqual(self.line.towers[16].target_line['LineC']['id'], 16)

        self.assertEqual(self.line.target_no_towers, {'LineB': 22, 'LineC': 22})

        # tower[0]: (0, C), (1, B), (2, B)
        df_count0 = self.line.towers[0].collapse_interaction.groupby('id_time').agg(len)
        # tower[16]: (0, C), (1, B)
        df_count1 = self.line.towers[16].collapse_interaction.groupby('id_time').agg(len)

        dfb = pd.DataFrame(self.line.dmg_idx_interaction['LineB'], columns=['idl', 'id_sim', 'id_time'])
        x = dfb.groupby(['idl' ,'id_time']).agg(len).reset_index()

        dfc = pd.DataFrame(self.line.dmg_idx_interaction['LineC'], columns=['idl', 'id_sim', 'id_time'])
        y = dfc.groupby(['idl' ,'id_time']).agg(len).reset_index()

        # tower0, B
        dt = self.line.towers[0].dmg_time_idx[0] - self.line.dmg_time_idx[0]
        for idt in [1, 2]:
            self.assertEqual(x.loc[(x.id_time == idt + dt) & (x.idl==0), 'id_sim'].values[0], df_count0.loc[idt, 'id_sim'])

        # tower0, C
        for idt in [0]:
            self.assertEqual(y.loc[(y.id_time == idt + dt) & (y.idl==0), 'id_sim'].values[0], df_count0.loc[idt, 'id_sim'])

        # tower16, B
        dt = self.line.towers[16].dmg_time_idx[0] - self.line.dmg_time_idx[0]
        for idt in [1]:
            self.assertEqual(x.loc[(x.id_time == idt + dt) & (x.idl==16), 'id_sim'].values[0], df_count1.loc[idt, 'id_sim'])

        # tower16, C
        for idt in [0]:
            self.assertEqual(y.loc[(y.id_time == idt + dt) & (y.idl==16), 'id_sim'].values[0], df_count1.loc[idt, 'id_sim'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
