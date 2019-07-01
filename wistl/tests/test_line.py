
import unittest
import os
import numpy as np
import copy
import logging
import pandas as pd
from scipy.stats import itemfreq

from wistl.config import Config
from wistl.line import Line
from wistl.tests.test_config import assertDeepAlmostEqual
# from wistl.transmission_network import read_shape_file, populate_df_lines, \
#     populate_df_towers

ATOL = 0.001
RTOL = 0.05
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class TestLine1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'), logger=cls.logger)

        event_name = 'test1'
        path_event = os.path.join(cls.cfg.path_wind_scenario_base,
                                  event_name)
        # cls.cfg.no_sims = 10000
        # cls.all_towers = read_shape_file(cls.cfg.file_shape_tower)
        # cls.all_towers.loc[1, 'Function'] = 'Suspension'
        # populate_df_towers(cls.all_towers, cls.cfg)
        #
        # cls.all_lines = read_shape_file(cls.cfg.file_shape_line)
        # populate_df_lines(cls.all_lines)

        # LineB
        dic_line = cls.cfg.lines['LineB'].copy()
        cls.no_sims = 1000000
        dic_line.update({'no_sims': cls.no_sims,
                         'damage_states': cls.cfg.damage_states,
                         'non_collapse': cls.cfg.non_collapse,
                         'scale': 1.0,
                         'rnd_state': np.random.RandomState(0),
                         'path_event': path_event,
                         'dic_towers': cls.cfg.towers_by_line['LineB']})

        cls.line = Line(name='LineB', **dic_line)

        for _, tower in cls.line.towers.items():
            tower.wind['ratio'] = 1.0
            tower._damage_prob = None
            tower._damage_prob_sim = None

        cls.line.compute_damage_prob()

    def test_set_towers(self):

        self.assertEqual(self.line.no_towers, 22)
        self.assertEqual(self.line.no_time, 3)

        self.assertEqual(self.line.towers[0].name, 'T23')
        self.assertEqual(self.line.towers[0].idl, 0)
        self.assertEqual(self.line.towers[0].idn, 24)
        self.assertEqual(self.line.towers[0].no_time, 3)
        self.assertEqual(self.line.towers[0].no_sims, self.no_sims)
        self.assertEqual(self.line.towers[0].damage_states, ['minor', 'collapse'])
        self.assertAlmostEqual(self.line.towers[0].scale, 1.0)

    def test_compute_damage_prob(self):

        # for name, tower in self.line1.towers.items():

        tower = self.line.towers[10]
        self.assertEqual(tower.name, 'T33')

        # T33 (in the middle)
        # o----o----x----o----o
        # collapse
        p0c = 0.0519
        p0gn2 = p0c * 0.125
        p0gn1 = p0c * 0.575
        p0gp1 = p0c * 0.575
        p0gp2 = p0c * 0.125
        pc = 1 - (1-p0c)*(1-p0gn2)*(1-p0gn1)*(1-p0gp1)*(1-p0gp2)

        self.assertAlmostEqual(p0c, tower.dmg['collapse'][0], places=3)
        self.assertAlmostEqual(p0gn2, tower.collapse_adj[8][0], places=3)
        self.assertAlmostEqual(p0gn1, tower.collapse_adj[9][0], places=3)
        self.assertAlmostEqual(p0gp1, tower.collapse_adj[11][0], places=3)
        self.assertAlmostEqual(p0gp2, tower.collapse_adj[12][0], places=3)
        self.assertAlmostEqual(
            pc, self.line.damage_prob['collapse']['T33'][0], places=3)

        # T33 (in the middle)
        # o----o----x----o----o
        # minor
        p0m = 0.1610
        pm = min(p0m - p0c + pc, 1.0)

        self.assertAlmostEqual(p0m, tower.dmg['minor'][0], places=3)
        self.assertAlmostEqual(
            pm, self.line.damage_prob['minor']['T33'][0], places=3)

    def test_compute_damage_prob_sim(self):

        self.line.compute_damage_prob_sim()

        tower = self.line.towers[10]
        self.assertEqual(tower.name, 'T33')

        # T33 (in the middle)
        # o----o----x----o----o
        # collapse
        p0c = 0.0519
        p0gn2 = p0c * 0.125
        p0gn1 = p0c * 0.575
        p0gp1 = p0c * 0.575
        p0gp2 = p0c * 0.125
        pc = 1 - (1-p0c)*(1-p0gn2)*(1-p0gn1)*(1-p0gp1)*(1-p0gp2)

        # np.testing.assert_allclose(
        #     pc, self.line.damage_prob_sim['collapse']['T33'][0],
        #     atol=ATOL,
        #     rtol=RTOL)

        msg = 'P(C) Theory: {:.4f}, Simulation: {:.4f}, Analytical: {:.4f}'
        self.logger.info(msg.format(pc, self.line.damage_prob_sim['collapse']['T33'][0],
              self.line.damage_prob['collapse']['T33'][0]))

        # T33 (in the middle)
        # o----o----x----o----o
        # minor
        p0m = 0.1610
        pm = min(p0m - p0c + pc, 1.0)

        # np.testing.assert_allclose(
        #     pm, self.line.damage_prob_sim['minor']['T33'][0],
        #     atol=ATOL,
        #     rtol=RTOL)

        msg = 'P(M) Theory: {:.4f}, Simulation: {:.4f}, Analytical: {:.4f}'
        self.logger.info(msg.format(pm, self.line.damage_prob_sim['minor']['T33'][0],
              self.line.damage_prob['minor']['T33'][0]))
    """

    def test_compute_stats(self):

        no_sims = 10
        rnd_state = np.random.RandomState(1)
        event_name = 'test1'
        path_event = os.path.join(self.cfg.path_wind_scenario_base,
                                  event_name)

        # LineB
        dic_line = self.cfg.lines['LineB'].copy()
        dic_line.update({'no_sims': no_sims,
                         'damage_states': self.cfg.damage_states,
                         'non_collapse': self.cfg.non_collapse,
                         'event_name': event_name,
                         'scale': 1.0,
                         'rnd_state': rnd_state,
                         'path_event': path_event,
                         'dic_towers': self.cfg.towers_by_line['LineB']})

        line = Line(name='LineB', **dic_line)

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

    # def test_sort_by_location(self):
    #
    #     # Calaca - Amadeo
    #     self.assertEqual([3, 4, 5, 0, 2, 1], self.line1.sort_by_location())
    #     self.assertEqual([0, 1, 2, 3, 4, 5], self.line1.id_by_line)

    # def test_assign_id_both_sides(self):
    #
    #     # Calaca - Amadeo
    #     expected = {0: (-1, 1), 1: (0, 2), 2: (1, 3),
    #                 3: (2, 4), 4: (3, 5), 5: (4, -1)}
    #     for i in range(5):
    #         outcome = self.line0.assign_id_both_sides(i)
    #         self.assertEqual(expected[i], outcome)
    #
    #     for i in range(5):
    #         outcome = self.line1.assign_id_both_sides(i)
    #         self.assertEqual(expected[i], outcome)

    # def test_assign_id_adj_towers(self):
    #
    #     # Calaca - Amadeo
    #     id_line = 0
    #     df_line = self.all_lines.loc[id_line, :]
    #     tf = self.all_towers['LineRoute'] == df_line['LineRoute']
    #     df_towers = copy.deepcopy(self.all_towers.loc[tf, :])
    #     df_towers.loc[:, 'Function'] = 'Suspension'
    #     line = Line(self.cfg, df_towers, df_line)
    #
    #     expected = {0: [-1, -1, 0, 1, 2],
    #                 1: [-1, 0, 1, 2, 3],
    #                 2: [0, 1, 2, 3, 4],
    #                 3: [1, 2, 3, 4, 5],
    #                 4: [2, 3, 4, 5, -1],
    #                 5: [3, 4, 5, -1, -1]}
    #     for i in line.id_by_line:
    #         outcome = line.assign_id_adj_towers(i)
    #         # max_no = line.towers[line.id2name[tid]].max_no_adj_towers
    #         self.assertEqual(expected[i], outcome)
    #
    #     # # a bit of change
    #     expected = {0: [-1, -1, 0, 1, 2],
    #                 1: [-1, 0, 1, 2, 3],
    #                 2: [0, 1, 2, 3, 4],
    #                 3: [-1, -1, -1, 0, 1, 2, 3, 4, 5, -1, -1, -1, -1],
    #                 4: [2, 3, 4, 5, -1],
    #                 5: [3, 4, 5, -1, -1]}
    #
    #     for i in self.line1.id_by_line:
    #         outcome = self.line1.assign_id_adj_towers(i)
    #         # print('{}:{}'.format(outcome, expected[i]))
    #         self.assertEqual(expected[i], outcome)

    # def test_create_list_idx(self):
    #
    #     self.assertEqual(self.line0.create_list_idx(0, 4, +1),
    #                      [1, 2, 3, 4])
    #     self.assertEqual(self.line0.create_list_idx(2, 4, -1),
    #                      [1, 0, -1, -1])

    # def test_update_id_adj_by_filtering_strainer(self):
    #
    #     # tower[3] is strainer
    #     expected = {0: [-1, -1, 0, 1, 2],
    #                 1: [-1, 0, 1, 2, -1],
    #                 2: [0, 1, 2, -1, 4],
    #                 3: [-1, -1, -1, 0, 1, 2, -1, 4, 5, -1, -1, -1, -1],
    #                 4: [2, -1, 4, 5, -1],
    #                 5: [-1, 4, 5, -1, -1]}
    #
    #     for _, tower in self.line1.towers.items():
    #         # print('{}:{}:{}'.format(tower.id,
    #         #                         tower.id_adj,
    #         #                         expected[tower.id]))
    #         self.assertEqual(tower.id_adj, expected[tower.id])

    # def test_calculate_cond_pc_adj(self):
    #
    #     # FIXME: Need to explain the methodology
    #
    #     """
    #
    #     - AC-099 : Terminal
    #     [-1, -1, 1, 5, 4]
    #
    #     probability          list
    #       0.075             (0, 1)  (0, 1)
    #       0.075            (-1, 0)  x
    #       0.350         (-1, 0, 1)  (0, 1)
    #       0.025     (-2, -1, 0, 1)  (0, 1)
    #       0.025      (-1, 0, 1, 2)  (0, 1, 2)
    #       0.100  (-2, -1, 0, 1, 2)  (0, 1, 2)
    #
    #       rel_idx: (0, 1, 2), (0,1),
    #       cum_prob: 0.125, 0.45+0.125
    #
    #     - AC-100: Suspension
    #     [-1, 1, 5, 4, -1]
    #     probability          list
    #       0.075             (0, 1)  (0, 1)
    #       0.075            (-1, 0)  (-1, 0)
    #       0.350         (-1, 0, 1)  (-1, 0, 1)
    #       0.025     (-2, -1, 0, 1)  (-1, 0, 1)
    #       0.025      (-1, 0, 1, 2)  (-1, 0, 1)
    #       0.100  (-2, -1, 0, 1, 2)  (-1, 0, 1)
    #
    #       rel_idx: (0, 1), (-1, 0), (-1, 0, 1)
    #       cum_prob: 0.075, 0.075+0.075, 0.5+0.15
    #
    #     - AC-101: Suspension
    #     [1, 5, 4, -1, 3]
    #     probability          list
    #       0.075             (0, 1)  x
    #       0.075            (-1, 0)  (-1, 0)
    #       0.350         (-1, 0, 1)  (-1, 0)
    #       0.025     (-2, -1, 0, 1)  (-2, -1, 0)
    #       0.025      (-1, 0, 1, 2)  (-1, 0)
    #       0.100  (-2, -1, 0, 1, 2)  (-2, -1, 0)
    #
    #       rel_idx: (-2, -1, 1), (-1, 0)
    #       cum_prob: 0.125, 0.125+0.45
    #
    #     - AC-102: Strainer
    #     [-1, -1, -1, 1, 5, 4, (-1), 3, 2, -1, -1, -1, -1]
    #     probability                                      list
    #        0.05                                     (-1, 0, 1) (-1,0,1)
    #        0.08                              (-2, -1, 0, 1, 2) (-2,-1,0,1,2)
    #        0.10                       (-3, -2, -1, 0, 1, 2, 3) (-3,-2,-1,0,1,2)
    #        0.08                (-4, -3, -2, -1, 0, 1, 2, 3, 4) (-3,-2,-1,0,1,2)
    #        0.05         (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5) (-3,-2,-1,0,1,2)
    #        0.62  (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6) (-3,-2,-1,0,1,2)
    #
    #       rel_idx: (-1, 0, 1), (-2, -1, 0, 1, 2), (-3, -2, -1, 0, 1, 2)
    #       cum_prob: 0.05, 0.08+0.05, 0.85+0.13
    #
    #     - AC-103: Suspension
    #     [4, -1, (3), 2, -1]
    #     probability          list
    #       0.075             (0, 1)  (0, 1)
    #       0.075            (-1, 0)  x
    #       0.350         (-1, 0, 1)  (0, 1)
    #       0.025     (-2, -1, 0, 1)  (0, 1)
    #       0.025      (-1, 0, 1, 2)  (0, 1)
    #       0.100  (-2, -1, 0, 1, 2)  (0, 1)
    #
    #       rel_idx: (0, 1)
    #       cum_prob: 0.575
    #
    #     - AC-104: Suspension
    #     [-1, 3, (2), -1, -1
    #     probability          list
    #       0.075             (0, 1)  x
    #       0.075            (-1, 0)  (-1, 0)
    #       0.350         (-1, 0, 1)  (-1, 0)
    #       0.025     (-2, -1, 0, 1)  (-1, 0)
    #       0.025      (-1, 0, 1, 2)  (-1, 0)
    #       0.100  (-2, -1, 0, 1, 2)  (-1, 0)
    #
    #       rel_idx: (-1, 0)
    #       cum_prob: 0.575
    #
    #     """
    #
    #     expected = {'AC-099': {1: 0.575, 2: 0.125},
    #                 'AC-100': {-1: 0.075 + 0.5, 1: 0.075 + 0.5},
    #                 'AC-101': {-2: 0.125, -1: 0.125 + 0.45},
    #                 'AC-102': {-3: 0.85, -2: 0.08 + 0.85,
    #                            -1: 0.05 + 0.08 + 0.85,
    #                            1: 0.05 + 0.08 + 0.85,
    #                            2: 0.08 + 0.85},
    #                 'AC-103': {1: 0.575},
    #                 'AC-104': {-1: 0.575}}
    #
    #     expected_cond_pc_adj_sim = {
    #         'AC-099': {'rel_idx': [(0, 1, 2), (0, 1)],
    #                    'cum_prob': np.array([0.125, 0.45 + 0.125])},
    #         'AC-100': {'rel_idx': [(0, 1), (-1, 0), (-1, 0, 1)],
    #                    'cum_prob': np.array([
    #                        0.075, 0.075 + 0.075, 0.5 + 0.15])},
    #         'AC-101': {'rel_idx': [(-2, -1, 0), (-1, 0)],
    #                    'cum_prob': np.array([0.125, 0.125 + 0.45])},
    #         'AC-102': {'rel_idx': [(-1, 0, 1), (-2, -1, 0, 1, 2),
    #                                (-3, -2, -1, 0, 1, 2)],
    #                    'cum_prob': np.array([0.05, 0.08 + 0.05, 0.85 + 0.13])},
    #         'AC-103': {'rel_idx': [(0, 1)],
    #                    'cum_prob': np.array([0.575])},
    #         'AC-104': {'rel_idx': [(-1, 0)],
    #                    'cum_prob': np.array([0.575])}}
    #
    #     for name, val in self.line1.towers.items():
    #         assertDeepAlmostEqual(self, val.cond_pc_adj, expected[name])
    #         assertDeepAlmostEqual(self, val.cond_pc_adj_sim,
    #                               expected_cond_pc_adj_sim[name])
    #         # print("{}:{}".format(name, val.cond_pc_adj))
    #         # print("{}:{}:{}".format(name, val.cond_pc_adj_sim,
    #         # expected_cond_pc_adj_sim[name]))

    """
    def test_compute_damage_probability_simulation_alt(self):

        event_id = 'test2'
        scale = 2.5
        self.line1.event_tuple = (event_id, scale)

        seed = self.cfg.seed[event_id][self.line1.name]
        rnd_state = np.random.RandomState(seed)

        rv = rnd_state.uniform(size=(self.cfg.no_sims,
                                     len(self.line1.time_index)))

        # tf_ds = pd.Panel(np.zeros((self.cfg.no_sims,
        #                            len(self.line1.time_index),
        #                            self.line1.no_towers), dtype=bool),
        #                  items=range(self.cfg.no_sims),
        #                  major_axis=self.line1.time_index,
        #                  minor_axis=self.line1.name_by_line)

        tf_ds = np.zeros((self.line1.no_towers,
                          self.cfg.no_sims,
                          len(self.line1.time_index)), dtype=bool)

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

        event_name = 'test1'
        path_event = os.path.join(cls.cfg.path_wind_scenario_base,
                                  event_name)
        # cls.cfg.no_sims = 10000
        # cls.all_towers = read_shape_file(cls.cfg.file_shape_tower)
        # cls.all_towers.loc[1, 'Function'] = 'Suspension'
        # populate_df_towers(cls.all_towers, cls.cfg)
        #
        # cls.all_lines = read_shape_file(cls.cfg.file_shape_line)
        # populate_df_lines(cls.all_lines)

        # LineB
        dic_line = cls.cfg.lines['LineB'].copy()
        cls.no_sims = 1000000
        dic_line.update({'no_sims': cls.no_sims,
                         'damage_states': cls.cfg.damage_states,
                         'non_collapse': cls.cfg.non_collapse,
                         'event_name': event_name,
                         'scale': 1.0,
                         'rnd_state': np.random.RandomState(0),
                         'path_event': path_event,
                         'dic_towers': cls.cfg.towers_by_line['LineB']})

        cls.line = Line(name='LineB', **dic_line)

        for _, tower in cls.line.towers.items():
            tower.wind['ratio'] = 1.0
            tower._damage_prob = None
            tower._damage_prob_sim = None

        cls.line.compute_damage_prob()

    def test_compute_damage_prob_sim_no_cascading(self):
        pass

    def test_write_hdf5(self):
        pass

    def test_compute_damage_per_line(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
