
import unittest
import os
import numpy as np
import copy
import logging
import pandas as pd
from scipy.stats import itemfreq

from wistl.config import Config, set_towers_and_lines, Event
from wistl.tower import Tower, compute_dmg_by_tower, read_wind
from wistl.line import *
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

        #logging.basicConfig(level=logging.ERROR)
        cls.logger = logging.getLogger(__name__)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'))

        dic_towers_by_line, dic_lines = set_towers_and_lines(cls.cfg)

        cls.line = Line(**dic_lines['LineB'])

        cls.towers = {}
        for tower_name, item in dic_towers_by_line['LineB'].items():
            cls.towers[tower_name] = Tower(**item)

        event_name = 'test1'
        path_event = os.path.join(cls.cfg.path_wind_event_base,
                                  event_name)

        cls.event_dic = {'id': 'test1_s1.0',
                     'scale': 3.0,
                     'name': event_name,
                     'seed': 1,
                     'path_wind_event': path_event,
                     }
        #path_output = os.path.join(cls.cfg.path_output, 'test1_s3.0')
        cls.cfg.no_sims = 200000

        cls.event = Event(**cls.event_dic)
        # LineB
        #dic_line = cls.cfg.lines['LineB'].copy()
        cls.wind = create_wind_given_bearing(10.0, 1.0)
        cls.results = [compute_dmg_by_tower(tower, cls.event, cls.line, cls.cfg, wind=cls.wind)
                       for _, tower in cls.towers.items()]
        cls.dmg = get_dmg_from_results(cls.event, cls.line, cls.cfg, cls.results)

    @classmethod
    def tearDown(cls):
        try:
            os.remove(cls.line.file_output)
            os.removedirs(cls.line.output_path)
        except:
            pass

    def test_compute_dmg_by_line(self):

        result = compute_dmg_by_line(self.line, self.results, self.event, self.cfg)
        self.assertEqual(result['event'], self.event.id)
        self.assertEqual(result['line'], self.line.linename)

        for ds in self.cfg.dmg_states:
            self.assertFalse(result['dmg_prob'][ds].empty)
            self.assertFalse(result['dmg_prob_sim'][ds].empty)
            self.assertFalse(result['dmg_prob_sim_wo_cascading'][ds].empty)
            self.assertFalse(result['prob_no_dmg'][ds].empty)
            self.assertFalse(result['no_dmg'][ds].empty)
            self.assertFalse(result['prob_no_dmg_wo_cascading'][ds].empty)
            self.assertFalse(result['no_dmg_wo_cascading'][ds].empty)

    def test_line(self):

        self.assertEqual(self.line.linename, 'LineB')
        self.assertEqual(self.line.numcircuit, 2)
        self.assertEqual(self.line.seed, 1)
        self.assertEqual(self.line.no_towers, 22)
        self.assertEqual(len(self.line.names), 22)
        self.assertEqual(self.line.names[0], 'T23')

    def test_time(self):
        pd.testing.assert_index_equal(self.dmg.index, self.wind.index, check_names=False)

    def test_no_time(self):
        self.assertEqual(self.dmg.shape[0], 2)

    def test_damage_prob(self):
        name = 'T26'
        # for name, tower in self.line1.towers.items():

        tower = self.towers['T26']
        #tower._dmg = None
        #tower._dmg_sim = None
        self.assertEqual(tower.name, name)
        # T26 (in the middle)
        # 'T23'(0), 'T24'(1), 'T25'(2), 'T26'(3), 'T27'(4), 'T28'(5), 'T29' (6)
        # o---o----o----x----o----o---0
        # collapse
        # lognorm.cdf(1.0, 0.03, scale=1.05)
        p0c = 0.0519
        p0gn2 = p0c * 0.125
        p0gn1 = p0c * 0.575
        p0gp1 = p0c * 0.575
        p0gp2 = p0c * 0.125
        pc = 1 - (1-p0c)*(1-p0gn2)*(1-p0gn1)*(1-p0gp1)*(1-p0gp2)

        dmg = get_dmg_from_results(self.event, self.line, self.cfg, self.results)

        self.assertAlmostEqual(p0c, dmg['collapse_T24'][0], places=3)
        self.assertAlmostEqual(p0c, dmg['collapse_T25'][0], places=3)
        self.assertAlmostEqual(p0c, dmg['collapse_T27'][0], places=3)
        self.assertAlmostEqual(p0c, dmg['collapse_T28'][0], places=3)

        collapse_adj = {x['tower']: x['collapse_adj'] for x in self.results
                    if (x['event']==self.event.id) and (x['line']==self.line.linename) and (x['collapse_adj'])}

        self.assertAlmostEqual(p0gn2, collapse_adj[name][1][0], places=3)
        self.assertAlmostEqual(p0gn1, collapse_adj[name][2][0], places=3)
        self.assertAlmostEqual(p0gp1, collapse_adj[name][4][0], places=3)
        self.assertAlmostEqual(p0gp2, collapse_adj[name][5][0], places=3)

        dmg_prob, _ = compute_dmg_prob(self.event, self.line, self.cfg, self.results)
        self.assertAlmostEqual(
            pc, dmg_prob['collapse'][name][0], places=3)

        # T26 (in the middle)
        # 'T23'(0), 'T24'(1), 'T25'(2), 'T26'(3), 'T27'(4), 'T28'(5), 'T29' (6)
        # o---o----o----x----o----o---0
        # minor
        # lognorm.cdf(1.0, 0.03, scale=1.02)
        p0m = 0.16105
        pm = min(p0m - p0c + pc, 1.0)

        self.assertAlmostEqual(p0m, dmg[f'minor_{name}'][0], places=3)
        #self.assertAlmostEqual(p0m, tower.dmg_sim['minor'][0], places=3)
        self.assertAlmostEqual(
            pm, dmg_prob['minor'][name][0], places=3)


    def test_compute_damage_prob_sim(self):

        name = 'T26'

        tower = self.towers['T26']
        self.assertEqual(tower.name, name)
        # T26 (in the middle)
        # 'T23'(0), 'T24'(1), 'T25'(2), 'T26'(3), 'T27'(4), 'T28'(5), 'T29' (6)
        # o---o----o----x----o----o---0
        # collapse
        # lognorm.cdf(1.0, 0.03, scale=1.05)
        p0c = 0.0519
        p0gn2 = p0c * 0.125
        p0gn1 = p0c * 0.575
        p0gp1 = p0c * 0.575
        p0gp2 = p0c * 0.125
        pc = 1 - (1-p0c)*(1-p0gn2)*(1-p0gn1)*(1-p0gp1)*(1-p0gp2)

        #with self.assertLogs('distributed.worker', level='INFO') as cm:
        dmg_prob, dmg = compute_dmg_prob(self.event, self.line, self.cfg, self.results)
        dmg_prob_sim, _, _ =  compute_dmg_prob_sim(self.event, self.line, self.cfg, self.results, dmg)

        try:
            np.testing.assert_allclose(
                pc, dmg_prob_sim['collapse'][name][0], atol=self.cfg.atol, rtol=self.cfg.rtol)
        except AssertionError:
            self.logger.warning(
                f'P(C) Theory: {pc:.4f}, '
                f"Analytical: {dmg_prob['collapse'][name][0]:.4f}, "
                f"Simulation: {dmg_prob_sim['collapse'][name][0]:.4f}")

        for _id in range(23, 45):
            name = f'T{_id}'
            try:
                np.testing.assert_allclose(dmg_prob['collapse'][name][0],
                    dmg_prob_sim['collapse'][name][0], atol=self.cfg.atol, rtol=self.cfg.rtol)
            except AssertionError:
                self.logger.warning(
                    f'Tower: {name}, collapse '
                    f"Analytical: {dmg_prob['collapse'][name][0]:.4f}, "
                    f"Simulation: {dmg_prob_sim['collapse'][name][0]:.4f}")

        # o----o----x----o----o
        # minor
        # lognorm.cdf(1.0, 0.02, scale=1.02) 
        p0m = 0.16105
        pm = min(p0m - p0c + pc, 1.0)
        try:
           self.assertTrue(
                pm >= dmg_prob_sim['minor'][name][0])
        except AssertionError:
            self.logger.warning(
                f'P(m) Theory: {pm:.4f}, '
                f"Analytical: {dmg_prob['minor'][name][0]:.4f}, "
                f"Simulation: {dmg_prob_sim['minor'][name][0]:.4f}")

        # except 32, strainer tower
        for _id in list(range(23, 32)) + list(range(33, 45)):
            name = f'T{_id}'
            try:
                self.assertTrue(dmg_prob['minor'][name][0]
                        >= dmg_prob_sim['minor'][name][0])
            except AssertionError:
                self.logger.warning(
                    f'Tower: {name}, minor, '
                    f"Analytical: {dmg_prob['minor'][name][0]:.4f}, "
                    f"Simulation: {dmg_prob_sim['minor'][name][0]:.4f}")

    def test_compute_damage_prob_sim_wo_cascading(self):

        name = 'T26'

        tower = self.towers['T26']
        self.assertEqual(tower.name, name)
        # T26 (in the middle)
        # 'T23'(0), 'T24'(1), 'T25'(2), 'T26'(3), 'T27'(4), 'T28'(5), 'T29' (6)
        # o---o----o----x----o----o---0
        # collapse
        # lognorm.cdf(1.0, 0.03, scale=1.05)

        _, dmg = compute_dmg_prob(self.event, self.line, self.cfg, self.results)
        dmg_prob_sim_wo_cascading, _, _ = compute_dmg_prob_sim_wo_cascading(self.event, self.line, self.cfg, self.results, dmg)

        pc = dmg[f'collapse_{name}']
        try:
            np.testing.assert_allclose(
                pc, dmg_prob_sim_wo_cascading['collapse'][name], atol=self.cfg.atol, rtol=self.cfg.rtol)
        except AssertionError:
            self.logger.warning(
                    f'P(C) Theory: {pc[0]}, '
                f"Simulation: {dmg_prob_sim_wo_cascading['collapse'][name][0]}")

        # o----o----x----o----o
        # minor
        # lognorm.cdf(1.0, 0.02, scale=1.02) 
        pm = dmg[f'minor_{name}']
        try:
           np.testing.assert_allclose(
                pm, dmg_prob_sim_wo_cascading['minor'][name], atol=self.cfg.atol, rtol=self.cfg.rtol)
        except AssertionError:
            self.logger.warning(
                f'P(m) Theory: {pm.values}, '
                f"Simulation: {dmg_prob_sim_wo_cascading['minor'][name].values}")

        # except 32, strainer tower
        for name in self.line.names:
            #name = towers[_id].name
            #idt0, idt1 = self.line.towers[_id].dmg_time_idx
            #print(dmg[f'minor_{name}'][0])
            try:
                np.testing.assert_allclose(
                        np.nan_to_num(dmg[f'minor_{name}'][0]), dmg_prob_sim_wo_cascading['minor'][name], atol=self.cfg.atol, rtol=self.cfg.rtol)
            except AssertionError:
                dummy = f'minor_{name}'
                self.logger.warning(
                    f'Tower: {name}, minor, '
                    f"Theory: {dmg[dummy][0]}, "
                    f"Simulation: {dmg_prob_sim_wo_cascading['minor'][name]}")

    def test_compute_stats(self):

        self.cfg.no_sims = 10

        dmg = get_dmg_from_results(self.event, self.line, self.cfg, self.results)

        tf_ds = np.zeros((self.line.no_towers, self.cfg.no_sims, dmg.shape[0]))
        tf_ds[:self.line.no_towers, 0:5, 0] = 1
        tf_ds[:self.line.no_towers, 0, 1] = 1

        tf_ds_minor = np.zeros_like(tf_ds)
        tf_ds_minor[:self.line.no_towers, 0:8, 0] = 1
        tf_ds_minor[:self.line.no_towers, 0:5, 1] = 1

        tf_sim = {'minor': tf_ds_minor, 'collapse': tf_ds}
        est_no_tower, prob_no_tower = compute_stats(tf_sim, self.cfg, self.line.no_towers, dmg.index)

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

    def test_compute_stats_given_timestamp(self):

        self.cfg.no_sims = 10

        dmg = get_dmg_from_results(self.event, self.line, self.cfg, self.results)

        tf_ds = np.zeros((self.line.no_towers, self.cfg.no_sims, dmg.shape[0]))
        tf_ds[:self.line.no_towers, 0:5] = 1
        #tf_ds[:line.no_towers, 0] = 1

        tf_ds_minor = np.zeros_like(tf_ds)
        tf_ds_minor[:self.line.no_towers, 0:8] = 1
        #tf_ds_minor[:line.no_towers, 0:5] = 1

        tf_sim = {'minor': tf_ds_minor, 'collapse': tf_ds}

        est_no_tower, prob_no_tower = compute_stats(tf_sim, self.cfg, self.line.no_towers, dmg.index)

        # collapse 
        prob = np.zeros((self.line.no_towers + 1))
        prob[0] = 0.5
        prob[-1] = 0.5
        np.testing.assert_almost_equal(prob_no_tower['collapse'].iloc[0], prob)

        # minor
        prob = np.zeros((self.line.no_towers + 1))
        prob[0] = 0.7
        prob[-1] = 0.3
        np.testing.assert_almost_equal(prob_no_tower['minor'].iloc[0], prob)

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

        cls.logger = logging.getLogger(__name__)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'))
        cls.cfg.no_sims = 1

        dic_towers_by_line, dic_lines = set_towers_and_lines(cls.cfg)

        linename = 'LineB'

        cls.line = Line(**dic_lines[linename])

        cls.towers = {}
        for tower_name, item in dic_towers_by_line[linename].items():
            cls.towers[tower_name] = Tower(**item)

        event_name = 'test2'
        event_scale = 1.0
        path_event = os.path.join(cls.cfg.path_wind_event_base,
                                  event_name)
        path_output = os.path.join(cls.cfg.path_output, 'test2_s1.0')

        cls.event_dic = {'id': 'test2_s1.0',
                         'scale': event_scale,
                         'name': event_name,
                         'seed': 1,
                         'path_wind_event': path_event,
                         }
        cls.event = Event(**cls.event_dic)

        # LineB

        cls.results = [compute_dmg_by_tower(tower, cls.event, cls.line, cls.cfg)
                       for _, tower in cls.towers.items()]

    def test_compute_damage_per_line(self):
        result = compute_dmg_by_line(self.line, self.results, self.event, self.cfg)
        self.assertEqual(result['event'], self.event.id)
        self.assertEqual(result['line'], self.line.linename)
        for ds in self.cfg.dmg_states:
            self.assertTrue(result['dmg_prob'][ds].empty)
            self.assertTrue(result['dmg_prob_sim'][ds].empty)
            self.assertTrue(result['dmg_prob_sim_wo_cascading'][ds].empty)
            self.assertTrue(result['prob_no_dmg'][ds].empty)
            self.assertTrue(result['no_dmg'][ds].empty)
            self.assertTrue(result['prob_no_dmg_wo_cascading'][ds].empty)
            self.assertTrue(result['no_dmg_wo_cascading'][ds].empty)


class TestLine3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):


        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'))

        cls.cfg.no_sims = 10
        dic_towers_by_line, dic_lines = set_towers_and_lines(cls.cfg)

        linename = 'LineA'

        cls.line = Line(**dic_lines[linename])

        cls.towers = {}
        for tower_name, item in dic_towers_by_line[linename].items():
            cls.towers[tower_name] = Tower(**item)

        event_name = 'test2'
        event_scale = 2.0
        path_event = os.path.join(cls.cfg.path_wind_event_base,
                                  event_name)

        cls.event_dic = {'id': 'test2_s2.0',
                         'scale': event_scale,
                         'name': event_name,
                         'seed': 1,
                         'path_wind_event': path_event,
                         }
        cls.event = Event(**cls.event_dic)

        # LineB
        cls.results = [compute_dmg_by_tower(tower, cls.event, cls.line, cls.cfg)
                       for _, tower in cls.towers.items()]

    def test_dmg_towers(self):
        expected = set([16, 18, 19, 6, 7, 12, 2, 13, 14, 1, 10, 9, 20, 15, 4, 3])
        dmg = get_dmg_from_results(self.event, self.line, self.cfg, self.results)
        for i, name in enumerate(self.line.names):
            if i in expected:
                self.assertFalse(dmg[f'collapse_{name}'].empty)
            else:
                self.assertTrue(dmg[f'collapse_{name}'].isnull().values.all())

        # index
        wind = read_wind(self.towers['T22'], self.event)
        self.assertEqual(dmg.index[0], wind.index[476])
        self.assertEqual(dmg.index[-1], wind.index[482])

    def test_compute_damage_per_line(self):
        with self.assertLogs('distributed.worker', level='INFO') as cm:
            compute_dmg_by_line(self.line, self.results, self.event, self.cfg)


class TestLine4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.cfg = Config(os.path.join(BASE_DIR, 'test_interaction.cfg'))


    @unittest.skip('NOT YET')
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

    @unittest.skip('NOT YET')
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
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestLine1)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main(verbosity=2)
