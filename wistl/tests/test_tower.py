#!/usr/bin/env python

__author__ = 'Hyeuk Ryu'

import unittest
import logging
import pandas as pd
import os
import numpy as np

from wistl.config import Config
from wistl.constants import RTOL, ATOL
from wistl.tower import Tower
from wistl.tests.test_config import assertDeepAlmostEqual

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class TestTower(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.basicConfig(level=logging.INFO)

        cls.logger = logging.getLogger(__name__)

        cls.cfg = Config(os.path.join(BASE_DIR, 'test.cfg'), logger=cls.logger)

        tower_dic = cls.cfg.towers_by_line['LineA'][0].copy()
        path_event = os.path.join(cls.cfg.path_wind_scenario_base, 'test1')
        tower_dic.update({'path_event': path_event,
                          'no_sims': 2000,
                          'damage_states': cls.cfg.damage_states,
                          'scale': 1.0,
                          'rnd_state': np.random.RandomState(1),
                          'event_name': 'test1'})

        cls.tower = Tower(tower_id=0, **tower_dic)

        cls.tower.wind['ratio'] = 1.082  # 1.05*np.exp(0.03)

        # cls.tower = Tower(tower_id=0, logger=logge**cls.cfg.towers.loc[0])
        # cls.network = TransmissionNetwork(cfg=cls.cfg, event_id='test2', scale=2.5)
        #
        # cls.tower = cls.network.lines['Calaca - Amadeo'].towers['AC-100']
        # cls.ps_tower = cls.tower.ps_tower
        # cls.tower.event_tuple = (cls.tower.file_wind, 3.0)

        # set wind file, which also sets wind and time_index
        # cls.tower.file_wind = file_wind

        # compute prob_damage_isolation and prob_damage_adjacent
        # cls.tower.compute_damage_prob_isolation()
        #cls.tower.compute_pc_adj()

    def test_damage_states(self):

        self.assertEqual(set(self.tower.damage_states),
                         set(self.cfg.fragility_metadata['limit_states']))

    def test_file_wind(self):

        assert self.tower.name == 'T14'
        expected = os.path.join(self.cfg.path_wind_scenario_base, 'test1',
                                'ts.T14.csv')
        self.assertEqual(self.tower.file_wind, expected)

    def test_compute_directional_wind_speed(self):

        tower_dic = self.cfg.towers_by_line['LineA'][0].copy()
        path_event = os.path.join(self.cfg.path_wind_scenario_base, 'test1')

        bearings = np.arange(0.0, 360.0, 22.5)

        for bearing in bearings:
            tower_dic.update({'path_event': path_event,
                              'ratio_z_to_10': 1.0,
                              'axisaz': 0.0})
            tower = Tower(tower_id=0, **tower_dic)

            wind = pd.Series({'Speed': 1.0, 'Bearing': bearing})
            result = tower.compute_directional_wind_speed(wind)
            try:
                self.assertAlmostEqual(result, 1.0)
            except AssertionError:
                print(bearing, result)

    def test_wind(self):
        # FIXME
        pass

    def test_damage_prob(self):

        self.assertAlmostEqual(self.tower.collapse_capacity, 75.0)
        self.assertAlmostEqual(self.tower.frag_arg, {'collapse': 0.03,
                                                     'minor': 0.02})
        self.assertAlmostEqual(self.tower.frag_scale, {'collapse': 1.05,
                                                       'minor': 1.02})

        self.tower.wind['ratio'] = 1.02
        self.tower._damage_prob = None
        self.assertAlmostEqual(self.tower.damage_prob['minor'].values[0], 0.5,
                               places=2)

        self.tower.wind['ratio'] = 1.02*np.exp(0.02)  # 1.04
        self.tower._damage_prob = None
        self.assertAlmostEqual(self.tower.damage_prob['minor'].values[0], 0.84,
                               places=2)

        self.tower.wind['ratio'] = 1.02*np.exp(-0.02)  # 1.0
        self.tower._damage_prob = None
        self.assertAlmostEqual(self.tower.damage_prob['minor'].values[0], 0.16,
                               places=2)

        self.tower.wind['ratio'] = 1.05
        self.tower._damage_prob = None
        self.assertAlmostEqual(self.tower.damage_prob['collapse'].values[0], 0.5,
                               places=2)

        self.tower.wind['ratio'] = 1.082  # 1.05*np.exp(0.03)
        self.tower._damage_prob = None
        self.assertAlmostEqual(self.tower.damage_prob['collapse'].values[0], 0.84,
                               places=2)

        self.tower.wind['ratio'] = 1.019  # 1.05*np.exp(-0.03)
        self.tower._damage_prob = None
        self.assertAlmostEqual(self.tower.damage_prob['collapse'].values[0], 0.16,
                               places=2)

    def test_collapse_prob_adjacent(self):

        cond_pc_adj = {11: 0.125, 12: 0.575, 14: 0.575, 15: 0.125}
        assertDeepAlmostEqual(self, dict(self.tower.cond_pc_adj), cond_pc_adj)

        self.tower.wind['ratio'] = 1.082  # 1.05*np.exp(0.03)

        self.assertEqual(self.tower.id_adj, [11, 12, 13, 14, 15])
        self.tower._damage_prob = None

        for id_abs, value in cond_pc_adj.items():

            self.assertAlmostEqual(self.tower.collapse_prob_adj[id_abs][0],
                                   value * 0.84, places=2)

    def test_damage_prob_mc(self):

        # 1. determine damage state of tower due to wind
        rnd_state = np.random.RandomState(1)
        rv = rnd_state.uniform(size=(self.tower.no_sims, self.tower.no_time))  # 10, 1045
        self.tower.wind['ratio'] = 1.082  # 1.05*np.exp(0.03)
        self.tower._damage_prob = None
        self.tower._damage_prob_mc = None
        self.assertAlmostEqual(self.tower.damage_prob['collapse'].values[0],
                               0.842, places=2)
        self.assertAlmostEqual(self.tower.damage_prob['minor'].values[0],
                               0.998, places=2)

        a = np.array([rv < self.tower.damage_prob[ds].values
                      for ds in self.tower.damage_states]).sum(axis=0)

        b = np.array([rv[:, :, np.newaxis] <
                      self.tower.damage_prob.values])[0].sum(axis=2)

        np.testing.assert_array_equal(a, b)

        # comparing with analytical output
        for ids, ds in enumerate(self.tower.damage_states, 1):
            prob_mc = (a >= ids).sum(axis=0) / self.tower.no_sims
            isclose = np.isclose(prob_mc[0], self.tower.damage_prob[ds].values[0],
                                 rtol=RTOL, atol=ATOL)
            if not isclose:
                self.logger.warning('PE of {}: {:.3f} vs {:.3f}'.format(
                    ds, prob_mc[0], self.tower.damage_prob[ds].values[0]))

    def test_compare_damage_prob_vs_mc(self):

        # damage_prob vs. damage_prob_mc
        prob_mc = 0
        for ds in self.tower.damage_states[::-1]:
            no = len(self.tower.damage_prob_mc[ds]['id_sim'][
                 self.tower.damage_prob_mc[ds]['id_time'] == 1])
            prob_mc += no / self.tower.no_sims
            isclose = np.isclose(prob_mc,
                                 self.tower.damage_prob[ds].values[1],
                                 rtol=RTOL, atol=ATOL)
            if not isclose:
                self.logger.warning('PE of {}: {:.3f} vs {:.3f}'.format(
                    ds, prob_mc, self.tower.damage_prob[ds].values[1]))

    def test_collapse_prob_adj_mc(self):

        # tower14 (lid: 13,
        tower_dic = self.cfg.towers_by_line['LineA'][0].copy()
        path_event = os.path.join(self.cfg.path_wind_scenario_base, 'test1')
        tower_dic.update({'path_event': path_event,
                          'no_sims': 35000,
                          'damage_states': self.cfg.damage_states,
                          'scale': 1.0,
                          'rnd_state': np.random.RandomState(1),
                          'event_name': 'test1'})

        tower = Tower(tower_id=0, **tower_dic)

        tower.wind['ratio'] = 1.082  # 1.05*np.exp(0.03)
        tower._damage_prob = None
        tower._damage_prob_mc = None
        tower._collapse_prob_adj = None
        tower._collapse_prob_adj_mc = None

        msg = 'P({}|{}) at {}: analytical {:.3f} vs. simulation {:.3f}'

        # comparing with analytical output
        id_adj_removed = [x for x in tower.id_adj if x >=0]
        id_adj_removed.remove(tower.lid)

        for id_time, grouped in tower.collapse_prob_adj_mc.groupby('id_time'):

            for lid in id_adj_removed:

                prob = grouped['id_adj'].apply(lambda x: lid in x).sum() / tower.no_sims
                try:
                    assert np.isclose(tower.collapse_prob_adj[lid][id_time], prob,
                                      atol=ATOL, rtol=RTOL)
                except AssertionError:
                    self.logger.warning(msg.format(
                        lid,
                        tower.name,
                        id_time,
                        tower.collapse_prob_adj[lid][id_time],
                        prob
                        ))

    """
    def test_determine_damage_state_by_simulation(self):

        seed = 11
        no_sims = 10000
        rnd_state = np.random.RandomState(seed)

        tower = copy.deepcopy(self.tower)
        wp = pd.DataFrame(np.ones(shape=(5, 2)),
                          index=pd.date_range('1/1/2000', periods=5),
                          columns=['minor', 'collapse'])
        wp['minor'] = [0.1, 0.3, 0.5, 0.7, 0.9]
        wp['collapse'] = [0.05, 0.2, 0.4, 0.6, 0.75]

        tower.prob_damage_isolation = wp

        rv = rnd_state.uniform(size=(no_sims, len(wp.index)))

        tf_array = np.array([rv < wp[ds].values for ds in
                             self.cfg.damage_states])
        ds_wind = np.sum(tf_array, axis=0)

        for i, (_, value) in enumerate(wp.iterrows()):
            for j, ds in enumerate(self.cfg.damage_states, 1):
                sim_result = np.sum(ds_wind[:, i] >= j) / float(no_sims)
                # print('{}:{}'.format(sim_result, value[ds]))
                np.testing.assert_almost_equal(sim_result, value[ds],
                                               decimal=1)
        tower.determine_damage_isolation_mc(rv)

        wind_mc = {ds: dict() for ds in self.cfg.damage_states}
        for i, ds in enumerate(self.cfg.damage_states, 1):
            wind_mc[ds]['id_sim'], wind_mc[ds]['id_time'] = \
                np.where(ds_wind == i)
            np.testing.assert_almost_equal(wind_mc[ds]['id_sim'],
                                          tower.damage_isolation_mc[ds]['id_sim'])
            np.testing.assert_almost_equal(wind_mc[ds]['id_time'],
                                          tower.damage_isolation_mc[ds]['id_time'])
            # assertDeepAlmostEqual(self, tower.damage_mc[ds],
            #                      wind_mc[ds])

        no_collapse_ignoring_time = np.sum(ds_wind == 2)

        rnd_state = np.random.RandomState(seed + 100)

        rv2 = rnd_state.uniform(size=no_collapse_ignoring_time)

        print('{}:{}'.format(tower.cond_pc_adj_mc['rel_idx'],
                             tower.cond_pc_adj_mc['cum_prob']))
        expected_abs_list = (0, 2)

        expected = [expected_abs_list if x <= tower.cond_pc_adj_mc['cum_prob']
                    else None for x in rv2]

        ps_ = pd.Series(expected,
                        index=tower.damage_isolation_mc['collapse'].index,
                        name='id_adj')

        adj_mc = pd.concat([tower.damage_isolation_mc['collapse'], ps_],
                           axis=1)
        adj_mc = adj_mc[pd.notnull(ps_)]

        #print('{}'.format(pd.unique(tower.damage_isolation_mc['collapse']['id_sim'])))

        damage_adjacent_mc = dict()
        for (id_time_, id_adj_), grouped in adj_mc.groupby(['id_time', 'id_adj']):
            damage_adjacent_mc.setdefault(id_time_, {})[id_adj_] = \
                grouped.id_sim.tolist()

        #print('{}'.format(damage_adjacent_mc[0][expected_abs_list]))

        tower.compute_mc_adj(rv, seed)

        # print('{}'.format(tower.damage_mc['collapse']['abs_idx']))
        # self.assertEqual(damage_adjacent_mc, tower.damage_mc['collapse']['id_adj'].tolist())
        assertDeepAlmostEqual(self, tower.damage_isolation_mc, wind_mc)
        # assertDeepAlmostEqual(self, tower.damage_adjacent_mc, damage_adjacent_mc)


        for i in range(5):
            print('{}:{}'.format(
                len(tower.damage_adjacent_mc[i][(0, 2)]),
                len(damage_adjacent_mc[i][(0, 2)])))

        print('{}'.format(len(wind_mc['collapse']['id_time'])))
"""

if __name__ == '__main__':
    unittest.main(verbosity=2)
