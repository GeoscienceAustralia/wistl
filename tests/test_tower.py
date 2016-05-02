#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import unittest
import StringIO
import pandas as pd
import os
import numpy as np
import copy
import matplotlib
matplotlib
import matplotlib.pyplot as plt

from wistl.config import TransmissionConfig
from wistl.transmission_network import TransmissionNetwork
from wistl.tower import Tower
from test_config import assertDeepAlmostEqual

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class TestTower(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.cfg = TransmissionConfig(os.path.join(BASE_DIR, 'test.cfg'))
        cls.network = TransmissionNetwork(cls.cfg)
        cls.network.event_tuple = ('test2', 2.5)

        cls.tower = cls.network.lines['Calaca - Amadeo'].towers['AC-100']
        cls.ps_tower = cls.tower.ps_tower
        cls.tower.event_tuple = (cls.tower.file_wind, 3.0)

        # set wind file, which also sets wind and time_index
        # cls.tower.file_wind = file_wind

        # compute prob_damage_isolation and prob_damage_adjacent
        # cls.tower.compute_damage_prob_isolation()
        #cls.tower.compute_pc_adj()

    def test_get_cond_collapse_prob(self):

        list_function = ['Suspension'] * 2 + ['Terminal'] * 2
        list_value = [20.0, 50.0] * 2

        for func_, value_ in zip(list_function, list_value):

            ps_tower = copy.deepcopy(self.ps_tower)
            cfg = copy.deepcopy(self.cfg)

            ps_tower['Function'] = func_
            ps_tower['Height'] = value_
            tower = Tower(cfg, ps_tower)

            if tower.cond_pc:
                tmp = cfg.cond_collapse_prob[func_]
                expected = tmp.set_index('list').to_dict()['probability']
                assertDeepAlmostEqual(self, tower.cond_pc, expected)

        list_function = ['Strainer'] * 2
        list_value = ['low', 'high']

        for func_, value_ in zip(list_function, list_value):

            ps_tower = copy.deepcopy(self.ps_tower)
            ps_tower['Function'] = func_
            cfg = copy.deepcopy(self.cfg)

            cfg.design_value[self.ps_tower['LineRoute']]['level'] = value_
            tower = Tower(cfg, ps_tower)

            if tower.cond_pc:
                tmp = cfg.cond_collapse_prob[func_]
                expected = tmp.loc[tmp['design_level'] == value_, :].set_index('list').to_dict()['probability']
                assertDeepAlmostEqual(self, tower.cond_pc, expected)

    def test_assign_design_speed(self):

        expected = 75.0 * 1.2
        cfg = copy.deepcopy(self.cfg)
        cfg.adjust_design_by_topo = True
        cfg.topo_multiplier[self.ps_tower['Name']] = 1.15
        tower = Tower(cfg, self.ps_tower)
        self.assertEqual(tower.design_speed, expected)

        expected = 75.0
        cfg.adjust_design_by_topo = False
        tower = Tower(self.cfg, self.ps_tower)
        self.assertEqual(tower.design_speed, expected)

    def test_compute_collapse_capacity(self):

        ps_tower = copy.deepcopy(self.ps_tower)
        cfg = copy.deepcopy(self.cfg)

        cfg.design_value[ps_tower['LineRoute']]['span'] = 20.0
        ps_tower['actual_span'] = 10.0
        tower = Tower(cfg, ps_tower)

        self.assertEqual(tower.collapse_capacity,
                         tower.design_speed/np.sqrt(0.75))

        # design wind span < actual span
        cfg.design_value[ps_tower['LineRoute']]['span'] = 20.0
        ps_tower['actual_span'] = 25.0
        tower = Tower(cfg, ps_tower)

        self.assertEqual(tower.collapse_capacity,
                         tower.design_speed/1.0)

    def test_convert_10_to_z(self):

        # terrain category
        # ASNZS 1170.2:2011
        category = [1, 2, 3, 4]
        mzcat10 = [1.12, 1.0, 0.83, 0.75]

        height_z = 15.4   # Suspension
        for cat_, mzcat10_ in zip(category, mzcat10):
            cfg = copy.deepcopy(self.cfg)
            cfg.design_value[self.ps_tower['LineRoute']]['cat'] = cat_
            ps_tower = copy.deepcopy(self.ps_tower)
            ps_tower.Function = 'Suspension'
            expected = np.interp(height_z, cfg.terrain_multiplier['height'],
                                 cfg.terrain_multiplier['tc'+ str(cat_)])/mzcat10_
            tower = Tower(cfg, ps_tower)
            self.assertEqual(tower.convert_factor, expected)

        height_z = 12.2  # Strainer, Terminal
        for cat_, mzcat10_ in zip(category, mzcat10):
            cfg = copy.deepcopy(self.cfg)
            # ps_tower = copy.deepcopy(self.ps_tower)
            # ps_tower['Function'] = 'Strainer'
            cfg.design_value[self.ps_tower['LineRoute']]['cat'] = cat_
            expected = np.interp(height_z, cfg.terrain_multiplier['height'],
                                 cfg.terrain_multiplier['tc'+str(cat_)])/mzcat10_
            tower = Tower(cfg, self.ps_tower)
            self.assertEqual(tower.convert_factor, expected)

    '''
    def test_calculate_cond_pc_adj(self):
        """included in test_transmission_line.py"""
        pass
    '''

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

if __name__ == '__main__':
    unittest.main()
