#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import numpy as np
#import pandas as pd
#import itertools


class Tower(object):

    """
    class Tower
    Tower class represent an individual transmission tower.
    """
    #id_gen = itertools.count()

    def __init__(self, conf, df_tower):

        self.conf = conf
        self.df_tower = df_tower

        self.id = df_tower['id']
        self.name = df_tower['Name']
        self.ttype = df_tower['Type']  # Lattice Tower or Steel Pole
        self.funct = df_tower['Function']  # e.g., suspension, terminal, strainer
        self.line_route = df_tower['LineRoute']  # string
        self.no_circuit = 2  # double circuit (default value)
        self.design_speed = self.assign_design_speed()  # design wind speed
        self.design_span = self.conf.design_value[self.line_route]['span']  # design wind span
        self.terrain_cat = self.conf.design_value[self.line_route]['cat']  # Terrain Cateogry
        self.design_level = self.conf.design_value[self.line_route]['level']  # design level
        self.strong_axis = df_tower['AxisAz']  # azimuth of strong axis relative to North (deg)
        self.dev_angle = df_tower['DevAngle']  # deviation angle
        self.height = float(df_tower['Height'])
        self.height_z = conf.drag_height[self.funct]
        self.actual_span = df_tower['actual_span']  # actual wind span on eith side
        self.collapse_capacity = self.compute_collapse_capacity()
        self.max_no_adj_towers = self.assign_max_no_adj_towers()

        # assigned
        self.id_sides = None  # (left, right)
        self.id_adj = None  # (23,24,0,25,26) ~ idfy_adj_list (function specific)
        self.cond_pc_adj = None  # dict ~ cal_cond_pc_adj
        self.cond_pc_adj_mc = {'rel_idx': None, 'cum_prob': None}  # ~ cal_cond_pc_adj

    def assign_max_no_adj_towers(self):
        ''' determine max no. adjacent tower based on function and design level
        '''
        if self.funct == 'Strainer':
            funct_ = self.funct
            design_level_ = self.design_level

        else:  # Suspension or Terminal
            funct_ = 'Suspension'
            thr = float(self.conf.cond_collapse_prob[funct_]['threshold'])
            if self.height > thr:
                design_level_ = 'higher'
            else:
                design_level_ = 'lower'

        return self.conf.cond_collapse_prob[funct_][design_level_]['max_adj']

    def assign_design_speed(self):
        """ determine design speed """
        design_speed = self.conf.design_value[self.line_route]['speed']
        if self.conf.flag_adjust_design_by_topo:
            id_topo = np.sum(self.conf.topo_multiplier[self.name] >=
                             self.conf.design_adjustment_factor_by_topo['threshold'])
            design_speed *= self.conf.design_adjustment_factor_by_topo[id_topo]

        return design_speed

    def compute_collapse_capacity(self):
        """
        calculate adjusted collapse wind speed for a tower
        Vc = Vd(h=z)/sqrt(u)
        where u = 1-k(1-Sw/Sd)
        Sw: actual wind span
        Sd: design wind span (defined by line route)
        k: 0.33 for a single, 0.5 for double circuit
        """

        # k: 0.33 for a single, 0.5 for double circuit
        k_factor = {1: 0.33, 2: 0.5}  # hard-coded

        # calculate utilization factor
        u = min(1.0, 1.0 - k_factor[self.no_circuit] *
                (1.0 - self.actual_span / self.design_span))  # 1 in case sw/sd > 1
        #self.u_val = 1.0/np.sqrt(u)
        return self.design_speed/np.sqrt(u)

    def cal_cond_pc_adj(self, cond_pc, id2name):
        """
        calculate conditional collapse probability of jth tower given ith tower
        P(j|i)
        """

        if self.funct == 'Strainer':
            cond_pc_ = cond_pc['Strainer'][self.design_level]['prob']

        else:  # Suspension or Terminal
            thr = float(cond_pc['Suspension']['threshold'])
            if self.height > thr:
                cond_pc_ = cond_pc['Suspension']['higher']['prob']
            else:
                cond_pc_ = cond_pc['Suspension']['lower']['prob']

        idx_m1 = np.array([i for i in range(len(self.adj_list))
            if self.adj_list[i] == -1]) - self.max_no_adj_towers # rel_index

        try:
            max_neg = np.max(idx_m1[idx_m1<0]) + 1
        except ValueError:
            max_neg = - self.max_no_adj_towers

        try:
            min_pos = np.min(idx_m1[idx_m1>0])
        except ValueError:
            min_pos = self.max_no_adj_towers + 1

        bound_ = set(range(max_neg, min_pos))

        cond_prob = {}
        for item in cond_pc_.keys():
            w = list(set(item).intersection(bound_))
            w.sort()
            w = tuple(w)
            if cond_prob.has_key(w):
                cond_prob[w] += cond_pc_[item]
            else:
                cond_prob[w] = cond_pc_[item]

        if cond_prob.has_key((0,)):
            cond_prob.pop((0,))

        # sort by cond. prob
        rel_idx = sorted(cond_prob, key=cond_prob.get)
        prob = map(lambda v: cond_prob[v], rel_idx)

        cum_prob = np.cumsum(np.array(prob))

        self.cond_pc_adj_mc['rel_idx'] = rel_idx
        self.cond_pc_adj_mc['cum_prob'] = cum_prob

        # sum by node
        cond_pc_adj = dict()
        for key_ in cond_prob:
            for i in key_:
                try:
                    cond_pc_adj[i] += cond_prob[key_]
                except KeyError:
                    cond_pc_adj[i] = cond_prob[key_]

        if 0 in cond_pc_adj:
            cond_pc_adj.pop(0)

        self.cond_pc_adj = cond_pc_adj

        return
