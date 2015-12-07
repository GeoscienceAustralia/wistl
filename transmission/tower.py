import numpy as np
import pandas as pd
import itertools


class Tower(object):

    """
    class Tower
    Tower class represent an individual transmission tower.
    """
    fid_gen = itertools.count()

    def __init__(self, conf, df_tower, adj=None):
        self.conf = conf
        self.df_tower = df_tower
        self.fid = next(self.fid_gen)

        self.ttype = self.df_tower['Type']  # Lattice Tower or Steel Pole
        self.funct = self.df_tower['Function']  # e.g., suspension, terminal, strainer
        self.line_route = self.df_tower['LineRoute']  # string
        self.no_circuit = 2  # double circuit (default value)
        self.design_speed = self.determine_design_speed()  # design wind speed
        self.design_span = self.conf.design_value[self.line_route]['span']  # design wind span
        self.terrain_cat = self.conf.design_value[self.line_route]['cat']  # Terrain Cateogry
        self.design_level = self.conf.design_value[self.line_route]['level']  # design level
        self.strong_axis = self.df_tower['AxisAz']  # azimuth of strong axis relative to North (deg)
        self.dev_angle = self.df_tower['DevAngle']  # deviation angle
        self.height = float(self.df_tower['Height'])
        self.height_z = conf.drag_height[self.funct]

        # to be assigned
        self.actual_span = None  # actual wind span on eith side
        self.adj = adj  # (left, right)
        self.adj_list = None  # (23,24,0,25,26) ~ idfy_adj_list (function specific)
        self.adj_design_speed = None
        self.max_no_adj_towers = None  #
        self.cond_pc_adj = None  # dict ~ cal_cond_pc_adj
        self.cond_pc_adj_mc = {'rel_idx': None, 'cum_prob': None}  # ~ cal_cond_pc_adj

    # def height_z_based_on_tower_funct(self):
    #     # typical drag height by tower type
    #     # drag height (FIXME: typical value by type, but vary across towers)
    #     height_z_dic = {'Suspension': 15.4, 'Strainer': 12.2, 'Terminal': 12.2}
    #     return height_z_dic[self.funct]

    def determine_design_speed(self):

        design_speed = self.conf.design_value[self.line_route]['speed']

        if self.conf.flag_adjust_design_by_topo:
            ix_topo = np.sum(self.conf.topo_multiplier[name_] >=
                self.conf.design_adjustment_factor_by_topo['threshold'])
            design_speed *= self.conf.design_adjustment_factor_by_topo[ix_topo]

        return design_speed


    def calc_adj_collapse_wind_speed(self):
        """
        calculate adjusted collapse wind speed for a tower
        Vc = Vd(h=z)/sqrt(u)
        where u = 1-k(1-Sw/Sd)
        Sw: actual wind span
        Sd: design wind span (defined by line route)
        k: 0.33 for a single, 0.5 for double circuit
        """

        # k: 0.33 for a single, 0.5 for double circuit
        k_factor = {1: 0.33, 2: 0.5}

        # calculate utilization factor
        try:
            u = min(1.0, 1.0 - k_factor[self.no_circuit] *
                (1.0 - self.actual_span / self.design_span))  # 1 in case sw/sd > 1
        except KeyError:
            return {'error': "no. of circuit {} is not valid: {}"
                    .format(self.fid, self.no_circuit)}
        self.u_val = 1.0/np.sqrt(u)
        vc = self.design_speed/np.sqrt(u)

        self.adj_design_speed = vc
        return

    def idfy_adj_list(self, tower, fid2name, cond_pc, flag_strainer=None):
        """
        identify list of adjacent towers which can influence on collapse
        """

        def create_list_idx(idx, nsteps, flag):
            """
                create list of adjacent towers in each direction (flag=+/-1)
            """

            list_idx = []
            for i in range(nsteps):
                try:
                    idx = tower[fid2name[idx]].adj[flag]
                except KeyError:
                    idx = -1
                list_idx.append(idx)
            return list_idx

        def mod_list_idx(list_):
            """
            replace id of strain tower with -1
            """
            for i, item in enumerate(list_):

                if item != -1:
                    tf = False
                    try:
                        tf = tower[fid2name[item]].funct in flag_strainer
                    except KeyError:
                        print "KeyError {}".format(fid2name[item])

                    if tf is True:
                        list_[i] = -1
            return list_

        if self.funct == 'Strainer':
            self.max_no_adj_towers =\
                cond_pc['Strainer'][self.design_level]['max_adj']

        else:  # Suspension or Terminal
            thr = float(cond_pc['Suspension']['threshold'])
            if self.height > thr:
                self.max_no_adj_towers =\
                    cond_pc['Suspension']['higher']['max_adj']
            else:
                self.max_no_adj_towers =\
                    cond_pc['Suspension']['lower']['max_adj']

        list_left = create_list_idx(self.fid, self.max_no_adj_towers, 0)
        list_right = create_list_idx(self.fid, self.max_no_adj_towers, 1)

        if flag_strainer is None:
            self.adj_list = list_left[::-1] + [self.fid] + list_right
        else:
            self.adj_list = (mod_list_idx(list_left)[::-1] + [self.fid] +
                             mod_list_idx(list_right))

        return

    def cal_cond_pc_adj(self, cond_pc, fid2name):
        """
        calculate conditional collapse probability of jth tower given ith tower
        P(j|i)
        """

        if self.funct == 'Strainer':
            cond_pc_ = cond_pc['Strainer'][self.design_level]['prob']

        else: # Suspension or Terminal
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
