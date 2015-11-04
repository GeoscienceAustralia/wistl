import numpy as np
from scipy.stats import lognorm
import pandas as pd


class Event(object):

    """
    class Event
    """
    def __init__(self, fid):

        self.fid = fid
        # to be assigned
        self.wind = None  # pd.DataFrame
        self.pc_wind = None  # pd.DataFrame <- cal_pc_wind
        self.pc_adj = None  # dict (ntime,) <- cal_pc_adj_towers
        self.mc_wind = None  # dict(nsims, ntime)
        self.mc_adj = None  # dict

    # originally a part of Tower class but moved wind to pandas timeseries
    def cal_pc_wind(self, asset, frag, ntime, ds_list, nds):
        """
        compute probability of damage due to wind
        - asset: instance of Tower object
        - frag: dictionary by asset.const_type
        - ntime:  
        - ds_list: [('collapse', 2), ('minor', 1)]
        - nds:
        """

        pc_wind = np.zeros((ntime, nds))

        vratio = self.wind.dir_speed.values/asset.adj_design_speed

        self.vratio = vratio

        try:
            fragx = frag[asset.ttype][asset.funct]
            idf = np.sum(fragx['dev_angle'] <= asset.dev_angle)

            for (ds, ids) in ds_list: # damage state
                med = fragx[idf][ds]['param0']
                sig = fragx[idf][ds]['param1']

                temp = lognorm.cdf(vratio, sig, scale=med)
                pc_wind[:,ids-1] = temp # 2->1

        except KeyError:        
                print "fragility is not defined for %s" %asset.const_type

        self.pc_wind = pd.DataFrame(pc_wind, columns = [x[0] for x in ds_list], 
           index = self.wind.index)
                
        return

    def cal_pc_adj(self, asset, cond_pc): # only for analytical approach 
        """
        calculate collapse probability of jth tower due to pull by the tower
        Pc(j,i) = P(j|i)*Pc(i)
        """
        # only applicable for tower collapse

        pc_adj = {}
        for rel_idx in asset.cond_pc_adj.keys():
            abs_idx = asset.adj_list[rel_idx + asset.max_no_adj_towers]
            pc_adj[abs_idx] = (self.pc_wind.collapse.values * 
                                      asset.cond_pc_adj[rel_idx])

        self.pc_adj = pc_adj

        return

    def cal_mc_adj(self, asset, nsims, ntime, ds_list, nds, rv, idx):
        """
        2. determine if adjacent tower collapses or not due to pull by the tower
        jtime: time index (array)
        idx: multiprocessing thread id
        """

        # if rv is None:  # perfect correlation
        #     prng = np.random.RandomState()
        #     rv = prng.uniform(size=(nsims, ntime))

        # 1. determine damage state of tower due to wind
        val = np.array([rv < self.pc_wind[ds[0]].values for ds in ds_list]) # (nds, nsims, ntime)

        ds_wind = np.sum(val, axis=0) # (nsims, ntime) 0(non), 1, 2 (collapse)

        #tf = event.pc_wind.collapse.values > rv # true means collapse
        mc_wind = {}
        for (ds, ids) in ds_list:
            (mc_wind.setdefault(ds,{})['isim'], 
             mc_wind.setdefault(ds,{})['itime']) = np.where(ds_wind == ids)

        #if unq_itime == None:

        # for collapse
        unq_itime = np.unique(mc_wind['collapse']['itime'])

        nprob = len(asset.cond_pc_adj_mc['cum_prob']) # 

        mc_adj = {}  # impact on adjacent towers

        # simulaiton where none of adjacent tower collapses    
        #if max_idx == 0:

        if nprob > 0:

            for jtime in unq_itime:

                jdx = np.where(mc_wind['collapse']['itime'] == jtime)[0]
                idx_sim = mc_wind['collapse']['isim'][jdx] # index of simulation
                nsims = len(idx_sim)
                if idx:
                    prng = np.random.RandomState(idx)
                else:
                    prng = np.random.RandomState()
                rv = prng.uniform(size=(nsims))

                list_idx_cond = []
                for rv_ in rv:
                    idx_cond = sum(rv_ >= asset.cond_pc_adj_mc['cum_prob'])
                    list_idx_cond.append(idx_cond)

                # ignore simulation where none of adjacent tower collapses    
                unq_list_idx_cond = set(list_idx_cond) - set([nprob])

                for idx_cond in unq_list_idx_cond:

                    # list of idx of adjacent towers in collapse
                    rel_idx = asset.cond_pc_adj_mc['rel_idx'][idx_cond]

                    # convert relative to absolute fid
                    abs_idx = [asset.adj_list[j + asset.max_no_adj_towers] for 
                               j in rel_idx]

                    # filter simulation          
                    isim = [i for i, x in enumerate(list_idx_cond) if x == idx_cond]
                    mc_adj.setdefault(jtime, {})[tuple(abs_idx)] = idx_sim[isim]

        self.mc_wind = mc_wind
        self.mc_adj = mc_adj
        return
