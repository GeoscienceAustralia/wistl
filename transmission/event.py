import numpy as np
from scipy.stats import lognorm
import pandas as pd
import matplotlib.pyplot as plt


def dir_wind_speed(speed, bearing, t0):

    # angle between wind direction and tower conductor
    phi = np.abs(bearing - t0)

    tf = (phi <= np.pi/4) | (phi > np.pi/4*7) | ((phi > np.pi/4*3) & (phi <= np.pi/4*5))

    cos_ = abs(np.cos(np.pi/4.0-phi))
    sin_ = abs(np.sin(np.pi/4.0-phi))

    adj = speed*np.max(np.vstack((cos_, sin_)), axis=0)

    dir_speed = np.where(tf, adj, speed)  # adj if true, otherwise speed

    return dir_speed


class Event(object):

    """
    class Event
    Inputs:
    tower: instance of tower class
    vel_file: velocity file containing velocity time history at this tower location.
    """
    def __init__(self, tower, vel_file):
        self.tower = tower
        self.vel_file = vel_file
        self.vratio = self.wind.dir_speed.values/tower.adj_design_speed
        self.pc_adj = None  # dict (ntime,) <- cal_pc_adj_towers
        self.mc_wind = None  # dict(nsims, ntime)
        self.mc_adj = None  # dict
        self.idx_time = self.wind.index
        # self.frag, self.ds_list, self.nds = self.read_frag()

    @property
    def wind(self):
        # Time,Longitude,Latitude,Speed,UU,VV,Bearing,Pressure
        data = pd.read_csv(self.vel_file, header=0, parse_dates=[0], index_col=[0],
            usecols=[0, 3, 6], names=['', '', '', 'speed', '', '', 'bearing', ''])

        speed = data['speed'].values
        bearing = np.deg2rad(data['bearing'].values)  # degree

        # angle of conductor relative to NS
        t0 = np.deg2rad(self.tower.strong_axis) - np.pi/2.0

        convert_factor = self.convert_10_to_z()
        dir_speed = convert_factor * dir_wind_speed(speed, bearing, t0)

        data['dir_speed'] = pd.Series(dir_speed, index=data.index)
        return data

    def convert_10_to_z(self):
        """
        Mz,cat(h=10)/Mz,cat(h=z)
        tc: terrain category (defined by line route)
        asset is a Tower class instance.
        """
        terrain_height = self.tower.terrain_height
        tc_str = 'tc' + str(self.tower.terrain_cat)  # Terrain

        try:
            mzcat_z = np.interp(self.tower.height_z, terrain_height['height'], terrain_height[tc_str])
        except KeyError:
            print "%s is not defined" %tc_str
            return {'error': "{} is not defined".format(tc_str)}  # these errors should be handled properly

        mzcat_10 = terrain_height[tc_str][terrain_height['height'].index(10)]
        return mzcat_z/mzcat_10

    @property
    def pc_wind(self):
        """
        compute probability of damage due to wind
        - asset: instance of Tower object
        - frag: dictionary by asset.const_type
        - ntime:  
        - ds_list: [('collapse', 2), ('minor', 1)]
        - nds:
        """
        frag, ds_list, nds = self.read_frag()
        pc_wind = np.zeros(shape=(len(self.idx_time), nds))

        try:
            fragx = frag[self.tower.ttype][self.tower.funct]
            idf = np.sum(fragx['dev_angle'] <= self.tower.dev_angle)

            for (ds, ids) in ds_list:  # damage state
                med = fragx[idf][ds]['param0']
                sig = fragx[idf][ds]['param1']

                temp = lognorm.cdf(self.vratio, sig, scale=med)
                pc_wind[:, ids-1] = temp  # 2->1

        except KeyError:        
                print "fragility is not defined for %s" % self.tower.const_type

        return pd.DataFrame(pc_wind, columns=[x[0] for x in ds_list], index=self.wind.index)

    def cal_pc_adj(self, asset, cond_pc):  # only for analytical approach
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

    @property
    def cond_pc(self):
        """read condition collapse probability defined by tower function

        >>> txt = '''FunctionType, # of collapse, probability, start, end
        ... suspension, 1, 0.075, 0, 1
        ... suspension, 1, 0.075, -1, 0
        ... suspension, 2, 0.35, -1, 1
        ... suspension, 3, 0.025, -1, 2
        ... suspension, 3, 0.025, -2, 1
        ... suspension, 4, 0.10, -2, 2
        ... strainer, 1, 0.075, 0, 1
        ... strainer, 1, 0.075, -1, 0
        ... strainer, 2, 0.35, -1, 1
        ... strainer, 3, 0.025, -1, 2
        ... strainer, 3, 0.025, -2, 1
        ... strainer, 4, 0.10, -2, 2'''
        ... strainer, 5, 0.10, -2, 2'''
        ... strainer, 5, 0.10, -2, 2'''
        ... strainer, 5, 0.10, -2, 2'''
        >>> cond_pc = read_cond_prob(StringIO(txt))
        >>> cond_pc'''
        {'strainer': {'max_adj': 2,
          (-2, -1, 0, 1): 0.025,
          (-2, -1, 0, 1, 2): 0.1,
          (-1, 0): 0.075,
          (-1, 0, 1): 0.35,
          (-1, 0, 1, 2): 0.025,
          (0, 1): 0.075},
         'suspension': {'max_adj': 2,
          (-2, -1, 0, 1): 0.025,
          (-2, -1, 0, 1, 2): 0.1,
          (-1, 0): 0.075,
          (-1, 0, 1): 0.35,
          (-1, 0, 1, 2): 0.025,
          (0, 1): 0.075}}
        """

        data = pd.read_csv(self.tower.conf.file_frag, skipinitialspace=1)
        cond_pc = {}
        for line in data.iterrows():
            func, cls_str, thr, pb, n0, n1 = [line[1][x] for x in
                                    ['FunctionType', 'class', 'threshold', 'probability', 'start', 'end']]
            list_ = range(int(n0), int(n1)+1)
            cond_pc.setdefault(func, {})['threshold'] = thr
            cond_pc[func].setdefault(cls_str, {}).setdefault('prob', {})[tuple(list_)] = float(pb)

        for func in cond_pc.keys():
            cls_str = cond_pc[func].keys()
            cls_str.remove('threshold')
            for cls in cls_str:
                max_no_adj_towers = np.max(np.abs([j for k in cond_pc[func][cls]['prob'].keys()
                                for j in k]))
                cond_pc[func][cls]['max_adj'] = max_no_adj_towers

        return cond_pc

    def read_frag(self, flag_plot=None):
        """read collapse fragility parameter values
        >>> txt = '''ConstType, damage, index, function, param0, param1
        ... Unknown, minor, 1, lognorm, 0.85, 0.05'''
        ... Unknown, collpase, 2, lognorm, 1.02, 0.05'''
        >>> data = read_collapse_frag(StringIO(txt))
        >>> data'''
        {'Unknown': {'collapse': {'function': 'lognorm',
           'param0': 1.02,
           'param1': 0.05},
          'minor': {'function': 'lognorm', 'param0': 0.85, 'param1': 0.05}}}
        """

        data = pd.read_csv(self.tower.conf.file_frag, skipinitialspace=1)

        frag = {}
        for ttype in data['tower type'].unique():
            for func in data['function'].unique():
                idx = (data['tower type'] == ttype) & (data['function'] == func)
                dev0_ = data['dev0'].ix[idx].unique()
                dev1_ = data['dev1'].ix[idx].unique()
                dev_ = np.sort(np.union1d(dev0_, dev1_))

                frag.setdefault(ttype, {}).setdefault(func, {})['dev_angle'] = dev_

                for j, val in enumerate(dev0_):

                    idx2 = np.where(idx & (data['dev0'] == val))[0]

                    for k in idx2:
                        ds_ = data.ix[k]['damage']
                        idx_ = data.ix[k]['index']
                        cdf_ = data.ix[k]['cdf']
                        param0_ = data.ix[k]['param0']
                        param1_ = data.ix[k]['param1']

                        # dev_angle (0, 5, 15, 30, 360) <= tower_angle 0.0 => index
                        # angle is less than 360. if 360 then 0.
                        frag[ttype][func].setdefault(j+1, {}).setdefault(ds_, {})['idx'] = idx_
                        frag[ttype][func][j+1][ds_]['param0'] = param0_
                        frag[ttype][func][j+1][ds_]['param1'] = param1_
                        frag[ttype][func][j+1][ds_]['cdf'] = cdf_

        ds_list = [(x, frag[ttype][func][1][x]['idx']) for x in frag[ttype][func][1].keys()]
        ds_list.sort(key=lambda tup: tup[1])  # sort by ids

        nds = len(ds_list)

        if flag_plot:
            x = np.arange(0.5, 1.5, 0.01)
            line_style = {'minor': '--', 'collapse': '-'}

            for ttype in frag.keys():
                for func in frag[ttype].keys():
                    plt.figure()

                    for idx in frag[ttype][func].keys():
                        try:
                            for ds in frag[ttype][func][idx].keys():
                                med = frag[ttype][func][idx][ds]['param0']
                                sig = frag[ttype][func][idx][ds]['param1']
                                y = lognorm.cdf(x, sig, scale=med)
                                plt.plot(x,y, line_style[ds])
                        except AttributeError:
                            print "no"

                    plt.legend(['collapse', 'minor'], 2)
                    plt.xlabel('Ratio of wind speed to adjusted design wind speed')
                    plt.ylabel('Probability of exceedance')
                    plt.title(ttype+':'+func)
                    plt.yticks(np.arange(0, 1.1, 0.1))
                    plt.grid(1)
                    plt.savefig(ttype + '_' + func + '.png')

        return frag, ds_list, nds


if __name__ == '__main__':
    from config_class import TransmissionConfig
    conf = TransmissionConfig()
    from read import TransmissionNetwork
    network = TransmissionNetwork(conf)
    tower, sel_lines, fid_by_line, fid2name, lon, lat = network.read_tower_gis_information(conf)

