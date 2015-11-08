"""
simulation of collapse of transmission towers 
based on empricially derived conditional probability


required input
    -. tower geometry whether rectangle or squre (v) - all of them are square
    -. design wind speed (requiring new field in input) v
    -. tower type and associated collapse fragility v
    -. conditional probability by tower type (suspension and strainer) v
    -. idenfity strainer tower v

Todo:
    -. creating module (v)
    -. visualisation (arcGIS?) (v)
    -. postprocessing of mc results (v)
    -. think about how to sample random numbers (spatial correlation) (v)
    -. adding additional damage state (v)
    -. adj_list by function type (v)
    -. update cond adj each simulation/time
    -. assigning cond (different)
    -. with our without cascading effect (mc simulation) - priority
"""

'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from mpl_toolkits.basemap import Basemap
from scipy import stats
import shapefile
import csv
import time
import cPickle as pickle
import simplekml
import pandas as pd
from scipy.optimize import minimize_scalar
'''
import numpy as np
import parmap

#from compute import compute_
#import read
from read import (read_frag, read_cond_prob, read_tower_GIS_information, 
                 read_velocity_profile)

#import compute
from compute import (cal_collapse_of_towers_analytical,
    cal_collapse_of_towers_mc, cal_exp_std, cal_exp_std_no_cascading)

from tower import Tower
from event import Event

###############################################################################
# main procedure
###############################################################################


def sim_towers(conf):
    print "==============>conf.test:", conf.test
    shape_file_tower = conf.shape_file_tower
    shape_file_line = conf.shape_file_line
    dir_wind_timeseries = conf.dir_wind_timeseries
    file_frag = conf.file_frag
    file_cond_pc = conf.file_cond_pc
    file_design_value = conf.file_design_value
    file_terrain_height = conf.file_terrain_height
    file_topo_value= conf.file_topo_value
    flag_strainer = conf.flag_strainer
    flag_save = conf.flag_save
    dir_output = conf.dir_output
    nsims = conf.nsims

    # read GIS information
    (tower, sel_lines, fid_by_line, fid2name, lon, lat) = \
        read_tower_GIS_information(shape_file_tower, shape_file_line, file_design_value, file_topo_value)

    # read collapse fragility by asset type
    (frag, ds_list, nds) = read_frag(file_frag)

    # read conditional collapse probability
    cond_pc = read_cond_prob(file_cond_pc)

    # calculate conditional collapse probability
    for i in tower.keys():
        tower[i].idfy_adj_list(tower, fid2name, cond_pc, flag_strainer)
        tower[i].cal_cond_pc_adj(cond_pc, fid2name)

    # read wind profile and design wind speed
    event = read_velocity_profile(Event, dir_wind_timeseries, tower, file_terrain_height)
    idx_time = event[event.keys()[0]].wind.index
    ntime = len(idx_time)

    for i in tower.keys():
        event[i].cal_pc_wind(tower[i], frag, ntime, ds_list, nds)  #
        event[i].cal_pc_adj(tower[i], cond_pc)  #

    # analytical approach
    pc_collapse = {}
    for line in sel_lines:
        pc_collapse[line] = cal_collapse_of_towers_analytical(fid_by_line[line], 
            event, fid2name, ds_list, idx_time, ntime)       
        if flag_save:
            for (ds, _) in ds_list:
                csv_file = dir_output + "/pc_line_" + ds + '_' + line.replace(' - ','_') + ".csv"
                pc_collapse[line][ds].to_csv(csv_file)
            
    print "Analytical calculation is completed"

    # mc approach
    # realisation of tower collapse in each simulation
    tf_sim_all = dict()
    prob_sim_all = dict()
    est_ntower_all = dict()
    prob_ntower_all = dict()
    est_ntower_nc_all = dict()
    prob_ntower_nc_all = dict()
    import time
    tic = time.clock()
    if conf.parallel:
        print "parallel MC run on......"
        mc_returns = parmap.map(mc_loop, range(len(sel_lines)), conf, sel_lines, ntime,
                                fid_by_line, event, tower, fid2name, ds_list, nds, idx_time)

        for id, line in enumerate(sel_lines):
            tf_sim_all[line] = mc_returns[id][0]
            prob_sim_all[line] = mc_returns[id][1]
            est_ntower_all[line] = mc_returns[id][2]
            prob_ntower_all[line] = mc_returns[id][3]
            est_ntower_nc_all[line] = mc_returns[id][4]
            prob_ntower_nc_all[line] = mc_returns[id][5]
    else:
        for id, line in enumerate(sel_lines):
            tf_sim, prob_sim, est_ntower, prob_ntower, est_ntower_nc, prob_ntower_nc = mc_loop(id, conf, sel_lines,
                ntime, fid_by_line, event, tower, fid2name, ds_list, nds, idx_time)
            tf_sim_all[line] = tf_sim
            prob_sim_all[line] = prob_sim
            est_ntower_all[line] = est_ntower
            prob_ntower_all[line] = prob_ntower
            est_ntower_nc_all[line] = est_ntower_nc
            prob_ntower_nc_all[line] = prob_ntower_nc
    print '------------>>>>>>time taken', time.clock() - tic

    print "MC calculation is completed"
    return tf_sim_all, prob_sim_all, est_ntower_all, prob_ntower_all, est_ntower_nc_all, prob_ntower_nc_all


def mc_loop(id, conf, lines, ntime, fid_by_line, event, tower, fid2name, ds_list, nds, idx_time):
    line = lines[id]
    if conf.test:
        print "we are in test, Loop", id
        prng = np.random.RandomState(id)
    else:
        print "MC sim, Loop:", id
        prng = np.random.RandomState()
    rv = prng.uniform(size=(conf.nsims, ntime))  # perfect correlation within a single line

    for i in fid_by_line[line]:
        event[fid2name[i]].cal_mc_adj(tower[fid2name[i]], conf.nsims, ntime, ds_list, nds, rv, id)

    # compute estimated number and probability of towers without considering
    # cascading effect
    (est_ntower_nc, prob_ntower_nc) = cal_exp_std_no_cascading(
        fid_by_line[line], event, fid2name, ds_list, conf.nsims, idx_time, ntime)

    # compute collapse of tower considering cascading effect
    (tf_sim, prob_sim) = (cal_collapse_of_towers_mc(fid_by_line[line], event,
                                                    fid2name, ds_list, conf.nsims, idx_time, ntime))
    (est_ntower, prob_ntower) = cal_exp_std(tf_sim, ds_list, idx_time)
    if conf.flag_save:
        for (ds, _) in ds_list:
            npy_file = conf.dir_output + "/tf_line_mc_" + ds + '_' + line.replace(' - ','_') + ".npy"
            np.save(npy_file, tf_sim[ds])

            csv_file = conf.dir_output + "/pc_line_mc_" + ds + '_' + line.replace(' - ','_') + ".csv"
            prob_sim[ds].to_csv(csv_file)

            csv_file = conf.dir_output + "/est_ntower_" + ds + '_' + line.replace(' - ','_') + ".csv"
            est_ntower[ds].to_csv(csv_file)

            npy_file = conf.dir_output + "/prob_ntower_" + ds + '_' + line.replace(' - ','_') + ".npy"
            np.save(npy_file, prob_ntower[ds])

            csv_file = conf.dir_output + "/est_ntower_nc_" + ds + '_' + line.replace(' - ','_') + ".csv"
            est_ntower_nc[ds].to_csv(csv_file)

            npy_file = conf.dir_output + "/prob_ntower_nc_" + ds + '_' + line.replace(' - ','_') + ".npy"
            np.save(npy_file, prob_ntower_nc[ds])

    return tf_sim, prob_sim, est_ntower, prob_ntower, est_ntower_nc, prob_ntower_nc


if __name__ == '__main__':
    from config_class import TransmissionConfig
    conf = TransmissionConfig()
    sim_towers(conf)
