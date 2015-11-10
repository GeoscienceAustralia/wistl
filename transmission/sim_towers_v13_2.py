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

import numpy as np
import parmap
import time

from read import read_velocity_profile
from compute import cal_collapse_of_towers_analytical, cal_collapse_of_towers_mc, cal_exp_std, cal_exp_std_no_cascading
from read import TransmissionNetwork
from event import Event


def sim_towers(conf):
    if conf.test:
        print "==============>testing"
    # file_frag = conf.file_frag
    # file_cond_pc = conf.file_cond_pc
    flag_strainer = conf.flag_strainer
    flag_save = conf.flag_save
    dir_output = conf.dir_output

    # read GIS information
    network = TransmissionNetwork(conf)
    tower, sel_lines, fid_by_line, fid2name, lon, lat = network.read_tower_gis_information(conf)

    # read collapse fragility by asset type
    # frag, ds_list, nds = read_frag(file_frag)

    # read conditional collapse probability
    # cond_pc = read_cond_prob(file_cond_pc)
    print tower
    import os
    # for name in tower.keys():
    #     file_name = 'ts.' + name + '.csv'
    #     vel_file = os.path.join(conf.dir_wind_timeseries, file_name)
    #     event_dummy = Event(tower[name], vel_file)

    # calculate conditional collapse probability
    for i, name in enumerate(tower.keys()):
        print i, name
        file_name = 'ts.' + name + '.csv'
        vel_file = os.path.join(conf.dir_wind_timeseries, file_name)
        event_dummy = Event(tower[name], vel_file)
        tower[name].idfy_adj_list(tower, fid2name, event_dummy.cond_pc, flag_strainer)
        tower[name].cal_cond_pc_adj(event_dummy.cond_pc, fid2name)

    # read wind profile and design wind speed
    event = read_velocity_profile(conf, tower)
    if 'error' in event:
        return event

    idx_time = event[event.keys()[0]].wind.index
    # ntime = len(idx_time)

    for i in tower.keys():
        event[i].pc_wind(tower[i])  #
        event[i].cal_pc_adj(tower[i], event_dummy.cond_pc)  #

    # analytical approach
    pc_collapse = {}
    for line in sel_lines:
        pc_collapse[line] = cal_collapse_of_towers_analytical(fid_by_line[line], event, fid2name,
                                                              event_dummy.ds_list, idx_time)
        if flag_save:
            for (ds, _) in event_dummy.ds_list:
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
    tic = time.clock()
    if conf.parallel:
        print "parallel MC run on......"
        mc_returns = parmap.map(mc_loop, range(len(sel_lines)), conf, sel_lines,
                                fid_by_line, event, tower, fid2name, event_dummy.ds_list, event_dummy.nds, idx_time)

        for id, line in enumerate(sel_lines):
            tf_sim_all[line], prob_sim_all[line], est_ntower_all[line], prob_ntower_all[line], \
            est_ntower_nc_all[line], prob_ntower_nc_all[line] = \
                mc_returns[id][0], mc_returns[id][1], mc_returns[id][2], mc_returns[id][3], \
                mc_returns[id][4], mc_returns[id][5]
    else:
        for id, line in enumerate(sel_lines):
            tf_sim, prob_sim, est_ntower, prob_ntower, est_ntower_nc, prob_ntower_nc = mc_loop(id, conf, sel_lines,
                fid_by_line, event, tower, fid2name, event_dummy.ds_list, event_dummy.nds, idx_time)
            tf_sim_all[line] = tf_sim
            prob_sim_all[line] = prob_sim
            est_ntower_all[line] = est_ntower
            prob_ntower_all[line] = prob_ntower
            est_ntower_nc_all[line] = est_ntower_nc
            prob_ntower_nc_all[line] = prob_ntower_nc
    print 'MC simulation took {} seconds'.format(time.clock() - tic)

    print "MC calculation is completed"
    return tf_sim_all, prob_sim_all, est_ntower_all, prob_ntower_all, est_ntower_nc_all, prob_ntower_nc_all, sel_lines


def mc_loop(id, conf, lines, fid_by_line, event, tower, fid2name, ds_list, nds, idx_time):
    ntime = len(idx_time)
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
    est_ntower_nc, prob_ntower_nc = cal_exp_std_no_cascading(
        fid_by_line[line], event, fid2name, ds_list, conf.nsims, idx_time, ntime)

    # compute collapse of tower considering cascading effect
    tf_sim, prob_sim = (cal_collapse_of_towers_mc(fid_by_line[line], event,
                                                    fid2name, ds_list, conf.nsims, idx_time, ntime))
    est_ntower, prob_ntower = cal_exp_std(tf_sim, idx_time)
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
