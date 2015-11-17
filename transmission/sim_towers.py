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
from event import cal_mc_adj


def sim_towers(conf):
    if conf.test:
        print "==============>testing"

    # read GIS information
    network = TransmissionNetwork(conf)
    tower, sel_lines, fid_by_line, fid2name, lon, lat = network.read_tower_gis_information()

    # calculate conditional collapse probability
    for i, name in enumerate(tower.keys()):
        tower[name].idfy_adj_list(tower, fid2name, conf.cond_pc, conf.flag_strainer)
        tower[name].cal_cond_pc_adj(conf.cond_pc, fid2name)

    # read wind profile and design wind speed
    event = read_velocity_profile(conf, tower)  # dictionary of event class instances

    # Exception handling
    if 'error' in event:
        return event

    idx_time = event[event.keys()[0]].wind.index

    for i in tower.keys():
        event[i].cal_pc_wind()
        event[i].call_pc_adj()

    # analytical approach
    pc_collapse = {}
    for line in sel_lines:
        pc_collapse[line] = cal_collapse_of_towers_analytical(fid_by_line[line],
                                                    event, fid2name, conf.damage_states, idx_time)
        if conf.flag_save:
            print 'Saving....'
            for (ds, _) in conf.damage_states:
                csv_file = conf.dir_output + "/pc_line_" + ds + '_' + line.replace(' - ', '_') + ".csv"
                pc_collapse[line][ds].to_csv(csv_file)

    print "Analytical calculation is completed"

    # realisation of tower collapse in each simulation
    tf_sim_all = dict()
    prob_sim_all = dict()
    est_ntower_all = dict()
    prob_ntower_all = dict()
    est_ntower_nc_all = dict()
    prob_ntower_nc_all = dict()

    tic = time.time()

    if conf.parallel:
        print "Parallel MC run on......"
        mc_returns = parmap.map(mc_loop, range(len(sel_lines)), conf, sel_lines,
                                fid_by_line, event, tower, fid2name, idx_time)

        for idx, line in enumerate(sel_lines):
            tf_sim_all[line], prob_sim_all[line], est_ntower_all[line], prob_ntower_all[line], \
                est_ntower_nc_all[line], prob_ntower_nc_all[line] = \
                mc_returns[idx][0], mc_returns[idx][1], mc_returns[idx][2], mc_returns[idx][3], \
                mc_returns[idx][4], mc_returns[idx][5]
    else:
        print "Serial MC run on......"
        for idx, line in enumerate(sel_lines):
            tf_sim, prob_sim, est_ntower, prob_ntower, est_ntower_nc, prob_ntower_nc = mc_loop(idx, conf, sel_lines,
                fid_by_line, event, tower, fid2name, idx_time)
            tf_sim_all[line] = tf_sim
            prob_sim_all[line] = prob_sim
            est_ntower_all[line] = est_ntower
            prob_ntower_all[line] = prob_ntower
            est_ntower_nc_all[line] = est_ntower_nc
            prob_ntower_nc_all[line] = prob_ntower_nc

    print 'MC simulation took {} seconds'.format(time.time() - tic)

    return tf_sim_all, prob_sim_all, est_ntower_all, prob_ntower_all, est_ntower_nc_all, prob_ntower_nc_all, sel_lines


def mc_loop(id, conf, lines, fid_by_line, events, tower, fid2name, idx_time):
    ntime = len(idx_time)
    line = lines[id]
    damage_states = conf.damage_states
    if conf.test:
        print "we are in test, Loop", id
        prng = np.random.RandomState(id)
    else:
        print "MC sim, Loop:", id
        prng = np.random.RandomState()
        id = None  # required for true random inside cal_mc_adj
    rv = prng.uniform(size=(conf.nsims, ntime))  # perfect correlation within a single line


    # for i in fid_by_line[line]:
    #     print '------------>>>>', i, fid2name[i]
    #     event[fid2name[i]].cal_mc_adj(tower[fid2name[i]], damage_states, rv, id)

    tower_ids = [fid2name[l] for l in fid_by_line[line]]

    if conf.parallel_towers:
        events_list = parmap.map(cal_mc_adj, tower_ids, events, tower, damage_states, rv, id)
        events = {t: events_list[i] for i, t in enumerate(tower_ids)}
    else:
        for i, l in enumerate(fid_by_line[line]):
            tower_id = fid2name[l]
            events[tower_id] = cal_mc_adj(tower_id, events, tower, damage_states, rv, id)

    # compute estimated number and probability of towers without considering
    # cascading effect
    est_ntower_nc, prob_ntower_nc = cal_exp_std_no_cascading(
        fid_by_line[line], events, fid2name, damage_states, conf.nsims, idx_time, ntime)

    # compute collapse of tower considering cascading effect
    tf_sim, prob_sim = (cal_collapse_of_towers_mc(fid_by_line[line], events,
                                                    fid2name, damage_states, conf.nsims, idx_time, ntime))
    est_ntower, prob_ntower = cal_exp_std(tf_sim, idx_time)
    if conf.flag_save:
        for (ds, _) in damage_states:
            npy_file = conf.dir_output + "/tf_line_mc_" + ds + '_' + line.replace(' - ', '_') + ".npy"
            np.save(npy_file, tf_sim[ds])

            csv_file = conf.dir_output + "/pc_line_mc_" + ds + '_' + line.replace(' - ', '_') + ".csv"
            prob_sim[ds].to_csv(csv_file)

            csv_file = conf.dir_output + "/est_ntower_" + ds + '_' + line.replace(' - ', '_') + ".csv"
            est_ntower[ds].to_csv(csv_file)

            npy_file = conf.dir_output + "/prob_ntower_" + ds + '_' + line.replace(' - ', '_') + ".npy"
            np.save(npy_file, prob_ntower[ds])

            csv_file = conf.dir_output + "/est_ntower_nc_" + ds + '_' + line.replace(' - ', '_') + ".csv"
            est_ntower_nc[ds].to_csv(csv_file)

            npy_file = conf.dir_output + "/prob_ntower_nc_" + ds + '_' + line.replace(' - ', '_') + ".npy"
            np.save(npy_file, prob_ntower_nc[ds])
    print 'loop finished'
    return tf_sim, prob_sim, est_ntower, prob_ntower, est_ntower_nc, prob_ntower_nc


if __name__ == '__main__':
    from config_class import TransmissionConfig
    conf = TransmissionConfig()
    sim_towers(conf)

