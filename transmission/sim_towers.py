#!/usr/bin/env python
from __future__ import print_function

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

#import parmap
import time
import sys
from multiprocessing import Pool

#from read import read_velocity_profile
#from compute import cal_collapse_of_towers_analytical,\
#    cal_collapse_of_towers_mc, cal_exp_std, cal_exp_std_no_cascading
#from read import TransmissionNetwork
#from transmission_line import TransmissionLine
#from transmission_network import TransmissionNetwork
from event_set import create_event_set


def sim_towers(conf):
    if conf.test:
        print("==============>testing")

    # read wind profile and design wind speed
    event_set = create_event_set(conf)

    tic = time.time()

    if conf.parallel:

        print("Parallel run on......")

        # Make the Pool of workers
        pool = Pool(2)
        for event_key, damage_network in event_set.iteritems():
            damage_prob = pool.map(mc_compute, damage_network.lines.values())
        pool.close()

        # damage_prob = {}
        #     print('computing damage probability for {}'.format(event_key))
        #     damage_prob[event_key] = \
        #         parmap.map(mc_compute, damage_network.lines.keys(), damage_network)

        print('MC simulation took {} seconds'.format(time.time() - tic))

    else:
        print('Serial run on.....')

        damage_prob = {}
        for event_key, damage_network in event_set.iteritems():
            print('computing damage probability for {}'.format(event_key))
            for line_key, damage_line in damage_network.lines.iteritems():
                damage_prob.setdefault(event_key, {})[line_key] =\
                    damage_line.compute_collapse_of_towers_analytical()


            # tf_sim, prob_sim, est_ntower, prob_ntower, est_ntower_nc, \
            #     prob_ntower_nc = mc_loop(idx, conf, sel_lines, fid_by_line,
            #                              event, tower, id2name, idx_time)
            # tf_sim_all[line] = tf_sim
            # prob_sim_all[line] = prob_sim
            # est_ntower_all[line] = est_ntower
            # prob_ntower_all[line] = prob_ntower
            # est_ntower_nc_all[line] = est_ntower_nc
            # prob_ntower_nc_all[line] = prob_ntower_nc

        print('MC simulation took {} seconds'.format(time.time() - tic))

    return damage_prob

#    return tf_sim_all, prob_sim_all, est_ntower_all, prob_ntower_all, \
#        est_ntower_nc_all, prob_ntower_nc_all, sel_lines

def mc_compute(damage_line):

    #pc_collapse = dict()
    #print('MC sim, Loop: {}'.format(damage_line))
    pc_collapse = damage_line.compute_collapse_of_towers_analytical()
    #print('loop {} finished'.format(damage_line))

    return pc_collapse


# def mc_compute(damage_network):

#     pc_collapse = dict()
#     print('MC sim, Loop: {}'.format(pid))
#     pc_collapse[pid] = damage_network.lines[pid].compute_collapse_of_towers_analytical()
#     print('loop {} finished'.format(pid))

#     return pc_collapse[pid]


# def mc_loop(id, conf, lines, fid_by_line, event, tower, id2name, idx_time):
#     ntime = len(idx_time)
#     line = lines[id]
#     damage_states = conf.damage_states
#     if conf.test:
#         print('we are in test, Loop {}'.format(id))
#         prng = np.random.RandomState(id)
#     else:
#         print('MC sim, Loop: {}'.format(id))
#         prng = np.random.RandomState()
#         id = None  # required for true random inside cal_mc_adj
#     rv = prng.uniform(size=(conf.nsims, ntime))  # perfect correlation within a single line

#     for i in fid_by_line[line]:
#         event[id2name[i]].cal_mc_adj(tower[id2name[i]], damage_states, rv, id)

#     # compute estimated number and probability of towers without considering
#     # cascading effect
#     est_ntower_nc, prob_ntower_nc = cal_exp_std_no_cascading(fid_by_line[line],
#                                                              event,
#                                                              id2name,
#                                                              damage_states,
#                                                              conf.nsims,
#                                                              idx_time,
#                                                              ntime)

#     # compute collapse of tower considering cascading effect
#     tf_sim, prob_sim = cal_collapse_of_towers_mc(fid_by_line[line],
#                                                  event,
#                                                  id2name,
#                                                  damage_states,
#                                                  conf.nsims,
#                                                  idx_time,
#                                                  ntime)
#     est_ntower, prob_ntower = cal_exp_std(tf_sim, idx_time)
#     if conf.flag_save:
#         line_ = line.replace(' - ', '_')
#         for (ds, _) in damage_states:
#             npy_file = os.path.join(conf.dir_output,
#                                     'tf_line_mc_{}_{}.npy'.format(ds, line_))
#             np.save(npy_file, tf_sim[ds])

#             csv_file = os.path.join(conf.dir_output,
#                                     'pc_line_mc_{}_{}.csv'.format(ds, line_))
#             prob_sim[ds].to_csv(csv_file)

#             csv_file = os.path.join(conf.dir_output,
#                                     'est_ntower_{}_{}.csv'.format(ds, line_))
#             est_ntower[ds].to_csv(csv_file)

#             npy_file = os.path.join(conf.dir_output,
#                                     'prob_ntower_{}_{}.npy'.format(ds, line_))
#             np.save(npy_file, prob_ntower[ds])

#             csv_file = os.path.join(conf.dir_output,
#                                     'est_ntower_nc_{}_{}.csv'.format(ds, line_))
#             est_ntower_nc[ds].to_csv(csv_file)

#             npy_file = os.path.join(conf.dir_output,
#                                     'prob_ntower_nc_{}_{}.npy'.format(ds,
#                                                                       line_))
#             np.save(npy_file, prob_ntower_nc[ds])
#     print('loop {} finished'.format(id))
#     return tf_sim, prob_sim, est_ntower, prob_ntower, est_ntower_nc,\
#         prob_ntower_nc


if __name__ == '__main__':

    args = sys.argv[1:]

    if not args:
        print('python sim_towers.py <config-file>')
        sys.exit(1)

    from config_class import TransmissionConfig
    conf = TransmissionConfig(cfg_file=args[0])
    sim_towers(conf)
