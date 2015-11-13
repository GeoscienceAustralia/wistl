'''
functions to compute collapse probability of tranmission towers

- pc_wind

- pc_adj_towers

- mc_adj
-
'''

import numpy as np
import pandas as pd
from scipy.stats import itemfreq


def cal_collapse_of_towers_analytical(list_fid, event, fid2name, ds_list,
                                      idx_time):
    """
    calculate collapse of towers analytically

    Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
    whre Pd: collapse due to direct wind
    Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))

    pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
    """
    ntime = len(idx_time)
    ntower = len(list_fid)
    cds_list = [x[0] for x in ds_list]  # only string
    cds_list.remove('collapse')  # non-collapse

    pc_adj_agg = np.zeros((ntower, ntower, ntime))

    for index, i in enumerate(list_fid):
        for j in event[fid2name[i]].pc_adj.keys():
            jcol = list_fid.index(j)
            pc_adj_agg[index, jcol, :] = event[fid2name[i]].pc_adj[j]
        pc_adj_agg[index, index, :] = event[fid2name[i]].pc_wind.collapse.values

    pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)  # (ntower, ntime)

    # prob of non-collapse damage
    prob = {}
    for ds in cds_list:

        temp = np.zeros_like(pc_collapse)
        for index, i in enumerate(list_fid):

            val = (event[fid2name[i]].pc_wind[ds].values
                   - event[fid2name[i]].pc_wind.collapse.values
                   + pc_collapse[index, :])

            val = np.where(val > 1.0, 1.0, val)

            temp[index, :] = val

        prob[ds] = pd.DataFrame(temp.T, columns=[fid2name[x] for x in list_fid],
                                index=idx_time)

    prob['collapse'] = pd.DataFrame(pc_collapse.T,
                                    columns=[fid2name[x] for x in list_fid],
                                    index=idx_time)
    return prob


def cal_collapse_of_towers_mc(list_fid, event, fid2name, ds_list, nsims,
                              idx_time, ntime):

    ntower = len(list_fid)
    cds_list = ds_list[:]
    cds_list.reverse()  # [(collapse, 2), (minor, 1)]

    tf_ds = np.zeros((ntower, nsims, ntime), dtype=bool)

    for i in list_fid:

        # collapse by adjacent towers
        for j in event[fid2name[i]].mc_adj.keys():  # time

            for k in event[fid2name[i]].mc_adj[j].keys():  # fid

                isim = event[fid2name[i]].mc_adj[j][k]

                for l in k:  # each fid

                    tf_ds[list_fid.index(l), isim, j] = True

    prob_sim, tf_sim = {}, {}

    # append damage stae by direct wind
    for ds, _ in cds_list:
        for i in list_fid:
            for (j1, k1) in zip(event[fid2name[i]].mc_wind[ds]['isim'],
                                event[fid2name[i]].mc_wind[ds]['itime']):
                tf_ds[list_fid.index(i), j1, k1] = True

        temp = np.sum(tf_ds, axis=1)/float(nsims)

        prob_sim[ds] = pd.DataFrame(temp.T,
                                    columns=[fid2name[x] for x in list_fid],
                                    index=idx_time)

        tf_sim[ds] = np.copy(tf_ds)

    return tf_sim, prob_sim


def cal_exp_std_no_cascading(list_fid, event, fid2name, ds_list, nsims,
                             idx_time, ntime):

    tf_sim_line = {}

    ntowers = len(list_fid)

    for ds, _ in ds_list:

        tmp = np.zeros((ntowers, nsims, ntime), dtype=bool)

        for i in list_fid:
            isim = event[fid2name[i]].mc_wind[ds]['isim']
            itime = event[fid2name[i]].mc_wind[ds]['itime']
            tmp[list_fid.index(i), isim, itime] = True

        tf_sim_line[ds] = np.copy(tmp)

    est_ntower, prob_ntower = cal_exp_std(tf_sim_line, idx_time)

    return est_ntower, prob_ntower


def cal_exp_std(tf_sim_line, idx_time):
    """
    compute mean and std of no. of ds
    tf_collapse_sim.shape = (ntowers, nsim, ntime)
    """

    est_ntower = {}
    prob_ntower = {}

    for ds in tf_sim_line:

        ntowers, nsims, ntime = tf_sim_line[ds].shape

        # mean and standard deviation
        x_ = np.array(range(ntowers+1))[:, np.newaxis]  # (ntowers, 1)
        x2_ = np.power(x_, 2.0)

        no_ds_acr_towers = np.sum(tf_sim_line[ds], axis=0)  # (nsims, ntime)
        no_freq = np.zeros(shape=(ntime, ntowers+1))  # (ntime, ntowers)

        for i in range(ntime):
            val = itemfreq(no_ds_acr_towers[:, i])  # (value, freq)
            no_freq[i, [int(x) for x in val[:, 0]]] = val[:, 1]

        prob = no_freq / float(nsims)  # (ntime, ntowers)

        exp_ntower = np.dot(prob, x_)
        std_ntower = np.sqrt(np.dot(prob, x2_) - np.power(exp_ntower, 2))

        est_ntower[ds] = pd.DataFrame(np.hstack((exp_ntower, std_ntower)),
                                      columns=['mean', 'std'], index=idx_time)

        prob_ntower[ds] = prob

    return est_ntower, prob_ntower
