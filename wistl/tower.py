#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import logging
import bisect
from scipy import stats

from wistl.config import unit_vector_by_bearing, angle_between_unit_vectors


logger = logging.getLogger(__name__)


def compute_damage_by_tower(tower, event, lines, cfg):
    """
    compute probability exceedance of damage of tower in isolation (Pc)
    Note: dmg index is not identical to wind index
    """
    line = lines[tower.lineroute]
    seed = (tower.idl +
            line.seed * line.no_towers +
            event.seed * line.no_towers * len(cfg.selected_lines))
    rnd_state = np.random.RandomState(seed=seed)

    logger.info(f'Process {os.getpid()} done processing record {event.name} - {tower.name}')

    wind = read_wind(tower, event)

    dmg = set_dmg(tower, wind, cfg)

    # compute collapse_adj
    collapse_adj = set_collapse_adj(tower, dmg)

    # dmg by simulation
    dmg_state_sim = set_dmg_state_sim(dmg, cfg, rnd_state)

    dmg_sim = set_dmg_sim(tower, dmg_state_sim, dmg, event, cfg)

    collapse_adj_sim = set_collapse_adj_sim(tower, dmg_state_sim, collapse_adj, rnd_state, cfg)

    return {'event': event.id,
            'line': tower.lineroute,
            'tower': tower.name,
            'dmg': dmg,
            'collapse_adj': collapse_adj,
            'dmg_state_sim': dmg_state_sim,
            'collapse_adj_sim': collapse_adj_sim,
            }


def set_dmg(tower, wind, cfg):

    df = wind.apply(compute_damage_using_directional_vulnerability, args=(tower,), axis=1)

    # apply thresholds
    valid = np.where(df['minor'] > cfg.dmg_threshold)[0]
    try:
        #idt0 = max(min(valid) - 1, 0)
        idt0 = min(valid)
    except ValueError:
        #logger.info(f'{tower.name} sustains no damage')
        dmg = pd.DataFrame(None)
    else:
        #idt1 = max(valid) + 2
        idt1 = max(valid) + 1
        dmg = df.iloc[idt0:idt1]
        dmg.index = wind.index[idt0:idt1]

    return dmg


def set_dmg_state_sim(dmg, cfg, rnd_state):
    """
    determine damage state of tower in isolation by simulation
    # PD not PE = 0(non), 1, 2 (collapse)
    TODO: changed to for loop by no_sims and using delayed?
    """

    dmg_state_sim = {ds: pd.DataFrame(None) for ds in cfg.damage_states}

    if not dmg.empty:

        no_time = len(dmg.index)

        # 1. determine damage state of tower due to wind
        rv = rnd_state.uniform(size=(cfg.no_sims, no_time))

        # ds_wind.shape == (no_sims, no_time)
        # PD not PE = 0(non), 1, 2 (collapse)
        # dmg_state_sim.shape == (no_sims, no_time)
        _array = (rv[:, :, np.newaxis] < dmg.values).sum(axis=2)

        for ids, ds in enumerate(cfg.damage_states, 1):

            # Note that id_time always starts from 0, irrespective of dmg_time_idx 
            id_sim, id_time = np.where(_array == ids)

            # convert to wind time index for aggregation at line level
            #id_time += self.dmg_time_idx[0]

            dmg_state_sim[ds] = pd.DataFrame(
                np.vstack((id_sim, id_time)).T, columns=['id_sim', 'id_time'])

            # using tuple instead of dataframe (ALT) 
            #dmg_state_sim[ds] = tuple([x, y] for (x, y) in zip(id_sim, id_time)])

    return dmg_state_sim


def set_dmg_sim(tower, dmg_state_sim, dmg, event, cfg):
    """
    calls self.dmg_state_sim and compare against dmg
    PE not PD 1, 2 (collapse)
    """

    if not dmg.empty:

        dmg_sim = {}

        pb = 0.0

        for ds in cfg.damage_states[::-1]:  # collapse first

            # res = Counter(map(itemgetter(1), dmg_state_sim[ds])
            _array = dmg_state_sim[ds].groupby('id_time').agg(len)['id_sim']
            dmg_sim[ds] = (pb + _array).fillna(_array).fillna(pb)
            pb = _array
            dmg_sim[ds] = dmg_sim[ds] / cfg.no_sims

            # check whether MC simulation is close to analytical
            dmgx = dmg.iloc[dmg_sim[ds].index][ds].values
            diff = np.abs(dmg_sim[ds].values - dmgx) - (cfg.atol + cfg.rtol * dmgx)

            if len(diff[diff > 0]):
                idx = np.argmax(diff)
                logger.warning(f'PE of {ds} of {tower.name} by {event.id}: {dmg_sim[ds].iloc[idx]:.3f} vs {dmgx[idx]:.3f}')

        return dmg_sim


def set_collapse_adj(tower, dmg):
    """
    used only for analytical approach
    calculate collapse probability of jth tower due to pull by the tower
    Pc(j,i) = P(j|i)*Pc(i)
    """
    # only applicable for tower collapse
    if not dmg.empty:

        collapse_adj = {}
        for key, value in tower.cond_pc_adj.items():
            collapse_adj[key] = dmg['collapse'].values * value

        return collapse_adj


def set_collapse_adj_sim(tower, dmg_state_sim, collapse_adj, rnd_state, cfg):
    """
    : calls self.dmg_state_sim
    :param seed: seed is None if no seed number is provided
    :return:
    """

    if tower.cond_pc_adj_sim_idx and not dmg_state_sim['collapse'].empty:

        df = dmg_state_sim['collapse'].copy()

        # generate regardless of time index
        rv = rnd_state.uniform(size=len(dmg_state_sim['collapse']['id_sim']))

        df['id_adj'] = (rv[:, np.newaxis] >= tower.cond_pc_adj_sim_prob).sum(axis=1)

        # remove case with no adjacent tower collapse
        # copy to avoid warning 
        df = df[df['id_adj'] < len(tower.cond_pc_adj_sim_prob)].copy()

        # replace index with tower id
        df['id_adj'] = df['id_adj'].apply(lambda x: tower.cond_pc_adj_sim_idx[x])

        # check against collapse_adj
        tmp = df.groupby(['id_time', 'id_adj']).apply(len).reset_index()

        for idl in tower.cond_pc_adj.keys():

            prob = tmp.loc[tmp['id_adj'].apply(lambda x: idl in x)].groupby('id_time').sum() / cfg.no_sims
            try:
                np.testing.assert_allclose(prob[0], collapse_adj[idl], atol=cfg.atol, rtol=cfg.rtol)
            except AssertionError:
                logger.debug(
                    f'Pc({idl}|{tower.name}): '
                    f'simulation {prob[0].values} vs. '
                    f'analytical {collapse_adj[idl]}')

        collapse_adj_sim = df.copy()

        return collapse_adj_sim



def angle_between_two(deg1, deg2):
    """
    :param: deg1: angle 1 (0, 360)
            deg2: angle 2 (0, 360)
    """
    assert (deg1 >= 0) and (deg1 <= 360)
    assert (deg2 >= 0) and (deg2 <= 360)

    # angle between wind and tower strong axis (normal1)
    v1 = unit_vector_by_bearing(deg1)
    v2 = unit_vector_by_bearing(deg1 + 180)

    u = unit_vector_by_bearing(deg2)

    angle = min(angle_between_unit_vectors(u, v1),
                angle_between_unit_vectors(u, v2))

    return angle


def get_file_wind(event, tower):

    file_wind = os.path.join(event.path_wind_event,
                             tower.file_wind_base_name)
    try:
        assert os.path.exists(file_wind)
    except AssertionError:
        logger.error(f'Invalid file_wind {file_wind}')

    return file_wind


def read_wind(tower, event):
    """
    set the wind given a file_wind
    """
    try:
        file_wind = get_file_wind(event, tower)
        tmp = pd.read_csv(file_wind,
                          parse_dates=[0],
                          index_col=['Time'],
                          usecols=['Time', 'Speed', 'Bearing'])
    except IOError:
        msg = f'Invalid file_wind {file_wind}'
        logger.critical(msg)
    else:
        wind = tmp.loc[tmp.isnull().sum(axis=1) == 0].copy()
        wind['Speed'] *= event.scale * tower.ratio_z_to_10
        wind['ratio'] = wind['Speed'] / tower.collapse_capacity

    return wind


def compute_damage_using_directional_vulnerability(row, tower):
    """
    :param row: pandas Series of wind
    """

    key = get_directional_vulnerability(tower, row['Bearing'])

    dmg = {}
    for ds, (fn, param1, param2) in tower.frag_dic[key].items():
        value = getattr(stats, fn).cdf(row['ratio'], float(param2), scale=float(param1))
        dmg[ds] = np.nan_to_num(value, 0.0)

    return pd.Series(dmg)


def get_directional_vulnerability(tower, bearing):
    """

    :param row: pandas Series of wind
    :return:
             | North
        ------------
       |     |      |
       |     |______| strong axis
       |            |
       |            |
        ------------

    """
    sorted_frag_dic_keys = sorted(tower.frag_dic.keys())

    if len(sorted_frag_dic_keys) > 1:

        try:
            angle = angle_between_two(bearing, tower.axisaz)
        except AssertionError:
            logger.error(f'Something wrong in bearing: {bearing}, axisaz: {tower.axisaz} of {tower.name}')
        else:
            # choose fragility given angle
            loc = min(bisect.bisect_right(sorted_frag_dic_keys, angle),
                  len(sorted_frag_dic_keys) - 1)
    else:
        loc = 0

    return sorted_frag_dic_keys[loc]


'''
def set_dmg_time_idx(dmg, wind):
    """
    return starting and edning index of dmg relative to wind time index
    """
    if not dmg.empty:
        idt = wind.index.intersection(dmg.index)
        idt0 = wind.index.get_loc(idt[0])
        idt1 = wind.index.get_loc(idt[-1]) + 1
        return (idt0, idt1)
'''
