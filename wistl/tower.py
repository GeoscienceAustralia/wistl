#!/usr/bin/env python

import os
import bisect
import collections
import pandas as pd
import dask.array as da
import numpy as np

from scipy import stats
from distributed.worker import logger

from wistl.config import FIELDS_TOWER


params_tower = FIELDS_TOWER + [
                'coord',
                'coord_lat_lon',
                'point',
                'frag_dic',
                'file_wind_base_name',
                'ratio_z_to_10',
                'actual_span',
                'u_factor',
                'collapse_capacity',
                'cond_pc',
                'max_no_adj_towers',
                'id_adj', 'idl',
                'cond_pc_adj',
                'cond_pc_adj_sim_idx',
                'cond_pc_adj_sim_prob'
               ]

# define Tower
Tower = collections.namedtuple('Tower', params_tower)


def compute_dmg_by_tower(tower, event, line, cfg, wind=None):
    """
    compute probability exceedance of damage of tower in isolation (Pc)
    Note: dmg index is not identical to wind index
    Args:
        tower: instance of Tower class
        event: instance of Event class
        lines: instance of Line class
        cfg: instance of Config class
        wind: DataFrame(time, columns=['ratio', 'Bearing']
    Returns:
        dict(key=['event', 'line', 'tower',
                  'dmg', 'collapse_adj',
                  'dmg_state_sim', 'collapse_adj_sim'])
    """

    #logger = logging.getLogger(__name__)

    logger.info(f'Processing {event.id}: {line.linename}: {tower.name}')

    #line = lines[tower.lineroute]
    seed = (tower.idl +
            line.seed * line.no_towers +
            event.seed * line.no_towers * len(cfg.selected_lines))
    rnd_state = np.random.RandomState(seed=seed)


    if wind is None:
        wind = read_wind(tower, event)

    dmg = set_dmg(tower, wind, cfg)

    collapse_adj = set_collapse_adj(tower, dmg)

    # dmg by simulation
    if cfg.options['run_simulation'] or cfg.options['run_simulation_wo_cascading']:

        dmg_state_sim = set_dmg_state_sim(dmg, cfg, rnd_state)

        _ = check_sim_accuracy(tower, dmg_state_sim, dmg, event, cfg)

        collapse_adj_sim = set_collapse_adj_sim(tower, dmg_state_sim, collapse_adj, rnd_state, cfg)

    else:

        dmg_state_sim = {ds: pd.DataFrame(None, columns=['id_sim', 'id_time'])
                         for ds in cfg.dmg_states}

        collapse_adj_sim = pd.DataFrame(None, columns=['id_sim', 'id_time', 'id_adj'])


    return {'event': event.id,
            'line': tower.lineroute,
            'tower': tower.name,
            'dmg': dmg,
            'collapse_adj': collapse_adj,
            'dmg_state_sim': dmg_state_sim,
            'collapse_adj_sim': collapse_adj_sim,
            }


def set_dmg(tower, wind, cfg):
    """
    compute dmg using directional vulnerability and apply damage threshold
    Args:
        tower: instance of Tower class
        wind: DataFrame(index=Time, columns=['Speed', 'Bearing', 'ratio'])
        cfg: instance of Config class
    Returns:
        dmg: DataFrme(index=Time, columns=cfg.dmg_states)
             can be empty DataFrame
    """

    #logger = logging.getLogger(__name__)

    df = wind.apply(compute_dmg_using_directional_vulnerability, args=(tower,), axis=1)

    # apply thresholds
    valid = np.where(df['minor'] > cfg.dmg_threshold)[0]
    try:
        #idt0 = max(min(valid) - 1, 0)
        idt0 = min(valid)
    except ValueError:
        logger.info(f'{tower.name} sustains no damage')
        dmg = pd.DataFrame(None, columns=cfg.dmg_states)
    else:
        #idt1 = max(valid) + 2
        idt1 = max(valid) + 1
        dmg = df.iloc[idt0:idt1]
        dmg.index = wind.index[idt0:idt1]
    return dmg


def set_dmg_state_sim(dmg, cfg, rnd_state):
    """
    determine damage state of tower in isolation by simulation
    PD not PE = 0(non), 1, 2 (collapse)
    TODO: changed to for loop by no_sims and using delayed?
    Args:
        dmg: DataFrame(index=wind.index, columns=cfg.dmg_states)
        cfg: instance of Config class
        rnd_state: instance of np.random.RandomState
    Returns:
        dmg_state_sim: Dict(key=ds, value=DataFrme(columns=[id_sim, id_time]))
                       can be empty DataFrame
    """

    dmg_state_sim = {}

    no_time = len(dmg.index)

    # 1. determine damage state of tower due to wind
    rv = rnd_state.uniform(size=(cfg.no_sims, no_time))
    #rv = da.from_array(rv)
    # ds_wind.shape == (no_sims, no_time)
    # PD not PE = 0(non), 1, 2 (collapse)
    # dmg_state_sim.shape == (no_sims, no_time)
    _array = (rv[:, :, np.newaxis] < dmg.values).sum(axis=2)
    #_array = da.from_array(_array, chunks=(4000, 1280))

    for ids, ds in enumerate(cfg.dmg_states, 1):

        # Note that id_time always starts from 0, irrespective of dmg_time_idx 
        id_sim, id_time = np.where(_array == ids)

        # convert to wind time index for aggregation at line level
        #id_time += self.dmg_time_idx[0]

        dmg_state_sim[ds] = pd.DataFrame(
            np.vstack((id_sim, id_time)).T, columns=['id_sim', 'id_time'])

        # using tuple instead of dataframe (ALT) 
        #dmg_state_sim[ds] = tuple([x, y] for (x, y) in zip(id_sim, id_time)])

    return dmg_state_sim


def check_sim_accuracy(tower, dmg_state_sim, dmg, event, cfg):
    """
    calls dmg_state_sim and compare against dmg
    PE not PD 1, 2 (collapse)
    Args:
        tower: instance of Tower class
        dmg_state_sim: Dict(key=ds, value=DataFrme(columns=[id_sim, id_time]))
                       can be empty DataFrame
        dmg: DataFrame(index=wind.index, columns=cfg.dmg_states)
        event: instance of Event class
        cfg: instance of Config class
    Returns:

        dmg_sim: DataFrame(index=wind.index, columns=cfg.dmg_states)
        dmg_sim: DataFrame(key=ds, value=DataFrme(columns=[id_sim, id_time]))
                       can be empty DataFrame

    """

    #logger = logging.getLogger(__name__)

    if dmg.empty:

        dmg_sim = pd.DataFrame(None, columns=cfg.dmg_states)

    else:

        dmg_sim = {}

        pb = pd.DataFrame(None, columns=['id_sim'])

        for ds in cfg.dmg_states[::-1]:  # collapse first

            # res = Counter(map(itemgetter(1), dmg_state_sim[ds])
            df = dmg_state_sim[ds].groupby('id_time').agg(len)/cfg.no_sims
            df = df.add(pb, fill_value=0)
            dmg_sim[ds] = df.rename(columns={'id_sim': ds})
            # copy for the next step
            pb = df

        dmg_sim = pd.concat([dmg_sim[ds] for ds in cfg.dmg_states], axis=1).fillna(0.0)
        if not dmg_sim.empty:
            dmg_sim['Time'] = dmg_sim.apply(lambda x: dmg.index[x.name], axis=1)
            dmg_sim.set_index('Time', inplace=True)

        # check whether MC simulation is close to analytical
        dfc = pd.merge(dmg_sim, dmg, left_index=True, right_index=True,
                       how='outer').fillna(0.0)
        for ds in cfg.dmg_states:
            dfc[f'diff_{ds}'] = dfc.apply(
                lambda x: np.abs(x[f'{ds}_x'] - x[f'{ds}_y']) -
                (cfg.atol + cfg.rtol * x[f'{ds}_y']), axis=1)
            if (dfc[f'diff_{ds}'] > 0).sum():
                idx = dfc[f'diff_{ds}'].idxmax()
                x = dfc.loc[idx, f'{ds}_x']
                y = dfc.loc[idx, f'{ds}_y']
                logger.warning(f'PE({ds} of {tower.name}|{event.id}): (S) {x:.4f} vs (A) {y:.4f}')

        #dmgx = dmg.iloc[dmg_sim[ds].index][ds].values
        #diff = np.abs(dmg_sim[ds].values - dmgx) - (cfg.atol + cfg.rtol * dmgx)

            #if len(diff[diff > 0]):
            #    idx = np.argmax(diff)
            #    logger.warning(f'{event.id}, {tower.name}, {ds}: (S) {dmg_sim[ds].iloc[idx]:.4f} vs (A) {dmgx[idx]:.4f}')

    return dmg_sim


def set_collapse_adj(tower, dmg):
    """
    calculate collapse probability of jth tower due to pull by the tower
    Pc(j,i) = P(j|i)*Pc(i)
    used only for analytical approach
    Args:
        tower: instance of Tower class
        dmg: DataFrame(index=wind.index, columns=cfg.dmg_states)
    Returns:
        Dict(key=tower.idl, value=np.array)
        can be empty dict
    """
    # only applicable for tower collapse
    return {key: dmg['collapse'].values * value
            for key, value in tower.cond_pc_adj.items()
            if not dmg['collapse'].empty}

def check_against_collapse_adj(collapse_adj_sim, collapse_adj, tower, cfg):
    """
    Args:
        collapse_adj_sim: DataFrame(columns=['id_sim','id_time','id_adj']
        collapse_adj: Dict(key=tower.idl, value=np.array)
        tower: instance of Tower class
    Returns:
        DataFrame(columns=['id_sim','id_time','id_adj'])
        can be empty DataFrame
    """

    if not collapse_adj_sim.empty:

        tmp = collapse_adj_sim.groupby(['id_time', 'id_adj']).apply(len).reset_index()

        for idl in tower.cond_pc_adj.keys():

            prob = (tmp.loc[tmp['id_adj'].apply(lambda x: idl in x)].groupby('id_time').sum() / cfg.no_sims)[0].values
            if prob.size:
                try:
                    np.testing.assert_allclose(prob, collapse_adj[idl], atol=cfg.atol, rtol=cfg.rtol)
                except AssertionError:
                    try:
                        idmax = np.abs(prob - collapse_adj[idl]).argmax()
                    except ValueError:
                        idmax = 0
                    finally:
                        logger.warning(f"""
Pc({idl}|{tower.name}): (S) {prob[idmax]:.4f} vs. (A) {collapse_adj[idl][idmax]:.4f}""")

            else:
                logger.warning(f"""
Pc({idl}|{tower.name}): (S) NA vs. (A) {collapse_adj[idl][0]:.4f}""")

        return prob

def set_collapse_adj_sim(tower, dmg_state_sim, collapse_adj, rnd_state, cfg):
    """
    calls self.dmg_state_sim
    :param seed: seed is None if no seed number is provided
    Args:
        tower: instance of Tower class
        dmg_state_sim: Dict(key=ds, value=DataFrme(columns=[id_sim, id_time]))
        collapse_adj: Dict(key=tower.idl, value=np.array)
        rnd_state: instance of np.random.RandomState
        cfg: instance of Config class
    Returns:
        DataFrame(columns=['id_sim','id_time','id_adj'])
        can be empty DataFrame
    """
    if tower.cond_pc_adj_sim_idx and not dmg_state_sim['collapse'].empty:

        # copy to prevent modify the original
        # df: DataFrame(colums=['id_sim','id_time'])
        df = dmg_state_sim['collapse'].copy()

        # generate regardless of time index
        rv = rnd_state.uniform(size=len(df['id_sim']))
        #rv = da.from_array(rv)

        #_array = (rv[:, np.newaxis] >= tower.cond_pc_adj_sim_prob).sum(axis=1)
        #df['id_adj'] = da.from_array(_array)
        df['id_adj'] = (rv[:, np.newaxis] >= tower.cond_pc_adj_sim_prob).sum(axis=1)

        # remove case with no adjacent tower collapse
        # copy to avoid warning 
        df = df[df['id_adj'] < len(tower.cond_pc_adj_sim_prob)].copy()

        # replace index with tower id
        df['id_adj'] = df['id_adj'].apply(lambda x: tower.cond_pc_adj_sim_idx[x])

        # check against collapse_adj
        check_against_collapse_adj(df, collapse_adj,  tower, cfg)

    else:
        df = pd.DataFrame(None, columns=['id_sim', 'id_time', 'id_adj'])

    return df


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



def read_wind(tower, event):
    """
    set the wind given a file_wind
    Args:
        tower: instance of Tower class
        event: instance of Event class
    Returns:
        DataFrame(index=Time,
                  columns=['Speed', 'Bearing', 'ratio'])
    """
    file_wind = os.path.join(event.path_wind_event,
                             tower.file_wind_base_name)
    try:
        assert os.path.exists(file_wind)
    except AssertionError:
        logger.error(f'{__name__}: Invalid file_wind {file_wind}')
    else:
        tmp = pd.read_csv(file_wind,
                          parse_dates=[0],
                          index_col=['Time'],
                          usecols=['Time', 'Speed', 'Bearing'])

        wind = tmp.loc[tmp.isnull().sum(axis=1) == 0].copy()
        wind['Speed'] *= event.scale * tower.ratio_z_to_10
        wind['ratio'] = wind['Speed'] / tower.collapse_capacity

        return wind


def compute_dmg_using_directional_vulnerability(row, tower):
    """
    compute PE of damage using directional vulnerability
    Args:
        row: Series; row of wind DataFrame
        tower: instance of tower class
    Returns:
        Series(index=cfg.dmg_states)
    """

    key = get_directional_vulnerability(tower, row['Bearing'])

    dmg = {}
    for ds, (fn, param1, param2) in tower.frag_dic[key].items():
        value = getattr(stats, fn).cdf(row['ratio'], float(param2), scale=float(param1))
        dmg[ds] = np.nan_to_num(value, 0.0)

    return pd.Series(dmg)


def get_directional_vulnerability(tower, bearing):
    """
    get directional vulnerability
            | North
        ---------
       |    |____| strong axis
       |         |
        ---------
    Args:
        tower: instance of tower class
        bearing: angle from North
    Returns:
        key for frag_dic
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


def unit_vector_by_bearing(angle_deg):
    """
    return unit vector given bearing
    :param angle_deg: 0-360
    :return: unit vector given bearing in degree
    """
    angle_rad = np.deg2rad(angle_deg)
    return np.array([-1.0 * np.sin(angle_rad), -1.0 * np.cos(angle_rad)])


def angle_between_unit_vectors(v1, v2):
    """
    compute angle between two unit vectors
    :param v1: vector 1
    :param v2: vector 2
    :return: the angle in degree between vectors 'v1' and 'v2'

    """
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


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
