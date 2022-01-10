#!/usr/bin/env python

import os
import collections
import pandas as pd
import numpy as np
#import h5py
#import logging
from distributed.worker import logger

from wistl.config import FIELDS_LINE


params_line = FIELDS_LINE + [
              'coord',
              'coord_lat_lon',
              'line_string',
              'name_output',
              'no_towers',
              'actual_span',
              'seed',
              #'id2name',
              #'ids',
              'names',
              #'name2id',
              ]

# lineroute used as key
params_line.remove('lineroute')


# define Line
Line = collections.namedtuple('Line', params_line)


def compute_dmg_by_line(line, results, event, cfg):
    """
    mc simulation over transmission line

    Args:
        line: instance of Line class
        results: nested list of output of compute_dmg_by_tower
        evnet: instance of Event class
        cfg: instance of Config class
    Returns:
        dict(key=['event', 'line',
                  'dmg_prob',
                  'dmg_prob_sim',
                  'no_dmg', 'prob_no_dmg',
                  'dmg_prob_sim_wo_cascading',
                  'no_dmg_wo_cascading', 'prob_no_dmg_wo_cascading'])
    """

    #logger = logging.getLogger(__name__)

    #logger.info(f'computing damage of {self.name} for {self.event_id}')

    # compute damage probability analytically
    dmg_prob, dmg = compute_dmg_prob(event, line, cfg, results)

    # perfect correlation within a single line
    if cfg.options['run_simulation']:
        dmg_prob_sim, no_dmg, prob_no_dmg =  compute_dmg_prob_sim(event, line, cfg, results, dmg)
    else:
        dmg_prob_sim = {
            ds: pd.DataFrame(None, columns=line.names) for ds in cfg.dmg_states}
        no_dmg = {
            ds: pd.DataFrame(None, columns=['mean', 'std']) for ds in cfg.dmg_states}
        prob_no_dmg = {
            ds: pd.DataFrame(None, columns=range(line.no_towers + 1)) for ds in cfg.dmg_states}

    if cfg.options['run_simulation_wo_cascading']:
        dmg_prob_sim_wo_cascading, no_dmg_wo_cascading, prob_no_dmg_wo_cascading = compute_dmg_prob_sim_wo_cascading(event, line, cfg, results, dmg)
    else:
        dmg_prob_sim_wo_cascading = {
            ds: pd.DataFrame(None, columns=line.names) for ds in cfg.dmg_states}
        no_dmg_wo_cascading = {
            ds: pd.DataFrame(None, columns=['mean', 'std']) for ds in cfg.dmg_states}
        prob_no_dmg_wo_cascading = {
            ds: pd.DataFrame(None, columns=range(line.no_towers + 1)) for ds in cfg.dmg_states}

    # compare simulation against analytical
    if dmg_prob is not None and cfg.options['run_simulation']:

        for ds in cfg.dmg_states:
            diff = np.abs(dmg_prob_sim[ds] - dmg_prob[ds]) - (cfg.atol + cfg.rtol * dmg_prob[ds])

            if len(diff[diff > 0]):
                idx = diff[diff > 0].max(axis=0).idxmax()
                idt = diff[diff > 0][idx].idxmax()
                logger.warning(f"""
Simulation results not close to analytical - {event.id}, {line.linename}
{idx}, {ds}: (S) {dmg_prob_sim[ds][idx].loc[idt]:.4f} vs (A) {dmg_prob[ds][idx].loc[idt]:.4f}""")

    # save
    #if cfg.options['save_output']:
    #   write_output()

    return {'event': event.id,
            'line': line.linename,
            'dmg_prob': dmg_prob,
            'dmg_prob_sim': dmg_prob_sim,
            'no_dmg': no_dmg,
            'prob_no_dmg': prob_no_dmg,
            'dmg_prob_sim_wo_cascading': dmg_prob_sim_wo_cascading,
            'no_dmg_wo_cascading': no_dmg_wo_cascading,
            'prob_no_dmg_wo_cascading': prob_no_dmg_wo_cascading,
            }


def compute_stats(dic_tf_ds, cfg, no_towers, idx_time):
    """
    compute mean and std of no. of ds
    tf_collapse_sim.shape = (no_towers, no_sim, no_time)
    :param tf_sim:
    :return:
    """
    #tic = time.time()
    no_dmg = {}
    prob_no_dmg = {}
    columns = [str(x) for x in range(no_towers + 1)]
    no_time = len(idx_time)

    # (no_towers, 1)
    x_tower = np.array(range(no_towers + 1))[:, np.newaxis]
    x2_tower = x_tower ** 2.0

    tf_ds = np.zeros((no_towers,
                      cfg.no_sims,
                      no_time), dtype=bool)

    # from collapse and minor
    for ds in cfg.dmg_states[::-1]:

        tf_ds = np.logical_xor(dic_tf_ds[ds], tf_ds)

        # mean and standard deviation
        # no_ds_across_towers.shape == (no_sims, no_time)
        no_ds_across_towers = tf_ds.sum(axis=0)
        prob = np.zeros(shape=(no_time, no_towers + 1))

        for i in range(no_time):
            value, freq = np.unique(no_ds_across_towers[:, i], return_counts=True)  # (value, freq)
            prob[i, [int(x) for x in value]] = freq

        prob /= cfg.no_sims  # (no_time, no_towers)

        _exp = np.dot(prob, x_tower)
        _std = np.sqrt(np.dot(prob, x2_tower) - _exp ** 2)

        no_dmg[ds] = pd.DataFrame(np.hstack((_exp, _std)),
            columns=['mean', 'std'], index=idx_time)

        prob_no_dmg[ds] = pd.DataFrame(prob, columns=columns, index=idx_time)

    return no_dmg, prob_no_dmg



def compute_dmg_prob(event, line, cfg, results):
    """
    calculate damage probability of towers analytically
    Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
    where Pd: collapse due to direct wind
    Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))
    pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
    Args:
        evnet: instance of Event class
        line: instance of Line class
        cfg: instance of Config class
        results: nested list of output of compute_damage_by_tower
    Returns:
        dmg_prob: dict(key=ds, value=DataFrme(index=Time, columns=line.names))
                       can be empty DataFrame
        dmg: DataFrame(columns=[ds_tower.name]) can be empty DataFrame
    """

    dmg_prob = {}

    # dmg (DataFrame(index: time, columns: {ds}_{name})
    # can be empty dataframe
    dmg = get_dmg_from_results(event, line, cfg, results)
    # remove columns with NaN
    #dmg.dropna(axis=1, how='all', inplace=True)

    # collapse_adj Dict(key=(index, name), value=
    # can be empty dict
    collapse_adj = {x['tower']: x['collapse_adj'] for x in results
                    if (x['event']==event.id) and (x['line']==line.linename) and (x['collapse_adj'])}

    pc_adj_agg = np.zeros((line.no_towers, line.no_towers, dmg.shape[0]))

    # prob of collapse
    #for k, name in enumerate(line.names):
    for name, val in collapse_adj.items():

        # adjust tower dmg_time_idx to line 
        k = line.names.index(name)

        for idl, prob in val.items():

            new = dmg[f'collapse_{name}'].copy()
            try:
                new.loc[~new.isna()] = prob
            except ValueError:
                logger.warning(f'different length provided for {name}')
            else:
                pc_adj_agg[k, idl] = new.fillna(0.0)

        pc_adj_agg[k, k] = dmg[f'collapse_{name}'].fillna(0.0).values

    # pc_collapse.shape == (no_tower, no_time)
    pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)

    dmg_prob['collapse'] = pd.DataFrame(pc_collapse.T,
            columns=line.names, index=dmg.index)

    # non_collapse state
    for ds in cfg.non_collapse:

        temp = np.zeros_like(pc_collapse)

        for k, name in enumerate(line.names):

            dmg_ds = dmg[f'{ds}_{name}'].fillna(0.0).values
            dmg_collapse = dmg[f'collapse_{name}'].fillna(0.0).values

            # P(DS>ds) - P(collapse directly) + P(collapse induced)
            try:
                value = dmg_ds - dmg_collapse + pc_collapse[k]
            except ValueError:
                self.logger.critical(f'PE of {ds} of {name} can not be used, {dmg_ds.shape} and {dmg_collapse.shape} is incompatible with {pc_collapse.shape}')
            else:
                temp[k] = np.where(value > 1.0, [1.0], value)

        dmg_prob[ds] = pd.DataFrame(temp.T,
                columns=line.names, index=dmg.index)

    return dmg_prob, dmg



def compute_dmg_prob_sim(event, line, cfg, results, dmg=None):

    if dmg is None:
        dmg = get_dmg_from_results(event, line, cfg, results)

    if dmg.empty:

        dmg_prob_sim = {
            ds: pd.DataFrame(None, columns=line.names) for ds in cfg.dmg_states}
        no_dmg = {
            ds: pd.DataFrame(None, columns=['mean', 'std']) for ds in cfg.dmg_states}
        prob_no_dmg = {
            ds: pd.DataFrame(None, columns=range(line.no_towers + 1)) for ds in cfg.dmg_states}

    else:

        # dmg_state_sim Dict(key=name, value=Dict(key=ds, value=DataFrame(columns=[id_sim, id_time])))
        # can be empty
        dmg_state_sim = {x['tower']: x['dmg_state_sim'] for x in results
                        if (x['event']==event.id) and (x['line']==line.linename) and not x['dmg_state_sim']['collapse'].empty}
        # collapse_adj_sim
        collapse_adj_sim = {x['tower']: x['collapse_adj_sim'] for x in results
                        if (x['event']==event.id) and (x['line']==line.linename) and not x['collapse_adj_sim'].empty}

        #collapse_adj_sim_name = [(name, collapse_adj_sim[name]) for k, name in enumerate(line.names)]

        # perfect correlation within a single line
        dmg_prob_sim = {}
        #self.dmg_idx = {}  # tuple of idl, id_sim, id_time

        #idx_by_ds = {ds: None for ds in self.dmg_states}
        dic_tf_ds = {ds: None for ds in cfg.dmg_states}
        tf_ds = np.zeros((line.no_towers, cfg.no_sims, dmg.shape[0]), dtype=bool)

        # collapse by adjacent towers
        for name, value in collapse_adj_sim.items():

            #try:
            #    collapse_adj_sim[name]['id_adj']
            #except KeyError:
            #    pass
            #else:
            # adjust time index from tower to line
            _ = (~dmg[f'collapse_{name}'].isna()).idxmax()
            ida = dmg[f'collapse_{name}'].index.get_loc(_)

            for id_adj, grp in value.groupby('id_adj'):
                for idl in id_adj:
                    tf_ds[idl, grp['id_sim'].values, grp['id_time'].values + ida] = True

            #else:
            #    print(f'collapse_adj_sim of {self.name} for {self.towers[k].name},{self.towers[k].idl} is None')
        # append damage state by direct wind
        for ds in cfg.dmg_states[::-1]:  # collapse first

            for name, value in dmg_state_sim.items():

                # adjust time index from tower to line
                _ = (~dmg[f'collapse_{name}'].isna()).idxmax()
                ida = dmg[f'collapse_{name}'].index.get_loc(_)

                #dmg_state_sim = self.towers[k].dmg_state_sim[ds].loc[idx].copy()

                tf_ds[line.names.index(name), value[ds]['id_sim'].values,
                      value[ds]['id_time'].values + ida] = True

            # dmg index for line interaction
            #self.dmg_idx[ds] = np.where(tf_ds)

            # PE(DS)
            dmg_prob_sim[ds] = pd.DataFrame(tf_ds.sum(axis=1).T / cfg.no_sims,
                columns=line.names, index=dmg.index)

            dic_tf_ds[ds] = tf_ds.copy()
            #idx = np.where(tf_ds)
            #idx_by_ds[ds] = [(id_time, id_sim, id_tower) for
            #                 id_tower, id_sim, id_time in zip(*idx)]

        # compute mean, std of no. of damaged towers
        no_dmg, prob_no_dmg = compute_stats(dic_tf_ds, cfg, line.no_towers, dmg.index)

        # checking against analytical value
        #for name in self.names:
        #    try:
        #        np.testing.assert_allclose(self.dmg_prob['collapse'][name].values,
        #            self.dmg_prob_sim['collapse'][name].values, atol=self.atol, rtol=self.rtol)
        #    except AssertionError:
        #        self.logger.warning(f'Simulation results of {name}:collapse are not close to the analytical')
    return dmg_prob_sim, no_dmg, prob_no_dmg


def get_dmg_from_results(event, line, cfg, results):
    """
    get dmg from results

    """

    dump_str, dump_df = zip(*[(x['tower'], x['dmg']) for x in results
                          if (x['event']==event.id) and (x['line']==line.linename)])
    dmg = pd.concat(dump_df, join='outer', axis=1)

    _str = [[f'{ds}_{name}' for name in dump_str] for ds in cfg.dmg_states]
    dmg.columns = [x for z in zip(*_str) for x in z]

    return dmg


def compute_dmg_prob_sim_wo_cascading(event, line, cfg, results = None, dmg=None):

    if results is None:
        assert not dmg.empty, logger.warning('Either dmg or results should be provided')
        assert dmg_state_sim is not None, logger.warning('Either dmg_state_sim or results should be provided')

    if dmg is None:
        dmg = get_dmg_from_results(event, line, cfg, results)

    if dmg.empty:
        dmg_prob_sim_wo_cascading = {
            ds: pd.DataFrame(None, columns=line.names) for ds in cfg.dmg_states}
        no_dmg_wo_cascading = {
            ds: pd.DataFrame(None, columns=['mean', 'std']) for ds in cfg.dmg_states}
        prob_no_dmg_wo_cascading = {
            ds: pd.DataFrame(None, columns=range(line.no_towers + 1)) for ds in cfg.dmg_states}

    else:

        # dmg_state_sim
        dmg_state_sim = {x['tower']: x['dmg_state_sim'] for x in results
                        if (x['event']==event.id) and (x['line']==line.linename) and not x['dmg_state_sim']['collapse'].empty}
        # collapse_adj_sim
        collapse_adj_sim = {x['tower']: x['collapse_adj_sim'] for x in results
                        if (x['event']==event.id) and (x['line']==line.linename) and not x['collapse_adj_sim'].empty}


        dmg_prob_sim_wo_cascading = {}

        dic_tf_ds = {ds: None for ds in cfg.dmg_states}

        tf_ds = np.zeros((line.no_towers, cfg.no_sims, dmg.shape[0]), dtype=bool)

        for ds in cfg.dmg_states[::-1]:  # collapse first

            for name,value in dmg_state_sim.items():

                # adjust time index from tower to line
                _ = (~dmg[f'collapse_{name}'].isna()).idxmax()
                ida = dmg[f'collapse_{name}'].index.get_loc(_)

                tf_ds[line.names.index(name), value[ds]['id_sim'].values,
                      value[ds]['id_time'].values + ida] = True

            # PE(DS)
            dmg_prob_sim_wo_cascading[ds] = pd.DataFrame(
                tf_ds.sum(axis=1).T / cfg.no_sims,
                columns=line.names,
                index=dmg.index)

            dic_tf_ds[ds] = tf_ds.copy()

        no_dmg_wo_cascading, prob_no_dmg_wo_cascading = compute_stats(dic_tf_ds, cfg, line.no_towers, dmg.index)

        # checking against analytical value for non collapse damage states
        #for k in self.dmg_towers:
        #    name = self.towers[k].name
        #    idt0, idt1 = [x - self.dmg_time_idx[0] for x in self.towers[k].dmg_time_idx]
        #    try:
        #        np.testing.assert_allclose(self.towers[k].dmg[ds].values,
        #                self.dmg_prob_sim_no_cascading[ds].iloc[idt0:idt1][name].values, atol=self.atol, rtol=self.rtol)
        #    except AssertionError:
        #        self.logger.warning(f'Simulation results of {name}:{ds} are not close to the analytical')
    return dmg_prob_sim_wo_cascading, no_dmg_wo_cascading, prob_no_dmg_wo_cascading













class Line_OLD(object):
    """ class for a collection of towers """

    registered = ['name',
                  'coord',
                  'coord_lat_lon',
                  'dic_towers',
                  'id2name',
                  'ids',
                  'line_string',
                  'name_output',
                  'names',
                  'no_towers',
                  'target_no_towers',
                  'no_sims',
                  'dmg_states',
                  'non_collapse',
                  'scale',
                  'event_id',
                  'rnd_state',
                  'rtol',
                  'atol',
                  'dmg_threshold',
                  'line_interaction',
                  'path_event',
                  'path_output']

    def __init__(self, logger=None, **kwargs):

        self.logger = logger or logging.getLogger(__name__)

        self.name = None
        self.coord = None
        self.coord_lat_lon = None
        self.dic_towers = None
        self.id2name = None
        self.ids = None
        self.event_id = None
        self.line_string = None
        self.name_output = None
        self.names = None
        self.no_towers = None
        self.target_no_towers = None
        self.no_sims = None
        self.dmg_states = None
        self.non_collapse = None
        self.path_event = None
        self.path_output = None
        self.scale = None

        for key, value in kwargs.items():
            if key in self.registered:
                setattr(self, key, value)

        self._towers = None
        self._dmg_towers = None

        # analytical method
        self.dmg_prob = None

        # simulation method
        self.dmg_prob_sim = None
        self.no_dmg = None
        self.prob_no_dmg = None
        self.dmg_idx = None

        # no cascading collapse
        self.dmg_prob_sim_no_cascading = None
        self.no_dmg_no_cascading = None
        self.prob_no_dmg_no_cascading = None

        # line interaction
        self._dmg_idx_interaction = None
        self.dmg_prob_interaction = None
        self.prob_no_dmg_interaction = None
        self.no_dmg_interaction = None

        self._time = None
        self._dmg_time_idx = None
        self._no_time = None


        self._file_output = None

    def __repr__(self):
        return f'Line(name={self.name}, no_towers={self.no_towers}, event_id={self.event_id})'

    #def __getstate__(self):
    #    d = self.__dict__.copy()
    #    if 'logger' in d:
    #        d['logger'] = d['logger'].name
    #    return d

    #def __setstate__(self, d):
    #    if 'logger' in d:
    #        d['logger'] = logging.getLogger(d['logger'])
    #    self.__dict__.update(d)

    @property
    def towers(self):

        if self._towers is None:

            self._towers = {}

            for idn in self.dic_towers:
                dic = h_tower(line=self, idn=idn)
                self._towers[dic['idl']] = Tower(**dic)

        return self._towers

    @property
    def file_output(self):
        if self._file_output is None:
            self._file_output = os.path.join(self.path_output, f'{self.event_id}_{self.name}')
        return self._file_output

    @property
    def time(self):
        if self._time is None and self.dmg_towers:
            try:
                self._time = self.towers[0].wind.index[
                        self.dmg_time_idx[0]:self.dmg_time_idx[1]]
            except AttributeError:
                self.logger.error(
                        f'Can not set time for Line:{self.name}')
        return self._time

    @property
    def dmg_time_idx(self):
        """
        index of dmg relative to wind time index
        """
        if self._dmg_time_idx is None:
            # get min of dmg_time_idx[0], max of dmg_time_idx[1]
            tmp = [self.towers[k].dmg_time_idx for k in self.dmg_towers]
            try:
                id0 = list(map(min, zip(*tmp)))[0]
            except IndexError:
                self.logger.info(f'Line:{self.name} sustains no damage')
            else:
                id1 = list(map(max, zip(*tmp)))[1]
                self._dmg_time_idx = (id0, id1)
        return self._dmg_time_idx

    @property
    def dmg_time_idx1(self):
        """
        index of dmg relative to wind time index
    """
        if self._dmg_time_idx is None:
            # get min of dmg_time_idx[0], max of dmg_time_idx[1]
            tmp = [self.towers[k].dmg_idxmax for k in self.dmg_towers]
            flatten = [y for x in tmp for y in x]
            if flatten:
                id0 = max(0, min(flatten) - 1)
                id1 = min(max(flatten) + 1, len(self.towers[0].wind) + 1)
                self._dmg_time_idx1 = id0, id1
            else:
                self.logger.info(f'Line:{self.name} sustains no damage')
        return self._dmg_time_idx1

    @property
    def dmg_towers(self):
        if self._dmg_towers is None:
            self._dmg_towers = [k for k, v in self.towers.items() if v.no_time]
        return self._dmg_towers

    @property
    def no_time(self):
        if self._no_time is None and self.dmg_towers:
            self._no_time = len(self.time)
        return self._no_time

    @property
    def dmg_idx_interaction(self):
        """
        compute damage probability due to parallel line interaction using MC
        simulation
        :param seed:
        :return:
        """
        if self._dmg_idx_interaction is None:

            self._dmg_idx_interaction = defaultdict(list)

            for k in self.dmg_towers:

                # determine damage state by line interaction
                # collapse_interaction['id_sim', 'id_time', 'no_collapse']
                for id_time, grp in self.towers[k].collapse_interaction.groupby('id_time'):

                    # check if wind id_time is consistent with collapse_interaction
                    wind_vector = unit_vector_by_bearing(
                        self.towers[k].wind['Bearing'][self.towers[k].dmg_time_idx[0] + id_time])

                    angle = {}
                    for line_name, value in self.towers[k].target_line.items():
                        angle[line_name] = angle_between_unit_vectors(wind_vector,
                                                                      value['vector'])

                    target_line = min(angle, key=angle.get)

                    if angle[target_line] < 180:

                        target_tower_id = self.towers[k].target_line[target_line]['id']
                        max_no_towers = self.target_no_towers[target_line]

                        for no_collapse, subgrp in grp.groupby('no_collapse'):

                            no_towers = int(no_collapse / 2)

                            right = create_list_idx(target_tower_id, no_towers, max_no_towers, 1)
                            left = create_list_idx(target_tower_id, no_towers, max_no_towers, -1)

                            id_towers = [x for x in right + [target_tower_id] + left if x >= 0]

                            # adjust id_time
                            list_id = list(itertools.product(id_towers, subgrp['id_sim'], [id_time + self.towers[k].dmg_time_idx[0] - self.dmg_time_idx[0]]))

                            self._dmg_idx_interaction[target_line] += list_id

                    else:
                        msg = f'tower:{tower.name}, angle: {angle[target_line]}, wind:{wind_vector}'
                        self.logger.warning(msg)
        return self._dmg_idx_interaction

    def write_csv_output(self, idt_max, key, dic):
        """
        """
        _file = self.file_output + f'_{key}.csv'
        df = pd.DataFrame(None)
        for k, v in dic.items():
            df[k] = v.loc[idt_max]
        df.to_csv(_file)
        self.logger.info(f'{_file} is saved')

    def write_output(event_id, line_name,
                     dmg_prob,
                     dmg_prob_sim, no_dmg, prob_no_dmg,
                     dmg_prob_sim_wo_cascading, no_dmg_wo_cascading, prob_no_dmg_wo_cascading):

        items = ['dmg_prob', 'dmg_prob_sim', 'dmg_prob_sim_wo_cascading',
                 'no_dmg', 'no_dmg_wo_cascading',
                 'prob_no_dmg', 'prob_no_dmg_wo_cascading']

        columns_by_item = {'dmg_prob': self.names,
                           'dmg_prob_sim': self.names,
                           'dmg_prob_sim_no_cascading': self.names,
                           'no_dmg': ['mean', 'std'],
                           'no_dmg_no_cascading': ['mean', 'std'],
                           'prob_no_dmg': range(self.no_towers + 1),
                           'prob_no_dmg_no_cascading': range(self.no_towers + 1)
                           }

        if self.no_time:

            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                self.logger.info(f'{self.path_output} is created')

            # save no_dmg and prob_no_dmg csv
            try:
                idt_max = self.no_dmg['collapse']['mean'].idxmax()
            except TypeError:
                pass
            else:
                # save no_dmg to csv
                self.write_csv_output(idt_max, 'no_dmg', self.no_dmg)

                # sve prob_no_dmg to csv
                self.write_csv_output(idt_max, 'prob_no_dmg', self.prob_no_dmg)

                # save dmg_prob to csv
                _file = self.file_output + f'_dmg_prob.csv'
                df = pd.DataFrame(None)
                for k, v in self.dmg_prob.items():
                    df[k] = v.loc[idt_max]
                for k, v in self.dmg_prob_sim.items():
                    df[f'{k}_sim'] = v.loc[idt_max]
                df.to_csv(_file)
                self.logger.info(f'{_file} is saved')

            # save no_dmg_non_cascading and prob_no_dmg_non_cascading csv
            try:
                idt_max = self.no_dmg_no_cascading['collapse']['mean'].idxmax()
            except TypeError:
                pass
            else:
                # save no_dmg to csv
                self.write_csv_output(idt_max, 'no_dmg_no_cascading', self.no_dmg_no_cascading)
                # sve prob_no_dmg to csv
                self.write_csv_output(idt_max, 'prob_no_dmg_no_cascading', self.prob_no_dmg_no_cascading)

            _file = self.file_output + '.h5'
            with h5py.File(_file, 'w') as hf:

                for item in items:

                    group = hf.create_group(item)

                    for ds in self.dmg_states:

                        try:
                            value = getattr(self, item)[ds]
                        except TypeError:
                            self.logger.debug(f'cannot get {item}{ds}')
                        else:
                            data = group.create_dataset(ds, data=value)
                            data.attrs['columns'] = ','.join(f'{x}' for x in columns_by_item[item])

                # metadata
                #hf.attrs['nrow'], data.attrs['ncol'] = value.shape
                hf.attrs['time_start'] = str(self.time[0])
                try:
                    hf.attrs['time_freq'] = str(self.time[1]-self.time[0])
                except IndexError:
                    hf.attrs['time_freq'] = str(self.time[0])
                hf.attrs['time_period'] = self.time.shape[0]

            self.logger.info(f'{self.file_output} is saved')
        else:
            self.logger.info(f'No output for {self.name} by {self.event_id}')


'''
def compute_damage_per_line(line, cfg):
    """
    mc simulation over transmission line
    :param line: instance of transmission line
           cfg: instance of config
    :return: None but update attributes of
    """

    logger = logging.getLogger(__name__)

    logger.info(f'computing damage of {line.name} for {line.event_id}')

    # compute damage probability analytically
    line.compute_dmg_prob()

    # perfect correlation within a single line
    if cfg.options['run_simulation']:
        line.compute_dmg_prob_sim()

        if cfg.options['run_no_cascading_collapse']:
            line.compute_dmg_prob_sim_no_cascading()

    # compare simulation against analytical
    #for ds in cfg.dmg_states:
    #    idx_not_close = np.where(~np.isclose(line.dmg_prob_sim[ds].values,
    #                             line.dmg_prob[ds].values,
    #                             atol=ATOL,
    #                             rtol=RTOL))
    #    for idc in idx_not_close[1]:
    #        logger.warning(f'Simulation not CLOSE {ds}:{line.towers[idc].name}')

    # save
    if cfg.options['save_output']:
        line.write_output()

    return line
'''

