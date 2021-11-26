#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
#import h5py
import logging




def compute_damage_by_line(line, results, event, cfg):
    """
    mc simulation over transmission line
    :param line: instance of transmission line
           cfg: instance of config
    :return: None but update attributes of
    """

    logger = logging.getLogger(__name__)

    #logger.info(f'computing damage of {self.name} for {self.event_id}')

    # compute damage probability analytically
    damage_prob = compute_damage_prob(event, line, cfg, results)

    # perfect correlation within a single line
    if cfg.options['run_simulation']:
        damage_prob_sim, no_damage, prob_no_damage =  compute_damage_prob_sim(event, line, cfg, results)

        if cfg.options['run_simulation_wo_cascading']:
            damage_prob_sim_wo_cascading, no_damage_wo_cascading, prob_no_damage_wo_cascading = compute_damage_prob_sim_wo_cascading(event, line, cfg, results)
        else:
            damage_prob_sim_wo_cascading = None
            no_damage_wo_cascading = None
            prob_no_damage_wo_cascading = None
    else:
        damage_prob_sim = None
        no_damage = None
        prob_no_damage = None

    # compare simulation against analytical
    if damage_prob is not None:

        for ds in cfg.damage_states:
            diff = np.abs(damage_prob_sim[ds] - damage_prob[ds]) - (cfg.atol + cfg.rtol * damage_prob[ds])

            if len(diff[diff > 0]):
                idx = diff[diff > 0].max(axis=0).idxmax()
                idt = diff[diff > 0][idx].idxmax()
                logger.warning(f"""
Simulation results not close to analytical - {event.id}, {line.linename}
{idx}, {ds}: (S) {damage_prob_sim[ds][idx].loc[idt]:.4f} vs (A) {damage_prob[ds][idx].loc[idt]:.4f}""")

    # save
    #if cfg.options['save_output']:
    #   write_output()

    return {'event': event.id,
            'line': line.linename,
            'damage_prob': damage_prob,
            'damage_prob_sim': damage_prob_sim,
            'no_damage': no_damage,
            'prob_no_damage': prob_no_damage,
            'dmg_prob_sim_wo_cascading': damage_prob_sim_wo_cascading,
            'no_damage_wo_cascading': no_damage_wo_cascading,
            'prob_no_damage_wo_cascading': prob_no_damage_wo_cascading,
            }


def compute_stats(dic_tf_ds, cfg, no_towers, idx_time):
    """
    compute mean and std of no. of ds
    tf_collapse_sim.shape = (no_towers, no_sim, no_time)
    :param tf_sim:
    :return:
    """
    #tic = time.time()
    no_damage = {}
    prob_no_damage = {}
    columns = [str(x) for x in range(no_towers + 1)]
    no_time = len(idx_time)

    # (no_towers, 1)
    x_tower = np.array(range(no_towers + 1))[:, np.newaxis]
    x2_tower = x_tower ** 2.0

    tf_ds = np.zeros((no_towers,
                      cfg.no_sims,
                      no_time), dtype=bool)

    # from collapse and minor
    for ds in cfg.damage_states[::-1]:

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

        no_damage[ds] = pd.DataFrame(np.hstack((_exp, _std)),
            columns=['mean', 'std'], index=idx_time)

        prob_no_damage[ds] = pd.DataFrame(prob, columns=columns, index=idx_time)

    return no_damage, prob_no_damage



def compute_damage_prob(event, line, cfg, results):
    """
    calculate damage probability of towers analytically
    Pc(i) = 1-(1-Pd(i))x(1-Pc(i,1))*(1-Pc(i,2)) ....
    where Pd: collapse due to direct wind
    Pi,j: collapse probability due to collapse of j (=Pd(j)*Pc(i|j))
    pc_adj_agg[i,j]: probability of collapse of j due to ith collapse
    """

    # dmg
    dump_str, dump_df = zip(*[(x['tower'], x['dmg']) for x in results
                              if (x['event']==event.id) and (x['line']==line.linename)])
    dmg = pd.concat(dump_df, join='outer', axis=1)

    if not dmg.empty:

        damage_prob = {}

        _str = [[f'{ds}_{name}' for name in dump_str] for ds in cfg.damage_states]
        dmg.columns = [x for z in zip(*_str) for x in z]

        # collapse_adj
        collapse_adj = {x['tower']: x['collapse_adj'] for x in results
                        if (x['event']==event.id) and (x['line']==line.linename)}

        pc_adj_agg = np.zeros((line.no_towers, line.no_towers, dmg.shape[0]))

        # prob of collapse
        for k, name in enumerate(line.names):

            # adjust tower dmg_time_idx to line 
            for idl, prob in collapse_adj[name].items():

                new = dmg[f'collapse_{name}'].copy()
                try:
                    new.loc[~new.isna()] = prob
                except ValueError:
                    logger.warning(f'different length provided for {tower_name}')
                else:
                    pc_adj_agg[k, idl] = new.fillna(0.0)

            pc_adj_agg[k, k] = dmg[f'collapse_{name}'].fillna(0.0).values

        # pc_collapse.shape == (no_tower, no_time)
        pc_collapse = 1.0 - np.prod(1 - pc_adj_agg, axis=0)

        damage_prob['collapse'] = pd.DataFrame(pc_collapse.T,
                columns=line.names, index=dmg.index)

        # non_collapse state
        for ds in cfg.non_collapse:

            temp = np.zeros_like(pc_collapse)

            for k, name in enumerate(line.names):

                dmg_ds = dmg[f'{ds}_{name}'].fillna(0.0).values
                dmg_collapse = dmg[f'collapse_{name}'].fillna(0.0).values

                if len(dmg_ds):

                    # P(DS>ds) - P(collapse directly) + P(collapse induced)
                    try:
                        value = dmg_ds - dmg_collapse + pc_collapse[k]
                    except ValueError:
                        self.logger.critical(f'PE of {ds} of {name} can not be used, {dmg_ds.shape} and {dmg_collapse.shape} is incompatible with {pc_collapse.shape}')
                    else:
                        temp[k] = np.where(value > 1.0, [1.0], value)

                else:
                    temp[k] = pc_collapse[k]

            damage_prob[ds] = pd.DataFrame(temp.T,
                    columns=line.names,
                    index=dmg.index)

        return damage_prob



def compute_damage_prob_sim(event, line, cfg, results, dmg=None):

    if dmg is None:
        dump_str, dump_df = zip(*[(x['tower'], x['dmg']) for x in results
                              if (x['event']==event.id) and (x['line']==line.linename)])
        dmg = pd.concat(dump_df, join='outer', axis=1)

        if not dmg.empty:
            _str = [[f'{ds}_{name}' for name in dump_str] for ds in cfg.damage_states]
            dmg.columns = [x for z in zip(*_str) for x in z]

    if dmg.empty:
        return None, None, None

    else:

        # dmg_state_sim
        dmg_state_sim = {x['tower']: x['dmg_state_sim'] for x in results
                        if (x['event']==event.id) and (x['line']==line.linename)}

        # collapse_adj_sim
        collapse_adj_sim = {x['tower']: x['collapse_adj_sim'] for x in results
                        if (x['event']==event.id) and (x['line']==line.linename)}

        # perfect correlation within a single line
        damage_prob_sim = {}
        #self.dmg_idx = {}  # tuple of idl, id_sim, id_time

        #idx_by_ds = {ds: None for ds in self.damage_states}
        dic_tf_ds = {ds: None for ds in cfg.damage_states}
        tf_ds = np.zeros((line.no_towers, cfg.no_sims, dmg.shape[0]), dtype=bool)

        # collapse by adjacent towers
        for name in line.names:

            try:
                collapse_adj_sim[name]['id_adj']
            except KeyError:
                pass
            else:
                # adjust time index from tower to line
                _ = (~dmg[f'collapse_{name}'].isna()).idxmax()
                ida = dmg[f'collapse_{name}'].index.get_loc(_)

                for id_adj, grp in collapse_adj_sim[name].groupby('id_adj'):
                    for idl in id_adj:
                        tf_ds[idl, grp['id_sim'].values, grp['id_time'].values + ida] = True

            #else:
            #    print(f'collapse_adj_sim of {self.name} for {self.towers[k].name},{self.towers[k].idl} is None')
        # append damage state by direct wind
        for ds in cfg.damage_states[::-1]:  # collapse first

            for k, name in enumerate(line.names):

                # adjust time index from tower to line
                _ = (~dmg[f'collapse_{name}'].isna()).idxmax()
                ida = dmg[f'collapse_{name}'].index.get_loc(_)

                #dmg_state_sim = self.towers[k].dmg_state_sim[ds].loc[idx].copy()

                tf_ds[k, dmg_state_sim[name][ds]['id_sim'].values,
                      dmg_state_sim[name][ds]['id_time'].values + ida] = True

            # dmg index for line interaction
            #self.dmg_idx[ds] = np.where(tf_ds)

            # PE(DS)
            damage_prob_sim[ds] = pd.DataFrame(tf_ds.sum(axis=1).T / cfg.no_sims,
                columns=line.names, index=dmg.index)

            dic_tf_ds[ds] = tf_ds.copy()
            #idx = np.where(tf_ds)
            #idx_by_ds[ds] = [(id_time, id_sim, id_tower) for
            #                 id_tower, id_sim, id_time in zip(*idx)]

        # compute mean, std of no. of damaged towers
        no_damage, prob_no_damage = compute_stats(dic_tf_ds, cfg, line.no_towers, dmg.index)

        # checking against analytical value
        #for name in self.names:
        #    try:
        #        np.testing.assert_allclose(self.damage_prob['collapse'][name].values,
        #            self.damage_prob_sim['collapse'][name].values, atol=self.atol, rtol=self.rtol)
        #    except AssertionError:
        #        self.logger.warning(f'Simulation results of {name}:collapse are not close to the analytical')
        return damage_prob_sim, no_damage, prob_no_damage


def compute_damage_prob_sim_wo_cascading(event, line, cfg, results = None, dmg=None, dmg_state_sim=None):

    if results is None:
        assert dmg is not None, logger.warning('Either dmg or results should be provided')
        assert dmg_state_sim is not None, logger.warning('Either dmg_state_sim or results should be provided')

    if dmg is None:
        dump_str, dump_df = zip(*[(x['tower'], x['dmg']) for x in results
                              if (x['event']==event.id) and (x['line']==line.linename)])
        dmg = pd.concat(dump_df, join='outer', axis=1)

        if not dmg.empty:
            _str = [[f'{ds}_{name}' for name in dump_str] for ds in cfg.damage_states]
            dmg.columns = [x for z in zip(*_str) for x in z]

    if dmg_state_sim is None:
        # dmg_state_sim
        dmg_state_sim = {x['tower']: x['dmg_state_sim'] for x in results
                        if (x['event']==event.id) and (x['line']==line.linename)}

    if dmg.empty:
        return None, None, None
    else:

        damage_prob_sim_wo_cascading = {}

        dic_tf_ds = {ds: None for ds in cfg.damage_states}

        tf_ds = np.zeros((line.no_towers, cfg.no_sims, dmg.shape[0]), dtype=bool)

        for ds in cfg.damage_states[::-1]:  # collapse first

            for k, name in enumerate(line.names):

                # adjust time index from tower to line
                _ = (~dmg[f'collapse_{name}'].isna()).idxmax()
                ida = dmg[f'collapse_{name}'].index.get_loc(_)

                tf_ds[k, dmg_state_sim[name][ds]['id_sim'].values,
                      dmg_state_sim[name][ds]['id_time'].values + ida] = True

            # PE(DS)
            damage_prob_sim_wo_cascading[ds] = pd.DataFrame(
                tf_ds.sum(axis=1).T / cfg.no_sims,
                columns=line.names,
                index=dmg.index)

            dic_tf_ds[ds] = tf_ds.copy()

        no_damage_wo_cascading, prob_no_damage_wo_cascading = compute_stats(dic_tf_ds, cfg, line.no_towers, dmg.index)

        # checking against analytical value for non collapse damage states
        #for k in self.dmg_towers:
        #    name = self.towers[k].name
        #    idt0, idt1 = [x - self.dmg_time_idx[0] for x in self.towers[k].dmg_time_idx]
        #    try:
        #        np.testing.assert_allclose(self.towers[k].dmg[ds].values,
        #                self.damage_prob_sim_no_cascading[ds].iloc[idt0:idt1][name].values, atol=self.atol, rtol=self.rtol)
        #    except AssertionError:
        #        self.logger.warning(f'Simulation results of {name}:{ds} are not close to the analytical')
    return damage_prob_sim_wo_cascading, no_damage_wo_cascading, prob_no_damage_wo_cascading













class Line(object):
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
                  'damage_states',
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
        self.damage_states = None
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
        self.damage_prob = None

        # simulation method
        self.damage_prob_sim = None
        self.no_damage = None
        self.prob_no_damage = None
        self.dmg_idx = None

        # no cascading collapse
        self.damage_prob_sim_no_cascading = None
        self.no_damage_no_cascading = None
        self.prob_no_damage_no_cascading = None

        # line interaction
        self._dmg_idx_interaction = None
        self.damage_prob_interaction = None
        self.prob_no_damage_interaction = None
        self.no_damage_interaction = None

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
                     damage_prob,
                     damage_prob_sim, no_damage, prob_no_damage,
                     damage_prob_sim_wo_cascading, no_damage_wo_cascading, prob_no_damage_wo_cascading):

        items = ['damage_prob', 'damage_prob_sim', 'damage_prob_sim_wo_cascading',
                 'no_damage', 'no_damage_wo_cascading',
                 'prob_no_damage', 'prob_no_damage_wo_cascading']

        columns_by_item = {'damage_prob': self.names,
                           'damage_prob_sim': self.names,
                           'damage_prob_sim_no_cascading': self.names,
                           'no_damage': ['mean', 'std'],
                           'no_damage_no_cascading': ['mean', 'std'],
                           'prob_no_damage': range(self.no_towers + 1),
                           'prob_no_damage_no_cascading': range(self.no_towers + 1)
                           }

        if self.no_time:

            if not os.path.exists(self.path_output):
                os.makedirs(self.path_output)
                self.logger.info(f'{self.path_output} is created')

            # save no_damage and prob_no_damage csv
            try:
                idt_max = self.no_damage['collapse']['mean'].idxmax()
            except TypeError:
                pass
            else:
                # save no_damage to csv
                self.write_csv_output(idt_max, 'no_damage', self.no_damage)

                # sve prob_no_damage to csv
                self.write_csv_output(idt_max, 'prob_no_damage', self.prob_no_damage)

                # save damage_prob to csv
                _file = self.file_output + f'_damage_prob.csv'
                df = pd.DataFrame(None)
                for k, v in self.damage_prob.items():
                    df[k] = v.loc[idt_max]
                for k, v in self.damage_prob_sim.items():
                    df[f'{k}_sim'] = v.loc[idt_max]
                df.to_csv(_file)
                self.logger.info(f'{_file} is saved')

            # save no_damage_non_cascading and prob_no_damage_non_cascading csv
            try:
                idt_max = self.no_damage_no_cascading['collapse']['mean'].idxmax()
            except TypeError:
                pass
            else:
                # save no_damage to csv
                self.write_csv_output(idt_max, 'no_damage_no_cascading', self.no_damage_no_cascading)
                # sve prob_no_damage to csv
                self.write_csv_output(idt_max, 'prob_no_damage_no_cascading', self.prob_no_damage_no_cascading)

            _file = self.file_output + '.h5'
            with h5py.File(_file, 'w') as hf:

                for item in items:

                    group = hf.create_group(item)

                    for ds in self.damage_states:

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
    line.compute_damage_prob()

    # perfect correlation within a single line
    if cfg.options['run_simulation']:
        line.compute_damage_prob_sim()

        if cfg.options['run_no_cascading_collapse']:
            line.compute_damage_prob_sim_no_cascading()

    # compare simulation against analytical
    #for ds in cfg.damage_states:
    #    idx_not_close = np.where(~np.isclose(line.damage_prob_sim[ds].values,
    #                             line.damage_prob[ds].values,
    #                             atol=ATOL,
    #                             rtol=RTOL))
    #    for idc in idx_not_close[1]:
    #        logger.warning(f'Simulation not CLOSE {ds}:{line.towers[idc].name}')

    # save
    if cfg.options['save_output']:
        line.write_output()

    return line
'''

