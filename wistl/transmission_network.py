from __future__ import print_function

import os
import shapefile
import pandas as pd
import numpy as np
import geopy.distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shapely.geometry import LineString, Point
from geopy.distance import great_circle

from wistl.transmission_line import TransmissionLine


def create_damaged_network(conf):
    """ create dict of transmission network
    :param conf: instance of config class
    :return: dictionary of damaged_networks
    """

    damaged_networks = dict()

    for event_id, scale_list in conf.scale.iteritems():

        for scale in scale_list:

            event_id_scale = conf.event_id_scale_str.format(event_id=event_id,
                                                            scale=scale)

            if event_id_scale in damaged_networks:
                msg = '{} is already assigned'.format(event_id_scale)
                raise KeyError(msg)

            path_output_scenario = os.path.join(conf.path_output,
                                                event_id_scale)

            if not os.path.exists(path_output_scenario) and conf.save:
                os.makedirs(path_output_scenario)
                print('{} is created'.format(path_output_scenario))

            damaged_networks[event_id_scale] = TransmissionNetwork(conf)
            damaged_networks[event_id_scale].event_tuple = (event_id, scale)

    return damaged_networks


class TransmissionNetwork(object):
    """ class for a collection of wistl lines"""

    def __init__(self, conf):

        self.conf = conf

        self._event_tuple = None
        self.event_id_scale = None
        self.event_id = None
        self.scale = None
        self.path_event = None
        self.time_index = None

        self.df_towers = read_shape_file(self.conf.file_shape_tower)
        self.df_towers = populate_df_towers(self.df_towers, self.conf)

        self.df_lines = read_shape_file(self.conf.file_shape_line)
        self.df_lines = populate_df_lines(self.df_lines)

        self.lines = dict()
        for line_name, grouped in self.df_towers.groupby('LineRoute'):

            if line_name in self.conf.selected_lines:
                try:
                    idx = self.df_lines[self.df_lines.LineRoute ==
                                        line_name].index[0]
                except IndexError:
                    msg = '{} not in the line shapefile'.format(line_name)
                    raise IndexError(msg)

                self.lines[line_name] = TransmissionLine(
                    conf=self.conf,
                    df_towers=grouped,
                    ps_line=self.df_lines.loc[idx])

        if conf.line_interaction:
            self._set_line_interaction()

            if conf.figure:
                self._plot_line_interaction()

    @property
    def event_tuple(self):
        return self._event_tuple

    @event_tuple.setter
    def event_tuple(self, value):
        try:
            event_id, scale = value
        except ValueError:
            raise ValueError("Pass a tuple of event_id and scale")
        else:
            self.event_id = event_id
            self.scale = scale
            self.path_event = os.path.join(self.conf.path_wind_scenario_base,
                                           event_id)
            self.event_id_scale = self.conf.event_id_scale_str.format(
                event_id=event_id, scale=scale)
            self._set_damage_line()

    def _set_damage_line(self):
        # assign event information to instances of TransmissionLine

        line = None
        for line in self.lines.itervalues():
            line.event_tuple = (self.event_id, self.scale)

        # assuming same time index for each tower in the same network
        self.time_index = line.time_index

    def _set_line_interaction(self):

        for line_name, line in self.lines.iteritems():

            for target_line in self.conf.line_interaction[line_name]:

                line_string = self.lines[target_line].line_string
                line_coord = self.lines[target_line].coord

                for tower in line.towers.itervalues():

                    closest_pt_on_line = line_string.interpolate(
                        line_string.project(tower.point))

                    closest_pt_coord = np.array(closest_pt_on_line.coords)

                    closest_pt_lat_lon = closest_pt_coord[:, ::-1]

                    # compute distance
                    dist_from_line = geopy.distance.great_circle(
                        tower.coord_lat_lon, closest_pt_lat_lon).meters

                    if dist_from_line < tower.height:

                        id_pt = find_id_nearest_pt(closest_pt_coord,
                                                   line_coord)

                        tower.id_on_target_line[target_line] = \
                            self.lines[target_line].id_by_line[id_pt]

    def _plot_line_interaction(self):

        for line_name, line in self.lines.iteritems():

            plt.figure()
            plt.plot(line.coord[:, 0],
                     line.coord[:, 1], '-', label=line_name)

            for target_line in self.conf.line_interaction[line_name]:

                plt.plot(self.lines[target_line].coord[:, 0],
                         self.lines[target_line].coord[:, 1],
                         '--', label=target_line)

                for tower in line.towers.itervalues():

                    try:
                        id_pt = tower.id_on_target_line[target_line]
                    except KeyError:
                        plt.plot(tower.coord[0], tower.coord[1], 'ko')
                    else:
                        target_tower_name = self.lines[
                            target_line].name_by_line[id_pt]
                        target_tower = self.lines[target_line].towers[
                            target_tower_name]

                        plt.plot([tower.coord[0], target_tower.coord[0]],
                                 [tower.coord[1], target_tower.coord[1]],
                                 'ro-',
                                 label='{}->{}'.format(tower.name,
                                                       target_tower_name))

            plt.title(line_name)
            plt.legend(loc=0)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            png_file = os.path.join(self.conf.path_output,
                                    'line_interaction_{}.png'.format(line_name))
            plt.savefig(png_file)
            plt.close()


def populate_df_lines(df_lines):
    """
    add columns to df_lines
    :param df_lines:
    :return:
    """
    # coord, coord_lat_lon, line_string
    df_lines['coord'] = df_lines['Shapes'].apply(lambda x: x.points)
    df_lines['coord_lat_lon'] = df_lines['Shapes'].apply(
        lambda x: np.array(x.points)[:, :-1].tolist())
    df_lines['line_string'] = df_lines['Shapes'].apply(
        lambda x: LineString(x.points))

    # name_output
    df_lines['name_output'] = df_lines['LineRoute'].apply(
        lambda x: '_'.join(x for x in x.split() if x.isalnum()))

    df_lines['actual_span'] = df_lines['coord_lat_lon'].apply(
        calculate_distance_between_towers)
    return df_lines


def populate_df_towers(df_towers, conf):
    """
    create columns of tower attributes: coord, coord_lat_lon, Point,
    design_span, design_level, design_speed, terrain_cat, no_circuit,
    file_wind_base_name, frag_func, frag_scale, frag_arg, cond_pc,
    max_no_adj_towers, height_z, file_wind_base_name,

    :param df_towers:
    :param conf:
    :return:
    """

    # coord, coord_lat_lon, Point
    df_towers['coord'] = df_towers['Shapes'].apply(lambda x: x.points[0])
    df_towers['coord_lat_lon'] = df_towers['Shapes'].apply(
        lambda x: np.array(x.points[0])[::-1].tolist())
    df_towers['point'] = df_towers['Shapes'].apply(lambda x: Point(x.points[0]))

    # design_span, design_level, design_speed, terrain_cat
    df_towers['design_span'] = df_towers['LineRoute'].apply(
        lambda x: conf.design_value[x]['span'])
    df_towers['design_level'] = df_towers['LineRoute'].apply(
        lambda x: conf.design_value[x]['level'])
    df_towers['design_speed'] = df_towers['LineRoute'].apply(
        lambda x: conf.design_value[x]['speed'])
    df_towers['terrain_cat'] = df_towers['LineRoute'].apply(
        lambda x: conf.design_value[x]['cat'])

    # adjust design speed
    if conf.adjust_design_by_topography:
        df_towers['design_speed'] *= df_towers['Name'].apply(
            adjust_design_speed, args=(conf,))

    # height_z
    df_towers['height_z'] = df_towers['Function'].apply(
        lambda x: conf.drag_height[x])

    # file_wind_base_name
    df_towers['file_wind_base_name'] = df_towers['Name'].apply(
        lambda x: conf.wind_file_head + x + conf.wind_file_tail)
    df_towers['no_circuit'] = 2  # double circuit for all towers

    # frag_func, frag_scale, frag_arg
    df_towers = df_towers.merge(df_towers.apply(assign_fragility_parameters,
                                                args=(conf,), axis=1),
                                left_index=True, right_index=True)

    # cond_pc, max_no_adj_towers
    df_towers = df_towers.merge(df_towers.apply(assign_cond_collapse_prob,
                                                args=(conf,), axis=1),
                                left_index=True, right_index=True)

    df_towers['factor_10_to_z'] = df_towers.apply(convert_10_to_z, args=(conf,),
                                                  axis=1)

    return df_towers


def mc_loop_over_line(damage_line):
    """
    mc simulation over transmission line
    :param damage_line: instance of transmission line
    :return: None but update attributes of
    """

    event_id = damage_line.event_id
    line_name = damage_line.name

    if damage_line.conf.random_seed:
        try:
            seed = damage_line.conf.seed[event_id][line_name]
        except KeyError:
            msg = '{}:{} is undefined. Check the config file'.format(
                event_id, line_name)
            raise KeyError(msg)
    else:
        seed = None

    # perfect correlation within a single line
    damage_line.compute_damage_probability_simulation(seed)

    if not damage_line.conf.skip_non_cascading_collapse:
        damage_line.compute_damage_probability_simulation_non_cascading()

    try:
        damage_line.conf.line_interaction[line_name]
    except (TypeError, KeyError):
        pass
    else:
        damage_line.compute_damage_probability_simulation_line_interaction()


# def assign_design_values(line_route, conf):
#     """
#     assign design values
#     :param line_route:
#     :param conf:
#     :return: pandas series
#     """


def assign_cond_collapse_prob(ps_tower, conf):
    """ get dict of conditional collapse probabilities
    :return: cond_pc, max_no_adj_towers
    """

    ps_cond = pd.Series({'cond_pc': dict(),
                        'max_no_adj_towers': None})

    att_value = ps_tower[conf.cond_collapse_prob_metadata['by']]
    df_prob = conf.cond_collapse_prob[att_value]

    att = conf.cond_collapse_prob_metadata[att_value]['by']
    att_type = conf.cond_collapse_prob_metadata[att_value]['type']
    tf_array = None
    if att_type == 'string':
        tf_array = df_prob[att] == ps_tower[att]
    elif att_type == 'numeric':
        tf_array = (df_prob[att + '_lower'] <= ps_tower[att]) & \
                   (df_prob[att + '_upper'] > ps_tower[att])

    # change to dictionary
    ps_cond['cond_pc'] = dict(zip(df_prob.loc[tf_array, 'list'],
                                  df_prob.loc[tf_array, 'probability']))
    ps_cond['max_no_adj_towers'] = \
        conf.cond_collapse_prob_metadata[att_value]['max_adj']

    return ps_cond


def convert_10_to_z(ps_tower, conf):
    """
    compute Mz,cat(h=z) / Mz,cat(h=10)
    :param ps_tower:
    :param conf:
    :return: ratio of Mz,cat(z) to Mz,cat(10)
    """
    try:
        terrain_cat = 'tc' + str(ps_tower.terrain_cat)
    except AttributeError:
        msg = 'attribute:{} is not found'.format('terrain_cat')
        raise AttributeError(msg)

    try:
        terrain_multiplier_z = np.interp(
            ps_tower.height_z,
            conf.terrain_multiplier['height'],
            conf.terrain_multiplier[terrain_cat])
    except KeyError:
        msg = '{} is undefined in {}'.format(
            terrain_cat, conf.file_terrain_multiplier)
        raise KeyError(msg)
    except AttributeError:
        msg = 'attribute:{} is not found'.format('height_z')
        raise AttributeError(msg)

    id10 = conf.terrain_multiplier['height'].index(10)
    terrain_multiplier_10 = conf.terrain_multiplier[terrain_cat][id10]
    return terrain_multiplier_z / terrain_multiplier_10


# def assign_shapely_point(x):
#     """
#     assign shapely point
#     :param shape: shapefile._Shape instance
#     :return: pd.Series('coord': [lon, lat],
#                        'coord_lat_lon': [lat, lon],
#                        'point': Shapely.Point instance)
#     """
#     coord = shape.points[0]
#     return pd.Series({'coord': coord,
#                       'coord_lat_lon': np.array(coord)[::-1].tolist(),
#                       'point': Point(coord)})


# def assign_shapely_line(shape):
#     """
#     add pandas series data
#     :param shape: shapefile._Shape instance
#     :return: pd.Series('coord': list of [lon, lat],
#                        'coord_lat_lon': list of [lat, lon],
#                        'line_string': Shapely.LineString instance)
#     """
#     coord = shape.points
#     return pd.Series({'coord': coord,
#                       'coord_lat_lon': np.array(coord)[:, ::-1].tolist(),
#                       'line_string': LineString(coord)})


def assign_fragility_parameters(ps_tower, conf):
    """

    :param ps_tower:
    :param conf:
    :return:
    """
    tf_array = np.ones((conf.fragility.shape[0],), dtype=bool)
    for att, att_type in zip(conf.fragility_metadata['by'],
                             conf.fragility_metadata['type']):
        if att_type == 'string':
            try:
                tf_array *= conf.fragility[att] == ps_tower[att]
            except AttributeError:
                msg = 'attribute:{} is not defined'.format(att)
                raise AttributeError(msg)
        elif att_type == 'numeric':
            try:
                tf_array *= (conf.fragility[att + '_lower'] <= ps_tower[att]) &\
                            (conf.fragility[att + '_upper'] > ps_tower[att])
            except AttributeError:
                msg = 'attribute:{} is not defined'.format(att)
                raise AttributeError(msg)
        else:
            msg = 'attrib. type should be either string or numeric'
            raise TypeError(msg)

    ps_frag = pd.Series({'frag_scale': dict(), 'frag_arg': dict(),
                        'frag_func': None})
    for ds in conf.damage_states:
        idx = tf_array & (conf.fragility['limit_states'] == ds)
        ps_selected = conf.fragility.loc[idx]
        assert sum(idx) == 1
        ps_frag['frag_func'] = ps_selected[
            conf.fragility_metadata['function']].values[0]
        frag_func = conf.fragility_metadata[ps_frag['frag_func']]
        ps_frag['frag_scale'][ds] = ps_selected[frag_func['scale']].values[0]
        ps_frag['frag_arg'][ds] = ps_selected[frag_func['arg']].values[0]
    return ps_frag


def adjust_design_speed(tower_name, conf):
    """
    compute design adjustment factor based on topography
    :param tower_name: tower name
    :param conf:
    :return:
    """
    id_topo = sum(conf.topo_multiplier[tower_name] >=
                  conf.design_adjustment_factor_by_topo['threshold'])
    return conf.design_adjustment_factor_by_topo[id_topo]


def calculate_distance_between_towers(coord_lat_lon):
    """ calculate actual span between the towers """
    coord_lat_lon = np.stack(coord_lat_lon)
    dist_forward = np.zeros(len(coord_lat_lon) - 1)
    for i, (pt0, pt1) in enumerate(zip(coord_lat_lon[0:-1],
                                       coord_lat_lon[1:])):
        dist_forward[i] = great_circle(pt0, pt1).meters

    actual_span = 0.5 * (dist_forward[0:-1] + dist_forward[1:])
    actual_span = np.insert(actual_span, 0, [0.5 * dist_forward[0]])
    actual_span = np.append(actual_span, [0.5 * dist_forward[-1]])
    return actual_span


def find_id_nearest_pt(pt_coord, line_coord):
    """
    :param pt_coord: (,1)
    :param line_coord: (,2)
    :return:
    """
    assert pt_coord.shape[1] == 2
    assert line_coord.shape[1] == 2
    diff = np.linalg.norm(line_coord - pt_coord, axis=1)

    return np.argmin(diff)


def read_shape_file(file_shape):
    """
    read shape file and return data frame
    :param file_shape:
    :return data_frame:
    """
    sf = shapefile.Reader(file_shape)
    shapes = sf.shapes()
    records = sf.records()
    fields = [x[0] for x in sf.fields[1:]]
    fields_type = [x[1] for x in sf.fields[1:]]

    shapefile_type = {'C': object, 'F': np.float64, 'N': np.int64}

    data_frame = pd.DataFrame(records, columns=fields)

    for name_, type_ in zip(data_frame.columns, fields_type):
        if data_frame[name_].dtype != shapefile_type[type_]:
            data_frame[name_] = data_frame[name_].astype(shapefile_type[type_])

    if 'Shapes' in fields:
        raise KeyError('Shapes is already in the fields')
    else:
        data_frame['Shapes'] = shapes

    return data_frame
