'''
read
    - collapse fragility
    - conditional proability
    - shape file
    - wind velocity profile

'''

import pandas as pd
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from geopy.distance import great_circle
from tower import Tower
from event import Event
import shapefile


def read_topo_value(file_):
    """read topograhpic multipler value from the input file
    :param file_: input file
    :type file_: string
    :returns: topography value at each site
    :rtype: dict
    """
    names_str = ['Name', '', '', '', '', '', '', '', '', 'Mh', 'Mhopp']
    data = pd.read_csv(file_, usecols=[0, 9, 10], skiprows=1,
                       names=names_str)
    data['topo'] = data[['Mh', 'Mhopp']].max(axis=1)
    return data.set_index('Name').to_dict()['topo']


def read_design_value(file_):
    """read design values by line
    """
    data = pd.read_csv(file_, skipinitialspace=True, skiprows=1,
                       names=['lineroute', 'speed', 'span', 'cat', 'level'],
                       index_col=0)

    design_value = data.transpose().to_dict()
    return design_value.keys(), design_value

def read_adjust_by_topo(file_):
    """read topographic multipler for design speed adjustment
    :param file_: input file
    :type file_: string

    :returns: design speed adjustment factor
    :rtype: dict

    """

    # design speed adjustment factor
    temp = pd.read_csv(file_, skiprows=2)
    data = temp.set_index('key').to_dict()['value']

    # threshold
    temp = pd.read_csv(file_, skiprows=1)
    data['threshold'] = np.array([float(x) for x in temp.columns])

    assert len(data['threshold']) == len(data.keys()) - 2
    return data

class TransmissionNetwork(object):
    """
    Tranmission Network which contains towers and GIS information
    """

    def __init__(self, conf):
        self.conf = conf

    def read_tower_gis_information(self, flag_figure=None):
        """
        read geospational information of towers

        Usage:
        read_tower_gis_information(file_shape_tower, file_shape_line,
            flag_figure=None, flag_load = 1, sel_lines = None)

        :returns
            tower = Dictionary of towers
            sel_lines =  line names provided in the input file.
            TODO: Hyeuk please describe: fid_by_line, fid2name, lon, lat
        """
        shapes_tower, records_tower, fields_tower =\
            read_shape_file(self.conf.file_shape_tower)
        shapes_line, records_line, fields_line =\
            read_shape_file(self.conf.file_shape_line)

        sel_lines, design_value = read_design_value(self.conf.file_design_value)

        if self.conf.file_topo_value:
            topo_value = read_topo_value(self.conf.file_topo_value)
            adjust_by_topo = read_adjust_by_topo(self.conf.file_adjust_by_topo)

        sel_keys = ['Type', 'Name', 'Latitude', 'Longitude', 'DevAngle',
                    'AxisAz', 'Mun', 'Barangay', 'ConstType', 'Function',
                    'LineRoute', 'Height']

        sel_idx = {str_: get_field_index(fields_tower, str_)
                   for str_ in sel_keys}

        # processing shp file
        tower, fid2name = {}, {}
        fid, line_route, lat, lon = [], [], [], []
        fid_ = 0
        for item in records_tower:

            line_route_ = item[sel_idx['LineRoute']]

            if line_route_ in sel_lines:

                name_ = item[sel_idx['Name']]
                lat_ = item[sel_idx['Latitude']]
                lon_ = item[sel_idx['Longitude']]

                if self.conf.file_topo_value:
                    idx_topo = np.sum(topo_value[name_] >=
                                      adjust_by_topo['threshold'])
                    design_speed = design_value[line_route_]['speed'] *\
                        adjust_by_topo[idx_topo]
                else:
                    design_speed = design_value[line_route_]['speed']

                tower[name_] = Tower(self.conf, line_route_, design_speed,
                                     design_value, sel_idx, item)
                fid2name[fid_] = name_
                fid.append(fid_)
                line_route.append(line_route_)
                lat.append(lat_)
                lon.append(lon_)
                fid_ += 1

        fid = np.array(fid)
        line_route = np.array(line_route)
        lat = np.array(lat, dtype=float)
        lon = np.array(lon, dtype=float)

        lineroute_line = get_data_per_polygon(records_line, fields_line,
                                              'LineRoute')

        # generate connectivity for each of line routes
        fid_by_line = {}

        for line in sel_lines:

            # pts
            tf = (line_route == line)
            fid_pt = fid[tf]
            xy_pt = np.array([lon[tf], lat[tf]], dtype=float).T

            # need to change once line shape file is corrected
            i = lineroute_line.index(line)
            xy_line = np.array(shapes_line[i].points)

            idx_sorted = []
            for item in xy_line:

                diff = xy_pt - np.ones((xy_pt.shape[0], 1))*item[np.newaxis, :]
                temp = diff[:, 0]*diff[:, 0]+diff[:, 1]*diff[:, 1]
                idx = np.argmin(temp)
                tf_ = np.allclose(temp[idx], 0.0, 1.0e-4)
                if tf_:
                    idx_sorted.append(fid_pt[idx])
                else:
                    print 'Something wrong in {}, {}, or {}'.format(line, item,
                                                                    xy_pt[idx])

            fid_by_line[line] = idx_sorted

            # calculate distance between towers
            ntower_sel = sum(tf)
            dist_forward = []

            for i in range(ntower_sel-1):

                j0 = fid_by_line[line][i]
                j1 = fid_by_line[line][i+1]
                pt0 = (lat[j0], lon[j0])
                pt1 = (lat[j1], lon[j1])

                dist_forward.append(great_circle(pt0, pt1).meters)

            if flag_figure:
                pt_x, pt_y = [], []
                for k in idx_sorted:
                    pt_x.append(lon[fid == k])
                    pt_y.append(lat[fid == k])

                plt.figure()
                plt.plot(xy_line[:, 0], xy_line[:, 1], 'ro-', pt_x, pt_y, 'b-')
                plt.title(line)

            # assign adj, actual_span, adj_design_speed
            idx_sorted = np.insert(idx_sorted, 0, -1)  # start
            idx_sorted = np.append(idx_sorted, -1)  # end

            for j in range(1, ntower_sel+1):
                name_ = fid2name[idx_sorted[j]]
                tower[name_].adj = (idx_sorted[j-1], idx_sorted[j+1])

                # compute actual wind span
                if j == 1:
                    val = 0.5*dist_forward[0]
                elif j == ntower_sel:
                    val = 0.5*dist_forward[ntower_sel-2]
                else:
                    val = 0.5*(dist_forward[j-2]+dist_forward[j-1])

                tower[name_].actual_span = val
                tower[name_].calc_adj_collapse_wind_speed()

        return tower, sel_lines, fid_by_line, fid2name, lon, lat


def read_shape_file(file_shape):
    """
    read shape file
    """
    sf = shapefile.Reader(file_shape)
    shapes = sf.shapes()
    records = sf.records()
    fields = sf.fields
    fields = fields[1:]

    return shapes, records, fields


def get_field_index(fields, key_string):
    """
    retrieve field index
    """
    f = False
    j = np.NaN
    for j, f in enumerate(fields):
        if f[0] == key_string:
            break
    if f[0] == key_string:
        return j


def get_data_per_polygon(records, fields, key_string):
    """
    retrieve field data
    """
    return map(lambda y: y[get_field_index(fields, key_string)], records)


def read_velocity_profile(conf, tower):
    """
    read velocity time history at each tower location

    Usage:
    read_velocity_profile(Wind, dir_wind_timeseries, tower)
    :return
     event: dictionary of event class instances
    """
    event = dict()  # dictionary of event class instances

    for name in tower:
        file_name = 'ts.' + name + '.csv'
        vel_file = os.path.join(conf.dir_wind_timeseries, file_name)
        try:
            event[name] = Event(tower[name], vel_file)

        except IOError:
            import inspect
            print 'File not found:', vel_file
            return {'error': 'File {vel_file} not found in function {func}'
                    .format(vel_file=vel_file, func=inspect.stack()[0][3])}
        except Exception as e:
            import inspect
            print e
            return {'error': 'Something went wrong in {func}.'
                    .format(func=inspect.stack()[0][3])}

    return event


if __name__ == '__main__':
    from config_class import TransmissionConfig
    conf = TransmissionConfig()
