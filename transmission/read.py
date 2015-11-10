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
from StringIO import StringIO
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from geopy.distance import great_circle

from tower import Tower
from event import Event
import shapefile


def read_topo_value(file_):
    '''
    read topograhpic multipler value 
    '''
    data = pd.read_csv(file_, header=0, usecols=[0,9,10],
        names=['Name', '', '', '', '', '', '', '', '', 'Mh', 'Mhopp'])
    data['topo'] = np.max([data['Mh'].values, data['Mhopp'].values], axis=0)

    val = {}
    for item in data['Name']:
        val[item] = data[data['Name']==item]['topo'].values[0]
    return val


def read_design_value(file_):
    """read design values by line
    """
    data = pd.read_csv(file_, skipinitialspace=1)
    design_value = {}
    for line in data.iterrows():
        lineroute_, speed_, span_, cat_, level_ = [ line[1][x] for x in 
        ['lineroute', 'design wind speed', 'design wind span', 'terrain category', 'design level']]
        design_value.setdefault(lineroute_, {})['speed'] = speed_
        design_value.setdefault(lineroute_, {})['span'] = span_
        design_value.setdefault(lineroute_, {})['cat'] = cat_
        design_value.setdefault(lineroute_, {})['level'] = level_

    sel_lines = design_value.keys()
    return (sel_lines, design_value)


def distance(origin, destination):
    # origin, desttination (lat, lon) tuple
    # distance in km 
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371.0  # km
 
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c

    return d


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
        read_tower_gis_information(shape_file_tower, shape_file_line,
            flag_figure=None, flag_load = 1, sel_lines = None)

        :returns
            tower = Dictionary of towers
            sel_lines =  line names provided in the input file.
            TODO: Hyeuk please describe: fid_by_line, fid2name, lon, lat
        """
        shapes_tower, records_tower, fields_tower = read_shape_file(self.conf.shape_file_tower)
        shapes_line, records_line, fields_line = read_shape_file(self.conf.shape_file_line)

        sel_lines, design_value = read_design_value(self.conf.file_design_value)

        if self.conf.file_topo_value:
            topo_value = read_topo_value(self.conf.file_topo_value)
            topo_dic = {'threshold': np.array([1.05, 1.1, 1.2, 1.3, 1.45]), 0: 1.0, 1: 1.1, 2: 1.2,
                        3: 1.3, 4: 1.45, 5: 1.6}

        sel_keys = ['Type', 'Name', 'Latitude', 'Longitude', 'DevAngle',
                    'AxisAz', 'Mun', 'Barangay', 'ConstType', 'Function', 'LineRoute', 'Height']

        sel_idx = {str_: get_field_index(fields_tower, str_) for str_ in sel_keys}

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
                    idx_topo = np.sum(topo_value[name_] >= topo_dic['threshold'])
                    design_speed = design_value[line_route_]['speed']*topo_dic[idx_topo]
                else:
                    design_speed = design_value[line_route_]['speed']

                tower[name_] = Tower(self.conf, line_route_, design_speed, design_value, sel_idx, item)

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

        lineroute_line = get_data_per_polygon(records_line, fields_line, 'LineRoute')

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
                    print 'Something wrong in {}, {}, or {}'.format(line, item, xy_pt[idx])

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


def check_shape_files_tower_line(shape_file_tower, shape_file_line):

    """
    check consistency of shape files of tower and line
    Not used at the moment?
    """

    (shapes_tower, records_tower, fields_tower) = read_shape_file(shape_file_tower)
    (shapes_line, records_line, fields_line) = read_shape_file(shape_file_line)

    sel_keys = ['Name', 'Latitude', 'Longitude', 'POINT_X',
                     'POINT_Y', 'Mun', 'Barangay', 'ConstType', 'Function', 'LineRoute']

    sel_idx = {}
    for str_ in sel_keys:
        sel_idx[str_] = get_field_index(fields_tower, str_)

    # processing shp file

    fid, name, line_route, lat, lon = [], [], [], [], []
    for fid_, item in enumerate(records_tower):
        name_ = item[sel_idx['Name']] #
        line_route_ = item[sel_idx['LineRoute']]
        lat_ = item[sel_idx['Latitude']]
        lon_ = item[sel_idx['Longitude']]

        fid.append(fid_)
        name.append(name_)
        line_route.append(line_route_)
        lat.append(lat_)
        lon.append(lon_)

    fid = np.array(fid)
    name = np.array(name)
    line_route = np.array(line_route)
    lat = np.array(lat, dtype=float)
    lon = np.array(lon, dtype=float)

    unq_line_route = np.unique(line_route)
    nline = len(unq_line_route)

    # validate line route information stored in line shapefile
    # unnecessary if line shape file corrected
    orig_list = range(len(records_line))
    correct_lineroute_mapping = {}

    for line in unq_line_route:

        idx = np.where(line_route == line)[0]
        idx = np.random.choice(idx,1)[0]

        lon_sel, lat_sel = lon[idx], lat[idx]

        for i in orig_list:

            xy = np.array(shapes_line[i].points)
            #print xy.shape, lon_sel, lat_sel

            diff = xy - np.ones((xy.shape[0],1))*np.array([[lon_sel, lat_sel]])
            abs_diff = np.min(diff[:,0]*diff[:,0]+diff[:,1]*diff[:,1])
            tf = np.allclose(abs_diff, 0.0, 1.0e-4)

            if tf == True:
                print "line: %s, LineRoute: %s, line ID: %s" %(line, records_line[i][5], str(i))
                #orig_list.remove(i)
                correct_lineroute_mapping[line] = i
                break
            else:
                pass

    return fid, name, line_route


def read_velocity_profile(conf, tower):
    """
    read velocity time history at each tower location

    Usage:
    read_velocity_profile(Wind, dir_wind_timeseries, tower)
    :return
     event: dictionary of event classes
    """
    event = dict()  # dictionary of event class instances

    for name in tower:
        file_name = 'ts.' + name + '.csv'
        vel_file = os.path.join(conf.dir_wind_timeseries, file_name)
        try:
            event[name] = Event(tower[name], vel_file)

        except IOError:
            print 'File not found:', vel_file
            import inspect
            return {'error': 'File {} not found in function {}'.format(vel_file, inspect.stack()[0][3])}
        except Exception as e:
            print e
            raise

    return event


if __name__ == '__main__':
    from config_class import TransmissionConfig
    conf = TransmissionConfig()

