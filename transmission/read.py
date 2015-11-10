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


def read_frag(file_, flag_plot=None):
    """read collapse fragility parameter values
    >>> txt = '''ConstType, damage, index, function, param0, param1
    ... Unknown, minor, 1, lognorm, 0.85, 0.05'''
    ... Unknown, collpase, 2, lognorm, 1.02, 0.05'''
    >>> data = read_collapse_frag(StringIO(txt))
    >>> data'''
    {'Unknown': {'collapse': {'function': 'lognorm',
       'param0': 1.02,
       'param1': 0.05},
      'minor': {'function': 'lognorm', 'param0': 0.85, 'param1': 0.05}}}
    """

    data = pd.read_csv(file_, skipinitialspace=1)

    frag = {}
    for ttype in data['tower type'].unique():
        for func in data['function'].unique():
            idx = (data['tower type'] == ttype) & (data['function'] == func)
            dev0_ = data['dev0'].ix[idx].unique()
            dev1_ = data['dev1'].ix[idx].unique()
            dev_ = np.sort(np.union1d(dev0_, dev1_))

            frag.setdefault(ttype, {}).setdefault(func, {})['dev_angle'] = dev_

            for j, val in enumerate(dev0_):

                idx2 = np.where(idx & (data['dev0'] == val))[0]

                for k in idx2:
                    ds_ = data.ix[k]['damage']
                    idx_ = data.ix[k]['index']
                    cdf_ = data.ix[k]['cdf']
                    param0_ = data.ix[k]['param0']
                    param1_ = data.ix[k]['param1']

                    # dev_angle (0, 5, 15, 30, 360) <= tower_angle 0.0 => index
                    # angle is less than 360. if 360 then 0.
                    frag[ttype][func].setdefault(j+1, {}).setdefault(ds_, {})['idx'] = idx_
                    frag[ttype][func][j+1][ds_]['param0'] = param0_
                    frag[ttype][func][j+1][ds_]['param1'] = param1_
                    frag[ttype][func][j+1][ds_]['cdf'] = cdf_

    ds_list = [(x, frag[ttype][func][1][x]['idx']) for x in frag[ttype][func][1].keys()]
    ds_list.sort(key=lambda tup: tup[1])  # sort by ids

    nds = len(ds_list)

    if flag_plot:
        x = np.arange(0.5, 1.5, 0.01)
        line_style = {'minor': '--', 'collapse': '-'}

        for ttype in frag.keys():
            for func in frag[ttype].keys():
                plt.figure()

                for idx in frag[ttype][func].keys():
                    try:
                        for ds in frag[ttype][func][idx].keys():
                            med = frag[ttype][func][idx][ds]['param0']
                            sig = frag[ttype][func][idx][ds]['param1']
                            y = lognorm.cdf(x, sig, scale=med)
                            plt.plot(x,y, line_style[ds])
                    except AttributeError:
                        print "no"

                plt.legend(['collapse', 'minor'], 2)
                plt.xlabel('Ratio of wind speed to adjusted design wind speed')
                plt.ylabel('Probability of exceedance')
                plt.title(ttype+':'+func)
                plt.yticks(np.arange(0, 1.1, 0.1))
                plt.grid(1)
                plt.savefig(ttype + '_' + func + '.png')

    return frag, ds_list, nds


def read_cond_prob(file_):
    """read condition collapse probability defined by tower function 

    >>> txt = '''FunctionType, # of collapse, probability, start, end
    ... suspension, 1, 0.075, 0, 1
    ... suspension, 1, 0.075, -1, 0
    ... suspension, 2, 0.35, -1, 1
    ... suspension, 3, 0.025, -1, 2
    ... suspension, 3, 0.025, -2, 1
    ... suspension, 4, 0.10, -2, 2
    ... strainer, 1, 0.075, 0, 1
    ... strainer, 1, 0.075, -1, 0
    ... strainer, 2, 0.35, -1, 1
    ... strainer, 3, 0.025, -1, 2
    ... strainer, 3, 0.025, -2, 1
    ... strainer, 4, 0.10, -2, 2'''
    ... strainer, 5, 0.10, -2, 2'''
    ... strainer, 5, 0.10, -2, 2'''
    ... strainer, 5, 0.10, -2, 2'''
    >>> cond_pc = read_cond_prob(StringIO(txt))
    >>> cond_pc'''
    {'strainer': {'max_adj': 2,
      (-2, -1, 0, 1): 0.025,
      (-2, -1, 0, 1, 2): 0.1,
      (-1, 0): 0.075,
      (-1, 0, 1): 0.35,
      (-1, 0, 1, 2): 0.025,
      (0, 1): 0.075},
     'suspension': {'max_adj': 2,
      (-2, -1, 0, 1): 0.025,
      (-2, -1, 0, 1, 2): 0.1,
      (-1, 0): 0.075,
      (-1, 0, 1): 0.35,
      (-1, 0, 1, 2): 0.025,
      (0, 1): 0.075}}
    """

    data = pd.read_csv(file_, skipinitialspace=1)
    cond_pc = {}
    for line in data.iterrows():
        func, cls_str, thr, pb, n0, n1 = [ line[1][x] for x in 
                           ['FunctionType', 'class', 'threshold', 'probability', 'start', 'end']]
        list_ = range(int(n0), int(n1)+1)
        cond_pc.setdefault(func,{})['threshold'] = thr
        cond_pc[func].setdefault(cls_str,{}).setdefault('prob',{})[tuple(list_)] = float(pb)

    for func in cond_pc.keys():
        cls_str = cond_pc[func].keys()
        cls_str.remove('threshold')
        for cls in cls_str:    
            max_no_adj_towers = np.max(np.abs([j for k in cond_pc[func][cls]['prob'].keys() 
                            for j in k]))
            cond_pc[func][cls]['max_adj'] = max_no_adj_towers

    return cond_pc 


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

