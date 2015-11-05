'''
read 
    - collapse fragility
    - conditional proability
    - shape file
    - wind velocity profile

'''

import pandas as pd
import numpy as np
from StringIO import StringIO
import matplotlib.pyplot as plt
from scipy.stats import lognorm

def read_topo_value(file_):
    '''
    read topograhpic multipler value 
    '''
    data = pd.read_csv(file_, header=0, usecols=[0,9,10],
        names=['Name','','','','','','','','','Mh','Mhopp'])
    data['topo'] = np.max([data['Mh'].values, data['Mhopp'].values],axis=0)

    val = {}
    for item in data['Name']:
        val[item] = data[data['Name']==item]['topo'].values[0]
    return val

def read_terrain_height_multiplier(file_):
    """read terrain height multiplier (ASNZ 1170.2:2011 Table 4.1)
    """

    data = pd.read_csv(file_, header=0, skipinitialspace = True)
    height, cat1, cat2, cat3, cat4 = [], [], [], [], []
    for line in data.iterrows():
        height_, cat1_, cat2_, cat3_, cat4_ = [ line[1][x] for x in 
        ['height(m)', 'terrain category 1', 'terrain category 2', 
        'terrain category 3', 'terrain category 4']]
        height.append(height_)
        cat1.append(cat1_)
        cat2.append(cat2_)
        cat3.append(cat3_)
        cat4.append(cat4_)

    terrain_height = {}
    terrain_height['height'] = height
    terrain_height['tc1'] = cat1
    terrain_height['tc2'] = cat2
    terrain_height['tc3'] = cat3
    terrain_height['tc4'] = cat4

    return terrain_height


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
            dev_ = np.sort(np.union1d(dev0_,dev1_))

            frag.setdefault(ttype, {}).setdefault(func,{})['dev_angle'] = dev_

            for (j, val) in enumerate(dev0_):

                idx2 = np.where(idx & (data['dev0'] == val))[0]

                for k in idx2:
                    ds_ = data.ix[k]['damage']
                    idx_ = data.ix[k]['index']
                    cdf_ = data.ix[k]['cdf']
                    param0_ = data.ix[k]['param0']
                    param1_ = data.ix[k]['param1']

                    # dev_angle (0, 5, 15, 30, 360) <= tower_angle 0.0 => index
                    # angle is less than 360. if 360 then 0.
                    frag[ttype][func].setdefault(j+1,{}).setdefault(ds_,{})['idx'] = idx_
                    frag[ttype][func][j+1][ds_]['param0'] = param0_
                    frag[ttype][func][j+1][ds_]['param1'] = param1_
                    frag[ttype][func][j+1][ds_]['cdf'] = cdf_

    ds_list = [(x, frag[ttype][func][1][x]['idx']) 
              for x in frag[ttype][func][1].keys()]

    ds_list.sort(key=lambda tup:tup[1]) # sort by ids
    #ds_list = [('minor', 1), ('collapse', 2)]

    nds = len(ds_list)

    if flag_plot:
        x = np.arange(0.5, 1.5, 0.01)
        line_style = {'minor':'--','collapse':'-'}

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

                plt.legend(['collapse','minor'],2)
                plt.xlabel('Ratio of wind speed to adjusted design wind speed')
                plt.ylabel('Probability of exceedance')
                plt.title(ttype+':'+func)
                plt.yticks(np.arange(0,1.1,0.1))
                plt.grid(1)
                plt.savefig(ttype +'_' + func +'.png')

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
    radius = 6371.0 # km
 
    dlat = np.radians(lat2-lat1)
    dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c

    return d

def read_tower_GIS_information(Tower, shape_file_tower, shape_file_line,
    file_design_value,  file_topo_value = None, flag_figure=None, flag_load=1):
    """
    read geospational information of towers

    Usage:
    read_tower_GIS_information(Tower, shape_file_tower, shape_file_line, 
        flag_figure=None, flag_load = 1, sel_lines = None)
    """

    (shapes_tower, records_tower, fields_tower) = read_shape_file(shape_file_tower)
    (shapes_line, records_line, fields_line) = read_shape_file(shape_file_line)

    (sel_lines, design_value) = read_design_value(file_design_value)

    if file_topo_value != None:
        topo_value = read_topo_value(file_topo_value)
        topo_dic = {'threshold': np.array([1.05, 1.1, 1.2, 1.3, 1.45]), 
        0: 1.0, 1: 1.1, 2: 1.2, 3: 1.3, 4: 1.45, 5: 1.6}

    km2m = 1000.0

    # typical drag height by tower type
    height_z_dic = {'Suspension': 15.4, 'Strainer': 12.2, 'Terminal': 12.2}

    sel_keys = ['Type', 'Name', 'Latitude', 'Longitude', 'DevAngle', 
                'AxisAz', 'Mun', 'Barangay', 'ConstType', 'Function', 
                'LineRoute', 'Height']
 
    sel_idx = {}
    for str_ in sel_keys:
        sel_idx[str_] = get_field_index(fields_tower, str_)
    
    # processing shp file
    tower, fid2name = {}, {}
    fid, line_route, lat, lon = [], [], [], []
    fid_ = 0
    for item in records_tower:

        line_route_ = item[sel_idx['LineRoute']]

        if line_route_ in sel_lines:

            name_ = item[sel_idx['Name']] #
            ttype_ = item[sel_idx['Type']]
            funct_ = item[sel_idx['Function']]

            lat_ = item[sel_idx['Latitude']]
            lon_ = item[sel_idx['Longitude']]
            strong_axis_ = item[sel_idx['AxisAz']]
            dev_angle_ = item[sel_idx['DevAngle']]
            height_ = float(item[sel_idx['Height']])

            # FIXME
            height_z_ = height_z_dic[funct_] 

            if file_topo_value:
                idx_topo = np.sum(topo_value[name_] >= topo_dic['threshold'])
                designSpeed_ = design_value[line_route_]['speed']*topo_dic[idx_topo]
            else:
                designSpeed_ = design_value[line_route_]['speed']               

            designSpan_ = design_value[line_route_]['span']
            terrainCat_ = design_value[line_route_]['cat']
            designLevel_ = design_value[line_route_]['level']

            tower[name_] = Tower(fid_, ttype_, funct_, 
                                 line_route_, designSpeed_, designSpan_, designLevel_,  
                                 terrainCat_, strong_axis_, dev_angle_, height_, height_z_)

            #print "%s, %s, %s" %(name_, tower[name_].sd, sd_)

            fid2name[fid_] = name_
            fid.append(fid_)
            line_route.append(line_route_)
            lat.append(lat_)
            lon.append(lon_)

            fid_ += 1 # increase by 1

    fid = np.array(fid)
    line_route = np.array(line_route)
    lat = np.array(lat, dtype=float)
    lon = np.array(lon, dtype=float)
    #unq_line_route = np.unique(line_route)
    #nline = len(unq_line_route)
    nline = len(sel_lines)

    lineroute_line = list(get_data_per_polygon(records_line, fields_line, 
                          'LineRoute')) 

    # generate connectivity for each of line routes
    fid_by_line = {}

    #for line in unq_line_route:
    for line in sel_lines:

        # pts
        tf = (line_route == line)
        fid_pt = fid[tf]
        xy_pt = np.array([lon[tf], lat[tf]], dtype=float).T

        # need to change once line shape file is corrected
        #i = correct_lineroute_mapping[line]
        i = lineroute_line.index(line)
        xy_line = np.array(shapes_line[i].points)

        idx_sorted = []
        for item in xy_line:

            diff = xy_pt - np.ones((xy_pt.shape[0],1))*item[np.newaxis,:]
            temp = diff[:,0]*diff[:,0]+diff[:,1]*diff[:,1]
            idx = np.argmin(temp)
            tf_ = np.allclose(temp[idx], 0.0, 1.0e-4)
            if tf_ == True:
                idx_sorted.append(fid_pt[idx])
            else:
                print 'Something wrong %s, %s, %s' %(line, item, xy_pt[idx])    

        fid_by_line[line] = idx_sorted

        # calculate distance between towers
        ntower_sel = sum(tf)
        dist_forward = []

        for i in range(ntower_sel-1):

            j0 = fid_by_line[line][i]
            j1 = fid_by_line[line][i+1]

            pt0 = (lat[j0], lon[j0])
            pt1 = (lat[j1], lon[j1])

            temp = distance(pt0, pt1)*km2m

            dist_forward.append(temp)

        if flag_figure:
            pt_x, pt_y = [], []
            for k in idx_sorted:
                pt_x.append(lon[fid==k])
                pt_y.append(lat[fid==k])

            plt.figure()
            plt.plot(xy_line[:,0],xy_line[:,1],'ro-',pt_x,pt_y,'b-')
            plt.title(line)

        # assign adj, actual_span, adj_design_speed    
        idx_sorted = np.insert(idx_sorted, 0, -1) # start 
        idx_sorted = np.append(idx_sorted, -1) # end

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
    
    return (tower, sel_lines, fid_by_line, fid2name, lon, lat)

def read_shape_file(file_shape):
    """
    read shape file
    """    

    import shapefile

    sf = shapefile.Reader(file_shape)
    shapes = sf.shapes()
    records = sf.records()
    fields = sf.fields
    fields = fields[1:]
    
    return (shapes, records, fields)

def get_field_index(fields, key_string):
    """
    retrieve field index
    """
    i, tf = -1, False
    while (tf == False) and (i < len(fields)-1):
        i += 1
        tf = fields[i][0] == key_string
    if tf == True:
        return i
    else:
        return np.NaN

def get_data_per_polygon(records, fields, key_string):
    """
    retrieve field data
    """
    x = []
    i = get_field_index(fields, key_string)
    for item in records:
        x.append(item[i])
    return np.array(x)

def check_shape_files_tower_line(shape_file_tower, shape_file_line):

    """check consistency of shape files of tower and line"""

    (shapes_tower, records_tower, fields_tower) = read_shape_file(shape_file_tower)
    (shapes_line, records_line, fields_line) = read_shape_file(shape_file_line)

    sel_keys = ['Name', 'Latitude', 'Longitude', 'POINT_X', 
                     'POINT_Y', 'Mun', 'Barangay', 'ConstType', 'Function', 
                     'LineRoute']

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

    return (fid, name, line_route)

def dir_wind_speed(speed, bearing, t0):

    # angle between wind direction and tower conductor
    phi = np.abs(bearing-t0)

    tf = (phi <= np.pi/4) | (phi > np.pi/4*7) | ((phi > np.pi/4*3) & 
        (phi <= np.pi/4*5))

    cos_ = abs(np.cos(np.pi/4.0-phi))
    sin_ = abs(np.sin(np.pi/4.0-phi))

    adj = speed*np.max(np.vstack((cos_, sin_)), axis=0)

    dir_speed = np.where(tf,adj,speed) # adj if true, otherwise speed 

    return dir_speed

def convert_10_to_z(asset, terrain_height):
    """
    Mz,cat(h=10)/Mz,cat(h=z)
    tc: terrain category (defined by line route)
    """
    tc_str = 'tc' + str(asset.terrain_cat) # Terrain 

    try:
        mzcat_z = np.interp(asset.height_z, terrain_height['height'], terrain_height[tc_str])
    except KeyError:
        print "%s is not defined" %tc_str

    mzcat_10 = terrain_height[tc_str][terrain_height['height'].index(10)]

    return (mzcat_z/mzcat_10)


def read_velocity_profile(Wind, dir_wind_timeseries, tower, file_terrain_height):
    """
    read velocity time history at each tower location

    Usage:
    read_velocity_profile(Wind, dir_wind_timeseries, tower)
    Wind: Event class
    """

    # read velocity profile for each of the towers

    terrain_height = read_terrain_height_multiplier(file_terrain_height)

    event = dict()  # dictionary of event class instances

    for name in tower.keys():

        vel_file = dir_wind_timeseries + '/ts.' + name + '.csv'

        try:
            event[name] = Wind(tower[name].fid)

            # Time,Longitude,Latitude,Speed,UU,VV,Bearing,Pressure
            data = pd.read_csv(vel_file, header=0, parse_dates=[0], index_col=[0],
                usecols=[0,3,6],names=['','','','speed','','','bearing',''])

            speed = data['speed'].values
            bearing = np.deg2rad(data['bearing'].values) # degree

            # angle of conductor relative to NS
            t0 = np.deg2rad(tower[name].strong_axis) - np.pi/2.0 

            convert_factor = convert_10_to_z(tower[name], terrain_height)

            dir_speed = convert_factor * dir_wind_speed(speed, bearing, t0)

            event[name].convert_factor = convert_factor

            # convert velocity at 10m to dragt height z

            #data['EW'] = pd.Series(speed*np.cos(bearing+np.pi/2.0), 
            #             index=data.index) # x coord
            #data['NS'] = pd.Series(speed*np.sin(bearing-np.pi/2.0), 
            #             index=data.index) # y coord
            data['dir_speed'] = pd.Series(dir_speed, index=data.index)

            event[name].wind = data

        except ValueError:
            print vel_file

    return event