# a script to manipulate the existing shapefile

"""
# http://gis.stackexchange.com/questions/72421/appending-using-pyshp
# Polygon shapefile we are updating.
# We must include a file extension in
# this case because the file name
# has multiple dots and pyshp would get
# confused otherwise.
file_name = "ep202009.026_5day_pgn.shp"
# Create a shapefile reader
r = shapefile.Reader(file_name)
# Create a shapefile writer
# using the same shape type
# as our reader
w = shapefile.Writer(r.shapeType)
# Copy over the existing dbf fields
w.fields = list(r.fields)
# Copy over the existing dbf records
w.records.extend(r.records())
# Copy over the existing polygons
w._shapes.extend(r.shapes())
# Add a new polygon
w.poly(parts=[[[-104,24],[-104,25],[-103,25],[-103,24],[-104,24]]])
# Add a new dbf record for our polygon making sure we include
# all of the fields in the original file (r.fields)
w.record("STANLEY","TD","091022/1500","27","21","48","ep")
# Overwrite the old shapefile or change the name and make a copy
w.save(file_name)
"""

import shapefile
import numpy as np

file_line = \
    './gis_data/Lines_NGCP_with_synthetic_attributes_WGS84_selection.shp'
new_file = './gis_data/Lines_parallel_line_interaction.shp'
sf_line = shapefile.Reader(file_line)

shapes_line = sf_line.shapes()
# records_line = sf_line.records()
# fields_line = sf_line.fields
# fields_line = [x[0] for x in sf.fields[1:]]

w = shapefile.Writer(sf_line.shapeType)
w.fields = list(sf_line.fields)
w.records.extend(sf_line.records())
w._shapes.extend(sf_line.shapes())

# Add a new polygon
coord0 = np.array(shapes_line[0].points)
coord1 = np.array(shapes_line[1].points)[::-1]  # noticed reverse order
coord2 = coord0 - (coord1 - coord0)

# list_coord2 = coord2[:, np.newaxis].tolist()
# w.poly(parts=list_coord2)

w.line(parts=coord2[:, np.newaxis].tolist())
# w.line(parts=[[[120.80379437, 13.93787626],
#                [120.80270786, 13.93673612],
#                [120.799997, 13.93543783],
#                [120.79805622, 13.93453621],
#                [120.79430776, 13.93354953],
#                [120.79191467, 13.93339797]]])

w.record('Amadeox - Calacax',
         'HV Transmission Line',
         'Unknown',
         'NGCP',
         'NGCP',
         'Amadeox - Calacax',
         '            ',
         '          ',
         '                         ',
         '                                                  ',
         '0.00000000000e+000',
         '0.00000000000e+000',
         'Created from NGCP tower location data',
         230,
         'Unknown',
         0,
         'Unknown',
         '1',
         0,
         '3.56524047953e+004',
         '               ')
w.save(new_file)

# modify towers shapefile

file_towers = './gis_data/Towers_with_extra_strainers_WGS84_selection.shp'
new_file = './gis_data/Towers_parallel_line_interaction.shp'
sf_ = shapefile.Reader(file_towers)

w = shapefile.Writer(sf_.shapeType)
w.fields = list(sf_.fields)
w.records.extend(sf_.records())
w._shapes.extend(sf_.shapes())

# Add a new
for i, pt in enumerate(coord2):
    w.point(pt[0], pt[1])

    name = 'DA-' + '{:0>3d}'.format(i)
    lon = '{:1.11e}'.format(pt[0])
    lat = '{:1.11e}'.format(pt[1])
    w.record('Lattice Tower',
             name,
             'NGCP',
             'NGCP',
             'CALACA',
             'Bagong Tubig',
             lat,
             lon,
             'Synthetic data only',
             'Unknown',
             'Suspension',
             0,
             134,
             '0.00000000000e+000',
             '1.78376227264e+001',
             0,
             'NGCP',
             '2.62787614054e+005',
             '1.54200016210e+006',
             '041007001',
             'Digitized from stereo pair derived from IFSAR data and ORI',
             'Amadeox - Calacax',
             0)

w.save(new_file)

