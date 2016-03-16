#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import shapefile
import pandas as pd
from wistl.transmission_line import TransmissionLine


class TransmissionNetwork(object):
    """ class for a collection of wistl lines"""

    def __init__(self, conf):
        self.conf = conf
        self.df_towers = read_shape_file(self.conf.file_shape_tower)
        self.df_lines = read_shape_file(self.conf.file_shape_line)

        self.lines = dict()
        for name, grouped in self.df_towers.groupby('LineRoute'):
            if name in self.conf.sel_lines:
                try:
                    idx = self.df_lines[self.df_lines['LineRoute']
                                        == name].index[0]
                except IndexError:
                    print ('{} not in the line shapefile'.format(name))
                    raise

                self.lines[name] = TransmissionLine(
                    conf=self.conf,
                    df_towers=grouped,
                    df_line=self.df_lines.loc[idx])


def read_shape_file(file_shape):
    """ read shape file and return data frame
    """
    sf = shapefile.Reader(file_shape)
    shapes = sf.shapes()
    records = sf.records()
    fields = [x[0] for x in sf.fields[1:]]
    fields_type = [x[1] for x in sf.fields[1:]]

    shapefile_type = {'C': 'object', 'F': 'np.float64', 'N': 'np.int64'}

    data_frame = pd.DataFrame(records, columns=fields)

    for name_, type_ in zip(data_frame.columns, fields_type):
        if data_frame[name_].dtype != eval(shapefile_type[type_]):
            data_frame[name_] = \
                data_frame[name_].astype(eval(shapefile_type[type_]))

    if 'Shapes' in fields:
        raise KeyError('Shapes is already in the fields')
    else:
        data_frame['Shapes'] = shapes

    return data_frame

# if __name__ == '__main__':
#     from config_class import TransmissionConfig
#     conf = TransmissionConfig()
