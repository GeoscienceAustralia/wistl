#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import shapefile
import pandas as pd

from transmission_line import TransmissionLine


class TransmissionNetwork(object):
    """ class for a collection of transmission lines"""

    def __init__(self, conf):
        self.conf = conf
        self.df_towers = read_shape_file(self.conf.file_shape_tower)
        self.df_lines = read_shape_file(self.conf.file_shape_line)

        self.lines = dict()
        for name, grouped in self.df_towers.groupby('LineRoute'):
            if name in self.conf.sel_lines:
                tf = self.df_lines['LineRoute'] == name
                if not tf.sum():
                    raise KeyError('{} not in the line shapefile'.format(name))
                self.lines[name] = self.add_line(df_towers=grouped,
                                                 df_line=self.df_lines[tf])

    def add_line(self, df_towers, df_line):
        """ add transmission line
        """
        name = df_line['LineRoute'].values[0]
        if name in self.lines:
            raise KeyError('{} is already assigned'.format(name))
        self.lines[name] = TransmissionLine(self.conf, df_towers, df_line)
        return self.lines[name]


def read_shape_file(file_shape):
    """ read shape file and return data frame
    """
    sf = shapefile.Reader(file_shape)
    shapes = sf.shapes()
    records = sf.records()
    fields = [x[0] for x in sf.fields[1:]]

    data_frame = pd.DataFrame(records, columns=fields)
    if 'Shapes' in fields:
        raise KeyError('Shapes is already in the fields')
    else:
        data_frame['Shapes'] = shapes
    return data_frame

# if __name__ == '__main__':
#     from config_class import TransmissionConfig
#     conf = TransmissionConfig()
