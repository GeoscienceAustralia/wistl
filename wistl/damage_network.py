#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import os
import numpy as np

from wistl.transmission_network import TransmissionNetwork
from wistl.damage_line import TransmissionLine


class DamageNetwork(TransmissionNetwork):
    """ class for a collection of damage to lines
    """

    def __init__(self, conf, event_id):

        self.event_id = event_id
        super(DamageNetwork, self).__init__(conf)

        # self.lines here are instances of TransmissionLine

        # line is a TransmissionLine instance
        for key, line in self.lines.iteritems():
            self.lines[key] = TransmissionLine(line, event_id)

        # after the for loop self.lines are DamageLine instances

        # assuming same time index for each tower in the same network
        self.time_index = self.lines[key].time_index


