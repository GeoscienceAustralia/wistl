#!/usr/bin/env python
from __future__ import print_function

__author__ = 'Hyeuk Ryu'

import os
from event import Event


class EventSet(object):
    """ class for a collection of wind events
        network: instance of TransmissionNetwork class
    """

    def __init__(self, network):
        self.network = network

        self.events = dict()
        for path_wind in self.network.conf.path_wind_timeseries:
            event_id = path_wind.split('/')[-1]
            self.events[event_id] = self.add_event(event_id)

    def add_event(self, event_id):
        """ add wind event
        """
        path_ = [x for x in self.network.conf.path_wind_timeseries
                 if event_id in x][0]
        towers = dict()
        for _, line in self.network.lines.iteritems():
            for name, tower in line.towers.iteritems():
                vel_file = os.path.join(path_, tower.wind_file)
                towers[name] = Event(tower, vel_file)
        return towers


# class Other(object):

#     def override(self):
#         print "OTHER override()"

#     def implicit(self):
#         print "OTHER implicit()"

#     def altered(self):
#         print "OTHER altered()"

# class Child(object):

#     def __init__(self):
#         self.other = Other()

#     def implicit(self):
#         self.other.implicit()

#     def override(self):
#         print "CHILD override()"

#     def altered(self):
#         print "CHILD, BEFORE OTHER altered()"
#         self.other.altered()
#         print "CHILD, AFTER OTHER altered()"



# class Add_Event(object):

#     def __init__(self,)
