# script to set data for sim_towers_v13-2
import os

from config_class import TransmissionConfig
from sim_towers_v13_2 import sim_towers

if __name__ == '__main__':
    conf = TransmissionConfig()
    sim_towers(conf)
