# script to set data for sim_towers_v13-2
import os

from config_class import TransmissionConfig

if __name__ == '__main__':
    from sim_towers_v13_2 import main
    conf = TransmissionConfig()
    main(conf)
