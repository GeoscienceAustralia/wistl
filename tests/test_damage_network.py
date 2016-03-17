__author__ = 'sudipta'
import os
import pandas as pd
import unittest

WISTL = os.environ['WISTL']


class DamagedNetowrkTest(unittest.TestCase):

    def setUp(self):
        from wistl.damage_network import create_damaged_network
        from wistl.config_class import TransmissionConfig
        conf_path = os.path.join(WISTL, 'tests', 'test.cfg')
        assert os.path.exists(conf_path), 'config file path wrong'
        self.conf = TransmissionConfig(conf_path)
        self.damaged_networks = create_damaged_network(self.conf)

    def test_time_index(self):
        keys = self.damaged_networks.keys()  # get one key
        d_n0 = self.damaged_networks.pop(keys[0])  # get the time index

        # compare with the rest of them
        for k, d_n in self.damaged_networks.iteritems():
            self.assertTrue(d_n0.time_index.equals(d_n.time_index))

    def test_number_of_damaged_networks(self):
        self.assertEqual(len(self.damaged_networks),
                         len(self.conf.path_wind_scenario))

    def test_damaged_network_type(self):
        from wistl.damage_network import DamageNetwork
        from wistl.transmission_network import TransmissionNetwork
        for d in self.damaged_networks.itervalues():
            self.assertTrue(isinstance(d, DamageNetwork))
            self.assertTrue(isinstance(d, TransmissionNetwork))

if __name__ == '__main__':
    unittest.main()
