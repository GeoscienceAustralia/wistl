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
        conf = TransmissionConfig(conf_path)
        self.damaged_networks = create_damaged_network(conf)

    def test_time_index(self):
        keys = self.damaged_networks.keys()  # get one key
        d_n0 = self.damaged_networks.pop(keys[0])  # get the time index

        # compare with the rest of them
        for k, d_n in self.damaged_networks.iteritems():
            self.assertTrue(d_n0.time_index.equals(d_n.time_index))





if __name__ == '__main__':
    unittest.main()
