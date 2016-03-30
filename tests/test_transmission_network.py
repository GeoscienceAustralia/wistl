__author__ = 'sudipta'
import os
import unittest
from wistl.transmission_network import create_damaged_network
from wistl.config_class import TransmissionConfig

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class TransmissionNetworkTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        conf_path = os.path.join(BASE_DIR, 'test.cfg')
        assert os.path.exists(conf_path), 'config file path wrong'
        cls.conf = TransmissionConfig(conf_path)
        cls.damaged_networks = create_damaged_network(cls.conf)

    def test_time_index(self):
        keys = self.damaged_networks.keys()  # get the keys
        d_n0 = self.damaged_networks.pop(keys[0])  # get the time index

        # compare with the rest of them
        for k, d_n in self.damaged_networks.iteritems():
            self.assertTrue(d_n0.time_index.equals(d_n.time_index))

    def test_number_of_damaged_networks(self):
        self.assertEqual(len(self.damaged_networks),
                         len(self.conf.path_wind_scenario))

    def test_damaged_network_type(self):
        from wistl.transmission_network import TransmissionNetwork
        for d in self.damaged_networks.itervalues():
            self.assertTrue(isinstance(d, TransmissionNetwork))

if __name__ == '__main__':
    unittest.main()
