__author__ = 'sudipta'
import os
import unittest
from wistl.event import create_event
from wistl.config import Config

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class EventTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        conf_path = os.path.join(BASE_DIR, 'test.cfg')
        assert os.path.exists(conf_path), 'config file path wrong'
        cls.cfg = Config(conf_path)
        cls.event = create_event(cfg=cls.cfg,
                                 event=cls.cfg.events[0])

    def test_time_index(self):
        # compare time_index
        for line in self.event:
            self.assertTrue(line.time_index.equals(self.event[0].time_index))

    def test_number_of_damaged_networks(self):
        self.assertEqual(len(self.event), len(self.cfg.lines))

    # def test_damaged_network_type(self):
    #     from wistl.transmission_network import TransmissionNetwork
    #     for _, d in self.damaged_networks.items():
    #         self.assertTrue(isinstance(d, TransmissionNetwork))

if __name__ == '__main__':
    unittest.main()