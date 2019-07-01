__author__ = 'sudipta'
import os
import logging
import unittest
from wistl.scenario import create_scenario
from wistl.config import Config

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class EventTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        file_cfg = os.path.join(BASE_DIR, 'test.cfg')

        assert os.path.exists(file_cfg), 'config file does not exist'
        cls.cfg = Config(file_cfg, logger)
        cls.event = create_scenario(cfg=cls.cfg,
                                 event=cls.cfg.events[0])

    def test_time_index(self):
        # compare time_index
        for line in self.event:
            self.assertTrue(line.time_index.equals(self.event[0].time_index))

    def test_number_of_damaged_networks(self):
        self.assertEqual(len(self.event), len(self.cfg.lines))

    def test_no_lines(self):
        pass

    def test_path_output(self):
        pass



    # def test_damaged_network_type(self):
    #     from wistl.transmission_network import TransmissionNetwork
    #     for _, d in self.damaged_networks.items():
    #         self.assertTrue(isinstance(d, TransmissionNetwork))


if __name__ == '__main__':
    unittest.main(verbosity=2)
