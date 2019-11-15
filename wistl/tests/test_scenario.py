import os
import logging
import unittest
from wistl.scenario import Scenario
from wistl.config import Config

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class EventTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        logging.basicConfig(level=logging.DEBUG)
        cls.logger = logging.getLogger(__name__)

        file_cfg = os.path.join(BASE_DIR, 'test.cfg')

        assert os.path.exists(file_cfg), 'config file does not exist'
        cls.cfg = Config(file_cfg, cls.logger)
        cls.scenario = Scenario(cfg=cls.cfg,
                                event=cls.cfg.events[0],
                                logger=cls.logger)

    def test_time(self):
        # compare time_index
        for line in self.scenario.list_lines:
            self.assertTrue(line.time.equals(self.scenario.lines['LineA'].time))

    def test_number_of_lines(self):
        self.assertEqual(len(self.scenario.list_lines), len(self.cfg.lines))

    def test_no_lines(self):
        self.assertEqual(self.scenario.no_lines, 2)

    def test_path_output(self):
        self.assertEqual(self.cfg.options['save_output'], True)
        self.assertEqual(os.path.exists(self.scenario.path_output), True)
        expected = os.path.join(BASE_DIR, 'output/test1_s3.0')
        self.assertEqual(self.scenario.path_output, expected)

        os.removedirs(self.scenario.path_output)
        self.scenario._path_output = None
        msg = '{} is created'.format(expected)
        with self.assertLogs(logger=self.logger, level='INFO') as cm:
            _ = self.scenario.path_output
            self.assertEqual(
                '{}:{}'.format(cm.output[0].split(':')[0],cm.output[0].split(':')[-1]),
                'INFO:{}'.format(msg))


    # def test_damaged_network_type(self):
    #     from wistl.transmission_network import TransmissionNetwork
    #     for _, d in self.damaged_networks.items():
    #         self.assertTrue(isinstance(d, TransmissionNetwork))


if __name__ == '__main__':
    unittest.main(verbosity=2)
