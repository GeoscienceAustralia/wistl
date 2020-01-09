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
        cls.cfg.events[0] = ('test1', 3.0, 1)
        cls.scenario = Scenario(cfg=cls.cfg,
                                event=('test1', 20.0, 1),
                                logger=cls.logger)

    @classmethod
    def tearDownClass(cls):
        os.removedirs(cls.scenario.path_output)

    def test_id(self):
        self.assertEqual(self.scenario.id, 'test1_s20.0')

    def test_time(self):
        # compare time_index
        for name, line in self.scenario.lines.items():
            self.assertTrue(line.time.equals(self.scenario.lines[name].time))

    def test_number_of_lines(self):
        self.assertEqual(len(self.scenario.lines), len(self.cfg.lines))

    def test_no_lines(self):
        self.assertEqual(self.scenario.no_lines, 2)

    def test_path_output(self):
        self.assertEqual(self.cfg.options['save_output'], True)
        self.assertEqual(os.path.exists(self.scenario.path_output), True)
        expected = os.path.join(BASE_DIR, 'output/test1_s20.0')
        self.assertEqual(self.scenario.path_output, expected)

        os.removedirs(self.scenario.path_output)
        self.scenario._path_output = None
        msg = '{} is created'.format(expected)
        with self.assertLogs(logger=self.logger, level='INFO') as cm:
            _ = self.scenario.path_output
            self.assertEqual(
                '{}:{}'.format(cm.output[0].split(':')[0],cm.output[0].split(':')[-1]),
                'INFO:{}'.format(msg))

if __name__ == '__main__':
    unittest.main(verbosity=2)
