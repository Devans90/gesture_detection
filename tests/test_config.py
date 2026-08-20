from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from gesture_detection.config import AppConfig


class ConfigTests(unittest.TestCase):
    def test_write_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'config.json'
            config = AppConfig()
            config.sensor.driver = 'mock'
            config.write(path)

            loaded = AppConfig.from_path(path)

            self.assertEqual(loaded.sensor.driver, 'mock')
            self.assertEqual(loaded.capture.sample_hz, config.capture.sample_hz)


class CliTests(unittest.TestCase):
    def test_init_config_does_not_require_default_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            output = repo_root / 'generated.json'

            project_root = Path(__file__).resolve().parents[1]
            env = dict(os.environ)
            env['PYTHONPATH'] = str(project_root / 'src')
            result = subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'gesture_detection.cli',
                    '--config',
                    str(repo_root / 'missing-default.json'),
                    'init-config',
                    '--output',
                    str(output),
                ],
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn('Wrote', result.stdout)
            self.assertTrue(output.exists())
            self.assertFalse((repo_root / 'missing-default.json').exists())


if __name__ == '__main__':
    unittest.main()
