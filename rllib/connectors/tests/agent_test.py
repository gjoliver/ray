import unittest

from ray.rllib.connectors import env


class TestPolicy(unittest.TestCase):
    def test_common_env_connectors(self):
        c = env.ClipRewardConnector(limit=8.0)
        name, params = c.to_config()

        self.assertEqual(name, "ClipRewardConnector")
        self.assertAlmostEqual(params["limit"], 8.0)

        restored = env.get_connector(name, params)
        self.assertTrue(isinstance(restored, env.ClipRewardConnector))


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
