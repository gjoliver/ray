import unittest

from ray.rllib.connectors import connector


class TestPolicy(unittest.TestCase):
    def test_connector_pipeline(self):
        connectors = [connector.DoNothingConnector()]
        pipeline = connector.ConnectorPipeline(connectors)
        name, params = pipeline.to_config()
        restored = connector.get_connector(name, params)
        self.assertTrue(isinstance(restored, connector.ConnectorPipeline))


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
