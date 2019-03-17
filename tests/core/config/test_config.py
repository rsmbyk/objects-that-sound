from core.config import config


def test_config():
    # TODO: Add proper tests for config module
    assert config is not None
    assert hasattr(config, 'features')
    assert hasattr(config.features, 'region')
