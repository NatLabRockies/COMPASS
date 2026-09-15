"""Reusable fixtures for offline integration scenarios"""

import pytest

from integration_harness import OfflineScenario


@pytest.fixture
def offline_scenario_factory(monkeypatch, tmp_path):
    """Create and install one offline scenario for an integration test"""

    def create(config_or_path):
        cache_dir = tmp_path / "replay_cache"
        if isinstance(config_or_path, dict):
            scenario = OfflineScenario(config_or_path, cache_dir)
        else:
            scenario = OfflineScenario.from_file(config_or_path, cache_dir)
        scenario.install(monkeypatch)
        return scenario

    return create

