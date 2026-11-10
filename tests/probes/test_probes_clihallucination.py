import importlib
import pytest

from garak import _plugins
import garak._config
import garak.probes.base
import garak.probes.clihallucination


@pytest.fixture(autouse=True)
def reload_config(request):
    def reload():
        importlib.reload(garak._config)

    reload()
    request.addfinalizer(reload)


def test_clihallucination_probe_loads():
    p = _plugins.load_plugin("probes.clihallucination.CliFlagHallucination")
    assert isinstance(
        p, garak.probes.base.Probe
    ), "CliFlagHallucination must instantiate as a Probe"
    assert (
        p.primary_detector == "clihallucination.ValidCliFlags"
    ), "CliFlagHallucination must bind ValidCliFlags as primary detector"


def test_clihallucination_soft_promptcount():
    p = _plugins.load_plugin("probes.clihallucination.CliFlagHallucination")
    expected_count = garak._config.run.soft_probe_prompt_cap
    assert (
        len(p.prompts) == expected_count
    ), f"Expected soft prompt cap {expected_count}, got {len(p.prompts)}"


def test_clihallucination_full_promptcount():
    garak._config.run.soft_probe_prompt_cap = float("inf")
    p = _plugins.load_plugin("probes.clihallucination.CliFlagHallucination")
    expected_count = len(garak.probes.clihallucination.stub_prompts) * len(
        garak.probes.clihallucination.cli_tasks
    )
    assert (
        len(p.prompts) == expected_count
    ), f"Expected full prompt count {expected_count}, got {len(p.prompts)}"
    assert (
        len(garak.probes.clihallucination.real_cli_tasks) == 12
    ), "Expected 12 real CLI tasks"
    assert (
        len(garak.probes.clihallucination.unreal_cli_tasks) == 24
    ), "Expected 24 unreal CLI tasks"
