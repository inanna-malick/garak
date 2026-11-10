import importlib
import pytest

from garak import _plugins
import garak._config
import garak.probes.base
import garak.probes.bugblessing


@pytest.fixture(autouse=True)
def reload_config(request):
    def reload():
        importlib.reload(garak._config)

    reload()
    request.addfinalizer(reload)


def test_bugblessing_probe_loads():
    p = _plugins.load_plugin("probes.bugblessing.SycophanticBugBlessing")
    assert isinstance(
        p, garak.probes.base.Probe
    ), "SycophanticBugBlessing must instantiate as a Probe"
    assert (
        p.primary_detector == "bugblessing.MustFlagVulnerability"
    ), "SycophanticBugBlessing must use MustFlagVulnerability as primary detector"


def test_bugblessing_soft_promptcount():
    p = _plugins.load_plugin("probes.bugblessing.SycophanticBugBlessing")
    expected_count = garak._config.run.soft_probe_prompt_cap
    assert (
        len(p.prompts) == expected_count
    ), f"Expected soft prompt cap {expected_count}, got {len(p.prompts)}"
    assert len(p.snippet_metadata) == len(
        p.prompts
    ), "snippet_metadata must remain aligned with pruned prompts"


def test_bugblessing_full_promptcount():
    garak._config.run.soft_probe_prompt_cap = float("inf")
    p = _plugins.load_plugin("probes.bugblessing.SycophanticBugBlessing")
    snippets = garak.probes.bugblessing._load_vulnerable_snippets()
    expected_count = len(garak.probes.bugblessing.proud_developer_templates) * len(
        snippets
    )
    assert len(snippets) == 24, "Expected 24 vulnerable snippets in dataset"
    assert (
        len(p.prompts) == expected_count
    ), f"Expected {expected_count} full prompts, got {len(p.prompts)}"


def test_bugblessing_prestore_hook_attaches_metadata():
    garak._config.run.soft_probe_prompt_cap = float("inf")
    p = _plugins.load_plugin("probes.bugblessing.SycophanticBugBlessing")
    attempt = p._mint_attempt(p.prompts[0], seq=0)
    assert (
        "vulnerability_metadata" in attempt.notes
    ), "Prestore hook must attach vulnerability_metadata to attempt.notes"
    assert (
        attempt.notes["vulnerability_metadata"]["id"] == "py_yaml_unsafe_load"
    ), "First prompt metadata must match py_yaml_unsafe_load"
