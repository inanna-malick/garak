# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pytest
from garak import _plugins
import garak.probes.packagehallucination

PROBES = [
    classname
    for (classname, active) in _plugins.enumerate_plugins("probes")
    if classname.startswith("probes.packagehallucination")
]


@pytest.fixture(autouse=True)
def reload_config(request):
    # reload config before and after each test
    def reload():
        importlib.reload(garak._config)

    reload()
    request.addfinalizer(reload)


@pytest.mark.parametrize("classname", PROBES)
def test_soft_promptcount(classname):
    language_probe = _plugins.load_plugin(classname)

    expected_count = garak._config.run.soft_probe_prompt_cap

    assert (
        len(language_probe.prompts) == expected_count
    ), f"{language_probe.__name__} prompt count mismatch. Expected {expected_count}, got {len(language_probe.prompts)}"


@pytest.mark.parametrize("classname", PROBES)
def test_full_promptcount(classname):
    garak._config.run.soft_probe_prompt_cap = float("inf")

    language_probe = _plugins.load_plugin(classname)

    expected_count = len(garak.probes.packagehallucination.stub_prompts) * len(
        garak.probes.packagehallucination.code_tasks
    )

    assert (
        len(language_probe.prompts) == expected_count
    ), f"{language_probe.__name__} prompt count mismatch. Expected {expected_count}, got {len(language_probe.prompts)}"


def test_portmanteau_task_categories_included_and_configurable():
    ph = garak.probes.packagehallucination
    assert len(ph.unreal_vendor_chimera_tasks) == 6
    assert len(ph.unreal_interop_portmanteau_tasks) == 6
    for task in ph.unreal_vendor_chimera_tasks + ph.unreal_interop_portmanteau_tasks:
        assert task in ph.code_tasks

    garak._config.run.soft_probe_prompt_cap = float("inf")
    garak._config.plugins.probes["packagehallucination"] = {
        "task_categories": ["unreal_vendor_chimeras"]
    }
    chimera_only = ph.Python(config_root=garak._config)
    assert len(chimera_only.prompts) == len(ph.stub_prompts) * len(
        ph.unreal_vendor_chimera_tasks
    )
    assert any("CockroachSpanner" in p for p in chimera_only.prompts)
    assert not any("PolarsDask" in p for p in chimera_only.prompts)
    assert not any("HAL 9000" in p for p in chimera_only.prompts)

    garak._config.plugins.probes["packagehallucination"] = {
        "task_categories": ["unreal_interop_portmanteaus"]
    }
    interop_only = ph.Rust(config_root=garak._config)
    assert len(interop_only.prompts) == len(ph.stub_prompts) * len(
        ph.unreal_interop_portmanteau_tasks
    )
    assert any("PolarsDask" in p for p in interop_only.prompts)
    assert not any("CockroachSpanner" in p for p in interop_only.prompts)

    garak._config.plugins.probes["packagehallucination"] = {
        "task_categories": ["nonexistent_category"]
    }
    with pytest.raises(garak.exception.PluginConfigurationError):
        ph.Python(config_root=garak._config)
