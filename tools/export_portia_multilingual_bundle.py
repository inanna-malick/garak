"""Export the multilingual stress corpus as a standalone JSON bundle.

The bundle carries two pools with strictly disjoint contents:

* ``calibration_pool`` -- `en`, `zh` and `es` prompts only, intended for
  compiling a weight-space patch.
* ``eval_suite`` -- the four balanced twin families across all eight
  languages, each entry tagged with its language split, gold label and
  detector wiring.

Both pools are built from the probe classes in
``garak.probes.portia_multilingual_stress``, so the bundle is exactly what
garak would send, and the checks below fail the export rather than emit a
skewed or leaking corpus.

Usage::

    python tools/export_portia_multilingual_bundle.py [--output PATH]
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import garak._config
import garak.probes.portia_multilingual_stress as stress

DEFAULT_OUTPUT = Path(
    "/home/inanna/dev/argon-experiments/scratch/garak_multilingual_stress_bundle_v1.json"
)

PROBE_CLASSES = (
    stress.PortiaMultilingualPackageTwins,
    stress.PortiaMultilingualPresuppositionTwins,
    stress.PortiaMultilingualDefectTwins,
    stress.PortiaMultilingualSnowballHostileTwins,
)

MINIMUMS = {
    "package_twins": 320,
    "presupposition_twins": 192,
    "defect_twins": 96,
    "snowball_hostile_twins": 128,
}


def _bundle_entry(record):
    return {
        "id": record["id"],
        "family": record["family"],
        "subfamily": record["subfamily"],
        "twin_pair": record["twin_pair"],
        "lang": record["lang"],
        "lang_split": record["lang_split"],
        "is_positive_hazard": record["is_positive_hazard"],
        "gold_label": record["gold_label"],
        "prompt": record["prompt"],
        "detector_metadata": {
            "primary_detector": record["detector"],
            "extended_detectors": record["detector_extended"],
            "gold_label": record["gold_label"],
            "prompt_lang": record["lang"],
            "lang_split": record["lang_split"],
            "entity_key": record["entity_key"],
            "gold_label_source": 'attempt.notes["portia_multilingual"]["gold_label"]',
        },
    }


def collect_eval_suite():
    entries = []
    for probe_class in PROBE_CLASSES:
        probe = probe_class(config_root=garak._config)
        if len(probe.prompts) != len(probe.prompt_records):
            raise SystemExit(
                f"{probe_class.__name__}: prompt/record misalignment "
                f"({len(probe.prompts)} vs {len(probe.prompt_records)})"
            )
        for record, prompt in zip(probe.prompt_records, probe.prompts):
            if record["prompt"] != prompt:
                raise SystemExit(f"{probe_class.__name__}: record/prompt drift")
            entries.append(_bundle_entry(record))
    return entries


def collect_calibration_pool():
    entries = []
    for record in stress.build_calibration_records():
        entry = _bundle_entry(record)
        entry["pool"] = "calibration"
        entries.append(entry)
    return entries


def check_balance(entries):
    per_family_lang = defaultdict(Counter)
    per_pair_lang = defaultdict(Counter)
    for entry in entries:
        key = "hazard" if entry["is_positive_hazard"] else "benign"
        per_family_lang[(entry["family"], entry["lang"])][key] += 1
        per_pair_lang[(entry["twin_pair"], entry["lang"])][key] += 1
    problems = []
    for buckets, label in (
        (per_family_lang, "family"),
        (per_pair_lang, "twin pair"),
    ):
        for (name, lang), counts in sorted(buckets.items()):
            if counts["hazard"] != counts["benign"]:
                problems.append(
                    f"{label} {name} in {lang}: {counts['hazard']} hazard vs "
                    f"{counts['benign']} benign"
                )
    return problems


def check_language_splits(entries):
    problems = []
    for lang in stress.LANGS_CALIBRATION:
        if stress.LANG_SPLIT[lang] != "calibration_langs":
            problems.append(f"{lang} should be a calibration language")
    for lang in stress.LANGS_ZEROSHOT:
        if stress.LANG_SPLIT[lang] != "zeroshot_langs":
            problems.append(f"{lang} should be a zero-shot language")
    seen = {entry["lang"] for entry in entries}
    missing = set(stress.LANG_ORDER) - seen
    if missing:
        problems.append(f"languages absent from the eval suite: {sorted(missing)}")
    for entry in entries:
        if entry["lang_split"] != stress.LANG_SPLIT[entry["lang"]]:
            problems.append(f"{entry['id']}: wrong lang split tag")
    return problems


def check_no_leak(eval_entries, calibration_entries):
    problems = []
    eval_entities = {entry["detector_metadata"]["entity_key"] for entry in eval_entries}
    calibration_entities = {
        entry["detector_metadata"]["entity_key"] for entry in calibration_entries
    }
    shared_entities = eval_entities & calibration_entities
    if shared_entities:
        problems.append(f"entity overlap: {sorted(shared_entities)}")

    eval_prompts = {entry["prompt"] for entry in eval_entries}
    calibration_prompts = {entry["prompt"] for entry in calibration_entries}
    shared_prompts = eval_prompts & calibration_prompts
    if shared_prompts:
        problems.append(f"{len(shared_prompts)} identical prompts in both pools")

    calibration_langs = {entry["lang"] for entry in calibration_entries}
    unexpected = calibration_langs - set(stress.LANGS_CALIBRATION)
    if unexpected:
        problems.append(f"calibration pool leaked into languages {sorted(unexpected)}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    eval_entries = collect_eval_suite()
    calibration_entries = collect_calibration_pool()

    problems = []
    problems += check_balance(eval_entries)
    problems += check_language_splits(eval_entries)
    problems += check_no_leak(eval_entries, calibration_entries)
    by_family = Counter(entry["family"] for entry in eval_entries)
    for family, minimum in MINIMUMS.items():
        if by_family[family] < minimum:
            problems.append(f"{family} has {by_family[family]} prompts, need {minimum}")
    if len(eval_entries) < 640:
        problems.append(f"eval suite has {len(eval_entries)} prompts, need 640")
    if len({entry["id"] for entry in eval_entries}) != len(eval_entries):
        problems.append("duplicate prompt ids in the eval suite")
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        raise SystemExit(1)

    bundle = {
        "bundle": "garak_multilingual_stress_bundle",
        "bundle_version": "v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/export_portia_multilingual_bundle.py",
        "probe_module": "garak.probes.portia_multilingual_stress",
        "detector_module": "garak.detectors.portia_multilingual_epistemic",
        "languages": {
            lang: {
                "name": stress.LANG_NAMES[lang],
                "split": stress.LANG_SPLIT[lang],
            }
            for lang in stress.LANG_ORDER
        },
        "detectors": {
            "package_twins": stress.DETECTOR_PACKAGE,
            "presupposition_twins": stress.DETECTOR_PRESUPPOSITION,
            "defect_twins": stress.DETECTOR_DEFECT,
            "snowball_hostile_twins": stress.DETECTOR_SNOWBALL,
            "all_families": stress.DETECTOR_FIDELITY,
        },
        "counts": {
            "eval_total": len(eval_entries),
            "calibration_total": len(calibration_entries),
            "eval_by_family": dict(sorted(by_family.items())),
            "eval_by_language": dict(
                sorted(Counter(entry["lang"] for entry in eval_entries).items())
            ),
            "calibration_by_language": dict(
                sorted(Counter(entry["lang"] for entry in calibration_entries).items())
            ),
            "hazard_total": sum(
                1 for entry in eval_entries if entry["is_positive_hazard"]
            ),
            "benign_total": sum(
                1 for entry in eval_entries if not entry["is_positive_hazard"]
            ),
        },
        "checks": {
            "twin_balance": "exact 50/50 per family per language and per twin pair",
            "entity_overlap": 0,
            "prompt_overlap": 0,
            "calibration_languages": list(stress.LANGS_CALIBRATION),
            "zeroshot_languages": list(stress.LANGS_ZEROSHOT),
        },
        "calibration_pool": calibration_entries,
        "eval_suite": eval_entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=1)
        handle.write("\n")

    print(f"wrote {args.output}")
    print(
        f"eval_suite={len(eval_entries)} calibration_pool={len(calibration_entries)} "
        f"hazard={bundle['counts']['hazard_total']} benign={bundle['counts']['benign_total']}"
    )
    print(f"by_family={bundle['counts']['eval_by_family']}")
    print(f"by_language={bundle['counts']['eval_by_language']}")


if __name__ == "__main__":
    main()
