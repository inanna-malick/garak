"""Export the multilingual stress corpus as a standalone JSON bundle.

The bundle carries two pools with strictly disjoint contents:

* ``calibration_pool`` -- `en`, `zh` and `es` prompts only, intended for
  compiling a weight-space patch.
* ``eval_suite`` -- the four balanced twin families across all eight
  languages, each entry tagged with its language split, gold label and
  detector wiring.

Both pools are built from the probe classes in
``garak.probes.portia_multilingual_stress``, plus a deterministic seeded
slice from ``garak.probes.portia_parametric_fuzzer`` whose gold labels come
from executed oracles rather than a hand-written answer key. The parametric
slice is generated from its own template tables, so the two pools still share
no entity, no prompt and no prompt frame. The checks below fail the export
rather than emit a skewed, leaking or oracle-less corpus.

Usage::

    python tools/export_portia_multilingual_bundle.py [--output PATH]
        [--seed 20261113] [--parametric-per-family-per-lang 6]
        [--parametric-calibration-per-family-per-lang 3]
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import garak._config
import garak.probes.portia_multilingual_stress as stress
import garak.probes.portia_parametric_fuzzer as fuzzer

DEFAULT_OUTPUT = Path("portia_multilingual_stress_bundle_v1.json")

PROBE_CLASSES = (
    stress.PortiaMultilingualPackageTwins,
    stress.PortiaMultilingualPresuppositionTwins,
    stress.PortiaMultilingualDefectTwins,
    stress.PortiaMultilingualSnowballHostileTwins,
)

PARAMETRIC_PROBE_CLASSES = fuzzer.PARAMETRIC_PROBES

MINIMUMS = {
    "package_twins": 320,
    "presupposition_twins": 192,
    "defect_twins": 96,
    "snowball_hostile_twins": 128,
}


def _bundle_entry(record):
    metadata = {
        "primary_detector": record["detector"],
        "extended_detectors": record["detector_extended"],
        "gold_label": record["gold_label"],
        "prompt_lang": record["lang"],
        "lang_split": record["lang_split"],
        "entity_key": record["entity_key"],
        "template_id": record["template_id"],
        "gold_label_source": 'attempt.notes["portia_multilingual"]["gold_label"]',
    }
    # the parametric generator carries the oracle evidence with the prompt
    metadata.update(record.get("detector_metadata") or {})
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
        "template_id": record["template_id"],
        "generator": record.get("generator", "garak.probes.portia_multilingual_stress"),
        "detector_metadata": metadata,
    }


def collect_eval_suite(seed=None, per_family_per_lang=None):
    """Return the eval entries: curated families first, then the seeded slice."""
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
    if not PARAMETRIC_PROBE_CLASSES:
        return entries
    seed = fuzzer.DEFAULT_SEED if seed is None else seed
    per_family_per_lang = (
        fuzzer.DEFAULT_PER_FAMILY_PER_LANG
        if per_family_per_lang is None
        else per_family_per_lang
    )
    for probe_class in PARAMETRIC_PROBE_CLASSES:
        records = fuzzer.build_parametric_records(
            probe_class.family, seed, per_family_per_lang, fuzzer.EVAL_SPLIT
        )
        if (seed, per_family_per_lang) == (
            fuzzer.DEFAULT_SEED,
            fuzzer.DEFAULT_PER_FAMILY_PER_LANG,
        ):
            # the shipped probe must send exactly what the generator emits
            probe = probe_class(config_root=garak._config)
            if [record["prompt"] for record in records] != list(probe.prompts):
                raise SystemExit(
                    f"{probe_class.__name__}: probe prompts drifted from the "
                    f"parametric generator"
                )
        for record in records:
            entries.append(_bundle_entry(record))
    return entries


def collect_calibration_pool(seed=None, per_family_per_lang=None):
    """Return the calibration entries: curated frames plus a seeded slice."""
    entries = []
    for record in stress.build_calibration_records():
        entry = _bundle_entry(record)
        entry["pool"] = "calibration"
        entries.append(entry)
    seed = fuzzer.DEFAULT_SEED if seed is None else seed
    per_family_per_lang = (
        fuzzer.DEFAULT_CALIBRATION_PER_FAMILY_PER_LANG
        if per_family_per_lang is None
        else per_family_per_lang
    )
    for record in fuzzer.generate_parametric_twins(
        seed, per_family_per_lang, fuzzer.CALIBRATION_SPLIT
    ):
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

    shared_frames = template_overlap(eval_entries, calibration_entries)
    if shared_frames:
        problems.append(
            f"prompt-frame overlap between pools: {len(shared_frames)} shared templates"
        )
    return problems


def template_overlap(eval_entries, calibration_entries):
    """Return the prompt frames (by content digest) rendered in both pools.

    Two pools that reuse the same template -- even with a different noun in the
    slot -- collide here, which is what exact-prompt comparison alone misses."""
    eval_frames = {stress.template_digest(entry["template_id"]) for entry in eval_entries}
    calibration_frames = {
        stress.template_digest(entry["template_id"]) for entry in calibration_entries
    }
    return eval_frames & calibration_frames


def _pair_key(entry):
    """Group a hazard entry and its benign twin under one key."""
    identifier = entry["id"]
    return identifier[:-4] if identifier.endswith(("-haz", "-ben")) else identifier


def check_parametric_oracles(entries, label):
    """Every parametric twin must carry the evidence its gold label claims.

    Code-defect hazards must show a recorded counterexample and their sound
    twins must show none; graph twins must differ by exactly one edge over the
    same node set; blends must sit at the required edit distance from the
    snapshot; relational pairs must differ only in the filled slot.
    """
    problems = []
    parametric = [
        entry
        for entry in entries
        if entry.get("generator") == "portia_parametric_fuzzer"
    ]
    if not parametric:
        return problems
    by_pair = defaultdict(list)
    for entry in parametric:
        metadata = entry["detector_metadata"]
        if metadata.get("verification") != fuzzer.VERIFICATION_KINDS[entry["twin_pair"]]:
            problems.append(
                f"{entry['id']}: verification strength is not disclosed for its twin pair"
            )
        if entry["twin_pair"] == fuzzer.TWIN_PAIR_CODE:
            witness = metadata.get("counterexample_witness")
            if entry["is_positive_hazard"]:
                if not witness:
                    problems.append(f"{entry['id']}: defective twin has no counterexample")
                    continue
                for field in ("reason", "args", "expected", "observed", "mutation"):
                    if not witness.get(field):
                        problems.append(f"{entry['id']}: witness lacks {field}")
            elif witness is not None:
                problems.append(f"{entry['id']}: sound twin carries a counterexample")
            if metadata.get("oracle_inputs_checked", 0) < fuzzer.MIN_ORACLE_INPUTS:
                problems.append(
                    f"{entry['id']}: only {metadata.get('oracle_inputs_checked')} "
                    f"oracle inputs"
                )
        elif entry["twin_pair"] == fuzzer.TWIN_PAIR_NAMESPACE:
            if entry["is_positive_hazard"]:
                if metadata.get("distance_to_nearest_real", 0) < 2:
                    problems.append(
                        f"{entry['id']}: blend sits closer than two edits to a real name"
                    )
            elif metadata.get("distance_to_nearest_real") != 0:
                problems.append(f"{entry['id']}: real twin is not a snapshot member")
        by_pair[_pair_key(entry)].append(entry)

    for key, pair in sorted(by_pair.items()):
        hazards = [entry for entry in pair if entry["is_positive_hazard"]]
        benign = [entry for entry in pair if not entry["is_positive_hazard"]]
        if len(hazards) != 1 or len(benign) != 1:
            problems.append(f"{key}: twin pair does not have one hazard and one benign")
            continue
        hazard, safe = hazards[0], benign[0]
        if hazard["twin_pair"] != safe["twin_pair"]:
            problems.append(f"{key}: twins disagree on their twin pair")
            continue
        if hazard["twin_pair"] == fuzzer.TWIN_PAIR_GRAPH:
            hazard_meta, safe_meta = hazard["detector_metadata"], safe["detector_metadata"]
            if safe_meta["edge_count"] != hazard_meta["edge_count"] + 1:
                problems.append(f"{key}: bridge twin is not exactly one edge longer")
            if safe_meta["nodes"] != hazard_meta["nodes"]:
                problems.append(f"{key}: graph twins disagree on their node set")
            if hazard_meta["reachable"] or not safe_meta["reachable"]:
                problems.append(f"{key}: graph reachability contradicts the gold label")
            if not 3 <= (safe_meta["shortest_path_length"] or 0) <= 6:
                problems.append(f"{key}: bridge distance is outside [3, 6]")
        elif hazard["twin_pair"] == fuzzer.TWIN_PAIR_RELATIONAL:
            hazard_meta, safe_meta = hazard["detector_metadata"], safe["detector_metadata"]
            left = hazard["prompt"].replace(hazard_meta["filler"], "\x00")
            right = safe["prompt"].replace(safe_meta["filler"], "\x00")
            if left != right:
                problems.append(f"{key}: relational twins differ outside the filled slot")
            if hazard_meta["operator"] != safe_meta["operator"]:
                problems.append(f"{key}: relational twins disagree on their operator")
        elif hazard["twin_pair"] == fuzzer.TWIN_PAIR_CODE:
            metadata = hazard["detector_metadata"]
            if metadata["ast_mutation"] == "none":
                problems.append(f"{key}: defective twin records no mutation")
    return problems


def check_parametric_seeded_regularity(entries, label):
    """The seeded slice must balance 50/50 per pair and language by itself."""
    problems = []
    per_pair_lang = defaultdict(Counter)
    for entry in entries:
        if entry.get("generator") != "portia_parametric_fuzzer":
            continue
        key = "hazard" if entry["is_positive_hazard"] else "benign"
        per_pair_lang[(entry["lang"], entry["twin_pair"])][key] += 1
    for (lang, pair), counts in sorted(per_pair_lang.items()):
        if counts["hazard"] != counts["benign"]:
            problems.append(
                f"{label} parametric {pair} in {lang}: {counts['hazard']} hazard vs "
                f"{counts['benign']} benign"
            )
    return problems


def check_calibration_coverage(entries):
    """Every calibration subfamily must be present and balanced in en/zh/es."""
    problems = []
    per_lang_pair = defaultdict(Counter)
    per_lang_subfamily = defaultdict(Counter)
    for entry in entries:
        key = "hazard" if entry["is_positive_hazard"] else "benign"
        per_lang_pair[(entry["lang"], entry["twin_pair"])][key] += 1
        per_lang_subfamily[(entry["lang"], entry["subfamily"])][key] += 1
    for (lang, pair), counts in sorted(per_lang_pair.items()):
        if pair == "continuation":
            # neutral background prompts have no hazard twin by design
            continue
        if counts["hazard"] != counts["benign"]:
            problems.append(
                f"calibration pair {pair} in {lang}: {counts['hazard']} hazard vs "
                f"{counts['benign']} benign"
            )
    per_lang_totals = defaultdict(Counter)
    for (lang, subfamily), counts in per_lang_subfamily.items():
        for key, value in counts.items():
            per_lang_totals[subfamily][key] += value
    required = {
        "unreal_python",
        "real_niche_python",
        "false_presupposition",
        "counterintuitive_true_presupposition",
        "disconnected_graph_gold_no",
        "connected_graph_gold_yes",
        "vulnerable_or_buggy_snippet",
        "patched_sound_snippet",
        "benign_continuation",
    }
    missing = required - set(per_lang_totals)
    if missing:
        problems.append(f"calibration subfamilies absent: {sorted(missing)}")
    for subfamily in sorted(required & set(per_lang_totals)):
        langs = {
            lang for (lang, name) in per_lang_subfamily if name == subfamily
        }
        if langs != set(stress.LANGS_CALIBRATION):
            problems.append(
                f"calibration {subfamily} languages {sorted(langs)} are not "
                f"{list(stress.LANGS_CALIBRATION)}"
            )
        counts = {tuple(sorted(per_lang_subfamily[(lang, subfamily)].items())) for lang in langs}
        if len(counts) > 1:
            problems.append(f"calibration {subfamily} is unbalanced across languages")
    return problems


def build_bundle(eval_entries, calibration_entries, seed=None, per_family_per_lang=None):
    """Assemble the export payload from the two already-collected pools."""
    by_family = Counter(entry["family"] for entry in eval_entries)
    parametric_eval = [
        entry
        for entry in eval_entries
        if entry.get("generator") == "portia_parametric_fuzzer"
    ]
    parametric_calibration = [
        entry
        for entry in calibration_entries
        if entry.get("generator") == "portia_parametric_fuzzer"
    ]
    bundle = {
        "bundle": "garak_multilingual_stress_bundle",
        "bundle_version": "v1",
        "generated": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/export_portia_multilingual_bundle.py",
        "probe_module": "garak.probes.portia_multilingual_stress",
        "parametric_module": "garak.probes.portia_parametric_fuzzer",
        "detector_module": "garak.detectors.portia_multilingual_epistemic",
        "parametric": {
            "seed": fuzzer.DEFAULT_SEED if seed is None else seed,
            "per_family_per_lang": (
                fuzzer.DEFAULT_PER_FAMILY_PER_LANG
                if per_family_per_lang is None
                else per_family_per_lang
            ),
            "calibration_per_family_per_lang": (
                fuzzer.DEFAULT_CALIBRATION_PER_FAMILY_PER_LANG
            ),
            "twin_pairs": list(fuzzer.PARAMETRIC_TWIN_PAIRS),
            "verification": dict(fuzzer.VERIFICATION_KINDS),
            "oracles": {
                fuzzer.TWIN_PAIR_CODE: "independent reference function, executed",
                fuzzer.TWIN_PAIR_GRAPH: "exact breadth-first search",
                fuzzer.TWIN_PAIR_NAMESPACE: (
                    "vendored package snapshot: collision and edit distance"
                ),
                fuzzer.TWIN_PAIR_RELATIONAL: (
                    "authored tuple table with type-matched fillers"
                ),
            },
            "min_oracle_inputs": fuzzer.MIN_ORACLE_INPUTS,
        },
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
            "calibration_by_subfamily": dict(
                sorted(
                    Counter(entry["subfamily"] for entry in calibration_entries).items()
                )
            ),
            "hazard_total": sum(
                1 for entry in eval_entries if entry["is_positive_hazard"]
            ),
            "benign_total": sum(
                1 for entry in eval_entries if not entry["is_positive_hazard"]
            ),
            "parametric_eval_total": len(parametric_eval),
            "parametric_eval_pairs_total": len(parametric_eval) // 2,
            "parametric_calibration_total": len(parametric_calibration),
            "parametric_eval_by_twin_pair": dict(
                sorted(Counter(entry["twin_pair"] for entry in parametric_eval).items())
            ),
        },
        "checks": {
            "twin_balance": "exact 50/50 per family per language and per twin pair",
            "entity_overlap": 0,
            "prompt_overlap": 0,
            "template_id_overlap": 0,
            "calibration_languages": list(stress.LANGS_CALIBRATION),
            "zeroshot_languages": list(stress.LANGS_ZEROSHOT),
            "parametric_oracles": "every defective twin carries an executed witness",
        },
        "calibration_pool": calibration_entries,
        "eval_suite": eval_entries,
    }

    return bundle


def write_bundle(bundle, output):
    """Write ``bundle`` to ``output`` as UTF-8 JSON and return the path."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=fuzzer.DEFAULT_SEED)
    parser.add_argument(
        "--parametric-per-family-per-lang",
        type=int,
        default=fuzzer.DEFAULT_PER_FAMILY_PER_LANG,
    )
    parser.add_argument(
        "--parametric-calibration-per-family-per-lang",
        type=int,
        default=fuzzer.DEFAULT_CALIBRATION_PER_FAMILY_PER_LANG,
    )
    args = parser.parse_args()

    eval_entries = collect_eval_suite(args.seed, args.parametric_per_family_per_lang)
    calibration_entries = collect_calibration_pool(
        args.seed, args.parametric_calibration_per_family_per_lang
    )

    problems = []
    problems += check_balance(eval_entries)
    problems += check_language_splits(eval_entries)
    problems += check_no_leak(eval_entries, calibration_entries)
    problems += check_calibration_coverage(calibration_entries)
    problems += check_parametric_oracles(eval_entries, "eval")
    problems += check_parametric_oracles(calibration_entries, "calibration")
    problems += check_parametric_seeded_regularity(eval_entries, "eval")
    problems += check_parametric_seeded_regularity(calibration_entries, "calibration")
    by_family = Counter(entry["family"] for entry in eval_entries)
    for family, minimum in MINIMUMS.items():
        if by_family[family] < minimum:
            problems.append(f"{family} has {by_family[family]} prompts, need {minimum}")
    if len(eval_entries) < 1280:
        problems.append(f"eval suite has {len(eval_entries)} prompts, need 1280")
    if len({entry["id"] for entry in eval_entries}) != len(eval_entries):
        problems.append("duplicate prompt ids in the eval suite")
    if len({entry["id"] for entry in calibration_entries}) != len(calibration_entries):
        problems.append("duplicate prompt ids in the calibration pool")
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        raise SystemExit(1)

    bundle = build_bundle(
        eval_entries, calibration_entries, args.seed, args.parametric_per_family_per_lang
    )

    write_bundle(bundle, args.output)

    print(f"wrote {args.output}")
    print(
        f"eval_suite={len(eval_entries)} calibration_pool={len(calibration_entries)} "
        f"hazard={bundle['counts']['hazard_total']} benign={bundle['counts']['benign_total']}"
    )
    print(f"by_family={bundle['counts']['eval_by_family']}")
    print(f"by_language={bundle['counts']['eval_by_language']}")
    print(f"parametric={bundle['counts']['parametric_eval_by_twin_pair']}")


if __name__ == "__main__":
    main()
