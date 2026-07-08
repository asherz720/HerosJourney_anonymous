#!/usr/bin/env python
"""Build the static data file consumed by site/index.html.

Extracts *structured* example data (not just the formatted text) for every
registered task, under both semantic and nonce surface conditions, and bundles
it together with the Table 6 leaderboard numbers from the paper.

Output: site/data.js  ->  defines window.HJ_TASKS and window.HJ_LEADERBOARD
so the page works opened directly from disk (file://) as well as on GitHub
Pages, with no fetch()/CORS problems and no build step.

Run from the repo root:  python experiments/build_site_data.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import random

from herosjourney import get_task
from herosjourney.core.elements import fill_elements, load_lexicons
from herosjourney.core.generator import (
    get_template, generate_tasks, _CANONICAL_PROPERTY_DISTRACTOR,
)
from herosjourney.core.demo_generator import generate_mixed_demos, generate_demos
from herosjourney.env.env import AdventureEnv
from herosjourney.world_info.actions import ACTION_REGISTRY


# Order + display metadata for the eight tasks. Keys are registry names.
TASKS = [
    ("additive",      "A-Add",   "Attribute",  "Additive"),
    ("compositional", "A-Comp",  "Attribute",  "Compositional"),
    ("conditional",   "A-Cond",  "Attribute",  "Conditional"),
    ("override",      "A-Over",  "Attribute",  "Override"),
    ("proc_add",      "P-Add",   "Procedural", "Additive (procedural)"),
    ("proc_comp",     "P-Comp",  "Procedural", "Compositional (procedural)"),
    ("proc_cond",     "P-Cond",  "Procedural", "Conditional (procedural)"),
    ("proc_over",     "P-Over",  "Procedural", "Override (procedural)"),
]

SEED = 0
NUM_DISTRACTORS = 2  # distractor demos mixed into each variant


def _distractor_tasks(spec, use_nonce, dist_spec, dist_sem, dist_nonce):
    """Generate a few distractor demos (arbitrary item assignment, no pattern).

    Uses the flat distractor template, matching the evaluation pipeline. Returns
    [] if the framework can't build distractors for this task.
    """
    try:
        tmpl = get_template(spec.distractor_template or spec.template_name)
        delems = fill_elements(dist_spec, dist_sem, dist_nonce, seed=SEED)
        all_d = generate_tasks(delems, process=None, split=None, use_nonce=use_nonce,
                               task_type="distractor", template_name=tmpl.name, seed=SEED)
        if NUM_DISTRACTORS >= len(all_d):
            return all_d
        return random.Random(SEED).sample(all_d, NUM_DISTRACTORS)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"    (no distractors: {type(exc).__name__}: {exc})")
        return []


# ---------------------------------------------------------------------------
# Structured extraction helpers
# ---------------------------------------------------------------------------

def _entity_record(task) -> Dict[str, Any]:
    """Pull (name, ordered attributes, location) off a task tree.

    Mirrors the node-scan in demo_generator.build_world_listing so the page
    shows exactly what the world listing shows.
    """
    entity_node = None
    for node in task.tree.nodes.values():
        if "attribute_names" in node.properties:
            entity_node = node
            break
    if entity_node is None:
        entity_node = task.tree.nodes[task.tree.root_id]

    props = entity_node.properties
    names = props.get("attribute_names") or task.metadata.get("attribute_names", [])
    values = props.get("attribute_values") or task.metadata.get("attribute_values", [])
    location = props.get("location") or task.metadata.get("entity_location")

    return {
        "name": entity_node.argument,
        "attrs": [{"name": n, "value": v} for n, v in zip(names, values) if n],
        "location": location,
    }


def _collect_items(tasks) -> List[Dict[str, Any]]:
    """Unique acquisition items (buy/get) across the given tasks."""
    seen: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        for node in task.tree.nodes.values():
            action = node.meta.get("incoming_edge")
            adef = ACTION_REGISTRY.get(action) if action else None
            if adef and adef.is_acquisition:
                name = node.argument
                if name not in seen:
                    seen[name] = {
                        "name": name,
                        "location": node.properties.get("location", ""),
                        "cost": node.properties.get("cost"),
                    }
    return [seen[k] for k in sorted(seen)]


def _hidden_steps(task) -> List[Dict[str, str]]:
    """The ordered steps the agent must *infer* (those hidden from its rules).

    For attribute tasks this is the buy step (which item); for procedural tasks
    it is the perform/drink ritual step(s). Derived by matching the solution
    against rules_to_skip (the multiset of hidden argument strings).
    """
    skip = list(task.rules_to_skip)
    steps: List[Dict[str, str]] = []
    for step in task.tree.get_solution():
        verb, _, arg = step.partition(" ")
        if arg in skip:
            steps.append({"action": verb, "correct": arg})
            skip.remove(arg)  # consume one occurrence (handles repeats)
    return steps


def _hidden_signature(task) -> str:
    """Readable string for the full hidden requirement of an entity.

    Collapses consecutive identical steps into 'action arg ×N' so additive
    procedural tasks (perform ritual N times) read cleanly. This string is the
    quiz answer; the set of distinct signatures across entities forms the
    multiple-choice options.
    """
    steps = _hidden_steps(task)
    groups: List[list] = []  # [[(action, arg), count], ...]
    for s in steps:
        key = (s["action"], s["correct"])
        if groups and groups[-1][0] == key:
            groups[-1][1] += 1
        else:
            groups.append([key, 1])
    parts = []
    for (act, arg), n in groups:
        parts.append(f"{act} {arg}" + (f" ×{n}" if n > 1 else ""))
    return ", then ".join(parts) if parts else "(nothing extra)"


def _visible_rules(task) -> str:
    """The rules text the agent actually sees for this entity (hidden steps removed)."""
    env = AdventureEnv([(task.tree, task.tree.root_id)],
                       initial_currency=500, initial_location="GameStart")
    return env.reset(tree_index=0, initial_currency=500, initial_location="GameStart",
                     seed=SEED, rules_to_skip=task.rules_to_skip, task_label="Task")


def _demo_record(demo) -> Dict[str, Any]:
    """A demonstration episode as structured steps."""
    return {
        "entity_name": demo.entity_name,
        "goal": demo.goal,
        "rules_text": demo.rules_text,
        "is_distractor": bool(demo.metadata.get("is_distractor")),
        "steps": [{"cmd": cmd, "result": res} for cmd, res in demo.trace],
    }


def build_variant(spec, use_nonce: bool, sem_lex, nonce_lex,
                  dist_spec, dist_sem, dist_nonce) -> Dict[str, Any]:
    """All structured example data for one task under one surface condition."""
    with open(spec.rules) as f:
        rule = json.load(f)
    elements = fill_elements(rule, sem_lex, nonce_lex, seed=SEED, split_spec=spec.split)

    source = spec.gen_fn(elements, split="source", use_nonce=use_nonce)
    gen = spec.gen_fn(elements, split="gen", use_nonce=use_nonce)
    distractors = _distractor_tasks(spec, use_nonce, dist_spec, dist_sem, dist_nonce)

    # Include gen tasks so the world listing/items cover every possible item,
    # including ones only bought in gen combinations (matches what the agent sees).
    demos = generate_mixed_demos(source, distractor_tasks=distractors,
                                 gen_tasks=gen, seed=SEED)

    # World listing matches the real benchmark: it lists every entity and item,
    # property + distractor, so distractors are indistinguishable up front. The
    # demo episodes (below) are what reveal which ones carry a pattern.
    listing_tasks = source + gen + distractors
    entities = [_entity_record(t) for t in listing_tasks]
    items = _collect_items(listing_tasks)
    distractor_names = [_entity_record(t)["name"] for t in distractors]
    gen_names = [_entity_record(t)["name"] for t in gen]

    # Full executed answer trace for each held-out entity, so the page can let
    # the reader step through it (like a source episode) after attempting it.
    gen_demos = generate_demos(gen, seed=SEED)
    gen_demo_by_name = {d.entity_name: d for d in gen_demos}

    gen_entities = []
    for t in gen:
        rec = _entity_record(t)
        rec["hidden_steps"] = _hidden_steps(t)
        rec["hidden_signature"] = _hidden_signature(t)
        rec["solution"] = t.tree.get_solution()
        rec["visible_rules"] = _visible_rules(t)
        d = gen_demo_by_name.get(rec["name"])
        rec["trace"] = [{"cmd": c, "result": r} for c, r in d.trace] if d else []
        gen_entities.append(rec)

    return {
        "world": {"entities": entities, "items": items,
                  "distractor_names": distractor_names, "gen_names": gen_names},
        "demos": [_demo_record(d) for d in demos],
        "gen_entities": gen_entities,
    }


# ---------------------------------------------------------------------------
# Leaderboard — Table 6 (Appendix D) of the paper.
# ECSR and RV (RV normalised to 0-1) per model and task. Verified against the
# PDF. Task key order matches TASKS above.
# ---------------------------------------------------------------------------

TASK_KEYS = ["A-Add", "A-Comp", "A-Cond", "A-Over", "P-Add", "P-Comp", "P-Cond", "P-Over"]

# Each row: model -> [ (ecsr, rv) per task, in TASK_KEYS order ].
LEADERBOARD_ROWS = {
    "Qwen3.5-27B":          [(0.22, 0.12), (0.42, 0.30), (0.29, 0.03), (0.33, 0.07),
                             (0.41, 0.00), (0.01, 0.00), (0.26, 0.03), (0.17, 0.03)],
    "Gemini-3.1-Flash":     [(0.49, 0.47), (0.45, 0.47), (0.29, 0.00), (0.60, 0.00),
                             (0.48, 0.10), (0.00, 0.03), (0.05, 0.00), (0.18, 0.00)],
    "GPT-OSS-120B":         [(0.85, 0.47), (0.91, 0.80), (0.38, 0.20), (0.70, 0.42),
                             (0.45, 0.45), (0.03, 0.42), (0.26, 0.00), (0.47, 0.53)],
    "GPT-5.4-mini":         [(0.65, 0.47), (0.93, 0.60), (0.35, 0.00), (0.83, 0.42),
                             (0.35, 0.42), (0.12, 0.07), (0.16, 0.00), (0.21, 0.30)],
    "Llama-4":              [(0.13, 0.03), (0.58, 0.20), (0.30, 0.00), (0.48, 0.00),
                             (0.42, 0.00), (0.00, 0.00), (0.00, 0.00), (0.04, 0.10)],
    "Olmo3.1-32B-Instruct": [(0.36, 0.05), (0.20, 0.00), (0.23, 0.00), (0.25, 0.00),
                             (0.31, 0.00), (0.01, 0.00), (0.00, 0.00), (0.05, 0.05)],
    "Humans (avg.)":        [(0.63, 0.71), (0.75, 0.79), (0.57, 0.64), (0.78, 1.00),
                             (0.61, 0.71), (0.57, 0.57), (0.62, 0.57), (0.58, 0.71)],
}


def build_leaderboard() -> Dict[str, Any]:
    rows = []
    for model, cells in LEADERBOARD_ROWS.items():
        scores = {}
        for key, (ecsr, rv) in zip(TASK_KEYS, cells):
            scores[key] = {"ecsr": ecsr, "rv": rv}
        # Convenience averages for sorting / overview.
        attr = [c for k, c in zip(TASK_KEYS, cells) if k.startswith("A-")]
        proc = [c for k, c in zip(TASK_KEYS, cells) if k.startswith("P-")]
        rows.append({
            "model": model,
            "is_human": model.startswith("Humans"),
            "scores": scores,
            "avg_ecsr": round(sum(c[0] for c in cells) / len(cells), 3),
            "avg_rv": round(sum(c[1] for c in cells) / len(cells), 3),
            "attr_ecsr": round(sum(c[0] for c in attr) / len(attr), 3),
            "proc_ecsr": round(sum(c[0] for c in proc) / len(proc), 3),
        })
    return {
        "task_keys": TASK_KEYS,
        "task_labels": {tk: label for _, tk, _, label in TASKS},
        "task_family": {tk: fam for _, tk, fam, _ in TASKS},
        "metrics": ["ecsr", "rv"],
        "rows": rows,
        "source": "Anonymous manuscript, Appendix D, Table 6.",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sem_lex, nonce_lex = load_lexicons()
    with open(_CANONICAL_PROPERTY_DISTRACTOR) as f:
        dist_spec = json.load(f)
    dist_sem, dist_nonce = load_lexicons(dist_spec.get("lexicon", "default"))

    tasks_out: Dict[str, Any] = {}
    order: List[str] = []
    for name, key, family, label in TASKS:
        spec = get_task(name)
        tasks_out[key] = {
            "name": name,
            "key": key,
            "family": family,
            "label": label,
            "description": getattr(spec, "description", "") or "",
            "correct_rule": getattr(spec, "correct_rule", "") or "",
            "variants": {
                "semantic": build_variant(spec, False, sem_lex, nonce_lex,
                                          dist_spec, dist_sem, dist_nonce),
                "nonce": build_variant(spec, True, sem_lex, nonce_lex,
                                       dist_spec, dist_sem, dist_nonce),
            },
        }
        order.append(key)
        print(f"  {key:8s} ({name}) ok")

    payload_tasks = {"order": order, "tasks": tasks_out}
    leaderboard = build_leaderboard()

    out_dir = Path(__file__).resolve().parent.parent / "site"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "data.js"
    with open(out_path, "w") as f:
        f.write("// Auto-generated by experiments/build_site_data.py -- do not edit by hand.\n")
        f.write("window.HJ_TASKS = ")
        json.dump(payload_tasks, f, indent=1)
        f.write(";\n\n")
        f.write("window.HJ_LEADERBOARD = ")
        json.dump(leaderboard, f, indent=1)
        f.write(";\n")

    print(f"\nWrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
