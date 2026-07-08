"""Deterministic metadata for source-demo coverage experiments."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from herosjourney.core.generator import GeneratedTask


Pair = Tuple[int, ...]


def _input_names(rule_spec: Dict[str, Any]) -> List[str]:
    return [spec["name"] for spec in rule_spec.get("inputs", [])]


def _input_n_values(rule_spec: Dict[str, Any]) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for spec in rule_spec.get("inputs", []):
        if "n_values" in spec:
            values[spec["name"]] = int(spec["n_values"])
    return values


def _task_pair(task: GeneratedTask, names: Sequence[str]) -> Pair:
    attrs = task.metadata.get("attributes", {})
    return tuple(int(attrs[name]) for name in names)


def _pair_records(tasks: Sequence[GeneratedTask], names: Sequence[str]) -> List[Dict[str, Any]]:
    records = []
    for task in tasks:
        attrs = {name: int(task.metadata.get("attributes", {}).get(name)) for name in names}
        records.append({
            "entity": task.metadata.get("entity_instance"),
            "attributes": attrs,
            "pair": [attrs[name] for name in names],
        })
    return records


def _pair_lists(pairs: Iterable[Pair]) -> List[List[int]]:
    return [list(pair) for pair in sorted(set(pairs))]


def _composition_info(rule_spec: Dict[str, Any]) -> Dict[str, Any]:
    comp = rule_spec.get("output", {}).get("composition")
    if isinstance(comp, str):
        return {"fn": comp}
    if isinstance(comp, dict):
        info = {"fn": comp.get("fn", "dict")}
        for key in (
            "selector",
            "base_fn",
            "base_fns",
            "override_attr",
            "override_value",
            "override_dim",
            "override_dim_value",
            "override_dims",
        ):
            if key in comp:
                info[key] = comp[key]
        return info
    return {"fn": None}


def _connected_bipartite(edges: set[Tuple[int, int]]) -> bool:
    if not edges:
        return False
    adj: Dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        l_node = f"0:{left}"
        r_node = f"1:{right}"
        adj[l_node].add(r_node)
        adj[r_node].add(l_node)
    nodes = set(adj)
    visited = {next(iter(nodes))}
    queue: deque[str] = deque(visited)
    while queue:
        node = queue.popleft()
        for nxt in adj[node]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return visited == nodes


def _degree_map(pairs: set[Pair], dim: int) -> Dict[int, set[int]]:
    other = 1 - dim
    out: Dict[int, set[int]] = defaultdict(set)
    for pair in pairs:
        out[pair[dim]].add(pair[other])
    return out


def _all_values_seen(
    observed: Dict[str, List[int]],
    n_values: Dict[str, int],
) -> bool:
    for name, n_val in n_values.items():
        if set(observed.get(name, [])) != set(range(n_val)):
            return False
    return True


def _heuristic_identifiable(
    task_type: Optional[str],
    rule_spec: Dict[str, Any],
    selected_pairs: set[Pair],
    all_source_pairs: set[Pair],
    observed_values: Dict[str, List[int]],
) -> Tuple[bool, Dict[str, Any]]:
    names = _input_names(rule_spec)
    n_values = _input_n_values(rule_spec)
    comp = _composition_info(rule_spec)
    fn = comp.get("fn")
    source_complete = bool(all_source_pairs) and selected_pairs == all_source_pairs

    checks: Dict[str, Any] = {
        "source_split_complete": source_complete,
        "all_input_values_seen": _all_values_seen(observed_values, n_values),
    }

    if len(names) < 2:
        return source_complete, checks

    two_dim_pairs = {(pair[0], pair[1]) for pair in selected_pairs}

    if task_type in {"additive", "compositional"}:
        deg0 = _degree_map(selected_pairs, 0)
        deg1 = _degree_map(selected_pairs, 1)
        min_degree_ok = (
            all(len(deg0.get(v, set())) >= 2 for v in range(n_values[names[0]]))
            and all(len(deg1.get(v, set())) >= 2 for v in range(n_values[names[1]]))
        )
        connected = _connected_bipartite(two_dim_pairs)
        checks.update({
            "min_two_partners_per_value": min_degree_ok,
            "coverage_graph_connected": connected,
        })
        return min_degree_ok and connected, checks

    if task_type == "conditional":
        selector = comp.get("selector", {})
        regime_map = selector.get("map") or []
        n_role = n_values.get(names[1], 0)
        class_to_roles = _degree_map(selected_pairs, 0)
        regime_to_roles: Dict[int, set[int]] = defaultdict(set)
        for class_value, regime in enumerate(regime_map):
            regime_to_roles[int(regime)].update(class_to_roles.get(class_value, set()))
        class_degree_ok = all(
            len(class_to_roles.get(class_value, set())) >= 2
            for class_value in range(len(regime_map))
        )
        regime_role_ok = all(
            regime_to_roles.get(int(regime), set()) == set(range(n_role))
            for regime in set(regime_map)
        )
        checks.update({
            "min_two_roles_per_class": class_degree_ok,
            "all_roles_seen_within_each_regime": regime_role_ok,
            "roles_by_regime": {
                str(regime): sorted(values)
                for regime, values in sorted(regime_to_roles.items())
            },
        })
        return class_degree_ok and regime_role_ok, checks

    if task_type in {"override", "proc_over"} or fn == "override":
        override_attr = comp.get("override_attr", names[1])
        override_value = int(comp.get("override_value", n_values.get(override_attr, 1) - 1))
        override_idx = names.index(override_attr)
        base_idx = 1 - override_idx
        base_name = names[base_idx]
        non_override_base_values = {
            pair[base_idx] for pair in selected_pairs if pair[override_idx] != override_value
        }
        override_seen = any(pair[override_idx] == override_value for pair in selected_pairs)
        base_values_ok = non_override_base_values == set(range(n_values[base_name]))
        checks.update({
            "override_value": override_value,
            "override_seen": override_seen,
            "all_base_values_seen_without_override": base_values_ok,
        })
        return override_seen and base_values_ok, checks

    if task_type in {"proc_add", "proc_comp", "proc_cond"}:
        return source_complete, checks

    return source_complete, checks


def summarize_source_coverage(
    rule_spec: Dict[str, Any],
    selected_tasks: Sequence[GeneratedTask],
    *,
    all_source_tasks: Optional[Sequence[GeneratedTask]] = None,
    task_type: Optional[str] = None,
    requested_source_demos: Optional[int] = None,
    source_subset_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize which rule-relevant source-demo components are present."""
    names = _input_names(rule_spec)
    n_values = _input_n_values(rule_spec)
    selected_pairs = [_task_pair(task, names) for task in selected_tasks]
    all_pairs = (
        [_task_pair(task, names) for task in all_source_tasks]
        if all_source_tasks is not None else selected_pairs
    )
    selected_set = set(selected_pairs)
    all_set = set(all_pairs)

    observed_values: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        observed_values[name] = sorted({pair[idx] for pair in selected_pairs})

    identifiable, checks = _heuristic_identifiable(
        task_type,
        rule_spec,
        selected_set,
        all_set,
        observed_values,
    )

    source_size = len(all_set)
    selected_size = len(selected_set)
    return {
        "task_type": task_type,
        "requested_source_demos": requested_source_demos,
        "actual_source_demos": len(selected_tasks),
        "source_subset_seed": source_subset_seed,
        "source_size_k_star": source_size,
        "source_pair_coverage": selected_size / source_size if source_size else 0.0,
        "input_names": names,
        "input_n_values": n_values,
        "composition": _composition_info(rule_spec),
        "selected_source": _pair_records(selected_tasks, names),
        "selected_pairs": _pair_lists(selected_set),
        "all_source_pairs": _pair_lists(all_set),
        "covered_input_values": observed_values,
        "heuristic_identifiable": identifiable,
        "coverage_checks": checks,
    }
