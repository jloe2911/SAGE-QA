import itertools
import json
import math
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple


# =========================================================
# Parsing
# =========================================================


def normalize_ws(text: str) -> str:
    return " ".join(str(text).strip().split())


def local_name(uri: str) -> str:
    if not uri:
        return ""
    if uri.startswith("<") and uri.endswith(">"):
        uri = uri[1:-1]
    if "#" in uri:
        return uri.split("#")[-1]
    if "/" in uri:
        return uri.rstrip("/").split("/")[-1]
    return uri


def extract_query_signature(question: str, sparql_query: str) -> Dict:
    """
    Extract query entities/properties in a generic way from simple ASK/SELECT triple patterns.
    """
    entities = set()
    properties = set()

    q = normalize_ws(sparql_query)

    # Extract <uri> tokens
    uris = re.findall(r"<([^>]+)>", q)
    locals_ = [local_name(u) for u in uris]

    # Heuristic: in triple patterns, 1st and 3rd are entities/classes, 2nd is property
    # This is generic enough for the current benchmark formats.
    if len(locals_) >= 3:
        for i in range(0, len(locals_) - 2, 3):
            s = locals_[i]
            p = locals_[i + 1]
            o = locals_[i + 2]
            if s and not s.startswith("?"):
                entities.add(s)
            if p and not p.startswith("?"):
                properties.add(p)
            if o and not o.startswith("?"):
                entities.add(o)

    return {
        "question": question,
        "sparql_query": sparql_query,
        "query_entities": sorted(entities),
        "query_properties": sorted(properties),
    }


@dataclass
class ParsedAxiom:
    raw_axiom: str
    axiom_type: str  # fact | rule
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    rule_type: Optional[str] = None
    property: Optional[str] = None
    property1: Optional[str] = None
    property2: Optional[str] = None
    subproperty: Optional[str] = None
    superproperty: Optional[str] = None

    def entities(self) -> Set[str]:
        out = set()
        if self.subject:
            out.add(self.subject)
        if self.object:
            out.add(self.object)
        return out

    def properties(self) -> Set[str]:
        out = set()
        for x in [
            self.predicate,
            self.property,
            self.property1,
            self.property2,
            self.subproperty,
            self.superproperty,
        ]:
            if x:
                out.add(x)
        return out


def parse_axiom(raw_axiom: str) -> ParsedAxiom:
    ax = normalize_ws(raw_axiom)

    # -------- rules --------
    m = re.match(r"^SymmetricObjectProperty\(([^)]+)\)$", ax)
    if m:
        return ParsedAxiom(
            raw_axiom=ax,
            axiom_type="rule",
            rule_type="SymmetricObjectProperty",
            property=m.group(1),
        )

    m = re.match(r"^TransitiveObjectProperty\(([^)]+)\)$", ax)
    if m:
        return ParsedAxiom(
            raw_axiom=ax,
            axiom_type="rule",
            rule_type="TransitiveObjectProperty",
            property=m.group(1),
        )

    m = re.match(r"^FunctionalObjectProperty\(([^)]+)\)$", ax)
    if m:
        return ParsedAxiom(
            raw_axiom=ax,
            axiom_type="rule",
            rule_type="FunctionalObjectProperty",
            property=m.group(1),
        )

    m = re.match(r"^InverseObjectProperties\(([^,]+),([^)]+)\)$", ax)
    if m:
        return ParsedAxiom(
            raw_axiom=ax,
            axiom_type="rule",
            rule_type="InverseObjectProperties",
            property1=m.group(1).strip(),
            property2=m.group(2).strip(),
        )

    m = re.match(r"^SubObjectPropertyOf\(([^,]+),([^)]+)\)$", ax)
    if m:
        return ParsedAxiom(
            raw_axiom=ax,
            axiom_type="rule",
            rule_type="SubObjectPropertyOf",
            subproperty=m.group(1).strip(),
            superproperty=m.group(2).strip(),
        )

    m = re.match(r"^EquivalentObjectProperties\(([^,]+),([^)]+)\)$", ax)
    if m:
        return ParsedAxiom(
            raw_axiom=ax,
            axiom_type="rule",
            rule_type="EquivalentObjectProperties",
            property1=m.group(1).strip(),
            property2=m.group(2).strip(),
        )

    # -------- facts --------
    parts = ax.split()
    if len(parts) == 3:
        return ParsedAxiom(
            raw_axiom=ax,
            axiom_type="fact",
            subject=parts[0],
            predicate=parts[1],
            object=parts[2],
        )

    return ParsedAxiom(raw_axiom=ax, axiom_type="unknown")


# =========================================================
# Symbolic composer
# =========================================================


class SymbolicComposer:
    def __init__(
        self,
        top_k_subgraph: int = 0,
        edge_threshold: float = 0.50,
        connectivity_bonus: float = 0.30,
        fact_rule_mix_bonus: float = 0.20,
        query_alignment_bonus: float = 0.18,
        bridge_bonus: float = 0.22,
        redundancy_penalty: float = 0.05,
        beam_width: int = 96,
        max_combinations: int = 5000,
        learned_weights_path: str = "",
    ):
        self.top_k_subgraph = top_k_subgraph
        self.edge_threshold = edge_threshold
        self.connectivity_bonus = connectivity_bonus
        self.fact_rule_mix_bonus = fact_rule_mix_bonus
        self.query_alignment_bonus = query_alignment_bonus
        self.bridge_bonus = bridge_bonus
        self.redundancy_penalty = redundancy_penalty
        self.beam_width = beam_width
        self.max_combinations = max_combinations

        if learned_weights_path:
            self.load_learned_weights(learned_weights_path)

    def load_learned_weights(self, path: str) -> None:
        """
        Load offline-fitted symbolic weights from JSON.

        Expected keys match the constructor names:
        connectivity_bonus, fact_rule_mix_bonus, query_alignment_bonus,
        bridge_bonus, redundancy_penalty.
        """
        with open(path, "r", encoding="utf-8") as f:
            weights = json.load(f)

        for name in (
            "connectivity_bonus",
            "fact_rule_mix_bonus",
            "query_alignment_bonus",
            "bridge_bonus",
            "redundancy_penalty",
        ):
            if name in weights:
                setattr(self, name, float(weights[name]))

    # -----------------------------------------------------
    # Edge building
    # -----------------------------------------------------
    def _edge_between(self, a: ParsedAxiom, b: ParsedAxiom) -> Optional[Dict]:
        a_ents = a.entities()
        b_ents = b.entities()
        a_props = a.properties()
        b_props = b.properties()

        shared_entities = a_ents & b_ents
        shared_properties = a_props & b_props

        # fact-fact link
        if a.axiom_type == "fact" and b.axiom_type == "fact" and shared_entities:
            return {
                "type": "shared_entity",
                "weight": 0.60,
                "reason": f"Facts share entities: {', '.join(sorted(shared_entities))}",
            }

        # rule-rule link
        if a.axiom_type == "rule" and b.axiom_type == "rule" and shared_properties:
            return {
                "type": "shared_property_family",
                "weight": 0.55,
                "reason": f"Rules overlap on properties: {', '.join(sorted(shared_properties))}",
            }

        # fact-rule link
        if a.axiom_type == "fact" and b.axiom_type == "rule":
            if shared_properties:
                return {
                    "type": "fact_rule_property_match",
                    "weight": 0.85,
                    "reason": f"Fact predicate matches rule property family: {', '.join(sorted(shared_properties))}",
                }

        if a.axiom_type == "rule" and b.axiom_type == "fact":
            if shared_properties:
                return {
                    "type": "fact_rule_property_match",
                    "weight": 0.85,
                    "reason": f"Fact predicate matches rule property family: {', '.join(sorted(shared_properties))}",
                }

        return None

    def _build_graph(self, parsed_nodes: List[Dict]) -> List[Dict]:
        edges = []
        for i, j in itertools.combinations(range(len(parsed_nodes)), 2):
            pa = parsed_nodes[i]["parsed_obj"]
            pb = parsed_nodes[j]["parsed_obj"]
            edge = self._edge_between(pa, pb)
            if edge and edge["weight"] >= self.edge_threshold:
                edges.append(
                    {
                        "source": parsed_nodes[i]["node_id"],
                        "target": parsed_nodes[j]["node_id"],
                        "type": edge["type"],
                        "weight": edge["weight"],
                        "reason": edge["reason"],
                    }
                )
        return edges

    # -----------------------------------------------------
    # Scoring helpers
    # -----------------------------------------------------
    def _node_score(self, node: Dict) -> float:
        return float(node["score"])

    def _subgraph_edge_count(self, node_ids: Set[str], edges: List[Dict]) -> int:
        c = 0
        for e in edges:
            if e["source"] in node_ids and e["target"] in node_ids:
                c += 1
        return c

    def _is_connected(self, node_ids: Set[str], edges: List[Dict]) -> bool:
        if len(node_ids) <= 1:
            return True

        adj = {nid: set() for nid in node_ids}
        for e in edges:
            s, t = e["source"], e["target"]
            if s in node_ids and t in node_ids:
                adj[s].add(t)
                adj[t].add(s)

        start = next(iter(node_ids))
        stack = [start]
        seen = set()

        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj[cur] - seen)

        return len(seen) == len(node_ids)

    def _fact_rule_mix(self, nodes: List[Dict]) -> bool:
        kinds = {n["parsed_obj"].axiom_type for n in nodes}
        return "fact" in kinds and "rule" in kinds

    def _query_alignment(
        self, nodes: List[Dict], query_entities: Set[str], query_properties: Set[str]
    ) -> float:
        hit = 0.0
        for n in nodes:
            p = n["parsed_obj"]
            if p.entities() & query_entities:
                hit += 1.0
            if p.properties() & query_properties:
                hit += 1.0
        return hit / max(len(nodes), 1)

    def _bridge_score(
        self, nodes: List[Dict], query_entities: Set[str], query_properties: Set[str]
    ) -> float:
        """
        Reward subgraphs that collectively touch multiple query signatures,
        especially if facts/rules combine to bridge them.
        """
        covered_entities = set()
        covered_properties = set()
        fact_hits = 0
        rule_hits = 0

        for n in nodes:
            p = n["parsed_obj"]
            ents = p.entities() & query_entities
            props = p.properties() & query_properties
            covered_entities |= ents
            covered_properties |= props

            if p.axiom_type == "fact" and (ents or props):
                fact_hits += 1
            if p.axiom_type == "rule" and (ents or props):
                rule_hits += 1

        score = 0.0
        if query_entities:
            score += len(covered_entities) / len(query_entities)
        if query_properties:
            score += len(covered_properties) / len(query_properties)

        if fact_hits > 0 and rule_hits > 0:
            score += 0.5

        return score / 2.5

    def _redundancy(self, nodes: List[Dict]) -> float:
        """
        Penalize overly repetitive rule-only selections on the same property family.
        """
        prop_bag = []
        for n in nodes:
            prop_bag.extend(list(n["parsed_obj"].properties()))
        if not prop_bag:
            return 0.0
        unique = len(set(prop_bag))
        total = len(prop_bag)
        if total == 0:
            return 0.0
        return max(0.0, 1.0 - (unique / total))

    def _score_nodes(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        query_entities: Set[str],
        query_properties: Set[str],
    ) -> Dict:
        node_ids = {n["node_id"] for n in nodes}

        base = sum(self._node_score(n) for n in nodes)
        connected = self._is_connected(node_ids, edges)
        mix = self._fact_rule_mix(nodes)
        q_align = self._query_alignment(nodes, query_entities, query_properties)
        bridge = self._bridge_score(nodes, query_entities, query_properties)
        redundancy = self._redundancy(nodes)

        score = base
        if connected:
            score += self.connectivity_bonus
        if mix:
            score += self.fact_rule_mix_bonus
        score += self.query_alignment_bonus * q_align
        score += self.bridge_bonus * bridge
        score -= self.redundancy_penalty * redundancy

        return {
            "node_ids": sorted(node_ids),
            "score": round(score, 6),
            "connected": connected,
            "fact_rule_mix": mix,
            "query_alignment": round(q_align, 6),
            "bridge_score": round(bridge, 6),
            "nodes": [
                {
                    "node_id": n["node_id"],
                    "raw_axiom": n["raw_axiom"],
                    "score": n["score"],
                    "candidate_kind": n["candidate_kind"],
                    "axiom_type": n["parsed_obj"].axiom_type,
                    "parsed": asdict(n["parsed_obj"]),
                }
                for n in nodes
            ],
        }

    def _estimated_combination_count(self, n: int, max_size: int) -> int:
        total = 0
        for size in range(1, max_size + 1):
            total += math.comb(n, size)
            if total > self.max_combinations:
                break
        return total

    def _beam_candidates(
        self,
        parsed_nodes: List[Dict],
        edges: List[Dict],
        max_size: int,
        query_entities: Set[str],
        query_properties: Set[str],
    ) -> List[Dict]:
        node_by_id = {n["node_id"]: n for n in parsed_nodes}
        adjacency = {n["node_id"]: set() for n in parsed_nodes}
        for edge in edges:
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])

        frontier = [
            (n["node_id"],)
            for n in sorted(parsed_nodes, key=lambda item: item["score"], reverse=True)
        ][: max(1, self.beam_width)]
        seen = set(frontier)
        candidates = []

        for size in range(1, max_size + 1):
            for combo_ids in frontier:
                candidates.append(
                    self._score_nodes(
                        [node_by_id[nid] for nid in combo_ids],
                        edges,
                        query_entities,
                        query_properties,
                    )
                )

            if size == max_size:
                break

            expansions = {}
            for combo_ids in frontier:
                combo_set = set(combo_ids)
                neighbors = set()
                for nid in combo_ids:
                    neighbors.update(adjacency[nid])

                for nxt in neighbors - combo_set:
                    expanded = tuple(sorted((*combo_ids, nxt)))
                    if expanded in seen:
                        continue
                    seen.add(expanded)
                    candidate = self._score_nodes(
                        [node_by_id[nid] for nid in expanded],
                        edges,
                        query_entities,
                        query_properties,
                    )
                    expansions[expanded] = candidate

            if not expansions:
                break

            frontier = [
                tuple(candidate["node_ids"])
                for candidate in sorted(
                    expansions.values(),
                    key=lambda item: item["score"],
                    reverse=True,
                )[: max(1, self.beam_width)]
            ]

        return candidates

    # -----------------------------------------------------
    # Main compose
    # -----------------------------------------------------
    def compose(
        self, question: str, retrieved_units: List[Dict], sparql_query: str = ""
    ) -> Dict:
        parsed_nodes = []
        for idx, item in enumerate(retrieved_units):
            parsed = parse_axiom(item["axiom"])
            parsed_nodes.append(
                {
                    "node_id": f"n{idx}",
                    "raw_axiom": item["axiom"],
                    "candidate_kind": item.get("candidate_kind", "unknown"),
                    "score": float(item["score"]),
                    "label": int(item.get("label", 0)),
                    "parsed_obj": parsed,
                }
            )

        edges = self._build_graph(parsed_nodes)

        qsig = extract_query_signature(question=question, sparql_query=sparql_query)
        query_entities = set(qsig["query_entities"])
        query_properties = set(qsig["query_properties"])

        best = None

        max_size = (
            len(parsed_nodes)
            if self.top_k_subgraph <= 0
            else min(self.top_k_subgraph, len(parsed_nodes))
        )
        if (
            self._estimated_combination_count(len(parsed_nodes), max_size)
            <= self.max_combinations
        ):
            candidates = [
                self._score_nodes(
                    list(combo),
                    edges,
                    query_entities,
                    query_properties,
                )
                for size in range(1, max_size + 1)
                for combo in itertools.combinations(parsed_nodes, size)
            ]
        else:
            candidates = self._beam_candidates(
                parsed_nodes=parsed_nodes,
                edges=edges,
                max_size=max_size,
                query_entities=query_entities,
                query_properties=query_properties,
            )

        for candidate in candidates:
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        if best is None:
            best = {
                "node_ids": [],
                "score": 0.0,
                "connected": False,
                "fact_rule_mix": False,
                "query_alignment": 0.0,
                "bridge_score": 0.0,
                "nodes": [],
            }

        return {
            "question": question,
            "query_signature": qsig,
            "best_subgraph": best,
            "edges": edges,
        }
