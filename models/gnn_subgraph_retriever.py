import re
import string
from typing import List, Dict, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.symbolic_composer import parse_axiom, extract_query_signature


# =========================================================
# Text encoder helpers
# =========================================================


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def normalize_prop(prop: str) -> str:
    prop = str(prop).strip()
    if prop == "rdf:type":
        return "type"
    return prop


def serialize_axiom_for_encoder(axiom: str) -> str:
    """
    A stable text form for encoding axiom nodes.
    """
    axiom = " ".join(str(axiom).strip().split())

    kg = parse_kg_triple_unit(axiom)
    if kg is not None:
        s, p, o = kg
        return f"fact subject {s} predicate {p} object {o}"

    sent = parse_sentence_unit(axiom)
    if sent is not None:
        title, _, text = sent
        return f"sentence title {title} text {text}"

    if axiom.startswith("SymmetricObjectProperty("):
        p = axiom[len("SymmetricObjectProperty(") : -1]
        return f"rule symmetric object property {p}"

    if axiom.startswith("TransitiveObjectProperty("):
        p = axiom[len("TransitiveObjectProperty(") : -1]
        return f"rule transitive object property {p}"

    if axiom.startswith("FunctionalObjectProperty("):
        p = axiom[len("FunctionalObjectProperty(") : -1]
        return f"rule functional object property {p}"

    if axiom.startswith("SubObjectPropertyOf(") and axiom.endswith(")"):
        inside = axiom[len("SubObjectPropertyOf(") : -1]
        parts = [x.strip() for x in inside.split(",", 1)]
        if len(parts) == 2:
            return f"rule sub object property {parts[0]} subproperty of {parts[1]}"

    if axiom.startswith("InverseObjectProperties(") and axiom.endswith(")"):
        inside = axiom[len("InverseObjectProperties(") : -1]
        parts = [x.strip() for x in inside.split(",", 1)]
        if len(parts) == 2:
            return f"rule inverse object properties {parts[0]} inverse of {parts[1]}"

    if axiom.startswith("EquivalentObjectProperties(") and axiom.endswith(")"):
        inside = axiom[len("EquivalentObjectProperties(") : -1]
        parts = [x.strip() for x in inside.split(",", 1)]
        if len(parts) == 2:
            return f"rule equivalent object properties {parts[0]} equivalent to {parts[1]}"

    if " subClassOf " in axiom:
        a, b = axiom.split(" subClassOf ", 1)
        return f"rule subclass {a} subclass of {b}"

    if " domain " in axiom:
        p, c = axiom.split(" domain ", 1)
        return f"rule domain property {p} domain {c}"

    if " range " in axiom:
        p, c = axiom.split(" range ", 1)
        return f"rule range property {p} range {c}"

    if " disjointWith " in axiom:
        a, b = axiom.split(" disjointWith ", 1)
        return f"rule disjoint classes {a} disjoint with {b}"

    parts = axiom.split()
    if len(parts) == 3:
        s, p, o = parts
        return f"fact subject {s} predicate {p} object {o}"

    return axiom


def parse_kg_triple_unit(unit: str) -> Optional[Tuple[str, str, str]]:
    parts = str(unit).split("::", 3)
    if len(parts) == 4 and parts[0] == "KG":
        return parts[1].strip(), parts[2].strip(), parts[3].strip()
    return None


def parse_sentence_unit(unit: str) -> Optional[Tuple[str, int, str]]:
    parts = str(unit).split("::", 3)
    if len(parts) != 4 or parts[0] != "SENT":
        return None
    try:
        idx = int(parts[2])
    except Exception:
        idx = -1
    return parts[1].strip(), idx, parts[3].strip()


_ARTICLES = {"a", "an", "the"}


def normalize_text_for_match(text: str) -> str:
    text = str(text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def token_set_for_match(text: str) -> Set[str]:
    return {
        tok
        for tok in normalize_text_for_match(text).split()
        if tok and tok not in _ARTICLES and len(tok) > 1
    }


def kg_unit_signature(unit: str) -> Tuple[Set[str], Set[str]]:
    kg = parse_kg_triple_unit(unit)
    if kg is None:
        return set(), set()
    s, p, o = kg
    return {s, o}, {normalize_prop(p)}


def text_mentions_entity(text: str, entity: str) -> bool:
    entity_norm = normalize_text_for_match(entity)
    text_norm = normalize_text_for_match(text)
    if not entity_norm or not text_norm:
        return False
    if entity_norm in text_norm:
        return True
    entity_tokens = token_set_for_match(entity)
    text_tokens = token_set_for_match(text)
    if not entity_tokens:
        return False
    # Require substantial overlap to avoid connecting every Hesse-like page.
    return len(entity_tokens & text_tokens) / len(entity_tokens) >= 0.6


def sentence_kg_edge(sentence_unit: str, kg_unit: str) -> bool:
    sent = parse_sentence_unit(sentence_unit)
    kg = parse_kg_triple_unit(kg_unit)
    if sent is None or kg is None:
        return False
    title, _, text = sent
    s, _, o = kg
    haystack = f"{title} {text}"
    return (
        text_mentions_entity(haystack, s)
        or text_mentions_entity(haystack, o)
        or text_mentions_entity(f"{s} {o}", title)
    )


def extract_rule_properties(unit: str) -> set:
    unit = str(unit).strip()
    props = set()

    patterns = [
        r"SymmetricObjectProperty\((.*?)\)",
        r"TransitiveObjectProperty\((.*?)\)",
        r"FunctionalObjectProperty\((.*?)\)",
    ]

    for pat in patterns:
        m = re.match(pat, unit)
        if m:
            props.add(normalize_prop(m.group(1)))

    m = re.match(r"SubObjectPropertyOf\((.*?),(.*?)\)", unit)
    if m:
        props.add(normalize_prop(m.group(1)))
        props.add(normalize_prop(m.group(2)))

    m = re.match(r"InverseObjectProperties\((.*?),(.*?)\)", unit)
    if m:
        props.add(normalize_prop(m.group(1)))
        props.add(normalize_prop(m.group(2)))

    m = re.match(r"EquivalentObjectProperties\((.*?),(.*?)\)", unit)
    if m:
        props.add(normalize_prop(m.group(1)))
        props.add(normalize_prop(m.group(2)))

    if " domain " in unit:
        props.add(normalize_prop(unit.split(" domain ", 1)[0]))

    if " range " in unit:
        props.add(normalize_prop(unit.split(" range ", 1)[0]))

    return props


# =========================================================
# Graph construction helpers
# =========================================================


def edge_between_axioms(ax1: str, ax2: str) -> bool:
    """
    Same symbolic graph intuition as the previous subgraph generator:
    connect axioms if they share entities or properties.
    """
    kg1 = parse_kg_triple_unit(ax1)
    kg2 = parse_kg_triple_unit(ax2)
    sent1 = parse_sentence_unit(ax1)
    sent2 = parse_sentence_unit(ax2)

    if kg1 is not None and sent2 is not None:
        return sentence_kg_edge(ax2, ax1)

    if sent1 is not None and kg2 is not None:
        return sentence_kg_edge(ax1, ax2)

    if kg1 is not None and kg2 is not None:
        ents1, props1 = kg_unit_signature(ax1)
        ents2, props2 = kg_unit_signature(ax2)
        return bool((ents1 & ents2) or (props1 & props2))

    # Keep sentence-sentence edges conservative. KG bridge nodes carry the
    # structural signal for text benchmarks; capitalized phrase overlap tends
    # to create dense, noisy graphs.
    if sent1 is not None and sent2 is not None:
        return False

    p1 = parse_axiom(ax1)
    p2 = parse_axiom(ax2)

    ents1 = p1.entities()
    ents2 = p2.entities()
    props1 = {normalize_prop(p) for p in p1.properties()}
    props2 = {normalize_prop(p) for p in p2.properties()}

    rule_props1 = extract_rule_properties(ax1)
    rule_props2 = extract_rule_properties(ax2)

    props1 = props1 | rule_props1
    props2 = props2 | rule_props2

    if ents1 & ents2:
        return True

    if props1 & props2:
        return True

    return False


def build_edge_index(candidate_axioms: List[str], device=None) -> torch.Tensor:
    """
    Builds a bidirectional edge_index tensor of shape [2, num_edges].

    Includes self-loops.
    """
    edges = []
    n = len(candidate_axioms)

    for i in range(n):
        edges.append((i, i))

    for i in range(n):
        for j in range(i + 1, n):
            if edge_between_axioms(candidate_axioms[i], candidate_axioms[j]):
                edges.append((i, j))
                edges.append((j, i))

    if not edges:
        edges = [(i, i) for i in range(n)]

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index


def compute_node_symbolic_features(
    candidate_axioms: List[str],
    question: str,
    sparql_query: str,
) -> torch.Tensor:
    """
    Per-node symbolic features.

    Features:
      0. is_fact
      1. is_rule
      2. overlaps_query_entity
      3. overlaps_query_property
      4. rule_mentions_query_property
      5. fact_mentions_query_entity
      6. degree proxy placeholder, filled later
      7. bias feature = 1
    """
    qsig = extract_query_signature(question=question, sparql_query=sparql_query)
    query_entities = set(qsig.get("query_entities", []))
    query_properties = {normalize_prop(p) for p in qsig.get("query_properties", [])}

    rows = []

    for ax in candidate_axioms:
        kg = parse_kg_triple_unit(ax)
        if kg is not None:
            subject, predicate, obj = kg
            entity_tokens = token_set_for_match(subject) | token_set_for_match(obj)
            property_tokens = token_set_for_match(predicate)
            question_tokens = token_set_for_match(question)

            overlaps_query_entity = (
                1.0 if entity_tokens and entity_tokens & question_tokens else 0.0
            )
            overlaps_query_property = (
                1.0 if property_tokens and property_tokens & question_tokens else 0.0
            )
            rows.append(
                [
                    1.0,
                    0.0,
                    overlaps_query_entity,
                    overlaps_query_property,
                    0.0,
                    overlaps_query_entity,
                    0.0,
                    1.0,
                ]
            )
            continue

        sent = parse_sentence_unit(ax)
        if sent is not None:
            title, _, text = sent
            question_tokens = token_set_for_match(question)
            sent_tokens = token_set_for_match(f"{title} {text}")
            title_tokens = token_set_for_match(title)
            overlap = 1.0 if sent_tokens & question_tokens else 0.0
            title_overlap = 1.0 if title_tokens & question_tokens else 0.0
            rows.append(
                [
                    0.0,
                    0.0,
                    title_overlap,
                    overlap,
                    0.0,
                    title_overlap,
                    0.0,
                    1.0,
                ]
            )
            continue

        parsed = parse_axiom(ax)

        unit_entities = set(parsed.entities())
        unit_properties = {normalize_prop(p) for p in parsed.properties()}
        rule_properties = extract_rule_properties(ax)

        is_fact = 1.0 if parsed.axiom_type == "fact" else 0.0
        is_rule = 1.0 if parsed.axiom_type == "rule" else 0.0

        overlaps_query_entity = 1.0 if unit_entities & query_entities else 0.0
        overlaps_query_property = (
            1.0 if (unit_properties | rule_properties) & query_properties else 0.0
        )

        rule_mentions_query_property = (
            1.0 if (parsed.axiom_type == "rule" and rule_properties & query_properties) else 0.0
        )

        fact_mentions_query_entity = (
            1.0 if (parsed.axiom_type == "fact" and unit_entities & query_entities) else 0.0
        )

        rows.append(
            [
                is_fact,
                is_rule,
                overlaps_query_entity,
                overlaps_query_property,
                rule_mentions_query_property,
                fact_mentions_query_entity,
                0.0,
                1.0,
            ]
        )

    return torch.tensor(rows, dtype=torch.float)


def fill_degree_feature(node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Fill feature index 6 with normalized node degree.
    """
    if node_features.numel() == 0:
        return node_features

    n = node_features.size(0)
    degrees = torch.zeros(n, dtype=node_features.dtype, device=node_features.device)

    src = edge_index[0]
    degrees.index_add_(0, src, torch.ones_like(src, dtype=node_features.dtype))

    if degrees.max() > 0:
        degrees = degrees / degrees.max().clamp(min=1.0)

    node_features = node_features.clone()
    node_features[:, 6] = degrees
    return node_features


# =========================================================
# Simple GraphSAGE layer without PyG dependency
# =========================================================


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_dim]
        edge_index: [2, num_edges]
        """
        num_nodes = x.size(0)
        src, dst = edge_index

        neigh_sum = torch.zeros_like(x)
        neigh_sum.index_add_(0, dst, x[src])

        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)

        neigh_mean = neigh_sum / deg

        out = self.self_proj(x) + self.neigh_proj(neigh_mean)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


# =========================================================
# GNN Subgraph Retriever
# =========================================================


class GNNSubgraphRetriever(nn.Module):
    """
    GNN-based support-subgraph scorer.

    The model scores candidate support subgraphs by:
      1. Encoding the question.
      2. Encoding all candidate axiom nodes.
      3. Running message passing over the full local evidence graph.
      4. Pooling embeddings of the candidate subgraph nodes.
      5. Concatenating query embedding + subgraph embedding + symbolic subgraph features.
      6. Predicting a support score.

    This is still subgraph prediction, not node classification.
    """

    def __init__(
        self,
        model_name: str = "prajjwal1/bert-mini",
        node_symbolic_dim: int = 8,
        subgraph_symbolic_dim: int = 8,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        from utils.model_loader import load_encoder

        self.encoder = load_encoder(model_name)
        encoder_hidden = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.node_symbolic_dim = node_symbolic_dim
        self.subgraph_symbolic_dim = subgraph_symbolic_dim
        self.gnn_hidden_dim = gnn_hidden_dim

        self.node_input_projection = nn.Sequential(
            nn.Linear(encoder_hidden + node_symbolic_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gnn_layers = nn.ModuleList(
            [
                GraphSAGELayer(
                    in_dim=gnn_hidden_dim,
                    out_dim=gnn_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(gnn_layers)
            ]
        )

        self.subgraph_feature_projection = nn.Sequential(
            nn.Linear(subgraph_symbolic_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                encoder_hidden + gnn_hidden_dim + classifier_hidden_dim,
                classifier_hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def encode_texts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None,
        )
        return mean_pool(outputs.last_hidden_state, attention_mask)

    def encode_graph(
        self,
        node_text_embeddings: torch.Tensor,
        node_symbolic_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns contextualized node embeddings after GNN message passing.
        """
        x = torch.cat([node_text_embeddings, node_symbolic_features], dim=-1)
        x = self.node_input_projection(x)

        for layer in self.gnn_layers:
            residual = x
            x = layer(x, edge_index)
            x = x + residual

        return x

    def pool_subgraph(
        self,
        node_embeddings: torch.Tensor,
        subgraph_node_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool the node embeddings belonging to the candidate subgraph.
        """
        selected = node_embeddings[subgraph_node_ids]
        return selected.mean(dim=0)

    def score_one_graph(
        self,
        query_embedding: torch.Tensor,
        node_text_embeddings: torch.Tensor,
        node_symbolic_features: torch.Tensor,
        edge_index: torch.Tensor,
        subgraph_node_ids: torch.Tensor,
        subgraph_symbolic_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Scores one candidate subgraph inside one local evidence graph.
        """
        node_embeddings = self.encode_graph(
            node_text_embeddings=node_text_embeddings,
            node_symbolic_features=node_symbolic_features,
            edge_index=edge_index,
        )

        pooled_subgraph = self.pool_subgraph(
            node_embeddings=node_embeddings,
            subgraph_node_ids=subgraph_node_ids,
        )

        symbolic_repr = self.subgraph_feature_projection(
            subgraph_symbolic_features.unsqueeze(0)
        ).squeeze(0)

        combined = torch.cat(
            [
                query_embedding,
                pooled_subgraph,
                symbolic_repr,
            ],
            dim=-1,
        )

        logit = self.classifier(combined).squeeze(-1)
        prob = torch.sigmoid(logit)

        return {
            "logit": logit,
            "prob": prob,
            "pooled_subgraph": pooled_subgraph,
        }

    def forward_batch_graphs(
        self,
        query_embeddings: List[torch.Tensor],
        node_text_embeddings: List[torch.Tensor],
        node_symbolic_features: List[torch.Tensor],
        edge_indices: List[torch.Tensor],
        subgraph_node_ids: List[torch.Tensor],
        subgraph_symbolic_features: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Scores a batch of candidate subgraphs.

        Each item may belong to a different local graph, so this function
        processes them as a Python list instead of forcing a dense batch.
        This is slower but simpler and robust for the first implementation.
        """
        logits = []
        probs = []

        for q_emb, node_emb, node_sym, edge_idx, sg_ids, sg_sym in zip(
            query_embeddings,
            node_text_embeddings,
            node_symbolic_features,
            edge_indices,
            subgraph_node_ids,
            subgraph_symbolic_features,
        ):
            out = self.score_one_graph(
                query_embedding=q_emb,
                node_text_embeddings=node_emb,
                node_symbolic_features=node_sym,
                edge_index=edge_idx,
                subgraph_node_ids=sg_ids,
                subgraph_symbolic_features=sg_sym,
            )
            logits.append(out["logit"])
            probs.append(out["prob"])

        logits = torch.stack(logits, dim=0)
        probs = torch.stack(probs, dim=0)

        return {
            "logits": logits,
            "probs": probs,
        }


# =========================================================
# Feature computation for training/evaluation scripts
# =========================================================


def compute_subgraph_symbolic_features(
    question: str,
    sparql_query: str,
    subgraph_units: List[str],
    use_gold_features: bool = False,
    exact_match_any_gold: bool = False,
    contains_any_gold_explanation: bool = False,
) -> List[float]:
    """
    Same 8-dimensional feature vector used by the previous subgraph retriever.

    In final/inference-safe mode, keep use_gold_features=False.
    """
    qsig = extract_query_signature(question=question, sparql_query=sparql_query)
    query_entities = set(qsig.get("query_entities", []))
    query_properties = {normalize_prop(p) for p in qsig.get("query_properties", [])}

    if not subgraph_units:
        return [0.0] * 8

    entity_overlap_count = 0
    property_overlap_count = 0
    any_rule_mentions_query_property = 0.0
    any_fact_mentions_query_entity = 0.0

    has_fact = False
    has_rule = False

    for unit in subgraph_units:
        parsed = parse_axiom(unit)

        unit_entities = set(parsed.entities())
        unit_properties = {normalize_prop(p) for p in parsed.properties()}
        rule_properties = extract_rule_properties(unit)

        if unit_entities & query_entities:
            entity_overlap_count += 1

        if (unit_properties | rule_properties) & query_properties:
            property_overlap_count += 1

        if parsed.axiom_type == "rule":
            has_rule = True
            if rule_properties & query_properties:
                any_rule_mentions_query_property = 1.0

        if parsed.axiom_type == "fact":
            has_fact = True
            if unit_entities & query_entities:
                any_fact_mentions_query_entity = 1.0

    frac_entity_overlap = entity_overlap_count / len(subgraph_units)
    frac_property_overlap = property_overlap_count / len(subgraph_units)
    fact_rule_mix = 1.0 if has_fact and has_rule else 0.0
    size_norm = min(len(subgraph_units), 5) / 5.0

    exact = 1.0 if (use_gold_features and exact_match_any_gold) else 0.0
    contains = 1.0 if (use_gold_features and contains_any_gold_explanation) else 0.0

    return [
        frac_entity_overlap,
        frac_property_overlap,
        any_rule_mentions_query_property,
        any_fact_mentions_query_entity,
        fact_rule_mix,
        size_norm,
        exact,
        contains,
    ]


def build_graph_inputs_for_example(
    candidate_axioms: List[str],
    question: str,
    sparql_query: str,
    tokenizer,
    model_device,
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Builds graph tensors for one local evidence graph.

    This function tokenizes node texts but does not run the encoder.
    The training script can pass the tokenized tensors through the model encoder.
    """
    node_texts = [serialize_axiom_for_encoder(ax) for ax in candidate_axioms]

    encoded_nodes = tokenizer(
        node_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encoded_query = tokenizer(
        [question],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    edge_index = build_edge_index(candidate_axioms, device=model_device)

    node_symbolic = compute_node_symbolic_features(
        candidate_axioms=candidate_axioms,
        question=question,
        sparql_query=sparql_query,
    ).to(model_device)

    node_symbolic = fill_degree_feature(node_symbolic, edge_index)

    return {
        "node_input_ids": encoded_nodes["input_ids"].to(model_device),
        "node_attention_mask": encoded_nodes["attention_mask"].to(model_device),
        "node_token_type_ids": encoded_nodes.get("token_type_ids", None).to(model_device)
        if "token_type_ids" in encoded_nodes
        else None,
        "query_input_ids": encoded_query["input_ids"].to(model_device),
        "query_attention_mask": encoded_query["attention_mask"].to(model_device),
        "query_token_type_ids": encoded_query.get("token_type_ids", None).to(model_device)
        if "token_type_ids" in encoded_query
        else None,
        "edge_index": edge_index,
        "node_symbolic_features": node_symbolic,
    }
