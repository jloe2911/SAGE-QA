from evaluation.compare_ontology_atomic_pool_builders import (
    graph_expansion_atomic_pool,
)


def test_graph_expansion_reaches_schema_through_shared_relation():
    axioms = [
        "alice hasParent bob",
        "SubObjectPropertyOf(hasParent,hasAncestor)",
        "TransitiveObjectProperty(hasAncestor)",
        "unrelated likes pizza",
    ]

    pool = graph_expansion_atomic_pool(
        axioms,
        question="Who is an ancestor of Alice?",
        sparql_query="SELECT ?x WHERE { <urn:alice> <urn:hasParent> ?x . }",
        max_context_units=4,
        hops=2,
    )

    assert "alice hasParent bob" in pool
    assert "SubObjectPropertyOf(hasParent,hasAncestor)" in pool
    assert "TransitiveObjectProperty(hasAncestor)" in pool
    assert "unrelated likes pizza" not in pool


def test_graph_expansion_uses_natural_language_seed_without_formal_uri_seed():
    pool = graph_expansion_atomic_pool(
        ["margherita hasTopping mozzarella", "alice knows bob"],
        question="Which pizza has mozzarella topping?",
        sparql_query="SELECT ?x WHERE { ?x ?p ?y . }",
        max_context_units=1,
        hops=2,
    )

    assert pool == ["margherita hasTopping mozzarella"]


def test_graph_expansion_has_no_arbitrary_fallback_when_ungrounded():
    pool = graph_expansion_atomic_pool(
        ["alice knows bob"],
        question="Completely unmatched wording",
        sparql_query="SELECT ?x WHERE { ?x ?p ?y . }",
        max_context_units=40,
        hops=2,
    )

    assert pool == []
