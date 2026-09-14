# HotpotQA and 2Wiki Retrieval Data

> **Scope:** This is a specialized technical note about text-evidence representation and
> retrieval scoring. It is not the overall SAGE-QA architecture or reproduction guide.
> Start with [`README.md`](README.md), then use
> [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for protocol generations and
> [`REPRODUCE.md`](REPRODUCE.md) for commands and dependency boundaries.

This note explains what one text QA example becomes inside SAGE-QA. The same
schema is used for HotpotQA and 2WikiMultiHopQA: raw benchmark examples are
converted into candidate support-subgraph rows, then a retriever ranks those
rows for each question.

## Core Mapping

```text
raw context                         -> SENT::<title>::<sent_id>::<sentence>
raw/provided or constructed triples -> KG::<subject>::<predicate>::<object>
raw supporting_facts                -> gold_support_units
```

`SENT::...` units are sentence evidence. They are the only units counted as
predicted support facts.

`KG::...` units are graph-context bridge nodes. They can help the GNN rank a
candidate, but they are not exported as support facts and are not compared
against gold support labels.

## Running 2Wiki Example

Raw 2Wiki question:

```text
What is the place of birth of Princess Caroline Of Hesse-Darmstadt's husband?
```

Gold answer:

```text
Bad Homburg vor der Hohe
```

2Wiki provides structured evidence triples, for example:

```text
Caroline of Hesse-Darmstadt --spouse--> Frederick V, Landgrave of Hesse-Homburg
Frederick V, Landgrave of Hesse-Homburg --place of birth--> Bad Homburg vor der Hohe
```

It also provides gold supporting facts as sentence labels:

```text
Princess Caroline of Hesse-Darmstadt, sentence 0
Frederick V, Landgrave of Hesse-Homburg, sentence 0
```

The builder resolves those labels through `context`, producing:

```text
SENT::Princess Caroline of Hesse-Darmstadt::0::Caroline of Hesse-Darmstadt ...
SENT::Frederick V, Landgrave of Hesse-Homburg::0::Frederick V Louis William Christian ...
```

A generated candidate row then has the same `question`, `answer`, and
`example_id`, but a specific candidate support set:

```json
{
  "example_id": "2WikiMultiHopQA__test__89de3bee0bd911eba7f7acde48001122",
  "subgraph_units": [
    "SENT::Frederick V, Landgrave of Hesse-Homburg::0::...",
    "SENT::Princess Caroline of Hesse-Darmstadt::0::..."
  ],
  "graph_context_units": [
    "KG::Caroline of Hesse-Darmstadt::spouse::Frederick V, Landgrave of Hesse-Homburg",
    "KG::Frederick V, Landgrave of Hesse-Homburg::place of birth::Bad Homburg vor der Hohe"
  ],
  "gold_support_units": [
    "SENT::Princess Caroline of Hesse-Darmstadt::0::...",
    "SENT::Frederick V, Landgrave of Hesse-Homburg::0::..."
  ],
  "label": 1,
  "rank_target": 1.0
}
```

Here the candidate is perfect: ignoring order, `subgraph_units` equals
`gold_support_units`. Therefore `label = 1`, `rank_target = 1.0`,
`best_set_f1_to_gold = 1.0`, and `exact_match_any_gold = true`.

Other rows with the same `example_id` are competing candidates for the same
question, such as one gold sentence only, the gold set plus a distractor, or
distractor-only sentence sets.

## Running HotpotQA Example

Raw HotpotQA question:

```text
What National Hockey League (NHL) season saw the Dallas Stars finish the season
in a lower position than the Nashville Predators?
```

Gold answer:

```text
2016-17 NHL season
```

HotpotQA gives page-level context and gold support labels, but it normally does
not provide gold evidence triples. The builder therefore flattens the Wikipedia
context into sentence units and constructs optional KG bridge triples from the
text or LLM KG construction.

For this example, gold support includes:

```text
2016-17 Dallas Stars season, sentence 0
2016-17 Dallas Stars season, sentence 1
2016-17 NHL season, sentence 0
2016-17 NHL season, sentence 2
```

Those labels become:

```text
SENT::2016-17 Dallas Stars season::0::The 2016-17 Dallas Stars season ...
SENT::2016-17 Dallas Stars season::1::The Stars missed the playoffs ...
SENT::2016-17 NHL season::0::The 2016-17 NHL season was ...
SENT::2016-17 NHL season::2::The 2017 Stanley Cup playoffs began ...
```

One generated row may contain only one of those four support sentences:

```json
{
  "example_id": "HotpotQA__test__5a84a5dc5542992a431d1a82",
  "subgraph_units": [
    "SENT::2016-17 Dallas Stars season::0::..."
  ],
  "graph_context_units": [
    "KG::2016-17 NHL season::regular_season_start::October 12, 2016",
    "KG::Pittsburgh Penguins::defeated_team::Nashville Predators"
  ],
  "gold_support_units": [
    "SENT::2016-17 Dallas Stars season::0::...",
    "SENT::2016-17 Dallas Stars season::1::...",
    "SENT::2016-17 NHL season::0::...",
    "SENT::2016-17 NHL season::2::..."
  ],
  "label": 0,
  "rank_target": 0.4
}
```

This row has partial overlap with the gold set, so it can receive a nonzero
ranking target, but `label = 0` because it does not contain the complete gold
support explanation.

## What The Retriever Learns

The generated JSONL is a ranking dataset, not a new QA dataset. For one
question, many rows share the same `example_id`:

```text
same question + same answer + different candidate subgraph_units
```

The retriever scores each candidate row. Training uses:

- `label`: positive when the candidate contains the complete gold support set.
- `rank_target`: set-F1 overlap between the candidate and the gold support.
- `symbolic_features`: gold-free lexical and structural features.
- `graph_context_units`: KG bridge nodes available to the GNN graph.

## Retrieval Evaluation

At evaluation time:

```text
1. group rows by example_id
2. rank candidate rows by retriever score
3. take the top k rows
4. union their subgraph_units
5. compare that union to gold_support_units
```

The final reported retrieval metrics are support precision, recall and F1 at `k` over that top-k union. `KG::...` units are never counted as predicted support. They only affect the ranking model's internal graph representation.

Final HotpotQA and 2Wiki support prediction export converts selected
`SENT::<title>::<sent_id>::<sentence>` units back to benchmark format:

```text
[title, sent_id]
```

The clean interpretation is:

```text
subgraph_units      = candidate/predicted sentence support
gold_support_units  = gold sentence support
graph_context_units = auxiliary KG context for ranking only
```
