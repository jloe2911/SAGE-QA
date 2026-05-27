# GNN-RAG Local Changes

This directory records local changes made under `third_party/GNN-RAG` without
committing to the upstream GNN-RAG repository.

- Base GNN-RAG commit: `28d31bc0db3d54f0930e6125f1c98b5b1ace1ba4`
- Source patch: `GNN-RAG-local-changes.patch`

Apply from `third_party/GNN-RAG` with:

```powershell
git apply ../../patches/GNN-RAG-local-changes.patch
```

Untracked result files that existed when this patch was exported:

```text
llm/results/NeSyQA-GNN-RAG/nesyqa-2wiki/gpt-4.1-mini/test/no_rule/False/args.txt
llm/results/NeSyQA-GNN-RAG/nesyqa-2wiki/gpt-4.1-mini/test/no_rule/False/detailed_eval_result.jsonl
llm/results/NeSyQA-GNN-RAG/nesyqa-2wiki/gpt-4.1-mini/test/no_rule/False/eval_result.txt
llm/results/NeSyQA-GNN-RAG/nesyqa-2wiki/gpt-4.1-mini/test/no_rule/False/predictions.jsonl
llm/results/NeSyQA-GNN-RAG/nesyqa-familyowl_1hop/gpt-4.1-mini/test/no_rule/False/args.txt
llm/results/NeSyQA-GNN-RAG/nesyqa-familyowl_1hop/gpt-4.1-mini/test/no_rule/False/detailed_eval_result.jsonl
llm/results/NeSyQA-GNN-RAG/nesyqa-familyowl_1hop/gpt-4.1-mini/test/no_rule/False/eval_result.txt
llm/results/NeSyQA-GNN-RAG/nesyqa-familyowl_1hop/gpt-4.1-mini/test/no_rule/False/predictions.jsonl
llm/results/NeSyQA-GNN-RAG/nesyqa-familyowl_2hop/gpt-4.1-mini/test/no_rule/False/args.txt
llm/results/NeSyQA-GNN-RAG/nesyqa-familyowl_2hop/gpt-4.1-mini/test/no_rule/False/detailed_eval_result.jsonl
llm/results/NeSyQA-GNN-RAG/nesyqa-familyowl_2hop/gpt-4.1-mini/test/no_rule/False/eval_result.txt
llm/results/NeSyQA-GNN-RAG/nesyqa-familyowl_2hop/gpt-4.1-mini/test/no_rule/False/predictions.jsonl
llm/results/NeSyQA-GNN-RAG/nesyqa-hotpotqa/gpt-4.1-mini/test/no_rule/False/args.txt
llm/results/NeSyQA-GNN-RAG/nesyqa-hotpotqa/gpt-4.1-mini/test/no_rule/False/detailed_eval_result.jsonl
llm/results/NeSyQA-GNN-RAG/nesyqa-hotpotqa/gpt-4.1-mini/test/no_rule/False/eval_result.txt
llm/results/NeSyQA-GNN-RAG/nesyqa-hotpotqa/gpt-4.1-mini/test/no_rule/False/predictions.jsonl
llm/results/gnn/nesyqa-2wiki/rearev-lstm/test.info
llm/results/gnn/nesyqa-familyowl_1hop/rearev-lstm/test.info
llm/results/gnn/nesyqa-familyowl_1hop_smoke/rearev-lstm/test.info
llm/results/gnn/nesyqa-familyowl_2hop/rearev-lstm/test.info
llm/results/gnn/nesyqa-hotpotqa/rearev-lstm/test.info
```
