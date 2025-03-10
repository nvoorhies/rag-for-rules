#!/bin/bash

# Evaluate the title only model
# Usage: bash src/scripts/eval_title_only.sh

python data/QA/evaluators/evaluate_qa.py \
    --qa-pairs tmp/single_qa_3.json \
    --srd tmp/srd.morefull.json \
    --output tmp/finetune_title_only_eval.json \
    --system hierarchical \
    --top-k 10 \
    --model title-finetuned-mpnet-base-v2

