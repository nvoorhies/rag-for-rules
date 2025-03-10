python data/QA/evaluators/evaluate_qa.py \
    --qa-pairs tmp/single_qa_3.json \
    --srd tmp/srd.morefull.json \
    --output tmp/title_path_references_scope_eval.json \
    --system augmented \
    --top-k 10 \
    --verbose