python data/QA/evaluators/evaluate_qa.py \
    --qa-pairs tmp/single_qa_3.json \
    --srd tmp/srd.morefull.json \
    --output tmp/finetune_title_path_references_scope_eval.json \
    --system augmented \
    --top-k 10 \
    --model path-references-scope-finetuned-mpnet-base-v2 \
    --verbose

