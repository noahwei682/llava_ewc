#!/bin/bash
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR \
    --question-file /blob/weiwei/playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file /blob/weiwei/playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-7b-$CKPT_DIR.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --text_embed_name mmbench_dev_20230712 \


mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /blob/weiwei/playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /blob/weiwei/playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir /blob/weiwei/playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b-$CKPT_DIR
