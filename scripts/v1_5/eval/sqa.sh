#!/bin/bash
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi
python -m llava.eval.model_vqa_science \
    --model-path /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR \
    --question-file /blob/weiwei/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /blob/weiwei/playground/data/eval/scienceqa/images/test \
    --answers-file /blob/weiwei/playground/data/eval/scienceqa/answers/llava-v1.5-7b-$CKPT_DIR.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /blob/weiwei/playground/data/eval/scienceqa \
    --result-file /blob/weiwei/playground/data/eval/scienceqa/answers/llava-v1.5-7b-$CKPT_DIR.jsonl \
    --output-file /blob/weiwei/playground/data/eval/scienceqa/answers/llava-v1.5-7b-${CKPT_DIR}_output.jsonl \
    --output-result /blob/weiwei/playground/data/eval/scienceqa/answers/llava-v1.5-7b-${CKPT_DIR}_result.json
