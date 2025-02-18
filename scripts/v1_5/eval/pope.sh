#!/bin/bash
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi
python -m llava.eval.model_vqa_loader \
    --model-path /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR \
    --question-file /blob/weiwei/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /blob/weiwei/playground/data/eval/pope/val2014 \
    --answers-file /blob/weiwei/playground/data/eval/pope/answers/llava-v1.5-7b-$CKPT_DIR.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --text_embed_name llava_pope_test \

python llava/eval/eval_pope.py \
    --annotation-dir /blob/weiwei/playground/data/eval/pope/coco \
    --question-file /blob/weiwei/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /blob/weiwei/playground/data/eval/pope/answers/llava-v1.5-7b-$CKPT_DIR.jsonl
