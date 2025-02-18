#!/bin/bash
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi
python -m llava.eval.model_vqa \
    --model-path /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR \
    --question-file /blob/weiwei/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /blob/weiwei/playground/data/eval/mm-vet/images \
    --answers-file /blob/weiwei/playground/data/eval/mm-vet/answers/llava-v1.5-7b-$CKPT_DIR.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --text_embed_name llava-mm-vet \


mkdir -p /blob/weiwei/playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /blob/weiwei/playground/data/eval/mm-vet/answers/llava-v1.5-7b-$CKPT_DIR.jsonl \
    --dst /blob/weiwei/playground/data/eval/mm-vet/results/llava-v1.5-7b-$CKPT_DIR.json
cd /blob/weiwei/playground/data/eval/mm-vet/MM-Vet
python mm-vet.py
cd -

