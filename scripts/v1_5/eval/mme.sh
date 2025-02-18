#!/bin/bash
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi
python -m llava.eval.model_vqa_loader \
    --model-path /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR \
    --question-file /blob/weiwei/playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /blob/weiwei/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /blob/weiwei/playground/data/eval/MME/answers/llava-v1.5-7b-$CKPT_DIR.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --text_embed_name llava_mme \

cd /blob/weiwei/playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b-$CKPT_DIR

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-$CKPT_DIR
