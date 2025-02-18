#!/bin/bash
# 要检测的环境变量名称
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi

python -m llava.eval.model_vqa \
    --model-path /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR \
    --question-file /blob/weiwei/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder /blob/weiwei/playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file /blob/weiwei/playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b-$CKPT_DIR.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --text_embed_name llava-bench-in-the-wild-questions \


mkdir -p /blob/weiwei/playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question /blob/weiwei/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context /blob/weiwei/playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        /blob/weiwei/playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        /blob/weiwei/playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b-$CKPT_DIR.jsonl \
    --output \
        /blob/weiwei/playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-7b-$CKPT_DIR.jsonl

python llava/eval/summarize_gpt_review.py -f /blob/weiwei/playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-7b-$CKPT_DIR.jsonl
