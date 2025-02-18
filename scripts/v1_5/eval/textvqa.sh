#!/bin/bash
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi
python -m llava.eval.model_vqa_loader \
    --model-path /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR \
    --question-file /blob/weiwei/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /blob/weiwei/playground/data/eval/textvqa/train_images \
    --answers-file /blob/weiwei/playground/data/eval/textvqa/answers/llava-v1.5-7b-$CKPT_DIR.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --text_embed_name llava_textvqa_val_v051_ocr \

python -m llava.eval.eval_textvqa \
    --annotation-file /blob/weiwei/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /blob/weiwei/playground/data/eval/textvqa/answers/llava-v1.5-7b-$CKPT_DIR.jsonl
