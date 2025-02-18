#!/bin/bash
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-$CKPT_DIR"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR \
        --question-file /blob/weiwei/playground/data/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder /blob/weiwei/playground/data/eval/seed_bench \
        --answers-file /blob/weiwei/playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --text_embed_name llava-seed-bench 
done

wait

output_file=/blob/weiwei/playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /blob/weiwei/playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file /blob/weiwei/playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file /blob/weiwei/playground/data/eval/seed_bench/answers_upload/llava-v1.5-7b-$CKPT_DIR.jsonl

