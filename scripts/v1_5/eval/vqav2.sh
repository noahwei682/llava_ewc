#!/bin/bash
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-$CKPT_DIR"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR \
        --question-file /blob/weiwei/playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /blob/weiwei/playground/data/eval/vqav2/test2015 \
        --answers-file /blob/weiwei/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --text_embed_name llava_vqav2_mscoco_test-dev2015 \

done

wait

output_file=/blob/weiwei/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /blob/weiwei/playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

