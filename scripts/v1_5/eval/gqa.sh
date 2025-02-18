#!/bin/bash
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-$CKPT_DIR"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/blob/weiwei/playground/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR \
        --question-file /blob/weiwei/playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder /blob/weiwei/playground/data/eval/gqa/data/images \
        --answers-file /blob/weiwei/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --text_embed_name llava_gqa_testdev_balanced \

done

wait

output_file=/blob/weiwei/playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /blob/weiwei/playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst /blob/weiwei/playground/data/eval/gqa/answers/$SPLIT/$CKPT/testdev_balanced_predictions.json

cd $GQADIR
python eval.py --tier testdev_balanced  --predictions ../answers/$SPLIT/$CKPT/{tier}_predictions.json
