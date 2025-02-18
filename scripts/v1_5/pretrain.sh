#!/bin/bash
# 要检测的环境变量名称
bash ./scripts/v1_5/eval/check_ckpt_dit.sh
if [ $? -ne 0 ]; then
  exit 1
fi

export PYTHONPATH=/home/aiscuser/mycode/llava_clipSingleText/llava_aoqi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FORCE_TORCHRUN=8
deepspeed --num_gpus=8 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /blob/weiwei/codes/llava_aoqi/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --text_embedding_path /blob/weiwei/text_embedding/blip_laion_cc_sbu_558k.pt \
    --image_folder /blob/weiwei/codes/llava_dynaCrossClipOfflinetext/llava_aoqi/playground/data/LLaVA-Pretrain/images \
    --vision_tower EVA02-CLIP-L-14-336 \
    --eva_ckpt_path $EVA_CKPT \
    --run_name llava-v1.5-7b-$CKPT_DIR-pretrain \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /blob/weiwei/llava_checkpoint/llava-v1.5-7b-$CKPT_DIR-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --seed 0 \
    --vision_encoder_lr 1e-4 \
    --training_stage 1.0 \
