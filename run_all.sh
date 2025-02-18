#!/bin/bash
bash scripts/v1_5/pretrain.sh
# bash scripts/v1_5/finetune15.sh
bash scripts/v1_5/finetune.sh

nohup python /blob/thinking.py &

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/vqav2.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/gqa.sh
# CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/vizwiz.sh
# CUDA_VISIBLE_DEVICES=1 bash scripts/v1_5/eval/sqa.sh
# CUDA_VISIBLE_DEVICES=2 bash scripts/v1_5/eval/textvqa.sh
# CUDA_VISIBLE_DEVICES=3 bash scripts/v1_5/eval/pope.sh
# CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
# CUDA_VISIBLE_DEVICES=1 bash scripts/v1_5/eval/mmbench.sh
# CUDA_VISIBLE_DEVICES=2 bash scripts/v1_5/eval/mmbench_cn.sh
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/seed.sh
# CUDA_VISIBLE_DEVICES=3 bash scripts/v1_5/eval/llavabench.sh
# CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmvet.sh
