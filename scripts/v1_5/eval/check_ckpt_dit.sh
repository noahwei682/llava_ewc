#!/bin/bash

VAR_NAME="CKPT_DIR"

# 检测环境变量是否设置
if [ -z "${!VAR_NAME}" ]; then
  echo "环境变量 $VAR_NAME 未设置，脚本退出。"
  exit 1
else
  echo "环境变量 $VAR_NAME 已设置，值为: ${!VAR_NAME}"
fi
VAR_NAME="EVA_CKPT"

# 检测环境变量是否设置
if [ -z "${!VAR_NAME}" ]; then
  echo "环境变量 $VAR_NAME 未设置，脚本退出。"
  exit 1
else
  echo "环境变量 $VAR_NAME 已设置，值为: ${!VAR_NAME}"
fi