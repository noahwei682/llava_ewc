import subprocess
import multiprocessing
import time
import pickle
import os
import sys
import argparse
import logging
import shlex
import subprocess
import time
from dataclasses import dataclass
from io import StringIO

import torch
from torchvision.models import resnet101
def check_gpu_memory(gpu_index, threshold):
    # 获取指定GPU的内存使用情况
    cmd = f'nvidia-smi --query-gpu=memory.free --format=csv,nounits --id={gpu_index}'
    output = subprocess.check_output(cmd, shell=True)
    free_memory = int(output.decode().strip().split('\n')[1])
    print(f"GPU {gpu_index} free memory: {free_memory}MB")
    return free_memory > threshold
def check_keywords_running(keywords):
    assert isinstance(keywords, list), "keywords should be a list of strings"
    num = 0
    for keyword in keywords:
        cmd = f"ps -ef | grep {keyword}|grep -v grep|wc -l"
        output = subprocess.check_output(cmd, shell=True)
        num = int(output.decode())
        if num > 0:
            print(f"Number of processes with keyword {keyword}: {num}")
            return num
        print(f"Number of processes with keyword {keyword}: {num}")
    return num 
def run_task(task_name, gpu_index):
    cmd = F"CUDA_VISIBLE_DEVICES={gpu_index} python /home/aiscuser/mycode/llava_dynaCrossClipOfflinetextEval/llava_aoqi/llava/train/train_mem.py"
    subprocess.run(cmd, shell=True,stdout=open("/dev/null",mode="w"))
    

def schedule_tasks(task_names, gpu_list, threshold):
    # gpus = list(range(num_gpus))
    gpus = gpu_list
    running_tasks = {}
    # print(env)
    while task_names:
        for gpu in gpus:
            if not task_names:
                break
            if gpu not in running_tasks or not running_tasks[gpu].is_alive():
                if check_gpu_memory(gpu, threshold):
                    # time.sleep(60)
                    # if not check_gpu_memory(gpu, threshold):
                    #     print(f"GPU {gpu} is occupied")
                    #     continue
                    task_name = task_names.pop(0)
                    p = multiprocessing.Process(target=run_task, args=(task_name, gpu))
                    p.start()
                    running_tasks[gpu] = p

        # 检查正在运行的任务，如果已经完成，则移除
        for gpu, process in list(running_tasks.items()):
            if not process.is_alive():
                del running_tasks[gpu]
        time.sleep(60*5)
def run_one_task(cmd):
    subprocess.run(cmd, shell=True,stdout=open("/dev/null",mode="w"))
if __name__ == "__main__":
    keywords = ["run_all"]
    while check_keywords_running(keywords):
        time.sleep(60)
    
    os.chdir("/home/aiscuser/mycode/llava_dynaCrossClipOfflinetextEval/llava_aoqi")

    CKPT_DIR = os.environ.get("CKPT_DIR", None)
    if CKPT_DIR is None:
        raise ValueError("Please set the CKPT_DIR environment variable")
    res_log_dir = os.path.join("/blob/weiwei/llava_log", f"{CKPT_DIR}_logs")
    if not os.path.exists(res_log_dir):
        os.makedirs(res_log_dir)
    # 执行第二个任务组
    task_list1 = [
        "CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/vizwiz.sh",
        "CUDA_VISIBLE_DEVICES=1 bash scripts/v1_5/eval/sqa.sh",
        "CUDA_VISIBLE_DEVICES=2 bash scripts/v1_5/eval/textvqa.sh",
        "CUDA_VISIBLE_DEVICES=3 bash scripts/v1_5/eval/pope.sh",
        "CUDA_VISIBLE_DEVICES=4 bash scripts/v1_5/eval/mme.sh",
        "CUDA_VISIBLE_DEVICES=5 bash scripts/v1_5/eval/mmbench.sh",
        "CUDA_VISIBLE_DEVICES=6 bash scripts/v1_5/eval/mmbench_cn.sh",
        "CUDA_VISIBLE_DEVICES=7 bash scripts/v1_5/eval/mmvet.sh",
    ]
    def add_log_file(cmd, log_dir):
        log_file = os.path.join(log_dir, f"{cmd.split()[-1].split('/')[-1]}.log")
        return f"{cmd} > {log_file}"
    task_list1 = [add_log_file(cmd, res_log_dir) for cmd in task_list1]
    p_list1 = []
    for cmd in task_list1:
        p = multiprocessing.Process(target=run_one_task, args=(cmd,))
        p.start()
        p_list1.append(p)

    for p in p_list1:
        p.join()

    # 执行第2个任务组
    task_list2 = [
        "CUDA_VISIBLE_DEVICES=1 bash scripts/v1_5/eval/vqav2.sh",
        "CUDA_VISIBLE_DEVICES=2 bash scripts/v1_5/eval/gqa.sh"
    ]
    task_list2 = [add_log_file(cmd, res_log_dir) for cmd in task_list2]
    p_list2 = []
    for cmd in task_list2:
        p = multiprocessing.Process(target=run_one_task, args=(cmd,))
        p.start()
        p_list2.append(p)
    for p in p_list2:
        p.join()
    task_list3 = [
        "CUDA_VISIBLE_DEVICES=3 bash scripts/v1_5/eval/llavabench.sh",
        "CUDA_VISIBLE_DEVICES=4 bash scripts/v1_5/eval/seed.sh"
    ]
    task_list3 = [add_log_file(cmd, res_log_dir) for cmd in task_list3]
    subprocess.run(task_list3[0], shell=True,timeout=60*10)
    subprocess.run(task_list3[1], shell=True)


import os
# 使用 nohup 运行你的 Python 脚本
os.system("nohup python /blob/thinking.py > output.log 2>&1 &")
