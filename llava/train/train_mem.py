import logging

# 配置根日志记录器
logging.basicConfig(level=logging.INFO)
# 创建一个 FileHandler，将日志写入指定文件
file_handler = logging.FileHandler('all_logs.log')
file_handler.setLevel(logging.INFO)  # 设置要捕捉的日志级别

# 创建一个日志格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 获取根日志记录器并添加 FileHandler
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
def mannual_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import os
seed = int(os.environ.get("SEED", 0))
assert seed is not None, "Please set the SEED environment variable"
mannual_seed(seed)

# import os
# # 获取当前工作目录
# current_directory = os.getcwd()
# print(f"当前工作目录: {current_directory}")
# # 设置新的工作目录
# new_directory = '/home/aiscuser/mycode/llava_unfreezeCLIP/llava_aoqi'
# os.chdir(new_directory)
# # 验证是否切换到新目录
# print(f"新的工作目录: {os.getcwd()}")


from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
