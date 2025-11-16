#对称模块引入
# CUDA_VISIBLE_DEVICES=4 python train_yolo.py --cfg train_config_3ch_sym.yaml训练命令
from ultralytics import YOLO
import yaml
import argparse
import torch
import random
import numpy as np

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def start_training(config_file):
    set_seed(42)
    
    # 1. 加载您的 YAML 配置文件
    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: Failed to load training config file {config_file}. {e}")
        return
    
    # 2. 实例化 YOLO 模型，并传入预训练权重
    model_path = cfg.pop('model', 'yolov11n.pt') 
    model = YOLO(model_path)
    
    # 3. 从配置中移除不适用于 .train() 的键
    if 'task' in cfg: del cfg['task']
    if 'mode' in cfg: del cfg['mode']
    
    # 5. 检查并移除 'fl_gamma'
    if 'fl_gamma' in cfg:
        del cfg['fl_gamma']
        print("Removed 'fl_gamma' parameter to prevent SyntaxError (not a valid YOLO argument).")

    print(f"Starting training on device {cfg['device']} using config from {config_file}...")
    
    # 6. 启动训练，将 YAML 中的所有参数 (cfg) 传入
    model.train(**cfg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to the training config file (.yaml)')
    args = parser.parse_args()
    
    start_training(args.cfg)