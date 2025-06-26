# src/utils.py
"""
放置可重複使用的輔助函式，例如資料載入和繪圖。
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, List
from .config import Config

def get_dataloaders() -> Tuple[Dict[str, DataLoader], Dict[str, int], torch.Tensor, List[str]]:
    """
    準備資料轉換、載入資料集、建立 DataLoader，並計算類別權重。
    
    Returns:
        Tuple: 包含 (dataloaders, dataset_sizes, class_weights, class_names) 的元組。
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3), transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Grayscale(num_output_channels=3), transforms.Resize(256),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    print(f"從 '{Config.DATA_DIR}' 載入資料...")
    image_datasets = {k: datasets.ImageFolder(os.path.join(Config.DATA_DIR, k), data_transforms[k]) for k in ['train', 'validation']}
    dataloaders = {k: DataLoader(v, batch_size=Config.BATCH_SIZE, shuffle=True if k=='train' else False, num_workers=Config.NUM_WORKERS) for k, v in image_datasets.items()}
    dataset_sizes = {k: len(v) for k, v in image_datasets.items()}
    class_names = image_datasets['train'].classes
    print(f"資料集類別: {class_names}")

    class_counts = np.bincount(image_datasets['train'].targets)
    class_weights = [sum(class_counts) / (len(class_counts) * count) for count in class_counts]
    
    return dataloaders, dataset_sizes, torch.tensor(class_weights, dtype=torch.float), class_names

def setup_matplotlib_chinese_font():
    """設定 Matplotlib 以支援中文。"""
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang TC', 'Heiti TC'] 
        plt.rcParams['axes.unicode_minus'] = False
        print("Matplotlib 中文字體設定完成。")
    except Exception:
        print("警告：設定中文字體失敗。")

def plot_history(history: Dict[str, List[float]]):
    """繪製訓練歷史圖表"""
    setup_matplotlib_chinese_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(history['train_acc'], label='Training Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.set_title(f'{Config.MODEL_NAME} Accuracy over Epochs')
    ax1.legend(); ax1.grid(True); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    
    ax2.plot(history['train_loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{Config.MODEL_NAME} Loss over Epochs')
    ax2.legend(); ax2.grid(True); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()