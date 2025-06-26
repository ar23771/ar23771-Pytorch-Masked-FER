# src/evaluate.py
"""
評估腳本：載入訓練好的模型，並在驗證集上產生詳細報告與混淆矩陣。
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from typing import Dict, List

# 從本地模組匯入
from .config import Config
from .model import build_model
from .utils import get_dataloaders, setup_matplotlib_chinese_font

def evaluate_and_report(
    model: torch.nn.Module, 
    dataloaders: Dict[str, torch.utils.data.DataLoader], 
    class_names: List[str], 
    device: torch.device
):
    """在驗證集上進行評估並產生報告與圖表。"""
    model.eval()
    y_pred, y_true = [], []

    print("\n正在驗證集上進行全面評估...")
    for inputs, labels in tqdm(dataloaders['validation']):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
    
    print("\n" + "="*50)
    print("           模型最終評估報告 (Final Evaluation Report)")
    print("="*50)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    print("="*50)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label (預測標籤)')
    plt.ylabel('True Label (真實標籤)')
    plt.title('Confusion Matrix (Binary Classification)')
    plt.show()

def main():
    """評估腳本的主執行函式。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 建立模型並載入已訓練的權重
    model = build_model().to(device)
    print(f"正在從 '{Config.SAVE_PATH}' 載入已訓練的模型...")
    model.load_state_dict(torch.load(Config.SAVE_PATH, map_location=device))
    
    # 準備資料
    dataloaders, _, _, class_names = get_dataloaders()
    
    # 設定繪圖並執行評估
    setup_matplotlib_chinese_font()
    evaluate_and_report(model, dataloaders, class_names, device)

if __name__ == '__main__':
    main()