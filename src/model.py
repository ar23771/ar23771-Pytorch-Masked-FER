# src/model.py
"""
定義模型架構和自訂的損失函數。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .config import Config # 使用相對路徑匯入設定

def build_model() -> nn.Module:
    """
    根據設定檔建立並返回一個預訓練模型。
    
    Returns:
        nn.Module: 一個準備好進行遷移學習的模型。
    """
    print(f"正在建立模型: {Config.MODEL_NAME}")
    if Config.MODEL_NAME == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif Config.MODEL_NAME == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        raise ValueError("不支援的模型名稱，請在 config.py 中選擇 'resnet18' 或 'resnet34'")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)
    return model

class FocalLoss(nn.Module):
    """
    Focal Loss 的標準實作，用於解決類別不平衡和困難樣本問題。
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss