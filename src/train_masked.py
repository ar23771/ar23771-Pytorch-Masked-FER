# src/train.py
"""
主訓練腳本：組織整個訓練流程，包括兩階段訓練、儲存模型和繪圖。
"""
import torch
import torch.optim as optim
import copy
from tqdm import tqdm
from typing import Dict, List

# 從本地模組匯入
from .config import Config
from .model import build_model, FocalLoss
from .utils import get_dataloaders, plot_history

def train_model(
    model: torch.nn.Module, 
    criterion: torch.nn.Module, 
    dataloaders: Dict[str, torch.utils.data.DataLoader], 
    dataset_sizes: Dict[str, int], 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler, 
    num_epochs: int, 
    device: torch.device
) -> tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    模型訓練的核心迴圈。

    Args:
        (各種模型、資料、優化器等物件)

    Returns:
        Tuple: 包含 (訓練好的最佳模型, 訓練歷史紀錄) 的元組。
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}' + ' | ' + '-'*10)
        for phase in ['train', 'validation']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().cpu().item() / dataset_sizes[phase]

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                if scheduler: scheduler.step()
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f'此階段最佳驗證準確率: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, history

def main():
    """主執行函式，組織所有訓練步驟。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, dataset_sizes, class_weights, _ = get_dataloaders()
    model = build_model().to(device)
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=Config.GAMMA)
    print(f"已啟用 Focal Loss (gamma={Config.GAMMA})")

    # STAGE 1
    print("\n" + "="*25 + "\n STAGE 1: 訓練分類頭\n" + "="*25)
    for param in model.parameters(): param.requires_grad = False
    for param in model.fc.parameters(): param.requires_grad = True
    optimizer_head = optim.Adam(model.fc.parameters(), lr=Config.LR_HEAD)
    scheduler_head = optim.lr_scheduler.StepLR(optimizer_head, step_size=3, gamma=0.1)
    model, history_head = train_model(model, criterion, dataloaders, dataset_sizes, optimizer_head, scheduler_head, Config.EPOCHS_HEAD, device)

    # STAGE 2
    print("\n" + "="*25 + "\n STAGE 2: 全局微調\n" + "="*25)
    for param in model.parameters(): param.requires_grad = True
    optimizer_finetune = optim.Adam(model.parameters(), lr=Config.LR_FINETUNE)
    scheduler_finetune = optim.lr_scheduler.StepLR(optimizer_finetune, step_size=5, gamma=0.1)
    model, history_finetune = train_model(model, criterion, dataloaders, dataset_sizes, optimizer_finetune, scheduler_finetune, Config.EPOCHS_FINETUNE, device)

    # 儲存與繪圖
    print(f"\n訓練完成！正在儲存模型至 '{Config.SAVE_PATH}'")
    torch.save(model.state_dict(), Config.SAVE_PATH)
    print("模型儲存完畢。")
    
    full_history = {k: history_head[k] + history_finetune[k] for k in history_head}
    plot_history(full_history)

if __name__ == '__main__':
    main()