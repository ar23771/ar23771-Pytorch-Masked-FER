"""
專案設定檔：集中管理所有可調整的參數。
"""

class Config:
    # --- 路徑設定 ---
    # 資料集根目錄
    DATA_DIR = './data/masked_dataset_binary/'
    # 訓練好的模型儲存路徑
    SAVE_PATH = './saved_models/binary_mood_model_final.pth'
    
    # --- 模型設定 ---
    MODEL_NAME: str = 'resnet34'  # 可選 'resnet18', 'resnet34'
    NUM_CLASSES: int = 2

    # --- 訓練超參數 ---
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 2
    
    # --- Focal Loss 參數 ---
    # 聚焦參數，數值越大，越關注困難樣本。推薦 1.0 ~ 5.0
    GAMMA: float = 2.0
    
    # --- 兩階段訓練參數 ---
    # 階段一 (僅訓練分類頭)
    LR_HEAD: float = 0.001
    EPOCHS_HEAD: int = 5
    
    # 階段二 (全局微調)
    LR_FINETUNE: float = 0.0001
    EPOCHS_FINETUNE: int = 20