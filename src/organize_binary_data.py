# src/organize_binary_data.py
"""
資料準備腳本：
將原始的7分類資料集，自動整理成模型訓練所需的二元分類格式。
這個腳本會讀取 config.py 中的路徑設定。
"""
import os
import shutil
from typing import List

# 從本地模組匯入 Config
from .config import Config

def create_binary_dataset(
    source_base: str, 
    dest_base: str, 
    positive_classes: List[str], 
    negative_classes: List[str]
):
    """
    從來源路徑建立一個結構正確的二元分類資料集。

    Args:
        source_base (str): 原始7分類資料集的根目錄路徑。
        dest_base (str): 新的二元分類資料集的目標根目錄路徑。
        positive_classes (List[str]): 要被歸類為正面的類別名稱列表。
        negative_classes (List[str]): 要被歸類為負面的類別名稱列表。
    """
    # 如果目標資料夾已存在，先刪除以確保乾淨的開始
    if os.path.exists(dest_base):
        print(f"偵測到已存在的 '{dest_base}'，正在將其刪除...")
        shutil.rmtree(dest_base)
    
    print(f"正在建立新的目標資料夾 '{dest_base}'...")
    
    # 遍歷 train 和 validation
    for data_type in ['train', 'validation']:
        print(f"\n--- 正在處理 {data_type} 資料夾 ---")
        
        # 建立目標子資料夾 (e.g., ./data/masked_dataset_binary/train/0_positive)
        dest_positive_dir = os.path.join(dest_base, data_type, '0_positive')
        dest_negative_dir = os.path.join(dest_base, data_type, '1_negative')
        os.makedirs(dest_positive_dir, exist_ok=True)
        os.makedirs(dest_negative_dir, exist_ok=True)
        
        # 複製 Positive 類別的檔案
        copied_count = 0
        for class_name in positive_classes:
            source_class_path = os.path.join(source_base, data_type, class_name)
            if not os.path.isdir(source_class_path): continue
            for filename in os.listdir(source_class_path):
                shutil.copy(os.path.join(source_class_path, filename), dest_positive_dir)
                copied_count += 1
        print(f"已複製 {copied_count} 個檔案至 '{dest_positive_dir}'")

        # 複製 Negative 類別的檔案
        copied_count = 0
        for class_name in negative_classes:
            source_class_path = os.path.join(source_base, data_type, class_name)
            if not os.path.isdir(source_class_path): continue
            for filename in os.listdir(source_class_path):
                shutil.copy(os.path.join(source_class_path, filename), dest_negative_dir)
                copied_count += 1
        print(f"已複製 {copied_count} 個檔案至 '{dest_negative_dir}'")
    
    print("\n==========================================")
    print("新的二元分類資料集建立完畢！")
    print("==========================================")

def main():
    """主執行函式"""
    # 定義分類規則
    positive_classes = ['happy', 'neutral']
    negative_classes = ['angry', 'disgust', 'fear', 'sad', 'surprise']
    
    # 原始資料夾的路徑，這裡我們需要手動指定，因為它是在新資料夾產生前的來源
    # 注意：Config.DATA_DIR 指的是轉換後的路徑，這裡要用轉換前的
    source_dataset_path = './data/masked_dataset/' # 假設原始資料集放在這裡
    
    if not os.path.isdir(source_dataset_path):
        print(f"!!! 致命錯誤：找不到原始資料集路徑 '{source_dataset_path}'")
        print("請先遵照 README.md 的指示下載並放置好原始的7分類資料集。")
        return
        
    create_binary_dataset(
        source_base=source_dataset_path,
        dest_base=Config.DATA_DIR, # 目標路徑從 config 讀取
        positive_classes=positive_classes,
        negative_classes=negative_classes
    )

if __name__ == '__main__':
    main()