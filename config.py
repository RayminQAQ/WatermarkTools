# WatermarkTools/config.py

class Args:
    # 指定要使用的數據集名稱，這裡假設使用 "cifar10"
    dataset = 'cifar10'
    # 如果數據集路徑為 None，表示沒有指定路徑，自動下載到預設路徑
    dataset_path = None  
    # 水印演算法名稱（示意用）
    watermark = 'default_watermark'
    # 模型名稱（示意用）
    model = 'default_model'

args = Args()
