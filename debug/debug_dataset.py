# WatermarkTools/run.py
from WatermarkTools.datasets.cifar10 import CIFAR10DataModule

def main():
    data_module = CIFAR10DataModule()
    data_module.setup()
    
    print("訓練數據集大小：", len(data_module.get_train_dataset()))
    print("驗證數據集大小：", len(data_module.get_eval_dataset()))
    print("測試數據集大小：", len(data_module.get_test_dataset()))

if __name__ == '__main__':
    main()
