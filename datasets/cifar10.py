# WatermarkTools/datasets/cifar10.py
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

from WatermarkTools.config import args

class CIFAR10Dataset(Dataset):
    """封裝 torchvision CIFAR-10 數據集"""
    def __init__(self, root, train=True, transform=None, download=False):
        if transform is None:
            transform = transforms.ToTensor()
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download
        )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class CIFAR10DataModule:
    """
    數據模組：根據 config.py 中的參數設定，決定 CIFAR-10 數據集的路徑與是否自動下載。
    
    此模組除了提供訓練集與測試集外，也從訓練集中隨機分割出驗證集（eval set）。
    """
    def __init__(self):
        # 如果 args.dataset_path 為 None，則自動下載到預設路徑
        if args.dataset_path is None:
            self.data_dir = './data'
            self.download = True
        else:
            self.data_dir = args.dataset_path
            self.download = False

        self.transform = transforms.ToTensor()
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def setup(self):
        # 先載入整個訓練數據集
        full_train_dataset = CIFAR10Dataset(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=self.download
        )
        # 分割出 90% 作為訓練集，10% 作為驗證集
        train_size = int(0.9 * len(full_train_dataset))
        eval_size = len(full_train_dataset) - train_size
        self.train_dataset, self.eval_dataset = random_split(full_train_dataset, [train_size, eval_size])
        
        # 初始化測試數據集
        self.test_dataset = CIFAR10Dataset(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=self.download
        )

    def get_train_dataset(self):
        return self.train_dataset

    def get_eval_dataset(self):
        return self.eval_dataset

    def get_test_dataset(self):
        return self.test_dataset
