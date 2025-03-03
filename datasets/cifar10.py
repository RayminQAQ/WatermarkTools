from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms

class cifar10(Dataset):
    def __init__(self, dataset_path=None, transform=None, is_train=True):
        """
        Custom CIFAR-10 dataset wrapper.

        Args:
            dataset_path (str or None): Path to the dataset directory. If None, it downloads automatically.
            transform (callable or None): Transformations to apply to the images.
            is_train (bool): Whether to load the training set (True) or test set (False).
        """
        super(cifar10, self).__init__()
        
        self.dataset_path = dataset_path  # Default: None
        self.transform = transform  # Default: None

        # Load CIFAR-10 dataset.
        # 若 dataset_path 為 None 或空字串，則下載並存放在 "./data" 目錄下
        if self.dataset_path is None or self.dataset_path == "":
            self.dataset = datasets.CIFAR10(root="./data", train=is_train, download=True, transform=self.transform)
        else:
            self.dataset = datasets.CIFAR10(root=self.dataset_path, train=is_train, download=False, transform=self.transform)
        
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves a single image from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            The transformed image.
        """
        image, _ = self.dataset[index]  # CIFAR-10 provides (image, label); 忽略 label
        return image

    def get_feat(self):
        """
        Returns a sample image after applying transformation.
        """
        sample_image, _ = self.dataset[0]
        return sample_image
    
    def get_featdim(self):
        """
        Returns the shape of a sample image after transformation.
        """
        sample_image, _ = self.dataset[0]
        return sample_image.shape

    
class cifar10_interface:
    def __init__(self, dataset_path=None, transform=None):
        """
        CIFAR-10 dataset interface to facilitate dataset loading.

        Args:
            dataset_path (str or None): Path to the dataset directory. If None, it downloads automatically.
            transform (callable or None): Transformations to apply to the images.
        """
        self.dataset_path = dataset_path if dataset_path is not None else None
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def get_train_dataset(self):
        """
        Returns the training dataset.
        """
        return cifar10(self.dataset_path, self.transform, is_train=True)

    def get_eval_dataset(self):
        """
        Returns the evaluation (test) dataset.
        """
        return cifar10(self.dataset_path, self.transform, is_train=False)
