"""
    Template for a generic dataset class
"""

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, root_dir=None, transform=None, is_train=True):
        """
        Base dataset template for PyTorch's Dataset class.

        Args:
            root_dir (str or None): Path to the dataset directory. If None, a default location may be used.
            transform (callable or None): Transformations to apply to the data.
            is_train (bool): Indicates whether the dataset is for training or evaluation.
        """
        super().__init__()

        # Variables
        self.dataset = None  # Placeholder for the actual dataset
        self.root_dir = root_dir  # Default: None
        self.transform = transform  # Default: None
        self.is_train = is_train  # Whether to load training or test data

        # Load dataset - Users should implement dataset-specific loading logic here
        self.load_dataset()

        # Error handling
        if self.dataset is None:
            raise ValueError("A dataset instance must be initialized in the load_dataset() method.")

        if self.transform is None:
            print("Warning: No transform has been provided for BaseDataset.")

    def load_dataset(self):
        """
        Load the dataset based on the given root directory.
        This method should be overridden in subclasses to load dataset-specific data.
        """
        raise NotImplementedError("Subclasses must implement the load_dataset() method.")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            Any: Data sample (image, text, etc.) after applying transformations.
        """
        data = self.dataset[index]

        # Apply transformations if available
        if self.transform:
            data = self.transform(data)

        return data

    def get_featdim(self):
        """
        Returns the feature dimension of a sample.

        Returns:
            tuple: Shape of the dataset sample after transformations.
        """
        sample = self.dataset[0]

        # Apply transformation if provided
        if self.transform:
            sample = self.transform(sample)

        return sample.shape
    
class BaseDataset_interface:
    def __init__(self, dataset_path, transform):
        """
        BaseDataset dataset interface to facilitate dataset loading.

        Args:
            - dataset_path.
            - transform (callable or None): Transformations to apply to images.
        """
        # Store dataset path, defaulting to None if not provided
        self.dataset_path = dataset_path if dataset_path is not None else None
        self.transform = transform  # Store the transformation function

    def get_train_dataset(self):
        """
        Returns the training dataset.

        Returns: Instance of the custom BaseDataset dataset with training data.
        """
        return BaseDataset(self.dataset_path, self.transform, is_train=True)

    def get_eval_dataset(self):
        """
        Returns the evaluation (test) dataset.

        Returns: Instance of the custom BaseDataset dataset with test data.
        """
        return BaseDataset(self.dataset_path, self.transform, is_train=False)