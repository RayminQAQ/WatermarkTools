from torch.utils.data import Dataset
from torchvision import datasets

class mnist(Dataset):
    def __init__(self, dataset_path=None, transform=None, is_train=True):
        """
        Custom MNIST dataset wrapper.

        Args:
            dataset_path (str or None): Path to the dataset directory. If None, it downloads automatically.
            transform (callable or None): Transformations to apply to the images.
            is_train (bool): Whether to load the training set (True) or test set (False).
        """
        super(mnist, self).__init__()
        
        # Local variable for storing the dataset
        self.dataset = -1  
        self.dataset_path = dataset_path  # Default: None
        self.transform = transform  # Default: None
        
        # Load dataset, automatically saves the file into the default location if dataset_path is not provided.
        if self.dataset_path is None:
            # If no dataset path is specified, download and store MNIST in "./data"
            self.dataset = datasets.MNIST(root="./data", train=is_train, download=True)
        else:
            # If dataset path is provided, attempt to load MNIST from that location
            self.dataset = datasets.MNIST(root=self.dataset_path, train=is_train, download=False)
        
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves a single image from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            Transformed image if a transform function is provided, otherwise returns the raw image.
        """
        image, _ = self.dataset[index]  # MNIST dataset provides (image, label), but label is ignored
        if self.transform:
            image = self.transform(image)  # Apply transformations if provided
        return image

    def get_featdim(self):
        """
        Returns the dimensions of the image features.

        Returns:
            tuple: Shape of a sample image after transformation (e.g., (1, 28, 28) for grayscale images).
        """
        sample_image, _ = self.dataset[0]  # Retrieve the first image
        if self.transform:
            sample_image = self.transform(sample_image)  # Apply transformation if provided
        return sample_image.shape  # Return the shape of the transformed image

    
class mnist_interface:
    def __init__(self, dataset_path=None, transform=None):
        """
        MNIST dataset interface to facilitate dataset loading.

        Args:
            - dataset_path
            - transform (callable or None): Transformations to apply to images.
        """
        # Store dataset path, defaulting to None if not provided
        self.dataset_path = dataset_path if dataset_path is not None else None
        self.transform = transform  # Store the transformation function

    def get_train_dataset(self):
        """
        Returns the training dataset.

        Returns:
            mnist: Instance of the custom MNIST dataset with training data.
        """
        return mnist(self.dataset_path, self.transform, is_train=True)

    def get_eval_dataset(self):
        """
        Returns the evaluation (test) dataset.

        Returns:
            mnist: Instance of the custom MNIST dataset with test data.
        """
        return mnist(self.dataset_path, self.transform, is_train=False)