# This is a data sampler. It fits around a dataset when it is passed to the data loader.
# This code is heavily based on the data sampler available at:
# https://github.com/ufoym/imbalanced-dataset-sampler/commit/3aab47c49ab5e045cafb2e2107c71e7d859f887e

"""
## Usage

Simply pass an `ImbalancedDatasetSampler` for the parameter `sampler` when creating a `DataLoader`.
For example:

```python
from sampler import ImbalancedDatasetSampler
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    sampler=ImbalancedDatasetSampler(train_dataset),
    batch_size=args.batch_size, 
    **kwargs
)
"""

import torch
from torch.utils import data


class wormDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # Make a set of indices to iterate through
        self.indices = list(range(len(dataset)))

        # Get the number of samples in the dataset
        self.num_samples = len(self.indices)

        # Make a dictionary with labels as keys and number of samples with that label as values
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)
        # I think these weights are what is passed on to the DataLoader, presumably it knows what to do from there

    def _get_label(self, dataset, idx):
    	# Get the label from the dataset
        # In the wormDataset, each sample is a 1, 2 tensor with an array representing the image + the class
        # tensor[image_array, class_label]
        sample = dataset[idx]
        label = sample[1]
        return label

        #image_import.wormDataset

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
