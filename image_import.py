# Home made image import for pytorch, better accountability on what is happening to the image at each stage
# and what that looks like

def count_images(file_path):
    # Finds class folders, makes a list of classes, and prints the classes and how many images are in each class
    # Useful for checking how many images are in the train and test sets to start with
    # Arguments:
    #   file_path: path to the train or test folder. This should include /train/ or /test/
    # Returns:
    #   image_counter: vector whose elements are the number of images in a given class. Order is the same as class_names
    #   class_names: vector containing string class names. Order matches the counts in image counter, and should
    #       match the order in the dataset, dataloader, and model output as well
    import os
    from pathlib import Path
    
    image_counter = []
    class_names = []
    
    for class_name in sorted(os.listdir(file_path)):
        # Exclude .DS_Store
        if class_name != '.DS_Store':
            
            class_names.append(class_name)

            # Make a Path to the class directory
            class_dir = Path(file_path) / class_name

            # Note that this is set to work with .png images and needs modification
            # to work with other types
            image_counter.append(len(os.listdir(class_dir)))
                          
    return image_counter, class_names

def count_loader_samples(some_dataloader):
    # Iterate through images in the train or test data loader to see what classes (now labels) they have
    # and how many samples there are for each label. This is especially useful when using the data sampler
    # which changes the number of samples from what is in the dataset
    # Arguments:
    #   some_dataloader: a torch dataloader object
    # Returns:
    #   label_dict: dictionary whose keys are the labels present in the dataloader. The values are the
    #       number of samples with that label
    label_dict = {}
    for data in some_dataloader:
        images, labels = data
        for label in labels:
            # Convert the label into a number
            label = torch.Tensor.numpy(label)
            label = label.item()
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1
    return label_dict

def process_image(file_path, means, stds, transform = None):
    # Function loads a black and white .png image and transforms it into
    # a tensor suitable for use with pytorch
    # Inputs:
    #    file_path: string path to the image (currently complete path)
    #               PosixPaths also work
    #    norms: list of 3 means from the model original training data,
    #        corresponding to RGB
    #    stds: list of 3 standard deviations (RGB) from training data
    # Returns:
    #   tensor_RGB: torch tensor with dimensions 224 x 224 x 3 corresponding to the image at the file path

    # This is intended to be a more getting into the nitty gritty method of importing images for use with pytorch
    # Based on the tutorial available here:
    # https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad

    from PIL import Image
    import numpy as np
    import torch
    import random
    
    # Load the image
    img = Image.open(file_path)
    
    # Re-size the image to 256 x 256
    width, height = img.size
    if width > height:
        img.thumbnail((256, 1000000)) # Constrain width to 256 pixels, thumbnail adjusts heigh proportionally
    else:
        img.thumbnail((1000000, 256)) # Constrain height to 256 pixels
        
    # Crop out center 224 by 224
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Convert the image into a numpy array
    img = np.array(img)
    
    # Divide by the max value to get range 0 to 1
    img = img / np.ptp(img)
    
    # It is easiest to do the transforms at this stage
    if transform == 'hflip':
        # Flip the image left to right
        img = np.fliplr(img)
        
    elif transform == 'vflip':
        # Flip the image upside down
        img = np.flipud(img)
        
    else:
        # If transform is anything else, don't transform the image
        pass
    
    Rm, Gm, Bm = means # provided mean
    Rstd, Gstd, Bstd = stds # provided std
    
    # Make 3 normalized arrays from one 224 x 224 array
    R = (img - Rm)/ Rstd
    G = (img - Gm)/ Gstd
    B = (img - Bm)/ Bstd
    
    # Stack the three normalized arrays to make an "RGB" image
    img_RGB = np.stack([R, G, B])
    
    # Convert the array into a tensor
    tensor_RGB = torch.from_numpy(img_RGB).type(torch.FloatTensor)
    
    return tensor_RGB

def de_process_image(tensor, means, stds):
    # Function takes a tensor corresponding to an image from pytorch and
    # converts it back into a numpy array (may add PIL image). This is useful
    # to check on what is being passed to the network at the end of the full import process
    # Inputs:
    #    tensor: 3D RGB pytorch tensor corresponding to an image
    #    means: list of 3 means from the model original training data,
    #        corresponding to RGB
    #    stds: list of 3 standard deviations (RGB) from training data
    # Returns:
    #   img: image as a 2D numpy array
    # Note that I haven't bothered to undo any transforms or convert all the way to PIL image
    # in this function. This is easy enough to do with the numpy array if needed for some
    # reason

    from PIL import Image
    import numpy as np
    import torch
    
    # Convert the tensor into a numpy array
    img_RGB = torch.Tensor.numpy(tensor)

    # Get the means and stds
    Rm, Gm, Bm = means # provided mean
    Rstd, Gstd, Bstd = stds # provided std
    
    # Take one 224 x 224 stack off the 3 x 224 x 224 "RGB" image
    img = img_RGB[1, :, :]
    
    # Breakout the means and stds. These are different for each of
    # the layers, I am making the assumption that the one I took out
    # is red. This could cause problems if the mean and std for green
    # or blue is very different.
    Rm, Gm, Bm = means # provided mean
    Rstd, Gstd, Bstd = stds # provided std
    
    # De-normalize using mean and std for red
    img = img * Rstd + Rm
    
    # At this point I am only returning the de-normalized numpy array
    # If a PIL image is desired code will need to be added to do that
    
    return img
 
# Make a dataset loosely based on ImageFolder, but that uses the more hands on import
import torch
from torch.utils import data
from pathlib import Path
import os
import glob

class wormDataset(data.Dataset):
# Loads images from the specified folder into a format suitable for use with pytorch dataloader
# Images for import must be sorted into folders labelled with the desired class name. The name of the
# folder will become the label for the class.
# The classes will be ordered alphanumerically, and this order will persist through to the output
# of a model trained using the data in the dataset.
# Arguments:
#   file_path: string or Path object filename of the directory where the class folders are kept.
#       This is generally the test or train folder.
#   means: list of 3 means from the model's original training data corresponding to RGB
#   stds: list of 3 standard deviations (RGB) from training data
#   tranform: string indicating a transform to be performed for data augmentation.
#       hflip = flip horizontally
#       vflip = flip vertically
#       default is None
# Returns:
#   Dataset object containing the specified data suitable for use with Pytorch

    def __init__(self, file_path, means, stds, transform = None):

        self.file_path = Path(file_path)
        self.means = means
        self.stds = stds
        #Initialization
        self.classes = [] # Empty list to append class names onto
        self.transform = transform

        # Indexed list of class_folder/image.png
        #self.list_IDs = list_IDs
        self.image_paths = []

        # Find class folders on the file path
        for class_name in sorted(os.listdir(file_path)):
            # Use of sorted is important, numbered classes will import in order which is really helpful later

            # Exclude .DS_Store
            if class_name != '.DS_Store':

                # Save the class name to the classes list
                self.classes.append(class_name)

                # Make a Path to the class directory
                class_dir = self.file_path / class_name

                # Note that this is set to work with .png images and needs modification
                # to work with other types
                for image in class_dir.glob('*.png'):
                    # Add the path to the image to the list of image paths
                    self.image_paths.append(image)
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Needs a list with all the filenames in the list
        # Here, this is the name of the specific image + the enclosing class folder

        # Uses the index to get class_folder/image_name
        image_path = self.image_paths[index]

        # Load data and get label
        sample = process_image(image_path, self.means, self.stds, self.transform)

        # Convert the class folder into the string name of the class
        class_name = str(image_path.parent.stem)
        # Use this to convert the string name into the corresponding numerical label
        label = self.classes.index(class_name)


import torch
from torch.utils import data

class wormDatasetSampler(torch.utils.data.sampler.Sampler):
# Randomly samples from the dataset. Pulls fewer images from classes with a large number of samples and duplicates
# some samples from classes with a small number of samples to even out the number of samples per class and
# prevent bias in training.
# This code is heavily based on this repo: https://github.com/ufoym/imbalanced-dataset-sampler
# Most modifications are so it will work with the wormDataset
# Arguments:
#   dataset: Pytorch dataset object containing samples to load. Assumes created with wormDataset 
# Returns:
#   Dataloader object that can be used for training models in Pytorch
# Syntax for using the sampler:
#   trainloader = torch.utils.data.DataLoader(traindata, sampler = wormDatasetSampler(traindata), batch_size=num_train, shuffle=False)
#   Note that shuffle has to be False since shuffling is now being directed by the sampler.

    def __init__(self, dataset, indices=None):

        # Make a set of indices to iterate through
        self.indices = list(range(len(dataset)))

        
        # Don't replace samples - samples have already been agumented, take one that isn't in the set already
        self.replacement = False

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
        
        # Limit the number of samples to 3x the smallest population
        min_pop = 1000000
        for label in label_to_count:
            if label_to_count[label] < min_pop:
                min_pop = label_to_count[label]
        self.num_samples = 3 * min_pop
        
        

    def _get_label(self, dataset, idx):
        # Get the label from the dataset
        # In the wormDataset, each sample is a 1, 2 tensor with an array representing the image + the class
        # tensor[image_array, class_label]
        sample = dataset[idx]
        label = sample[1]
        return label

        #image_import.wormDataset

    def __iter__(self):
        
        # The sampler has to specify how to iterate through itself
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))     

    def __len__(self):
        return self.num_samples


        return sample, label



