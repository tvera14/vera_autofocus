# Home made image import for pytorch, better accountability on what is happening to the image at each stage
# and what that looks like

def process_image(file_path, means, stds):
    # Function loads a black and white .png image and transforms it into
    # a tensor suitable for use with pytorch
    # Inputs:
    #    file_path: string path to the image (currently complete path)
    #               PosixPaths also work
    #    norms: list of 3 means from the model original training data,
    #        corresponding to RGB
    #    stds: list of 3 standard deviations (RGB) from training data

    # This is intended to be a more getting into the nitty gritty method of importing images for use with pytorch
    # Based on the tutorial available here:
    # https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad

    from PIL import Image
    import numpy as np
    import torch
    
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
    #    tensor: pytorch tensor corresponding to an image
    #    norms: list of 3 means from the model original training data,
    #        corresponding to RGB
    #    stds: list of 3 standard deviations (RGB) from training data
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
  'Characterizes a dataset for PyTorch'
  def __init__(self, file_path, means, stds):
    # Assumes that the images are sorted by class into folders, which are named
    # with the class name
        self.file_path = Path(file_path)
        self.means = means
        self.stds = stds
        #Initialization
        self.classes = [] # Empty list to append class names onto

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
        sample = process_image(image_path, self.means, self.stds)

        # Convert the class folder into the string name of the class
        class_name = str(image_path.parent.stem)
        # Use this to convert the string name into the corresponding numerical label
        label = self.classes.index(class_name)

        return sample, label









