# This is intended to be a more getting into the nitty gritty method of importing images for use with pytorch
# Based on the tutorial available here:
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad

from PIL import Image
import numpy as np

# Get one image
image_path = '/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/microscope_images/test/acceptable/16_0.png'
img = Image.open(image_path)

# Re-size the image to 256 x 256
width, height = img.size
if width > height:
	img.thumbnail((256, 1000000)) # Constrain width to 256 pixels, thumbnail adjusts heigh proportionally
else:
	img.thumbnail((1000000, 256)) # Constrain height to 256 pixels
# Thumbnail is a better choice than resize since it resizes both axes based on the constraint for just one.
# No need to do calculations to figure out what the second axis is when the larger is 256.
# Thumbnail also operates on the image in place, rather than generating a new image and using up memory.

# Crop out center 224 by 224
left_margin = (img.width-224)/2
bottom_margin = (img.height-224)/2
right_margin = left_margin + 224
top_margin = bottom_margin + 224
img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

# Convert the image into a numpy array
img = np.array(img) / 255



def imshow(image, ax=None, title=None):

    # PyTorch tensors assume the color channel is first
    # but matplotlib assumes is the third dimension
    #image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #image = std * image + mean
    
    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax