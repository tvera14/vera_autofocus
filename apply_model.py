# Based on the tutorial:
# https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

# These are the steps to take a trained model and apply it on images

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable # This import statement is missing in the original tutorial
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib as plt

data_dir = '/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/microscope_images/test'
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('autofocus_resnet.pth')
model.eval() # This line puts the model in eval, as opppsed to train mode

def predict_image(image):
	# Give an image to the model and get a prediction
	# The image needs to be a PIL image
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0) # Squeeze removes inputs of dimension 1 so the tensor has the shape the network is expecting
    input = Variable(image_tensor) # Variable is a wrapper for tensors that brings additional tools
    input = input.to(device)
    output = model(input)
    # Get an error at this point if the input is in the wrong shape. What is the right shape?
    # I think unsqueeze may be a step to getting the correct shape
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(num):
	# Pull random images from the data files to test the model on
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes # classes = ['acceptable', 'slightly_out_neg', etc.]

    # Select random images from the data set
    indices = list(range(len(data))) # Create a list of numbers to match the number of images in the dataset
    np.random.shuffle(indices)
    idx = indices[:num] # List of random numbers, which will correspond to image indices
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx) # Use SubsetRandomSampler to pull just the images corresponding to the random indices
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num) # Load the sampled images
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels
    # images are arrays
    # labels is a tensor with numbers from 0 to 4.


to_pil = transforms.ToPILImage()
images, labels = get_random_images(5)
#fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()