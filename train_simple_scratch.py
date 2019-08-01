# Import images using the custom image pipeline for BW images.
# Train a very basic conv net
# The conv net and training code is based on the Udacity FMNIST tutorial


# Import all needed libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim import Adam
from torch.autograd import Variable

# Import custom image handling functions
from image_import_BW import get_mean_std, process_image_BW, wormDataset_BW, wormDatasetSampler, count_loader_samples

# These last two are used to save info about how the training progressed
import pickle
import datetime

# Set the full path to the main image directory
# Paths on minimac
#train_dir = '/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/microscope_images_5cat/train'
#test_dir = '/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/microscope_images_5cat/test'

# Paths on Squidward
#train_dir = '/home/vera/VeraPythonScripts/vera_autofocus/microscope_images/train'
#test_dir = '/home/vera/VeraPythonScripts/vera_autofocus/microscope_images/test'

# Practice paths
train_dir = '/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/microscope_images_3cat/practice'
test_dir = '/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/microscope_images_3cat/practice'

num_train = 32
num_test = 32

# Get mean and std of images in the dataset
mean, std = get_mean_std(train_dir, 30)

# Load the images into the dataset
testdata = wormDataset_BW(test_dir, mean, std)

# Augment the training set with vertical and horizontal flip
traindata = wormDataset_BW(train_dir, mean, std)
traindata_hflip = wormDataset_BW(train_dir, mean, std, 'hflip')
traindata_vflip = wormDataset_BW(train_dir, mean, std, 'vflip')
augmented_traindata = torch.utils.data.ConcatDataset([traindata, traindata_hflip, traindata_vflip])

# Get the classes
class_names = augmented_traindata.datasets[0].classes
print('Detected ' + str(len(class_names)) + ' classes in training data')
print(class_names)

# Print out how many images are in the trainloader and testloader
print('Traindata length = ' + str(len(traindata)) + ', testdata length = ' + str(len(testdata)))

trainloader = torch.utils.data.DataLoader(augmented_traindata, batch_size=num_train, shuffle=False)
testloader = torch.utils.data.DataLoader(testdata, batch_size=num_test, shuffle=False)
# Shuffle has to be set to False when using a sampler. If you want shuffling it needs to happen in the sampler
# Removed sampler for now because it slows things down. Replace once the model is working
# sampler = wormDatasetSampler(augmented_traindata)
# sampler = wormDatasetSampler(testdata)

# Check number of samples and distribution into classes for both the test and trainloaders
#train_dict = count_loader_samples(trainloader)
#test_dict = count_loader_samples(testloader)
#print('Classes and sample counts in train loader:')
#print(train_dict)
#print('Classes and sample counts in test loader:')
#print(test_dict)


# Check if cuda is available, and set pytorch to run on GPU or CPU as appropriate
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Cuda available, running on GPU')
else:
    device = torch.device("cpu")
    print('Cuda is not available, running on CPU')
    # Give the user a message so they know what is going on

class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)

    def forward(self, input):

        print(np.shape(input))
        output = self.conv1(input)
        output = self.relu1(output)
        print('Conv layer 1')
        print(np.shape(output))

        output = self.conv2(output)
        output = self.relu2(output)
        print('Conv layer 2')
        print(np.shape(output))

        output = self.pool(output)
        print('Pool layer 1')
        print(np.shape(output))

        output = self.conv3(output)
        output = self.relu3(output)
        print('Conv layer 3')
        print(np.shape(output))

        output = self.conv4(output)
        output = self.relu4(output)
        print('Conv layer 4')
        print(np.shape(output))

        output = self.pool(output)
        print('Pool layer 2')
        print(np.shape(output))

        output = output.view(-1, 16 * 16 * 24)
        print('Reshape')
        print(np.shape(output))

        output = self.fc(output)
        print('Fully connected')
        print(np.shape(output))

        return output

print('model defined')
# Create an instance of the model
model = SimpleNet()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Select a loss function
loss_fn = nn.CrossEntropyLoss()
print('model created')

#Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):

    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch):
    torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
    print("Checkpoint saved")

def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(testloader):
      
        test_acc = []

        #Predict classes using images from the test set
        images.unsqueeze_(0) # Color is 1D, make the tensor have a 4D shape anyway (color x batch x height x width)
        images.transpose_(0, 1) # Transpose to get expected order (batch x color x height x width)

        outputs = model(images)
        _,prediction = torch.max(outputs.data, 1)
        test_acc.append(torch.sum(prediction == labels.data))
        print('Test accuracy: ' + str(test_acc))
    
    return accuracy
        


    #Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / 10000

    return test_acc

def train(num_epochs, losses):

    for epoch in range(num_epochs):

        # Put the model in training mode
        model.train()

        for i, (images, labels) in enumerate(trainloader):
            
            #Clear all accumulated gradients
            optimizer.zero_grad()
            
            # Pass the batch of images to the model, get outputs
            images.unsqueeze_(0) # Color is 1D, make the tensor have a 4D shape anyway (color x batch x height x width)
            images.transpose_(0, 1) # Transpose to get expected order (batch x color x height x width)
            outputs = model(images)

            # Pass the outputs and labels to the loss function, calculate loss
            loss = loss_fn(outputs,labels)
            #Backpropagate the loss
            loss.backward()

            #Adjust parameters according to the computed gradients
            optimizer.step()
            print('Loss: ' + str(loss.item()))
            losses.append(loss.item())

        #Call the learning rate adjustment function
        adjust_learning_rate(epoch)

    return losses

losses = []
losses = train(3, losses)
accuracy = test()

