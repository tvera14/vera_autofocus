# Based on the Image Classifier in Pytorch tutorial from:
# https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
# Now using the MNIST image set, converted into png
# https://github.com/pjreddie/mnist-csv-png

# Originally used RGB images with the resnet50 pretrained cnn. This is modified to work with
# black and white .png versions of the MNIST dataset.

# ImageFolder will convert the images to RGB, it's actually
# a challenge to keep them in black and white if that is desired for some reason.

# After getting this to work with the MNIST PNG images, it is now modified to work with the
# PNG microscope images generated in the lab.

# Modified more using code snipped from the offical Pytorch transfer learning tutorial:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

# To run individual lines in terminal, type ipython then hit enter

# Import all needed libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
# These last two are used to save info about how the training progressed
import pickle
import datetime


# This tutorial originally had a custom function to import the image data. However, I don't 
# have that data set so I replaced the import code with code that imports the dogs_cats dataset
# from Kaggle:
# https://www.kaggle.com/c/dogs-vs-cats

data_dir = 'microscope_images'
# Make sure you are running the code from the folder that contains data_dir, or it won't work
num_train = 10
num_test = 10

# Define transforms for the training data and testing data
resnet_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


# Load the images from the image folder
traindata = datasets.ImageFolder(data_dir + '/train', transform=resnet_transforms)
testdata = datasets.ImageFolder(data_dir + '/test', transform=resnet_transforms)
# Got a lot of errors with this initially, the issue is that the images need to be in subfolders based on their
# label. This is how pytorch detects and assigns the correct labels. Originally I had all the images 
# stored in one folder and the only indicator was the file name.
# Issue and error described here https://github.com/pytorch/examples/issues/236

# Split the data into non-overlapping sets
# This line causes problems when it is run. It isn't needed for the MNIST data set since that is
# already split into training and test labelled data. Splitting the training and test data
# into folders, rather than trying to split while the program is running, seems to be the
# easier solution.
#train_data, test_data = torch.utils.data.random_split(dogs_cats_data, [num_train, (len(dogs_cats_data) - num_train)]) 

# Load from the training and test sets
trainloader = torch.utils.data.DataLoader(traindata, batch_size=num_train, shuffle=True)
testloader = torch.utils.data.DataLoader(testdata, batch_size=num_test, shuffle=True)

# Get the classes
class_names = traindata.classes
print('Detected ' + str(len(class_names)) + ' classes in training data')
print(class_names)

# Print out how many images are in the trainloader and testloader
print("Train batch size = " + str(num_train) + ', test batch size = ' + str(num_test))
print('Trainloder length = ' + str(len(trainloader)) + ', testloader length = ' + str(len(testloader)))

# Check if cuda is available, and set pytorch to run on GPU or CPU as appropriate
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Cuda available, running on GPU')
else:
    device = torch.device("cpu")
    print('Cuda is not available, running on CPU')
    # Give the user a message so they know what is going on

model = models.resnet50(pretrained=True)
#print(model) 
# Printing the model shows some of the internal layers, not expected to
# understand these but neat to see

# Freeze the pre-trained layers, no need to update featue detection
for param in model.parameters():
    param.requires_grad = False

# Get the number of features the model expects in the final fully connected layer, this is different
# in different models
num_ftrs = model.fc.in_features

# Re-define the final fully connected layer (model.fc, fc = fully connected)
model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), # 2048 inputs to 512 outputs 
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 # The next line needs to be modified for the number of classes
                                 # in the data set. For the microscope images I currently have 
                                 # five classes, so there are 5 outputs
                                 nn.Linear(512, 5), # 512 inputs to 5 outputs
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

# In the next section of code, this error occurred in "logps = model.forward(inputs)"
# RuntimeError: Given groups=1, weight of size 64 3 7 7, expected input[64, 1, 28, 28] to have 3 channels, but got 1 channels instead
# This is because the model and tutorial expect colored data (3 dimensional) while
# the MNIST data set is black and white (1 dimensional)

# The input is (batch size, number of channels, height, width)

# Train the network
epochs = 3
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader)) 
            accuracy_tracker.append(accuracy/len(testloader))                     
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'autofocus_resnet50.pth')

# Save the information about how training went
# Get a unique date and time to id this training round
now = datetime.datetime.now()
time_string = (':').join([str(now.hour), str(now.minute)]) 
date_string = ('-').join([str(now.month), str(now.day), str(now.year)])
file_name = ('_').join(['resnet18_training', date_string, time_string])

fileObject = open(file_name, 'wb')
training_data = [train_losses, test_losses, accuracy_tracker]
pickle.dump(training_data, fileObject)
fileObject.close
fileObject.close()

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()