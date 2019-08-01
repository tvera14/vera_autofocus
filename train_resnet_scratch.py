# Import images using the custom image pipeline for BW images.
# Make a version of Resnet 50 that is modified to take single channel (BW) images
# instead of color.
# Modification to Resnet is based on this post:
# https://discuss.pytorch.org/t/modify-resnet-or-vgg-for-single-channel-grayscale/22762

# Import all needed libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

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
train_dir = '/home/vera/VeraPythonScripts/vera_autofocus/microscope_images/train'
test_dir = '/home/vera/VeraPythonScripts/vera_autofocus/microscope_images/test'

num_train = 10
num_test = 10

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

trainloader = torch.utils.data.DataLoader(augmented_traindata, sampler = wormDatasetSampler(augmented_traindata), batch_size=num_train, shuffle=False)
testloader = torch.utils.data.DataLoader(testdata, sampler = wormDatasetSampler(testdata), batch_size=num_test, shuffle=False)
# Shuffle has to be set to False when using a sampler. If you want shuffling it needs to happen in the sampler

# Check number of samples and distribution into classes for both the test and trainloaders
train_dict = count_loader_samples(trainloader)
test_dict = count_loader_samples(testloader)
print('Classes and sample counts in train loader:')
print(train_dict)
print('Classes and sample counts in test loader:')
print(test_dict)


'''
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
train_losses, test_losses, accuracy_tracker = [], [], []
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

'''