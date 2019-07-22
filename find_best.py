
# Import libraries
# Import the image processing functions and class
from image_import import process_image, de_process_image, wormDataset

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
import time

# Define functions
def get_prediction(image_path, means, stds)
	# Imports and processes an image, then passes it to the model to generate a prediction

	image = process_image(image_path, means, stds)
	image.unsqueeze_(0) # Unsqueeze to add a "dummy" dimension - this would be the batch size in a multi image set
	output = model(image)

	# Convert the output to the top prediction
	_, prediction_index = torch.max(output.data, 1)
	
	return prediction_index


# Start logging time
start = time.time()

# load the model
model_path = '/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/autofocus_resnet50.pth'
model = torch.load(model_path)
model.eval() # Put the model in eval mode

# Set means and stds for the model
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
# Means and stds for Resnet50, update these if using a different model

# Index for the class "acceptabe" - images that are in focus. This index will change based on how
# many classes the classifer has
acceptable_index = 1

prediction_list = []
plane_list = []

# Set the upper and lower bounds on planes
plane_range = 

# Set the starting plane

while not acceptable_focus:

	# Get an image
	# For now this is pulled from a focus stack using the plane to specify the file name
	# Get the current step position
	image = process_image(image_path, means, stds)
	image.unsqueeze_(0) # Unsqueeze to add a "dummy" dimension - this would be the batch size in a multi image set
	output = model(image)

	# Convert the output to the top prediction
	_, prediction_index = torch.max(output.data, 1)
	prediction = class_names[prediction_index] 

	# Check if the image is acceptable
	if prediction == 'acceptable':
		# Double check
		# Give the image to the model again (could functionalize this)
		# Get output and convert
		if prediction == 'acceptable':
			acceptable_focus = TRUE
		else:
			# Continue on with the rest of the loop, proceeding with the new prediction

	# Capture the current plane + prediction
	prediction_list.append(prediction)
	plane_list.append(plane)

	# Use the class ranges + current plane to decide how many planes to move and in which direction

	# Generate new plane position to go to

# Stop logging execution time
end = time.time()
run_time = end - start

# After the best plane has been identified, open RisWidget to let the user view the image that has been deemed acceptable

# Print out the number of steps, planes + predictions, and total run time
print('Run time: ' + str(run_time))
