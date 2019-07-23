
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
import time
from pathlib import Path

# Define functions
def get_prediction(image_path, class_names, means, stds):
	# Imports and processes an image, then passes it to the model to generate a prediction

	image = process_image(image_path, means, stds)
	image.unsqueeze_(0) # Unsqueeze to add a "dummy" dimension - this would be the batch size in a multi image set
	output = model(image)

	# Convert the output to the top prediction
	_, prediction_index = torch.max(output.data, 1)

	# Conver the index, which is a tensor, into a numpy array
	prediction = torch.Tensor.numpy(prediction_index)
	# Get the value out of the numpy array
	prediction = prediction.item()
	
	# Convert the prediction into a class name
	prediction = int(class_names[prediction])

	# The focus classes are indicated by numbers, so the index is equivalent to the class name
	return prediction


def get_new_plane(upper_bound, lower_bound, current_plane, prediction):
	# Determine which focus plane to move to to get another image

	# Define a "steps dictionary" - how many steps is it from a given class to get to acceptable?
	# Goal is to slightly overshoot, so the max number of steps is given
	# The details of the steps_dict are dependent on the sorter used to class images prior to training the model
	steps_dict = {
		0 : 23, # full range / 2 + 1 
		1 : 13, # 2 x 6 steps + 1
		2 : 7, # 6 steps + 1	
		3 : 0, # Placeholder, this should never actually get called
		4 : -7, # Values are negative for out of focus in the positive direction
		5 : -13,
		6 : -23
		}

	steps = steps_dict[prediction]

	if steps < 0:
		# Split the difference between the current plane and the lower bound
		new_plane = int(lower_bound + ((current_plane - lower_bound) // 2))
	else:
		# Splot the difference between the current plane and the upper bound
		 new_plane = int(upper_bound - ((upper_bound - current_plane) // 2))

	return new_plane


# Set the range limits for the focus stack
plane_range = [0, 44]

# Start logging time
start = time.time()

# load the model
model_path = '/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/compare_num_classes/resnet50_7cat.pth'
model = torch.load(model_path)
model.eval() # Put the model in eval mode

# Enter the class names list, for some models the folders load in order but in others they don't. Check
# the Jupyter notebook that documents training the model to verify.

# Set means and stds for the model
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
# Means and stds for Resnet50, update these if using a different model

# Index for the class "acceptabe" - images that are in focus. This index will change based on how
# many classes the classifer has
acceptable = 3

class_names = ['0', '6', '1', '4', '3', '2', '5']

prediction_list = []
plane_list = []

# Set the starting plane
start_plane = 30

# Set a path to a focus stack
focus_stack = Path('/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead/06/2019-05-23t0923 focus')

# Use this for testing functions
#image_path = '/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead/06/2019-06-14t1105 focus/30.png'

upper_bound = plane_range[1]
lower_bound = plane_range[0]

current_plane = start_plane # Current plane is expected to be an integer
acceptable_focus = False
while not acceptable_focus:

	print('Upper bound: ' + str(upper_bound))
	print('Lower bound: ' + str(lower_bound))

	# Get an image
	# For now this is pulled from a focus stack using the plane to specify the file name

	if current_plane > 9:
		image_file = str(int(current_plane)) + '.png'
	else:
		image_file = '0' + str(int(current_plane)) + '.png'
	image_path = focus_stack / image_file

	prediction = get_prediction(image_path, class_names, means, stds)

	print('Prediction: ' + str(prediction))

	# Check if the image is acceptable
	if prediction == acceptable:
		acceptable_focus = True
		# Double check, give the image to the model and see if the prediction matches
		#prediction = get_prediction(image_path, class_names, means, stds)
		#if prediction == acceptable_index:
			# The image is in focus, the loop can terminate
		#	acceptable_focus = True
		
	elif upper_bound == (lower_bound + 1):
		# Stop the loop when the boundaries converge
		acceptable_focus = True


	# Re-set the upper or lower bound
	elif prediction < acceptable: # The focus plane is below the best focus plane
		lower_bound = current_plane
	else: # The focus plane is above the best focus plane
		upper_bound = current_plane

	# Capture the current plane + prediction
	prediction_list.append(prediction)
	plane_list.append(current_plane)

	# Use upper and lower bound + current plane to update the plane
	current_plane = get_new_plane(upper_bound, lower_bound, current_plane, prediction)

# Stop logging execution time
end = time.time()
run_time = end - start

# After the best plane has been identified, open RisWidget to let the user view the image that has been deemed acceptable
print('Found best: ' + image_file)

# Print out the number of steps, planes + predictions, and total run time
print('Run time: ' + str(run_time))

from ris_widget import ris_widget  
rw = ris_widget.RisWidget()                                                                                  
import freeimage  
import pathlib  

rw.add_image_files_to_flipbook(image_path)
