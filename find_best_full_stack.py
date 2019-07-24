# Loads a trained model, and then tries it on every image in the dataset. Takes the median of the "acceptable"
# focus planes as the best focus.


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
import statistics

# Define functions
def get_prediction(image_path, means, stds):
	# Imports and processes an image, then passes it to the model to generate a prediction
	with torch.no_grad():
		image = process_image(image_path, means, stds)
		image.unsqueeze_(0) # Unsqueeze to add a "dummy" dimension - this would be the batch size in a multi image set
		
		model_start = time.time()
		output = model(image)
		model_end = time.time()

		# Convert the output to the top prediction
		_, prediction_index = torch.max(output.data, 1)
		
		# Conver the index, which is a tensor, into a numpy array
		prediction = torch.Tensor.numpy(prediction_index)
		# Get the value out of the numpy array
		prediction = prediction.item()

	# The focus classes are indicated by numbers, so the index is equivalent to the class name
	return prediction, model_start, model_end


# Set the range limits for the focus stack
plane_range = [0, 44]

# Check if cuda is available, and set pytorch to run on GPU or CPU as appropriate
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Cuda available, running on GPU')
else:
    device = torch.device("cpu")
    print('Cuda is not available, running on CPU')
    # Give the user a message so they know what is going on


# load the pre-trained model
#model_path = '/mnt/purplearray/Vera/vera_autofocus/compare_num_classes/resnet50_5cat.pth'
model_path = '/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/compare_num_classes/resnet50_5cat.pth'
model = torch.load(model_path)
model.eval() # Put the model in eval mode

# Start logging time
start = time.time()

# Set means and stds for the model
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
# Means and stds for Resnet50, update these if using a different model

# Index for the class "acceptabe" - images that are in focus. This index will change based on how
# many classes the classifer has
acceptable = 2

prediction_list = []
plane_list = list(range(plane_range[0], (plane_range[1] + 1)))

# Set a path to a focus stack
#focus_stack = Path('/mnt/purplearray/Pittman_Will/20190521_cyclo_dead/06/2019-05-23t0923 focus')
focus_stack = Path('/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead/06/2019-06-08t2148 focus')
# mod to /mnt/purplearray/ for linux

# Use this for testing functions
#image_path = '/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead/06/2019-06-14t1105 focus/30.png'

model_starts = []
model_ends = []

for current_plane in plane_list: # Iterate through every image in the stack
	# Get an image
	# For now this is pulled from a focus stack using the plane to specify the file name
	if current_plane > 9:
		image_file = str(int(current_plane)) + '.png'
	else:
		image_file = '0' + str(int(current_plane)) + '.png'
	image_path = focus_stack / image_file

	prediction, model_start, model_end = get_prediction(image_path, means, stds)
	model_starts.append(model_start)
	model_ends.append(model_end)

	print(current_plane)
	print('Prediction: ' + str(prediction))
	# Capture the current plane + prediction
	prediction_list.append(prediction)

# Use the plane and prediction list to get the median plane with an acceptable focus
acceptable_planes = []
for i in range(len(plane_list)):
	if prediction_list[i] == acceptable:
		acceptable_planes.append(plane_list[i])
print(acceptable_planes)

# Take the median acceptable focus plane as the best
best_focus = statistics.median(acceptable_planes)

# Stop logging execution time
end = time.time()
run_time = end - start

# After the best plane has been identified, open RisWidget to let the user view the image that has been deemed acceptable
print('Found best: ' + str(best_focus))

# Print out the number of steps, planes + predictions, and total run time
print('Run time: ' + str(run_time))

model_time = np.array(model_ends) -np.array(model_starts)
model_time = np.sum(model_time)
print('Time in model: ', model_time)

from ris_widget import ris_widget  
rw = ris_widget.RisWidget()                                                                                  
import freeimage  
import pathlib  

if best_focus > 9:
	image_file = str(int(best_focus)) + '.png'
else:
	image_file = '0' + str(int(best_focus)) + '.png'
image_path = focus_stack / image_file


rw.add_image_files_to_flipbook(image_path)
