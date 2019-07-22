

# load the model


# Start logging time

prediction_list = []
step_list = []

# Set the upper and lower bounds on planes
plane_range = 

# Set the starting plane

while not acceptable_focus:

	# Get an image
	# For now this is pulled from a focus stack using the plane to specify the file name
	# Get the current step position

	# Give the image to the model

	# Is there a way to get the classes from the model, or do they need to be hard coded?

	# Convert the output to the top prediction

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


