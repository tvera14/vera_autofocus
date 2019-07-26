# Looks through a directory for focus stacks that have been labelled, takes the images
# and uses filenames to caluclate how far the image is from the best focus. Then 
# copies the images into folders that correspond to how far off it is. This prepares
#. the images for import and use with Pytorch using ImageFolder.

# Originally the three categories were acceptable, slightly out, and very out. Further testing
# showed however that the model is REALLY good at telling whether the focal plane is too far
# above or below the best focus, and not so good at telling really out from a little out. Therefor
# the three categories are now above, acceptable, and below.


# How to run this from Squidward (GPU)
#ssh zplab@squidward.wucon.wustl.edu # Connect to Squidward
#zplab@squidward.wucon.wustl.edu's password: # Enter zplab password
#zplab@squidward:~$ ls /mnt # Check if the drive is mounted
#zplab@squidward:~$ cd /mnt/purplearray/ # Get into purple array
# Can find more on this and other linux procedures in linux notes in Will's folder


from pathlib import Path
import os
import glob
import shutil

# Specify a directory to copy the sorted images into
sorted_dir = Path('/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/microscope_images_3cat/')
# Squidward
#sorted_dir = Path('/home/vera/VeraPythonScripts/vera_autofocus/microscope_images')  

# This version of the sorter makes 3 classes:
# 0) Out of focus negative (below)
# 1) Acceptable 
# 2) Out of focus positive (above)

# Make folders for the 3 classes
for subfolder in ['train', 'test']:
	subdir = sorted_dir / subfolder
	for i in range(3):
		(subdir / str(i)).mkdir()

train_test_flipper = 'train' # Use this variable to alternate saving images in the train and test folders
sorted_dir = sorted_dir / train_test_flipper

# Set a directory to search in
experiment_dir = Path('/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead/')
# Squidward
#experiment_dir = Path('/mnt/purplearray//Pittman_Will/20190521_cyclo_dead/')

# Set the step ranges here to make them easy to change in the future
acceptable_range = 1 # Images within this many steps of the correct focus are deemed acceptable

# Keep track of how many images and classes have been generated
image_counter = [0, 0, 0]

# Iterate through all sub directories looking for best_focus.txt files.
for worm in os.listdir(experiment_dir):
	worm_dir = experiment_dir / worm
	print('New worm')

	for stack in worm_dir.glob('* focus'):
		stack_dir = worm_dir / stack
		print('Found a stack')

		q = stack_dir / 'best_focus.txt'
		# Check if best focus has been noted for this stack
		if q.exists():

			# Switch between saving images in the train and test folders
			sorted_dir = sorted_dir.parent
			if train_test_flipper == 'train':
				train_test_flipper = 'test'
			else:
				train_test_flipper = 'train'
			sorted_dir = sorted_dir / train_test_flipper

			# Read the filename of the best focus image out of the textfile, then convert the
			# filename (ex. 20.png) into an integer
			best_focus = int(str(q.read_text()).split('.')[0])

			# Go through the directory to find all .png images
			for image in stack_dir.glob('*.png'):

				# Convert the stem of the filename into an int, and calculate the distance (in steps)
				# from that image to the best focus plane
				distance = int(image.stem) - best_focus
				print(distance)

				# Use the absolute value of the distance to decide which folder the image should be saved in
				if distance <= (-1 * acceptable_range):
					class_folder = sorted_dir / '0'
					image_counter[0] += 1
				elif distance <= acceptable_range:
					class_folder = sorted_dir / '1'
					image_counter[1] += 1
				else:
					class_folder = sorted_dir / '2'
					image_counter[2] += 1

				name_counter = 0
				image_name = str(image.stem) + '_' + str(name_counter) + '.png'
				# Check if the image name + counter already exists in the folder. If it does, increase
				# the counter to get a unique image name
				while (class_folder / image_name).exists():
					name_counter = name_counter + 1
					image_name = str(image.stem) + '_' + str(name_counter) + '.png'

				# Save the image file in the appropriate class folder
				shutil.copy(str(image), (str(class_folder) + '/' + image_name))

				

print(image_counter)


