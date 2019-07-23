# Looks through a directory for focus stacks that have been labelled, takes the images
# and uses filenames to caluclate how far the image is from the best focus. Then 
# copies the images into folders that correspond to how far off it is. This prepares
#. the images for import and use with Pytorch using ImageFolder.


# How to run this from Squidward (GPU)
#ssh vera@squidward.wucon.wustl.edu # Connect to Squidward
# The code to create train and test directories with the subdirectories for categories does
# not work on Squidward AT ALL, and I do not know why. So long as the correct sorted and
# experiment directories are provided, the loops to go through folders and copy images work great.

# Can use choose_best.py with the path to the sorted images to verify that the
# sorter is working


from pathlib import Path
import os
import glob
import shutil

# Specify a directory to copy the sorted images into
sorted_dir = Path('/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/microscope_images/')
# Squidward
#sorted_dir = Path('/home/vera/VeraPythonScripts/vera_autofocus/microscope_images')  


# This version of the sorter makes 7 classes:
# 0) Very out negative
# 1) Slightly out negative
# 2) Acceptable
# 3) Slightly out positive
# 4) Very out positive

# This method of making directories does not work on Squidward. I get a "file not found" error
# and I'm like.....that's the point? Have to go in and manually make all the directories. Boo.
for subfolder in ['train', 'test']:
	subdir = sorted_dir / subfolder
	for i in range(5):
		(subdir / str(i)).mkdir()
train_test_flipper = 'train' # Use this variable to alternate saving images in the train and test folders
sorted_dir = sorted_dir / train_test_flipper

# Set a directory to search in
experiment_dir = Path('/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead/')
# Squidward
#experiment_dir = Path('/mnt/purplearray//Pittman_Will/20190521_cyclo_dead/')

# Set the step ranges here to make them easy to change in the future
acceptable = 1
slightly_out = 6

# Keep track of how many images and classes have been generated
image_counter = [0, 0, 0, 0, 0]

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

				# Use the distance to decide which folder the image should be saved in
				if distance < (-1 * slightly_out):
					class_folder = sorted_dir / '0'
					image_counter[0] += 1
				elif distance < (-1 * acceptable):
					class_folder = sorted_dir / '1'
					image_counter[1] += 1
				elif distance <= acceptable:
					class_folder = sorted_dir / '2'
					image_counter[2] += 1
				elif distance <= slightly_out:
					class_folder = sorted_dir / '3'
					image_counter[3] += 1
				else:
					class_folder = sorted_dir / '4'
					image_counter[4] += 1

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













#Path('.').glob('**/* focus')

# Set a directory to copy the images to. They should be copied because the
# originals are kept in a system of directories to identify them.



# When copying the images, rename using parent directory info so there is some
# record of where the image came from





# Set bounds - images greater than a certain amount out of focus go in a greater
# than category. Do this here so it can easily be adjusted.




# Collect info on how many directories were found, how many images total, and per
# category counts.


