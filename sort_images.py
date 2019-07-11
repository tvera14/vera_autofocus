# Looks through a directory for focus stacks that have been labelled, takes the images
# and uses filenames to caluclate how far the image is from the best focus. Then 
# copies the images into folders that correspond to how far off it is. This prepares
#. the images for import and use with Pytorch using ImageFolder.


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
sorted_dir = Path('/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/microscope_images/train')


# Set a directory to search in
experiment_dir = Path('/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead/')
# In order to access a purple array directory from Squidward include /mnt/purplearray/
# instead of Volumes/purplearray

# Iterate through all sub directories looking for best_focus.txt files.
#for worm in os.listdir(experiment_dir):
	 #for focus_stack in Path(worm).glob('**/*focus'):
	 	#print('Found focus stack')
	 	#best_focus = focus_stack / 'best_focus.txt'
	 	#if best_focus.exists():
	 	#	print('Found best focus')

worm_dir = Path('/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead/04/')

for stack in worm_dir.glob('* focus'):
	stack_dir = worm_dir / stack
	q = stack_dir / 'best_focus.txt'
	# Check if best focus has been noted for this stack
	if q.exists():
		# Read the filename of the best focus image out of the textfile, then convert the
		# filename (ex. 20.png) into an integer
		best_focus = int(str(q.read_text()).split('.')[0])

		# Go through the directory to find all .png images
		for image in stack_dir.glob('*.png'):
			# Convert the stem of the filename into an int, and calculate the distance (in steps)
			# from that image to the best focus plane
			distance = int(image.stem) - best_focus
			print(distance)

			# Check if a folder already exists in the sorted directory for this distance
			class_folder = sorted_dir / str(distance)
			if not class_folder.exists():
				# Make a folder for the class if it doesn't already exist
				class_folder.mkdir()

			# Save the image file in the appropriate class folder
			shutil.copy(str(image), str(class_folder))














#Path('.').glob('**/* focus')

# Set a directory to copy the images to. They should be copied because the
# originals are kept in a system of directories to identify them.



# When copying the images, rename using parent directory info so there is some
# record of where the image came from





# Set bounds - images greater than a certain amount out of focus go in a greater
# than category. Do this here so it can easily be adjusted.




# Collect info on how many directories were found, how many images total, and per
# category counts.


