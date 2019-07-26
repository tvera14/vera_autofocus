# Program allows the user to view a flopbook of images and choose the best focus. Type best() and a file with the
# filename of the best focused image will be saved in the same folder as the images.

# Run this by starting ipython, then "run load_flipbook.py". Just running the file outside ipython will cause it to
# immediately
# Run from inside the same directory as this script

from ris_widget import ris_widget  
from PyQt5 import Qt
import freeimage  
from pathlib import Path  
import os

# Define the RisWidget window
rw = ris_widget.RisWidget()  

# Define the layout in the window (so buttons can be added to it)
l = rw.flipbook.layout()

# Define functions that will be linked to buttons
# Define a function that will grab the filename of the best focus image
def best():
	# Run this once the best (most in focus image) is selected in the image viewer
	# A .txt file with the filename of the best image will be saved in the stack folder
	p = Path(rw.flipbook.current_page[0].name)
	best_focus = p.name
	folder = p.parent
	with open((str(folder) + '/best_focus.txt'), 'w') as f:
		f.write(best_focus)
	print('Best focus')

def no_worm():
	# Run this if there is no worm, the worm is too small, or the worm is exploded everywhere
	# Also use this in cases such as two worms, worm is split between focus planes
	p = Path(rw.flipbook.current_page[0].name)
	folder = p.parent
	with open((str(folder) + '/no_worm.txt'), 'w') as f:
		f.write('no worm')
	print('No worm')

def find_next_stack():
	# Experiment directory (this will have to be hard coded to change experiments)
	# Iterate through all sub directories looking for best_focus.txt files.
	experiment_dir = Path('/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead')

	for worm in os.listdir(experiment_dir):
		worm_dir = experiment_dir / worm

		for stack in worm_dir.glob('* focus'):
			stack_dir = worm_dir / stack
			print('Found stack')

			# Check if best focus or no worm has been noted for this stack
			q = stack_dir / 'best_focus.txt'
			nw = stack_dir / 'no_worm.txt'
			if q.exists() or nw.exists():
				# If the stack is annotated, add it to the visited list
				visited.append(stack_dir)
				print('Stack annotated')
			else:
				# The stack hasn't been annotated, it is the next stack
				next_stack = stack_dir
				print('Found next stack')
				break
				# End the loop and return the stack that has been found
	return next_stack


def next_stack():
	# Loads the next stack into RisWidget for choosing
	# Get the file path to the next stack
	next_stack = find_next_stack()

	# Clear images currently in the flipbook
	rw.flipbook_pages.clear()

	# Load images from the next stack
	rw.add_image_files_to_flipbook(str(next_stack) + '/*.png')

# Initialize a list of stacks visited in this session
visited = []

# Define the push buttons to add to RisWidget
best_button = Qt.QPushButton('BEST')
best_button.pressed.connect(best)
best_button.show()
l.addWidget(best_button)

nw_button = Qt.QPushButton('NO WORM')
nw_button.pressed.connect(no_worm)
nw_button.show()
l.addWidget(nw_button)

next_button = Qt.QPushButton('NEXT')
next_button.pressed.connect(next_stack)
next_button.show()
l.addWidget(next_button)

file_path = '/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead/07/2019-05-30t0942 focus'
rw.add_image_files_to_flipbook(file_path + '/*.png')



# Last line
# rw.run
# rw.input

"""
from ris_widget import ris_widget                                       
rw = ris_widget.RisWidget()
from PyQt5 import Qt
b = Qt.QPushButton?
b = Qt.QPushButton('button')
def do_something():
    print('done')
b.pressed.connect(do_something)
b.show()
b.hide()
rw.flipbook.layout
rw.flipbook.layout()
l = rw.flipbook.layout()
l.add?
l.add
l.addWidget(b)
b.show()
hist

"""


