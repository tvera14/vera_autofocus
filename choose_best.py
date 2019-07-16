# Program allows the user to view a flopbook of images and choose the best focus. Type best() and a file with the
# filename of the best focused image will be saved in the same folder as the images.

# Run this by starting ipython, then "run load_flipbook.py". Just running the file outside ipython will cause it to
# immediately
# Run from inside the same directory as this script

from ris_widget import ris_widget  
rw = ris_widget.RisWidget()                                                                                  
import freeimage  
import pathlib  

# Define a function that will grab the filename of the best focus image
def best():
	# Run this once the best (most in focus image) is selected in the image viewer
	# A .txt file with the filename of the best image will be saved in the stack folder
	p = pathlib.Path(rw.flipbook.current_page[0].name)
	best_focus = p.name
	folder = p.parent
	with open((str(folder) + '/best_focus.txt'), 'w') as f:
		f.write(best_focus)

def no_worm():
	# Run this if there is no worm, the worm is too small, or the worm is exploded everywhere
	# Also use this in cases such as two worms, worm is split between focus planes
	p = pathlib.Path(rw.flipbook.current_page[0].name)
	folder = p.parent
	with open((str(folder) + '/no_worm.txt'), 'w') as f:
		f.write('no worm')

# Enter the full path to the image stack
file_path = '/Volumes/purplearray/Pittman_Will/20190521_cyclo_dead/05/2019-05-27t0605 focus'
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


	