# Run this by starting ipython, then "run load_flipbook.py". Just running the file outside ipython will cause it to
# immediately

from ris_widget import ris_widget                                       
rw = ris_widget.RisWidget()                                             

import freeimage    
import glob
stack_images = glob.glob('/Users/zplab/Desktop/VeraPythonScripts/vera_autofocus/microscope_images/train/0/*.png')  

for image in stack_images:
	i = freeimage.read(image)
	rw.flipbook_pages.append([i])

