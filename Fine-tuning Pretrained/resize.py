from PIL import Image
import os, sys

path = "/home/asblab/Downloads/A3/DataSet/animalNew/"
pathNew = "/home/asblab/Downloads/A3/DataSet/animalNewNew/"
dirs = os.listdir( path )

for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
	    print item
            imResize = im.resize((128,128), Image.ANTIALIAS)
            imResize.save(pathNew + item, 'JPEG', quality=100)


