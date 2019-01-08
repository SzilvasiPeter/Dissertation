from PIL import Image
import os, sys

path = "/home/szilvasi/Documents/Dissertation/DissertationSource/Codes/CCR_usingKeras/test3/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((28,28), Image.ANTIALIAS)
            imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

resize()