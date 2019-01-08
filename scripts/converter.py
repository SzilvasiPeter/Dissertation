import numpy as np
from PIL import Image                                                            
import numpy                                                                     
import matplotlib.pyplot as plt                                                  
import glob
import os
import cv2

def img2array(path):
    img = Image.open(path).convert('F')
    return numpy.array(img).ravel()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

root_folder = '/home/szilvasi/Documents/Dissertation/DissertationSource/Codes/train1'
folders = [os.path.join(root_folder, x) for x in ('00', '01', '02', '03', '04')]
all_images = [img for folder in folders for img in load_images_from_folder(folder)]
print(len(all_images))
print(all_images[0])

# train_labels = []
# files = glob.glob ("/home/szilvasi/Documents/Dissertation/DissertationSource/Codes/resized-imgs/*.jpg")
# for myFile in files:
#     train_labels.append([1., 0.])

# train_labels = np.array(train_labels,dtype='float64')

# imageFolderPath = '/home/szilvasi/Documents/Dissertation/DissertationSource/Codes/resized-imgs'
# imagePath = glob.glob(imageFolderPath+'/*.jpg') 

# im_array = numpy.array([img2array(path) for path in imagePath])

# im_array = np.array(im_array, dtype='float32')

# print(im_array.shape)
# np.save('train', im_array)
# np.save('train_labels',train_labels)
# print(train_labels.shape)