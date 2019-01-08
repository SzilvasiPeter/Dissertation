import numpy as np
from PIL import Image
from scipy import misc
import os
from keras.models import load_model
import csv
import pandas as pd

import cv2

# Training set
dirs = '/home/szilvasi/Documents/Dissertation/DissertationSource/Codes/CCR_usingKeras/train3'
filelists = os.listdir(dirs)
dic = {}
i = 0
for file in filelists:
    dic[str(i)] = file
    i = i+1
    
csvfile = open('list.csv', 'w', newline='')
writer = csv.writer(csvfile)
m = 'filename'
m.encode('utf-8')
n = 'label'
n.encode('utf-8')


    
model = load_model('model.h5')

# Test set
dirs = '/home/szilvasi/Documents/Dissertation/DissertationSource/Codes/CCR_usingKeras/test3/resized'
filelists = os.listdir(dirs)
files = []
results = []
for file in filelists:
    files.append(file)
    path = dirs + '/' + file
    # im = Image.open(path)
    im = cv2.imread(path)
    x = np.array(im)
    #print(x.shape)
    x = misc.imresize(x,[28,28]).astype('float32')/255
    x = x.reshape((28,28,3))
    x = np.expand_dims(x,axis=0)
    x = model.predict(x)
    x = 1-x
    x = np.argsort(x)
    y = ''
    for j in x[0][:5]:
        i = []
        y = y + dic[str(j)]
    results.append(y)


dataframe = pd.DataFrame({'filename':files, 'label':results})
dataframe.to_csv('list.csv',index=False,sep=',',encoding='utf-8')