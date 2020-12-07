from skimage.transform import resize
import numpy as np
import os
import matplotlib.pyplot as plt

def load_and_split_image(path,size_x,size_y,channels=3):
    slices = []
    img = plt.imread(path)
    for x in range(0,img.shape[0],size_x):
        for y in range(0,img.shape[1],size_y):
            cropped = img[x:x+size_x,y:y+size_y]
            cropped = resize(cropped,(size_x,size_y,channels),mode="constant",preserve_range=True)
            slices.append(cropped)
    
    return slices
    
def load_images(path,size_x,size_y,base_type=".tif",mask_type=".png"):
    X = []
    Y = []
    ids = next(os.walk(path))[2]
    
    for i in range(len(ids)):
        img = plt.imread(path+ids[i])
        if "segmented" not in ids[i]:
            filename = ids[i].replace(".tif","")
            xSlices = load_and_split_image(path+filename+base_type,size_x,size_y,1)
            ySlices = load_and_split_image(path+filename+"_segmented"+mask_type,size_x,size_y,1)
            
            for i in range(len(xSlices)):
                X.append(xSlices[i])
                Y.append(ySlices[i])        
                
    return np.array(X),np.array(Y)


def load_full_images(path,base_type=".tif",mask_type=".png",chans=3):
    X = []
    Y = []
    ids = next(os.walk(path))[2]
    for i in range(len(ids)):
        img = plt.imread(path+ids[i])
        xSize = img.shape[0]
        ySize = img.shape[1]
        if "segmented" not in ids[i]:
            filename = ids[i].replace(".tif","")
            img = plt.imread(path+filename+base_type)
            mask = plt.imread(path+filename+"_segmented"+mask_type)
            img = resize(img,(xSize,ySize,chans),mode="constant",preserve_range=True)
            mask = resize(mask,(xSize,ySize,chans),mode="constant",preserve_range=True)
            X.append(img)
            Y.append(mask)
                
    return np.array(X),np.array(Y)
