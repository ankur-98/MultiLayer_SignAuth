#Packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import tqdm #percentage bar for tasks
import random
from PIL import Image

def test_view_img(file):
    ims=np.load(file);
    a,b,c,d = ims
    i = Image.fromarray(a[100])
    i.show()
#    plt.imshow(a[100])
    
def getImg(path,img):
    IMG_SIZE = 100
    img = cv2.imread(path,cv2.IMREAD_ANYCOLOR)
    img = pp(img)
    img = cv2.resize(img, (IMG_SIZE*2, IMG_SIZE), cv2.INTER_LINEAR)
    return img

def pp(img):
    # img = cv2.resize(img, (IMG_SIZE, None), cv2.INTER_LINEAR)
    return img

def create_data_set():    
    data_set = []
    for img in tqdm.tqdm(os.listdir(TRAIN_DIR_f)):
        path = os.path.join(TRAIN_DIR_f,img)
        img0 = getImg(path,img)
        data_set.append([np.array(img0),[0]])
        
    for img in tqdm.tqdm(os.listdir(TRAIN_DIR_g)):
        path = os.path.join(TRAIN_DIR_g,img)
        img1 = getImg(path,img)
        data_set.append([np.array(img1),[1]])
        
    random.shuffle(data_set)
#    np.save('data_set.npy',data_set)
    return data_set

def split_data_set(data,m):
    #data_set = [[array,[]],[array,[]],[array,[]],...,[array,[]]]    mx2
    #x = mx1
    #y = mx1
    x,y = [data[i][0] for i in range(m)],[data[i][1] for i in range(m)]
    return x,y

def get_train_test_data(x,y,m):
    size_train = int(0.8*m)
    size_test = m - size_train
    train_x,test_x,train_y,test_y = np.array([x[i] for i in range(0,size_train)]),np.array([x[i] for i in range(size_train,m)]),np.array([y[i] for i in range(0,size_train)]),np.array([y[i] for i in range(size_train,m)])  
    return train_x,test_x,train_y,test_y

TRAIN_DIR_f = '/media/ankur98/0F5E1B3E0F5E1B3E/Projects/Signature Authenticator/Single layer/TrainingSet/Offline Forgeries'
TRAIN_DIR_g = '/media/ankur98/0F5E1B3E0F5E1B3E/Projects/Signature Authenticator/Single layer/TrainingSet/Offline Genuine'
data_set = create_data_set()
m = len(data_set)
data_set_x,data_set_y = split_data_set(data_set,m)

train_x_org,test_x_org,train_y,test_y = get_train_test_data(data_set_x,data_set_y,m)
train_y = train_y.T
test_y = test_y.T
np.save('data_set.npy',[train_x_org,test_x_org,train_y,test_y])

# test_view_img('data_set.npy')
