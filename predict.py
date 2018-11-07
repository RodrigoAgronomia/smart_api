
# coding: utf-8

# In[1]:


import numpy as np
import torch
import pickle
from torch.autograd import Variable
import glob
import cv2
from PIL import Image as PILImage
from matplotlib import pyplot as plt
import Model as Net
import os
import time
from argparse import ArgumentParser
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


# In[2]:


def colorir(cat):
    h, w = cat.shape[:2]
    msk = np.zeros((h,w,3), dtype = 'uint8')
    msk[cat == 1] = [0,0,128]
    msk[cat == 2] = [0,0,255]
    msk[cat == 3] = [0,128,0]
    msk[cat == 4] = [0,255,0]
    return(msk)


# In[3]:


def calc_bb(tw, th, iw, ih):
    n_v = int(np.ceil(tw/iw))
    n_h = int(np.ceil(th/ih))

    in_height = np.int(np.ceil(th/n_h))
    in_width = np.int(np.ceil(tw/n_v))
    
    mrgmx = 80 * (1 + np.round(in_height/80)) - in_height
    mrgmy = 80 * (1 + np.round(in_width/80)) - in_width

    ex_height = np.int(in_height + 2 * mrgmx)
    ex_width = np.int(in_width + 2 * mrgmy)

    ex_height = np.int(8 * np.ceil(ex_height/8))
    ex_width = np.int(8 * np.ceil(ex_width/8))

    in_height = ex_height - 2 * mrgmx
    in_width = ex_width - 2 * mrgmy


    coordsx = mrgmx * np.array([[0,-2],[2,0]])
    for i in range(n_v - 2):
        coordsx = np.insert(coordsx, 1, [-mrgmx, mrgmx], axis = 1)

    coordsy = mrgmy * np.array([[0,-2],[2,0]])
    for i in range(n_h - 2):
        coordsy = np.insert(coordsy, 1, [-mrgmy, mrgmy], axis = 1)

    xc = np.array([np.array(range(n_v)), np.array(range(n_v)) + 1])
    yc = np.array([np.array(range(n_h)), np.array(range(n_h)) + 1])
    xc = np.int0(coordsx + in_width * xc)
    yc = np.int0(coordsy + in_height * yc)

    xc[0,n_v - 1] = tw - in_width - 2 * mrgmx
    xc[1,n_v - 1] = tw
    yc[0,n_h - 1] = th - in_height - 2 * mrgmy
    yc[1,n_h - 1] = th


    xco = -coordsx
    xco[1] = xco[0] + in_width
    xco[1] = xco[0] + in_width

    yco = -coordsy
    yco[1] = yco[0] + in_height
    yco[1] = yco[0] + in_height

    yco = np.int0(yco)
    xco = np.int0(xco)
    
    if n_h == 1:
        yc = yco = np.array([[0],[th]])
    if n_v == 1:
        xc = xco = np.array([[0],[tw]])
        
    return([xc, yc, xco, yco])


# In[4]:


parser = ArgumentParser()
parser.add_argument('--model', default="ESPNet", help='Model name')
parser.add_argument('--data_dir', default='./data', help='Data directory')
parser.add_argument('--scaleIn', type=int, default=8, help='For ESPNet-C, scaleIn=8. For ESPNet, scaleIn=1')
parser.add_argument('--max_epochs', type=int, default=1501, help='Max. number of epochs')
parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size. 12 for ESPNet-C and 6 for ESPNet. '
                                                               'Change as per the GPU memory')
parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs.')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
parser.add_argument('--savedir', default='./results', help='directory to save the results')
parser.add_argument('--resume', type=bool, default=False, help='Use this flag to load last checkpoint for training')  #
parser.add_argument('--classes', type=int, default=5, help='No of classes in the dataset. 20 for cityscapes')
parser.add_argument('--cached_data_file', default='data.p', help='Cached file name')
parser.add_argument('--logFile', default='trainValLog.txt', help='File that stores the training and validation logs')
parser.add_argument('--onGPU', default=False, help='Run on CPU or GPU. If TRUE, then GPU.')
parser.add_argument('--decoder', type=bool, default=False,help='True if ESPNet. False for ESPNet-C') # False for encoder
parser.add_argument('--pretrained', default='../pretrained/encoder/espnet_p_2_q_8.pth', help='Pretrained ESPNet-C weights. '
                                                                          'Only used when training ESPNet')
parser.add_argument('--p', default=2, type=int, help='depth multiplier')
parser.add_argument('--q', default=8, type=int, help='depth multiplier')

args, _ = parser.parse_known_args()


p = args.p
q = args.q
classes = args.classes
modelA = Net.ESPNet_Encoder(classes, p, q)
model_weight_file = 'data/model_2501.pth'
modelA.load_state_dict(torch.load(model_weight_file,map_location=device)) #Sem CUDA
modelA = modelA.to(device) #Utilizado pro CUDA

# set to evaluation mode
modelA.eval()

data = pickle.load(open('data/data.p', "rb"))

up = torch.nn.Upsample(scale_factor=8, mode='bilinear')
up = up.to(device)
            
print('Modelo Ok')


# In[8]:

# Global mean and std values
rgb_mean = data['mean']
rgb_std = data['std']
n_classes = len(data['classWeights'])
new_ext = '.png'


input_height = 1080
input_width = 960
 
n_frames = 123
def get_plants(img_name, out_img_name):
    start_time = time.time()
    print(img_name)
    
    img = cv2.imread(img_name)
    imgo = img.copy()
    img = img.astype(np.float32)
    img -= rgb_mean
    img /= rgb_std
    
    total_width, total_height = img.shape[:2]
    xc, yc, xco, yco = calc_bb(total_width, total_height, input_height, input_width)

    time_taken = time.time() - start_time
    print('PreProc time: %.2f' % time_taken)
    
    preds_l = []
    for yi in range(yc.shape[1]):
        for xi in range(xc.shape[1]):
            im = img[xc[0,xi]:xc[1,xi],yc[0,yi]:yc[1,yi]]   
            im = im.reshape(np.insert(im.shape, 0, 1))
            im = np.moveaxis(im, 3, 1)
            img_tensor = torch.from_numpy(im)
            img_variable = Variable(img_tensor)
            img_variable = img_variable.to(device)
            img_out = modelA(img_variable)
            img_out = up(img_out)
            preds = img_out.cpu().data.numpy()
            preds = preds[:,:,xco[0,xi]:xco[1,xi],yco[0,yi]:yco[1,yi]] 
            preds_l.append(preds)
            
            
    time_taken = time.time() - start_time - time_taken
    print('Prediction time: %.2f' % time_taken)
    
    pred = np.stack(preds_l)
    pred = np.moveaxis(pred, 2, 4)
    ps = pred.shape
    pred = pred.reshape((yc.shape[1], xc.shape[1], ps[2], ps[3], n_classes))
    pred = np.moveaxis(pred, 0, 2)
    pred = pred.reshape(xc.shape[1] * ps[2], yc.shape[1] * ps[3] , n_classes)
    cat = np.argmax(pred, 2)
    
    pred = colorir(cat)

    # apply the overlay
    alpha = 0.5
    cv2.addWeighted(pred, alpha, imgo, 1 - alpha, 0, imgo)
    cv2.imwrite(out_img_name, imgo)
    
    time_taken = time.time() - start_time
    print('Total time: %.2f' % time_taken)

