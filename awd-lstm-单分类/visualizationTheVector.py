# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:17:16 2019

@author: night
"""
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image
import matplotlib as mpl
#import os
import scipy.io
#import imageio
from PIL import Image
import sklearn.preprocessing as skp

from sklearn.metrics import confusion_matrix


mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

#取数据
#raw=scipy.io.loadmat('D:/OneDrive/EEG-python/awd-lstm-单分类/15629427777810395_visualization_LZ_VCEEGall_50-250ms_all_torch_9_part1.mat')
raw=scipy.io.loadmat('D:/EEG/model/AWD-LSTM/lz/50-250_9_vis/15631020158609197_ACC=0.88_loss=0.49_epoch=152_vis.mat')

#test='train'
test='val'

if test == 'train':
    output=raw['output'].astype('int32')   #预测值 1 x ANY
    targets=raw['targets'].astype('int32')   #.....真值   1 x ANY
    vector=raw['vector'].astype('float32')   #特征向量
else:
    output=raw['val_output'].astype('int32')
    targets=raw['val_targets'].astype('int32')
    vector=raw['val_vector'].astype('float32') 
del(raw)

#lz 求特征向量叠加平均和混淆矩阵
chaos=np.zeros([len(np.unique(targets)),len(np.unique(targets))])
vec=[[],[],[],[],[],[],[],[]]


for i in range(targets.size):
    if targets[0,i]==output[0,i]:
#        if chaos[targets[0,i],targets[0,i]]<
        chaos[targets[0,i],targets[0,i]]+=1
        vec[targets[0,i]].append(vector[i,:])
        
    else:
        chaos[targets[0,i],output[0,i]]+=1
        
vec=[np.array(j) for j in vec]


vec_mean=[q.mean(0).reshape(64,1) for q in vec]
#for i,j in enumerate(vec_mean):
#    plt.bar(np.arange(0,64),j.reshape(64,),1)
#    plt.axis('off')
#    plt.xticks([])
#    plt.yticks([])
#    plt.savefig('C:/Users/night/Desktop/{}_out_bar{}.png'.format(test,i))
#    plt.close()

y_true=[]
y_pred=[]
duoshaoge=[1,1,1,1,1,1,1,1]
for i in range(targets.size):
    for j in range(8):
        if targets[0,i] == j and duoshaoge[j] < np.random.randint(220,high=250):
            duoshaoge[j] +=1
            y_true.append(targets[0,i])
            y_pred.append(output[0,i])

#lz 画特征向量

#vec_mean=[np.tile(skp.minmax_scale(q.mean(0).reshape(64,1))*255,(1,20)) for q in vec]
#for i,j in enumerate(vec_mean):
#    im = Image.fromarray(vec_mean[i]).convert('L')
#    im=im.resize((20*10,64*10),Image.ANTIALIAS)
#    im.save('C:/Users/night/Desktop/out{}.png'.format(i))
# 
##lz 画混淆矩阵
#chaos_percent=chaos/(chaos.sum(1).reshape(4,1))
#chaos_percent=np.tile(chaos_percent.reshape((4,4,1)),(1,1,3))
#chaos_percent[:,:,0]=(1-chaos_percent[:,:,0])*160 + 80
#chaos_percent[:,:,1]=(1-chaos_percent[:,:,1])*60 + 180
#chaos_percent[:,:,2]=255
#chaos_percent=chaos_percent.astype('uint8')
#im = Image.fromarray(chaos_percent).convert('RGB')
#im=im.resize((4*80,4*80))
#im.save('C:/Users/night/Desktop/混淆矩阵.png')




#sklearn求混淆矩阵
def plot_confusion_matrix(y_true, y_pred, 
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):#classes,
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#    classes = classes[unique_labels(y_true, y_pred)]
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
    cmm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.text(x=-2,y=2.5,s='True label',fontsize=15,rotation=90)
    ax.text(x=1.8,y=9,s='Predicted label',fontsize=15)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries

           title=title)
           
#           xlabel='Predicted label')#       ylabel='True label',    xticklabels=classes, yticklabels=classes,

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    fmtt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="bottom",
                    color="white" if cm[i, j] > thresh else "black") # 字体颜色
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cmm[i, j], fmtt),
                    ha="center", va="top",
                    color="white" if cm[i, j] > thresh else "black") # 字体颜色
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plot_confusion_matrix(targets.reshape(targets.shape[1],1), output.reshape(output.shape[1],1), classes=['centre','around','cross','rightslant','corner','WASD','leftslant','space'],
#                      title='Confusion matrix ')
plot_confusion_matrix(y_true, y_pred,
                      title='混淆矩阵 ')# classes=['centre','around','cross','rightslant','corner','WASD','leftslant','space'],

plt.savefig('C:/Users/night/Desktop/混淆矩阵.png',dpi=500)


    
    












