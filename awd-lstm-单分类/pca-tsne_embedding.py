# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:34:13 2019

@author: night
"""

import numpy as np
from sklearn.decomposition import PCA #sklearn要求的输入为 Nx特征
from sklearn.manifold import TSNE
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from utils import mate_data
from PIL import Image


#plt.rcParams['xtick.direction']='in'
#plt.rcParams['ytick.direction']='in'
#plt.rcParams['ztick.direction']='in'
def plot_embedding_3d(X,y,c,title=None,camera=None):
#    #坐标缩放到[0,1]区间
#    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
#    X = (X - x_min) / (x_max - x_min)
    #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    label=['o','v','^','s','D','p','h']
#    label1=mate_data(to_torch=0)
#    label1=[i[0] for i in label1]
#    img = [Image.fromarray(i).convert('L') for i in label1]
    col_map=['b','g','r','c','m','y','k']#blue green red cyan magenta yellow black
    col_alpha=[0.8,0.6,0.6,0.8,0.8,0.9,0.6]
    
    fig = plt.figure(figsize=(14,8))#
#    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
#    plt.rcParams['figure.dpi'] = 800
    ax.grid(True)
#    ax.set(xlim=[-2,2],xticks=[-2,-1,0,1,2],ylim=[-2,2],yticks=[-2,-1,0,1,2],zlim=[-2,2],zticks=[-2,-1,0,1,2])
    for i in range(c):
        temp=[]
        for j in range(X.shape[0]):
            if i==y[j]:
                temp.append(X[j])
        temp=np.array(temp)
        ax.scatter(temp[:,0], temp[:,1], temp[:,2]
           ,color=col_map[i]
           ,marker=label[i]
           ,alpha=col_alpha[i]
           )
#    for i in range(X.shape[0]):
#        if y[i] == 7:
#            continue
#        ax.scatter(X[i,0], X[i,1], X[i,2]
#                   ,color=col_map[y[i]]
#                   ,marker=label[y[i]]
#                   ,alpha=col_alpha[y[i]]
#                   )
    ax.legend([' ',' ',' ',' ',' ',' ',' '])
#    if title is not None:
#        plt.title(title)
#    for ii in range(0,360,20):
#        for iii in range(0,360,20):
#            ax.view_init(elev=iii, azim=ii)
#            plt.savefig('C:/Users/night/Desktop/pca/movie%d' % ii + '_%d' % iii,dpi=500)
#    if camera is not None:
#        for n in range(camera.shape[1]):
#            ax.view_init(azim=camera[0,n], elev=camera[1,n])
#            plt.savefig('D:/OneDrive/EEG-python/文章/毕业论文/tsne/sub3movie%d.png' % n,dpi=500)
    plt.show()


#mat            = scipy.io.loadmat('D:/EEG/model/AWD-LSTM/lz/50-250_9_vis/15631020158609197_ACC=0.88_loss=0.49_epoch=152_vis.mat')
#mat            = scipy.io.loadmat('D:/EEG/model/AWD-LSTM/lz/50-250_9_part2/15629446062968898_ADAMmaxACC=0.84_loss=0.78_epoch=152_vis.mat')#Nx64
mat            = scipy.io.loadmat('D:/EEG/model/AWD-LSTM/Brother Qiang/50-250_9/15628726948027844_ADAMLr=0.005_acc=0.68_loss=1.13_epoch=150_vis.mat')#Nx64
#hunhemat       = scipy.io.loadmat('D:/EEG/model/AWD-LSTM/lz/50-250_9_vis/15631020158609197_ACC=0.88_loss=0.49_epoch=152_hunhe_vis.mat')#貌似没什么意义
#vec            = mat['vector']
val_vec        = mat['val_vector']
emb_val_vec_tar= mat['real_tar'].reshape(-1)
#emb_val_vec_tar= mat['val_targets'].reshape(-1)
#emb_hunhe_vec  = hunhemat['vector']


'''PCA'''
#pca            = PCA(n_components=3, copy=True, whiten=False)
#val_pca        = PCA(n_components=3, copy=True, whiten=False)
#hunhepca       = PCA(n_components=3, copy=True, whiten=False)
#emb_hunhe_vec  = hunhepca.fit_transform(emb_hunhe_vec)
#val_emb_vec    = val_pca.fit_transform(val_vec)


#label1=mate_data(to_torch=0)
#label1=[i[0] for i in label1]
#img = [Image.fromarray(i).convert('L') for i in label1]
#label=mate_data(to_torch=0)

#'''t-SNE'''
val_tsne=TSNE(n_components=3, init='pca', random_state=0)
val_emb_vec_tsne    = val_tsne.fit_transform(val_vec)





#'''plot it'''
camera=np.array([[-60,40]]).T
##camera=np.array([[-42,36],[-80,40],[-122,60]]).T
#plot_embedding_3d(val_emb_vec[0:400],emb_val_vec_tar[0:400],title='PCA embedding',camera=camera)
plot_embedding_3d(val_emb_vec_tsne[0:500],emb_val_vec_tar[0:500],8,title='t-SNE embedding',camera=camera)
#















