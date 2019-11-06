# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:21:41 2019
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
#  a>2 and a<5  X ; (a>2) and (a<5)  X  ;  a>2 & a<5  X   ;  (a>2) & (a<5)  √
#可以看出python里逻辑运算符 or and 啥的与 布尔运算符还是有区别的    而这两个用的时候都要将左右化为整体，不能产生优先级混乱
@author: night
"""
import numpy as np
import os
import scipy.io

import model
import torch
import torch.nn as nn

import pickle
from LRP_linear import *

from mne.viz import plot_topomap,tight_layout
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm


class Heatmap:
    def __init__(self):
        self.pos =   np.array([[0.09456747, 0.45      ],
                               [0.19883673, 0.6       ],
                               [0.29158346, 0.68      ],
                               [0.37592077, 0.72      ],
                               [0.4617831 , 0.73      ],
                               [0.54879398, 0.72      ],
                               [0.63632404, 0.68      ],
                               [0.72916309, 0.6       ],
                               [0.82952896, 0.45      ],
                               [0.182265  , 0.27      ],
                               [0.282265  , 0.38      ],
                               [0.372265  , 0.42      ],
                               [0.46226497, 0.43      ],
                               [0.552265  , 0.42      ],
                               [0.642265  , 0.38      ],
                               [0.742265  , 0.27      ],
                               [0.31271   , 0.07      ],
                               [0.28271   , 0.2       ],
                               [0.46271014, 0.2       ],
                               [0.64271   , 0.2       ],
                               [0.61271   , 0.07      ]])                      #channel position
    
        self.cmap =   col.LinearSegmentedColormap.from_list('heatmap_lrp',['#000000',  '#ff0033','#ffff33','#ffffff'])
    def topo_map(self,data,ax,label=''):
        #cmap='Reds'
#        ax.set(title="Mean\ncorrelation"+label) #带图像的标签
        ax.set(title="Mean\ncorrelation")
        return plot_topomap(data, self.pos,axes=ax,cmap='jet',vmin=np.min(data), vmax=np.max(data), show=0,point_of_view='back') #sensors='',
    # matplotlib中的colormap的风格 加 _r 就是翻转的意思, 所有colormaps 都有翻转
    def sample_map(self,data,ax,title='Heatmap',ticks=1,colorbar=1,label=''):
#        time_plot = 0.180  # For highlighting a specific time.
        max_corr = data.max()
        min_corr = data.min()
        X,Y      = np.arange(data.shape[0]),np.arange(data.shape[1])
        X,Y      = np.meshgrid(X,Y)
        
        def data_f(x,y):
            return data[x,y]
        
        im       =ax.contourf(X, Y, data_f(X,Y), cmap='jet' ,vmin=min_corr, vmax=max_corr
                              ,levels = MaxNLocator(nbins=80).tick_values(min_corr, max_corr)
                              ,alpha=1.)#
#        norm = colors.Normalize(vmin=0.4, vmax=0.75)
#        im.set_norm(norm)
        if ticks:
#            ax.set(xlabel='time(ms)', ylabel='Channel', title=title+label,#带标签
            ax.set(xlabel='time(ms)', ylabel='Channel', title=title,
                   xlim=[0, data.shape[0] - 1], ylim=[data.shape[1] - 1, 0],
                   xticks=np.arange(0, data.shape[0], 30),yticks=np.arange(0, data.shape[1], 5)
                   )
            plt.setp(ax.get_xticklabels(), rotation=45)
#            plt.xticks(fontsize=8)
        else:
            ax.set(
                   xlim=[0, data.shape[0] - 1], ylim=[data.shape[1] - 1, 0],
                   xticks=[],yticks=[]
                  )
        if colorbar:
            plt.colorbar(im,ax=ax, ticks=[0,0.5,1])#
        plt.tight_layout()
#        return im

'''
LSTM use numpy and LRP
'''
class LSTM_for_LRP:
    
    def __init__(self, path):
        
        
        # model weights
        f_model   = open(path,'rb')
        my_model     = pickle.load(f_model)
        f_model.close()
        
        # LSTM  encoder
        self.Wxh_Left  = my_model["rnns.0.weight_ih_l0"]  # shape 4d*e
        self.bxh_Left  = my_model["rnns.0.bias_ih_l0"]  # shape 4d 
        self.Whh_Left  = my_model["rnns.0.weight_hh_l0"]  # shape 4d*d
        self.bhh_Left  = my_model["rnns.0.bias_hh_l0"]  # shape 4d  

        # linear output layer
        self.Why_Left  = my_model["decoder.weight"]  # shape C*d

    

    def set_input(self, eeg, delete_pos=None):
        """
        Build the numerical input x/x_rev from the word indices w (+ initialize hidden layers h, c).
        Optionally delete words at positions delete_pos.
        """
        T      = len(eeg)                         #  input eeg sequence length 
        d      = int(self.Wxh_Left.shape[0]/4)     # hidden layer dimension

        x=eeg
#        if delete_pos is not None:
#            x[delete_pos, :] = np.zeros((len(delete_pos), eeg))
        
        self.eeg            = eeg
        self.x              = x

        
        self.h_Left         = np.zeros((T+1, d),dtype='float32')
        self.c_Left         = np.zeros((T+1, d),dtype='float32')

     
   
    def forward(self):
        """
        Standard forward pass.
        Compute the hidden layer values (assuming input x/x_rev was previously set)
        """
        T      = len(self.eeg)
#        print(self.eeg.shape)
#        print(len(self.eeg))                         
        d      = int(self.Wxh_Left.shape[0]/4) 
        # gate indices (assuming the gate ordering in the LSTM weights is i,g,f,o): 
        #!!But in pytorch it is (i f g o) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        idx    = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # 
        idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) 
          
        # initialize
        self.gates_xh_Left  = np.zeros((T, 4*d),dtype='float32')  
        self.gates_hh_Left  = np.zeros((T, 4*d),dtype='float32') 
        self.gates_pre_Left = np.zeros((T, 4*d),dtype='float32')  # gates pre-activation
        self.gates_Left     = np.zeros((T, 4*d),dtype='float32')  # gates activation
        

             
        for t in range(T): 
            self.gates_xh_Left[t]     = np.dot(self.Wxh_Left, self.x[t])
            self.gates_hh_Left[t]     = np.dot(self.Whh_Left, self.h_Left[t-1]) 
            self.gates_pre_Left[t]    = self.gates_xh_Left[t] + self.gates_hh_Left[t] + self.bxh_Left + self.bhh_Left
            self.gates_Left[t,idx]    = 1.0/(1.0 + np.exp(- self.gates_pre_Left[t,idx]))
            self.gates_Left[t,idx_g]  = np.tanh(self.gates_pre_Left[t,idx_g]) 
            self.c_Left[t]            = self.gates_Left[t,idx_f]*self.c_Left[t-1] + self.gates_Left[t,idx_i]*self.gates_Left[t,idx_g]
            self.h_Left[t]            = self.gates_Left[t,idx_o]*np.tanh(self.c_Left[t])
            

            
        self.y_Left  = np.dot(self.Why_Left,  self.h_Left[T-1])

        self.s       = self.y_Left
        
        return self.s.copy() # prediction scores
     
              
    def backward(self, eeg, sensitivity_class):
        """
        Standard gradient backpropagation pass.
        Compute the hidden layer gradients by backpropagating a gradient of 1.0 for the class sensitivity_class
        """
        # forward pass
        self.set_input(eeg)
        self.forward() 
        
        T      = len(self.eeg)
        d      = int(self.Wxh_Left.shape[0]/4)
        C      = self.Why_Left.shape[0]   # number of classes
        #!!But in pytorch it is (i f g o) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        idx    = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately
        
        # initialize
        self.dx               = np.zeros(self.x.shape,dtype='float32')

        
        self.dh_Left          = np.zeros((T+1, d),dtype='float32')
        self.dc_Left          = np.zeros((T+1, d),dtype='float32')
        self.dgates_pre_Left  = np.zeros((T, 4*d),dtype='float32')  # gates pre-activation
        self.dgates_Left      = np.zeros((T, 4*d),dtype='float32')  # gates activation
        

               
        ds                    = np.zeros((C),dtype='float32')
        ds[sensitivity_class] = 1.0
        dy_Left               = ds.copy()

        
        self.dh_Left[T-1]     = np.dot(self.Why_Left.T,  dy_Left)

        
        for t in reversed(range(T)): 
            self.dgates_Left[t,idx_o]    = self.dh_Left[t] * np.tanh(self.c_Left[t])  # do[t]
            self.dc_Left[t]             += self.dh_Left[t] * self.gates_Left[t,idx_o] * (1.-(np.tanh(self.c_Left[t]))**2) # dc[t]
            self.dgates_Left[t,idx_f]    = self.dc_Left[t] * self.c_Left[t-1]         # df[t]
            self.dc_Left[t-1]            = self.dc_Left[t] * self.gates_Left[t,idx_f] # dc[t-1]
            self.dgates_Left[t,idx_i]    = self.dc_Left[t] * self.gates_Left[t,idx_g] # di[t]
            self.dgates_Left[t,idx_g]    = self.dc_Left[t] * self.gates_Left[t,idx_i] # dg[t]
            self.dgates_pre_Left[t,idx]  = self.dgates_Left[t,idx] * self.gates_Left[t,idx] * (1.0 - self.gates_Left[t,idx]) # d ifo pre[t]
            self.dgates_pre_Left[t,idx_g]= self.dgates_Left[t,idx_g] *  (1.-(self.gates_Left[t,idx_g])**2) # d g pre[t]
            self.dh_Left[t-1]            = np.dot(self.Whh_Left.T, self.dgates_pre_Left[t])
            self.dx[t]                   = np.dot(self.Wxh_Left.T, self.dgates_pre_Left[t])
            

                    
        return self.dx.copy()  
    
                   
    def lrp(self, eeg, LRP_class, eps=0.001, bias_factor=0.0):
        """
        Layer-wise Relevance Propagation (LRP) backward pass.
        Compute the hidden layer relevances by performing LRP for the target class LRP_class
        (according to the papers:
            - https://doi.org/10.1371/journal.pone.0130140
            - https://doi.org/10.18653/v1/W17-5221 )
        """
        # forward pass
        self.set_input(eeg)
        self.forward() 
        
        T      = len(self.eeg)
        d      = int(self.Wxh_Left.shape[0]/4)
        e      = eeg.shape[1]
        C      = self.Why_Left.shape[0]  # number of classes
        idx    = np.hstack((np.arange(0,2*d), np.arange(3*d,4*d))).astype(int) # indices of gates i,f,o together
        #!!But in pytorch it is (i f g o) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        idx_i, idx_f, idx_g, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately
        
        # initialize
        Rx       = np.zeros(self.x.shape,dtype='float32')
#        Rx_rev   = np.zeros(self.x.shape)
        
        Rh_Left  = np.zeros((T+1, d),dtype='float32')
        Rc_Left  = np.zeros((T+1, d),dtype='float32')
        Rg_Left  = np.zeros((T,   d),dtype='float32') # gate g only

        
        Rout_mask            = np.zeros((C),dtype='float32')
        Rout_mask[LRP_class] = 1.0  
        
        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh_Left[T-1]  = lrp_linear(self.h_Left[T-1],  self.Why_Left.T , np.zeros((C)), self.s, self.s*Rout_mask, 2*d, eps, bias_factor, debug=False)
     
        for t in reversed(range(T)):
            Rc_Left[t]   += Rh_Left[t]
            Rc_Left[t-1]  = lrp_linear(self.gates_Left[t,idx_f]*self.c_Left[t-1],         np.identity(d), np.zeros((d)), self.c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=False)
            Rg_Left[t]    = lrp_linear(self.gates_Left[t,idx_i]*self.gates_Left[t,idx_g], np.identity(d), np.zeros((d)), self.c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=False)
            Rx[t]         = lrp_linear(self.x[t],        self.Wxh_Left[idx_g].T, self.bxh_Left[idx_g]+self.bhh_Left[idx_g], self.gates_pre_Left[t,idx_g], Rg_Left[t], d+e, eps, bias_factor, debug=False)
            Rh_Left[t-1]  = lrp_linear(self.h_Left[t-1], self.Whh_Left[idx_g].T, self.bxh_Left[idx_g]+self.bhh_Left[idx_g], self.gates_pre_Left[t,idx_g], Rg_Left[t], d+e, eps, bias_factor, debug=False)
            
               
        return  Rx, Rh_Left[-1].sum()+Rc_Left[-1].sum()


class utils:
    def get_test_seq(self,seq_idx,x,y):                                                 #取某一个序列
        
        
        test_seq       =   x[:,seq_idx,:]
        test_lab       =   y[seq_idx]
        
        
        return  test_seq,test_lab
        
    def get_test_class(self,data_loc):                                                  #取测试集数据
                
        mat            = scipy.io.loadmat(data_loc)
        valx           = mat['valx']
        valx           = np.rollaxis(valx,1)
        valy           = mat['valy']
                
        if valy.shape[0]<2:
            valy       = valy.reshape((-1,1))
            
        return  valx,valy                                                          #x:[数据长度：trial数：导联]  y：[trial数，1]
    
    def load_model_and_test(self,model_loc,x,y):                                        #测试模型的分类正确率
    
        x               = torch.from_numpy(x)
        y               = torch.from_numpy(y.reshape(len(y))).long().view(-1)      #y：[trial数，1]→[trial数，]
    
        global ceshi_model, optimizer
        
        with open(model_loc, 'rb') as f:                                           #load模型
            ceshi_model, optimizer = torch.load(f)
            ceshi_model.cpu()
            
        ceshi_model.eval()
        output          = ceshi_model(x,None)
        pred_y          = torch.max(output, 1)[1].data
        accuracy        = (pred_y == y).float().sum() / len(y)
        
        only_right={'label':[],'idx':[]}
        
        for i in range(y.shape[0]):                                                #y.size 是个属性  而shape的具体值是数
            
            if pred_y[i]==y[i]:
                only_right['label'].append(y[i].item())                            # y.data:还是tensor  y.item()：是纯数据
                only_right['idx'].append(i)
    
            
    #    print(ceshi_model)  
        print('|   val_acc={:5.2f}          |'.format(accuracy),'\n')
        
        return  only_right
    
    '''定义位置#############################################################################################'''
    def location(self,time,bias):
        if   time   == '50-200':
            if   bias == 'acc':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/50-200/15587959718969903_ADAMmaxACC=0.89_loss=0.36_epoch=50_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/50-200/15587959718969903_ADAMmaxACC=0.89_loss=0.36_epoch=50_para'
            elif bias == 'last':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/50-200/15587959718969903_ADAMLr=0.005_acc=0.88_loss=0.53_epoch=150_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/50-200/15587959718969903_ADAMLr=0.005_acc=0.88_loss=0.53_epoch=150_para'
            elif bias == 'loss':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/50-200/15587959718969903_ADAMminloss=0.34_acc=0.89_epoch=40_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/50-200/15587959718969903_ADAMminloss=0.34_acc=0.89_epoch=40_para'
            data_loc                = 'D:/EEG/mat_data/LZ_VCEEGall_50-200ms_all_torch_5.mat'
        elif time   == '100-250':
            if   bias == 'acc':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/100-250/15593976446819801_ADAMmaxACC=0.89_loss=0.66_epoch=110_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/100-250/15593976446819801_ADAMmaxACC=0.89_loss=0.66_epoch=110_para'
            elif bias == 'last':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/100-250/15593976446819801_ADAMLr=0.001_acc=0.89_loss=1.03_epoch=200_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/100-250/15593976446819801_ADAMLr=0.001_acc=0.89_loss=1.03_epoch=200_para'
            elif bias == 'loss':
                pass
            data_loc                = 'D:/EEG/mat_data/LZ_VCEEGall_100-250ms_all_torch_5.mat'
        elif time   == '100-300':
            if   bias == 'acc':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/100-300/1559405594709866_ADAMmaxACC=0.89_loss=0.86_epoch=152_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/100-300/1559405594709866_ADAMmaxACC=0.89_loss=0.86_epoch=152_para'
            elif bias == 'last':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/100-300/1559405594709866_ADAMLr=0.001_acc=0.88_loss=1.03_epoch=200_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/100-300/1559405594709866_ADAMLr=0.001_acc=0.88_loss=1.03_epoch=200_para'
            elif bias == 'loss':
                pass
            data_loc                = 'D:/EEG/mat_data/LZ_VCEEGall_100-300ms_all_torch_5.mat'
        elif time   == '100-200':
            if   bias == 'acc':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/100-200/1559567826064137_ADAMmaxACC=0.87_loss=0.39_epoch=33_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/100-200/1559567826064137_ADAMmaxACC=0.87_loss=0.39_epoch=33_para'
            elif bias == 'last':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/100-200/1559567826064137_ADAMLr=0.001_acc=0.87_loss=1.19_epoch=200_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/100-200/1559567826064137_ADAMLr=0.001_acc=0.87_loss=1.19_epoch=200_para'
            elif bias == 'loss':
                pass
            data_loc                = 'D:/EEG/mat_data/LZ_VCEEGall_100-200ms_all_torch_5.mat'
        elif time   == '0-300':
            if   bias == 'acc':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/0-300/15595730622401888_ADAMmaxACC=0.90_loss=0.51_epoch=102_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/0-300/15595730622401888_ADAMmaxACC=0.90_loss=0.51_epoch=102_para'
            elif bias == 'last':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/0-300/15595730622401888_ADAMLr=0.001_acc=0.88_loss=1.05_epoch=200_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/0-300/15595730622401888_ADAMLr=0.001_acc=0.88_loss=1.05_epoch=200_para'
            elif bias == 'loss':
                pass
            data_loc                = 'D:/EEG/mat_data/LZ_VCEEGall_0-300ms_all_torch_5.mat'
        elif time   == '0-250':
            if   bias == 'acc':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/0-250/15595755297490413_ADAMmaxACC=0.91_loss=0.42_epoch=101_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/0-250/15595755297490413_ADAMmaxACC=0.91_loss=0.42_epoch=101_para'
            elif bias == 'last':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/0-250/15595755297490413_ADAMLr=0.001_acc=0.89_loss=1.11_epoch=200_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/0-250/15595755297490413_ADAMLr=0.001_acc=0.89_loss=1.11_epoch=200_para'
            elif bias == 'loss':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/0-250/15595755297490413_ADAMminloss=0.33_acc=0.89_epoch=35_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/0-250/15595755297490413_ADAMminloss=0.33_acc=0.89_epoch=35_para'
            data_loc                = 'D:/EEG/mat_data/LZ_VCEEGall_0-250ms_all_torch_5.mat'
        elif time   == '50-250':
            if   bias == 'acc':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/50-250/15595882061793437_ADAMmaxACC=0.90_loss=0.39_epoch=91_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/50-250/15595882061793437_ADAMmaxACC=0.90_loss=0.39_epoch=91_para'
            elif bias == 'last':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/50-250/15595882061793437_ADAMLr=0.001_acc=0.89_loss=0.87_epoch=200_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/50-250/15595882061793437_ADAMLr=0.001_acc=0.89_loss=0.87_epoch=200_para'
            elif bias == 'loss':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/50-250/15595882061793437_ADAMminloss=0.33_acc=0.89_epoch=39_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/50-250/15595882061793437_ADAMminloss=0.33_acc=0.89_epoch=39_para'
            data_loc                = 'D:/EEG/mat_data/LZ_VCEEGall_50-250ms_all_torch_5.mat'
        elif time   == '50-250_8':
            if   bias == 'acc':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/50-250_8/1559848709058013_ADAMmaxACC=0.88_loss=0.51_epoch=105_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/50-250_8/1559848709058013_ADAMmaxACC=0.88_loss=0.51_epoch=105_para'
            elif bias == 'last':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/50-250_8/1559848709058013_lastModel_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/50-250_8/1559848709058013_lastModel_para'
            elif bias == 'loss':
                model_loc           = 'D:/EEG/model/AWD-LSTM/lz/50-250_8/1559848709058013_ADAMminloss=0.43_acc=0.86_epoch=78_model'
                para_loc            = 'D:/EEG/model/AWD-LSTM/lz/50-250_8/1559848709058013_ADAMminloss=0.43_acc=0.86_epoch=78_para'
            data_loc                = 'D:/EEG/mat_data/LZ_VCEEGall_50-250ms_all_torch_8.mat'
        model_loc='D:/EEG/model/AWD-LSTM/lz/50-250_9_part1/15629427777810395_ADAMmaxACC=0.82_loss=0.90_epoch=154_model'
        para_loc='D:/EEG/model/AWD-LSTM/lz/50-250_9_part1/15629427777810395_ADAMmaxACC=0.82_loss=0.90_epoch=154_para'
        data_loc='D:/EEG/mat_data/LZ_VCEEGall_50-250ms_all_torch_9_part1.mat'
        return model_loc,para_loc,data_loc


if __name__ == '__main__':
    '''获得验证集数据'''
#    util=utils()
#    model_loc,para_loc,data_loc = util.location('50-250_8','acc')
#    #     50-200  50-250  0-250  0-300  100-200  100-250  100-300
#    print(para_loc)
#    
#    x,y                         = util.get_test_class(data_loc)                                 #x(151,2865,21)  y(2865,1)
#    
#    '''测试pytorch模型是否对，并返回仅正确分类的label和索引，only_right{'idx'索引，'label'标签}'''
#    only_right                  = util.load_model_and_test(model_loc,x,y)
#    
##    
##    '''LRP实例'''
#    eps                         = 0.0001
#    bias_factor                 = 0.0
###    
#    net                         = LSTM_for_LRP(para_loc)
    heatmap                     = Heatmap()
    #Rx_sample,R_rest            = net.lrp(x[:,0,:],int(y[0,0]),eps,bias_factor)            #每个点相关性
    #R_channel                   = np.sum(Rx_sample , axis=0)                               #每个导联相关性
    #R_sample                    = np.sum(Rx_sample , axis=1)                               #每个时间步相关性
    ''''''
    
    test_label=0
#    t_l       =[' ■',' O',' +',' \\',' X',' ..''',' /'][test_label]
    time_dim  =x.shape[0]-6
    
    R_channel                   = np.zeros(x.shape[2],dtype='float32')
    #R_time                      = np.zeros(x.shape[0])
    #Rx_sample_mean              = np.zeros(x[:,0,:].shape)
    R_time                      = np.zeros(time_dim,dtype='float32')
    Rx_sample_mean              = np.zeros((time_dim,x[:,0,:].shape[1]),dtype='float32')
    
    wtf=[]   #存储有很大值或很小值的那种情况  用来实验
    idxxx=[]  #存储有wtf 的索引
#    for i in wtf:   #用来画存储的这些异常情况
#        fig= plt.subplots()
#        ax=plt.subplot(1,1,1)
#        heatmap.sample_map(i,ax,ticks=1,colorbar=1,label='')
#        plt.show()
    
    '''for one-one test'''
#    alpha=3
#    bate=5
#    idx_start=10
#    num=15
#    
#    ax_idx=1
#    howmany=0
#    fig                         = plt.figure(figsize=(12,8))
#    fig_idx                     = 1
#    for i,j in enumerate(only_right['label']):
#          if ax_idx > alpha*bate:
#              fig.savefig('C:/Users/night/Desktop/新建文件夹/{}.svg'.format(fig_idx))
#              if howmany==num:
#                  plt.show()
#                  break
#              fig                 = plt.figure(figsize=(12,8))
#              ax_idx              = 1
#              fig_idx            += 1
#          
#          if j==test_label and i > idx_start:
#              ax                  = fig.add_subplot(alpha,bate,ax_idx)
#              Rx_sample,R_rest    = net.lrp(x[:,only_right['idx'][i],:],test_label,eps,bias_factor)
#              if not Rx_sample_mean.shape[0] == Rx_sample.shape[0]:
#                  Rx_sample       = Rx_sample[0:time_dim,:]
#                  
#              Rx_sample_stand     = (Rx_sample-Rx_sample.min())/(Rx_sample.max()-Rx_sample.min())
#              R_channel           = np.sum(Rx_sample_stand , axis=0)
#              R_channel           = (R_channel-R_channel.min())/(R_channel.max()-R_channel.min())
#              R_time              = np.sum(Rx_sample_stand , axis=1)
#              R_time              = (R_time-R_time.min())/(R_time.max()-R_time.min())
#              Rx_sample_mean      = Rx_sample_stand
#              
#              ax_idx             += 1
#              howmany            += 1
#              
#          #        heatmap.topo_map(R_channel,ax)
#              heatmap.sample_map(Rx_sample_mean,ax,title='Relevance')
        
    
    
    '''for average test'''
    total_test_samp             = 30000   #共取多少样本进行平均
    start_test_num              = 0        #从第几个开始取
    fig                         = plt.subplots()
    class_LRP                   = []

    test_samp                   = 1
    for i,j in enumerate(only_right['label']):
        if test_samp==total_test_samp or i == len(only_right['label'])-1:
            ax1=plt.subplot(1,2,1)
            ax2=plt.subplot(1,2,2)
##            Rx_sample_mean  = Rx_sample_mean#/superposition
#            Rx_sample_mean  = (Rx_sample_mean-Rx_sample_mean.min())/(Rx_sample_mean.max()-Rx_sample_mean.min())
#            R_channel       = R_channel#/superposition
#            R_channel       = (R_channel-R_channel.min())/(R_channel.max()-R_channel.min())
#            R_time          = R_time#/superposition
##            R_time          = (R_time-R_time.min())/(R_time.max()-R_time.min())
            
            
            Rx_sample_mean  = (Rx_sample_mean-Rx_sample_mean.mean())/Rx_sample_mean.std()
            ttttt=Rx_sample_mean
            mask1           = np.where(Rx_sample_mean < 0. , 0.05 , 1.).astype('float32')
            mask2           = np.where(Rx_sample_mean < (Rx_sample_mean.max()*0.2) , 1.5 , 1.).astype('float32')
            mask2           = np.where(Rx_sample_mean > Rx_sample_mean.max()*0.4 , 0 , 1.).astype('float32')
            mmm=mask2
            mask=mask1*mask2#*mask3

            Rx_sample_mean  = Rx_sample_mean*mask
            Rx_sample_mean[np.where(Rx_sample_mean > Rx_sample_mean.max()*0.4)]    = Rx_sample_mean.max()*0.4
            
#            yyyyy=Rx_sample_mean
            
            Rx_sample_mean  = (Rx_sample_mean-Rx_sample_mean.min())/(Rx_sample_mean.max()-Rx_sample_mean.min())
            R_channel       = (R_channel-R_channel.mean())/R_channel.std()
            
            heatmap.topo_map(R_channel,ax1)
            heatmap.sample_map(Rx_sample_mean,ax2,ticks=1,colorbar=1)
            plt.show()
            break
        
        if j==test_label and i > start_test_num:
            
            Rx_sample,R_rest    = net.lrp(x[:,only_right['idx'][i],:],test_label,eps,bias_factor)
            class_LRP.append(net.s.copy())
            if Rx_sample.max() > 5000: #or Rx_sample.max() < 30:
                wtf.append(Rx_sample)
                idxxx.append(only_right['idx'][i])
                continue
                
            if  Rx_sample_mean.shape[0] != Rx_sample.shape[0]:
                Rx_sample       = Rx_sample[0:time_dim,:]

            
            Rx_sample_stand     = (Rx_sample-Rx_sample.min())/(Rx_sample.max()-Rx_sample.min())
#            Rx_sample_stand  = (Rx_sample-Rx_sample.mean())/Rx_sample.std()
            R_channel          += np.sum(Rx_sample_stand , axis=0)
            R_time             += np.sum(Rx_sample_stand , axis=1)
            Rx_sample_mean     += Rx_sample_stand
            
            test_samp    += 1
            
    
    '''测试时间动态性'''
    #fig,ax= plt.subplots()
    #heatmap.sample_map(x[:,0,:],ax)
    
    '''判断numpy模型是否对，已经验证是对的了'''
    #scores=0
    #lis=[]
    #ss=[]
    #k,l=only_right['idx'],only_right['label']
    #
    ##for i in range(len(l)):
    #for i in k:
    #    net.set_input(x[:,i,:])
    #    
    #    s=net.forward()
    #    lis.append(s)
    #    ss.append(np.argmax(s))
    #    if np.argmax(s)==y[i,0]:
    ##    if np.argmax(s)==y[i]:
    #        scores+=1
    #        
    #print(scores/len(l))
    
    ''''''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
