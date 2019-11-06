# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:02:04 2019

@author: night
"""

import numpy as np
import scipy.io

import torch
import torch.nn as nn



def get_test_class(data_loc):                                                  #取测试集数据
                
    mat            = scipy.io.loadmat(data_loc)
#    valx           = mat['hunhex']
    valx           = mat['valx']
    valx           = np.rollaxis(valx,1)
#    valy           = mat['hunhey']
    valy           = mat['valy']
          
    if valy.shape[0]<2:
        valy       = valy.reshape((-1,1))
      
    return  valx,valy   

def load_model_and_test(model_loc,x,y):                                        #测试模型的分类正确率

    batch=len(y)
    print(batch)
    Representation=np.zeros((batch,64))
    Representation_output=np.zeros(batch)
      
    x               = torch.from_numpy(x)
    y               = torch.from_numpy(y.reshape(batch)).long().view(-1)      #y：[trial数，1]→[trial数，]

    
    global ceshi_model, optimizer
    
    with open(model_loc, 'rb') as f:                                           #load模型
        ceshi_model, optimizer = torch.load(f)
        ceshi_model.cpu()
        
    ceshi_model.eval()
    output, hidden, rnn_hs, dropped_rnn_hs          = ceshi_model(x, None, return_h=1)
    pred_y          = torch.max(output, 1)[1].data
    accuracy        = (pred_y == y).float().sum() / len(y)
    print(accuracy)
    
    Representation_output[0:batch]=pred_y.clone().cpu().data.numpy().astype('int32')
    Representation[0:batch]=rnn_hs[0][-1,:,:].clone().cpu().data.numpy().astype('float32')

    
    return  Representation,Representation_output
#    return  output, rnn_hs







if __name__ == '__main__':
      x,y=get_test_class('D:/EEG/mat_data/Qiang_VCEEGall_50-250ms_all_torch_9.mat')
      R,R_out=load_model_and_test('D:/EEG/model/AWD-LSTM/Brother Qiang/50-250_9/15628726948027844_ADAMLr=0.005_acc=0.68_loss=1.13_epoch=150_model',x,y)
      scipy.io.savemat('D:/EEG/model/AWD-LSTM/Brother Qiang/50-250_9/15628726948027844_ADAMLr=0.005_acc=0.68_loss=1.13_epoch=150_vis.mat',
                       {'val_vector':R,
                       'val_output':R_out,
                       'real_tar':y
                       })
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       
                       