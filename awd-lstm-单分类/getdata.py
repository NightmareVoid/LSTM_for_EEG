# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 17:32:11 2019

@author: night
"""

import mne
#import sys
import numpy as np
np.random.seed(1337)
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import scipy.io
import  random
#import h5py
#import matplotlib.pyplot as plt



#channel_we_need=[43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]  #取导联,python中由于从0开始所以前移了一位，matlab中是从44开始
#channel_we_need=[47,53,54,55,56,57,60,61,62]
##channel_name=['P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']
#channel_name=['PZ','PO5','PO3','POz','PO4','PO6','O1','Oz','O2']
#channel_Num=len(channel_we_need)


#def readeegfromcnt(filepath):
#  date_dir=filepath#'D:\OneDrive\EEG-python'#数据目录
# # print(len(data))
# 
#  raw=mne.io.read_raw_cnt(data_dir,montage=None,preload=True)
#  raw.info['bads']=['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3',
# 'C1','CZ','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2']
#  raw.filter(l_freq=0.5, h_freq=None)
#  events = mne.find_events(raw, 'STI 014')
#  epochs = mne.Epochs(raw, events, event_id=[1,2,3,4], tmin=tmin, tmax=tmax,baseline=None,picks=channel_we_need)
#  
#  #数据标准化
#  datatemp=epochs.get_data()
#  print('datatemp_size:',len(datatemp))
#  for i in range(len(datatemp)):
#      for j in range(len(datatemp[i,])):
#          datatemp[i,j,]=scale(datatemp[i,j,])
#  print('datatemp_scale_size:',len(datatemp))  
#
#     
#   #数据提取       
#  for i in range(len(epochs.events)):
#      datatemp1=[epochs.events[i,3],datatemp[i]]
#      data.append(datatemp1)
#  print('data_size_temp:',len(data),'::::',data[-5][0][0],' =? ',epochs.event_id[-5],' =? ',epochs.events[-5,3])   

#########################################################
def getdata(rootpath):
    xdata=[]
    ydata=[]
    for root,dirs,files in os.walk(rootpath):
        for file in files:
            if 'cnt' in file:
#                print(file)
#                readeegfromcnt(os.path.join(root,file))
                raw=mne.io.read_raw_cnt(os.path.join(root,file),montage=None,preload=True)
                raw.info['bads']=['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3',
                                  'C1','CZ','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2','CB1','CB2','HEO','VEO','EKG','EMG','PO7','PO8',
                                  'P7','P5','P3','P4','P1','P6','P8','P2']
                events = mne.find_events(raw, 'STI 014')
                raw.filter(l_freq=1.0, h_freq=70,method='iir')
#                print(events)
                events_id=[]
                for elem in [1,2,3,4]:
                    if elem in events:
                        events_id.append(elem)
                events_id=[61,62,63,64 ,65, 66, 67, 68, 69, 70, 71, 72, 73, 74]
                classiser=[0,50,100,150,200,250,300,350,400,450,500,550,600,650]
#                print(events_id)
                epochs = mne.Epochs(raw, events, event_id=events_id, tmin=-0.85, tmax=0,baseline=None,picks=channel_we_need)
#                print(epochs.event_id)
#                epochs.plot(n_channels=2)
                
            
##            #数据标准化
                datatemp=epochs.get_data()
#              #  plt.plot(100000*datatemp[1,1,:])
##                print('datatemp_shape:',datatemp.shape)
                for i in range(len(datatemp)):
                    for j in range(len(datatemp[i,])):
                        datatemp[i,j,]=scale(datatemp[i,j,])
#                print('datatemp_scale_shape:',datatemp.shape)
#                plt.plot(datatemp[1,1,:],'r')
#                plt.show()
#            
#            #数据提取
                for i in range(len(epochs.events)):
#                    datatemp1=[np.array(epochs.events[i,2])]
#                    datatemp2=[datatemp[i]]
                    xdata.append(datatemp[i].T)
                    ydata.append(epochs.events[i,2]-60)
#                print('data_size_temp:',len(xdata),'::::',' =? ',epochs.events[-5][2])
#                plt.plot(xdata[1][1],'g')
#                print(ydata[1])
#                plt.show()
    xdata,ydata=np.array(xdata),np.array(ydata)
    xdata_random,ydata_random=shuffle(xdata,ydata)
#    xdata,ydata=shuffle(np.array(xdata),np.array(ydata))
    xdata=xdata[np.argsort(ydata)]
    ydata=ydata[np.argsort(ydata)]
    xdata=[xdata[i:i+50] for i in classiser]
    ydata=ydata.reshape((14,50))
    scipy.io.savemat('D:/EEG/dujiale_data/dujiale.mat',{'xdata':xdata,'ydata':ydata,'xdata_random':xdata_random,
                                  'ydata_random':ydata_random})
    raw.close()
           
  
#        print('data_size(should=data_size_temp):',len(data),'\n')
#    data=np.random.shuffle(data)#随机打乱顺序   
    return xdata,ydata

#################
def getdata2(rootpath):#


    tmin5=-0.25
    tmax5=0.0
    tmin0=0.5 
    tmax0=0.75 
    tmin1=1.0 
    tmax1=1.25 
    tmin2=1.5 
    tmax2=1.75 
    tmin3=2.0 
    tmax3=2.25 
    tmin4=2.5 
    tmax4=2.75 
    
    xdata=[]
    ydata=[]
    channel_we_need=[43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]  #取导联,python中由于从0开始所以前移了一位，matlab中是从44开始
#    channel_name=['P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']
    for root,dirs,files in os.walk(rootpath):
#        print(files)
        for file in files:
            if 'cnt' in file:
#                print(file)
#                readeegfromcnt(os.path.join(root,file))
                raw=mne.io.read_raw_cnt(os.path.join(root,file),montage=None,preload=True)
                raw.info['bads']=['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3',
                                  'C1','CZ','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2']
#                raw.set_eeg_reference(ref_channels='average')
                raw.filter(l_freq=1.0, h_freq=70,method='iir')
                events = mne.find_events(raw, 'STI 014')
                events_id=[]
#                if 8 in events[:,2] or 9 in events[:,2] or 11 in events[:,2] or 12 in events[:,2] or 13 in events[:,2]:#or 10 in events[:,2] 
                if 8 in events[:,2] or 9 in events[:,2] or 10 in events[:,2] or 11 in events[:,2] or 12 in events[:,2] or 13 in events[:,2]:#
                    continue
#                if 1 in events[:,2] or 2 in events[:,2] or 3 in events[:,2] or 4 in events[:,2] or 5 in events[:,2] or 6 in events[:,2] or 7 in events[:,2]:
#                    continue
#                for elem in [8,9,10,11,12,13]:
                for elem in [1,2,3,4,5,6,7]:
                    if elem in events:
                        events_id.append(elem)
                epochs0 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin0, tmax=tmax0,baseline=None,picks=channel_we_need)
                epochs1 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin1, tmax=tmax1,baseline=None,picks=channel_we_need)
                epochs2 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin2, tmax=tmax2,baseline=None,picks=channel_we_need)
                epochs3 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin3, tmax=tmax3,baseline=None,picks=channel_we_need)
                epochs4 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin4, tmax=tmax4,baseline=None,picks=channel_we_need)
#                if np.random.randint(10)<7: #避免空闲太多
#                    epochs5 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin5, tmax=tmax5,baseline=None,picks=channel_we_need)
#                #数据标准化5 空闲
#                    datatemp5=epochs5.get_data()
#                    for i in range(len(datatemp5)):
#                        for j in range(len(datatemp5[i,])):
#                            datatemp5[i,j,]=scale(datatemp5[i,j,])   
#    #            #数据提取5
#                    for i in range(len(epochs5.events)):
#                        xdata.append(datatemp5[i].T)
#                        ydata.append(np.array(7))
#                
             #数据标准化0
                datatemp0=epochs0.get_data()
                for i in range(len(datatemp0)):
                    for j in range(len(datatemp0[i,])):
                        datatemp0[i,j,]=scale(datatemp0[i,j,])   
#           #数据提取0
                for i in range(len(epochs0.events)):
                    xdata.append(datatemp0[i].T)
                    ydata.append(np.array(epochs0.events[i,2]-1))
                    
                    
#           #数据标准化1
                datatemp1=epochs1.get_data()
                for i in range(len(datatemp1)):
                    for j in range(len(datatemp1[i,])):
                        datatemp1[i,j,]=scale(datatemp1[i,j,])   
#           #数据提取1
                for i in range(len(epochs1.events)):
                    xdata.append(datatemp1[i].T)
                    ydata.append(np.array(epochs1.events[i,2]-1))
                    
              
            #数据标准化2
                datatemp2=epochs2.get_data()
                for i in range(len(datatemp2)):
                    for j in range(len(datatemp2[i,])):
                        datatemp2[i,j,]=scale(datatemp2[i,j,])   
#           #数据提取2
                for i in range(len(epochs2.events)):
                    xdata.append(datatemp2[i].T)
                    ydata.append(np.array(epochs2.events[i,2]-1))
                    
                    
                
            #数据标准化3
                datatemp3=epochs3.get_data()
                for i in range(len(datatemp3)):
                    for j in range(len(datatemp3[i,])):
                        datatemp3[i,j,]=scale(datatemp3[i,j,])   
#            #数据提取3
                for i in range(len(epochs3.events)):
                    xdata.append(datatemp3[i].T)
                    ydata.append(np.array(epochs3.events[i,2]-1))
                    
            #数据标准化4
                datatemp4=epochs4.get_data()
                for i in range(len(datatemp4)):
                    for j in range(len(datatemp4[i,])):
                        datatemp4[i,j,]=scale(datatemp4[i,j,])   
#            #数据提取4
                for i in range(len(epochs4.events)):
                    xdata.append(datatemp4[i].T)
                    ydata.append(np.array(epochs4.events[i,2]-1))    

#                return xdata,ydata
                    
    xdata,ydata=shuffle(np.array(xdata,dtype='float32'),np.array(ydata,dtype='float32'))
#    scipy.io.savemat('D:/EEG/mat_data/VCEEGall_180ms_all.mat',{'xdata':xdata,
#                                  'ydata':ydata})
    raw.close()
    print('-------------------Done-----------------------')
    
    return xdata,ydata


#######################
def getdatafrommat(rootpath,strr): 
    for elem in os.listdir(rootpath):
#        print(elem)
        if '{}.mat'.format(strr) in elem:
            mat=scipy.io.loadmat(rootpath+'\\'+elem)
            xdata=mat['xdata']
            ydata=mat['ydata']
#            ydata=ydata.T
#            mat.close()
    return xdata,ydata

def getdatafrommat2(rootpath,strr): 
    for elem in os.listdir(rootpath):
        if strr in elem:
            mat=scipy.io.loadmat(rootpath+'/'+elem)
            xdata=mat['trainx']
#            xdata=np.rollaxis(xdata,1)
#            xdata=xdata[:,0:20224,:]
            ydata=mat['trainy']
#            if ydata.shape[0]<2:
#                ydata=ydata.reshape((-1,1))
#            ydata=ydata[0:20224]
            valx=mat['valx']
#            valx=np.rollaxis(valx,1)
            valy=mat['valy']
#            if valy.shape[0]<2:
#                valy=valy.reshape((-1,1))
#            mat.close()

            
    return xdata,ydata,valx,valy
#######################################################
def getdatazxy(rootpath):
    xdata=[]
    ydata=[]
    
#    channel_name_zxy=['CPP5H', 'CPP3H', 'CPPZ', 'CPP4H', 'CPP6H', 'P9H', 'P7H', 'P5H', 'P3H', 'P1H', 'P2H', 'P4H', 
#     'P6H', 'P8H', 'P10H', 'PPO9H', 'PPO7H', 'PPO5H', 'PPO3H', 'PPO1H', 'PPOZ', 'PPO2H', 'PPO4H', 
#     'PPO6H', 'PPO8H', 'PPO10H', 'PO9', 'PO9H', 'PO7', 'PO5', 'PO3', 'PO1', 'M1', 'POZ', 'PO2', 'PO4',
#     'PO6', 'PO8', 'PO10H', 'PO10', 'POO9H', 'POO7', 'M2', 'POO5H', 'POO1', 'POOZ', 'POO2', 'POO6H',
#     'POO8', 'POO10H', 'O1', 'O1H', 'OZ', 'O2H', 'O2', 'OCB1', 'OCB1H', 'OCB2H', 'OCB2', 'CB1', 'CB1H', 'CBZ', 'CB2H', 
#     'CB2', 'STI 014']
#    channel_we_need_zxy=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41,43,44,45,
#                         46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]   ## Ori Ref
    channel_we_need_zxy=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41,43,44,45,
                         46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]    ###M1M2Ref
#    channel_we_need_zxy=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,41,43,44,45,
#                         46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]   ###AverRef

    events_id_all=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    
    for root,dirs,files in os.walk(rootpath):
        for file in files:
            if 'cnt' in file:
                raw=mne.io.read_raw_cnt(os.path.join(root,file),montage=None,preload=True)
                
                raw.info['bads']=[]
                raw.set_eeg_reference(ref_channels=['M1','M2'])
#                raw.set_eeg_reference(ref_channels=['M1','M2'])
                
                events = mne.find_events(raw, 'STI 014')
                raw.filter(l_freq=0,h_freq=70,method='iir')
                events_id=[]
                for elem in events_id_all:
                    if elem in events:
                        events_id.append(elem)
                
                
                epochs = mne.Epochs(raw, events, event_id=events_id, tmin=0, tmax=0.2,baseline=None,picks=channel_we_need_zxy)
                
            
##           
                datatemp=epochs.get_data()

#            
#            #数据提取
                for i in range(len(epochs.events)):
#                    datatemp1=[np.array(epochs.events[i,2])]
#                    datatemp2=[datatemp[i]]
                    xdata.append(datatemp[i].T)
                    ydata.append(epochs.events[i,2])
                    
    xdata,ydata=np.array(xdata,dtype='float32'),np.array(ydata,dtype='float32')
    xdata_random,ydata_random=shuffle(xdata,ydata)
#    xdata,ydata=shuffle(np.array(xdata),np.array(ydata))
    xdata=xdata[np.argsort(ydata)]
    ydata=ydata[np.argsort(ydata)]
    
    classiser=[np.where(ydata==i)[0][0] for i in events_id_all]
    classiser.append(len(ydata))
    
    xdata=[xdata[classiser[i]:classiser[i+1]] for i in range(len(classiser)-1)]
#    ydata=ydata.reshape((14,50))
    scipy.io.savemat('D:/EEG/zhouxiaoyu_data/zhouxiaoyu_M1M2Ref_data.mat',{'xdata':xdata,'ydata':ydata,'xdata_random':xdata_random,
                                  'ydata_random':ydata_random})
    raw.close()
           
  
#        print('data_size(should=data_size_temp):',len(data),'\n')
#    data=np.random.shuffle(data)#随机打乱顺序   
    return xdata,ydata

def order(x,y):
    xx=x
    yy=y
    x=x[np.argsort(y[:,0])]
    y=y[np.argsort(y[:,0])]
    classiser=[np.where(y==i)[0][0] for i in [0,1,2,3]]
    classiser.append(len(y))
    x=[x[classiser[i]:classiser[i+1]] for i in range(len(classiser)-1)]
#    scipy.io.savemat('D:/EEG/mat_data/VCEEGall_50-250ms_all_order.mat',{'xdata':x,'ydata':y,'xdata_random':xx,
#                                  'ydata_random':yy})
    return x,y

def gettrainandval(x,y,path):
    print(path)
    x,y=shuffle(x,y)
    i=1
    i=1
    i=1
    x,y=shuffle(x,y)
    i=1
    i=1
    i=1
    x,y=shuffle(x,y)
    valx=x[0:int(0.1*len(y))]
    valy=y[0:int(0.1*len(y))]
    trainx=x[int(0.1*len(y)):]
    trainy=y[int(0.1*len(y)):]
    print('\n ----------------开始保存--------------------')
    scipy.io.savemat(path,{'trainx':trainx,'trainy':trainy,'valx':valx,
                                  'valy':valy})
    return trainx,trainy,valx,valy

def getdata3(rootpath):#

    
    
    tmin0=0.05
    tmax0=0.2
    tmin1=0.55
    tmax1=0.7
    tmin2=1.05
    tmax2=1.2
    tmin3=1.55
    tmax3=1.7
    tmin4=2.05
    tmax4=2.20
    tmin5=2.55
    tmax5=2.7
    
    xdata=[]
    ydata=[]
    channel_we_need=[43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]  #取导联,python中由于从0开始所以前移了一位，matlab中是从44开始
#    channel_name=['P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']
    for root,dirs,files in os.walk(rootpath):
        print(22222222222222222222222222222222222222222222222222222222222222222)
        for file in files:
            if 'cnt' in file:
#                print(file)
#                readeegfromcnt(os.path.join(root,file))
                raw=mne.io.read_raw_cnt(os.path.join(root,file),montage=None,preload=True)
                raw.info['bads']=['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3',
                                  'C1','CZ','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','M2']
                raw.filter(l_freq=1.0, h_freq=70,method='iir')
                events = mne.find_events(raw, 'STI 014')
                if not(1 in events[:,2] or 2 in events[:,2] or 2 in events[:,2] or 2 in events[:,2]):
                    break
                events_id=[]
                for elem in [1,2,3,4]:
                    if elem in events:
                        events_id.append(elem)
                epochs0 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin0, tmax=tmax0,baseline=None,picks=channel_we_need)
                epochs1 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin1, tmax=tmax1,baseline=None,picks=channel_we_need)
                epochs2 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin2, tmax=tmax2,baseline=None,picks=channel_we_need)
                epochs3 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin3, tmax=tmax3,baseline=None,picks=channel_we_need)
                epochs4 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin4, tmax=tmax4,baseline=None,picks=channel_we_need)
                epochs5 = mne.Epochs(raw, events, event_id=events_id, tmin=tmin5, tmax=tmax5,baseline=None,picks=channel_we_need)

                
             #数据标准化0
                datatemp0=epochs0.get_data()
                for i in range(len(datatemp0)):
                    for j in range(len(datatemp0[i,])):
                        datatemp0[i,j,]=scale(datatemp0[i,j,])   
#           #数据提取0
                for i in range(len(epochs0.events)):
                    xdata.append(datatemp0[i].T)
                    ydata.append(np.array(epochs0.events[i,2]-1))
                    
                    
#           #数据标准化1
                datatemp1=epochs1.get_data()
                for i in range(len(datatemp1)):
                    for j in range(len(datatemp1[i,])):
                        datatemp1[i,j,]=scale(datatemp1[i,j,])   
#           #数据提取1
                for i in range(len(epochs1.events)):
                    xdata.append(datatemp1[i].T)
                    ydata.append(np.array(epochs1.events[i,2]-1))
                    
              
            #数据标准化2
                datatemp2=epochs2.get_data()
                for i in range(len(datatemp2)):
                    for j in range(len(datatemp2[i,])):
                        datatemp2[i,j,]=scale(datatemp2[i,j,])   
#           #数据提取2
                for i in range(len(epochs2.events)):
                    xdata.append(datatemp2[i].T)
                    ydata.append(np.array(epochs2.events[i,2]-1))
                    
                    
                
            #数据标准化3
                datatemp3=epochs3.get_data()
                for i in range(len(datatemp3)):
                    for j in range(len(datatemp3[i,])):
                        datatemp3[i,j,]=scale(datatemp3[i,j,])   
#            #数据提取3
                for i in range(len(epochs3.events)):
                    xdata.append(datatemp3[i].T)
                    ydata.append(np.array(epochs3.events[i,2]-1))
                    
            #数据标准化4
                datatemp4=epochs4.get_data()
                for i in range(len(datatemp4)):
                    for j in range(len(datatemp4[i,])):
                        datatemp4[i,j,]=scale(datatemp4[i,j,])   
#            #数据提取4
                for i in range(len(epochs4.events)):
                    xdata.append(datatemp4[i].T)
                    ydata.append(np.array(epochs4.events[i,2]-1))
                    
            #数据标准化4
                datatemp5=epochs5.get_data()
                for i in range(len(datatemp5)):
                    for j in range(len(datatemp5[i,])):
                        datatemp5[i,j,]=scale(datatemp5[i,j,])   
#            #数据提取4
                for i in range(len(epochs5.events)):
                    xdata.append(datatemp5[i].T)
                    ydata.append(np.array(epochs5.events[i,2]-1)) 
                    
                    
    xdata,ydata=shuffle(np.array(xdata,dtype='float32'),np.array(ydata,dtype='float32'))
#    scipy.io.savemat('D:/EEG/mat_data/VCEEGall_180ms_all.mat',{'xdata':xdata,
#                                  'ydata':ydata})
    raw.close()
    print('Done')
    
    return xdata,ydata
    

def divide_lz_data(x,y,valx,valy,name):
    
    part1x = x[0:int(x.shape[0]*0.5),:,:]
    part1y = y[:,0:int(x.shape[0]*0.5)]
    
    part2x = x[int(x.shape[0]*0.5):,:,:]
    part2y = y[:,int(x.shape[0]*0.5):]
    
    
    
    valx=valx
    valy=valy
    scipy.io.savemat('D:/EEG/mat_data/{}'.format(name[0]),{'trainx':part1x,'trainy':part1y,'valx':valx,
                    'valy':valy})
    scipy.io.savemat('D:/EEG/mat_data/{}'.format(name[1]),{'trainx':part2x,'trainy':part2y,'valx':valx,
                    'valy':valy})
    
    return part1x,part1y,part2x,part2y


if __name__  == "__main__":     
    rootpath='D:/EEG/EEG/qjt'
    x,y=getdata2(rootpath)
#    aa,bb,cc,dd= gettrainandval(x,y,path='D:/EEG/mat_data/yhq_VCEEGall_0-250ms_all_torch_9.mat')
    aa,bb,cc,dd= gettrainandval(x,y,path='D:/EEG/mat_data/qjt_VCEEGall_0-250ms_all_torch_8.mat')
#    xx,yy,vx,vy=getdatafrommat2('D:/EEG/mat_data/','LZ_VCEEGall_50-250ms_all_torch_9.mat')
#    a,b,c,d=divide_lz_data(xx,yy,vx,vy,name=['LZ_VCEEGall_50-250ms_all_torch_9_part1.mat','LZ_VCEEGall_50-250ms_all_torch_9_part2.mat'])
#    v,n=order(x,y)


    
    
    
    
    