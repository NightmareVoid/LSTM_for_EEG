# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:48:56 2019

@author: night
"""

import sys
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSizePolicy, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from LRP_eeg import *


class MyMplCanvas(FigureCanvas):
    """FigureCanvas的最终的父类其实是QWidget。"""

    def __init__(self, parent=None):#, width=5, height=6

        # 配置中文显示
        plt.rcParams['font.family']        = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 此处初始化子图一定要在初始化函数之前
        self.fig,self.ax                   = plt.subplots()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)


        '''定义FigureCanvas的尺寸策略，这部分的意思是设置FigureCanvas，使之尽可能的向外填充空间。'''
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class MatplotlibWidget_one(QWidget):#绘图类
    def __init__(self, parent=None):
        super(MatplotlibWidget_one, self).__init__(parent)

    def initplotUi(self,idx_start=0,time='50-250',bias='loss'):#初始化函数
        #定义qt控件
        self.layout  = QVBoxLayout(self)
        self.mpl     = MyMplCanvas(self)
        self.layout.addWidget(self.mpl)
        
        util                             = utils()
        '''获得验证集数据'''
        self.model_loc,self.para_loc,self.data_loc = util.location(time,bias)
        #     50-200  50-250  0-250  0-300  100-200  100-250  100-300
        self.x,self.y                    = util.get_test_class(self.data_loc)                                 #x(151,2865,21)  y(2865,1)
        
        '''测试pytorch模型是否对，并返回仅正确分类的label和索引，only_right{'idx'索引，'label'标签}'''
        self.only_right                  = util.load_model_and_test(self.model_loc,self.x,self.y)
#        self.only_right={'label':[1],'idx':[2]}
        '''LRP实例'''
        self.label                       = iter(self.only_right['label'][idx_start:])
        self.idx                         = iter(self.only_right['idx'][idx_start:])
        
        self.eps                         = 0.001
        self.bias_factor                 = 0.0
        
        self.net                         = LSTM_for_LRP(self.para_loc)
        self.heatmap                     = Heatmap()
        
        self.R_channel                   = np.zeros(self.x.shape[2])
        #R_time                      = np.zeros(x.shape[0])
        #Rx_sample_mean              = np.zeros(x[:,0,:].shape)
        self.R_time                      = np.zeros(self.x.shape[0]-1)
        self.Rx_sample_mean              = np.zeros((self.x[:,0,:].shape[0]-1,self.x[:,0,:].shape[1]))
        
        
    def pltlrp(self,test_label):
        #画图，并移动到下一个
        while 1:
            self.temp_label  = next(self.label)
            self.temp_idx    = next(self.idx)
#            print('idx:',self.temp_idx)
            
            if self.temp_label == test_label:
                print('idx:',self.temp_idx, 'label:',self.temp_label)
                self.Rx_sample,self.R_rest    = self.net.lrp(self.x[:,self.temp_idx,:],test_label,self.eps,self.bias_factor)
                self.Rx_sample                = self.Rx_sample[0:-1,:]
                self.Rx_sample_stand          = (self.Rx_sample-self.Rx_sample.min())/(self.Rx_sample.max()-self.Rx_sample.min())
                self.R_channel                = np.sum(self.Rx_sample_stand , axis=0)
                self.R_channel                = (self.R_channel-self.R_channel.min())/(self.R_channel.max()-self.R_channel.min())
                self.R_time                   = np.sum(self.Rx_sample_stand , axis=1)
                self.R_time                   = (self.R_time-self.R_time.min())/(self.R_time.max()-self.R_time.min())
                self.Rx_sample_mean           = self.Rx_sample_stand

                self.heatmap.sample_map(self.Rx_sample_mean,self.mpl.ax,title='Relevance',ticks=0,colorbar=0)
                self.mpl.draw()
                print(self.Rx_sample_mean.shape)
                break
            
            
    def send(self):
        return self.Rx_sample_mean


                    











