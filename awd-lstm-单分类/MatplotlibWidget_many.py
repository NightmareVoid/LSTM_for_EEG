# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 22:42:23 2019

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
import scipy.io as spi


class MyMplCanvas(FigureCanvas):
    """FigureCanvas的最终的父类其实是QWidget。"""

    def __init__(self, parent=None):

        # 配置中文显示
        plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 此处初始化子图一定要在初始化函数之前
#        self.fig                         = plt.figure(figsize=(8,7))
        self.fig                         = plt.figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        '''定义FigureCanvas的尺寸策略，这部分的意思是设置FigureCanvas，使之尽可能的向外填充空间。'''
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class MatplotlibWidget_many(QWidget):#绘图类
    def __init__(self, parent=None):
        super(MatplotlibWidget_many, self).__init__(parent)
        self.initplotUi()
        self.heatmap                     = Heatmap()
        self.all_Rx_sample_mean          = []

    def initplotUi(self):#初始化函数
        #定义qt控件
        self.layout = QVBoxLayout(self)
        self.mpl = MyMplCanvas(self)
        self.layout.addWidget(self.mpl)
        self.ax_idx=0    
        
    def save(self,name):
#        self.mpl.fig.savefig('C:/Users/night/Desktop/{}.png'.format(name))
        self.mpl.fig.clear()
        self.ax_idx=0
        spi.savemat('D:/OneDrive/SSVEP及视图解码/某著名PPT实验室/某PPT实验室/LRP分析/{}.mat'.format(name),{'Rx_sample_mean':self.all_Rx_sample_mean})
        self.all_Rx_sample_mean=[]


    def lrpplot(self,Rx_sample_mean):
        self.ax_idx +=1
        if self.ax_idx in [1,6,11]:
            ticks=1
        else:
            ticks=0
        if self.ax_idx in [5,10,15]:
            colorbar=1
        else:
            colorbar=0
        self.heatmap.sample_map(Rx_sample_mean,self.mpl.fig.add_subplot(3,5,self.ax_idx),title='Relevance',ticks=ticks,colorbar=colorbar)
        self.mpl.draw()
        
    def get_Rx_sample_mean(self,Rx_sample_mean):
        self.all_Rx_sample_mean.append(Rx_sample_mean)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        