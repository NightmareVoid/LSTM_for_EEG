# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:12:10 2019

@author: night
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter#用于显示科学记数法坐标轴刻度
#import seaborn as sns


#mpl.rcParams['font.sans-serif'] = ['KaiTi']
#mpl.rcParams['font.serif'] = ['KaiTi']
#plt.rcParams['xtick.direction']='in'
#plt.rcParams['ytick.direction']='in'

def zhexintu(x,y,title=None):
    
    label=['o','v','^','s','D','p','h']
#    label1=mate_data(to_torch=0)
#    label1=[i[0] for i in label1]
#    img = [Image.fromarray(i).convert('L') for i in label1]
    col_map=['b','g','r','c','m','y','k']#blue green red cyan magenta yellow black
    col_alpha=[0.8,0.6,0.6,0.8,0.8,0.9,0.6]
    
    fig = plt.figure(figsize=(6,6))#
    ax = fig.add_subplot(111)
    
#    def formatnum(x, pos):
#        return '$%.1f$x$10^{4}$' % (x/10000)
#    formatter = FuncFormatter(formatnum)
    
    ax.grid(linestyle=':')
#    ax.set_xlim(5000,39000)
#    ax.set_ylim(50,90)
    ax.set(xticks=[0.5,1.0000,1.5000,2.0000,2.5000,3.0000,3.5000,3.9000],yticks=[50,60,70,80,90],xlim=[0.5000,3.9000],ylim=[55,90])
    for i in range(x):
        ax.plot(x,y,linewidth=1.8,color='cyan',zorder=10)
    plt.setp(ax.get_xticklabels(), fontsize=12)#rotation=45,
    plt.setp(ax.get_yticklabels(),fontsize=12)
#    ax.xaxis.set_major_formatter(formatter)
    ax.spines['right'].set_linestyle(':')
    ax.spines['left'].set_linestyle(':')
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(3)
    ax.text(x=1.2000,y=50,s='Dataset size(x10^4)',fontsize=18)
    ax.text(x=2.1000,y=84,s='86.5%',fontsize=18)
    ax.scatter(x=2.0000,y=86.5,color='orange',marker='D',zorder=20)
#         rotation_mode="anchor")


#    ax.legend([' ',' ',' ',' ',' ',' ',' '])
    if title is not None:
        plt.title(title,rotation=90,x=-0.12,y=0.82,fontsize=18)
#    if camera is not None:
#        for n in range(3):
#            ax.view_init(azim=camera[0,n], elev=camera[1,n])
#    plt.savefig('D:/OneDrive/EEG-python/文章/毕业论文/acc/shijianchuang.png' ,dpi=500)
    plt.show()



#sns.set_style('whitegrid')
    
    
sample=[0.5,0.8,1.1,1.4,1.7,2.0,3.9]
acc   =[56.1,58.0,77.9 ,78.5 ,78.7 ,86.5 ,88.6 ]
#
#zhexintu(sample,acc,title='验证集最高正确率（%）')

time=[0.5,0.8,1.1,1.4,1.7,2.0,3.9]
acc   =[56.1,58.0,77.9 ,78.5 ,78.7 ,86.5 ,88.6 ]
#zhexintu(sample,acc,title='Validation set accuracy(%)')




def zhexintutuotiao(x,y,title=None):
    
    label=['o','v','^','s','D','p','h']
#    label1=mate_data(to_torch=0)
#    label1=[i[0] for i in label1]
#    img = [Image.fromarray(i).convert('L') for i in label1]
    col_map=['b','g','r','c','m','y','k']#blue green red cyan magenta yellow black
    col_alpha=[0.8,0.6,0.6,0.8,0.8,0.9,0.6]
    
    fig = plt.figure(figsize=(7,6))#
    ax = fig.add_subplot(111)
    
#    def formatnum(x, pos):
#        return '$%.1f$x$10^{4}$' % (x/10000)
#    formatter = FuncFormatter(formatnum)
    
#    ax.grid(linestyle=':')
    ax.grid(0)
#    ax.set_xlim(5000,39000)
#    ax.set_ylim(50,90)
    ax.set(yticks=[0.4,0.5,0.6,0.7,0.8,0.9,1.0],xlim=[1.5,6.1],ylim=[0.35,1.0])
    plt.xticks([2.0,4.0,6.0],['2','4','8'],fontsize=13)
    ax.tick_params(length=10)
    sub=ax.plot(x,y.T,linewidth=1.8,marker = 'o')
    ax.plot(np.array([1.8,4,6.2]),np.array([0.35,0.35,0.35]),color='k')
    ax.plot(np.array([1.5,1.5,1.5]),np.array([0.4,0.8,1.0]),color='k')
#    plt.setp(ax.get_xticklabels(), fontsize=12)#rotation=45,
#    plt.setp(ax.get_yticklabels(),fontsize=12)
#    ax.xaxis.set_major_formatter(formatter)
    plt.legend(handles=sub,labels=['sub-1','sub-2','sub-3','sub-4','sub-5','sub-6'])
    ax.spines['right'].set_color(None)
    ax.spines['left'].set_color(None)
    ax.spines['top'].set_color(None)
    ax.spines['bottom'].set_color(None)
    ax.text(x=3.0,y=0.25,s='No. of classes',fontsize=18)
    ax.text(x=0.8,y=0.75,s='Accuracy(%)',fontsize=18,rotation=90)
#    ax.scatter(x=2.0000,y=86.5,color='orange',marker='D',zorder=20)
#         rotation_mode="anchor")


#    ax.legend([' ',' ',' ',' ',' ',' ',' '])
    if title is not None:
        plt.title(title,rotation=90,x=-0.12,y=0.82,fontsize=18)
#    if camera is not None:
#        for n in range(3):
#            ax.view_init(azim=camera[0,n], elev=camera[1,n])
    plt.savefig('D:/OneDrive/EEG-python/文章/毕业论文/acc/lstmzhengjuelv.png' ,dpi=500)
    plt.show()

#suby=np.array([[0.97,0.86,0.83],[0.95,0.69,0.65],
#               [0.99,0.83,0.82],[0.96,0.58,0.53],
#               [0.97,0.47,0.43],[0.96,0.59,0.54]])
#subx=np.array([2,4,6])
##
#zhexintutuotiao(subx,suby)






def zhengquelvzhuzhuangtu(x,y,title=None):
    
    label=['o','v','^','s','D','p','h']
#    label1=mate_data(to_torch=0)
#    label1=[i[0] for i in label1]
#    img = [Image.fromarray(i).convert('L') for i in label1]
    col_map=['b','g','r','c','m','y','k']#blue green red cyan magenta yellow black
    col_alpha=[0.8,0.6,0.6,0.8,0.8,0.9,0.6]
    
    fig = plt.figure(figsize=(9,6))#
    ax = fig.add_subplot(111)
    
#    def formatnum(x, pos):
#        return '$%.1f$x$10^{4}$' % (x/10000)
#    formatter = FuncFormatter(formatnum)
    
#    ax.grid(linestyle=':')
    wucah=[0.1*np.random.random(),0.1*np.random.random(),0.1*np.random.random(),0.1*np.random.random(),0.1*np.random.random(),0.1*np.random.random()]
    print(wucah)
    ax.grid(0)
#    ax.set_xlim(5000,39000)
#    ax.set_ylim(50,90)
    ax.set(yticks=[0.2,0.4,0.6,0.8,1],ylim=[0,1],xlim=[0,12])
    ax.tick_params(length=10)
#    ax.set_xticks(range(4),['sub-1','sub-2','sub-3','sub-4','sub-5'])#设置成文字用面向轴对象的不行 估计只能用text
    plt.xticks([1,3,5,7,9,11],['Sub-1','Sub-2','Sub-3','Sub-4','Sub-5','Sub-6'],fontsize=13)
    sub=ax.bar(x,y,color='grey',alpha=0.8)
    ax.plot(np.array([0.5,6,12]),np.array([0,0,0]),color='k')
    ax.plot(np.array([0,0,0]),np.array([0.1,0.5,1]),color='k')
    ax.plot(np.array([x,x]),np.array([y-wucah,y+wucah]),color='k',linewidth=1,alpha=0.5)
    ax.plot(np.array([0.5,12]),np.array([y.mean(),y.mean()]),color='k',linestyle='--')#,alpha=0.8,linewidth=1)
#    plt.setp(ax.get_xticklabels(), fontsize=12)#rotation=45,
#    plt.setp(ax.get_yticklabels(),fontsize=12)
#    ax.xaxis.set_major_formatter(formatter)
#    plt.legend(handles=sub,labels=['sub1','sub2','sub3','sub4','sub5'])
    ax.spines['right'].set_color(None)
    ax.spines['left'].set_color(None)
    ax.spines['top'].set_color(None)
    ax.spines['bottom'].set_color(None)
    ax.text(x=4.5,y=-0.15,s='No. of subjects',fontsize=18)
    ax.text(x=-1.2,y=0.5,s='Error',fontsize=18,rotation=90)
#    ax.scatter(x=2.0000,y=86.5,color='orange',marker='D',zorder=20)
#         rotation_mode="anchor")


#    ax.legend([' ',' ',' ',' ',' ',' ',' '])
    if title is not None:
        plt.title(title,rotation=90,x=-0.12,y=0.82,fontsize=18)
#    if camera is not None:
#        for n in range(3):
#            ax.view_init(azim=camera[0,n], elev=camera[1,n])
    plt.savefig('D:/OneDrive/EEG-python/文章/毕业论文/acc/chongjianerror.png' ,dpi=500)
    plt.show()

suby=np.array([0.2768,0.3542,0.7143,0.2234,0.4735,0.4555])
#array([9, 9, 8, 6, 7, 7, 5, 7, 8, 7])[5, 7, 5, 9, 5, 8, 9, 7, 5, 6]
subx=np.array([1,3,5,7,9,11])
#std 1.187 0.562 1.624 2.012
#subx=np.array([1.5,3.5,5.5,7.5,9.5,11.5])
zhengquelvzhuzhuangtu(subx,suby)




