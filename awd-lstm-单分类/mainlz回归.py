# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:02:58 2019

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
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import os
import pickle
from tensorboardX import SummaryWriter
wri=SummaryWriter(log_dir='D:/EEG/tensorboard/temp/LZ_50-250_9')

import scipy.io
import model
from matplotlib import pyplot as plt

from utils import repackage_hidden,get_batchlz,mate_data

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='LZ_VCEEGall_50-250ms_all_torch_5.mat',
                    help='location of the root path')
parser.add_argument('--img_data', type=str, default='img_representation_4.mat',
                    help='location of the root path')
parser.add_argument('--rootpath', type=str, default='D:/EEG/mat_data/',
                    help='location of the data corpus')
parser.add_argument('--class_num', type=int, default=8,
                    help='有多少类')
parser.add_argument('--one_fold', type=bool, default=0,
                    help='one fold cross validation')
parser.add_argument('--eval_all', type=bool, default=0,
                    help='evaluate all data')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--ninp', type=int, default=21,
                    help='输入维度')
parser.add_argument('--nhid', type=int, default=64,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.005,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.15,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=201,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.5,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                    help='report interval') #报告时间间隔
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='D:/EEG/model/AWD-LSTM/'+randomhash,
                    help='path to save the final model')
parser.add_argument('--vis_save', type=str,  default='D:/OneDrive/EEG-python/awd-lstm-单分类/'+randomhash,
                    help='path to save the visualization data')
parser.add_argument('--alpha', type=float, default=1,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[150],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args                = parser.parse_args()
args.tied           = False

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

'''#################################################################
判断显卡是否可用
###################################################################'''
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

'''###############################################################################
# Load data
###############################################################################'''
def getdatafrommat(): 
    img_mat        = scipy.io.loadmat(args.rootpath+'/'+args.img_data)
    img=img_mat['vec']#4,64
    for elem in os.listdir(args.rootpath):
        if '{}'.format(args.data) in elem:
            mat            = scipy.io.loadmat(args.rootpath+'/'+elem)
            xdata          = mat['trainx'] #n*sample*ch
            xdata          = np.rollaxis(xdata,1)#sample n ch
            xdata          = xdata[:,0:len(xdata[1])//128*128,:]
            ydata          = mat['trainy'] # 1*n
            img_y          = img[[ydata.reshape(-1).astype('int32')]]
            if ydata.shape[0]<2:
                ydata      = ydata.reshape((-1,1))# n*1
            ydata          = ydata[0:len(ydata)//128*128]
            valx           = mat['valx']
            valx           = np.rollaxis(valx,1)
            valy           = mat['valy']
            if valy.shape[0]<2:
                valy       = valy.reshape((-1,1))
            
            print('train_num:',xdata.shape[1])
            print('train_lab:',ydata.shape)
            print('val_num:',valx.shape[1])
            print('val_lab:',valy.shape)
    
#    del(mat,img_mat)
#    return 
    return xdata,ydata,valx,valy,img_y
'''##############################################################################
#将numpy转为torch.tensor以及放到显卡上
###############################################################################'''
train_data,train_label,val_data,val_label,train_img  = getdatafrommat()
val_data,val_label                         = torch.from_numpy(val_data),torch.from_numpy(val_label.reshape(len(val_label))).long()
train_data,train_label                     = torch.from_numpy(train_data),torch.from_numpy(train_label.reshape(len(train_label))).long()
train_img                                  = torch.from_numpy(train_img).float()
if args.cuda:
    train_data,train_label                 = train_data.cuda(),train_label.cuda()#和nn.Module不同，调用tensor.cuda()只是返回这个tensor对象在GPU内存上的拷贝，而不会对自身进行改变。因此必须对tensor进行重新赋值，即tensor=tensor.cuda().
    val_data,val_label                     = val_data.cuda(),val_label.cuda()
    train_img                              = train_img.cuda()

eval_batch_size                            = len(val_label)#一次测试全部测试集数据
print(len(val_label))

'''###############################################################################
实时绘制回归图
################################################################################'''
def plt_vec(output_vec,img_vec,epoch):
       if epoch != 1:
              lines=[0,0,0,0]
              pass
       else:
              plt.ion()
              img_mat        = scipy.io.loadmat(args.rootpath+'/'+args.img_data)
              img=img_mat['vec']#4,64
              fig = plt.figure(figsize=(10,6))
              
              ax=[]
              col=['r','b','m','g']
              
              for i in range(4):
                     temp_ax=fig.add_subplot(2,2,i+1)
                     temp_ax.set(xticks=[0,30,64],yticks=[])
                     ax.append(temp_ax)
              for i in range(4):
                     ax[i].plot(img[i],color=col[i])
       

'''###############################################################################
#保存，load模型函数   #保存权重参数函数，后面LRP或者SA要用
###############################################################################'''
def model_save(fn,all_model=1,model_para=0):
    if all_model:
        with open(fn, 'wb') as f:
            torch.save([model, optimizer], f)
    if model_para:
        para_dic          = {}
        for name,para in model.named_parameters():
            para_dic[name]= para.clone().cpu().data.numpy()
        with open(fn,'wb') as f:
            pickle.dump(para_dic,f)

def model_load(fn):
    global model, optimizer
    with open(fn, 'rb') as f:
        model, optimizer  = torch.load(f)
#        
#'''###############################################################################
## 测试集验证   
################################################################################'''
#'''主要测试函数  不使用最终模型看看''' 
#def evaluate(data_source,data_source_target, batch_size=len(val_label)):
#    # Turn on evaluation mode which disables dropout.
#    model.eval()
#    if args.model == 'QRNN': model.reset()
#    
#    total_loss                             = 0
##    hidden                                 = model.init_hidden(batch_size) #hidden并不是权重！hidden是那个和细胞状态等同的那个h
#
#    data, targets                          = get_batchlz(data_source,data_source_target,0 ,batch_size, evaluation=True)
#
#    output, hidden, rnn_hs, dropped_rnn_hs = model(data,None,return_h=True)
#
#    total_loss                             = criterion(output,targets)
##    hidden                                 = repackage_hidden(hidden)
#    pred_y                                 = torch.max(output, 1)[1].data
#    accuracy                               = (pred_y == targets).float().sum() / len(targets)
#    
#    
#    wri.add_scalar('valraw_loss',total_loss,epoch)
#    wri.add_scalar('val_accuracy',accuracy,epoch)
#
##    if epoch==args.epochs:
##        scipy.io.savemat('{}_last_val_vis_{}'.format(args.save,args.data),{'output':pred_y.clone().cpu().data.numpy(),'targets':targets.clone().cpu().data.numpy(),
##                                                                     'vector':rnn_hs[0][-1,:,:].clone().cpu().data.numpy()})
##    if epoch % 50 == 0 and epoch != 0:
##        targets_image                      =image[targets]
##        wri.add_embedding(rnn_hs[0][-1,:,:].clone().cpu().data.numpy(),metadata=targets,label_img=targets_image.data,global_step=epoch/10)    
#
#    #我想把向量也可视化出来        
#    print('|   val_acc={:5.2f}   |   valraw_loss={:5.2f}  |'.format(accuracy,total_loss.item()),'\n')
#    
#    return total_loss.item(),accuracy#使用loss.item()可以从标量中获取Python数字
#
#'''#############################################
#获得 训练集 每幅图表征并保存，重建要用。用的方法和eval相似 而且要看看是否和测试集相似  不使用最终模型看看
##############################################'''
#def getvec(data_source,data_source_target,valx,valy,batch_size=128,name=''):
#    # Turn on evaluation mode which disables dropout.
#    model.eval()
#    if args.model == 'QRNN': model.reset()
#    
##    hidden = model.init_hidden(batch_size)
#    #train_set
#    Representation=np.zeros((len(data_source_target),64))
#    Representation_output=np.zeros(len(data_source_target))
#    
#    i=0
#    
#    while i < len(data_source_target):
#
#        data, targets = get_batchlz(data_source,data_source_target, i ,batch_size, evaluation=True)
#        output, hidden, rnn_hs, dropped_rnn_hs = model(data,None,return_h=True)
#        pred_y = torch.max(output, 1)[1].data
#        
#
#        Representation_output[i:i+batch_size]=pred_y.clone().cpu().data.numpy().astype('int32')
#        Representation[i:i+batch_size]=rnn_hs[0][-1,:,:].clone().cpu().data.numpy().astype('float32')
#        
#        i+=batch_size
#          
#    #val_set
#    val_data, val_targets                          = get_batchlz(valx,valy,0 ,len(valy), evaluation=True)
#    val_output, hidden, val_rnn_hs, dropped_rnn_hs = model(val_data,None,return_h=True)
#    val_pred_y                                 = torch.max(val_output, 1)[1].data
#    
#    scipy.io.savemat(name+'_vis',{'output':Representation_output,'targets':data_source_target.clone().cpu().data.numpy(),
#                     'vector':Representation,'val_output':val_pred_y.clone().cpu().data.numpy(),'val_targets':val_targets.clone().cpu().data.numpy(),
#                      'val_vector':val_rnn_hs[0][-1,:,:].clone().cpu().data.numpy().astype('float32')})
#    print('-----','save trainvec Down <(￣︶￣)↗[GO!]','--------------')    
#

#
'''###############################################################################
# Build the model 实例化模型
###############################################################################'''
model               = model.RNNModel(rnn_type=args.model, ninp=args.ninp, nhid=args.nhid, nlayers=args.nlayers, dropout=args.dropout, dropouth=args.dropouth, wdrop=args.wdrop, tie_weights=args.tied,class_num=args.class_num)
criterion           = nn.MSELoss()#回归训练
image               = mate_data()

if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr']                              = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout              = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout                    = args.wdrop

'''###将模型放到显卡上###'''
if args.cuda:
    model     = model.cuda()
    criterion = criterion.cuda()
   
'''###取模型所有参数（权重等）后面要用，打印部分网络参数###'''
params       = list(model.parameters()) #list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args,'\n')
print('Model total parameters:', total_params,'\n')
#
'''#################################################################
训练函数
###################################################################'''
#
lzarg                       =len(train_data[1]) // args.batch_size
def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss              = 0
    start_time              = time.time()
    hidden                  = model.init_hidden(args.batch_size)
    batch, i                = 0, 0
    while i < train_data.size(1) - 1 - 1:

#        seq_len=args.bptt

#        lr2                                    = optimizer.param_groups[0]['lr']
#        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()

        data, targets                          = get_imgbatchlz(train_data,train_img, i, args.batch_size)
        
        
        hidden                                 = repackage_hidden(hidden)
        optimizer.zero_grad()
        
        '''训练步'''
        output, hidden, rnn_hs, dropped_rnn_hs = model(data,None,return_h=True)
        raw_loss                               = criterion(output, targets)
        
        wri.add_scalar('raw_loss',raw_loss,epoch*lzarg+batch)
        

        loss                                   = raw_loss
        # Activiation Regularization
#        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
#        wri.add_scalar('alpha_loss',loss-raw_loss,epoch*lzarg+batch)
#        # Temporal Activation Regularization (slowness)
#        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
#        wri.add_scalar('alphabeta_loss',loss-raw_loss,epoch*lzarg+batch)
        
        loss.backward()
        
#        wri.add_scalar('loss',loss,epoch*179+batch)

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        
        optimizer.step()

        total_loss                       += raw_loss.data
#        optimizer.param_groups[0]['lr']   = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss                      = total_loss.item() / args.log_interval#log_interval平均loss
            wri.add_scalar('bat_curraw_loss',cur_loss,epoch*(lzarg//args.log_interval)+batch/args.log_interval)
            elapsed                       = time.time() - start_time
#            pred_y                        = torch.max(output, 1)[1].data
#            accuracy                      = (pred_y == targets).float().sum() / len(targets)
#            wri.add_scalar('train_accuracy',accuracy,epoch*(lzarg//args.log_interval)+batch/args.log_interval)
            print(' | epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    '平均loss {:5.2f} | '.format(
                epoch, batch, lzarg, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, ),'\n')
            total_loss = 0
            start_time = time.time()
        ###
        batch         += 1
        i             += args.batch_size
        
    '''# 保存梯度数据来可视化'''
#    if epoch % 10 == 0 and epoch != 0:
##        targets_image=image[targets]
##        wri.add_embedding(rnn_hs[0][-1,:,:].clone().cpu().data.numpy(),metadata=targets,label_img=targets_image.data,global_step=epoch/10)
#        for name, param in model.named_parameters():
#            wri.add_histogram(name, param.clone().cpu().data.numpy(), epoch/10)

#
'''###################################################################
训练逻辑
###################################################################'''
# Loop over epochs.
lr             = args.lr
best_val_loss  = []
stored_loss    = 10000
stored_acc     = 1

'''# At any point you can hit Ctrl + C to break out of training early.主要训练逻辑代码'''
try:
    print(model,'\n')
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer    == 'sgd':
        optimizer        = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer    == 'adam':
        optimizer        = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        
#        ######'''多次留出验证'''######
#        if args.one_fold:
#            one_fold_evaluate(val_data2,val_label2, batch_size=len(val_label2))
    

#        ######用ASGD训练#######
        if 't0' in optimizer.param_groups[0]:
               pass
#            tmp            = {}
#            for prm in model.parameters():
#                tmp[prm]   = prm.data.clone()
#                prm.data   = optimizer.state[prm]['ax'].clone()
#
#            val_loss2,val_acc2= evaluate(val_data,val_label,eval_batch_size)
#
#            if val_loss2 < stored_loss:
#                model_save('{}_ASGDminLOSS{:.2f}.e{}'.format(args.save,val_loss2,epoch))
#                print('Saving Averaged!')
#                stored_loss   = val_loss2
#                
#            if val_acc2 > stored_acc:
#                model_save('{}_ASGDmaxACC{:.2f}.e{}'.format(args.save,val_acc2,epoch))
#                print('Saving Averaged!')
#                stored_acc    = val_acc2
#
#            for prm in model.parameters():
#                prm.data      = tmp[prm].clone()
#                
#            if epoch in args.when:
#                print('Saving model before learning rate decreased')
#                model_save('{}_ASGDLr{}.e{}'.format(args.save, optimizer.param_groups[0]['lr'],epoch))
#                print('Dividing learning rate by 10')
#                optimizer.param_groups[0]['lr'] /= 10.
        
        ########用Adma训练########
        else:
            val_loss,val_acc = evaluate(val_data,val_label, eval_batch_size) 
            
#            if epoch <= len(lrr):
#                optimizer.param_groups[0]['lr']=lrr[epoch-1]

            if args.optimizer == 'adam':
                if val_loss < stored_loss and epoch > 50:
                    model_save('{}_ADAM_loss={:.2f}_epoch={}_model'.format(args.save,val_loss,epoch))
                    model_save('{}_ADAM_loss={:.2f}_epoch={}_para'.format(args.save,val_loss,epoch),0,1)
                    print('Saving model ADAMminLOSS')
#                    getvec(train_data,train_label,val_data,val_label,batch_size=128,name='{}_loss={:.2f}_acc={:.2f}_epoch={}'.format(args.save,val_loss,val_acc,epoch))
                    stored_loss = val_loss
                    
                if val_acc > stored_acc and epoch > 50:
                    model_save('{}_ADAM_ACC={:.2f}_loss={:.2f}_epoch={}_model'.format(args.save,val_acc,val_loss,epoch))
                    model_save('{}_ADAM_ACC={:.2f}_loss={:.2f}_epoch={}_para'.format(args.save,val_acc,val_loss,epoch),0,1)
#                    getvec(train_data,train_label,val_data,val_label,batch_size=128,name='{}_ACC={:.2f}_loss={:.2f}_epoch={}'.format(args.save,val_acc,val_loss,epoch))
                    print('Saving Averaged ADAMmaxACC!')
                    
                    stored_acc = val_acc
    
    
                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}_ADAMLr={}_loss={:.2f}_epoch={}_model'.format(args.save, optimizer.param_groups[0]['lr'],val_loss,epoch))
                    model_save('{}_ADAMLr={}_loss={:.2f}_epoch={}_para'.format(args.save, optimizer.param_groups[0]['lr'],val_loss,epoch),0,1)
                    print('Dividing learning rate by 5')
#                    optimizer.param_groups[0]['lr'] /= 10.
#                    optimizer.param_groups[0]['lr'] *= 10
                    optimizer.param_groups[0]['lr'] = 0.001
                
#            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
#                print('Switching to ASGD')
#                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
#

            best_val_loss.append(val_loss)
            
    wri.close()
    model_save('{}_lastModel_model'.format(args.save))
    model_save('{}_lastModel_para'.format(args.save),0,1)

     ###测试所有数据####
#    evaluate_all_data(train_data,train_label, batch_size=2000)
    
#
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    
#
#
##    if epoch==args.epochs:
##        scipy.io.savemat('C:/Users/night/Desktop/visualization.mat',{'output':pred_y.clone().cpu().data.numpy(),'targets':targets.clone().cpu().data.numpy(),
##                                                                     'vector':rnn_hs[0][-1,:,:].clone().cpu().data.numpy()})
##    if epoch % 10 == 0 and epoch != 0:
##        targets_image=image[targets]
##        wri.add_embedding(rnn_hs[0][-1,:,:].clone().cpu().data.numpy(),metadata=targets,label_img=targets_image.data,global_step=epoch/10)    
