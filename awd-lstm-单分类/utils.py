import torch
import numpy as np


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def get_batchlz(xdata,ydata,i, batch_size,evaluation=False):
    data = xdata[:,i:i+batch_size,:]
    target = ydata[i:i+batch_size].view(-1)
    return data, target

def get_imgbatchlz(xdata,img_data,i, batch_size,evaluation=False):
    data = xdata[:,i:i+batch_size,:]
    target = img_data[i:i+batch_size]
    return data, target
'''######################################################################        
#生成图片，这里因为显示的原因选择了实际图像的补
###################################################################'''
def mate_data(to_torch=1):
    shizi               = np.array([[1,1,0,0,1,1],[1,1,0,0,1,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,0,0,1,1],[1,1,0,0,1,1]])#十字
    zhongyang           = np.array([[1,1,1,1,1,1],[1,0,0,0,0,1],[1,0,0,0,0,1],[1,0,0,0,0,1],[1,0,0,0,0,1],[1,1,1,1,1,1]])#中央
    sizhou              = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,1,1,0,0],[0,0,1,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])#四周
    xiegang             = np.array([[0,0,1,1,1,1],[0,0,0,1,1,1],[1,0,0,0,1,1],[1,1,0,0,0,1],[1,1,1,0,0,0],[1,1,1,1,0,0]])#斜杠
    sidian              = np.array([[1,1,0,0,1,1],[1,1,0,0,1,1],[0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,0,0,1,1],[1,1,0,0,1,1]])#周围4点
    shangxia            = np.array([[0,0,1,1,0,0],[0,0,1,1,0,0],[1,1,0,0,1,1],[1,1,0,0,1,1],[0,0,1,1,0,0],[0,0,1,1,0,0]])
    fanxie              = np.array([[1,1,1,1,0,0],[1,1,1,0,0,0],[0,1,1,1,0,0],[0,0,1,1,1,0],[0,0,0,1,1,1],[0,0,0,0,1,1]])
    space               = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
    image               = [[zhongyang],[sizhou],[shizi],[xiegang],[sidian],[shangxia],[fanxie],[space]]
    if to_torch:
        image               = np.array([[zhongyang],[sizhou],[shizi],[xiegang],[sidian],[shangxia],[fanxie],[space]])
        image               = torch.from_numpy(image).float()
    return image




















