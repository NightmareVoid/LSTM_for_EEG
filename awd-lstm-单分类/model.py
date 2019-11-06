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

import torch
import torch.nn as nn

#from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
#('--dropout'help='dropout applied to layers (0 = no dropout)')
#('--dropouth', help='dropout for rnn layers (0 = no dropout)')
#('--dropouti',help='dropout for input embedding layers (0 = no dropout)')
#('--dropoute',help='dropout to remove words from embedding layer (0 = no dropout)')
#    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
#wdrop',help='amount of weight dropout to apply to the RNN hidden to hidden matrix')   
    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5,wdrop=0, tie_weights=False, class_num=4):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
#        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)

        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
#############################是否使用BN层，因为LRP没法穿过BN层##############
#        self.bn=nn.BatchNorm1d(nhid, momentum=0.5)
######################################################################### 
#############################是否使用bias，bias的使用会对LRP产生影响########               
#        self.decoder = nn.Linear(nhid,class_num,bias=0)
        self.decoder = nn.Linear(nhid, class_num)
#########################################################################
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
#        if tie_weights:
#            #if nhid != ninp:
#            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
#            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouth = dropouth
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1

        
#        self.decoder.bias.data.fill_(0)  ##
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input,hidden, return_h=False):#(self, input, hidden, return_h=False):
        raw_output =input 
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []#保存多层的经过lockdrop输出当只有一层时outputs【0】就是最后输出
        for l, rnn in enumerate(self.rnns):
#            current_input = raw_output
#            print(raw_output.shape)
            raw_output, new_h = rnn(raw_output)
#            print(raw_output.shape)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
#        print('output',output.shape)
        
        '''
        Lz
        '''
#        BNoutput=self.bn(output[-1,:,:])        
#        lzoutput=self.decoder(BNoutput)
        
        lzoutput=self.decoder(output[-1,:,:])

        '''
        Lz
        '''

        if return_h:
            return lzoutput, hidden, raw_outputs, outputs#hidden用来初始化下一次batch的
        return lzoutput #result, hidden

    def init_hidden(self, bsz):#目的也是为了分离
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
