import torch
import torch.nn as nn
import torch.nn.functional as F 
from modules.conv_tbc import ConvTBC 
from data_iterator import dataIterator
import torch.utils.data as data
import numpy as np 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1):
    pad = (kernel_size - 1) * dilation
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)


class CasualConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CausalConv1d(256, 256, kernel_size=3, dilation=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x[:, :, :-self.conv1.padding[0]]  # remove trailing padding
        return x

# class AttentionDecoder(nn.Module):
# 	def __init__(self, hidden_size, output_size, encoder_output):
# 		super(AttentionDecoder, self).__init__()
# 		self.output_size = output_size
# 		self.hidden_size = hidden_size
# 		self.embedding = nn.Embedding(self.output_size, 256)
# 		self.W = nn.Linear(256,1)
#         self.casual = CasualConv(in_channels=256, out_channels=256, kernel_size=3)
#         self.fc = nn.Linear(256, 1)
#         self.fc2 = nn.Linear(256, output_size)
# 		pass 

# 	def forward(self, encoder_output, ):
# 		pass

# 	def initHidden(self):
# 		pass 


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, in_channels, out_channels, kernel_size, padding, num_blocks=3, bmm=None, batch_size=2):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.W = nn.Linear(int(out_channels/2), embedding_dim)
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.conv_tbc = ConvTBC(
                            in_channels, 
                            out_channels, 
                            kernel_size, 
                            padding=padding
                            )
        self.bmm = bmm if bmm is not None else torch.bmm
        self.W_o = nn.Linear(embedding_dim, vocab_size) 
        pass 

    def forward(self, encoder_output, decoder_input, labels):
        # labels = labels.unsqueeze(0) #(1, 6, 10)
        print("Input size:", labels.size())
        print("Encoder output size:", encoder_output.size())
        H,W = encoder_output.size()[1:3]
        s = self.embedding(labels) #labels: (batch_size, samples ,length)
                                   #s: (batch_size, samples, length, embeding_dim)
        s = s.squeeze() #s: (batch_size, length, embbeding_dim)
        a_l = s
        s = torch.transpose(s, 0, 1) #s: (length, batch_size, embedding_dim)
        for i in range(self.num_blocks):
            a_l = torch.transpose(a_l, 0, 1) #s: (length, batch_size, embedding_dim)
            s_conv = self.conv_tbc(a_l) #(length, batch_size, out_channels)
            z_l = F.glu(s_conv, dim=2) #(length, batch_size, out_channels/2)
            # print(self.W(z_l).size(), s.size())
            h_l = self.W(z_l) + s #(length, batch_size, embedding_dim)
            h_l = torch.transpose(h_l, 0, 1) #(batch_size, length, embedding) 
            #encoder_output: (batch_size, H, W, embedding_dim)
            trans_decoder_input = decoder_input.view(self.batch_size, H*W, embedding_dim)
            trans_decoder_input = torch.transpose(trans_decoder_input, 1,2) #(batch_size, embedding_dim, HxW)
            h_l = h_l.float()
            trans_decoder_input = trans_decoder_input.float()
            h_f = self.bmm(h_l, trans_decoder_input) #(batch_size, length, HxW)
            exp_h_f = torch.exp(h_f)
            alpha = exp_h_f/torch.sum(exp_h_f, 2).unsqueeze(2).repeat(1,1,H*W) #alpha: (batch_size, length, HxW)
            alpha = alpha.unsqueeze(3).repeat(1,1,1, embedding_dim) #(batch_size, length, HxW, embedding_dim)
            residual = encoder_output + decoder_input #(batch_size, H, W, embedding_dim)
            residual = residual.view(batch_size, H*W, embedding_dim)
            residual = residual.unsqueeze(1).float()
            c_l = torch.sum(alpha * residual, 2)   
            z_l = torch.transpose(z_l, 0, 1)
            a_l = c_l + z_l

        p_t = self.W_o(a_l) 
        output = F.log_softmax(p_t, dim=2)
        return output, output
              
def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])
    print('total words/phones',len(lexicon))
    return lexicon


class custom_dset(data.Dataset):
    def __init__(self,train,train_label,batch_size):
        self.train = train
        self.train_label = train_label
        self.batch_size = batch_size

    def __getitem__(self, index):
        train_setting = torch.from_numpy(numpy.array(self.train[index]))
        label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(torch.LongTensor)

        size = train_setting.size()

        # print("size: ", size)
        train_setting = train_setting.view(1,size[2],size[3])
        # print("train set: ", train_setting.size())
        train_print = torch.rand(train_setting.size())
        # print("train print: ", train_print.size())
        label_setting = label_setting.view(-1)
        return train_setting,label_setting, train_print

    def __len__(self):
        return len(self.train)


if  __name__ == "__main__":
    # datasets=['./offline-train.pkl','./train_caption.txt']
    # valid_datasets=['./offline-test.pkl', './test_caption.txt']
    # dictionaries=['./dictionary.txt']
    # batch_Imagesize=500000
    # valid_batch_Imagesize=500000
    # # batch_size for training and testing
    # batch_size=1
    # batch_size_t=1
    # # the max (label length/Image size) in training and testing
    # # you can change 'maxlen','maxImagesize' by the size of your GPU
    # maxlen=48
    # maxImagesize= 100000
    # # hidden_size in RNN
    # hidden_size = 256
    # # teacher_forcing_ratio 
    # teacher_forcing_ratio = 1
    # # change the gpu id 
    # gpu = [0]
    # # learning rate
    # lr_rate = 0.0001
    # # flag to remember when to change the learning rate
    # flag = 0
    # # exprate
    # exprate = 0
    # worddicts = load_dict(dictionaries[0])
    # worddicts_r = [None] * len(worddicts)
    # for kk, vv in worddicts.items():
    #     worddicts_r[vv] = kk

    # #load train data and test data
    # train,train_label = dataIterator(
    #                                     datasets[0], datasets[1],worddicts,batch_size=1,
    #                                     batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize
    #                                 )

    # test,test_label = dataIterator(
    #                                     valid_datasets[0],valid_datasets[1],worddicts,batch_size=1,
    #                                     batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize
    #                                 )

    # len_test = len(test)
    # len_train = len(train)
    # off_image_train = custom_dset(train,train_label,batch_size)
    # off_image_test = custom_dset(test,test_label,batch_size)
    # pass
    embedding_dim = 2
    batch_size = 2
    y = torch.LongTensor([[[1,2,3,4, 5], [5,6,7,8, 9]]])

    e = torch.LongTensor([[[[1, 2], [3, 3], [4,4]], [[1, 2], [3, 3], [4,4]], [[1, 2], [3, 3], [4,4]]], [[[1, 2], [3, 3], [4,4]], [[1, 2], [3, 3], [4,4]], [[1, 2], [3, 3], [4,4]]]])
    f = torch.LongTensor([[[[1, 2], [3, 3], [4,4]], [[1, 2], [3, 3], [4,4]], [[1, 2], [3, 3], [4,4]]], [[[1, 2], [3, 3], [4,4]], [[1, 2], [3, 3], [4,4]], [[1, 2], [3, 3], [4,4]]]])

    decoder_block = Decoder(112,embedding_dim, 2, 4, 3, 1, num_blocks=3, batch_size=2)
    out = decoder_block(e, f, y)


        