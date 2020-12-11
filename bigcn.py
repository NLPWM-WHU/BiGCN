# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from torch.autograd import Variable
import constant
import torch_utils
import numpy as np
import scipy.sparse as sp

#普通GCN的卷积
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphConvolutionFRE(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionFRE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        deom = adj > 0.5
        deom = Variable(deom.float())
        denom = torch.sum(deom, dim=1, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphConvolutionRE(nn.Module):
    def __init__(self, in_features, out_features,rela_len, bias=True,frc_lin=False):
        super(GraphConvolutionRE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rela_len = rela_len
        self.weight = nn.Parameter(torch.FloatTensor(in_features,out_features//rela_len))
        self.frc_lin = frc_lin
        if self.frc_lin == True:
           self.fre_line = nn.Linear((out_features//rela_len)*rela_len, out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj_re):
        hidden = torch.matmul(text, self.weight)
        adj_re = adj_re.permute(1,0,2,3)
        denom1 = torch.sum(adj_re[0], dim=2, keepdim=True) + 1
        output = torch.matmul(adj_re[0], hidden) / denom1
        output = F.softmax(output,dim=-1)

        for i in range(1,self.rela_len):
            denom2= torch.sum(adj_re[i], dim=2, keepdim=True) + 1
            output2 = torch.matmul(adj_re[i], hidden) / denom2
            output2 = F.relu(output2)
            output = torch.cat([output,output2],dim=2)


        if self.frc_lin:
            output = self.fre_line(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class BiGCN(nn.Module):
    def __init__(self, embedding_matrix,common_adj,fre_embedding,post_vocab, opt):
        super(BiGCN, self).__init__()
        self.opt = opt
        D = opt.embed_dim
        Co = opt.kernel_num
        self.num_layer = opt.num_layer
        self.post_vocab = post_vocab
        self.post_size = len(post_vocab)
        self.common_adj = common_adj
        self.fre_embedding = fre_embedding
        self.hidden_dim = opt.hidden_dim
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.post_embed = nn.Embedding( self.post_size, opt.post_dim, padding_idx=constant.PAD_ID) if opt.post_dim > 0 else None
        self.text_lstm = DynamicLSTM(opt.embed_dim+opt.post_dim, opt.hidden_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolutionRE(opt.hidden_dim, opt.hidden_dim,5,frc_lin=True)
        self.gc2 = GraphConvolutionRE(opt.hidden_dim, opt.hidden_dim,5,frc_lin=True)
        self.gc3 = GraphConvolutionRE(opt.hidden_dim, opt.hidden_dim,8,frc_lin=True)
        self.gc4 = GraphConvolutionRE(opt.hidden_dim, opt.hidden_dim,8,frc_lin=True)
        self.gc5 = GraphConvolutionFRE(opt.hidden_dim, opt.hidden_dim)
        self.gc6 = GraphConvolutionFRE(opt.hidden_dim, opt.hidden_dim)
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K - 2) for K in [3]])
        self.fc_aspect = nn.Linear(128, 2*D)
        self.att_line = nn.Linear(opt.hidden_dim,2*opt.hidden_dim)

        self.weight = nn.Parameter(torch.FloatTensor(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.bias = nn.Parameter(torch.FloatTensor(2 * self.hidden_dim))

    def pre_frequency(self):
        em = F.relu(F.softmax(self.gc5(self.fre_embedding, self.common_adj),dim=-1))
        pre_em = F.relu(F.softmax(self.gc6(em, self.common_adj),dim=-1))
        return pre_em

    def cross_network(self,f0,fn):
        fn_weight = torch.matmul(fn,self.weight)
        fl = f0*fn_weight + self.bias + f0
        x = fl[:,:,0:self.hidden_dim]
        y = fl[:,:,self.hidden_dim:]
        return x,y


    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj, adj2,_,_, post_emb = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = [self.embed(text_indices)]
        if self.opt.post_dim > 0:
            text +=[self.post_embed(post_emb)]
        text = torch.cat(text,dim=2)
        text = self.text_embed_dropout(text)
        text_out, (_, _) = self.text_lstm(text, text_len)

        text_fre_embedding = self.pre_frequency()
        text_fre = nn.Embedding.from_pretrained(text_fre_embedding)
        text_out_fre = text_fre(text_indices)
        text_out_fre = self.text_embed_dropout(text_out_fre)

        f0 = torch.cat([text_out_fre,text_out],dim=2) #x:fre  y:syn
        numlayer = self.num_layer
        f_n = f0
        for i in range(numlayer):
            if i == 0:
                x, y = self.cross_network(f0,f0)
                x = F.softmax(self.gc3(self.position_weight(x,aspect_double_idx, text_len, aspect_len), adj2),dim=-1)
                y = F.softmax(self.gc1(self.position_weight(y,aspect_double_idx, text_len, aspect_len), adj),dim=-1)
                f_n = torch.cat([x,y],dim=2)
            else:#多层的更新
                x,y = self.cross_network(f0,f_n)
                x = F.softmax(self.gc4(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj2),dim=-1)
                y = F.softmax(self.gc2(self.position_weight(y, aspect_double_idx, text_len, aspect_len), adj),dim=-1)
                f_n = torch.cat([x,y],dim=2)

        aspect = self.embed(aspect_indices)
        aa = [F.relu(conv(aspect.transpose(1, 2))) for conv in self.convs3]
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)
        temp = self.fc_aspect(aspect_v).unsqueeze(2)
        aa2 = F.tanh(f_n + temp.transpose(1, 2))
        xaa = f_n * aa2
        xaa2 = xaa.transpose(1, 2)
        xaa2 = F.max_pool1d(xaa2, xaa2.size(2)).squeeze(2)

        f_n_mask = self.mask(f_n, aspect_double_idx)

        text_out = self.att_line(text_out)
        alpha_mat = torch.matmul(f_n_mask, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1)
        x = F.relu(x + xaa2)
        output = self.fc(x)
        return output