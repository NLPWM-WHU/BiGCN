# -*- coding: utf-8 -*-
# file: attention.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:词向量的维度
        :param hidden_dim:valued_key的维度，如果没有给出，等于embed_dim // n_head
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head  # 向下取整除
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            # nn.Parameter：A kind of Tensor that is to be considered a module parameter
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)  # register_parameter：Adds a parameter to the module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    # k(?, k_len, emb_dim)
    # q(?, emb_dim)
    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing->q(?, q_len=1, emb_dim)
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]  # k_len
        q_len = q.shape[1]  # q_len=1

        # kx(?, k_len, n_head*hidden_dim) = k(?, k_len, emb_dim) * w_k(emb_dim, n_head*hidden_dim)
        # kx(?, k_len, n_head, hidden_dim)
        # kx(?, k_len, n_head, hidden_dim) -> (n_head*?, k_len, hidden_dim)
        # tensor.permute-维度重排列
        # contiguous:view只能用在contiguous的tensor上,如果在view之前用了transpose,permute等,需要用contiguous()来返回一个contiguous copy
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)

        # qx(?, q_len, n_head*hidden_dim) = q(?, q_len, emb_dim) * w_q(emb_dim, n_head*hidden_dim)
        # qx(?, q_len, n_head, hidden_dim)
        # qx(?, q_len, n_head, hidden_dim) -> (n_head*?, q_len, hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)

        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)  # kt(n_head*?, hidden_dim, k_len)
            # score(n_head*?, q_len, k_len) = qx(n_head*?, q_len, hidden_dim) * kt(n_head*?, hidden_dim, k_len)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            # kx(n_head*?, k_len, hidden_dim)->kxx(n_head*?, q_len, k_len, hidden_dim)
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            # qx(n_head*?, q_len, hidden_dim)->qxx(n_head*?, q_len, k_len, hidden_dim)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            # kq(n_head*?, q_len, k_len, hidden_dim*2)
            kq = torch.cat((kxx, qxx), dim=-1)
            # score(n_head*?, q_len, k_len) = kq(n_head*?, q_len, k_len, hidden_dim*2) * self.weight(hidden_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            # qw(n_head*?, q_len, hidden_dim) = qx(n_head*?, q_len, hidden_dim) * self.weight(hidden_dim, hidden_dim)
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)  # kt(n_head?*, hidden_dim, k_len)
            score = torch.bmm(qw, kt)  # score(n_head*?, q_len, k_len)
        else:
            raise RuntimeError('invalid score_function')
        # score(n_head*?, q_len, k_len)
        score = F.softmax(score, dim=-1)
        # out_put(n_head*?, q_len, hidden_dim) = score(n_head*?, q_len, k_len) * kx(n_head*?, k_len, hidden_dim)
        output = torch.bmm(score, kx)
        # output(?, q_len, n_head*hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)
        # output(?, q_len, out_dim) = output(?, q_len, n_head*hidden_dim) * proj(n_head*hidden_dim, out_dim)
        # out_dim = emb_dim
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''

    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1,
                 dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))  # q-(q_len, emb_dim)
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]  # mb_size-128 k-(batch_size, k_len, emb_dim+aspect_dim)
        q = self.q.expand(mb_size, -1, -1)  # (batch_size, q_len, emb_dim+aspect_dim);expand:将原有的维度复制，形成新的维度
        return super(NoQueryAttention, self).forward(k, q)