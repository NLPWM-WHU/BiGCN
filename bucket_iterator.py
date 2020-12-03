# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_fre_graph = []
        batch_dependency_graph_no = []
        batch_fre_graph_no = []

        batch_post_emb = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            text_indices, context_indices, aspect_indices, left_indices, polarity, dependency_graph,fre_graph,dependency_graph_no,fre_graph_no,post_emb = \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph'],item['fre_graph'],item['dependency_graph_no'],item['fre_graph_no'],item['post_emb']
            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            post_padding = [0] *(max_len - len(post_emb))
            batch_text_indices.append(text_indices + text_padding)
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_polarity.append(polarity)
            dere_num = 5
            fere_num = 8
            redependency_graph = []
            refre_graph = []
            for i in range(dere_num):
                adj = numpy.pad(dependency_graph[i], \
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant')
                redependency_graph.append(adj)
            for i in range(fere_num):
                adj = numpy.pad(fre_graph[i], \
                ((0,max_len-len(text_indices)),(0,max_len-len(text_indices))), 'constant')
                refre_graph.append(adj)
            batch_dependency_graph.append(redependency_graph)
            batch_fre_graph.append(refre_graph)
            batch_dependency_graph_no.append(numpy.pad(dependency_graph_no,((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),
                                                    'constant'))
            batch_fre_graph_no.append(numpy.pad(fre_graph_no, ((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),
                                             'constant'))
            batch_post_emb.append(post_emb+post_padding)

        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                'context_indices': torch.tensor(batch_context_indices), \
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'polarity': torch.tensor(batch_polarity), \
                'dependency_graph': torch.tensor(batch_dependency_graph),
                'fre_graph': torch.tensor(batch_fre_graph),
                'dependency_graph_no': torch.tensor(batch_dependency_graph_no),
                'fre_graph_no': torch.tensor(batch_fre_graph_no),
                'post_emb':torch.tensor(batch_post_emb)
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
