# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch



def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32') 
            except:
                continue
    return word_vec



def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

def build_fre_matrix(feature_words_path,embed_dim,type):
    embedding_matrix_file_name = '{0}_embedding_frematrix.pkl'.format(type)
    fin = open(feature_words_path, 'rb')
    feature_words = pickle.load(fin)
    fin.close()
    if os.path.exists(embedding_matrix_file_name):
        print('loading embeddingfre_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(feature_words), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

        word_vec = {}
        for line in fin:
            tokens = line.rstrip().split()
            if feature_words is None or tokens[0] in feature_words:
                try:
                    word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
                except:
                    continue
        i = 0
        for word in feature_words:
            vec = word_vec.get(word)
            if vec is not None:

                embedding_matrix[i] = vec
                i = i + 1
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))

    return embedding_matrix

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                #每个word一个id
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words] #根据word2idx重组每句话
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer,post_vocab):
        #
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        fin = open(fname+'_hira.graph', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()

        fin = open(fname+'single_hira.graph', 'rb')
        fre_graphs = pickle.load(fin)
        fin.close()


        fin = open(fname + '.graph', 'rb')
        idx2gragh_no = pickle.load(fin)
        fin.close()

        fin = open(fname + 'single.graph', 'rb')
        fre_graphs_no = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            sentence = text_left+' '+aspect+' '+text_right
            sen_len = len(sentence.split())
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)

            aspect_len = len(aspect.split())
            left_len = len(text_left.split())
            right_len = sen_len - aspect_len - left_len

            position = list(range(-left_len,0)) + [0]*aspect_len + list(range(1,right_len + 1))
            post_emb = [post_vocab.stoi.get(t, post_vocab.unk_index) for t in position]

            polarity = int(polarity)+1
            dependency_graph = idx2gragh[i]
            fre_graph = fre_graphs[i]

            dependency_graph_no = idx2gragh_no[i]
            fre_graph_no = fre_graphs_no[i//3]
            data = {
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'fre_graph':fre_graph,
                'dependency_graph_no': dependency_graph_no,
                'fre_graph_no': fre_graph_no,
                'post_emb':post_emb

            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=300,post_vocab=None):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },

        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']]) #所有的text

        if os.path.exists(dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text) 
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)

        fre_embedding = self.embedding_matrix
        self.fre_embedding = torch.tensor(fre_embedding,dtype=torch.float)
        fin = open(fname[dataset]['train'] + 'fre_full_all' + '.graph', 'rb')
        common_adj = pickle.load(fin)
        len_1 = len(tokenizer.word2idx)

        row = len(common_adj[0])
        diff = len_1 - row
        fin.close()
        cp_row = np.zeros((diff, row))
        common_adj = np.insert(common_adj,0,values=cp_row,axis=0)

        cp_col = np.zeros((diff,len_1))
        common_adj = np.insert(common_adj,0,values=cp_col,axis=1)

        self.common_adj = torch.tensor(common_adj, dtype=torch.float)


        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer,post_vocab))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer,post_vocab))
    
