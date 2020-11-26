# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1

    return matrix

na = ['nsubj','compound','nsubjpass','nmod']
adj = ['amod','acomp']
adv = ['advmod','advcl']
v = ['prt','dobj']



def dependency_adj_matrix_mul(text):
   # https://spacy.io/docs/usage/processing-text
   document = nlp(text)
   seq_len = len(text.split())
   matrix = []
   matrix1 = np.zeros((seq_len, seq_len)).astype('float32')
   matrix2 = np.zeros((seq_len, seq_len)).astype('float32')
   matrix3 = np.zeros((seq_len, seq_len)).astype('float32')
   matrix4 = np.zeros((seq_len, seq_len)).astype('float32')
   matrix5 = np.zeros((seq_len, seq_len)).astype('float32')
   nnum, anum, adnum, vnum, pnum, other = 0,0,0,0,0,0
   for token in document:#对于每一个中心词
       if token.i < seq_len:
           #matrix[token.i][token.i] = 1
           # https://spacy.io/docs/api/token
           for child in token.children:#对于它的每个孩子
               if child.i < seq_len:
                   #newre = child.dep_
                   # if newre not in relation.keys():
                   #     relation[newre] = 1
                   # else:
                   #     relation[newre] += 1
                   #如果两个词的关系属于以上关系，
                   if child.dep_ in na:
                       matrix1[token.i][child.i] = 1
                       matrix1[child.i][token.i] = 1
                   elif child.dep_ in adj:
                        matrix2[token.i][child.i] = 1
                        matrix2[child.i][token.i] = 1
                   elif child.dep_ in adv:
                        matrix3[token.i][child.i] = 1
                        matrix3[child.i][token.i] = 1
                   elif child.dep_ in v:
                        matrix4[token.i][child.i] = 1
                        matrix4[child.i][token.i] = 1
                   else:
                        matrix5[token.i][child.i] = 1
                        matrix5[child.i][token.i] = 1
                        matrix5[token.i][token.i] = 1
   matrix.append(matrix1)
   matrix.append(matrix2)
   matrix.append(matrix3)
   matrix.append(matrix4)
   matrix.append(matrix5)
   return matrix


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'_hira'+'.graph', 'wb')
    for i in range(0, len(lines), 3):
        #d对于每句话，会有好几个邻接矩阵，全部放在一个列表里
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        graph_hira = dependency_adj_matrix_mul(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = graph_hira
    pickle.dump(idx2graph, fout)        
    fout.close() 

if __name__ == '__main__':
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')