import numpy as np
import pickle
from nltk.corpus import stopwords

import os
import collections
import numpy as np
from collections import Counter

class Coword_matrix(object):
    def __init__(self,Fulltext_cut_content,Full_Feature_word, Full_Feature_weight):
        self.Fulltext_cut_content=Fulltext_cut_content
        self.Full_Feature_word = Full_Feature_word
        self.Full_Feature_weight= Full_Feature_weight

        self.Common_matrix = np.empty((len(self.Full_Feature_word), len(self.Full_Feature_word)))
        for i in range(len(self.Full_Feature_word)):
            self.Common_matrix[i][i] = int(self.Full_Feature_weight[i])
            print(self.Common_matrix)
            for n in range(1, len(self.Full_Feature_word) - i):
                #构建两个词之间的频率
                word1 = self.Full_Feature_word[i]
                word2 = self.Full_Feature_word[i + n]
                Common_weight = 0
                for Single_Text_Cut in self.Fulltext_cut_content:
                #遍历每一句话中的所有句子，如果word1和word2在同一片句子中出现，则该两词的权重+1
                    if ((word1 in Single_Text_Cut) and (word2 in Single_Text_Cut)):
                        Common_weight += 1
                self.Common_matrix[i][i + n] = Common_weight
                self.Common_matrix[i + n][i] = Common_weight

    def get_Full_Feature_word(self):
        return self.Full_Feature_word

    def get_Common_matrix(self):
        return self.Common_matrix

    def return_word_row(self,word):
        if word not in self.Full_Feature_word:
            return 0,-1
        else:
            for row in range(len(self.Full_Feature_word)):
                if word == self.Full_Feature_word[row]:
                    return self.Common_matrix[row],row

def stopword():
    #得到所有的停用词
    stop_words = stopwords.words('english')
    for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's']:
        stop_words.append(w)
    return stop_words

# def frequencu_adj_matrix(text,common_adj,stop_words):
#     seq_len = len(text.split())
#     matrix = np.zeros((seq_len, seq_len)).astype('float32')
#     for i in range(seq_len-1):
#         word1 = text[i]
#         word2 = text[i+1]
#
#         print()
#         #如果word是stopwords的话，其和其他词的值就为0   剩下的值为该两个词出现的频率
#
#
#
#
#
#
#     print()

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.fregraph', 'wb')
    stop_words = stopword()
    all_sentence = []
    re_sentence = []
    Full_familiar_Feature = {}
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$") ]
        aspect = lines[i + 1].lower().strip()
        sentence = text_left+' '+aspect+' '+text_right
        re_sentence.append(sentence)
        sentence = [s for s in sentence.split() if s not in stop_words]
        all_sentence.append(sentence) #所有的句子存储在all_sentence中
        Single_text_feature_sort_dict = collections.Counter(sentence)  # 词频统计

        Full_familiar_Feature = dict(Counter(dict(Single_text_feature_sort_dict)) + Counter(
            dict(Full_familiar_Feature)))
    Full_Feature_word, Full_Feature_weight = list(Full_familiar_Feature.keys()), list( Full_familiar_Feature.values())

        # adj_matrix = frequencu_adj_matrix(text_left+' '+aspect+' '+text_right) #得到词汇共现矩阵
        # idx2graph[i] = adj_matrix
    # co_words = Coword_matrix(all_sentence,Full_Feature_word, Full_Feature_weight)
    # common_adj = co_words.get_Common_matrix()


    return Full_Feature_word

#Get the small word frequency map of each instance according to the stored global word frequency map
def process_single(fname):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()

    fin = open(fname + 'fre' + '.graph', 'rb')  # 使用之前构造的全部词频矩阵
    common_adj = pickle.load(fin)
    fin.close()
    idx2graph = {}
    stop_words = stopword()
    all_sentence = []#只有特征词的句子
    re_sentence = []#原始句子
    Full_familiar_Feature = {}
    for i in range(0, len(lines), 3):#所有的句子
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$") ] #去除停用词和标点
        aspect = lines[i + 1].lower().strip()
        sentence = text_left+' '+aspect+' '+text_right
        re_sentence.append(sentence) #原始句子
        sentence = [s for s in sentence.split() if s not in stop_words]
        all_sentence.append(sentence)
        Single_text_feature_sort_dict = collections.Counter(sentence)  # 词频统计
        Full_familiar_Feature = dict(Counter(dict(Single_text_feature_sort_dict)) + Counter(dict(Full_familiar_Feature)))  # 化为Counter后 作并集，再将结果化为dict

    Full_Feature_word= list(Full_familiar_Feature.keys()) #所有的特征词
    feat_dic = {}
    for i in range(len(Full_Feature_word)):
        feat_dic[Full_Feature_word[i]] = i #key：单词 value：位置
    #所有的特征词在Full_Feature_word中
    for i in range(len(re_sentence)):  # 得到所有的句子
        sentence = re_sentence[i] #对每一个句子
        sentence_split = sentence.split()
        seq_len = len(sentence_split) #一个句子中所有的单词
        adj_matrix = np.zeros((seq_len, seq_len)).astype('float32') #构建词典
        for j in range(seq_len - 1):
            #对每个单词
            word1 = sentence_split[j]
            if word1 in feat_dic.keys():
                 index1 = feat_dic.get(word1)#得到当前位置
                 adj_matrix[j][j] = common_adj[index1][index1] #对角线位置为该单词出现频率
                 for k in range(j + 1, seq_len):
                    word2 = sentence_split[k]
                    if word2 in feat_dic.keys():
                        index2 = feat_dic.get(word2)
                        adj_matrix[j][k] = common_adj[index1][index2]
                        adj_matrix[k][j] = common_adj[index1][index2]
            else:
                adj_matrix[j] = 0.5 #停用词啥的
                adj_matrix[:,j] = 0.5

        # adj_matrix  = single_graph(sentence,common_adj,stop_words)
        idx2graph[i] = adj_matrix
    fout = open(fname + 'single'+'.graph', 'wb')
    pickle.dump(idx2graph, fout)
    fout.close()


def process_single_test(fname):
    #!添加！为了处理测试集
    test_file = fname + 'test.raw'
    fin = open(test_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    train_file = fname +'train.raw'
    fin = open(train_file + 'fre' + '.graph', 'rb')  # 使用之前训练集构造的全部词频矩阵
    common_adj = pickle.load(fin)
    fin.close()
    idx2graph = {}
    stop_words = stopword()
    all_sentence = []#只有特征词的句子
    re_sentence = []#原始句子
    Full_familiar_Feature = {}
    for i in range(0, len(lines), 3):#所有的句子
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$") ] #去除停用词和标点
        aspect = lines[i + 1].lower().strip()
        sentence = text_left+' '+aspect+' '+text_right
        re_sentence.append(sentence) #原始句子
        sentence = [s for s in sentence.split() if s not in stop_words]
        all_sentence.append(sentence)
        Single_text_feature_sort_dict = collections.Counter(sentence)  # 词频统计
        Full_familiar_Feature = dict(Counter(dict(Single_text_feature_sort_dict)) + Counter(dict(Full_familiar_Feature)))  # 化为Counter后 作并集，再将结果化为dict

    Full_Feature_word= list(Full_familiar_Feature.keys()) #所有的特征词
    feat_dic = {}
    for i in range(len(Full_Feature_word)):
        feat_dic[Full_Feature_word[i]] = i #key：单词 value：位置
    #所有的特征词在Full_Feature_word中
    for i in range(len(re_sentence)):  # 得到所有的句子
        sentence = re_sentence[i] #对每一个句子
        sentence_split = sentence.split()
        seq_len = len(sentence_split) #一个句子中所有的单词
        adj_matrix = np.zeros((seq_len, seq_len)).astype('float32') #构建词典
        for j in range(seq_len - 1):
            #对每个单词
            word1 = sentence_split[j]
            if word1 in feat_dic.keys():
                 index1 = feat_dic.get(word1)#得到当前位置
                 adj_matrix[j][j] = common_adj[index1][index1] #对角线位置为该单词出现频率
                 for k in range(j + 1, seq_len):
                    word2 = sentence_split[k]
                    if word2 in feat_dic.keys():
                        index2 = feat_dic.get(word2)
                        adj_matrix[j][k] = common_adj[index1][index2]
                        adj_matrix[k][j] = common_adj[index1][index2]
                    else:#若测试集中词未在训练集出现过，则直接设置为0
                        adj_matrix[j][k] = 0.5
                        adj_matrix[k][j] = 0.5

            else:
                adj_matrix[j] = 0.5 #停用词啥的
                adj_matrix[:,j] = 0.5

        # adj_matrix  = single_graph(sentence,common_adj,stop_words)
        idx2graph[i] = adj_matrix #i是每个句子
    fout = open(test_file + 'single'+'.graph', 'wb')
    pickle.dump(idx2graph, fout)
    fout.close()

#根据每个instance的小词频图，把它们进行分层
#一共分为8类

def process_single_hira(fname):
    f1 = [1]
    f2 = [2]
    f3 = [3, 4]
    f4 = list(range(5, 9))
    f5 = list(range(9, 17))
    f6 = list(range(17, 33))
    f7 = list(range(33, 65))
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()#先得到数据集所有的instance 为了得到行数
    fin.close()
    # 句法依赖图
    fin = open(fname+'single.graph', 'rb')
    fre_graphs = pickle.load(fin)
    fin.close()
    idx2graph = {}
    for i in range(0, len(lines), 3):
        #对每句话，重新构造分层的词频图
        fre_graph = fre_graphs[i / 3] #注意这里

        seq_len = fre_graph.shape[0]
        matrix1 = np.zeros((seq_len, seq_len)).astype('float32')
        matrix2 = np.zeros((seq_len, seq_len)).astype('float32')
        matrix3 = np.zeros((seq_len, seq_len)).astype('float32')
        matrix4 = np.zeros((seq_len, seq_len)).astype('float32')
        matrix5 = np.zeros((seq_len, seq_len)).astype('float32')
        matrix6 = np.zeros((seq_len, seq_len)).astype('float32')
        matrix7 = np.zeros((seq_len, seq_len)).astype('float32')
        matrix8 = np.zeros((seq_len, seq_len)).astype('float32')
        #这样每个句子会有8个图；按照词频划分的
        #他的值就是原来的值
        for j in range(seq_len):
            matrix1[j][j] = 1
            matrix2[j][j] = 1
            matrix3[j][j] = 1
            matrix4[j][j] = 1
            matrix5[j][j] = 1
            matrix6[j][j] = 1
            matrix7[j][j] = 1
            matrix8[j][j] = 1
            for k in range(seq_len):

                value = int(fre_graph[j][k])
                if value in f1:
                    matrix1[j][k] = value
                    matrix1[k][j] = value
                elif value in f2:
                    matrix2[j][k] = value
                    matrix2[k][j] = value
                elif value in f3:
                    matrix3[j][k] = value
                    matrix3[k][j] = value
                elif value in f4:
                    matrix4[j][k] = value
                    matrix4[k][j] = value
                elif value in f5:
                    matrix5[j][k] = value
                    matrix5[k][j] = value
                elif value in f6:
                    matrix6[j][k] = value
                    matrix6[k][j] = value
                elif value in f7:
                    matrix7[j][k] = value
                    matrix7[k][j] = value
                elif value > 64:
                    matrix8[j][k] = value
                    matrix8[k][j] = value
        matrix1 = np.where(matrix1>0,matrix1,0.5)
        matrix2 = np.where(matrix2 > 0, matrix2, 0.5)
        matrix3 = np.where(matrix3 > 0, matrix3, 0.5)
        matrix4 = np.where(matrix4 > 0, matrix4, 0.5)
        matrix5 = np.where(matrix5 > 0, matrix5, 0.5)
        matrix6 = np.where(matrix6 > 0, matrix6, 0.5)
        matrix7 = np.where(matrix7 > 0, matrix7, 0.5)
        matrix8 = np.where(matrix8 > 0, matrix8, 0.5)
        idx2graph[i] = [matrix1, matrix2, matrix3, matrix4, matrix5, matrix6, matrix7, matrix8]
    fout = open(fname + 'single_hira' + '.graph', 'wb')
    pickle.dump(idx2graph, fout)
    fout.close()

if __name__ == '__main__':
    print()
    #process('./datasets/acl-14-short-data/train.raw')
    #process('./datasets/semeval14/restaurant_train.raw')
    # process('./datasets/semeval14/laptop_train.raw')
    # process('./datasets/semeval15/restaurant_train.raw')
    # process('./datasets/semeval16/restaurant_train.raw')

    process_single('./datasets/acl-14-short-data/train.raw')

    process_single('./datasets/semeval14/restaurant_train.raw')

    process_single('./datasets/semeval14/laptop_train.raw')

    process_single('./datasets/semeval15/restaurant_train.raw')

    process_single('./datasets/semeval16/restaurant_train.raw')
    #
    #The test set! Note that unlike the training set, only the word frequency of the training set is used as the adjacency matrix
    process_single_test('./datasets/acl-14-short-data/')
    process_single_test('./datasets/semeval14/restaurant_')
    process_single_test('./datasets/semeval14/laptop_')
    process_single_test('./datasets/semeval15/restaurant_')
    process_single_test('./datasets/semeval16/restaurant_')

    #
    process_single_hira('./datasets/acl-14-short-data/train.raw')
    process_single_hira('./datasets/semeval14/restaurant_train.raw')
    process_single_hira('./datasets/semeval14/laptop_train.raw')
    process_single_hira('./datasets/semeval15/restaurant_train.raw')
    process_single_hira('./datasets/semeval16/restaurant_train.raw')

    process_single_hira('./datasets/acl-14-short-data/test.raw')
    process_single_hira('./datasets/semeval14/restaurant_test.raw')
    process_single_hira('./datasets/semeval14/laptop_test.raw')
    process_single_hira('./datasets/semeval15/restaurant_test.raw')
    process_single_hira('./datasets/semeval16/restaurant_test.raw')