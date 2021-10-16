# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader
import pdb
import nltk
from nltk.corpus import stopwords


"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
  
def load_data(trainingdir, testdir, stemming=False, lowercase=True, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


def count_words(data_set,data_label):
    positive_dic = {}
    negative_dic = {}
    length = len(data_label)
    # total_words = 0
    # V = 0
    stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 
    'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 
    'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 
    'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 
    'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 
    'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 
    'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 
    'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 
    'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 
    'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 
    'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 
    'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 
    'how', 'further', 'was', 'here', 'than','I','imdb'}

    for i in range(length):
        sentence = data_set[i]
        if (data_label[i] == 1):     # positive comment
            for word in sentence:
                # if (word in stopwords):
                #     continue
                if word not in positive_dic:
                    positive_dic[word] = 1
                else:
                    positive_dic[word] += 1

        else:                       # negative comment
            for word in sentence:
                # if (word in stopwords):
                #     continue
                if word not in negative_dic:
                   negative_dic[word] = 1
                else:
                    negative_dic[word] += 1
    # print(V)
    return (positive_dic, negative_dic)

# n-number of words    countw-number of times w appear  apha-laplace  V-number of word TYPE
def cal_laplace(alpha,dic,n,V):
    prob_dic = dic

    for word in dic:
        countw = dic[word]
        prob_dic[word] = (countw+alpha)/(n+alpha*(V+1))

    prob_dic['UNK'] = alpha/(n+alpha*(V+1))

    # add = 0
    # for word in dic:
    #     add += prob_dic[word]
    # print(add)
    return prob_dic

def estimate(dic_posi,dic_nega,dev_set,pos_prior):
    dev_label = []
    posi_unk = math.log(dic_posi['UNK'])
    nega_unk = math.log(dic_nega['UNK'])
    for sentence in dev_set:
        # print(sentence)
        # if (sentence == ['best', 'best', 'worst', 'worst']):
        #     breakpoint()
        p_posi = math.log(pos_prior)
        p_nega = math.log(1-pos_prior)
        for word in sentence:
            if (word in dic_posi):
                temp = dic_posi[word]
                p_posi += math.log(temp)

            if (word not in dic_posi):
                p_posi += posi_unk

            if (word in dic_nega):
                temp = dic_nega[word]
                p_nega += math.log(temp)

            if (word not in dic_nega):
                p_nega += nega_unk
        
        # print(math.exp(p_posi))
        # print(math.exp(p_nega))
        if (p_posi >= p_nega):
            dev_label.append(1)
        else:
            dev_label.append(0)

    return dev_label

def cal_n_V(dic):
    n = 0
    for word in dic:
        n += dic[word]
    V = len(dic)
    return (n,V)
           
"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.005, pos_prior=0.76,silently=False):
    # Keep this in the provided template
    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        yhats.append(-1)

    print_paramter_vals(laplace,pos_prior)
   
    positive_dic = count_words(train_set,train_labels)[0]
    negative_dic = count_words(train_set,train_labels)[1]

    pos_n_v = cal_n_V(positive_dic)
    neg_n_v = cal_n_V(negative_dic)

    prob_posi = cal_laplace(laplace,positive_dic,pos_n_v[0],pos_n_v[1])
    prob_nega = cal_laplace(laplace,negative_dic,neg_n_v[0],neg_n_v[1])
    # print(pos_n_v,neg_n_v)
    
    to_return  = estimate(prob_posi,prob_nega,dev_set,pos_prior)
    return to_return

# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")

def count_bigram_words(data_set, data_label):
    positive_dic = {}
    negative_dic = {}
    length = len(data_label)

    for i in range(length):
        sentence = data_set[i]
        if (data_label[i] == 1):     # positive comment
            for j in range(len(sentence)-1):
                bigram = (sentence[j],sentence[j+1])
                if bigram not in positive_dic:
                    positive_dic[bigram] = 1
                else:
                    positive_dic[bigram] += 1

        else:                       # negative comment
            for j in range(len(sentence)-1):
                bigram = (sentence[j],sentence[j+1])
                if bigram not in negative_dic:
                    negative_dic[bigram] = 1
                else:
                    negative_dic[bigram] += 1
    # print(V)
    return (positive_dic, negative_dic)

def give_dev_label(pos_single,neg_single,pos_bigram,neg_bigram,lamda):
    length = len(pos_single)
    output = []
    for i in range(length):
        pos_result = (1-lamda)*pos_single[i] + lamda*pos_bigram[i]
        neg_result = (1-lamda)*neg_single[i] + lamda*neg_bigram[i]
        if (pos_result >= neg_result):
            output.append(1)
        else:
            output.append(0)
    return output

def estimate_bigram(dic_single_posi,dic_single_nega,dic_bigram_posi,dic_bigram_nega,dev_set,pos_prior,bigram_lambda):
    dev_label = []
    posi_single_unk = math.log(dic_single_posi['UNK'])
    nega_single_unk = math.log(dic_single_nega['UNK'])

    posi_bigram_unk = math.log(dic_bigram_posi['UNK'])
    nega_bigram_unk = math.log(dic_bigram_nega['UNK'])

    posi_single_arr = []
    nega_single_arr = []
    posi_bigram_arr = []
    nega_bigram_arr = []

    # single part
    for sentence in dev_set:
        # print(sentence)
        p_posi = math.log(pos_prior)
        p_nega = math.log(1-pos_prior)
        for word in sentence:
            if (word in dic_single_posi):
                temp = dic_single_posi[word]
                p_posi += math.log(temp)

            if (word not in dic_single_posi):
                p_posi += posi_single_unk

            if (word in dic_single_nega):
                temp = dic_single_nega[word]
                p_nega += math.log(temp)

            if (word not in dic_single_nega):
                p_nega += nega_single_unk
        
        posi_single_arr.append(p_posi)
        nega_single_arr.append(p_nega)

    # multiple part
    for sentence in dev_set:
        p_posi = math.log(pos_prior)
        p_nega = math.log(1-pos_prior)
        for index in range(len(sentence)-1):
            bigram = (sentence[index],sentence[index+1])
            if (bigram in dic_bigram_posi):
                temp = dic_bigram_posi[bigram]
                p_posi += math.log(temp)

            if (bigram not in dic_bigram_posi):
                p_posi += posi_bigram_unk

            if (bigram in dic_bigram_nega):
                temp = dic_bigram_nega[bigram]
                p_nega += math.log(temp)

            if (bigram not in dic_bigram_nega):
                p_nega += nega_bigram_unk
        
        posi_bigram_arr.append(p_posi)
        nega_bigram_arr.append(p_nega)

    dev_label = give_dev_label(posi_single_arr,nega_single_arr,posi_bigram_arr,nega_bigram_arr,bigram_lambda)
    return dev_label

# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.005, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.7, silently=False):

    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    
    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        yhats.append(-1)

    positive_single_dic = count_words(train_set,train_labels)[0]
    negative_single_dic = count_words(train_set,train_labels)[1]
    positive_bigram_dic = count_bigram_words(train_set,train_labels)[0]
    negative_bigram_dic = count_bigram_words(train_set,train_labels)[1]

    pos_single_n_v = cal_n_V(positive_single_dic)
    neg_single_n_v = cal_n_V(negative_single_dic)
    pos_bigram_n_v = cal_n_V(positive_bigram_dic)
    neg_bigram_n_v = cal_n_V(negative_bigram_dic)

    prob_single_posi = cal_laplace(unigram_laplace,positive_single_dic,pos_single_n_v[0],pos_single_n_v[1])
    prob_single_nega = cal_laplace(unigram_laplace,negative_single_dic,neg_single_n_v[0],neg_single_n_v[1])
    prob_bigram_posi = cal_laplace(bigram_laplace,positive_bigram_dic,pos_bigram_n_v[0],pos_bigram_n_v[1])
    prob_bigram_nega = cal_laplace(bigram_laplace,negative_bigram_dic,neg_bigram_n_v[0],neg_bigram_n_v[1])

    to_return  = estimate_bigram(prob_single_posi,prob_single_nega,prob_bigram_posi,prob_bigram_nega,dev_set,pos_prior,bigram_lambda)
    return to_return

