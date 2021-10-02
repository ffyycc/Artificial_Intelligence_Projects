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
  
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=True):
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
    total_words = 0

    for i in range(length):
        sentence = data_set[i]
        if (data_label[i] == 1):     # positive comment
            for word in sentence: 
                if word not in positive_dic:
                    positive_dic[word] = 1
                else:
                    positive_dic[word] += 1
                total_words += 1
        else:                       # negative comment
            for word in sentence:
                if word not in negative_dic:
                   negative_dic[word] = 1
                else:
                    negative_dic[word] += 1
                total_words += 1
    return (positive_dic, negative_dic,total_words)

# n-number of words    countw-number of times w appear  apha-laplace  V-number of word TYPE
def cal_laplace(n,alpha,dic):
    prob_dic = dic
    V  = len(dic)

    for word in dic:
        countw = dic[word]
        temp = (countw+alpha)/(n+alpha*(V+1))
        prob_dic[word] = temp 

    temp = alpha/(n+alpha*(V+1))
    prob_dic['UNK'] = temp
    return prob_dic

def estimate(dic_posi,dic_nega,dev_set,pos_prior):
    dev_label = []
    for sentence in dev_set:
        p_posi = math.log(pos_prior)
        p_nega = math.log(1-pos_prior)
        for word in sentence:
            if (word in dic_posi):
                temp = dic_posi[word]
                p_posi += math.log(temp)

            if (word not in dic_posi):
                temp = dic_posi['UNK']
                p_posi += math.log(temp)

            if (word in dic_nega):
                temp = dic_nega[word]
                p_nega += math.log(temp)

            if (word not in dic_posi):
                temp = dic_nega['UNK']
                p_nega += dic_nega['UNK']
        
        if (p_posi >= p_nega):
            dev_label.append(1)
        else:
            dev_label.append(0)

    return dev_label
           
"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5,silently=True):
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)
   
    positive_dic = count_words(train_set,train_labels)[0]
    negative_dic = count_words(train_set,train_labels)[1]
    total_words = count_words(train_set,train_labels)[2]

    prob_posi = cal_laplace(total_words,laplace,positive_dic)
    prob_nega = cal_laplace(total_words,laplace,negative_dic)
    # print(prob_nega)
    
    yhats = estimate(prob_posi,prob_nega,dev_set,pos_prior)
    
    # yhats = []
    # for doc in tqdm(dev_set,disable=silently):
    #     yhats.append(-1)
    return yhats


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=1.0,pos_prior=0.5, silently=False):

    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    
    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        yhats.append(-1)
    return yhats

