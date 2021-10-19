# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post @650 - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
import math
import pdb
def backtrack(start,end,dic):
    path = []
    path.append(end)

    while path[-1] != start:
        child = path[-1]
        path.append(dic[child])
    path.reverse()
    #print(path.reverse)
    return path

def count_tag(train,tag_list,tag_word_list):
    for sentence in train:
        for pair in sentence:
            word = pair[0]
            tag = pair[1]
            tag_word = (tag,word)
            if (tag not in tag_list):
                tag_list[tag] = 1
            else:
                tag_list[tag] += 1
            if (tag_word not in tag_word_list):
                tag_word_list[tag_word] = 1
            else:
                tag_word_list[tag_word] += 1
    return tag_list,tag_word_list

def count_pair_list(train,tag_pair_list):
    for sentence in train:
        for j in range(len(sentence)-1):
            tag_pair = (sentence[j][1],sentence[j+1][1])
            if tag_pair not in tag_pair_list:
                tag_pair_list[tag_pair] = 1
            else:
                tag_pair_list[tag_pair] += 1
    return tag_pair_list

def cal_total(table):
    return sum(table.values())

# alpha-laplace     dic - list    n-number of words   V-number of word TYPE
def cal_laplace_tag_pair(alpha,dic):
    n = 0
    V = 0
    prob_table = {}
    for tag_pair in dic:
        n += dic[tag_pair]
        V += 1
        
    for tag_pair in dic:
        countw = dic[tag_pair]
        prob_table[tag_pair] = math.log((countw+alpha)/(n+alpha*(V+1)))

    prob_table['UNK'] = math.log(alpha/(n+alpha*(V+1)))
    # print(cal_total(prob_table))
    return prob_table

def get_trellis_map(tags,sentence):
    out = []
    l = len(sentence)
    for i in range(l):
        temp = {}
        for j in tags:
            temp[j] = 0
        out.append(temp)
    return out

def find_max_key(dic):
    # breakpoint()
    high = -2**32
    k = None
    for key,value in dic.items():
        if (value > high):
            high = value
            k = key
    return k

def cal_viterbi(sentence,findparent,map,list_prob_tag_pair,list_prob_tag_word):
    # setup
    for key,value in map[0].items():
        if (key == 'START'):
            map[0]['START'] = list_prob_tag_word[('START', 'START')]
        else:
            map[0][key] = -11111
    
    length = len(map)

    p_e = 0
    p_t = 0
    p_prev = 0

    for time in range(1,length-1):
        # emission
        for key,value in map[time].items():
            max = -2**16
            max_key = None
            if ((key,sentence[time]) in list_prob_tag_word):
                p_e = list_prob_tag_word[(key,sentence[time])]
            else:
                p_e = list_prob_tag_word['UNK']
    
            # transition
            for key0,value0 in map[time-1].items():
                if ((key0,key) in list_prob_tag_pair):
                    p_t = list_prob_tag_pair[(key0,key)]
                else:
                    p_t = list_prob_tag_pair['UNK']

                p_prev = map[time-1][key0]
                temp = p_prev + p_t + p_e
                if (temp > max):
                    max_key0 = key0
                    max = temp

            if (time == 1):
                findparent[(time,key)] = (0,'START')
            else:
                # update findparent
                findparent[(time,key)] = (time-1,max_key0)
            # record into table
            map[time][key] = max

    last_tag_idx = length - 2
    last_tag = find_max_key(map[last_tag_idx])
    findparent[(length-1,'END')] = (last_tag_idx,last_tag)
    to_return = backtrack((0,'START'),(length-1,'END'),findparent)
    # print(to_return)
    return to_return

def main():
    test, emission, transition, output = read_files()
    emission, transition = get_nested_dictionaries(emission, transition)
    initial = transition["START"]
    prediction = []
    
    """WRITE YOUR VITERBI IMPLEMENTATION HERE"""

    print(emission)
    # print(initial)
    print(transition["START"])

    # for sentence in test:
    #     map = get_trellis_map(tag_list,sentence)
    #     findparent = {}
    #     tag_find = cal_viterbi(sentence,findparent,map,list_prob_tag_pair,list_prob_tag_word)

    #     for i in range(len(sentence)):
    #         tag_find[i] = (sentence[i],tag_find[i][1])
    #     prediction.append(tag_find)

    print('Your Output is:',prediction,'\n Expected Output is:',output)


if __name__=="__main__":
    main()