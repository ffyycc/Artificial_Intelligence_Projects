"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
from math import inf
import math
import pdb
def backtrack(start,end,dic):
    path = []
    path.append(end)

    while path[-1] != start:
        child = path[-1]
        path.append(dic[child])
    path.reverse()
    return path

def count_tag_word(train,tag_list,tag_word_list):
    word_list = []
    for sentence in train:
        for pair in sentence:
            word = pair[0]
            if (word != 'END'):
                if word not in word_list:
                    word_list.append(word)
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
    return tag_list,tag_word_list,word_list

def count_pair_list(train,tag_pair_list):
    for sentence in train:
        for j in range(len(sentence)-1):
            if (sentence[j+1][1] != 'END'):
                tag_pair = (sentence[j][1],sentence[j+1][1])
                if tag_pair not in tag_pair_list:
                    tag_pair_list[tag_pair] = 1
                else:
                    tag_pair_list[tag_pair] += 1
    return tag_pair_list

def cal_total(table):
    return sum(table.values())

# alpha-laplace     dic - list    n-number of words   V-number of word TYPE
def cal_laplace_transition(alpha,dic,tag_list):
    V = 0
    prob_table = {}
    V = len(tag_list)
    
    total_n = 0
    for time0 in tag_list:
        total_n += tag_list[time0]
        n = tag_list[time0]
        
        for tag_pair in dic:
            countw = dic[tag_pair]
            prob_table[tag_pair] = math.log((countw+alpha)/(n+alpha*(V+1)))
    prob_table['UNK'] = math.log(alpha/(total_n+alpha*(V+1)))
    return prob_table

def cal_laplace_emission(alpha,tag_word_list,tag_list,word_type):
    prob_table = {}
    V = word_type
    n_total = 0
    for tag_word in tag_word_list:
        tag = tag_word[0]
        n = tag_list[tag]
        countw = tag_word_list[tag_word]
        n_total += n
        prob_table[tag_word] = math.log((countw+alpha)/(n+alpha*(V+1)))
    prob_table['UNK'] = math.log(alpha/(n_total+alpha*(V+1)))
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
            map[0][key] = -inf
    
    length = len(map)

    p_e = 0
    p_t = 0
    p_prev = 0

    for time in range(1,length-1):
        # emission
        for key,value in map[time].items():
            max = -inf
            max_key0 = None
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
    return to_return

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_list = {}
    tag_pair_list = {}
    tag_word_list = {}
    tag_list,tag_word_list,word_list = count_tag_word(train,tag_list,tag_word_list)
    tag_pair_list = count_pair_list(train,tag_pair_list)

    transition_laplace = 0.000000025
    emission_laplace = 0.0000001
    list_prob_tag_pair = cal_laplace_transition(transition_laplace,tag_pair_list,tag_list)
    list_prob_tag_word = cal_laplace_emission(emission_laplace,tag_word_list,tag_list,len(word_list))
    
    output = []

    for sentence in test:
        map = get_trellis_map(tag_list,sentence)
        findparent = {}
        tag_find = cal_viterbi(sentence,findparent,map,list_prob_tag_pair,list_prob_tag_word)

        for i in range(len(sentence)):
            tag_find[i] = (sentence[i],tag_find[i][1])
        output.append(tag_find)
    return output