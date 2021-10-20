"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
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

def make_V_n_transition_table(dic,tag_list):
    table = {}
    total_V = 0
    total_n = 0
    for tag in tag_list:
        V = 0
        n = 0
        for tag_pair in dic:
            if (tag_pair[0] == tag):
                total_V += 1
                total_n += dic[tag_pair]
                V += 1
                n += dic[tag_pair]
        table[tag] = (V,n)
    return table,total_V -1,total_n  # exclude end

def make_V_n_emission_table(tag_word_list,tag_list):
    table = {}
    total_V = 0
    total_n = 0
    for tag in tag_list:
        if (tag != 'START' and tag != 'END'):
            V = 0
            n = 0
            for tag_pair in tag_word_list:
                if (tag_pair[0] == tag):
                    total_V += 1
                    total_n += tag_word_list[tag_pair]
                    V += 1
                    n += tag_word_list[tag_pair]
            table[tag] = (V,n)
    return table,total_V, total_n


# alpha-laplace     dic - list    n-number of words   V-number of word TYPE
def cal_laplace_transition(alpha,dic,tag_list):
    # initial
    V_n_table, total_V,total_n = make_V_n_transition_table(dic,tag_list)
    prob_table = {}
    # intial type
    V = V_n_table['START'][0]
    n = V_n_table['START'][1]
    for tag_pair in dic:
        t0 = tag_pair[0]
        t1 = tag_pair[1]
        if (t0 == 'START'):
            prob_table[tag_pair]=(dic[tag_pair]+alpha)/(n+alpha*(V+1))

    for tag_pair in dic:
        t0 = tag_pair[0]
        if (t0 != 'START'):
            V = V_n_table[t0][0]
            n = V_n_table[t0][1]
            countw = dic[tag_pair]
            prob_table[tag_pair] = (countw+alpha)/(n+alpha*(V+1))

    prob_table['UNK'] = alpha/(total_n+alpha*(total_V+1))
    return prob_table

def get_hapex_dic(hapex_dic,tag_word):
    tag = tag_word[0]
    word = tag_word[1]

    if (word not in hapex_dic):
        hapex_dic[word] = [1,tag]
    else:
        hapex_dic[word][0] += 1
    return hapex_dic

def make_hapex_prob(hapex_dic):
    tag_dic = {}
    n = 0
    for word in hapex_dic:
        times = hapex_dic[word][0]
        tag =hapex_dic[word][1]
        if (times == 1):
            n += 1
            if (tag not in tag_dic):
                tag_dic[tag] = 1
            else:
                tag_dic[tag] += 1
    
    V = len(tag_dic)
    alpha = 0.000001
    for tag in tag_dic:
        tag_dic[tag] = (alpha+tag_dic[tag])/(n+alpha*(V+1))

    tag_dic['UNK'] = alpha/(n+alpha*(V+1))
    return tag_dic

def get_hapex_prob_dic(tag_word_list):
    hapex_dic = {}
    for tag_word in tag_word_list:
        if (tag_word[0] != 'START' and tag_word[0] != 'END'):
            hapex_dic = get_hapex_dic(hapex_dic,tag_word)
    tag_hapex_prob = make_hapex_prob(hapex_dic)
    return tag_hapex_prob


def cal_laplace_emission(alpha,tag_word_list,tag_list):
    V_n_table, total_V,total_n = make_V_n_emission_table(tag_word_list,tag_list)
    prob_table = {}
    hapex_dic = get_hapex_prob_dic(tag_word_list)
    # print(hapex_dic)
    for tag_word in tag_word_list:
        if (tag_word[0] != 'START' and tag_word[0] != 'END'):
            V = V_n_table[tag_word[0]][0]
            n = V_n_table[tag_word[0]][1]
            countw = tag_word_list[tag_word]
            if (tag_word[0] not in hapex_dic):
                prob = hapex_dic['UNK']
            else:
                prob = hapex_dic[tag_word[0]]
            scaled_alpha = alpha*prob
            prob_table[tag_word] = (countw+scaled_alpha)/(n+scaled_alpha*(V+1))

    prob_table['UNK'] = alpha/(total_n+alpha*(total_V+1))
    return prob_table, hapex_dic

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

def cal_viterbi(sentence,findparent,map,list_prob_tag_pair,list_prob_tag_word,hapex_prob):
    # setup
    for key,value in map[0].items():
        if (key == 'START'):
            map[0]['START'] = 100
        else:
            map[0][key] = list_prob_tag_word['UNK']
    
    length = len(map)

    p_e = 0
    p_t = 0
    p_prev = 0
    # print(hapex_prob)

    for time in range(1,length-1):
        # emission
        for key,value in map[time].items():
            max = -2**16
            max_key0 = None
            if ((key,sentence[time]) in list_prob_tag_word):
                p_e = list_prob_tag_word[(key,sentence[time])]
            elif (key in hapex_prob):
                p_e = list_prob_tag_word['UNK']*hapex_prob[key]     # edit pt2
            else:
                p_e = list_prob_tag_word['UNK']*hapex_prob['UNK']
            # transition
            for key0,value0 in map[time-1].items():
                if ((key0,key) in list_prob_tag_pair):
                    p_t = list_prob_tag_pair[(key0,key)]
                else:
                    p_t = list_prob_tag_pair['UNK']

                p_prev = map[time-1][key0]
                temp = p_prev + math.log(p_t) + math.log(p_e)
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

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_list = {}
    tag_pair_list = {}
    tag_word_list = {}
    tag_list,tag_word_list= count_tag_word(train,tag_list,tag_word_list)
    tag_pair_list = count_pair_list(train,tag_pair_list)

    transition_laplace = 0.0000001
    emission_laplace = 0.001
    list_prob_tag_pair = cal_laplace_transition(transition_laplace,tag_pair_list,tag_list)
    list_prob_tag_word,hapex_prob = cal_laplace_emission(emission_laplace,tag_word_list,tag_list)
    output = []

    for sentence in test:
        map = get_trellis_map(tag_list,sentence)
        findparent = {}
        tag_find = cal_viterbi(sentence,findparent,map,list_prob_tag_pair,list_prob_tag_word,hapex_prob)

        for i in range(len(sentence)):
            tag_find[i] = (sentence[i],tag_find[i][1])
        output.append(tag_find)
    return output