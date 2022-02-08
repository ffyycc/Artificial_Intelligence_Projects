"""
Part 4: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""
from math import inf
import math
import pdb
def end_with(word,suffix):
    if word.endswith(suffix):
        return True
    return False

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
            prob_table[('START','UNK')] = alpha/(n+alpha*(V+1))

    for tag_pair in dic:
        t0 = tag_pair[0]
        if (t0 != 'START'):
            V = V_n_table[t0][0]
            n = V_n_table[t0][1]
            countw = dic[tag_pair]
            prob_table[tag_pair] = (countw+alpha)/(n+alpha*(V+1))
            prob_table[(t0,'UNK')] = alpha/(n+alpha*(V+1))
    return prob_table

def get_hapex_dic(hapex_dic,tag_word):
    tag = tag_word[0]
    word = tag_word[1]

    if (word not in hapex_dic):
        hapex_dic[word] = [1,tag]
    else:
        hapex_dic[word][0] += 1
    return hapex_dic

def give_suffix_dic(dic,word,tag,suffix,n):
    if (end_with(word,suffix)):
        n += 1
        if (tag not in dic):
            dic[tag] = 1
        else:
            dic[tag] += 1
    return dic,n

def cal_suffix_prob(dic,n,alpha):
    newdic = {}
    V = len(dic)
    for tag in dic:
        newdic[tag] = (alpha+dic[tag])/(n+alpha*(V+1))
    newdic['UNK'] = alpha/(n+alpha*(V+1))
    return newdic

def make_hapex_prob(hapex_dic):
    # print(hapex_dic)
    tag_dic = {}
    ly_dic = {}
    ing_dic ={}
    num_dic = {}
    pie_s_dic = {}
    ed_dic ={}
    tive_dic = {}
    est_dic = {}
    ous_dic = {}
    ful_dic = {}
    tion_dic = {}
    able_dic = {}
    less_dic = {}

    n = 0
    n_ly = 0
    n_num = 0
    n_ing = 0
    n_pie_s =0
    n_ed = 0
    n_tive = 0
    n_est = 0
    n_ous = 0
    n_ful = 0
    n_tion = 0
    n_able = 0
    n_less = 0
    for word in hapex_dic:
        times = hapex_dic[word][0]
        tag =hapex_dic[word][1]

        if (times == 1):
            if (end_with(word,'ly')):
                n_ly +=1
                if (tag not in ly_dic):
                    ly_dic[tag] = 1
                else:
                    ly_dic[tag] += 1

            ing_dic,n_ing = give_suffix_dic(ing_dic,word,tag,'ing',n_ing)
            pie_s_dic, n_pie_s = give_suffix_dic(pie_s_dic,word,tag,"'s",n_pie_s)
            ed_dic, n_ed = give_suffix_dic(ed_dic,word,tag,'ed',n_ed)
            tive_dic,n_tive = give_suffix_dic(tive_dic,word,tag,'tive',n_tive)
            est_dic,n_est = give_suffix_dic(est_dic,word,tag,'est',n_est)
            ous_dic,n_ous = give_suffix_dic(ous_dic,word,tag,'ous',n_ous)
            ful_dic,n_ful = give_suffix_dic(ful_dic,word,tag,'ful',n_ful)
            tion_dic,n_tion = give_suffix_dic(tion_dic,word,tag,'tion',n_tion)
            able_dic,n_able = give_suffix_dic(able_dic,word,tag,'able',n_able)
            less_dic,n_less = give_suffix_dic(less_dic,word,tag,'less',n_less)
            
            if (word.isnumeric()):
                n_num += 1
                if (tag not in num_dic):
                    num_dic[tag] = 1
                else:
                    num_dic[tag] += 1

        if (times == 1):
            n += 1
            if (tag not in tag_dic):
                tag_dic[tag] = 1
            else:
                tag_dic[tag] += 1

    V = len(tag_dic)
    V_ly = len(ly_dic)
    V_num = len(num_dic)

    alpha = 0.000001
    ing_dic = cal_suffix_prob(ing_dic,n_ing,alpha)
    pie_s_dic = cal_suffix_prob(pie_s_dic,n_pie_s,alpha) 
    ed_dic = cal_suffix_prob(ed_dic,n_ed,alpha)
    tive_dic = cal_suffix_prob(tive_dic,n_tive,alpha)
    est_dic = cal_suffix_prob(est_dic,n_est,alpha)
    ous_dic = cal_suffix_prob(ous_dic,n_ous,alpha)
    ful_dic =cal_suffix_prob(ful_dic,n_ful,alpha)
    tion_dic = cal_suffix_prob(tion_dic,n_tion,alpha)
    able_dic = cal_suffix_prob(able_dic,n_able,alpha)
    less_dic = cal_suffix_prob(less_dic,n_less,alpha)

    for tag in tag_dic:
        tag_dic[tag] = (alpha+tag_dic[tag])/(n+alpha*(V+1))
    tag_dic['UNK'] = alpha/(n+alpha*(V+1))

    for tag in ly_dic:
        ly_dic[tag] = (alpha+ly_dic[tag])/(n_ly+alpha*(V_ly+1))
    ly_dic['UNK'] = alpha/(n_ly+alpha*(V_ly+1))

    for tag in num_dic:
        num_dic[tag] = (alpha+num_dic[tag])/(n_num+alpha*(V_num+1))
    num_dic['UNK'] = alpha/(n_num+alpha*(V_num+1))
    return tag_dic,ly_dic,num_dic,ing_dic,pie_s_dic,ed_dic,tive_dic,est_dic,ous_dic,ful_dic,tion_dic,able_dic,less_dic

def get_hapex_prob_dic(tag_word_list):
    hapex_dic = {}

    for tag_word in tag_word_list:
        if (tag_word[0] != 'START' and tag_word[0] != 'END'):
            hapex_dic = get_hapex_dic(hapex_dic,tag_word)
    tag_hapex_prob,ly_prob,num_dic,ing_dic,pie_s_dic,ed_dic,tive_dic,est_dic,ous_dic,ful_dic,tion_dic,able_dic,less_dic = make_hapex_prob(hapex_dic)
    return tag_hapex_prob,ly_prob,num_dic,ing_dic,pie_s_dic,ed_dic,tive_dic,est_dic,ous_dic,ful_dic,tion_dic,able_dic,less_dic


def cal_laplace_emission(alpha,tag_word_list,tag_list):
    V_n_table, total_V,total_n = make_V_n_emission_table(tag_word_list,tag_list)
    prob_table = {}
    normal_prob_table = {}
    ly_prob_table = {}
    num_prob_table = {}
    ing_prob_table = {}
    pie_s_prob_table = {}
    ed_prob_table = {}
    tive_prob_table = {}
    est_prob_table = {}
    ous_prob_table = {}
    ful_prob_table = {}
    tion_prob_table = {}
    able_prob_table = {}
    less_prob_table = {}
    hapex_dic,ly_prob,num_dic,ing_dic,pie_s_dic,ed_dic,tive_dic,est_dic,ous_dic,ful_dic,tion_dic,able_dic,less_dic = get_hapex_prob_dic(tag_word_list)
    # print(hapex_dic)

    for tag_word in tag_word_list:
        tag = tag_word[0]
        word = tag_word[1]
        if (tag_word[0] != 'START' and tag_word[0] != 'END'):
            V = V_n_table[tag_word[0]][0]
            n = V_n_table[tag_word[0]][1]
            countw = tag_word_list[tag_word]

            normal_prob_table[tag_word] = (countw+alpha)/(n+alpha*(V+1))
            normal_prob_table[(tag_word[0],'UNK')] = alpha/(n+alpha*(V+1))

            if (end_with(word,'ly')):
                if (tag not in ly_prob):
                    prob = ly_prob['UNK']
                else:
                    prob = ly_prob[tag]
                scaled_alpha = alpha*prob
                ly_prob_table[(tag_word[0],'UNK-ly')] = scaled_alpha/(n+scaled_alpha*(V+1))

            elif (word.isnumeric()):
                if (tag not in num_dic):
                    prob = num_dic['UNK']
                else:
                    prob = num_dic[tag]
                scaled_alpha = alpha*prob
                num_prob_table[(tag_word[0],'UNK-num')] = scaled_alpha/(n+scaled_alpha*(V+1))
            elif (end_with(word,'ing')):
                if (tag not in ing_dic):
                    prob = ing_dic['UNK']
                else:
                    prob = ing_dic[tag]
                scaled_alpha = alpha*prob
                ing_prob_table[(tag_word[0],'UNK-ing')] = scaled_alpha/(n+scaled_alpha*(V+1))
            elif (end_with(word,"'s")):
                if (tag not in pie_s_dic):
                    prob = pie_s_dic['UNK']
                else:
                    prob = pie_s_dic[tag]
                scaled_alpha = alpha*prob
                pie_s_prob_table[(tag_word[0],'UNK-pie_s')] = scaled_alpha/(n+scaled_alpha*(V+1))

            elif (end_with(word,'ed')):
                if (tag not in ed_dic):
                    prob = ed_dic['UNK']
                else:
                    prob = ed_dic[tag]
                scaled_alpha = alpha*prob
                ed_prob_table[(tag_word[0],'UNK-ed')] = scaled_alpha/(n+scaled_alpha*(V+1))

            elif (end_with(word,'est')):
                if (tag not in est_dic):
                    prob = est_dic['UNK']
                else:
                    prob = est_dic[tag]
                scaled_alpha = alpha*prob
                est_prob_table[(tag_word[0],'UNK-est')] = scaled_alpha/(n+scaled_alpha*(V+1))

            elif (end_with(word,'tive')):
                if (tag not in tive_dic):
                    prob = tive_dic['UNK']
                else:
                    prob = tive_dic[tag]
                scaled_alpha = alpha*prob
                tive_prob_table[(tag_word[0],'UNK-tive')] = scaled_alpha/(n+scaled_alpha*(V+1))

            elif (end_with(word,'ous')):
                if (tag not in ous_dic):
                    prob = ous_dic['UNK']
                else:
                    prob = ous_dic[tag]
                scaled_alpha = alpha*prob
                ous_prob_table[(tag_word[0],'UNK-ous')] = scaled_alpha/(n+scaled_alpha*(V+1))

            elif (end_with(word,'ful')):
                if (tag not in ful_dic):
                    prob = ful_dic['UNK']
                else:
                    prob = ful_dic[tag]
                scaled_alpha = alpha*prob
                ful_prob_table[(tag_word[0],'UNK-ful')] = scaled_alpha/(n+scaled_alpha*(V+1))

            elif (end_with(word,'tion')):
                if (tag not in tion_dic):
                    prob = tion_dic['UNK']
                else:
                    prob = tion_dic[tag]
                scaled_alpha = alpha*prob
                tion_prob_table[(tag_word[0],'UNK-tion')] = scaled_alpha/(n+scaled_alpha*(V+1))

            elif (end_with(word,'able')):
                if (tag not in able_dic):
                    prob = able_dic['UNK']
                else:
                    prob = able_dic[tag]
                scaled_alpha = alpha*prob
                able_prob_table[(tag_word[0],'UNK-able')] = scaled_alpha/(n+scaled_alpha*(V+1))


            elif (end_with(word,'less')):
                if (tag not in less_dic):
                    prob = less_dic['UNK']
                else:
                    prob = less_dic[tag]
                scaled_alpha = alpha*prob
                less_prob_table[(tag_word[0],'UNK-less')] = scaled_alpha/(n+scaled_alpha*(V+1))

            else:
                if (tag_word[0] not in hapex_dic):
                    prob = hapex_dic['UNK']
                else:
                    prob = hapex_dic[tag_word[0]]
                # print(prob)
                scaled_alpha = alpha*prob
                prob_table[(tag_word[0],'UNK')] = scaled_alpha/(n+scaled_alpha*(V+1))
    # print(prob_table)
    return prob_table, hapex_dic,normal_prob_table,ly_prob_table,num_prob_table,ing_prob_table,pie_s_prob_table,ed_prob_table,tive_prob_table,est_prob_table,ous_prob_table,ful_prob_table,tion_prob_table,able_prob_table,less_prob_table

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

def get_tag(ly_prob):
    l = []
    for key,value in ly_prob.items():
        l.append(key[0])
    return l


def cal_viterbi(sentence,findparent,map,list_prob_tag_pair,list_prob_tag_word,hapex_prob,normal_prob,ly_prob,num_prob_table,ing_prob_table,pie_s_prob_table,ed_prob_table,tive_prob_table,est_prob_table,ous_prob_table,ful_prob_table,tion_prob_table,able_prob_table,less_prob_table):
    # setup
    for key,value in map[0].items():
        if (key == 'START'):
            map[0]['START'] = 100
        else:
            map[0][key] = 0.0001
    
    length = len(map)

    p_e = 0
    p_t = 0
    p_prev = 0
    ly_tag_list = get_tag(ly_prob)
    num_tag_list = get_tag(num_prob_table)
    ing_tag_list =get_tag(ing_prob_table)
    pie_s_tag_list =get_tag(pie_s_prob_table)
    ed_tag_list = get_tag(ed_prob_table)
    tive_tag_list = get_tag(tive_prob_table)
    est_tag_list = get_tag(est_prob_table)
    ous_tag_list = get_tag(ous_prob_table)
    ful_tag_list =get_tag(ful_prob_table)
    tion_tag_list = get_tag(tion_prob_table)
    able_tag_list =get_tag(able_prob_table)
    less_tag_list = get_tag(less_prob_table)
    # print(hapex_prob)
    # print(list_prob_tag_word)
    for time in range(1,length-1):
        # emission
        for key,value in map[time].items():
            max = -2**16
            max_key0 = None
            word = sentence[time]
            if ((key,sentence[time]) in normal_prob):
                p_e = normal_prob[(key,sentence[time])]
            elif (end_with(word,'ly')):
                if (key in ly_tag_list):
                    p_e = ly_prob[(key,'UNK-ly')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']
            elif (word.isnumeric()):
                if (key in num_tag_list):
                    p_e = num_prob_table[(key,'UNK-num')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']
            elif (end_with(word,'ing')):
                if (key in ing_tag_list):
                    p_e = ing_prob_table[(key,'UNK-ing')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']
            elif (end_with(word,"'s")):
                # breakpoint()
                if (key in pie_s_tag_list):
                    p_e = pie_s_prob_table[(key,'UNK-pie_s')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']
            elif (end_with(word,'ed')):
                if (key in ed_tag_list):
                    p_e = ed_prob_table[(key,'UNK-ed')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']

            elif (end_with(word,'tive')):
                if (key in tive_tag_list):
                    p_e = tive_prob_table[(key,'UNK-tive')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']
            elif (end_with(word,'est')):
                if (key in est_tag_list):
                    p_e = est_prob_table[(key,'UNK-est')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']
            elif (end_with(word,'ous')):
                if (key in ous_tag_list):
                    p_e = ous_prob_table[(key,'UNK-ous')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']

            elif (end_with(word,'ful')):
                if (key in ful_tag_list):
                    p_e = ful_prob_table[(key,'UNK-ful')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']

            elif (end_with(word,'tion')):
                if (key in tion_tag_list):
                    p_e = tion_prob_table[(key,'UNK-tion')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']

            elif (end_with(word,'able')):
                if (key in able_tag_list):
                    p_e = able_prob_table[(key,'UNK-able')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']

            elif (end_with(word,'less')):
                if (key in less_tag_list):
                    p_e = less_prob_table[(key,'UNK-less')]
                else:
                    p_e = 0.000000000000001*hapex_prob['UNK']

            elif (key in hapex_prob):
                p_e = list_prob_tag_word[(key,'UNK')]             # edit pt2
            else:
                p_e = 0.000000000000001*hapex_prob['UNK']
            # transition
            for key0,value0 in map[time-1].items():
                if (key0 == 'END'):
                    continue
                if ((key0,key) in list_prob_tag_pair):
                    p_t = list_prob_tag_pair[(key0,key)]
                else:
                    p_t = list_prob_tag_pair[(key0,'UNK')]

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

def viterbi_3(train, test):
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

    transition_laplace = 0.0001
    emission_laplace = 0.00001
    list_prob_tag_pair = cal_laplace_transition(transition_laplace,tag_pair_list,tag_list)
    list_prob_tag_word,hapex_prob,normal_prob,ly_prob,num_prob_table,ing_prob_table,pie_s_prob_table,ed_prob_table,tive_prob_table,est_prob_table,ous_prob_table,ful_prob_table,tion_prob_table,able_prob_table,less_prob_table= cal_laplace_emission(emission_laplace,tag_word_list,tag_list)

    output = []

    for sentence in test:
        map = get_trellis_map(tag_list,sentence)
        findparent = {}
        tag_find = cal_viterbi(sentence,findparent,map,list_prob_tag_pair,list_prob_tag_word,hapex_prob,normal_prob,ly_prob,num_prob_table,ing_prob_table,pie_s_prob_table,ed_prob_table,tive_prob_table,est_prob_table,ous_prob_table,ful_prob_table,tion_prob_table,able_prob_table,less_prob_table)

        for i in range(len(sentence)):
            tag_find[i] = (sentence[i],tag_find[i][1])
        output.append(tag_find)
    return output