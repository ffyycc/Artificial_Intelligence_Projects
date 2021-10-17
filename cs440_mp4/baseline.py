"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
import pdb

def create_word_tag_table(data):
    word_table = {}
    tag_table = {}
    for sentence in data:
        for word in sentence:
            w = word[0]
            tag = word[1]
            if ((tag != "START" and tag != "END")): #remove START and END tags in tag_table)
                if (tag not in tag_table):
                        tag_table[tag] = 1
                else:
                        tag_table[tag] += 1

                if (w not in word_table):   # word in word table
                        word_table[w] = {}
                        word_table[w][tag] = 1
                else:                       # word in word table
                   if (tag not in word_table[w]):
                        word_table[w][tag] = 1
                   else:
                        word_table[w][tag] += 1
    return word_table,tag_table

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # train data ex [('START', 'START'), ('cat', 'A'), ('cat', 'B')
    # test data ex ['START', 'cow', 'cat', 'dog', 'END']
    word_table, tag_table = create_word_tag_table(train)
    max_tag = max(tag_table, key=tag_table.get)

    output = []
    
    for sentence in test:
        temp = [('START','START')]
        for word in sentence:
            if (word != 'START' and word != 'END'):
                chosen_tag = 'NONE'
                if (word in word_table):
                        chosen_tag = max(word_table[word],key=word_table[word].get)
                else:
                        chosen_tag = max_tag
                
                temp.append((word,chosen_tag))
        temp.append(('END','END'))
        output.append(temp)
    # print(output)
    return output 