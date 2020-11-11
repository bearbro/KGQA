import sys
import ast
# sys.path.insert(0,'/Users/mingrongtang/PycharmProjects/TensorFlow/MyExperiment/rel_and_path_classification')
# sys.path.insert(0,'/home/aistudio/work/MyExperiment/classification')
sys.path.insert(0,'/home/hbxiong/QA2/classification')
import codecs
from keras_bert import Tokenizer
import re
import json

def transfer_path2sentence(path):
    path=path.replace('	','\t')
    sentence=''
    old_n = []
    for triple in path.split('\t'):
        if len(re.findall(pattern='\?',string=triple))==1:
            old_n.append(re.findall(pattern='\?\w*',string=triple)[0])
            sentence+='，'
            if triple.startswith('?'):
                sentence+='是'.join(triple.split('|||')[1:])
            else:
                sentence += '的'.join(triple.split('|||')[:-1])
        num_n=re.findall(pattern='\?\w*',string=triple)
        if len(num_n)==2:
            if num_n[1] not in old_n:
                sentence+='的'+triple.split('|||')[1]+'是'
            else:#对x->y这种路径进行特殊转换
                sentence=triple.split('|||')[1]+'是（'+sentence.lstrip('，')+'）'
    sentence=sentence.replace('<','').replace('>','').replace('_','')
    return sentence.lstrip('，')+'？'
# path='<龙卷风_（一种自然天气现象）>|||<外文名>|||?y	?x|||<英文名称>|||?y'
# print('select ?x where { '+path.replace('|||',' ').replace('\t',' . ')+'. }')
# print(path.replace('|||','').replace('<', '').replace('>', '').replace('  ','，').replace('\t', '，').replace('?x', '？').replace('?y','').replace('_',''))
# print(path.replace('>','').replace('<','').replace('_','').replace('|||', '').replace('\t', '，').replace('  ','，').replace('?', '').replace('ans',''))
# print(transfer_path2sentence(path))

def transfer_path2sentence_old(path):
    '''
    目前maxbert模型和预测均使用这种方法进行转换。
    :param path:
    :return:
    '''
    sentence=''
    for triple in path.split('\t'):
        if len(re.findall(pattern='\?',string=triple))==1:
            sentence+='，'
            if triple.startswith('?'):
                sentence+='是'.join(triple.split('|||')[1:])
            else:
                sentence += '的'.join(triple.split('|||')[:-1])
        if len(re.findall(pattern='\?',string=triple))==2:
            sentence+='的'+triple.split('|||')[1]
    sentence=sentence.replace('<','').replace('>','').replace('_','')
    return sentence[1:]+'？'

def transfer_path2sentence_maxbert(path):
    path=path.replace('	','\t')
    sentence=''
    for triple in path.split('\t'):
        if len(re.findall(pattern='\?',string=triple))==1:
            sentence+='，'
            if triple.startswith('?'):
                sentence+='是'.join(triple.split('|||')[1:])
            else:
                sentence += '的'.join(triple.split('|||')[:-1])
        num_n=re.findall(pattern='\?\w*',string=triple)
        if len(num_n)==2:
            sentence+='的'+triple.split('|||')[1]
    sentence=sentence.replace('<','').replace('>','').replace('_','')
    return sentence.lstrip('，')+'？'


def transfer_data_train(sentence,pos,neg,max_length,bert_vocab_path):
    print('begin transfer train/valid data')
    token_dict = {}
    with codecs.open(bert_vocab_path, 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    # token_dict_inv = {v: k for k, v in token_dict.items()}
    tokenizer = Tokenizer(token_dict)

    pos_indices = []
    pos_segments = []
    neg_indices=[]
    neg_segments=[]
    labels=[]
    for i in range(len(sentence)):
        # print(sentence[i])
        # pos_clear_sample = pos[i].replace('> ', '').replace(' <', '').replace('<', '').replace('>', '').replace(' ','').replace('  ','，').replace('\t', '，').replace('?x', '？').replace('?y', '').replace('_', '')
        #pos_clear_sample=pos[i].replace('>', '').replace('<', '').replace('|||','').replace('\t', '').replace('?', '').replace('ans','？')
        pos_clear_sample=transfer_path2sentence(pos[i])

        # print(pos_clear_sample)
        pos_indice, pos_segment = tokenizer.encode(first=sentence[i], second=pos_clear_sample,max_len=max_length)
        pos_indices.append(pos_indice)
        pos_segments.append(pos_segment)

        # neg_clear_sample = neg[i].replace('> ', '').replace(' <', '').replace('<', '').replace('>', '').replace('  ','，').replace('\t', '，').replace('?x', '？').replace('?y', '').replace('_', '')
        #neg_clear_sample=neg[i].replace('>', '').replace('<', '').replace('|||','').replace('\t', '').replace('?', '').replace('ans','？')
        neg_clear_sample=transfer_path2sentence(neg[i])

        # print(neg_clear_sample)
        neg_indice, neg_segment = tokenizer.encode(first=sentence[i], second=neg_clear_sample, max_len=max_length)
        neg_indices.append(neg_indice)
        neg_segments.append(neg_segment)

        labels.append([1,0])

    print('over transfer train/valid data')
    return pos_indices,pos_segments,neg_indices,neg_segments,labels
    
def transfer_data_test(x_sent,x_sample,max_length,bert_vocab_path):
    '''
    similarity模型转换训练数据集为可输入模型格式。
    将句子输入转化成bert相似度模型的输入形式，分词，向量化
    :param x_sent:列表
    :param x_sample:列表，长度和x_sample相同
    :param max_length:
    :return:
    '''
    # print('begin transfer train/valid data')
    token_dict = {}
    with codecs.open(bert_vocab_path, 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer=Tokenizer(token_dict)

    x_indices=[]
    x_segments=[]
    for i in range(len(x_sent)):
        # print(x_sample[i])
        # clear_sample=x_sample[i].replace('> ','').replace(' <','').replace('<', '').replace('>', '').replace('  ','，').replace('\t', '，').replace('?x', '？').replace('?y','').replace('_','')
        #clear_sample=x_sample[i].replace('>','').replace('<','').replace('_','').replace('|||', '').replace('\t', '').replace('  ','').replace('?', '').replace('ans','？')
        clear_sample=transfer_path2sentence(x_sample[i])
        
        #clear_sample='select ?a where { '+x_sample[i].replace('|||',' ').replace('\t',' . ').replace('?x','?a')+'. }'#.replace('?ans','?a')
        #print(clear_sample)
        indice,segment=tokenizer.encode(first=x_sent[i],second=clear_sample,max_len=max_length)
        x_indices.append(indice)
        x_segments.append(segment)
        # print(indice)
        # print(segment)
    # print('over transfer train/valid data')

    return x_indices,x_segments
    
def train_read_data(path):
    with open(path,'r',encoding='utf-8') as reader:
        reader_data=json.load(reader)
        # print(len(reader_data))
        sentence=[]
        pos=[]
        neg=[]
        for item in reader_data:
            sentence.append(item['sentence'])
            pos.append(item['pairs'][0])
            neg.append(item['pairs'][1])

        assert len(sentence)==len(pos)
        assert len(sentence)==len(neg)
    return sentence,pos,neg
