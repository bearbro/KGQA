import sys
import ast
# sys.path.insert(0,'/Users/mingrongtang/PycharmProjects/TensorFlow/MyExperiment/rel_and_path_similarity')
sys.path.insert(0,r'C:\Users\bear\Desktop\QA\rel_and_path_similarity')#room
# sys.path.insert(0, '/home/aistudio/work/MyExperiment/rel_and_path_similarity')#百度
import codecs
from keras_bert import Tokenizer
import re

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
                sentence=triple.split('|||')[1]+'是'+sentence.lstrip('，')
    sentence=sentence.replace('<','').replace('>','').replace('_','')
    return sentence.lstrip('，')+'？'

# print(transfer_path2sentence('?y|||<中文名>|||"两弹一星"	?y|||<原指>|||?x'))
def transfer_path2sentence_old(path):
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

def transfer_data(x_sent,x_sample,max_length,bert_vocab_path):
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
        # clear_sample=x_sample[i].replace('>','').replace('<','').replace('_','').replace('|||', '').replace('\t', '').replace('  ','').replace('?', '').replace('ans','？')
        clear_sample=transfer_path2sentence(x_sample[i])
        # print(clear_sample)
        indice,segment=tokenizer.encode(first=x_sent[i],second=clear_sample,max_len=max_length)
        x_indices.append(indice)
        x_segments.append(segment)
        # print(indice)
        # print(segment)
    # print('over transfer train/valid data')

    return x_indices,x_segments
