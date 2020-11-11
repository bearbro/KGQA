'''
对于不符合路径要求的路径，使用问题链接候选路径，获取正确答案
'''
from keras.metrics import binary_accuracy
import sys
# sys.path.insert(0,'/Users/mingrongtang/PycharmProjects/TensorFlow/MyExperiment/rel_and_path_similarity/data/maxbert_pred_data')
# sys.path.insert(0,'/home/unsw1/mrtang/MyExperiment/rel_and_path_similarity/data/maxbert_pred_data')
sys.path.insert(0,'/home/hbxiong/QA2/rel_and_path_similarity')

from some_function_maxbert import transfer_data
from keras_bert import load_trained_model_from_checkpoint
import keras
import json
from py2neo import Graph
import tqdm
import codecs
from keras_bert import Tokenizer,get_base_dict
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"#设置使用0号GPU
import numpy as np
import re
from elasticsearch import Elasticsearch
import keras.backend as k
from keras.layers import Layer
import tensorflow as tf
from keras import backend as K
from get_query_result_new import get_all_F1 
from pprint import pprint
import os
import time
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


model_tag='ckpt_similarity_bert_wwm_ext_f1.hdf5'
tag='bret_ext_pred_data_f12020-10-25-17-05-06'

print(tag)
class Config:

    data_dir = r'./data'
    train_data_path = os.path.join(data_dir, 'path_data/train_data_sample.json')
    valid_data_path = os.path.join(data_dir, 'path_data/valid_data_sample.json')
    trainvalid_data_path = os.path.join(data_dir, 'path_data/trainvalid_data_sample.json')

    linking_data_path = '../result_new_linking-no_n_59,r_0.8321,line_right_recall_0.9230,avg_n_2.6919.json'

    # bert_path
    # bert_path = '../../bert/bert_wwm_ext'  # 百度
    # bert_path = r'C:\Users\bear\OneDrive\ccks2020-onedrive\ccks2020\bert\tf-bert_wwm_ext'  # room
    bert_path = '/home/hbxiong/ccks/bert/tf-bert_wwm_ext'  # lab
    # bert_path = r'../../../ccks/bert/tf-bert_wwm_ext'  # colab
    bert_config_path = os.path.join(bert_path, 'bert_config.json')
    bert_ckpt_path = os.path.join(bert_path, 'bert_model.ckpt')
    bert_vocab_path = os.path.join(bert_path, 'vocab.txt')


    result_path = './data/%s'%tag
    true_answer_path = os.path.join(result_path, 'true_path_score.json')  # 模型在测试集正确路径上的预测得分
    ok_result_path = os.path.join(result_path, 'ok_result.txt')  # 保存为txt，使得发生错误时可以在当前问题继续训练
    pred_result_path = os.path.join(result_path, 'pred_result.txt')

    similarity_ckpt_path = './ckpt/%s'%model_tag  # 模型训练后，模型参数存储路径

    batch_size = 64
    epoches = 100
    learning_rate = 1e-5  # 2e-5
    neg_sample_number = 5
    max_length = 100  # neg3:64;100


config = Config()
for i in ['./ckpt',config.result_path]:
    if not os.path.exists(i):
        os.mkdir(i)

pprint(vars(Config))





def _get_true_result(true_path='./score/test.txt'):
    '''
    返回正确的题目id和答案的字典
    :param true_path: id2ans
    :return:
    '''
    with open(true_path,'r',encoding='utf-8') as reader:
        lines=reader.readlines()
        true_answer = {}
        sentence={}
        for i in range(766):
            true_answer[str(i)]=lines[i*4+2].strip('\n').split('\t')
            sentence[str(i)]=lines[i*4].strip('\n').split(':')[1].strip('?？')+'？'
            # print(true_answer[str(i)])
    return true_answer,sentence

def _get_pred_score(pred_path):
    id2score={}
    id2sentence={}
    id2path={}
    with open(pred_path, 'r', encoding='utf-8') as predict_reader:
        lines = predict_reader.readlines()
        for i in range(int(len(lines) / 2)):
            id = lines[i * 2].split(':')[0].lstrip('q')
            id2sentence[id]= lines[i * 2].split(':')[1].strip('\n')
            id2score[id]= lines[i * 2 + 1].split('---')[0]
            id2path[id]=lines[i * 2 + 1].split('---')[0].strip('\n')
    return id2sentence,id2score,id2path

def basic_network():
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path,
                                                    config.bert_ckpt_path,
                                                    seq_len=config.max_length,
                                                    training=False,
                                                    trainable=True)
    # 选择性某些层进行训练
    # bert_model.summary()
    # for l in bert_model.layers:
    #     # print(l)
    #     l.trainable = True
    x1 = keras.layers.Input(shape=(config.max_length,))
    x2 = keras.layers.Input(shape=(config.max_length,))
    bert_out = bert_model([x1, x2])  # 输出维度为(batch_size,max_length,768)
    # dense=bert_model.get_layer('NSP-Dense')
    bert_out = keras.layers.Lambda(lambda bert_out: bert_out[:, 0])(bert_out)
    # bert_out = keras.layers.Dropout(0.5)(bert_out)
    outputs = keras.layers.Dense(1, activation='sigmoid')(bert_out)

    model = keras.models.Model([x1, x2], outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


class EarlyStopByF1(keras.callbacks.Callback):
    def __init__(self,
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopByF1, self).__init__()

        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best = 0

        # 服务器上
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater  # >

    def on_epoch_end(self, epoch, logs=None):
        with open(config.valid_data_path, 'r', encoding='utf-8') as valid_reader:
            valid_data = json.load(valid_reader)
            valid_x_sent = valid_data['x_sent']
            valid_x_sample = valid_data['x_sample']
            valid_x_indices, valid_x_segments = transfer_data(valid_x_sent, valid_x_sample, config.max_length,config.bert_vocab_path)
        predict = self.model.predict([valid_x_indices, valid_x_segments]).ravel()  # validation_data[0])
        print('predict---', len(predict))
        f1 = 0
        with open('data/valid_f1_path.json', 'r', encoding='utf-8') as reader:
            data = json.load(reader)
            sent_split_list = data['sent_split_list']
            f1_list = data['f1_list']
        left = 0
        for i in range(len(sent_split_list)):
            sent_i_y = predict[left:sent_split_list[i]]
            max_index = left + np.argmax(sent_i_y)
            f1 += f1_list[max_index]
            left = sent_split_list[i]
        f1 = f1 / len(sent_split_list)
        print('f1-----', f1)
        # current = self.get_monitor_value(logs)

        if self.monitor_op(f1, self.best):  # f1 >best
            a = self.best
            self.best = f1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            print("f1 improved from %.5f to %.5f" % (a, self.best))
        else:
            print("f1 not improved from %.5f " % self.best)
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    
 

def append_threshold(threshold,pred_path,append_pred_path,append_ok_path):
    '''
    句子得分最大时，且大于某个阈值时不进行此步骤
    '''
    id2sentence,id2score,id2path=_get_pred_score(pred_path)
    _,true_id2sentence=_get_true_result()
    model = basic_network()
    model.load_weights(config.similarity_ckpt_path)
    #graph = Graph("http://47.114.86.211:57474", username='neo4j', password='pass')
    #url = {"host":'47.114.86.211', "port":59200, "timeout": 6000}
    graph = Graph("http://59.78.194.63:37474", username='neo4j', password='pass')
    url = {"host":'59.78.194.63', "port":59200, "timeout": 6000}#lab
    es = Elasticsearch([url])

    append_writer=open(append_pred_path,'w',encoding='utf-8')
    ok_paths=[]
    for i in range(766):
        print('问题%d:'%i)
        try:
            if float(id2score[str(i)])<threshold:
                x_sample=_link_a_sentence(id2sentence[str(i)],es)#输入句子，获取句子链接的one hop path
                max_score,max_path,ok_path=_pred_sentence(model,graph,x_sample,id2sentence[str(i)],i)
                append_writer.write('q'+str(i)+':'+id2sentence[str(i)]+'\n')
                append_writer.write(str(max_score)+'---'+max_path+'\n')
                print('sentence',str(i),str(max_score),'---',max_path)
                ok_paths.append(ok_path)
                append_writer.flush()
        except:
            x_sample = _link_a_sentence(true_id2sentence[str(i)], es)  # 输入句子，获取句子链接的one hop path
            max_score, max_path, ok_path = _pred_sentence(model, graph, x_sample, true_id2sentence[str(i)], i)
            append_writer.write('q' + str(i) + ':' + true_id2sentence[str(i)] + '\n')
            append_writer.write(str(max_score) + '---' + max_path + '\n')
            print('sentence', str(i), str(max_score), '---', max_path)
            ok_paths.append(ok_path)
            append_writer.flush()

    ok_writer=open(append_ok_path,'w',encoding='utf-8')
    json.dump(ok_paths,ok_writer,ensure_ascii=False)

def _get_next_hop_path_old(graph,path):
    path=path.replace("\'","\\\'").replace('\\','\\\\')
    path_list=path.split('|||')
    print('now query path---',path_list)
    assert len(path_list)==3
    cypher='match '
    sample=[]
    if  path_list[0]=='?x':
        cypher1=cypher+"(y)-[rel1:Relation{name:'"+path_list[1]+"'}]->(ent1:Entity{name:'"+path_list[2]+"'}) match (y)-[rel2]->(x) where ent1.name<>x.name return distinct rel2.name"
        # print(cypher1)
        answers = graph.run(cypher1).data()
        for ans in answers:
            one_ent=path.replace('?x','?y')+'\t'+'?y|||'+ans['rel2.name']+'|||?x'
            # print(one_ent)
            sample.append(one_ent)

        cypher2 = cypher + "(y)-[rel1:Relation{name:'" + path_list[1] + "'}]->(ent1:Entity{name:'" + path_list[2] + "'}) match (x)-[rel2]->(y) return distinct rel2.name"
        # print(cypher2)
        answers = graph.run(cypher2).data()
        for ans in answers:
            one_ent = path.replace('?x', '?y')+ '\t' +'?x|||' + ans['rel2.name'] + '|||?y'
            # print(one_ent)
            sample.append(one_ent)

        cypher3=cypher+"(x)-[rel1:Relation{name:'"+path_list[1]+"'}]->(:Entity{name:'"+path_list[2]+"'}) match (y)-[rel2]->(x) return distinct rel2.name,y.name"
        # print(cypher3)
        answers = graph.run(cypher3).data()
        for ans in answers:
            two_ent1=ans['y.name']+'|||'+ans['rel2.name']+'|||?x'+'\t'+path
            # print(two_ent)
            sample.append(two_ent1)

        cypher4 = cypher + "(x)-[rel1:Relation{name:'" + path_list[1] + "'}]->(ent1:Entity{name:'" + path_list[2] + "'}) match (x)-[rel2]->(y) where ent1.name<>y.name return distinct rel2.name,y.name"
        # print(cypher4)
        answers = graph.run(cypher4).data()
        for ans in answers:
            two_ent1= path + '\t' +'?x|||' + ans['rel2.name'] + '|||'+ans['y.name']
            # print(two_ent)
            sample.append(two_ent1)

    if path_list[2]=='?x':
        cypher5=cypher+"(:Entity{name:'"+path_list[0]+"'})-[rel1:Relation{name:'"+path_list[1]+"'}]->(y) match (y)-[rel2]->(x) return distinct rel2.name"
        # print(cypher5)
        answers = graph.run(cypher5).data()
        for ans in answers:
            one_ent=path.replace('?x','?y')+'\t'+'?y|||'+ans['rel2.name']+'|||?x'
            # print(one_ent)
            sample.append(one_ent)

        cypher6 = cypher + "(ent1:Entity{name:'" + path_list[0] + "'})-[rel1:Relation{name:'" + path_list[1] + "'}]->(y) match (x)-[rel2]->(y) where ent1.name<>x.name return distinct rel2.name"
        # print(cypher6)
        answers = graph.run(cypher6).data()
        for ans in answers:
            one_ent = path.replace('?x', '?y') + '\t' + '?x|||' + ans['rel2.name'] + '|||?y'
            # print(one_ent)
            sample.append(one_ent)

        cypher7=cypher+"(ent1:Entity{name:'" +path_list[0] + "'})-[rel1:Relation{name:'" + path_list[1] + "'}]->(x) match (y)-[rel2]->(x) where ent1.name<>y.name return distinct rel2.name,y.name"
        # print(cypher7)
        answers = graph.run(cypher7).data()
        for ans in answers:
            two_ent1=path+'\t'+ans['y.name']+'|||'+ans['rel2.name']+'|||?x'
            # print(two_ent)
            sample.append(two_ent1)

        cypher8 = cypher + "(:Entity{name:'" + path_list[0] + "'})-[rel1:Relation{name:'" + path_list[1] + "'}]->(x) match (x)-[rel2]->(y) return distinct rel2.name,y.name"
        # print(cypher8)
        answers = graph.run(cypher8).data()
        for ans in answers:
            two_ent = path + '\t' + '?x|||' + ans['rel2.name'] + '|||'+ans['y.name']
            # print(two_ent)
            sample.append(two_ent)

    return sample


def _get_next_hop_path_two(graph,path):
    path=path.replace('\\','\\\\').replace("\'","\\\'")
    # 输入一跳的路径，返回两跳的路径（包括一跳两个实体）
    # 若一跳路径答案不为属性，则选择一跳后续的两个实体路径
    # 若第一跳中关系为类型，则无第二跳
    path_list=path.split('|||')
    print('now query path---',path_list)
    assert len(path_list)==3
    if path_list[1]=='<类型>' and path_list[2]=='?x':
        return []
    cypher='match '
    sample=[]
    if  path_list[0]=='?x':
        cypher1=cypher+"(y)-[rel1:Relation{name:'"+path_list[1]+"'}]->(ent1:Entity{name:'"+path_list[2]+"'}) match (y)-[rel2]->(x) where rel2.name<>'<类型>' and ent1.name<>x.name return distinct rel2.name"
        # print(cypher1)
        answers = graph.run(cypher1).data()
        for ans in answers:
            one_ent=path.replace('?x','?y')+'\t'+'?y|||'+ans['rel2.name']+'|||?x'
            # print(one_ent)
            sample.append(one_ent)

        cypher2 = cypher + "(y)-[rel1:Relation{name:'" + path_list[1] + "'}]->(ent1:Entity{name:'" + path_list[2] + "'}) match (x)-[rel2]->(y) return distinct rel2.name"
        # print(cypher2)
        answers = graph.run(cypher2).data()
        for ans in answers:
            one_ent = path.replace('?x', '?y')+ '\t' +'?x|||' + ans['rel2.name'] + '|||?y'
            # print(one_ent)
            sample.append(one_ent)

        cypher3=cypher+"(x)-[rel1:Relation{name:'"+path_list[1]+"'}]->(:Entity{name:'"+path_list[2]+"'}) match (y)-[rel2]->(x) return distinct rel2.name,y.name"
        # print(cypher3)


        answers = graph.run(cypher3).data()
        if len(answers)<1000:
            for ans in answers:
                two_ent1=ans['y.name']+'|||'+ans['rel2.name']+'|||?x'+'\t'+path
                # print(two_ent)
                sample.append(two_ent1)

        cypher4 = cypher + "(x)-[rel1:Relation{name:'" + path_list[1] + "'}]->(ent1:Entity{name:'" + path_list[2] + "'}) match (x)-[rel2]->(y) where ent1.name<>y.name return distinct rel2.name,y.name"
        # print(cypher4)
        if len(answers)<1000:
            answers = graph.run(cypher4).data()
            for ans in answers:
                two_ent1= path + '\t' +'?x|||' + ans['rel2.name'] + '|||'+ans['y.name']
                # print(two_ent)
                sample.append(two_ent1)

    if path_list[2]=='?x':
        cypher5=cypher+"(:Entity{name:'"+path_list[0]+"'})-[rel1:Relation{name:'"+path_list[1]+"'}]->(y) match (y)-[rel2]->(x) where rel2.name<>'<类型>' return distinct rel2.name"
        # print(cypher5)
        answers = graph.run(cypher5).data()
        for ans in answers:
            one_ent=path.replace('?x','?y')+'\t'+'?y|||'+ans['rel2.name']+'|||?x'
            # print(one_ent)
            sample.append(one_ent)

        cypher6 = cypher + "(ent1:Entity{name:'" + path_list[0] + "'})-[rel1:Relation{name:'" + path_list[1] + "'}]->(y) match (x)-[rel2]->(y) where ent1.name<>x.name return distinct rel2.name"
        # print(cypher6)
        answers = graph.run(cypher6).data()
        for ans in answers:
            one_ent = path.replace('?x', '?y') + '\t' + '?x|||' + ans['rel2.name'] + '|||?y'
            # print(one_ent)
            sample.append(one_ent)


        if path_list[1] != '<国籍>' and path_list[1] != '<类型>' and path_list[1]!='<性别>':
            cypher7=cypher+"(ent1:Entity{name:'" +path_list[0] + "'})-[rel1:Relation{name:'" + path_list[1] + "'}]->(x) match (y)-[rel2]->(x) where ent1.name<>y.name return distinct rel2.name,y.name,x.name"
            # print(cypher7)
            answers = graph.run(cypher7).data()
            if len(answers) <= 1000:
                for ans in answers:
                    if ans['x.name'].startswith('"'):
                        continue
                    two_ent1=path+'\t'+ans['y.name']+'|||'+ans['rel2.name']+'|||?x'
                    # print(two_ent)
                    sample.append(two_ent1)

            cypher8 = cypher + "(:Entity{name:'" + path_list[0] + "'})-[rel1:Relation{name:'" + path_list[1] + "'}]->(x) match (x)-[rel2]->(y) return distinct rel2.name,y.name,x.name"
            # print(cypher8)
            answers = graph.run(cypher8).data()
            if len(answers) <= 1000:
                for ans in answers:
                    if ans['x.name'].startswith('"'):
                        continue
                    two_ent = path + '\t' + '?x|||' + ans['rel2.name'] + '|||'+ans['y.name']
                    # print(two_ent)
                    sample.append(two_ent)
            print('two hop path len',len(sample),'---',sample[:5])

    return sample


def _pred_sentence(model,graph,x_sample,sentence,id):
    beamsearch=10
    x_sent = [sentence] * len(x_sample)
    x_indices, x_segments = transfer_data(x_sent, x_sample, config.max_length,config.bert_vocab_path)

    result = model.predict([x_indices, x_segments], batch_size=config.batch_size)
    result = result.ravel()  # 将result展平变为一维数组
    # 将预测的正确答案写入pred_result文件
    assert len(x_sample) == len(result)

    top_beamsearch_one_hop = []  # 记录前k个候选的一跳路径
    all_max_score = result[result.argmax(-1)]  # 记录总体的最大得分
    all_max_path = x_sample[result.argmax(-1)]
    for i in range(beamsearch):
        assert len(result) != 0
        right_index = result.argmax(-1)
        now = {}
        now['level'] = '1'  # level的值表示path涉及的跳数
        now['score'] = str(result[right_index])
        now['path'] = x_sample[right_index]
        result = np.delete(result, [right_index])
        x_sample = np.delete(x_sample, [right_index])

        top_beamsearch_one_hop.append(now)
        print('one hop top ', beamsearch, ': ', now)

    top_beamsearch_two_hop = []  # 记录包含前k个一跳路径，且得分大于其包含的一跳路径的两跳候选路径
    for top in top_beamsearch_one_hop:
        max = float(top['score'])
        x_sample_next = _get_next_hop_path_two(graph, top['path'])

        if len(x_sample_next) == 0:  # 若没有第二跳路径，则看下一个
            continue

        x_sent_next = [sentence] * len(x_sample_next)
        x_indices, x_segments = transfer_data(x_sent_next, x_sample_next, config.max_length,config.bert_vocab_path)

        result = model.predict([x_indices, x_segments], batch_size=config.batch_size)
        result = result.ravel()  # 将result展平变为一维数组
        # print('two_hop---',result)
        for i in range(len(result)):
            if result[i] > max:
                if result[i] > all_max_score:
                    all_max_score = result[i]
                    all_max_path = x_sample_next[i]

                now = {}
                now['level'] = '2'
                now['score'] = str(result[i])
                now['path'] = x_sample_next[i]
                top_beamsearch_two_hop.append(now)
                print('two hop top :', now)  # 有符合条件加入twohop的path
    # 在two_hop的基础上向下找，直到没有可以加入的更大的得分

    ok_path={}
    ok_path['id']=id
    ok_path['sentence']=sentence
    ok_path['paths']=top_beamsearch_one_hop+top_beamsearch_two_hop

    return all_max_score,all_max_path,ok_path


def _link_a_sentence(sentence,es):
    '''
    输入句子，使用es链接path，返回不包含<类型>关系的128个one hop path
    :param sentence:
    :param es:
    :return:
    '''
    str=sentence
    pred_triples=[]
    #将日期格式化
    if re.search(r'\d+年\d+',str)!=None:
        str=str.replace('年','-')
        sub_month = re.search(r'\d+月', str).group()
        if len(sub_month) == 2:
            str = str.replace(sub_month, '0' + sub_month)
        # print('replace--',men)
    if re.search(r'\d+月\d+日',str)!=None:
        sub_month=re.search(r'\d+月', str).group()
        if len(sub_month)==2:
            str=str.replace(sub_month,'0'+sub_month)
        str=str.replace('月','-')
        # print(men)
        sub_day=re.search(r'\d+日',str).group()
        if len(sub_day)==2:
            str=str.replace(sub_day,'0'+sub_day)
        str=str.replace('日','')


    #去重可能影响index经过的不重要的字和词
    stop_words=['是什么','指什么','什么','有','哪些','哪里','如何','怎样','谁','多少','是','哪']
    for word in stop_words:
        str=str.replace(word,'')

    print('query---', str)

    #mention使用es检索,全匹配属性，或者分词匹配不同的索引
    dsl = {
        'size': 300,
        'query': {
            'match': {
                'clear_data':{
                    'query':str  # 对应字段名：检索字段
                }
            }
        }
    }
    r = es.search(index="ccks2019_path_index_no()", body=dsl)
    k=0
    if r['hits']['total']['value']!=0:
        print('查询结果数---', r['hits']['total']['value'])
        for item in r['hits']['hits']:
            print(item['_source']['old_data'])
            if len(re.findall('类型',item['_source']['old_data']))==0:
                pred_triples.append(item['_source']['old_data'])
                k=k+1
            if k>=30:
                break

    # r = es.search(index="ccks2019_path_index", body=dsl)
    # k = 0
    # if r['hits']['total']['value'] != 0:
    #     print('查询结果数---', r['hits']['total']['value'])
    #     for item in r['hits']['hits']:
    #         print('old data---', item['_source']['old_data'])
    #         print('clear data---', item['_source']['clear_data'])
    #         if len(re.findall('类型', item['_source']['old_data'])) == 0:
    #             pred_triples.append(item['_source']['old_data'])
    #             k = k + 1
    #         if k >= 80:
    #             break

    pred_triples = list(set(pred_triples))
    return pred_triples
    
dir, filename = os.path.split(config.pred_result_path)    

pred_path=config.pred_result_path
append_dir=dir+'_pathlink'
if not os.path.exists(append_dir):
    os.mkdir(append_dir)
append_pred=os.path.join(append_dir,'pred_result.txt')
append_ok=os.path.join(append_dir,'ok_result.json')
append_threshold(2,pred_path=pred_path,append_pred_path=append_pred,append_ok_path=append_ok)
get_all_F1(append_pred)
