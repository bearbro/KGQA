'''
生成实体消歧所需的数据
'''
def get_url():
    url = {"host": '59.78.194.63', "port": 59200, "timeout": 3000}#lab
    # url = {"host": '42.192.36.128', "port": 6692, "timeout": 1500}#room
    return url

import json,shutil,os
def extract_mention(question,label):
    label=[0]+label+[0]
    question_='-'+question+'-'
    begin=False
    str = ''
    mentions=[]
    for p_i in range(len(label)):
        # if begin==False and int(label[p_i])==2:
        if int(label[p_i])==2:
            begin=True
            str=str + question_[p_i]
        if begin and int(label[p_i])==1:
            str = str + question_[p_i]
        if begin and int(label[p_i])==0:
            begin=False
            mentions.append(str)
            str=''
    if len(mentions)==0:
        for p_i in range(len(label)):
            if int(label[p_i])==1:
                begin=True
                str=str + question_[p_i]
            if begin and int(label[p_i]) == 0:
                begin = False
                mentions.append(str)
                str = ''
    return mentions

def label_to_word(pre_path):
    with open(pre_path, 'r', encoding='utf-8') as file:
        file_list = file.readlines()
        true_entity = []
        pred_entity = []
        sentences = []
        for i in range(len(file_list)//3):
            sentence = file_list[i * 3].strip('\n')
            sentences.append(sentence)
            true_y = file_list[i * 3 + 1].strip('\n').strip('[]').split(',')
            begin = False
            str = ''
            a_true_entity = []
            for t_i in range(len(true_y)):
                if int(true_y[t_i]) == 2 or int(true_y[t_i]) == 1:#111
                    begin = True
                    str = str + sentence[t_i]
                if begin and int(true_y[t_i]) == 0:
                    begin = False
                    a_true_entity.append(str)
                    str = ''

            pred_y = file_list[i * 3 + 2].strip('\n').strip('[]').split(',')
            a_pred_entity=extract_mention(sentence,pred_y)
            # if true_y!=pred_y:
            #     print(sentence)
            #     print(true_y)
            #     print(pred_y)
            #     print(a_true_entity)
            #     print(a_pred_entity)
            true_entity.append(a_true_entity)
            pred_entity.append(a_pred_entity)
    return sentences, true_entity, pred_entity

def label_to_word_old(pre_path):
    '''
    从预测的序列标注结果，获取标注的实体集
    :param pre_path: 序列标注结果文件
    :return: 句子，真实实体，预测实体
    '''
    with open(pre_path,'r',encoding='utf-8') as file:
        file_list=file.readlines()
        true_entity=[]
        pred_entity=[]
        sentences=[]
        for i in range(len(file_list)//3):
            sentence= file_list[i * 3].strip('\n')
            sentences.append(sentence)
            true_y= file_list[i * 3 + 1].strip('\n').strip('[]').split(',')
            begin=False
            str=''
            a_true_entity=[]
            for t_i in range(len(true_y)):
                if int(true_y[t_i])==2 or int(true_y[t_i])==1:
                    begin=True
                    str = str + sentence[t_i]
                if begin and int(true_y[t_i])==0:
                    begin=False
                    a_true_entity.append(str)
                    str=''

            pred_y = file_list[i * 3 + 2].strip('\n').strip('[]').split(',')
            begin=False
            str = ''
            a_pred_entity=[]
            for p_i in range(len(pred_y)):
                '''
                211211，预测为一个实体，类似于企业家马云？？,不把111当做实体...?是否把识别出来的111当做一个单独的实体。
                '''
                # if int(pred_y[p_i])==2:
                #     begin=True
                #     str=str + sentence[p_i]
                # if begin and int(pred_y[p_i])==1:
                #     str = str + sentence[p_i]
                # if begin and int(pred_y[p_i])==0:
                #     begin=False
                #     a_pred_entity.append(str)
                #     str=''
                if int(pred_y[p_i])==2 or int(pred_y[p_i])==1:
                    begin=True
                    str=str + sentence[p_i]
                # if begin and int(pred_y[p_i])==1:
                #     str = str + sentence[p_i]
                # if begin and int(pred_y[p_i])==2:
                #     str = str + sentence[p_i]
                if begin and int(pred_y[p_i])==0:
                    begin=False
                    a_pred_entity.append(str)
                    str=''
            # if true_y!=pred_y:
            #     print(sentence)
            #     print(true_y)
            #     print(pred_y)
            # print(a_true_entity)
            # print(a_pred_entity)
            # print(set(a_true_entity)&set(a_pred_entity))
            true_entity.append(a_true_entity)
            pred_entity.append(a_pred_entity)
        return sentences,true_entity,pred_entity

def write_entity(pre_entity_path,sentence,true_entity,pred_entity):
    '''
    将实体结果写入json文件
    :param pre_entity_path:
    :param sentence:
    :param true_entity:
    :param pred_entity:
    :return:
    '''
    data={'sentence':sentence,
          'true_mention':true_entity,
          'pred_mention':pred_entity
          }
    with open(pre_entity_path,'w',encoding='utf-8') as entity_file:
        # for i in range(len(sentence)):
        #     entity_file.write(sentence[i]+'\n')
        #     entity_file.write(str(true_entity[i])+'\n')
        #     entity_file.write(str(pred_entity[i])+'\n')
        json.dump(data,entity_file,ensure_ascii=False)


def macro_valuate_set(true,pred):
    '''
    true和pred为集合的列表，先计算每个集合的recall和precision,F1，然后求平均
    :param true:
    :param pred:
    :return:
    '''
    assert len(true)==len(pred)
    num_sample=len(true)
    recall_list=[]
    precision_list=[]
    F1_list=[]
    for i in range(num_sample):
        pos=set(true[i])&set(pred[i])
        try:
            recall=len(pos)/len(true[i])
        except:
            recall=1
            print('mention true[%d] == []'%i)
        try:
            precision=len(pos)/len(pred[i])
        except:
            precision=0
        try:
            F1=2*recall*precision/(recall+precision)
        except:
            F1=0
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append(F1)
    # print(recall_list,precision_list,F1_list)
    print(sum(recall_list)/num_sample,sum(precision_list)/num_sample,sum(F1_list)/num_sample)
    return sum(recall_list)/num_sample,sum(precision_list)/num_sample,sum(F1_list)/num_sample

def micro_valuate_set(true,pred):
    assert len(true) == len(pred)
    num_sample = len(true)
    true_list = []
    positive_list = []
    true_positive_list = []
    for i in range(num_sample):
        pos = set(true[i]) & set(pred[i])
        true_list.append(len(true[i]))
        positive_list.append(len(pred[i]))
        true_positive_list.append(len(pos))
    recall=sum(true_positive_list)/sum(true_list)
    precision=sum(true_positive_list)/sum(positive_list)
    F1=2*recall*precision/(recall+precision)
    print(recall,precision,F1)
    return recall,precision,F1

def transfer(old_path,new_path):
    '''
    调整一些sent2ment文件格式
    '''    
    with open(old_path,'r',encoding='utf-8') as reader:
        transfers=[]
        data=json.load(reader)
        sentenes=data['sentence']
        mentions=data['pred_mention']
        for i in range(len(sentenes)):
            transfer={}
            transfer['id']=i
            transfer['sentence']=sentenes[i]
            transfer['mention']=mentions[i]
            transfers.append(transfer)

    with open(new_path,'w',encoding='utf-8') as writer:
        json.dump(transfers,writer,ensure_ascii=False)


'''
实体链接，使用实体抽取文件生成的json格式文件
'''

import json
from elasticsearch import Elasticsearch
import re
import jieba


def read_mention2entity(mention2entity_path):
    '''
    输入比赛给定的字典文件，返回字典,一个mention，对应多个entity组成的列表
    :param mention2entity_path:
    :return:
    '''
    men2ent_dict={}
    with open(mention2entity_path,'r',encoding='utf-8') as file:
        file_list=file.readlines()
        for line in file_list:
            line_list=line.strip('\n').split('\t')
            mention=line_list[0]
            entity=line_list[1]
            index=int(line_list[2])-1
            print(mention,index,entity)
            try:
                men2ent_dict[mention].insert(index,entity)
            except:
                men2ent_dict[mention]=[]
                men2ent_dict[mention].insert(index, entity)
        with open('./data/entity/men2ent_dict.json','w',encoding='utf-8') as write_file:
            json.dump(men2ent_dict,write_file,ensure_ascii=False)
        return men2ent_dict

def get_mention2entity(dict_path='./data/entity/men2ent_dict.json'):
    with open(dict_path,'r',encoding='utf-8') as dict_file:
        dict=json.load(dict_file)
        return dict

def read_mention2property(pro_path='./data/entity_and_property.txt',dict_path='./data/men2pro_dict.json'):
    dict={}
    with open(pro_path,'r',encoding='utf-8') as reader:
        for line in reader.readlines():
            if line.startswith('"'):
                value=line.strip('\n')
                key=value.strip('"')
                dict[key]=value
                # print(dict)
    with open(dict_path,'w',encoding='utf-8') as writer:
        json.dump(dict,writer,ensure_ascii=False)
    print('over reade mention2peoperty----')
# read_mention2property()

def get_mention2property(dict_path='./data/men2pro_dict.json'):
    with open(dict_path,'r',encoding='utf-8') as reader:
        dict=json.load(reader)
    return dict

def do_linking(men2ent_dict,men2pro_dict,mention_path,write_path):
    '''
    根据日期，给定词典，属性获取实体链接，其他表示为无实体链接，根据三元组链接来获取候选路径
    :param men2ent_dict:
    :param mention_path:
    :return:
    '''

    url = get_url()
    es = Elasticsearch([url])
    not_entity=0
    with open(mention_path,'r',encoding='utf-8') as file:
        data=json.load(file)
        result = []
        for item in data:
            a_sentence={}
            a_sentence['id']=item['id']
            a_sentence['sentence']=item['sentence']
            a_sentence['pred_entity']=[]
            a_sentence['mention']=item['mention']
            # result.append(a_sentence)
            # continue
            #首先格式化日期，便于之后的实体链接
            for men in item['mention']:
                #将日期格式化
                if re.search(r'\d+年\d+',men)!=None:
                    men=men.replace('年','-')
                    sub_month = re.search(r'\d+月', men).group()
                    if len(sub_month) == 2:#2
                        men = men.replace(sub_month, '0' + sub_month)
                    # print('replace--',men)
                if re.search(r'\d+月\d+日',men)!=None:
                    sub_month=re.search(r'\d+月', men).group()
                    if len(sub_month)==2:#2
                        men=men.replace(sub_month,'0'+sub_month)
                    men=men.replace('月','-')
                    # print(men)
                    sub_day=re.search(r'\d+日',men).group()
                    if len(sub_day)==2:#2
                        men=men.replace(sub_day,'0'+sub_day)
                    men=men.replace('日','')
                    # print('replace---',men)
                if men.endswith('年') :#将1991年转为1991
                    print(men)
                    # num=list(re.findall('\d{1}', men))[0]
                    a_sentence['mention'].append(men.rstrip('年'))
                    # item['mention'].append(men)
                if len(men)==4 and len(re.findall('\d{1}',men))==4:
                    print(men)
                    year=men+'年'
                    try:
                        for item in men2ent_dict[year]:
                            a_sentence['pred_entity'].append('<' + item + '>')
                    except:
                        print('not find mention in ent dict')


                # #完全匹配实体查询字典----！！！词典中的某些词对应知识库中无实体
                try:
                    for item in men2ent_dict[men]:
                        a_sentence['pred_entity'].append('<'+item+'>')
                except:
                    print('not find mention in ent dict')

                # #当mention不能完全链接到实体的时候，考虑当前的mention范围过大，进行分词之后再完全匹配
                # cut = thulac.thulac()
                # cut_list = []
                # for item in cut.cut(men, text=True).split(' '):
                #     cut_list.append(item.split('_')[0])
                # if len(cut_list)==2:
                #     print('jieba_list---',cut_list)
                #     for cut_men in cut_list:
                #         try:
                #             for item in men2ent_dict[cut_men]:
                #                 a_sentence['pred_entity'].append('<'+item+'>')
                #         except:
                #             print('not find mention in dict')

                #完全匹配属性查询字典
                try:
                    a_sentence['pred_entity'].append(men2pro_dict[men])
                except:
                    print('not find mention in pro dict')

                # a_sentence['pred_entity'] = list(set(a_sentence['pred_entity']))
                # if len(a_sentence['pred_entity'])>0:
                #     continue


                #模糊匹配实体
                dsl = {
                    'size': 10,
                    'query': {
                        'match': {
                            'clear_data': {
                                'query': men  # 对应字段名：检索字段
                            }
                        }
                    }
                }
                r = es.search(index="ccks2019_entity_and_property_no()_index", body=dsl)
                if r['hits']['total']['value'] != 0:
                    # print('查询结果数---', r['hits']['total']['value'])
                    max = r['hits']['hits'][0]['_score']
                    for item in r['hits']['hits']:
                        score = item['_score']
                        # 只返回得分最高的
                        if score < max:
                            break
                        entity = item['_source']['old_data']
                        # a_sentence['pred_entity'].append('<'+entity+'>')
                        if entity.startswith('<'):
                            a_sentence['pred_entity'].append(entity)

                # mention链接到属性，以及其他的完全匹配的实体，但是最好的应该是只链接实体，且完全对应及符号也应算入对应关系
                dsl = {
                    'size': 10,
                    'query': {
                        'match': {
                            'clear_data': {
                                'query': men  # 对应字段名：检索字段
                            }
                        }
                    }
                }
                r = es.search(index="ccks2019_entity_and_property_index", body=dsl)
                if r['hits']['total']['value'] != 0:
                    # print('查询结果数---', r['hits']['total']['value'])
                    max = r['hits']['hits'][0]['_score']
                    for item in r['hits']['hits']:
                        score = item['_score']
                        # 只返回得分最高的
                        if score < max:
                            break
                        entity = item['_source']['old_data']
                        # a_sentence['pred_entity'].append(entity)
                        if entity.startswith('<'):
                            a_sentence['pred_entity'].append(entity)
                        # if entity.startswith('"'):
                        #     a_sentence['pred_entity'].append()

            a_sentence['pred_entity'] = list(set(a_sentence['pred_entity']))



            #句子匹配查询实体和属性
            dsl = {
                'size': 10,
                'query': {
                    'match': {
                        'clear_data': {
                            'query': a_sentence['sentence']  # 对应字段名：检索字段
                        }
                    }
                }
            }
            r = es.search(index="ccks2019_entity_and_property_index", body=dsl)
            if r['hits']['total']['value'] != 0:
                # print('查询结果数---', r['hits']['total']['value'])
                max = r['hits']['hits'][0]['_score']
                for item in r['hits']['hits']:
                    score = item['_score']
                    # 只返回得分最高的
                    if score < max:
                        break
                    entity = item['_source']['old_data']
                    a_sentence['pred_entity'].append(entity)

            a_sentence['pred_entity']=list(set(a_sentence['pred_entity']))

            
            if len(a_sentence['pred_entity'])==0:#统计链接实体为0的情况
                not_entity+=1

            print('q',str(a_sentence['id']),':',a_sentence['sentence'],'---',len(a_sentence['pred_entity']),'---',a_sentence['pred_entity'])
            result.append(a_sentence)
    with open(write_path,'w',encoding='utf-8') as write_file:
        json.dump(result,write_file,ensure_ascii=False)
    print('over entity linking,not_entity number---',not_entity)

def evaluate_entity_linking(true_entity_path,pred_entity_path,no_right_linking_file=None):
    '''
    输入正确的实体和预测的实体，以一个句子为单位，求所有真实实体和预测的top k个实体的交集，算recall
    :param true_entity_path:
    :param pred_entity_path:
    :return:
    '''
    true_data=[]
    with open(true_entity_path,'r',encoding='utf-8') as data_file:
        true_data=json.load(data_file)

    pred_data=[]
    with open(pred_entity_path,'r',encoding='utf-8') as data_file:
        pred_data=json.load(data_file)

    all_right=0
    all_true=0
    all_pred=0
    all_line_right=0
    no_right_link=[]
    for i in range(len(true_data)):
        if true_data[i]['sentence']!=pred_data[i]['sentence'] and true_data[i]['sentence'].strip('?').strip('？').strip('?')!=pred_data[i]['sentence'].strip('?').strip('？').strip('?'):
            print(i,true_data[i]['sentence'],pred_data[i]['sentence'])
        
        # assert true_data[i]['sentence']==pred_data[i]['sentence']
        if 'entities' in true_data[i]:
            true_entity=true_data[i]['entities']
        else:
            true_entity=true_data[i]['true_entity_property']
        pred_entity=pred_data[i]['pred_entity']

        #转换为集合才可以计算交并补
        set_true_entity=set(true_entity)
        set_pred_entity=set(pred_entity)

        # print(true_data[i],'----',i)
        # print('ture_entity---',set_true_entity)
        # print('mention---',mention)
        # print('pred_entity---',set_pred_entity)
        right=len(set_true_entity&set_pred_entity)
        if (right)>0:
            all_line_right=all_line_right+1
        else:
            pred_data[i]['true_entity']=true_entity
            pred_data[i]['id']=i
            no_right_link.append(pred_data[i])
        # print(i,'---true---',len(true_entity),'right---',right)
        all_right=all_right+right
        all_true=all_true+len(set_true_entity)
        all_pred=all_pred+len(set_pred_entity)

    print('no_right_number----',len(no_right_link))
    if no_right_linking_file==None:
        with open('./no_right_linking.json','w',encoding='utf-8') as writer:
            json.dump(no_right_link,writer,ensure_ascii=False)
    else:
        with open(no_right_linking_file,'w',encoding='utf-8') as writer:
            json.dump(no_right_link,writer,ensure_ascii=False)
    print('recall----',all_right/all_true)
    print('line_right_recall----',all_line_right/len(true_data))
    print('average pred entity---',all_pred/len(true_data))

    return len(no_right_link),all_right/all_true,all_line_right/len(true_data),all_pred/len(true_data)



#文件复制
#封装成函数
def copy_function(src,target):
    with open(src,'rb') as rstream:
        container=rstream.read()
        with open(target,'wb') as wstream:
            wstream.write(container)



men2ent_dict=get_mention2entity('../data/men2ent_dict.json')
men2pro_dict=get_mention2property('../data/men2pro_dict.json')

log_file='./valid.log'

model_outfile_Path='ckpt_result'
ner_dir='./ner'


pre_mention_right='./pre_mention_right'
sent2ment_right='./sent2ment_right'
test_linking_right='./test_linking_right'
no_right_linking='./no_right_linking'
test_linking_right_finall='./test_linking_right_finall'

for i in [ner_dir,pre_mention_right,sent2ment_right,test_linking_right,no_right_linking,test_linking_right_finall]:
    if not os.path.exists(i):
        os.mkdir(i)
model_path_set=os.listdir(model_outfile_Path)
for idx,model_path_i in enumerate(model_path_set):
    if 'valid' in model_path_i:
        true_ner_path='../data/bio_ner_valid.txt'
        clear_test='../data/clear_valid.json'
    elif 'train' in model_path_i:
        true_ner_path='../data/bio_ner_train.txt'
        clear_test='../data/clear_train.json'
    elif 'test' in model_path_i:
        true_ner_path='../data/bio_ner_test.txt'
        clear_test='../data/clear_test.json'
    log_wf=open(log_file,'a',encoding='utf-8')
    print('-'*30)
    print('%d / %d'%(idx,len(model_path_set)))
    print(model_path_i)
    # if ',0.95' not in model_path_i and ',0.949' not in model_path_i  :
    #     continue
    tag=model_path_i[:-4]
    a=true_ner_path
    b=os.path.join(model_outfile_Path,model_path_i)
    log_wf.write(model_path_i+'\n')
    c=os.path.join(ner_dir,'ner-%s.txt'%tag)
    if not os.path.exists(c):
        with open(a,'r',encoding='utf-8') as f:
            lines_a = f.readlines()
        with open(b,'r',encoding='utf-8') as f:
            lines_b = f.readlines()
        # print(len(lines_a))
        # print(len(lines_b))
        lines_c=[]
        for i in range(len(lines_a)):
            if i%2==0:
                lines_c.append(lines_b[i])
            else:
                lines_c.append(lines_a[i])
                lines_c.append(lines_b[i])
        
        with open(c,'w',encoding='utf-8') as f:
            f.writelines(lines_c)
    
    infile=c
    sentence,true_mention,pred_mention=label_to_word(infile)
    r,p,f1=macro_valuate_set(true_mention,pred_mention)
    log_wf.write('ner:\tR:%.4f\tP:%.4f\tF1:%.4f\n'%(r,p,f1))
    tag+='_ner(%.4f,%.4f,%.4f)'%(r,p,f1)

    outfile=os.path.join(pre_mention_right,'%s.json'%tag)
    write_entity(outfile,sentence,true_mention,pred_mention)
    # with open(outfile, 'r', encoding='utf-8') as file:
    #     data = json.load(file)
    #     # print(data['sentence'])
    #     macro_valuate_set(data['true_mention'],data['pred_mention'])


    infile=outfile
    outfile=os.path.join(sent2ment_right,'%s.json'%tag)
    transfer(infile,outfile)


    infile=outfile
    outfile=os.path.join(test_linking_right,'%s.json'%tag)
    if not os.path.exists(outfile):
        do_linking(men2ent_dict,men2pro_dict,infile,outfile)

    no_right_linking_file=os.path.join(no_right_linking,'%s.json'%tag)
    if  os.path.exists(clear_test):
        no_n,r,line_right_recall,avg_n=evaluate_entity_linking(clear_test,outfile,no_right_linking_file)
        log_wf.write('entity_linking:\tno_n:%d\tR:%.4f\line_right_r:%.4f\tavg_n:%.4f\n'%(no_n,r,line_right_recall,avg_n))
        tag+='entity_linking:\tno_n:%d\tR:%.4f\tline_right_r:%.4f\tavg_n:%.4f'%(no_n,r,line_right_recall,avg_n)
        # tag=model_path_i+'%d\t%.4f\t%.4f\t%.4f'%(no_n,r,line_right_recall,avg_n)
        log_wf.write(tag+'\n')
        # copy_function(outfile, os.path.join(test_linking_right_finall,'%s.json'%tag))
        # shutil.copyfile(outfile, os.path.join(test_linking_right_finall,'%s.json'%tag))

    log_wf.close()
    