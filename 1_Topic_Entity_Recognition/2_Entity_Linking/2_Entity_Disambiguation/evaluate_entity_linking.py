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

    # url = {"host": '59.78.194.63', "port": 59200, "timeout": 1500}#lab
    url = {"host": '8.210.254.190', "port": 6692, "timeout": 1500}#lab
    es = Elasticsearch([url])
    with open(mention_path,'r',encoding='utf-8') as file:
        data=json.load(file)
        result = []
        for item in data:
            a_sentence={}
            a_sentence['id']=item['id']
            a_sentence['sentence']=item['sentence']
            a_sentence['pred_entity']=[]
            a_sentence['mention']=item['mention']

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

            not_entity=0
            if len(a_sentence['pred_entity'])==0:#统计链接实体为0的情况
                not_entity+=1

            print('q',str(a_sentence['id']),':',a_sentence['sentence'],'---',len(a_sentence['pred_entity']),'---',a_sentence['pred_entity'])
            result.append(a_sentence)
    with open(write_path,'w',encoding='utf-8') as write_file:
        json.dump(result,write_file,ensure_ascii=False)
    print('over entity linking,not_entity number---',not_entity)

def evaluate_entity_linking(true_entity_path,pred_entity_path):
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
    for i in range(766):
        assert true_data[i]['sentence']==pred_data[i]['sentence']
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
        print(i,'---true---',len(true_entity),'right---',right)
        all_right=all_right+right
        all_true=all_true+len(set_true_entity)
        all_pred=all_pred+len(set_pred_entity)

    print('no_right_number----',len(no_right_link))
    with open('./no_right_linking.json','w',encoding='utf-8') as writer:
        json.dump(no_right_link,writer,ensure_ascii=False)
    print('recall----',all_right/all_true)
    print('line_right_recall----',all_line_right/766)
    print('average pred entity---',all_pred/766)


# men2ent_dict=read_mention2entity('../data/ccks2019/PKUBASE/pkubase-mention2ent.txt')

men2ent_dict=get_mention2entity('../data/men2ent_dict.json')
men2pro_dict=get_mention2property('../data/men2pro_dict.json')

outfile='./result_new_linking-no_n_59,r_0.8321,line_right_recall_0.9230,avg_n_2.6919.json'
# no_right_number---- 59
# recall---- 0.8321013727560718
# line_right_recall---- 0.922976501305483
# average pred entity--- 2.691906005221932

# do_linking(men2ent_dict,men2pro_dict,infile,outfile)
evaluate_entity_linking('../data/clear_test.json',outfile)



