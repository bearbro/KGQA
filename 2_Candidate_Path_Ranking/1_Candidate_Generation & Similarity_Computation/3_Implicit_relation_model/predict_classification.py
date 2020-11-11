import sys
'''
先对一跳的句子进行搜索，选取最大的n个，然后再找n个实体相连的候选路径进行计算，选择得分最高的
'''

# sys.path.insert(0,'/home/aistudio/work/MyExperiment/classification')
sys.path.insert(0,'/home/hbxiong/QA2/classification')
sys.path.insert(0,r'C:\Users\bear\Desktop\QA\classification')
from keras_bert import load_trained_model_from_checkpoint
import keras
import json
from py2neo import Graph
from some_function_maxbert import transfer_data_test 
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"#设置使用0号GPU
import numpy as np
import re
import keras.backend as k
import tensorflow as tf
from get_query_result_new import get_all_F1
from pprint import pprint
import os
import time
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



tag=time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
print(tag)

model_tag='ckpt_similarity_bert_wwm_ext_p7.hdf5'
print(model_tag)

class Config:
    # data_path
    data_dir = r'./data/old_data'
    train_data_path = os.path.join(data_dir, 'siamese_train_data_sample.json')
    valid_data_path = os.path.join(data_dir, 'siamese_valid_data_sample.json')

    linking_data_path = '../result_new_linking-no_n_59,r_0.8321,line_right_recall_0.9230,avg_n_2.6919.json'

    # bert_path
    # bert_path = '../../bert/bert_wwm_ext'  # 百度
    bert_path = r'C:\Users\bear\OneDrive\ccks2020-onedrive\ccks2020\bert\tf-bert_wwm_ext'  # room
    # bert_path = r'../../../ccks/bert/tf-bert_wwm_ext'  # colab
    # bert_path = '/home/hbxiong/ccks/bert/tf-bert_wwm_ext'  # lab
    bert_config_path = os.path.join(bert_path, 'bert_config.json')
    bert_ckpt_path = os.path.join(bert_path, 'bert_model.ckpt')
    bert_vocab_path = os.path.join(bert_path, 'vocab.txt')

    parameter = 0.7  # 用于loss的计算

    # result_path
    result_path = './data/%s-%s' % (model_tag, tag)
    similarity_ckpt_path = './ckpt/%s' % model_tag  # 模型训练后，模型参数存储路径
    true_answer_path = os.path.join(result_path, 'true_path_score.json')  # 模型在测试集正确路径上的预测得分
    ok_result_path = os.path.join(result_path, 'ok_result.txt')  # 保存为txt，使得发生错误时可以在当前问题继续训练
    pred_result_path = os.path.join(result_path, 'pred_result.txt')




    batch_size = 64
    epoches = 1000
    learning_rate = 1e-5
    neg_sample_number = 10
    max_length = 100  # neg3:64;100


config = Config()

for i in ['./ckpt',config.result_path]:
    if not os.path.exists(i):
        os.mkdir(i)



pprint(vars(Config))


def basic_network():
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path,
                                                    config.bert_ckpt_path,
                                                    seq_len=config.max_length,
                                                    training=False,
                                                    trainable=True)

    x_1= keras.layers.Input(shape=(config.max_length,),name='input_x1')
    x_2= keras.layers.Input(shape=(config.max_length,),name='input_x2')

    bert_out=bert_model([x_1, x_2]) # 输出维度为(batch_size,max_length,768)

    # dense=bert_model.get_layer('NSP-Dense')
    bert_out = keras.layers.Lambda(lambda bert_out: bert_out[:, 0],name='bert_1')(bert_out)

    # bert_out=keras.layers.Dropout(0.5)(bert_out)

    outputs=keras.layers.Dense(1, activation='sigmoid',name='dense')(bert_out)

    model = keras.models.Model([x_1,x_2],outputs,name='basic_model')
    model.summary()
    return model


def triplet_model():
    basic_model=basic_network()
    pos_x_1= keras.layers.Input(shape=(config.max_length,))
    pos_x_2= keras.layers.Input(shape=(config.max_length,))
    neg_x_1= keras.layers.Input(shape=(config.max_length,))
    neg_x_2= keras.layers.Input(shape=(config.max_length,))
    pos_model=basic_model([pos_x_1,pos_x_2])
    neg_model=basic_model([neg_x_1,neg_x_2])
    out=keras.layers.concatenate([pos_model,neg_model],-1)
    model=keras.models.Model([pos_x_1,pos_x_2,neg_x_1,neg_x_2],out)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(config.learning_rate),loss=hinge_loss,metrics=['accuracy'])
    return model

def hinge_loss(y_true,y_pred):
    para=config.parameter
    kg=para-y_pred[:,0]+y_pred[:,1]
    kg=kg*np.array([[1],[0]])
    print('iii---',k.int_shape(kg))
    loss=k.max(kg,0)
    return k.mean(loss)

def get_one_ent_one_hop(graph, ent):
    '''
    输入graph实例和实体，返回实体周围一跳的路径
    :param graph:
    :param ent:
    :return:
    '''
    # ent->rel->?x
    cypher1 = "MATCH (ent1:Entity{name:'" + ent + "'})-[rel]->(x) RETURN DISTINCT rel.name"
    try:
        relations = graph.run(cypher1).data()
    except:
        relations=[]
        print('one_ent_one_hop cypher1 wrong')
    sample1 = []
    for rel in relations:
        sam = ent + '|||' + rel['rel.name'] + '|||?x'
        sample1.append(sam)
        # print(sam)

    # ?x->rel->ent
    cypher2 = "MATCH (ans)-[rel]->(ent1:Entity{name:'" + ent + "'}) RETURN DISTINCT rel.name"
    try:
        relations = graph.run(cypher2).data()
    except:
        relations=[]
        print('one_ent_one_hop cypher2 wrong')
    sample2 = []
    for rel in relations:
        sam = '?x|||' + rel['rel.name'] + '|||' + ent
        sample2.append(sam)
        # print(sam)
    return sample1 + sample2

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
        
        answers = graph.run(cypher4).data()
        if len(answers)<1000:
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
            cypher7=cypher+"(ent1:Entity{name:'" +path_list[0] + "'})-[rel1:Relation{name:'" + path_list[1] + "'}]->(x) match (y)-[rel2]->(x) where ent1.name<>y.name return distinct rel2.name,y.name"
            # print(cypher7)
            answers = graph.run(cypher7).data()
            if len(answers) <= 1000:
                for ans in answers:
                    two_ent1=path+'\t'+ans['y.name']+'|||'+ans['rel2.name']+'|||?x'
                    # print(two_ent)
                    sample.append(two_ent1)

            cypher8 = cypher + "(:Entity{name:'" + path_list[0] + "'})-[rel1:Relation{name:'" + path_list[1] + "'}]->(x) match (x)-[rel2]->(y) return distinct rel2.name,y.name"
            # print(cypher8)
            answers = graph.run(cypher8).data()
            if len(answers) <= 1000:
                for ans in answers:
                    two_ent = path + '\t' + '?x|||' + ans['rel2.name'] + '|||'+ans['y.name']
                    # print(two_ent)
                    sample.append(two_ent)
            print('two hop path len',len(sample),'---',sample[:5])

    return sample

def _get_next_hop_path_three(graph,path):
    '''
    输入任意跳路径，返回构建的多一跳路径（只包含y-rel->x）
    :param graph:
    :param path:
    :return:
    '''
    path = path.replace('\\', '\\\\').replace("\'", "\\\'")
    triple_list=path.split('\t')
    x_triple=triple_list[-1]
    x_list=x_triple.split('|||')
    if x_list[1]=='<类型>' and x_list[2]=='?x':
        return []
    cypher=''
    rel=[]
    sample=[]
    # print('path2---',path)
    for triple in triple_list:
        triple_cypher='match '
        item_list=triple.split('|||')
        if item_list[0].startswith('?'):
            triple_cypher=triple_cypher+"("+item_list[0].strip('?')+")"
        else:

            triple_cypher = triple_cypher +"(:Entity{name:'"+item_list[0]+"'})"
        triple_cypher = triple_cypher + "-[:Relation{name:'" + item_list[1] + "'}]"
        rel.append(item_list[1])
        if item_list[2].startswith('?'):
            triple_cypher = triple_cypher + "->(" + item_list[2].strip('?') + ") "
        else:

            triple_cypher = triple_cypher + "->(:Entity{name:'" + item_list[2] + "'}) "
        cypher=cypher+triple_cypher

    # #质包括y-rel->x
    # if len(re.findall(cypher,'(y)')) > 0:
    #     cypher1=cypher+'match (x)-[rel]->(a) where y.name<>a.name return distinct rel.name'
    # else:
    #     cypher1 = cypher + 'match (x)-[rel]->(a) where '
    #     for ent_item in ent:
    #         cypher1+="a.name<>'"+ent_item+"' and "
    #     cypher1=cypher1[:-4]+'return distinct rel.name'

    cypher1 = cypher + "match (x)-[rel]->(a) where rel.name<>'<类型>' and "
    for rel_item in rel:
        cypher1+="rel.name<>'"+rel_item+"' and "
    cypher1=cypher1[:-4]+'return distinct rel.name'

    answers = graph.run(cypher1).data()
    print('three hop cypher---',cypher1)
    for ans in answers:
        one_ent1_1=path.replace('?x','?y1')+'\t'+'?y1|||'+ans['rel.name']+'|||?x'
        sample.append(one_ent1_1)
    print('three hop path---',sample)
    return sample

def _get_next_hop_path_four(graph,path):
    '''
    输入任意跳路径，返回构建的多一跳路径（只包含y-rel->x）
    :param graph:
    :param path:
    :return:
    '''
    path = path.replace('\\', '\\\\').replace("\'", "\\\'")
    triple_list=path.split('\t')
    x_triple=triple_list[-1]
    x_list=x_triple.split('|||')
    if x_list[1]=='<类型>' and x_list[2]=='?x':
        return []
    cypher=''
    rel=[]
    sample=[]
    # print('path2---',path)
    for triple in triple_list:
        triple_cypher='match '
        item_list=triple.split('|||')
        if item_list[0].startswith('?'):
            triple_cypher=triple_cypher+"("+item_list[0].strip('?')+")"
        else:
            triple_cypher = triple_cypher +"(:Entity{name:'"+item_list[0]+"'})"
        triple_cypher = triple_cypher + "-[:Relation{name:'" + item_list[1] + "'}]"
        rel.append(item_list[1])
        if item_list[2].startswith('?'):
            triple_cypher = triple_cypher + "->(" + item_list[2].strip('?') + ") "
        else:
            triple_cypher = triple_cypher + "->(:Entity{name:'" + item_list[2] + "'}) "
        cypher=cypher+triple_cypher

    # #质包括y-rel->x
    # if len(re.findall(cypher,'(y1)')) > 0:
    #     cypher1=cypher+'match (x)-[rel]->(a) where y1.name<>a.name return distinct rel.name'
    # else:
    #     cypher1 = cypher + 'match (x)-[rel]->(a) where '
    #     for ent_item in ent:
    #         cypher1 += "a.name<>'" + ent_item + "' and "
    #     cypher1 = cypher1[:-4] + 'return distinct rel.name'

    cypher1 = cypher + "match (x)-[rel]->(a) where rel.name<>'<类型>' and "
    for rel_item in rel:
        cypher1+="rel.name<>'"+rel_item+"' and "
    cypher1 = cypher1[:-4] + 'return distinct rel.name'

    answers = graph.run(cypher1).data()
    print('four hop cypher---',cypher1)
    for ans in answers:
        one_ent1_1=path.replace('?x','?y2')+'\t'+'?y2|||'+ans['rel.name']+'|||?x'
        sample.append(one_ent1_1)
    print('four hop path---',sample)
    return sample

def _get_next_hop_path_(graph,path):
    '''
    输入任意跳路径，返回构建的多一跳路径（包含各种情况）
    :param graph:
    :param path:
    :return:
    '''
    path = path.replace('\\', '\\\\').replace("\'", "\\\'")
    triple_list=path.split('\t')
    x_triple=triple_list[-1]
    x_list=x_triple.split('|||')
    if x_list[1]=='<类型>' and x_list[2]=='?x':
        return []
    cypher=''
    ent=[]
    sample=[]
    # print('path2---',path)
    for triple in triple_list:
        triple_cypher='match '
        item_list=triple.split('|||')
        if item_list[0].startswith('?'):
            triple_cypher=triple_cypher+"("+item_list[0].strip('?')+")"
        else:
            ent.append(item_list[0])
            triple_cypher = triple_cypher +"(:Entity{name:'"+item_list[0]+"'})"
        triple_cypher = triple_cypher + "-[:Relation{name:'" + item_list[1] + "'}]"
        if item_list[2].startswith('?'):
            triple_cypher = triple_cypher + "->(" + item_list[2].strip('?') + ") "
        else:
            ent.append(item_list[2])
            triple_cypher = triple_cypher + "->(:Entity{name:'" + item_list[2] + "'}) "
        cypher=cypher+triple_cypher

    #第一种格式
    if len(re.findall(cypher,'(y)')) > 0:
        cypher1=cypher+'match (x)-[rel]->(a) where y.name<>a.name return distinct rel.name'
    else:
        cypher1 = cypher + 'match (x)-[rel]->(a) return distinct rel.name'
    answers = graph.run(cypher1).data()
    for ans in answers:
        one_ent1_1=path.replace('?x','?z')+'\t'+'?z|||'+ans['rel.name']+'|||?x'
        # print(one_ent)
        sample.append(one_ent1_1)
        # print(sample[-1])
    if len(re.findall(cypher,'(y)'))>0:
        cypher1=cypher+'match (x)-[rel]->(a) where y.name<>a.name return distinct rel.name,a.name'
    else:
        cypher1 = cypher + 'match (x)-[rel]->(a) return distinct rel.name,a.name'
    answers = graph.run(cypher1).data()
    for ans in answers:
        two_ent1_2=path+'\t?x|||'+ans['rel.name']+'|||'+ans['a.name']
        # print(two_ent)
        sample.append(two_ent1_2)
        # print(sample[-1])

    #第二种格式
    if len(re.findall(cypher,'(y)'))>0:
        cypher2=cypher+'match (a)-[rel]->(x) where y.name<>a.name return distinct rel.name'
    else:
        cypher2 = cypher + 'match (a)-[rel]->(x) return distinct rel.name'
    answers = graph.run(cypher2).data()
    for ans in answers:
        one_ent2_1=path.replace('?x','?z')+'\t'+'?x|||'+ans['rel.name']+'|||?z'
        # print(one_ent)
        sample.append(one_ent2_1)
        # print(sample[-1])
    if len(re.findall(cypher,'(y)'))>0:
        cypher2=cypher+'match (a)-[rel]->(x) where y.name<>a.name return distinct rel.name,a.name'
    else:
        cypher2 = cypher + 'match (a)-[rel]->(x) return distinct rel.name,a.name'
    answers = graph.run(cypher2).data()
    for ans in answers:
        two_ent2_2=path+'\t'+ans['a.name']+'|||'+ans['rel.name']+'|||?x'
        # print(two_ent)
        sample.append(two_ent2_2)
        # print(sample[-1])
    return sample


if __name__=='__main__':


    threshold=1#认为当得分大于等于此值时停止往下找

    model_triple = triplet_model()
    model_triple.load_weights(config.similarity_ckpt_path)
    model_triple.summary()
    model = model_triple.get_layer(name='basic_model')
    model.summary()

    graph = Graph("http://47.114.86.211:57474", username='neo4j', password='pass')
    # graph = Graph("http://59.78.194.63:37474", username='neo4j', password='pass')#lab

    reader = open(config.linking_data_path, 'r', encoding='utf-8')

    data = json.load(reader)
    pre_writer=open(config.pred_result_path,'w',encoding='utf-8')
    ok_writer=open(config.ok_result_path,'w',encoding='utf-8')

    assert len(data)==766

    beamsearch=[10,10,3,2]#top k
    all_sample=0
    all_number=0
    for k in range(len(data)):#k控制对第k个句子进行predict
        a_sent_data=data[k]
        print('问题',k,':',a_sent_data['sentence'])
        if len(a_sent_data['pred_entity'])==0:#若句子中没有链接的实体，则不进行处理
            continue
        else:
            x_sample=[]
            for candidate in  a_sent_data['pred_entity']:
                candidate = candidate.replace("'", "\\'")  # 数据库中某些实体存在'
                x_sample.extend(get_one_ent_one_hop(graph,candidate))#不包括关系为类型，所以可能存在实体有而无路径的情况
            if len(x_sample)==0:
                continue
            x_sent=[a_sent_data['sentence']]*len(x_sample)
            x_indices,x_segments = transfer_data_test(x_sent, x_sample, config.max_length,config.bert_vocab_path)

            sample_number=len(x_indices)
            all_sample=all_sample+sample_number

            result = model.predict([x_indices, x_segments],batch_size=config.batch_size)
            result=result.ravel()#将result展平变为一维数组
            assert len(x_sample)==len(result)

            top_beamsearch_one_hop=[]#记录前k个候选的一跳路径
            result_sorted = np.argsort(-np.array(result))
            if len(result) > beamsearch[0]:
                result_sorted = result_sorted[0:beamsearch[0]]
            all_max_score = result[result.argmax(-1)]  # 记录总体的最大得分
            all_max_path = x_sample[result.argmax(-1)]
            # print(all_max_score,result_sorted[0],result[result_sorted[0]])
            assert result[result_sorted[0]] == all_max_score
            for i in result_sorted:
                now = {}
                now['level'] = 1  # level的值表示path涉及的跳数
                now['score'] = result[i]
                now['path'] = x_sample[i]
                top_beamsearch_one_hop.append(now)
                print('one hop top ', beamsearch[0], ': ', now)


            top_beamsearch_two_hop = []  # 记录包含前k个一跳路径，且得分大于其包含的一跳路径的两跳候选路径,最多k
            if all_max_score < threshold:
                two_score = []
                two_sample = []
                for top in top_beamsearch_one_hop:
                    max = top['score']
                    try:
                        x_sample_next = _get_next_hop_path_two(graph, top['path'])
                    except:
                        x_sample_next=[]
                    if len(x_sample_next) == 0:  # 若没有第二跳路径，则看下一个
                        continue

                    x_sent_next = [a_sent_data['sentence']] * len(x_sample_next)
                    x_indices, x_segments = transfer_data_test(x_sent_next, x_sample_next, config.max_length,config.bert_vocab_path)

                    sample_number = len(x_indices)
                    all_sample = all_sample + sample_number
                    result = model.predict([x_indices, x_segments], batch_size=config.batch_size)
                    result = result.ravel()  # 将result展平变为一维数组
                    # print('two_hop---',result)
                    for i in range(len(result)):
                        if result[i] > max:
                            two_score.append(result[i])
                            two_sample.append(x_sample_next[i])
                            if result[i] > all_max_score:
                                all_max_score = result[i]
                                all_max_path = x_sample_next[i]

                next_sorted = np.argsort(-np.array(two_score))
                if len(next_sorted) > beamsearch[1]:
                    next_sorted = next_sorted[0:beamsearch[1]]
                for i in next_sorted:
                    now = {}
                    now['level'] = 2
                    now['score'] = two_score[i]
                    now['path'] = two_sample[i]
                    top_beamsearch_two_hop.append(now)
                    print('two hop top ',str(beamsearch[1]),' :', now)  # 有符合条件加入twohop的path
                # 在two_hop的基础上向下找，直到没有可以加入的更大的得分


            pre_writer.write('q'+str(k)+':' + a_sent_data['sentence'] + '\n')
            pre_writer.write(str(all_max_score)+'---'+all_max_path + '\n')
            pre_writer.flush()

            #将符合条件的路径写入ok文件
            ok_writer.write('q'+str(k)+':' + a_sent_data['sentence'] + '\n')
            for item in top_beamsearch_one_hop+top_beamsearch_two_hop:
                ok_writer.write(str(item['score'])+'---'+item['path']+'\n')
            print('问题', k,',得分',all_max_score,':',a_sent_data['sentence'],'predict over: ',all_max_path)
            all_number+=1

    pre_writer.close()
    ok_writer.close()

    print('平均的候选答案数量---',all_sample/all_number)
    get_all_F1(config.pred_result_path)    