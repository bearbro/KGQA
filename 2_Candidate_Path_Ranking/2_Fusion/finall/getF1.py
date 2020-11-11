'''
重写查询函数，对某些查询结果进行限制，如
    答案为对应的实体，则将两跳的候选路径改为一跳的候选路径
对答案进行某些限制
将预测的路径变为查询语句，获取预测的答案，存入query_result.txt
'''

from py2neo import Graph
import re

def get_Graph():
    return Graph("http://59.78.194.63:37474", username='neo4j', password='pass')
    # return Graph("http://localhost:7474", username='neo4j', password='pass')

def _get_true_result(true_path='./test.txt'):
    '''
    从原始文件test.txt中，返回正确的题目id和答案的字典
    :param true_path: id2ans
    :return:
    '''
    with open(true_path, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        true_answer = {}
        sentence = {}
        for i in range(766):
            true_answer[str(i)] = lines[i * 4 + 2].strip(' ').strip('\n').split('\t')
            sentence[str(i)] = lines[i * 4].strip(' ').strip('\n').split(':')[1]
            # print(true_answer[str(i)])
    return true_answer, sentence


def _get_pred_result(query_result_path):
    with open(query_result_path, 'r', encoding='utf-8') as reader:
        id2predanswers = {}
        id2predscore = {}
        id2path = {}
        for line in reader.readlines():
            # print(line.strip('\n'))
            line_json = eval(line.strip('\n'))
            id = line_json['id']
            pred_answer = line_json['answers']
            path = line_json['path']
            pred_score = line_json['pred_score']

            # if len(pred_answer)==0:#预测没有给出答案
            #     continue
            id2predanswers[str(id)] = pred_answer
            id2predscore[str(id)] = float(pred_score.split('|')[0])
            id2path[str(id)] = path
    return id2predanswers, id2predscore, id2path


def evaluate(query_result_path, wrong_path):
    '''
    每句的评估值然后算平均
    :param query_result_path:
    :param wrong_path:
    :param right_path:
    :return:
    '''
    id2answers, id2sentence = _get_true_result()
    assert len(id2answers.keys()) == 766
    assert len(id2sentence.keys()) == 766

    all_number = 0
    all_F1 = 0
    all_precision = 0
    all_recall = 0

    wrong_writer = open(wrong_path, 'w', encoding='utf-8')

    id2predanswers, id2predscore, id2path = _get_pred_result(query_result_path)

    print('统计的句子--', len(id2predanswers.keys()))

    for id in range(766):
        try:

            true_answer = set(id2answers[str(id)])
            pred_answer = set(id2predanswers[str(id)])
            right = len(pred_answer & true_answer)
            recall = right / len(true_answer)
            precison = right / len(pred_answer)
            if precison == 0 or recall == 0:
                F1 = 0
            else:
                F1 = (2 * recall * precison) / (recall + precison)

            if F1 == 0:
                # print(id,'---预测全错')
                wrong_writer.write(str(id) + '---' + id2sentence[str(id)] + '\n')
                wrong_writer.write(str(id2predscore[str(id)]) + '---' + id2path[str(id)] + '\n')
                # wrong_writer.write('pred_answers---'+str(pred_answer)+'\n')
                # wrong_writer.write('true_answers---'+str(true_answer)+'\n')

            all_number = all_number + 1
            all_F1 += F1
            all_precision += precison
            all_recall += recall

        except:
            print('此ID句子不存在，未统计', id)
    print('precision---', all_precision / 766, '\n')
    print('recall---', all_recall / 766, '\n')
    print('F1---', all_F1 / 766, '\n')
    print('number---', all_number)
    # return all_number, all_precision / all_number, all_recall / all_number, all_F1 / all_number
    return all_number, all_precision / 766, all_recall / 766, all_F1 / 766


def evaluate_threshold(query_result_path, threshold, wrong_path):
    '''
    句子得分最大时，且大于某个阈值时才作为正确答案
    :param query_result_path:
    :param threshold:
    :return:
    '''
    id2answers, id2sentence = _get_true_result()
    all_number = 0
    all_F1 = 0
    all_precision = 0
    all_recall = 0
    threshold_wrong = open(wrong_path, 'w', encoding='utf-8')
    id2predanswers, id2predscore, id2path = _get_pred_result(query_result_path)

    for id in range(766):
        try:
            if id2predscore[str(id)] < threshold:  # 低于某阈值则不进行统计
                continue

            true_answer = set(id2answers[str(id)])
            pred_answer = set(id2predanswers[str(id)])
            right = len(pred_answer & true_answer)
            recall = right / len(true_answer)
            precison = right / len(pred_answer)
            if precison == 0 or recall == 0:
                F1 = 0
            else:
                F1 = (2 * recall * precison) / (recall + precison)
            if F1 == 0:
                # print(id, '---预测全错')
                threshold_wrong.write(
                    str(id) + '---' + id2sentence[str(id)] + '---' + str(id2predscore[str(id)]) + '\n')
                # wrong_writer.write('pred_answers---'+str(pred_answer)+'\n')
                # wrong_writer.write('true_answers---'+str(true_answer)+'\n')
            all_number = all_number + 1
            all_F1 += F1
            all_precision += precison
            all_recall += recall

        except:
            print('threshold,此ID句子不存在，未统计')
    print('precision---', all_precision / all_number, '\n')
    print('recall---', all_recall / all_number, '\n')
    print('F1---', all_F1 / all_number, '\n')
    print('number---', all_number)
    return all_number, all_precision / all_number, all_recall / all_number, all_F1 / all_number


def get_answer(predict_result_path, ans_path, graph):
    ans_writer = open(ans_path, 'w', encoding='utf-8')
    with open(predict_result_path, 'r', encoding='utf-8') as predict_reader:
        lines = predict_reader.readlines()
        for i in range(int(len(lines) / 2)):
            result = {}  # 记录一个句子的path查询结果
            # q0:"成败一知己，生死两妇人"所说的人物有什么重大成就？
            result['id'] = lines[i * 2].split(':')[0].lstrip('q')
            result['question'] = lines[i * 2].split(':')[1].strip('\n')
            print('question---', result['question'])
            result['pred_score'] = lines[i * 2 + 1].split('---')[0]
            # 0.9438863---?y <制片人> <黄蓉_（金庸武侠小说《射雕英雄传》女主角）>	?y <制片地区> ?x
            path = lines[i * 2 + 1].split('---')[1].strip('\n')
            p, a = _path_to_ans(graph, result['question'], path)
            result['path'] = p
            result['answers'] = a
            # print('query result---',result)
            ans_writer.write(str(result) + '\n')


def _path_to_ans(graph, sentence, path):
    if len(path) == 0:
        return path, []
    triple_list = path.strip('\n').replace("'", "\\'").replace('\\', '\\\\').split('\t')  # 替换'为\'，避免查询语句错误
    result = []
    ent = []
    cypher = ''
    rel = []
    sample = []
    # print('path2---',path)
    for triple in triple_list:
        triple_cypher = 'match '
        item_list = triple.split('|||')
        if item_list[0].startswith('?'):
            triple_cypher = triple_cypher + "(" + item_list[0].strip('?') + ")"
        else:
            ent.append(item_list[0])
            triple_cypher = triple_cypher + "(:Entity{name:'" + item_list[0] + "'})"
        triple_cypher = triple_cypher + "-[:Relation{name:'" + item_list[1] + "'}]"
        rel.append(item_list[1])
        if item_list[2].startswith('?'):
            triple_cypher = triple_cypher + "->(" + item_list[2].strip('?') + ") "
        else:
            ent.append(item_list[2])
            triple_cypher = triple_cypher + "->(:Entity{name:'" + item_list[2] + "'}) "
        cypher = cypher + triple_cypher

    cypher += 'return x.name'

    print('query cypher---', cypher)
    answers = graph.run(cypher).data()

    for ans in answers:
        result.append(ans['x.name'])
    return path, list(set(result))


import os

graph =get_Graph()




def get_a_path_answer(sentence, path, graph):
    print(path)
    print(_path_to_ans(graph, sentence, path))


# get_a_path_answer('最大','?y|||<陵墓>|||<茂陵_（汉武帝陵寝）>\t?y1|||<在位皇帝>|||?y	?y1|||<朝代>|||?y2\t?y2|||<在位时间>|||?x')
def get_all_F1(file):
    print(file)
    filepath, fullflname = os.path.split(file)
    file_name = fullflname
    dir = os.path.join(filepath, '%s_score' % file_name[:-4])
    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(os.path.join(dir, 'query_' + file_name)):
        get_answer(file, os.path.join(dir, 'query_' + file_name), graph)
    all_number, average_precision, average_recall, average_F1 = evaluate(os.path.join(dir, 'query_' + file_name),
                                                                         os.path.join(dir, 'wrong_' + file_name))
    all_number_th, average_precision_th, average_recall_th, average_F1_th = evaluate_threshold(
        os.path.join(dir, 'query_' + file_name), 0.97, os.path.join(dir, 'z.txt'))
    print('{}---{}---{}---{}'.format(all_number, average_precision, average_recall, average_F1))
    print('{}---{}---{}---{}'.format(all_number_th, average_precision_th, average_recall_th, average_F1_th))
    with open(os.path.join(dir, 'f1score_' + file_name), 'w', encoding='utf-8') as wout:
        wout.write('{}---{}---{}---{}'.format(all_number, average_precision, average_recall, average_F1))


if __name__ == '__main__':
    # get_all_F1(r'C:\Users\77385\Desktop\QA\data\bert_wwm_ext_2hop_p72020-10-26-02-35-28\pred_result.txt')
    get_all_F1(r'C:\Users\77385\Desktop\QA\stacking\finall_top3\finallanswer.txt')
