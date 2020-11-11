"""
将所有预测的路径变为查询语句，获取预测的答案

得到的文件格式
q【问题id】：【问题】
【预测分数1】---【预测路径1】---【问题id】---【问题】---【答案1】
【预测分数2】---【预测路径2】---【问题id】---【问题】---【答案2】
例子
q2:"光武中兴"说的是哪位皇帝？
0.9999997---<建武中兴>|||<皇帝>|||?x---2---"光武中兴"说的是哪位皇帝？---'<后醍醐天皇>'

"""

from py2neo import Graph
import numpy as np




def get_answer(predict_result_path, ans_path, graph, use_f1=True, pre_sorces_k=None):
    with open(predict_result_path, 'r', encoding='utf-8') as predict_reader:
        lines = predict_reader.readlines()

        # result_path_id_2_question_id = [0] * len(lines)
        # q_idx = -1
        # for i in range(len(lines)):
        #     if lines[i][0] == 'q':
        #         q_idx = i
        #     result_path_id_2_question_id[i] = q_idx

        if pre_sorces_k is None:
            lines_topk = [1] * len(lines)
        else:
            # 选出得分高的前k个
            pre_idx = -1
            pre_sorces = []
            lines_topk = [0] * len(lines)
            for i in range(len(lines)):
                if lines[i][0] == 'q':
                    lines_topk[i] = 1
                    sorce_idx = np.argsort(pre_sorces)[-1 * pre_sorces_k:]
                    for j in sorce_idx:
                        lines_topk[pre_idx + j + 1] = 1
                    pre_sorces = []
                    pre_idx = i
                else:
                    pre_sorces.append(float(lines[i].split('---')[0]))

            sorce_idx = np.argsort(pre_sorces)[-1 * pre_sorces_k:]
            for j in sorce_idx:
                lines_topk[pre_idx + j + 1] = 1

        ans_writer = open(ans_path, 'w', encoding='utf-8')  # 中断继续？
        result = {}

        for i in range(len(lines)):
            if lines_topk[i] == 0:
                continue
            if lines[i][0] == 'q':  # result_path_id_2_question_id[i] == i:
                result['id'] = lines[i].split(':')[0].lstrip('q')
                result['question'] = lines[i].split(':')[1].strip('\n')
                ans_writer.write(lines[i])
                print(lines[i])
            else:
                result['pred_score'] = lines[i].split('---')[0]
                if use_f1:
                    triple_list = lines[i].split('---')[1].strip('\n').replace("'", "\\'") \
                        .split('\t')  # 替换'为\'，避免查询语句错误
                    try:
                        answers_i = _triple_list_to_ans(graph, triple_list)
                    except Exception as e:
                        print(e, lines[i])
                    answers_i.sort()
                    answers_i = list(map(lambda x: '\'' + x.replace('---', '') + '\'', answers_i))
                else:
                    answers_i = []

                ans_writer.write(lines[i][:-1] + '---' + result['id'] + '---' +
                                 result['question'] + '---' + '\t'.join(answers_i) + '\n')

        ans_writer.close()


def _triple_list_to_ans(graph, triple_list):
    # print('triple_list----', triple_list)

    triple_1 = triple_list[0].replace('?x', 'x').replace('?y', 'y')
    query_1 = triple_1.split('|||')
    cypher = "MATCH "

    if query_1[0] == 'x' or query_1[0] == 'y':

        cypher = cypher + "(" + query_1[0] + ")"
    else:
        cypher = cypher + "(:Entity{name:'" + query_1[0] + "'})"
    cypher = cypher + "-[:Relation{name:'" + query_1[1] + "'}]->"
    if query_1[2] == 'x' or query_1[2] == 'y':
        cypher = cypher + "(" + query_1[2] + ")"
    else:
        cypher = cypher + "(:Entity{name:'" + query_1[2] + "'})"

    if len(triple_list) == 1:
        cypher = cypher + " return x.name"
        result_1 = []
        try:
            # print('cypher---', cypher)
            answers = graph.run(cypher).data()
            for ans in answers:
                if ans['x.name'] != None:
                    result_1.append(ans['x.name'])
            result_1 = list(set(result_1))
            # print('answers----', answers)
        except:
            print('one_ent_one_hop cypher wrong')
        return result_1

    if len(triple_list) == 2:
        # cypher2="MATCH (ent1:Entity{name:'"+ent+"'})-[rel1]->(y)<-[rel2]-(x) RETURN DISTINCT rel1.name,rel2.name,x.name"
        cypher = cypher + " MATCH "
        triple_2 = triple_list[1].replace('?y', 'y').replace('?x', 'x')
        query_2 = triple_2.split('|||')

        if query_2[0] == 'x' or query_2[0] == 'y':
            cypher = cypher + "(" + query_2[0] + ")"
        else:
            cypher = cypher + "(:Entity{name:'" + query_2[0] + "'})"
        cypher = cypher + "-[:Relation{name:'" + query_2[1] + "'}]->"
        if query_2[2] == 'x' or query_2[2] == 'y':
            cypher = cypher + "(" + query_2[2] + ")"
        else:
            cypher = cypher + "(:Entity{name:'" + query_2[2] + "'})"

        cypher = cypher + " return x.name"
        result_2 = []
        try:
            # print('cypher---', cypher)
            answers = graph.run(cypher).data()
            for ans in answers:
                if ans['x.name'] != None:
                    result_2.append(ans['x.name'])
            result_2 = list(set(result_2))
            # print('answers----', answers)
        except:
            print('one_ent_two_hop cypher wrong')
        return result_2
    return []


if __name__ == '__main__':
    # txt = 'pred_result_neg5_dictABD3_top5_x.txt'
    # txt = 'pred_result_neg5_dictABD3_class_x.txt'
    # txt = 'ok_result_neg5_dictABD3_class_x.txt'
    # txt = 'ok_result_neg5_dictABD3_top5_x.txt'
    # txt = 'q78.txt'

    # txt='pred_result_new_pathlink_class.txt'
    # txt='pred_result_new_entlink_class.txt'
    # txt = 'ok_result_new_entlink_class.txt'
    # txt = 'pred_com_ent_path_class.txt'
    # txt = 'valid_ok_class.txt' # todo
    # txt = 'valid_pre_class.txt'
    # txt = 'ok_result_new_pathlink_class.txt'

    # for txt in ['pred_result_new_pathlink_class.txt','pred_result_new_entlink_class.txt',
    #             'ok_result_new_entlink_class.txt','pred_com_ent_path_class.txt',
    #             'valid_ok_class.txt','valid_pre_class.txt']:

    # txt='ok_result_new_entlink_sim.txt'
    # txt='pred_com_ent_path_sim.txt'
    # txt='pred_result_new_entlink_sim.txt'
    # txt='pred_result_new_pathlink_sim.txt'
    # txt='valid_ok_sim.txt'
    # txt='valid_pred_sim.txt'
    txt = 'ok_result_new_pathlink_sim.txt'
    pass
    # print(txt)
    # graph = Graph("http://59.78.194.63:57474", username='neo4j', password='pass')
    # get_answer(config.in_dir + '/' + txt, config.out_dir + '/answer_' + txt, graph, pre_sorces_k=10)
