"""

得到的文件格式
q【问题id】：【问题】
【预测分数1】---【预测路径1】---【问题id】---【问题】---【答案1】---【分数1】
【预测分数2】---【预测路径2】---【问题id】---【问题】---【答案2】---【分数2】
例子
q2:"光武中兴"说的是哪位皇帝？
0.9999997---<建武中兴>|||<皇帝>|||?x---2---"光武中兴"说的是哪位皇帝？---'<后醍醐天皇>'---0.0

"""
import numpy as np




def _get_true_result(true_path):
    '''
    返回正确的题目id和答案的字典
    :param true_path: id2ans
    :return:
    '''
    with open(true_path, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        true_answer = {}
        sentence = {}
        for i in range(766):
            true_answer[str(i)] = lines[i * 4 + 2].strip('\n').split('\t')
            sentence[str(i)] = lines[i * 4].strip('\n').split(':')[1]
            # print(true_answer[str(i)])
    return true_answer, sentence


def evaluate_threshold(query_result_path, sorce_path, true_path, use_f1=True, threshold=None):
    '''
    给各行答案打分
    '''
    id2answers, id2sentence = _get_true_result(true_path)
    F1_list = []
    id = -1
    with open(query_result_path, 'r', encoding='utf-8') as reader:
        sorce_writer = open(sorce_path, 'w', encoding='utf-8')
        for line in reader.readlines():
            if line[0] == 'q':
                id = int(line.split(':')[0][1:])
                sorce_writer.write(line)
            else:
                if use_f1:
                    pred_answer = line.split('---')[-1]

                    pred_answer = set(map(lambda x: x[1:-1], pred_answer[:-1].split('\t')))
                    # pred_answer = set(pred_answer[:-1].split('\t'))
                    # if len(pred_answer):
                    #     print('is zero')
                    true_answer = set(id2answers[str(id)])
                    right = len(pred_answer & true_answer)
                    recall = right / len(true_answer)

                    percison = right / len(pred_answer)

                    if percison == 0 or recall == 0:
                        F1 = 0
                    else:
                        F1 = (2 * recall * percison) / (recall + percison)
                else:
                    F1 = 0
                sorce_writer.write(line[:-1] + '---' + str(F1) + '\n')
                F1_list.append(F1)
        print(np.mean(F1_list), len(F1_list))


if __name__ == '__main__':
    pass
    # evaluate_threshold(dir + txt, dir + 'sorce_' + txt, true_path, use_f1=True, threshold=0)
