import json
def _get_true_result(true_path='./test.txt'):
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
            sentence[str(i)]=lines[i*4].strip('\n').split(':')[1]
            # print(true_answer[str(i)])
    return true_answer,sentence

def evaluate(query_result_path,wrong_path):
    '''
    每句的评估值然后算平均
    :param query_result_path:
    :param wrong_path:
    :param right_path:
    :return:
    '''
    id2answers,id2sentence=_get_true_result()
    all_number=0
    all_F1=0
    all_precision=0
    all_recall=0
    wrong_writer=open(wrong_path,'w',encoding='utf-8')
    with open(query_result_path,'r',encoding='utf-8') as reader:
        id2predpath={}
        id2predscore={}
        for line in reader.readlines():

            print(line.strip('\n'))
            line_json=eval(line.strip('\n'))
            id=line_json['id']
            pred_answer=set(line_json['answers'])
            pred_score=line_json['pred_score']
            if len(pred_answer)==0:#预测没有给出答案
                continue
            id2predpath[str(id)]=pred_answer
            id2predscore[str(id)]=pred_score

    print('统计的句子--',len(id2predpath.keys()))

    for id in range(766):
        try:
            true_answer=set(id2answers[str(id)])
            right=len(id2predpath[str(id)]&true_answer)
            recall=right/len(true_answer)
            precison = right / len(id2predpath[str(id)])
            if precison==0 or recall==0:
                F1=0
            else:
                F1=(2*recall*precison)/(recall+precison)

            if F1!=1:
                print(id,'---正确答案不全')
                wrong_writer.write(str(id)+'---'+id2sentence[str(id)]+'---'+id2predscore[str(id)]+'\n')
                # wrong_writer.write('pred_answers---'+str(pred_answer)+'\n')
                # wrong_writer.write('true_answers---'+str(true_answer)+'\n')

            all_number = all_number + 1
            all_F1+=F1
            all_precision+=precison
            all_recall+=recall
        except:
            print('此ID句子不存在，未统计')
    print('precision---', all_precision / all_number, '\n')
    print('recall---', all_recall / all_number, '\n')
    print('F1---',all_F1/all_number,'\n')
    print('number---',all_number)

def evaluate_threshold(query_result_path,threshold):
    '''
    句子得分最大时，且大于某个阈值时才作为正确答案
    :param query_result_path:
    :param threshold:
    :return:
    '''
    id2answers,id2sentence = _get_true_result()
    all_number = 0
    all_F1 = 0
    all_percision = 0
    all_recall = 0
    with open(query_result_path, 'r', encoding='utf-8') as reader:
        for line in reader.readlines():
            # print(line.strip('\n'))
            line_json = eval(line.strip('\n'))
            id = line_json['id']
            pred_answer = set(line_json['answers'])
            pred_socre=float(line_json['pred_score'].split('/')[0])#若最大值未达标，则不进行统计
            if len(pred_answer) == 0 or pred_socre<threshold:  # 预测没有给出答案
                continue
            true_answer = set(id2answers[str(id)])
            right = len(pred_answer & true_answer)
            recall = right / len(true_answer)

            percison = right / len(pred_answer)

            if percison == 0 or recall == 0:
                F1 = 0
            else:
                F1 = (2 * recall * percison) / (recall + percison)
            all_number = all_number + 1
            all_F1 += F1
            all_percision += percison
            all_recall += recall
    print('precision---', all_percision / all_number, '\n')
    print('recall---', all_recall / all_number, '\n')
    print('F1---', all_F1 / all_number, '\n')
    print('number---', all_number)


if __name__ == "__main__":
    # evaluate('./query_result_neg5_dictABD3_selected_?,.txt','./wrong_sentence_selected.txt')
    evaluate_threshold('./222.txt',0.99)
