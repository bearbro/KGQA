import json
import pickle
import sys
import time

from finall.getF1 import get_all_F1

sys.path.insert(0, r'/home/hbxiong/QA')
sys.path.insert(0, r'/home/hbxiong/QA/do_it')
sys.path.insert(0, r'/home/hbxiong/QA/stacking')
print(sys.path)

import os
import pandas as pd
from do_it.get_answer_sorce import evaluate_threshold
from do_it.get_select_answer import get_answer
from py2neo import Graph
import numpy as np
import random
import time
import multiprocessing


def get_Graph():
    return Graph("http://59.78.194.63:37474", username='neo4j', password='pass')
    # return Graph("http://localhost:7474", username='neo4j', password='pass')


def select_topN_from_top10(topN, fout, top10_path, sep):
    '''
        输入
            文件格式
                 qid¥pred_score¥path¥Q¥A¥score
                 0¥0.9702493¥<成败一知己，生死两妇人>|||<主要人物>|||?x¥"""成败一知己，生死两妇人""所说的人物有什么重大成就？"¥'<韩信_（汉初三杰之一）>'¥0.0
                 0¥0.94557583¥<成败一知己，生死两妇人>|||<类别>|||?x¥"""成败一知己，生死两妇人""所说的人物有什么重大成就？"¥'<典故>'¥0.0
    从里面每个问题先前topN个
    :param topN:
    :param fout:
    :param top10_path:
    :return:
    '''
    data10 = pd.read_csv(top10_path, sep=sep, error_bad_lines=False, encoding='utf-8')
    idxlist = None  # 需要的行idx
    maxqid = data10.qid.max()
    for qid in range(maxqid + 1):
        score = data10[data10.qid == qid].pred_score
        score_idx = data10[data10.qid == qid].index
        if len(score) == 0:
            continue
        idxi = score.argsort()[::-1][:topN]
        score_idx[idxi]
        if idxlist is None:
            idxlist = score_idx[idxi]
        else:
            idxlist = idxlist.append(score_idx[idxi])
    head = 'qid¥pred_score¥path¥Q¥A¥score'.split('¥')
    data = data10[data10.index.isin(idxlist)]
    data.to_csv(fout, index=None, columns=head, sep=sep, encoding='utf-8')


def make_it_ok(fin, fout, sep):
    """
    输入
        q【问题id】：【问题】
        【预测分数1】---【预测路径1】---【问题id】---【问题】---【答案1】---【分数1】
        【预测分数2】---【预测路径2】---【问题id】---【问题】---【答案2】---【分数2】
    输出
        qid¥pred_score¥path¥Q¥A¥score
        【问题id】¥【预测分数1】¥【预测路径1】¥【问题】¥【答案1】¥【分数1】
        【问题id】¥【预测分数2】¥【预测路径2】¥【问题】¥【答案2】¥【分数2】
        例子
    """
    with open(fin, 'r', encoding='utf-8') as file_read:
        lines = file_read.readlines()
        pred_score = []
        path = []
        qid = []
        question = []
        score = []
        answer = []
        for line in lines:
            if line[0] == 'q':
                continue
            data_i = line[:-1].split('---')
            pred_score.append(float(data_i[0]))
            path.append(data_i[1])
            qid.append(int(data_i[2]))
            question.append(data_i[3])
            answer.append(data_i[4])
            score.append(float(data_i[-1]))
    data = [qid, pred_score, path, question, answer, score]
    head = 'qid¥pred_score¥path¥Q¥A¥score'.split('¥')
    data = {head[i]: v for i, v in enumerate(data)}
    df = pd.DataFrame(data)
    df.to_csv(fout, index=None, columns=head, sep=sep, encoding='utf-8')


def make_data_ok(ok_list, true_path, use_f1, sep, topN=10, over_write=True):
    '''
    数据构造1
    输入：
        ok_list：预测的结果文件ok的列表（完整地址）
                各文件类似如下
                q0:"成败一知己，生死两妇人"所说的人物有什么重大成就？
                0.996529---<成败一知己，生死两妇人>|||<相关成语>|||?x
                0.13312617---<成败一知己，生死两妇人>|||<主要人物>|||?x
                0.010742664---<成败一知己，生死两妇人>|||<类别>|||?x
        true_path:正确答案文件的路径
                q1:"成败一知己，生死两妇人"所说的人物有什么重大成就？
                select ?y where { <成败一知己，生死两妇人> <主要人物> ?x . ?x <主要成就> ?y . }
                "垓下破 项羽"	"潍水杀 龙且"	"虏魏"	"下燕"	"破代"	"平赵"	"定齐"

                q2:葬于茂陵的皇帝在位于哪段时间？
                select ?y where { ?x <陵墓> <茂陵_（汉武帝陵寝）> . ?x <在位时间> ?y . }
                "公元前141年―公元前87年"
        use_f1:是否计算f1
        topN:各路径取前几个
        over_write:是否覆盖生成文件
    输出:
        :return 生成文件的list
            文件SN:(A，A_F1可能为空和0)，A中多个答案之间用\t划分
                  得分，A_path，Qid，Q，A，A_F1
            文件格式
                 qid¥pred_score¥path¥Q¥A¥score
                 0¥0.9702493¥<成败一知己，生死两妇人>|||<主要人物>|||?x¥"""成败一知己，生死两妇人""所说的人物有什么重大成就？"¥'<韩信_（汉初三杰之一）>'¥0.0
                 0¥0.94557583¥<成败一知己，生死两妇人>|||<类别>|||?x¥"""成败一知己，生死两妇人""所说的人物有什么重大成就？"¥'<典故>'¥0.0

    '''

    out_list = []
    for ok_i in ok_list:
        dir, file = os.path.split(ok_i)
        if 'ok' in file and topN != 10:
            fout = os.path.join(dir, 'top%dtrainN_sorce_answer_%s.csv' % (topN, file[:-4]))
        else:
            fout = os.path.join(dir, 'trainN_sorce_answer_%s.csv' % (file[:-4]))
        if os.path.exists(fout) and not over_write:
            out_list.append(fout)
            continue
        top10_path = os.path.join(dir, 'trainN_sorce_answer_%s.csv' % (file[:-4]))
        if not os.path.exists(fout) and \
                os.path.exists(top10_path) \
                and topN < 10 and 'ok' in file:  # 从前10挑出前topN
            select_topN_from_top10(topN, fout, top10_path, sep)
            out_list.append(fout)
            continue

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
        fin = os.path.join(dir, file)
        fout = os.path.join(dir, 'answer_' + file)  # 获得查询的答案
        graph = get_Graph()
        if not os.path.exists(fout) or over_write:
            get_answer(fin, fout, graph, use_f1=use_f1, pre_sorces_k=topN)
        """
        将生成的文件格式
        q【问题id】：【问题】
        【预测分数1】---【预测路径1】---【问题id】---【问题】---【答案1】---【分数1】
        【预测分数2】---【预测路径2】---【问题id】---【问题】---【答案2】---【分数2】
        例子
        q2:"光武中兴"说的是哪位皇帝？
        0.9999997---<建武中兴>|||<皇帝>|||?x---2---"光武中兴"说的是哪位皇帝？---'<后醍醐天皇>'---0.0 
        """
        fin = os.path.join(dir, 'answer_' + file)
        fout = os.path.join(dir, 'sorce_answer_' + file)
        if not os.path.exists(fout) or over_write:
            evaluate_threshold(fin, fout, true_path, use_f1=use_f1, threshold=0)  # 根据查询的答案和正确答案计算F1
        """
        qid¥pred_score¥path¥Q¥A¥score
        【预测分数1】¥【预测路径1】¥【问题id】¥【问题】¥【答案1】¥【分数1】
        【预测分数2】¥【预测路径2】¥【问题id】¥【问题】¥【答案2】¥【分数2】
        例子
        """
        fin = os.path.join(dir, 'sorce_answer_' + file)
        fout = os.path.join(dir, 'trainN_sorce_answer_' + file[:-4] + '.csv')
        if not os.path.exists(fout) or over_write:
            make_it_ok(fin, fout, sep=sep)  # 构造csv文件
        out_list.append(fout)
    return out_list


def make_data_ok_next(next_ok_list, fout, use_f1, sep, key=['qid', 'path'], kind_deal=2, kind_deal_args=[0.8, 0]):
    '''
    数据构造2
    输入：N模型的输出文件S1，...，SN
    qid¥pred_score¥path¥Q¥A¥score
    do：将key相同的行合并为一行(key=['qid','path'],use_f1=True时key=['qid','A']]
    对缺失值进行处理（方案一：直接为0，方案二：直接为同文件的最低分*deal_d，方案三：同文件该问题的最低分*deal_d√）
    根据A_path和A_path_true计算标签（也可以用F1？但查询耗时）
    输出：Q，A_path,分数1，...，分数3，标签
    '''
    data_out = None
    if key != ['qid', 'path'] and key != ['qid', 'A']:
        print('错误！')
        raise Exception('key 错误！ key=[\'qid\',\'path\'],use_f1=True时key=[\'qid\',\'A\']]')

    for idx, path in enumerate(next_ok_list):
        data = pd.read_csv(path, sep=sep, error_bad_lines=False, encoding='utf-8')
        if idx != 0:
            data.drop(['Q'], axis=1, inplace=True)
        if key == ['qid', 'path']:
            data.rename(columns={'pred_score': 'pred_score_%d' % idx, 'A': 'A_%d' % idx,
                                 'score': 'score_%d' % idx}, inplace=True)
            if data_out is None:
                data_out = data
                continue
            # 匹配
            data_out = pd.merge(data_out, data, left_on=key, right_on=key, how='outer')

        if key == ['qid', 'A'] and use_f1:
            data.rename(columns={'pred_score': 'pred_score_%d' % idx, 'path': 'path_%d' % idx,
                                 'score': 'score_%d' % idx}, inplace=True)
            if data_out is None:
                data_out = data
                continue
            # 匹配
            data_out = pd.merge(data_out, data, left_on=key, right_on=key, how='outer')

    # 合并列[path,A,f1]
    if key == ['qid', 'path']:
        cl = ['A', 'score']
    if key == ['qid', 'A'] and use_f1:
        cl = ['path', 'score']

    A_score = [data_out['%s_0' % i].copy() for i in cl]
    for i in range(len(data_out)):
        for name_i, name in enumerate(cl):
            ci = 0
            while (pd.isna(data_out['%s_%d' % (name, ci)][i])):
                ci += 1
                if ci >= len(next_ok_list):
                    ci = 0
                    print('error', i, name)
                    break
            A_score[name_i][i] = data_out['%s_%d' % (name, ci)][i]
    head = data_out.columns.to_list() + cl
    for name_i, name in enumerate(cl):
        data_out[name] = A_score[name_i]

    # 缺失值处理
    cl = ['pred_score_%d' % i for i, _ in enumerate(next_ok_list)]
    if kind_deal == 0:
        for ci in cl:
            data_out[ci] = data_out[ci].fillna(0)
    elif kind_deal == 1:
        # 方案二：max(同文件的最低分*k1-k2,0)零截断A
        for ci in cl:
            c_min = data_out[ci].min() * kind_deal_args[0] - kind_deal_args[1]
            if pd.isna(c_min):
                c_min = 0
            else:
                c_min = max(c_min, 0)
            data_out[ci] = data_out[ci].fillna(float(c_min))
    elif kind_deal == 2:
        # 方案三：max(同文件该问题的最低分*k1-k2,0)零截断
        for ci in cl:
            new_ci = data_out[ci].copy()
            for i in data_out[pd.isna(data_out[ci])].index.tolist():
                r_qid = data_out.qid[i]
                c_min = data_out[ci][data_out.qid == r_qid].min() * kind_deal_args[0] - kind_deal_args[1]
                if np.isnan(c_min):
                    c_min = 0
                    print(i, ci, 'is nan')
                else:
                    c_min = max(c_min, 0)
                new_ci[i] = c_min
            data_out[ci] = new_ci

    else:
        raise Exception('kind_deal 错误！ kind_deal in{0,1,2}')
    if fout != None:
        data_out.to_csv(fout, index=None, columns=head, sep=sep, encoding='utf-8')
    return data_out


def worker(new_ok_list, ok_idx, topN, q):
    for ok_list in new_ok_list:
        for i in ok_list:
            print(i)
        true_path = '../test.txt'
        use_f1 = True
        over_write = False
        sep = '¥'
        next_ok_list = make_data_ok(ok_list, true_path, use_f1, sep=sep, topN=topN, over_write=over_write)
        # xxx = pd.read_csv(next_ok_list[0], sep='¥', error_bad_lines=False,encoding='utf-8')
        # print(len(xxx))
        dir = '../stacking'
        # if not os.path.exists(dir):
        #     os.mkdir(dir)

        fout = None  # os.path.join(dir, 'merge_trainN_sorce_answer_sim.csv')
        data = make_data_ok_next(next_ok_list, fout, use_f1=True, sep=sep, key=['qid', 'path'], kind_deal=2,
                                 kind_deal_args=[0.8, 0])
        # xxx = pd.read_csv(fout, sep='¥', error_bad_lines=False,encoding='utf-8')
        # print(len(xxx))

        # 直接加权求和取最大
        w = [1] * len(next_ok_list)
        # data = pd.read_csv(fout, sep='¥', error_bad_lines=False,encoding='utf-8')
        cl = [i for i in data.columns.to_list() if 'pred_score_' in i]
        sum_pred_score = []
        for i in range(len(data)):
            sumi = 0
            for idx, j in enumerate(cl):
                if pd.isna(data[j][i]):
                    print(i, j)
                sumi += data[j][i] * w[idx]
            sum_pred_score.append(sumi)
        data['sum_pred_score'] = sum_pred_score

        f1 = []
        for qid in range(766):
            xxi = data[data.qid == qid]['sum_pred_score']
            if len(xxi) <= 0:
                print('no %d ' % qid)
                continue
            max_pred_score_idx = xxi.idxmax()
            max_pred_score = data['sum_pred_score'][max_pred_score_idx]
            f1i = data['score'][max_pred_score_idx]
            if not pd.isna(f1i):
                f1.append(f1i)
        t = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print(t, len(f1), np.mean(f1))
        if 'ok' in ok_list[0]:
            sss = 'ok'
        else:
            sss = 'pred'
        with open('top%d_%s_result_%d.txt' % (topN, sss, ok_idx), 'a', encoding='utf-8') as myw:
            for xxx in [np.mean(f1), len(f1), len(ok_list)]:
                myw.write(str(xxx) + '\t')
            xxx = '\', \''.join(ok_list)
            myw.write('[ \'')
            myw.write(xxx)
            myw.write(' \']')
            myw.write('\n')
        # q.put([np.mean(f1), len(f1), len(ok_list), ok_list])
        # # 网格调参 https://www.jianshu.com/p/35eed1567463
        # from hyperopt import fmin, tpe, hp, Trials
        #
        # # todo 缺失值填充时的选择min*x1-x2
        # space = dict()
        # for i in range(len(ok_list)):
        #     # space['w_%d' % i] = hp.uniform('w_%d' % i, 0, 1)
        #     # space['w_%d' % i] = hp.choice('w_%d' % i, [ix / 100 for ix in range(101)])
        #     space['w_%d' % i] = 1
        #
        # space['k1'] = hp.choice('k1', [ix / 100 for ix in range(50, 101)])
        # space['k2'] = hp.choice('k2', [ix / 100 for ix in range(50)])
        # space = {
        #     'w0': hp.uniform('w0', 0, 1),
        #     'w1': hp.uniform('w1', 0, 1),
        #     'w2': hp.uniform('w2', 0, 1),
        #     'w3': hp.uniform('w4', 0, 1),
        # }
        # data_set = dict()

        # #
        # def fn(params):
        #     if 'k1' in params:
        #         k1 = params['k1']
        #     else:
        #         k1 = 1
        #     if 'k2' in params:
        #         k2 = params['k2']
        #     else:
        #         k2 = 0
        #     if (k1, k2) not in data_set:
        #         data = make_data_ok_next(next_ok_list, fout=None, use_f1=True, sep=sep, key=['qid', 'path'], kind_deal=2,
        #                                  kind_deal_args=[k1, k2])
        #         data_set[(k1, k2)] = data
        #     else:
        #         data = data_set[(k1, k2)]
        #
        #     # 直接加权求和取最大
        #     i = 0
        #     w = [1]
        #     while ('w_%d' % i) in params:
        #          w.append(params['w_%d' % i])
        #          i += 1
        #     cl = [i for i in data.columns.to_list() if 'pred_score_' in i]
        #     if len(cl) != len(w):
        #         print(cl)
        #         print(w)
        #     sum_pred_score = []
        #     for i in range(len(data)):
        #         sumi = 0
        #         for idx, j in enumerate(cl):
        #             if pd.isna(data[j][i]):
        #                 print(i, j)
        #             sumi += data[j][i] * w[idx]
        #         sum_pred_score.append(sumi)
        #     data['sum_pred_score'] = sum_pred_score
        #     f1 = []
        #     for qid in range(766):
        #         xxi = data[data.qid == qid]['sum_pred_score']
        #         if len(xxi) <= 0:
        #             print('no %d ' % qid)
        #             continue
        #         max_pred_score_idx = xxi.idxmax()
        #         max_pred_score = data['sum_pred_score'][max_pred_score_idx]
        #         f1i = data['score'][max_pred_score_idx]
        #         if not pd.isna(f1i):
        #             f1.append(f1i)
        #     return np.mean(f1) * -1
        #
        #
        # trials = Trials()
        # best = fmin(
        #     fn=fn,
        #     space=space,
        #     algo=tpe.suggest,
        #     max_evals=200,
        #     verbose=True,
        #     trials=trials,
        #     max_queue_len=4)
        # print('trials')
        # for trial in trials.trials[:2]:
        #     print(trial)
        #
        # print(best)
        # for i in ok_list:
        #     print(i)


def get_finall_answer(next_ok_list, fout, sep, k1=0.8, k2=0):
    '''
    根据多个候选答案生成最终答案
    :param next_ok_list: make_data_ok的输出文件的路径， 文件内数据：qid¥pred_score¥path¥Q¥A¥score
    :param fout: 最终的答案文件
    :param sep: 分隔符
    :return:
    '''
    data = make_data_ok_next(next_ok_list, fout=None, use_f1=False, sep=sep, key=['qid', 'path'], kind_deal=2,
                             kind_deal_args=[k1, k2])
    # 直接加权求和取最大
    w = [1] * len(next_ok_list)
    # data = pd.read_csv(fout, sep='¥', error_bad_lines=False,encoding='utf-8')
    cl = [i for i in data.columns.to_list() if 'pred_score_' in i]
    sum_pred_score = []
    for i in range(len(data)):
        sumi = 0
        for idx, j in enumerate(cl):
            if pd.isna(data[j][i]):
                print(i, j)
            sumi += data[j][i] * w[idx]
        sum_pred_score.append(sumi)
    data['sum_pred_score'] = sum_pred_score
    out_w = open(fout, 'w', encoding='utf-8')
    for qid in range(766):
        xxi = data[data.qid == qid]['sum_pred_score']
        if len(xxi) <= 0:
            print('no %d ' % qid)
            continue
        max_pred_score_idx = xxi.idxmax()
        max_pred_score = data['sum_pred_score'][max_pred_score_idx] / len(next_ok_list)
        Q = data.Q[max_pred_score_idx]
        # Q = 'sorry I forget it'
        out_w.write('q%d:%s\n' % (qid, Q))
        A = data.A[max_pred_score_idx]
        if type(A) is str:
            pred_path = data.path[max_pred_score_idx]
            out_w.write('%f---%s\n' % (max_pred_score, pred_path))
            A_list = A.split('\t')
            A_list = [i[1:-1] for i in A_list]
            A = '\t'.join(A_list)
        else:
            A = 'no answer'
        # out_w.write('%s\n\n' % (A))
        out_w.flush()


import re
def xz_answer(in_file, out_file, true_path='./test.txt'):
    def jian1(matched):
        # 将匹配的数字 -1
        value = int(matched.group('value'))
        return 'q%d:'% (value - 1)

    with open(in_file, 'r', encoding='utf-8') as fr:
        rlines = fr.readlines()
    with open(true_path, 'r', encoding='utf-8') as fr:
        rlines2 = fr.readlines()
        rlines2 = rlines2[::4]
    for i in range(len(rlines)):
        if i % 2 == 0:
            rlines[i] = rlines2[i // 2]
            rlines[i]=re.sub('q(?P<value>\d+):', jian1, rlines[i])
    with open(out_file, 'w', encoding='utf-8') as fw:
        fw.writelines(rlines)


def get_answer_answer(path_file,path=False):
    infile=path_file.replace('.txt','_score/query_finallanswer.txt')
    if not os.path.exists(infile):
        get_all_F1(path_file)
    out_file=path_file.replace('.txt','_answer.txt')
    with open(path_file,'r',encoding='utf-8') as fr:
        rlines=fr.readlines()
    with open(infile,'r',encoding='utf-8') as fr:
        rlines2=fr.readlines()

    def jia1(matched):
        # 将匹配的数字 -1
        value = int(matched.group('value'))
        return 'q%d:'% (value + 1)
    for i in range(len(rlines)):
        if i % 2 == 0:
            rlines[i] = re.sub('q(?P<value>\d+):', jia1, rlines[i])
        if i%2==1:
            ans=eval(rlines2[i//2])['answers']
            # print('\t'.join(ans))
            rlines[i]='\t'.join(ans)+'\r'

    with open(out_file, 'w', encoding='utf-8') as fw:
        fw.writelines(rlines)




if __name__ == '__main__':
    ok_list = ['../data/bret_ext_pred_data_f12020-10-25-17-05-06/ok_result.txt',
               '../data/ckpt_similarity_bert_wwm_ext_f1_net_100,1e-05,8,val_monitor_f1-2020-10-27-08-05-20.hdf5-2020-10-27-18-09-32/ok_result.txt',
               '../data/bert_wwm_ext_2hop_p72020-10-26-02-35-28/ok_result.txt',
               '../data/ckpt_path_ent_rel_bert_wwm_ext-100,1e-05,8-2020-10-26-15-05-24.hdf5-2020-10-26-16-24-49/ok_result.txt',
               '../data/bret_ext_pred_data_f12020-10-25-17-05-06_pathlink/ok_result.txt '
               ]
    true_path = '../test.txt'
    use_f1 = True
    over_write = False
    topN = 3
    sep = '¥'
    next_ok_list = make_data_ok(ok_list, true_path, use_f1, sep=sep, topN=topN, over_write=over_write)
    
    finall_dir = 'finall_top%d' % topN
    if not os.path.exists(finall_dir):
        os.mkdir(finall_dir)
    out_file = os.path.join(finall_dir, 'finallanswer.txt')
    get_finall_answer(next_ok_list, out_file, sep, k1=0.7, k2=0.2)
    # 根据text.txt补全题目 题目存在 nan的情况 根据text修正
    xz_answer(out_file, out_file)

    get_all_F1(out_file)
    get_answer_answer(out_file)
