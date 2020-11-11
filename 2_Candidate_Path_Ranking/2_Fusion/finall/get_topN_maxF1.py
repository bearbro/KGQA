import os
import numpy as np

import pandas as pd
import warnings

from stacking.mainNselect1 import make_data_ok, make_data_ok_next

warnings.filterwarnings("ignore")
'''
前topN里最大的f1
'''

def get_Graph():
    return Graph("http://59.78.194.63:37474", username='neo4j', password='pass')
    # return Graph("http://localhost:7474", username='neo4j', password='pass')

def get_max_F1(next_ok_list, sep, topN, sorce_type='F1'):
    f1_list = []
    for ok_file in next_ok_list:
        f1_max = []
        dir, file = os.path.split(ok_file)
        data = pd.read_csv(ok_file, sep=sep, error_bad_lines=False,encoding='utf-8')
        out_w = open(os.path.join(dir, 'top%d_max_%s.txt' % (topN, sorce_type)), 'w', encoding='utf-8')
        for qid in range(766):
            pred_score = data[data.qid == qid].pred_score
            pred_score_idx = data[data.qid == qid].index
            idxi = pred_score.argsort()[::-1][:topN]
            xxi = data[data.index.isin(pred_score_idx[idxi])].score
            if len(xxi) <= 0:
                print('no %d ' % qid)
                continue
            max_score_idx = xxi.idxmax()
            max_pred_score = data['pred_score'][max_score_idx]
            f1_max.append(data['score'][max_score_idx])
            Q = data.Q[max_score_idx]
            out_w.write('q%d:%s\n' % (qid, Q))
            A = data.A[max_score_idx]
            if type(A) is str:
                pred_path = data.path[max_score_idx]
                out_w.write('%s---%s\n' % (str(max_pred_score), pred_path))
                A_list = A.split('\t')
                A_list = [i[1:-1] for i in A_list]
                A = '\t'.join(A_list)
            else:
                A = 'no answer'
            # out_w.write('%s\n\n' % (A))
            out_w.flush()
        f1_list.append(np.sum(f1_max) / 766)
    return f1_list





if __name__ == '__main__':

    ok_list = ['../data/bret_ext_pred_data_f12020-10-25-17-05-06/ok_result.txt',
               '../data/ckpt_similarity_bert_wwm_ext_f1_net_100,1e-05,8,val_monitor_f1-2020-10-27-08-05-20.hdf5-2020-10-27-18-09-32/ok_result.txt',
               '../data/bert_wwm_ext_2hop_p72020-10-26-02-35-28/ok_result.txt',
               '../data/ckpt_path_ent_rel_bert_wwm_ext-100,1e-05,8-2020-10-26-15-05-24.hdf5-2020-10-26-16-24-49/ok_result.txt',
               '../data/bret_ext_pred_data_f12020-10-25-17-05-06_pathlink/ok_result.txt '
               ]

    ok_list = [r'C:\Users\77385\Desktop\QA'+i[2:] for i in ok_list]
    true_path = './test.txt'
    use_f1 = True
    over_write = False
    sep = '¥'
    sorce_type = ['F1', 'P', 'R'][0]
    # topN = 1
    # next_ok_list = make_data_ok(ok_list, true_path, use_f1, sep=sep, topN=topN, over_write=over_write,sorce_type=sorce_type)
    # f1_list = get_max_F1(next_ok_list, sep, topN)

    # 全部
    next_ok_list = make_data_ok(ok_list, true_path, use_f1, sep=sep, topN=3, over_write=over_write)
    data = make_data_ok_next(next_ok_list, fout=None, use_f1=use_f1, sep=sep, key=['qid', 'path'],
                             kind_deal=2, kind_deal_args=[0.7, 0.2])

    # 直接加权求和取最大
    cl = [i for i in data.columns.to_list() if 'pred_score_' in i]
    sum_pred_score = []
    for i in range(len(data)):
        sumi = 0
        for idx, j in enumerate(cl):
            if pd.isna(data[j][i]):
                print(i, j)
            sumi += data[j][i]
        sum_pred_score.append(sumi/len(cl))
    data['pred_score'] = sum_pred_score
    data = data[['qid', 'path', 'pred_score']]
    data.sort_values("qid", inplace=True)
    data.reset_index(drop=True, inplace=True)
    with open('merge_ok_result.txt', 'w', encoding='utf-8') as fw:
        # fw.write(sep.join(['qid', 'path', 'pred_score']) + '\n')
        qqid = -1
        for idx, qid_i in enumerate(data.qid):
            if qqid != qid_i:
                fw.write('q%d:I dont know!\n' % qid_i)
                fw.write('%f---%s\n' % (data.pred_score[idx], data.path[idx]))
                qqid=qid_i
            else:
                fw.write('%f---%s\n' % (data.pred_score[idx], data.path[idx]))

    ok_list=['merge_ok_result.txt']


    rrr = []
    for topN in [10, 5, 3, 1]:
        next_ok_list = make_data_ok(ok_list, true_path, use_f1, sep=sep, topN=topN, over_write=over_write,
                                    sorce_type=sorce_type)
        f1_list = get_max_F1(next_ok_list, sep, topN)
        for i in f1_list:
            print(i)
            rrr.append(i)


    # dd=pd.read_csv('/Users/brobear/PycharmProjects/QA/data/class_2hop_p7/top1trainN_sorce_answer_ok_result.csv',sep=sep)
    idx = 0
    for topN in [10, 5, 3, 1]:
        print('top',topN, 'max', sorce_type)
        for i in ok_list:
            print(rrr[idx])
            idx += 1
