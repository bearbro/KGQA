# ! -*- coding: utf-8 -*-

import codecs
import csv
import gc
import os
import pickle
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import *
# 引入Tensorboard
from keras.callbacks import TensorBoard
from keras.layers import *
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert.backend import keras
from keras_contrib.layers import CRF
from sklearn.model_selection import KFold
from tqdm import tqdm

from GRU_model import MyGRU

topN=None

def all_args(args):
    r=[]

    def add_one(ri,args):
        ri=ri.copy()
        ri[-1]+=1
        x=len(args)-1
        while ri[x]>= len(args[x]):
            ri[x]=0
            x-=1
            ri[x] += 1
            if x==-1:
                return None
        return ri
    ri=[0]*len(args)
    r.append(ri)
    ri=add_one(ri, args)
    while ri!=None:
        r.append(ri)
        ri = add_one(ri, args)

    rr=[[args[idx][i] for idx,i in enumerate(ri)] for ri in r]
    return rr

#maxlen,learning_rate,min_learning_rate,bsize,need_one,lstm_unit,used_mask,used_mask0,used_wup,bert_kind,tagkind,model_type,acc_type
args=[[60],[5e-5],[1e-5],[16],[True],[64,[True],[True],[False],['tf-bert_wwm_ext'],['BIO'],['DT'],['sig_acc']]
for args_i in all_args(args):
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    maxlen,learning_rate,min_learning_rate,bsize,need_one,lstm_unit,used_mask,used_mask0,used_wup,bert_kind,tagkind,model_type,acc_type=args_i
    # maxlen = 55  #
    # learning_rate = 5e-5  #
    # min_learning_rate = 1e-5  # 1e-5
    # bsize = 16
    # need_one=True
    #
    # lstm_unit=64
    # used_mask=True
    # used_wup=False
    # bert_kind = ['chinese_L-12_H-768_A-12', 'tf-bert_wwm_ext'][1]
    # tagkind = ['BIO', 'BIOES'][0]
    # model_type=['DT','LSTM'][0]
    # acc_type=['sig_acc','all_acc','f1'][0]

    # path_root='/Users/brobear/OneDrive/ccks2020-onedrive/ccks2020/bert/'#mac
    # path_root=r'C:\Users\bear\OneDrive\ccks2020-onedrive\ccks2020\bert'#room
    path_root='/home/hbxiong/ccks/bert'#lab
    config_path = os.path.join(path_root, bert_kind, 'bert_config.json')
    checkpoint_path = os.path.join(path_root, bert_kind, 'bert_model.ckpt')
    dict_path = os.path.join(path_root, bert_kind, 'vocab.txt')


    model_save_path = "./ckpt"
    train_data_path = './data/bio_ner_train.txt'
    dev_data_path = './data/bio_ner_valid.txt'
    test_data_path = './data/bio_ner_test.txt'



    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    token_dict = {}

    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)


    class OurTokenizer(Tokenizer):
        def _tokenize(self, text):
            R = []
            for c in text:
                if c in self._token_dict:
                    R.append(c)
                elif self._is_space(c):
                    R.append('[unused1]')  # space类用未经训练的[unused1]表示
                else:
                    R.append('[UNK]')  # 剩余的字符是[UNK]
            return R


    tokenizer = OurTokenizer(token_dict)


    def delete_tag(s):
        s = re.sub('\{IMG:.?.?.?\}', '', s)  # 图片
        s = re.sub(re.compile(r'[a-zA-Z]+://[^\s]+'), '', s)  # 网址
        s = re.sub(re.compile('<.*?>'), '', s)  # 网页标签
        s = re.sub(re.compile('&[a-zA-Z]+;?'), '', s)  # 网页标签
        s = re.sub(re.compile('[a-zA-Z0-9]*[./]+[a-zA-Z0-9./]+[a-zA-Z0-9./]*'), '', s)
        s = re.sub("\?{2,}", "", s)
        # s = re.sub("（", ",", s)
        # s = re.sub("）", ",", s)
        s = re.sub(" \(", "（", s)
        s = re.sub("\) ", "）", s)
        s = re.sub("\u3000", "", s)
        # s = re.sub(" ", "", s)
        r4 = re.compile('\d{4}[-/年](\d{2}([-/月]\d{2}[日]{0,1}){0,1}){0,1}')  # 日期
        s = re.sub(r4, "▲", s)
        return s


    if tagkind == 'BIO':
        BIOtag = ['O','I','B']
    elif tagkind == 'BIOES':
        BIOtag = ['O',  'I','B', 'E', 'S']
    tag2id = {v: i for i, v in enumerate(BIOtag)}

    def make_data(path,test=False):
        with open(path,'r',encoding='utf-8') as fread:
            lines=fread.readlines()
        X=lines[::2]
        Y=lines[1::2]
        X=[i[:-1] for i in X]
        Y=[eval(i) for i in Y]
        # 验证
        error = [(X[i], Y[i]) for i in range(len(X)) if len(X[i])!=len(Y[i])]

        if tagkind!='BIO':
            # todo 修改BIOtag
            def BIO2BIOES(x):
                for idx,i in enumerate(x):
                    if i==tag2id['B']:
                        if idx<len(x)-1 and x[idx+1]==tag2id['I']:
                            pass
                        else:
                            x[idx]=tag2id['S']
                    elif i==tag2id['I']:
                        if idx<len(x)-1 and x[idx+1]==tag2id['I']:
                            pass
                        else:
                            x[idx]=tag2id['E']
                return x
            Y=[BIO2BIOES(i) for i in Y]
        if len(error)!=0:
            print('data error',error)
        if test:
            X=[(idx,i) for idx,i in enumerate(X)]
            return X,Y
        else:
            d=[(X[i],Y[i]) for i in range(len(X))]
            return d

    # 读取训练集

    train_data = make_data(train_data_path)
    print('最终训练集大小:%d' % len(train_data))

    print('-' * 30)
    dev_data = make_data(dev_data_path)
    print('最终验证集大小:%d' % len(dev_data))
    print('-' * 30)

    test_data,test_label = make_data(test_data_path,test=True)
    print('最终测试集大小:%d' % len(test_data))
    print('-' * 30)




    def getBIO(text, e):
        text = text[:maxlen]
        x1 = tokenizer.tokenize(text)
        p1=[tag2id['O']]+e+[tag2id['O']]
        # 根据实体获得 p1
        # p1 = [0] * len(x1)
        # for ei in e.split(';'):
        #     if ei == '':
        #         continue
        #     x2 = tokenizer.tokenize(ei)[1:-1]
        #     # print(x2)
        #     for i in range(len(x1) - len(x2)):
        #         if x2 == x1[i:i + len(x2)] and sum(p1[i:i + len(x2)]) == 0:
        #             if tagkind == 'BIO':
        #                 pei = [tag2id['I']] * len(x2)
        #                 pei[0] = tag2id['B']
        #             elif tagkind == 'BIOES':
        #                 pei = [tag2id['I']] * len(x2)
        #                 if len(x2) == 1:
        #                     pei[0] = tag2id['S']
        #                 else:
        #                     pei[0] = tag2id['B']
        #                     pei[-1] = tag2id['E']
        #             p1[i:i + len(x2)] = pei

        maxN = len(BIOtag)
        id2matrix = lambda i: [1 if x == i else 0 for x in range(maxN)]
        p1 = [id2matrix(i) for i in p1]

        return p1


    def seq_padding(X, padding=0, wd=1):
        L = [len(x) for x in X]
        ML = max(L)  # maxlen
        if wd == 1:
            return np.array([
                np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
            ])
        else:
            padding_wd = [padding] * len(X[0][0])
            padding_wd[tag2id['O']] = 1
            return np.array([
                np.concatenate([x, [padding_wd] * (ML - len(x))]) if len(x) < ML else x for x in X
            ])


    class data_generator:
        def __init__(self, data, batch_size=bsize):
            self.data = data
            self.batch_size = batch_size
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1

        def __len__(self):
            return self.steps

        def __iter__(self):
            while True:
                idxs = list(range(len(self.data)))
                np.random.shuffle(idxs)
                X1, X2, P = [], [], []
                for i in idxs:
                    d = self.data[i]
                    text = d[0][:maxlen]
                    e = d[1]
                    # todo 构造标签
                    p = getBIO(text, e)
                    x1, x2 = tokenizer.encode(text)
                    X1.append(x1)
                    X2.append(x2)
                    P.append(p)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        P = seq_padding(P, wd=2)
                        yield [X1, X2], P
                        X1, X2, P = [], [], []


    # 定义模型


    def modify_bert_model_3():
        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))  # 待识别句子输入
        x2_in = Input(shape=(None,))  # 待识别句子输入

        x1, x2 = x1_in, x2_in

        bert_out = bert_model([x1, x2])  # [batch,maxL,768]
        # todo [batch,maxL,768] -》[batch,maxL,3]
        xlen = 1
        a = Dense(units=xlen, use_bias=False, activation='tanh')(bert_out)  # [batch,maxL,1]
        b = Dense(units=xlen, use_bias=False, activation='tanh')(bert_out)
        c = Dense(units=xlen, use_bias=False, activation='tanh')(bert_out)
        outputs = Lambda(lambda x: K.concatenate(x, axis=-1))([a, b, c])  # [batch,maxL,3]
        outputs = Softmax()(outputs)
        model = keras.models.Model([x1_in, x2_in], outputs, name='basic_model')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()

        return model


    def modify_bert_model_3_masking():
        bert_model = load_trained_model_from_checkpoint(
            config_path, checkpoint_path,
            # output_layer_num=4
        )

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))  # 待识别句子输入
        x2_in = Input(shape=(None,))  # 待识别句子输入

        x1, x2 = x1_in, x2_in

        outputs = bert_model([x1, x2])  # [batch,maxL,768]
        # Masking
        xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
        outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
        outputs = Masking(mask_value=0)(outputs)  # error ?
        outputs = Dense(units=len(BIOtag), use_bias=False, activation='Softmax')(outputs)

        model = keras.models.Model([x1_in, x2_in], outputs, name='basic_masking_model')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()

        return model


    def modify_bert_model_bilstm_mask_crf():
        bert_model = load_trained_model_from_checkpoint(
            config_path, checkpoint_path,
            output_layer_num=4
        )

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))  # 待识别句子输入
        x2_in = Input(shape=(None,))  # 待识别句子输入

        x1, x2 = x1_in, x2_in

        outputs = bert_model([x1, x2])  # [batch,maxL,768]
        # Masking
        # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
        # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
        # outputs = Masking(mask_value=0)(outputs)
        outputs = Bidirectional(LSTM(units=lstm_unit, return_sequences=True))(outputs)
        outputs = Dropout(0.2)(outputs)
        mask = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(x1_in)
        crf = CRF(len(BIOtag), sparse_target=False)
        if used_mask:
            outputs = crf(outputs, mask=mask)
        else:
            outputs = crf(outputs)

        model = keras.models.Model([x1_in, x2_in], outputs, name='basic_bilstm_crf_model')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()

        return model


    def modify_bert_model_3_crf():
        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))  # 待识别句子输入
        x2_in = Input(shape=(None,))  # 待识别句子输入

        x1, x2 = x1_in, x2_in

        outputs = bert_model([x1, x2])  # [batch,maxL,768]
        #  [batch,maxL,768] -》[batch,maxL,len(BIOtag)]

        # # Masking
        # xm = Lambda(lambda x: tf.expand_dims(tf.clip_by_value(x1, 0, 1), -1))(x1)
        # outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, xm])
        # outputs = Masking(mask_value=0)(outputs)

        # outputs = Dense(units=len(BIOtag), use_bias=False, activation='tanh')(outputs)  # [batch,maxL,3]
        # outputs = Lambda(lambda x: x)(outputs)
        # outputs = Softmax()(outputs)

        # crf
        crf = CRF(len(BIOtag), sparse_target=False)
        outputs = crf(outputs)

        model = keras.models.Model([x1_in, x2_in], outputs, name='basic_crf_model')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()

        return model


    def modify_bert_model_biMyGRU_mask_crf():
        bert_model = load_trained_model_from_checkpoint(
            config_path, checkpoint_path,
            output_layer_num=4
        )

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))  # 待识别句子输入
        x2_in = Input(shape=(None,))  # 待识别句子输入

        x1, x2 = x1_in, x2_in

        outputs = bert_model([x1, x2])  # [batch,maxL,768]

        outputs = Bidirectional(MyGRU(units=lstm_unit, return_sequences=True, reset_after=True, name='MyGRU', tcell_num=3))(
            outputs)
        outputs = Dropout(0.2)(outputs)#n,n,600

        #MASK 排除padding
        if used_mask0:
            mask0 = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1_in)
            outputs=keras.layers.Multiply()([outputs,mask0])
        mask = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(x1_in)
        crf = CRF(len(BIOtag), sparse_target=False)
        if used_mask:
            outputs = crf(outputs,mask=mask)
        else:
            outputs = crf(outputs)


        model = keras.models.Model([x1_in, x2_in], outputs, name='basic_biMyGRU_crf_model')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()

        return model




    def decode(text_in, p_in,one=True):
        '''解码函数'''
        p = np.argmax(p_in, axis=-1)
        # _tokens = tokenizer.tokenize(text_in)
        _tokens=text_in
        if len(p)-2==len(text_in):
            _tokens = ' %s ' % text_in
        ei = ''
        r = []
        if tagkind == 'BIO':
            for i, v in enumerate(p):
                if ei == '':
                    if v == tag2id['B']:
                        ei += _tokens[i]
                else:
                    if v == tag2id['B']:
                        r.append(ei)
                        ei = _tokens[i]
                    elif v == tag2id['I']:
                        ei += _tokens[i]
                    elif v == tag2id['O']:
                        r.append(ei)
                        ei = ''
        elif tagkind == 'BIOES':
            for i, v in enumerate(p):
                if ei == '':
                    if v == tag2id['B']:
                        ei = _tokens[i]
                    elif v == tag2id['S']:
                        r.append(_tokens[i])
                else:
                    if v == tag2id['B']:
                        ei = _tokens[i]
                    elif v == tag2id['I']:
                        ei += _tokens[i]
                    elif v == tag2id['E']:
                        ei += _tokens[i]
                        r.append(ei)
                        ei = ''
                    elif v == tag2id['O']:
                        # r.append(ei)
                        ei = ''
                    elif v == tag2id['S']:
                        r.append(_tokens[i])
                        ei = ''
        if not one:
            r = [i for i in r if len(i) > 1]
        # r = list(set(r))
        # r.sort()
        return ';'.join(r)


    def extract_entity(text_in, batch=None):
        """解码函数,prb=True 返回解码前的值
        """
        if batch == None:
            text_in = text_in[:maxlen]
            _tokens = tokenizer.tokenize(text_in)
            _x1, _x2 = tokenizer.encode(text_in)
            _x1, _x2 = np.array([_x1]), np.array([_x2])
            _p = model.predict([_x1, _x2])[0]
            a = decode(text_in, _p,one=need_one)
            return a,_p
        else:
            text_in = [i[:maxlen] for i in text_in]
            ml = max([len(i) for i in text_in])+2
            x1x2 = [tokenizer.encode(i, max_len=ml) for i in text_in]
            _x1 = np.array([i[0] for i in x1x2])
            _x2 = np.array([i[1] for i in x1x2])
            _p = model.predict([_x1, _x2])
            # print(ml,_p.shape)
            a=[]
            p=[]
            for i in range(len(text_in)):
                a.append(decode(text_in[i], _p[i][1:len(text_in[i])+1],one=need_one))
                p.append(_p[i][1:len(text_in[i])+1])
            return a,p


    def myF1_P_R(y_true, y_pre):
        a = set(y_true.split(';'))
        b = set(y_pre.split(';'))
        TP = len(a & b)
        FN = len(a - b)
        FP = len(b - a)
        P = TP / (TP + FP) if TP + FP != 0 else 0
        R = TP / (TP + FN) if TP + FN != 0 else 0
        F1 = 2 * P * R / (P + R) if P + R != 0 else 0

        return F1, P, R


    def my_acc(d,Y_p, y_p):
        '''计算acc'''
        tokens_id, _x2 = tokenizer.encode(d)
        Y_p=np.argmax(Y_p,axis=1)  # 形如：[0,2,1,1,0,0,0]
        y_p=np.argmax(y_p,axis=1)  # 形如：[0,2,1,1,0,0,0]
        # print(d)
        # print(Y_p)
        # print(y_p)
        acc=(lambda x,y: np.equal(x, y).astype(int))(Y_p,y_p)
        # print(tokens_id)
        mask = (lambda x: np.greater(x, 0).astype(int)[1:-1])(tokens_id)  # 形如：[1,1,1,1,0,0,0]
        # 获得大于0的为1否则为0 分离出padding的id
        return np.sum(acc*mask)/np.sum(mask)



    def evaluate(dev_data, batch=1,tag=None):
        A = 1e-10
        F = 1e-10
        acc=1e-10
        if tag != None:
            error_f=open('ckpt/error-%s.txt'%tag, 'w', encoding='utf-8')
        for idx in tqdm(range(0, len(dev_data), batch)):
            d = [i[0][:maxlen] for i in dev_data[idx:idx + batch]]
            Y = [i[1] for i in dev_data[idx:idx + batch]]  # 形如：[[0,2,1,1,0,0,0],..]

            if type(Y[0][0]) not in [list,np.ndarray]:
                id2matrix = lambda i: [1 if x == i else 0 for x in range(len(tag2id))]
                Y = [[id2matrix(j) for j in i] for i in Y] #形如：[[1,0,0],[0,0,1],[0,1,0],[1,0,0]]
            y,y_p = extract_entity(d, batch) # y:实体1，实体2, y_p形如：[[1,0,0],[0,0,1],[0,1,0],[1,0,0]]
            # print(len(Y[0]))
            # print(len(y_p[0]))
            for j in range(len(d)):
                acc += my_acc(d[j], Y[j], y_p[j])
                if type(Y[j]) != str:
                    Y_d = decode(d[j], Y[j],one=True)
                else:
                    Y_d=Y[j]
                if tag!=None and Y_d!=y[j]:
                    error_f.write('%s\n'%(d[j]))
                    ss = np.argmax(Y[j], axis=1)
                    ss = '[%s]\t%s\n' % (', '.join(map(str, ss)),Y_d)
                    error_f.write(ss)
                    ss = np.argmax(y_p[j], axis=1)
                    ss = '[%s]\t%s\n' % (', '.join(map(str, ss)),y[j])
                    error_f.write(ss)
                f, p, r = myF1_P_R(Y_d, y[j])  # 求指标
                A += p
                F += f
        if tag != None:
            error_f.write("all_acc,f1,sig_acc\t%.4f\t%.4f\t%.4f\t"%(A / len(dev_data), F / len(dev_data),acc/len(dev_data)))
            error_f.close()
        return A / len(dev_data), F / len(dev_data),acc/len(dev_data)

    bast_epoch=0
    class Evaluate(Callback):
        def __init__(self, dev_data, model_path):
            self.ACC = []
            self.best = 0.
            self.passed = 0
            self.dev_data = dev_data
            self.model_path = model_path

        #调整学习率
        def on_batch_begin(self, batch, logs=None):
            """第一个epoch用来warmup，第二个epoch把学习率降到最低
            """
            if   used_wup:
                if self.passed < self.params['steps']:
                    lr = (self.passed + 1.) / self.params['steps'] * learning_rate
                    K.set_value(self.model.optimizer.lr, lr)
                    self.passed += 1
                elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
                    lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
                    lr += min_learning_rate
                    K.set_value(self.model.optimizer.lr, lr)
                    self.passed += 1

        def on_epoch_end(self, epoch, logs=None):

            all_acc,f1,sig_acc = evaluate(self.dev_data,batch=bsize )
            if acc_type=='f1':
                acc=f1
            elif acc_type=='all_acc':
                acc=all_acc
            elif acc_type=='sig_acc':
                acc=sig_acc
            self.ACC.append(acc)
            if acc > self.best:
                self.best = acc
                global bast_epoch
                bast_epoch=epoch
                print("save best model weights ...")
                model.save_weights(self.model_path)
            print('all_acc: %.4f, f1: %.4f, sig_acc: %.4f, best %s: %.4f\n' % (all_acc,f1,sig_acc,acc_type, self.best))


    def test(test_data, result_path, batch=1):
        F = open(result_path, 'w', encoding='utf-8')
        for idx in tqdm(range(0, len(test_data), batch)):
            d0 = [i[0] for i in test_data[idx:idx + batch]]
            d1 = [i[1] for i in test_data[idx:idx + batch]]
            y = extract_entity(d1, batch)[0]
            for i in range(len(d0)):
                s = u'%s,%s\n' % (d0[i], y[i])
                F.write(s)
            F.flush()
        F.close()


    def test_cv(test_data, batch=1):
        '''预测'''
        r = []
        for idx in tqdm(range(0, len(test_data), batch)):
            d0 = [i[0] for i in test_data[idx:idx + batch]]
            d1 = [i[1][:maxlen] for i in test_data[idx:idx + batch]]
            # d1=[i[:maxlen] for i in test_data[idx:idx + batch]]
            ml = max([len(i) for i in d1])+2
            x1x2 = [tokenizer.encode(i, max_len=ml) for i in d1]
            _x1 = np.array([i[0] for i in x1x2])
            _x2 = np.array([i[1] for i in x1x2])
            _p = model.predict([_x1, _x2])
            for i in range(len(d1)):
                r.append(_p[i])
        return r


    def test_cv_decode(test_data, result, result_path):
        '''获得cv的结果'''
        result_avg = np.mean(result, axis=0)
        F = open(result_path, 'w', encoding='utf-8')
        for idx, d in enumerate(test_data):
            # s = u'%s\n%s\n' % (d[1], decode(d[1], result_avg[idx]))
            ss=np.argmax(result_avg[idx],axis=1)
            ss= '[%s]' %', '.join(map(str,ss[1:-1]))
            s = u'%s\n%s\n' % (d[1], ss)
            F.write(s)
        F.close()

    # 返回实体及置信度
    def decode2(text_in, p_in):
        '''解码函数'''
        p = np.argmax(p_in, axis=-1)
        # _tokens = tokenizer.tokenize(text_in)
        _tokens = ' %s ' % text_in
        ei = ''
        eis=[]
        r = []
        rs=[]
        if tagkind == 'BIO':
            0/0
            #todo rs
            # for i, v in enumerate(p):
            #     if ei == '':
            #         if v == tag2id['B']:
            #             ei += _tokens[i]
            #     else:
            #         if v == tag2id['B']:
            #             r.append(ei)
            #             ei = _tokens[i]
            #         elif v == tag2id['I']:
            #             ei += _tokens[i]
            #         elif v == tag2id['O']:
            #             r.append(ei)
            #             ei = ''
        elif tagkind == 'BIOES':
            for i, v in enumerate(p):
                if ei == '':
                    if v == tag2id['B']:
                        ei = _tokens[i]
                        eis=[p_in[i][v]/sum(p_in[i])]
                    elif v == tag2id['S']:
                        r.append(_tokens[i])
                        rs.append([p_in[i][v] / sum(p_in[i])])
                else:
                    if v == tag2id['B']:
                        ei = _tokens[i]
                        eis = [p_in[i][v] / sum(p_in[i])]
                    elif v == tag2id['I']:
                        ei += _tokens[i]
                        eis.append( p_in[i][v] / sum(p_in[i]))
                    elif v == tag2id['E']:
                        ei += _tokens[i]
                        eis.append(p_in[i][v] / sum(p_in[i]))
                        r.append(ei)
                        rs.append(eis)
                        ei = ''
                        eis=[]
                    elif v == tag2id['O']:
                        # r.append(ei)
                        ei = ''
                        eis = []
                    elif v == tag2id['S']:
                        r.append(_tokens[i])
                        rs.append([p_in[i][v] / sum(p_in[i])])
                        ei = ''
                        eis = []
        r = [i for i in r if len(i) > 1]
        rs = [np.mean(i) for i in rs if len(i) > 1]
        r = list(set(r))
        if len(rs)==0:
          rs=0
        else:
          rs=np.max(rs)
        r.sort()
        return ';'.join(r),rs


    def test_cv_decode2(test_data, result, result_path):
        '''仅获取前topN个结果'''
        result_avg = np.mean(result, axis=0)
        r=[]
        for idx, d in enumerate(test_data):
            n,s=decode2(d[1], result_avg[idx])
            r.append((d[0],n,s ))
        if topN is not None:
            s_sort=[i[2] for i in r]
            s_sort.sort(reverse=True)
            print('topN',topN)
            ljz=s_sort[topN]
            print('ljz',ljz)
        F = open(result_path, 'w', encoding='utf-8')
        for idx, ri in enumerate(r):
            if topN is None or ri[2]>=ljz :
                s = u'%s,%s\n' % (ri[0], ri[1])
                F.write(s)
        F.close()

    # train

    train_ = train_data
    dev_ = dev_data
    if model_type=='DT':
        model = modify_bert_model_biMyGRU_mask_crf()
    elif model_type=='LSTM':
        model = modify_bert_model_bilstm_mask_crf()
    train_D = data_generator(train_)
    dev_D = data_generator(dev_)

    model_path = os.path.join(model_save_path, "modify_bert_biMyGRU_mask_crf_model".replace('MyGRU',model_type)+ ".weights")
    if not used_mask:
        model_path=model_path.replace('_mask','')

    ss=','.join(map(str, args_i))+'_'
    model_path=model_path.replace('modify_',ss+'modify_')

    if not os.path.exists(model_path):
        tbCallBack = TensorBoard(log_dir=os.path.join(model_save_path, 'logs_'),  # log 目录
                                 histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                                 batch_size=bsize,  # 用多大量的数据计算直方图
                                 write_graph=True,  # 是否存储网络结构图
                                 write_grads=False,  # 是否可视化梯度直方图
                                 write_images=True,  # 是否可视化参数
                                 embeddings_freq=0,
                                 embeddings_layer_names=None,
                                 embeddings_metadata=None)

        evaluator = Evaluate(dev_, model_path)
        H = model.fit_generator(train_D.__iter__(),
                                steps_per_epoch=len(train_D),
                                epochs=15,
                                callbacks=[evaluator, tbCallBack],
                                validation_data=dev_D.__iter__(),
                                validation_steps=len(dev_D),
                                verbose=2
                                )
        # f = open(model_path.replace('.weights', 'history.pkl'), 'wb')
        # pickle.dump(H, f, 4)
        # f.close()
        del tbCallBack,evaluator,H
        gc.collect()
    else:
        del model
        gc.collect()
        del sess
        gc.collect()
        K.clear_session()
        continue
    print("load best model weights ...")
    model.load_weights(model_path)

    print('val')
    score=evaluate(dev_, batch=1,tag='dev')
    print("valid evluation:", 'all_acc,f1,sig_acc',score)

    print('test')
    result_path = os.path.join(model_save_path, "test_result_k"+ ".txt")
    resulti_path=os.path.join(model_save_path, "test_result_k" + ".npy")
    #if False and os.path.exists(resulti_path):
    #    resulti = np.load(resulti_path,allow_pickle=True)
    #else:
    #    resulti = test_cv(test_data, batch=1)
    #    resulti=np.array(resulti)
        # np.save(resulti_path, resulti)  # 保存为.npy格式

    # test_cv_decode(test_data, [resulti], result_path)
    test_=[(test_data[i][1],test_label[i]) for i in range(len(test_data))]
    t_score=evaluate(test_, batch=1,tag='test')
    print("test evluation:", 'all_acc,f1,sig_acc',t_score)
    print('%.4f %.4f %.4f'%t_score)

    # %load_ext tensorboard  #使用tensorboard 扩展
    # %tensorboard --logdir logs  #定位tensorboard读取的文件目录
    ss0='maxlen,learning_rate,min_learning_rate,bsize,need_one,lstm_unit,used_mask,used_mask0,used_wup,bert_kind,tagkind,model_type,acc_type'.split(',')
    [print(ss0[i],args_i[i]) for i in range(len(ss0))]
    # print('model_type',model_type)
    # print('bert_kind',bert_kind,'\nbsize',bsize)
    # print('maxlen', maxlen)
    # print('tagkind',tagkind)
    # print('acc_type',acc_type)
    # print('lstm_unit',lstm_unit,'\nused_mask',used_mask,'\n'+'used_wup',used_wup)
    # print('bast_epoch',bast_epoch)
    # ss=str(bast_epoch)
    # if used_wup:
    #     ss+='wup-'
    # else:
    #     ss+='nowup-'
    # ss+=tagkind+'-'
    # if 'wwm' in bert_kind:
    #     ss+='wwm-'
    # ss+=str(lstm_unit)+'-'+str(maxlen)
    # ss+='(%f,%f,%f)'%t_score
    # print(ss)
    ss=','.join(map(str, args_i))+'_'+str(bast_epoch)+'(%f,%f,%f)'%t_score
    print(ss)
    del model,bast_epoch
    gc.collect()
    del sess
    gc.collect()
    K.clear_session()
