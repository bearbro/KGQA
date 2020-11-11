# ! -*- coding: utf-8 -*-

import codecs
import gc
import json
import shutil

import tensorflow as tf
from keras.backend import binary_crossentropy
from keras.callbacks import *

from keras.layers import *
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert.backend import keras
from sklearn.metrics import f1_score
from tqdm import tqdm


def all_args(args):
    r = []

    def add_one(ri, args):
        ri = ri.copy()
        ri[-1] += 1
        x = len(args) - 1
        while ri[x] >= len(args[x]):
            ri[x] = 0
            x -= 1
            ri[x] += 1
            if x == -1:
                return None
        return ri

    ri = [0] * len(args)
    r.append(ri)
    ri = add_one(ri, args)
    while ri != None:
        r.append(ri)
        ri = add_one(ri, args)

    rr = [[args[idx][i] for idx, i in enumerate(ri)] for ri in r]
    return rr


test_Xs = [
    '../data/60,5e-05,1e-05,16,True,64,True,True,False,tf-bert_wwm_ext,BIO,DT,sig_acc_modify_bert_biDT_mask_crf_model(0.8444,0.8408,0.9510)_ner(0.8494,0.8444,0.8392)entity_linkingno_n55R0.8522line_right_r0.9282avg_n6.6319.json']

#max_epoch,maxlen, learning_rate, bsize, bert_kind ,bert_layer, TFK, train_shuffle
args = [ [10],[75], [5e-6], [16, 32],['chinese_L-12_H-768_A-12'], [4], [5], [True]]
allargs=all_args(args)
for asidx,args_i in enumerate(allargs):
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    max_epoch,maxlen, learning_rate, bsize, bert_kind ,bert_layer, TFK,  train_shuffle  = args_i
    tag = ','.join(list(map(str, args_i)))
    print('*'*30)
    print('%d / %d'%(asidx,len(allargs)))
    print(tag)

    # path_root='/Users/brobear/OneDrive/ccks2020-onedrive/ccks2020/bert/'#mac
    # path_root = r'C:\Users\bear\OneDrive\ccks2020-onedrive\ccks2020\bert'  # room
    # path_root='/home/hbxiong/ccks/bert'#lab
    path_root = r'../../ccks/bert/'  # colab
    model_ckpt_path = "./ckpt_reduce_avg"
    model_save_path = "./ckpt_reduce_avg_result"
    model_save_path_finall = "./ckpt_reduce_avg_result_finall"
    for i in [model_ckpt_path, model_save_path,model_save_path_finall]:
        if not os.path.exists(i):
            os.mkdir(i)

    train_X = r'../data/bio_ner_train_ner(0.9998,0.9880,0.9880).json'
    train_Y = r'../data/clear_train.json'
    dev_X = r'../data/bio_ner_valid_ner(1.0000,0.9948,0.9948).json'
    dev_Y = r'../data/clear_valid.json'
    test_Y = r'../data/clear_test.json'
    mention_dict = dict()


    def make_data(train_X, train_Y, k=3):
        with open(train_Y, 'r', encoding='utf-8') as data_file:
            true_data = json.load(data_file)

        with open(train_X, 'r', encoding='utf-8') as data_file:
            pred_data = json.load(data_file)
        # sentence,mention,entity,label
        error_n = 0
        true_n = 0
        data = []
        for i in range(len(true_data)):
            assert true_data[i]['sentence'].strip('?').strip('？').strip('?') == pred_data[i]['sentence'].strip(
                '?').strip(
                '？').strip('?')
            if pred_data[i]['sentence'] in mention_dict:
                if mention_dict[pred_data[i]['sentence']] != pred_data[i]['mention']:
                    ax = 1
                assert mention_dict[pred_data[i]['sentence']] == pred_data[i]['mention'] or '大连' in pred_data[i][
                    'sentence'] or '清明' in pred_data[i]['sentence']
            mention_dict[pred_data[i]['sentence']] = pred_data[i]['mention']
            if 'true_entity_property' in true_data[i]:
                true_entity = true_data[i]['true_entity_property']
            else:
                true_entity = true_data[i]['entities']
            pred_entity = pred_data[i]['pred_entity']
            # 正
            true_entity = list(set(pred_entity)&set(true_entity))
            # 负
            error_entity = list(set(pred_entity) - set(true_entity))
            if k != -1:
                error_entity = error_entity[:max(1, len(true_entity) * k)]  # 正：负=1：k
            id = pred_data[i]['id']
            sentence = pred_data[i]['sentence']
            for entity in true_entity:
                # mention=true_data[i]['mention']
                data.append([id, sentence, entity, 1])
                true_n += 1
            for entity in error_entity:
                # mention=true_data[i]['mention']
                data.append([id, sentence, entity, 0])
                error_n += 1
        print('data:', len(data), 'T:F= %d / %d = %.3f' % (true_n, error_n, true_n / error_n))
        return data


    # todo 获得训练数据
    train_ = make_data(train_X, train_Y, TFK)
    dev_ = make_data(dev_X, dev_Y, -1)
    # test_ = make_data(test_X, test_Y, -1)
    # train_=train_data[:-len(train_data)//10]
    # dev_=train_data[-len(train_data)//10:]
    # test_=dev_
    print('训练集：', len(train_))
    print('验证集：', len(dev_))
    # print('测试集合：', len(test_))

    config_path = os.path.join(path_root, bert_kind, 'bert_config.json')
    checkpoint_path = os.path.join(path_root, bert_kind, 'bert_model.ckpt')
    dict_path = os.path.join(path_root, bert_kind, 'vocab.txt')
    token_dict = {}

    with codecs.open(dict_path, 'r', 'utf-8') as reader:
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


    def seq_padding(X, padding=0, wd=1):
        L = [len(x) for x in X]
        ML = max(L)  # maxlen
        if wd == 1:
            return np.array([
                np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
            ])
        else:
            0 / 0
        # else:
        #     padding_wd = [padding] * len(X[0][0])
        #     padding_wd[tag2id['O']] = 1
        #     return np.array([
        #         np.concatenate([x, [padding_wd] * (ML - len(x))]) if len(x) < ML else x for x in X
        #     ])


    class data_generator:
        def __init__(self, data, batch_size=bsize, shuffle=True):
            self.data = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1

        def __len__(self):
            return self.steps

        def __iter__(self):
            while True:
                idxs = list(range(len(self.data)))
                if self.shuffle:
                    np.random.shuffle(idxs)
                X1, X2, P = [], [], []
                for i in idxs:
                    d = self.data[i]
                    id = d[0]
                    e = d[2]
                    text = d[1][:maxlen - 3 - len(e)]
                    p = d[3]

                    x1, x2 = tokenizer.encode(text, e)
                    X1.append(x1)
                    X2.append(x2)
                    P.append(p)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        # batch 内打乱
                        if not self.shuffle:
                            idxss = list(range(len(P)))
                            np.random.shuffle(idxss)
                            a = np.array([X1[i] for i in idxss])
                            b = np.array([X2[i] for i in idxss])
                            c = np.array([P[i] for i in idxss])
                            X1, X2, P = a, b, c
                        yield [X1, X2], P
                        X1, X2, P = [], [], []


    # 定义模型

    def metrics_f1(y_true, y_pred):
        y_pred = tf.where(y_pred < 0.5, x=tf.zeros_like(y_pred), y=tf.ones_like(y_pred))
        equal_num = tf.count_nonzero(tf.multiply(y_true, y_pred))
        true_sum = tf.count_nonzero(y_true)
        pred_sum = tf.count_nonzero(y_pred)
        equal_num = tf.cast(equal_num, dtype=tf.float32)
        precision = equal_num / (tf.cast(pred_sum, dtype=tf.float32) + 1e5)
        recall = equal_num / (tf.cast(true_sum, dtype=tf.float32) + 1e5)
        f1 = (2 * precision * recall) / (precision + recall + 1e5)
        return f1


    def link_f1(y_true, y_pred):
        threshold_valud = 0.5
        y_true = np.reshape(y_true, (-1))
        y_pred = [1 if p > threshold_valud else 0 for p in np.reshape(y_pred, (-1))]
        equal_num = np.sum([1 for t, p in zip(y_true, y_pred) if t == p and t == 1 and p == 1])
        true_sum = np.sum(y_true)
        pred_sum = np.sum(y_pred)
        precision = equal_num / pred_sum
        recall = equal_num / true_sum
        f1 = (2 * precision * recall) / (precision + recall)
        print('all_num', len(y_true))
        print('equal_num:', equal_num)
        print('true_sum:', true_sum)
        print('pred_sum:', pred_sum)
        print('precision:', precision)
        print('recall:', recall)
        print('f1:', f1)

        return precision, recall, f1


    def modify_bert_model():
        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, output_layer_num=1)

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))  # 待识别句子输入
        x2_in = Input(shape=(None,))  # 待识别句子输入

        x1, x2 = x1_in, x2_in

        outputs = bert_model([x1, x2])  # [batch,maxL,768]
        outputs = keras.layers.Lambda(lambda bert_out: bert_out[:, 0], name='bert_cls')(outputs)  # [batch,768]
        outputs = Dense(units=128, activation='relu')(outputs)
        outputs = Dropout(0.15)(outputs)
        outputs = Dense(units=1, activation='sigmoid')(outputs)

        model = keras.models.Model([x1_in, x2_in], outputs, name='basic_model')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=binary_crossentropy,
                      metrics=[metrics_f1, 'acc'])
        model.summary()

        return model


    def evaluate_entity_linking(true_entity_path, pred_entity_path):
        '''
        输入正确的实体和预测的实体，以一个句子为单位，求所有真实实体和预测的top k个实体的交集，算recall
        :param true_entity_path:
        :param pred_entity_path:
        :return:
        '''
        true_data = []
        with open(true_entity_path, 'r', encoding='utf-8') as data_file:
            true_data = json.load(data_file)

        pred_data = []
        with open(pred_entity_path, 'r', encoding='utf-8') as data_file:
            pred_data = json.load(data_file)
        true_data = true_data[pred_data[0]['id']:]
        all_right = 0
        all_true = 0
        all_pred = 0
        all_line_right = 0
        no_right_link = []
        for i in range(len(true_data)):
            if true_data[i]['sentence'] != pred_data[i]['sentence']:
                print(true_data[i]['sentence'], pred_data[i]['sentence'])
            assert true_data[i]['sentence'] == pred_data[i]['sentence']
            true_entity = true_data[i]['true_entity_property']
            pred_entity = pred_data[i]['pred_entity']

            # 转换为集合才可以计算交并补
            set_true_entity = set(true_entity)
            set_pred_entity = set(pred_entity)

            # print(true_data[i],'----',i)
            # print('ture_entity---',set_true_entity)
            # print('mention---',mention)
            # print('pred_entity---',set_pred_entity)
            right = len(set_true_entity & set_pred_entity)
            if (right) > 0:
                all_line_right = all_line_right + 1
            else:
                pred_data[i]['true_entity'] = true_entity
                pred_data[i]['id'] = i
                no_right_link.append(pred_data[i])
            # print(i,'---true---',len(true_entity),'right---',right)
            all_right = all_right + right
            all_true = all_true + len(set_true_entity)
            all_pred = all_pred + len(set_pred_entity)

        # print('no_right_number----',len(no_right_link))
        # if no_right_linking_file==None:
        #     with open('./no_right_linking.json','w',encoding='utf-8') as writer:
        #         json.dump(no_right_link,writer,ensure_ascii=False)
        # else:
        #     with open(no_right_linking_file,'w',encoding='utf-8') as writer:
        #         json.dump(no_right_link,writer,ensure_ascii=False)
        # print('recall----',all_right/all_true)
        # print('line_right_recall----',all_line_right/766)
        # print('average pred entity---',all_pred/766)

        return len(no_right_link), all_right / all_true, all_line_right / len(true_data), all_pred / len(true_data)


    def extract_entity(text_in, a_in):
        text_in = text_in[:maxlen - len(a_in) - 3]
        # 构造位置id和字id
        token_ids, segment_ids = tokenizer.encode(first=text_in, second=a_in)
        p = train_model.predict([[token_ids], [segment_ids]])[0]

        return p


    def evaluate(dev_, use_id=False):
        r = []
        for d in tqdm(iter(dev_)):
            p = extract_entity(d[1], d[2])
            if p > 0.5:
                r.append(1)
            else:
                r.append(0)
        if not use_id:
            return f1_score([d[3] for d in dev_], r)
        else:
            pass
            return


    def test(test_, result_path, yz=0.5):
        r = []
        for d in tqdm(iter(test_)):
            p = extract_entity(d[1], d[2])
            r.append((d[0], d[1], d[2],p))
        rr=[1 if i[3]>yz else 0 for i in r]
        # 生成json
        result = []
        prid = -1
        a_sentence = dict()
        for id, sentence, pred_entity in r:
            if prid == id:
                if pred_entity != None:
                    a_sentence['pred_entity'].append(pred_entity)
            else:
                if prid != -1:
                    result.append(a_sentence)
                prid = id
                a_sentence = dict()
                a_sentence['id'] = id
                a_sentence['sentence'] = sentence
                a_sentence['pred_entity'] = []
                if pred_entity != None:
                    a_sentence['pred_entity'].append(pred_entity)
                a_sentence['mention'] = mention_dict[sentence]

        if a_sentence != dict():
            result.append(a_sentence)

        with open(result_path, 'w', encoding='utf-8') as write_file:
            json.dump(result, write_file, ensure_ascii=False)

        if len(test_[0]) == 4:
            # return sklearn.metrics.recall_score([d[3] for d in test_], rr)
            return f1_score([d[3] for d in test_],rr)

    def test_s(test_, result_paths, yzs):
        r = []
        for d in tqdm(iter(test_)):
            p = extract_entity(d[1], d[2])
            r.append((d[0], d[1], d[2],p))
        def help(a_sentence,yz):
            pe=a_sentence['pred_entity']
            if yz<1:
                a_sentence['pred_entity']=[i[0] for i in pe if i[1]>=yz]
            else:
                pe_s=[i[1] for i in pe]
                pe_s.sort(reverse=True)
                yz=pe_s[min(yz-1,len(pe_s)-1)]
                a_sentence['pred_entity'] = [i[0] for i in pe if i[1] >= yz]
            return  a_sentence

        f1=[]
        for idx,result_path in enumerate(result_paths):
            yz=yzs[idx]

            # 生成json
            result = []
            prid = -1
            a_sentence = dict()
            for ridx,(id, sentence, pred_entity,p) in enumerate(r):
                if prid == id:
                    a_sentence['pred_entity'].append((pred_entity,p))
                else:
                    if prid != -1:
                        result.append(help(a_sentence,yz))
                    prid = id
                    a_sentence = dict()
                    a_sentence['id'] = id
                    a_sentence['sentence'] = sentence
                    a_sentence['pred_entity'] = []
                    a_sentence['pred_entity'].append((pred_entity,p))
                    a_sentence['mention'] = mention_dict[sentence]

            if a_sentence != dict():
                result.append(help(a_sentence,yz))

            with open(result_path, 'w', encoding='utf-8') as write_file:
                json.dump(result, write_file, ensure_ascii=False)

            if len(test_[0]) == 4:
                pass
                # get rr
                #  sklearn.metrics.recall_score([d[3] for d in test_], rr)
                # f1.append(f1_score([d[3] for d in test_],rr))

        return f1
    def step_decay(epoch, initial_lrate=0.00001):
        if epoch < 3:
            lr = 1e-6
        else:
            lr = 1e-7
        return lr


    class Evaluate(Callback):
        def __init__(self, validation_data, filepath, stop_patience=5, used_recall=False,filepath2=None):
            self.dev_D = validation_data
            self.F1 = []
            self.R = []
            # self.val = dev_D
            self.best = 0.
            self.best_r = 0.
            self.f1_raise = 1
            self.used_recall = used_recall
            self.wait = 0
            self.stop_patience = stop_patience  # 早停忍耐次数
            self.filepath = filepath  # 模型参数保存路径
            self.filepath2=filepath2

        def on_epoch_end(self, epoch, logs=None):
            print('Evaluate:')
            precision, recall, f1, = self.evaluate()
            self.F1.append(f1)
            if f1 > self.best:
                self.best = f1
                self.model.save_weights(self.filepath)
            if not self.used_recall:
                print(' precision: %.6f, recall: %.6f,f1: %.6f, best f1: %.6f\n' % (
                    float(precision), float(recall), float(f1),  float(self.best)))
            else:
                self.R.append(recall)
                if recall > self.best_r:
                    self.best_r = recall
                    self.model.save_weights(self.filepath2)
                    print(' precision: %.6f, recall: %.6f,f1: %.6f, best f1: %.6f, best recall: %.6f\n' % (
                        float(precision), float(recall), float(f1), float(self.best),float(self.best_r)))

            # logging.debug(str(precision) + ' ' + str(recall) + ' ' + str(f1))

        def evaluate(self):
            # 分批
            pred = []
            label = []
            for i in range(len(self.dev_D)):
                [X1, X2], P = next(self.dev_D.__iter__())
                pred.extend(self.model.predict([X1, X2]))
                label.extend(P)
            return link_f1(label, pred)

        # def stop_train(self, F1, best_f1, stop_patience):
        #     stop = True
        #     for f in F1[-stop_patience:]:
        #         if f >= best_f1:
        #             stop = False
        #     if stop == True:
        #         print('EarlyStopping!!!')
        #         self.model.stop_training = True




    train_D = data_generator(train_,shuffle=train_shuffle)
    dev_D = data_generator(dev_)

    train_model = modify_bert_model()
    model_path = os.path.join(model_ckpt_path, "%s-modify_basic_model.weights"%tag)
    filepath_f1 =model_path.replace('.weights','-f1.weights')
    filepath_recall=model_path.replace('.weights','-recall.weights')
    if not os.path.exists(model_path):

        evaluate_f1_recall = Evaluate(dev_D, filepath_f1,used_recall=True,filepath2=filepath_recall)
        checkpoint_loss = ModelCheckpoint(model_path, monitor='val_loss', verbose=2, save_weights_only=True,
                                     save_best_only=True, mode='min')

        callbacks = [checkpoint_loss] #保存3种模型
        train_model.fit_generator(train_D.__iter__(),
                                  steps_per_epoch=len(train_D),
                                  epochs=max_epoch,
                                  callbacks=callbacks,
                                  validation_data=dev_D.__iter__(),
                                  validation_steps=len(dev_D))
        del evaluate_f1_recall, checkpoint_loss
        gc.collect()


    ckpt_kind_dict={'loss':model_path,'f1':filepath_f1,'recall':filepath_recall}
    for acc_type in ['loss']:
        train_model.load_weights(ckpt_kind_dict[acc_type])
        print('val')
        v_f1=evaluate(dev_)
        print('val:%.4f'%v_f1)

        for test_X in test_Xs:
            yzs=[0.3, 0.5, 2,3]
            tag2=tag+','+test_X[-47: -5]+','+acc_type
            test_ = make_data(test_X, test_Y, -1)
            # print('测试集合：', len(test_))

            result_paths = [os.path.join(model_save_path, "result_new_linking-%s-%s.json"%(tag2,str(i))) for i in yzs]
            if  not os.path.exists(result_paths[0]):
                print('test')
                test_f1 = test_s(test_, result_paths,yzs=yzs)
            for idx,result_path in enumerate(result_paths):
                # print('test_f1', test_f1[idx])
                no_n, r, line_right_recall, avg_n = evaluate_entity_linking(r'data/clear_test.json', result_path)
                result_s='no_n:%d\tr:%.4f\tline_right_recall:%.4f\tavg_n:%.4f' % (no_n, r, line_right_recall, avg_n)
                print(result_s)
                shutil.copyfile(result_path, os.path.join(model_save_path_finall, "result_new_linking-%s-%s-%s.json"%(tag2,str(yzs[idx]),result_s.replace('\t',','))))
            del test_

    del train_model,train_D,dev_D,train_,dev_,mention_dict
    gc.collect()
    del sess
    gc.collect()
    K.clear_session()
