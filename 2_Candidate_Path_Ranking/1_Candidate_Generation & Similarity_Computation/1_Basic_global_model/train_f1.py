import sys
# sys.path.insert(0,'/Users/mingrongtang/PycharmProjects/TensorFlow/MyExperiment/rel_and_path_similarity')
sys.path.insert(0,'/home/aistudio/work/MyExperiment/rel_and_path_similarity')
from keras.metrics import binary_accuracy
from tensorflow.contrib.metrics import accuracy

sys.path.insert(0, '/Users/brobear/Downloads/MyExperiment/rel_and_path_similarity')

from keras_bert import load_trained_model_from_checkpoint
import keras
from some_function_maxbert import transfer_data
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置使用0号GPU
import json
import numpy as np
import keras.backend as k
import keras

from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
from keras_bert import load_trained_model_from_checkpoint

from pprint import pprint
import os
import time
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model_tag=time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
print(model_tag)
class Config:
    # data_path对于交叉路径，固定在训练文件中
    data_dir = r'./data'
    train_data_path = os.path.join(data_dir, 'path_data/train_data_sample.json')
    valid_data_path = os.path.join(data_dir, 'path_data/valid_data_sample.json')
    trainvalid_data_path = os.path.join(data_dir, 'path_data/trainvalid_data_sample.json')

    linking_data_path = '../result_new_linking-no_n_59,r_0.8321,line_right_recall_0.9230,avg_n_2.6919.json'

    # bert_path
    # bert_path = '../../bert/bert_wwm_ext'  # 百度
    # bert_path = r'C:\Users\bear\OneDrive\ccks2020-onedrive\ccks2020\bert\tf-bert_wwm_ext'  # room
    # bert_path = r'../../../ccks/bert/tf-bert_wwm_ext'  # colab
    bert_path = '/home/hbxiong/ccks/bert/tf-bert_wwm_ext'  # lab
    bert_config_path = os.path.join(bert_path, 'bert_config.json')
    bert_ckpt_path = os.path.join(bert_path, 'bert_model.ckpt')
    bert_vocab_path = os.path.join(bert_path, 'vocab.txt')

    batch_size = 64
    epoches = 100
    learning_rate = 1e-5  # 2e-5
    neg_sample_number = 5
    max_length = 100  # neg3:64;100
    monitor = ['val_monitor_f1', 'val_my_accuracy'][0]

    model_tag = ','.join(map(str, [max_length, learning_rate, batch_size, monitor])) + '-' + model_tag


    similarity_ckpt_path = './ckpt/ckpt_similarity_bert_wwm_ext_f1_%s.hdf5'%model_tag  # 模型训练后，模型参数存储路径




config = Config()
for i in ['./ckpt']:
    if not os.path.exists(i):
        os.mkdir(i)

pprint(vars(Config))

def basic_network():
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path,
                                                    config.bert_ckpt_path,
                                                    seq_len=config.max_length,
                                                    training=False,
                                                    trainable=True)
    # 选择性某些层进行训练
    # bert_model.summary()
    # for l in bert_model.layers:
    #     # print(l)
    #     l.trainable = True
    x1 = keras.layers.Input(shape=(config.max_length,))
    x2 = keras.layers.Input(shape=(config.max_length,))
    bert_out = bert_model([x1, x2])  # 输出维度为(batch_size,max_length,768)
    # dense=bert_model.get_layer('NSP-Dense')
    bert_out = keras.layers.Lambda(lambda bert_out: bert_out[:, 0])(bert_out)
    # bert_out = keras.layers.Dropout(0.5)(bert_out)
    outputs = keras.layers.Dense(1, activation='sigmoid')(bert_out)

    model = keras.models.Model([x1, x2], outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(config.learning_rate),
        loss=my_loss,
        metrics=[my_accuracy, monitor_f1]
    )
    model.summary()
    return model


def my_accuracy(y_true, y_pred):
    '''

    :param y_true: ？,2
    :param y_pred: ?,
    :return: 1
    '''
    # y_true=tf.to_int32(y_true)
    y_true = k.expand_dims(y_true[:, 0],axis=-1)
    return binary_accuracy(y_true, y_pred)


def my_loss(y_true, y_pred):
    # y_true = tf.to_int32(y_true)
    y_true = k.expand_dims(y_true[:, 0])
    return k.binary_crossentropy(y_true, y_pred)


def monitor_f1(y_true, y_pred):
    '''
    统计预测为1或真实为1的样本的f1的平均值，弊端batch-wise的，解决：https://www.zhihu.com/question/53294625/answer/362401024
    :param y_true: ?,2
    :param y_pred: ?,
    :return: 1
    '''
    f1 = k.expand_dims(y_true[:, 1],axis=-1)
    y_true = k.expand_dims(y_true[:, 0],axis=-1)
    # 0.5 划分0，1
    one = tf.ones_like(y_pred)
    zero = tf.zeros_like(y_pred)
    # y_pred = tf.where(y_pred < 0.5, x=zero, y=one)
    # y= tf.where(y_pred == 1, x=one, y=y_true) #y_true 或 y_pred 为1的地方
    # 合并上面两个
    y= tf.where(y_pred > 0.5, x=one, y=y_true)
    return tf.div(k.sum(tf.multiply(y,f1)),k.sum(y))

import numpy as np

if __name__ == '__main__':
    training = True
    predicting = False
    if training:
        print('load data begin')
        with open(config.train_data_path, 'r', encoding='utf-8') as train_reader:
            train_data = json.load(train_reader)
            train_x_sent = train_data['x_sent']
            train_x_sample = train_data['x_sample']
            train_y = train_data['y']
            train_x_indices, train_x_segments = transfer_data(train_x_sent, train_x_sample, config.max_length,config.bert_vocab_path)
        train_y = [[i, 0] for i in train_y]
        train_y=np.array(train_y,dtype=int)
        with open('data/valid_f1_path.json', 'r', encoding='utf-8') as reader:
            data = json.load(reader)
            sent_split_list = data['sent_split_list']
            f1_list = data['f1_list']

        with open(config.valid_data_path, 'r', encoding='utf-8') as valid_reader:
            valid_data = json.load(valid_reader)
            valid_x_sent = valid_data['x_sent']
            valid_x_sample = valid_data['x_sample']
            valid_y = valid_data['y']
            valid_x_indices, valid_x_segments = transfer_data(valid_x_sent, valid_x_sample, config.max_length,config.bert_vocab_path)
        valid_y = [[v, f1_list[i]] for i, v in enumerate(valid_y)]
        valid_y = np.array(valid_y)
        print('load data over')

        checkpoint = keras.callbacks.ModelCheckpoint(config.similarity_ckpt_path,
                                                     monitor=config.monitor,
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max',
                                                     save_weights_only=True,
                                                     period=1)

        earlystop = keras.callbacks.EarlyStopping(monitor=config.monitor,
                                                  patience=config.neg_sample_number,
                                                  verbose=0,
                                                  mode='max')

        model = basic_network()
        model.fit([train_x_indices, train_x_segments],
                  train_y,
                  epochs=config.epoches,
                  callbacks=[checkpoint, earlystop],
                  validation_data=([valid_x_indices, valid_x_segments], valid_y),
                  batch_size=config.batch_size,
                  verbose=1,
                  )

        print('train over')
        # print(model.evaluate([valid_x_indices, valid_x_segments], valid_y))
        # print('val over')
    if predicting:
        '''
        对正确的路径进行打分
        '''

        model = basic_network()
        model.load_weights(config.similarity_ckpt_path)

        # 读取句子和答案路径
        with open('./data/old_data/clear_test.json', 'r', encoding='utf-8') as test_right:
            test_sent = []
            test_sample = []
            data = json.load(test_right)
            for item in data:
                test_sent.append(item['sentence'])
                if item['sqrql'].find('?y') != -1:  # 包含此字符串?y
                    test_sample.append(item['sqrql'].replace('?y', '?z').replace('?x', '?y').replace('?z', '?x'))
                else:
                    test_sample.append(item['sqrql'])
                print(test_sent[-1])

        x_indices, x_segments = transfer_data(test_sent, test_sample, config.max_length,config.bert_vocab_path)
        print('预测begin-----')
        writer = open(config.true_answer_path, 'w', encoding='utf-8')
        result = model.predict([x_indices, x_segments])
        result = result.ravel()
        true_result = []
        for i in range(len(result)):
            dict = {}
            dict['id'] = str(i)
            dict['score'] = str(result[i])
            dict['sentence'] = test_sent[i]
            dict['path'] = test_sample[i]
            true_result.append(dict)
        json.dump(true_result, writer, ensure_ascii=False)
        print('predict test right sample over')
