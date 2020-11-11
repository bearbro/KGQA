import sys
sys.path.insert(0,'/home/aistudio/work/MyExperiment/classification')
sys.path.insert(0, '/home/hbxiong/QA2/classification')
from keras_bert import load_trained_model_from_checkpoint
import keras
from some_function_maxbert import transfer_data_train,train_read_data,transfer_data_test
import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0"#设置使用0号GPU
import json
import numpy as np
import keras.backend as k
import tensorflow as tf
from pprint import pprint
import os
import time
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

model_tag=time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
print(model_tag)
class Config:
    # data_path对于交叉路径，固定在训练文件中
    data_dir=r'./data/old_data'
    train_data_path=os.path.join(data_dir,'siamese_train_data_sample.json')
    valid_data_path=os.path.join(data_dir,'siamese_valid_data_sample.json')

    linking_data_path = '../result_new_linking-no_n_59,r_0.8321,line_right_recall_0.9230,avg_n_2.6919.json'

    # bert_path
    # bert_path = '../../bert/bert_wwm_ext'  # 百度
    bert_path = r'C:\Users\bear\OneDrive\ccks2020-onedrive\ccks2020\bert\tf-bert_wwm_ext'  # room
    # bert_path = r'../../../ccks/bert/tf-bert_wwm_ext'  # colab
    # bert_path = '/home/hbxiong/ccks/bert/tf-bert_wwm_ext'  # lab
    # bert_path
    bert_config_path = os.path.join(bert_path, 'bert_config.json')
    bert_ckpt_path = os.path.join(bert_path, 'bert_model.ckpt')
    bert_vocab_path = os.path.join(bert_path, 'vocab.txt')

    parameter = 0.7
    batch_size = 64
    epoches = 100
    learning_rate = 1e-5  # 2e-5
    neg_sample_number = 10
    max_length = 100  # neg3:64;100
    monitor = ['val_loss', 'val_accuracy'][0]

    model_tag = ','.join(map(str, [max_length, learning_rate, batch_size, monitor,parameter,neg_sample_number])) + '-' + model_tag


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

if __name__=='__main__':

    training=True
    predicting=False
    if training:
        print('load data begin')

        train_sentence, train_pos, train_neg = train_read_data(config.train_data_path)
        train_pos_indices, train_pos_segments, train_neg_indices, train_neg_segments, train_labels = transfer_data_train(
            train_sentence, train_pos, train_neg,config.max_length,config.bert_vocab_path)
        train_labels = np.array(train_labels)

        valid_sentence, valid_pos, valid_neg = train_read_data(config.valid_data_path)
        valid_pos_indices, valid_pos_segments, valid_neg_indices, valid_neg_segments, valid_labels = transfer_data_train(
            valid_sentence, valid_pos, valid_neg,config.max_length,config.bert_vocab_path)
        valid_labels = np.array(valid_labels)

        print('load data over')

        checkpoint = keras.callbacks.ModelCheckpoint(config.similarity_ckpt_path,
                                                     monitor=config.monitor,
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='min' if 'loss' in config.monitor else 'max',
                                                     save_weights_only=True,
                                                     period=1)

        earlystop = keras.callbacks.EarlyStopping(monitor=config.monitor,
                                                  patience=config.neg_sample_number,
                                                  verbose=0,
                                                  mode='min' if 'loss' in config.monitor else 'max',)
        model=triplet_model()
        model.fit([train_pos_indices, train_pos_segments, train_neg_indices, train_neg_segments],
                  train_labels,
                  batch_size=config.batch_size,
                  epochs=config.epoches,
                  callbacks=[checkpoint, earlystop],
                  validation_data=(
                  [valid_pos_indices, valid_pos_segments, valid_neg_indices, valid_neg_segments], valid_labels),
                  verbose=1)

    if predicting:
        '''
        对正确的路径进行打分
        '''

        model=basic_network()
        model.load_weights(config.similarity_ckpt_path)

        #读取句子和答案路径
        with open('./data/old_data/clear_test.json','r',encoding='utf-8') as test_right:
            test_sent=[]
            test_sample=[]
            data=json.load(test_right)
            for item in data:
                test_sent.append(item['sentence'])
                if item['sqrql'].find('?y')!=-1:#包含此字符串?y
                    test_sample.append(item['sqrql'].replace('?y','?z').replace('?x','?y').replace('?z','?x'))
                else:
                    test_sample.append(item['sqrql'])
                print(test_sent[-1])

        x_indices,x_segments=transfer_data_test(test_sent,test_sample,config.max_length,config.bert_vocab_path)
        print('预测begin-----')
        writer=open(config.true_answer_path,'w',encoding='utf-8')
        result=model.predict([x_indices,x_segments])
        result=result.ravel()
        true_result=[]
        for i in range(len(result)):
            dict={}
            dict['id']=str(i)
            dict['score']=str(result[i])
            dict['sentence']=test_sent[i]
            dict['path']=test_sample[i]
            true_result.append(dict)
        json.dump(true_result,writer,ensure_ascii=False)
        print('predict test right sample over')


