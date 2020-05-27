# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import graph_util

from genIDCard  import *
import numpy as np
import time 


import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


OUTPUT_SHAPE = (32,256)

num_epochs = 150

num_hidden = 64
num_layers = 2

obj = gen_id_card()

num_classes = obj.len + 1 + 1  # 10位数字 + blank + ctc blank

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 50
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
MOMENTUM = 0.9

DIGITS='0123456789'
BATCHES = 10
BATCH_SIZE = 64
TRAIN_SIZE = BATCHES * BATCH_SIZE

def decode_sparse_tensor(sparse_tensor):
    #print("sparse_tensor = ", sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    #print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        #print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        #print(result)
    return result
    
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = DIGITS[spars_tensor[1][m]]
        decoded.append(str)
    # Replacing blank label to none
    #str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    #str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print("ffffffff", str_decoded)
    return decoded

def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0
    
    if len(original_list) != len(detected_list):
        print("------len(original_list)", len(original_list), "len(detected_list)", len(detected_list)," test and detect length desn't match")
        return


    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        if hit:
            true_numer = true_numer + 1
    
    acc = true_numer * 1.0 / len(original_list)
    print("-----------------------------------Test Accuracy:{}--------\n".format(acc))

    return acc


#转化一个序列列表为稀疏矩阵    
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    
    return indices, values, shape
    

# 生成一个训练batch
def get_next_batch(batch_size=128):
    obj = gen_id_card()
    #(batch_size,256,32)
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1],OUTPUT_SHAPE[0]])
    codes = []

    for i in range(batch_size):
        #生成不定长度的字串
        image, text, vec = obj.gen_image()
        #np.transpose 矩阵转置 (32*256,) => (32,256) => (256,32)
        inputs[i,:] = np.transpose(image.reshape((OUTPUT_SHAPE[0],OUTPUT_SHAPE[1])))
        codes.append(list(text))
    targets = [np.asarray(i) for i in codes]
    sparse_targets = sparse_tuple_from(targets)
    #(batch_size,) 值都是256
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]

    return inputs, sparse_targets, seq_len
    


def get_train_model():   
    inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]])
    
    #定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)
    
    #1维向量 序列长度 [batch_size,]
    #seq_len = tf.placeholder(tf.int32, [None])
    seq_len = [256]
    
    # define one layer LSTM ---cell
    #cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    # define multi-layer LSTM
    stack_lstm = []
    for i in range(2):
        stack_lstm.append(tf.contrib.rnn.BasicLSTMCell(num_hidden,state_is_tuple=True))
    stack = tf.contrib.rnn.MultiRNNCell(stack_lstm, state_is_tuple=True)
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)
    
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]
    
    outputs = tf.reshape(outputs, [-1, num_hidden])
    W = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
    
    logits = tf.matmul(outputs, W) + b

    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    
    logits = tf.transpose(logits, (1, 0, 2))
    
    return logits, inputs, targets, seq_len
    
def train():

    #-------------------------------------------
    logits, inputs, targets, seq_len = get_train_model()

    '''
    loss = tf.nn.ctc_loss(labels=targets,inputs=logits, sequence_length=seq_len)
    loss_mean = tf.reduce_mean(loss)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                global_step,
                                                DECAY_STEPS,
                                                LEARNING_RATE_DECAY_FACTOR,
                                                staircase=True)
    
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=MOMENTUM).minimize(cost, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
    '''

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    print('-----------------------final output--logits-----------------------------',logits)
    

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver.restore(session, './model_ocr/model_lstm_ocr-1310')

  
        test_inputs,test_targets,_ = get_next_batch(1)
 
        print('--------------test_inputs-------------------',test_inputs.shape)
        
        print('--------------------------label -----------------',test_targets[1])
        test_feed = {inputs: test_inputs}
        decoded_data  = session.run(decoded[0], test_feed)
        print('-------------------decoded_data------------------------',decoded_data[1])



        
        constant_graph = graph_util.convert_variables_to_constants(session, session.graph_def, ['transpose'])
        with tf.gfile.FastGFile('model_ocr/model_lstm_ocr.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        




if __name__ == '__main__':
    #inputs, sparse_targets,seq_len = get_next_batch(1)
    #result = decode_sparse_tensor(sparse_targets)

    train()
