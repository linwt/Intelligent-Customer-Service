import tensorflow as tf
import csv
import random
import numpy as np
from gensim.models import Word2Vec
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


class siameseLSTM(object):

    # bilstm
    def bi_lstm(self, rnn_size, layer_size, keep_prob):

        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_fw_cell_mul = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list), output_keep_prob=keep_prob)

        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_bw_cell_mul = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list), output_keep_prob=keep_prob)

        return lstm_fw_cell_mul, lstm_bw_cell_mul

    # 获取权重
    def get_weight(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name=name)

    # 获取偏度
    def get_bias(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    # 转置
    def transpose_inputs(self, inputs, rnn_size, sequence_len):
        # 将所有句子的第n个字向量组成一个维度
        inputs = tf.transpose(inputs, [1, 0, 2])    # (50, ?, 300)
        # 重塑矩阵为二维
        inputs = tf.reshape(inputs, [-1, rnn_size])
        # 将矩阵切分成sequence_len批，每一批为所有句子的第n个字向量的组合
        inputs = tf.split(inputs, sequence_len, 0)
        return inputs

    # 余弦相似度
    def cos_sim(self, x1, x2):
        mul_x1x2 = tf.reduce_sum(tf.multiply(x1, x2), 1)
        norm_x1 = tf.sqrt(tf.reduce_sum(tf.square(x1), 1))
        norm_x2 = tf.sqrt(tf.reduce_sum(tf.square(x2), 1))
        Ew = mul_x1x2 / (norm_x1 * norm_x2)
        return Ew

    # 对比损失
    def contrastive_loss(self, Ew, y):
        l_1 = 0.25 * tf.square(1 - Ew)
        l_0 = tf.square(tf.maximum(Ew, 0))
        loss = tf.reduce_sum(y * l_1 + (1 - y) * l_0)
        return loss

    # 初始化
    def __init__(self, rnn_size, layer_size, sequence_len, grad_clip, learning_rate, decay_rate, decay_step, embedding):

        self.input_x1 = tf.placeholder(tf.int32, shape=[None, sequence_len], name='input_x1')
        self.input_x2 = tf.placeholder(tf.int32, shape=[None, sequence_len], name='input_x2')
        self.y = tf.placeholder(tf.float32, shape=[None], name='y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # 由句子每个id获取对应向量
        with tf.name_scope('embedding'):
            inputs_x1 = tf.nn.embedding_lookup(embedding, self.input_x1)
            inputs_x2 = tf.nn.embedding_lookup(embedding, self.input_x2)
            print('inputs_x1', inputs_x1.shape)     # shape=(?, 50, 300)   ?为batch_size

            inputs_x1 = self.transpose_inputs(inputs_x1, rnn_size, sequence_len)
            inputs_x2 = self.transpose_inputs(inputs_x2, rnn_size, sequence_len)
            print('inputs_x1', inputs_x1)           # split:0 ~ split:49   shape=(?, 300)

        with tf.name_scope('output'):
            bilstm_fw, bilstm_bw = self.bi_lstm(rnn_size, layer_size, self.dropout_keep_prob)

            # 返回值是一个tuple(outputs, outputs_state_fw, output_state_bw )
            # outputs为一个长度为600的list，每一个元素都包含正向和反向的输出(output_fw, output_bw)
            outputs_x1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bilstm_fw, bilstm_bw, inputs_x1, dtype=tf.float32)
            print('outputs_x1：', outputs_x1)        # concat:0 ~ concat_49:0    shape=(?, 600)

            output_x1 = tf.reduce_max(outputs_x1, 0)        # Max:0     shape=(?, 600)
            print('output_x1：', output_x1)

            # 开启变量重用的开关，共享参数
            tf.get_variable_scope().reuse_variables()

            outputs_x2, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bilstm_fw, bilstm_bw, inputs_x2, dtype=tf.float32)
            output_x2 = tf.reduce_max(outputs_x2, 0)

        with tf.name_scope('dense_layer'):

            fc_w = self.get_weight([2*rnn_size, 128], 'fc_w')
            fc_b = self.get_bias([128], 'fc_b')

            logits_1 = tf.matmul(output_x1, fc_w) + fc_b
            logits_2 = tf.matmul(output_x2, fc_w) + fc_b

            print('fw(logits_x1) shape:', logits_1.shape)       # (?, 128)
            print('fw(logits_x2) shape:', logits_2.shape)

        with tf.name_scope('loss'):
            self.Ew = self.cos_sim(logits_1, logits_2)
            tf.add_to_collection('Ew', self.Ew)
            self.loss = self.contrastive_loss(self.Ew, self.y)

        with tf.name_scope('optimizer'):
            # 获得所有可进行训练的变量
            tvars = tf.trainable_variables()
            # 计算所有参数的梯度
            gradiends = tf.gradients(self.loss, tvars)
            # 修正梯度值，控制梯度爆炸。参数：梯度张量、截取的比率    返回：截取后的梯度张量、所有张量的全局范数
            grads, _ = tf.clip_by_global_norm(gradiends, grad_clip)

            global_steps = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(learning_rate=learning_rate, global_step=global_steps,
                                                            decay_steps=decay_step, decay_rate=decay_rate, staircase=False)
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            # 将新的梯度应用到变量上
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_steps)


tf.flags.DEFINE_integer('rnn_size', 300, 'hidden units of RNN, as well as dimentionality of character embedding')
tf.flags.DEFINE_integer('layer_size', 2, 'number of layers of RNN')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.flags.DEFINE_integer('sequence_len', 30, 'sequence length')
tf.flags.DEFINE_integer("num_epochs", 1, 'Number of training epochs')
tf.flags.DEFINE_integer('decay_step', 100, 'decay step')
tf.flags.DEFINE_float('dropout_keep_prob', 0.2, 'dropout keep probability')
tf.flags.DEFINE_float('grad_clip', 10.0, 'clip gradients at this value')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.flags.DEFINE_float('decay_rate', 0.9, 'decay rate')
tf.flags.DEFINE_string('train_file', '../input/final-data/final_regroup.csv', 'train raw file')
tf.flags.DEFINE_string('test_file', '../input/final-data/final_test.csv', 'test raw file')
tf.flags.DEFINE_string('standard_file', '../input/mycompete/standard.csv', 'standard raw file')
tf.flags.DEFINE_string('save_path', './model.ckpt', 'model save directory')

tf.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.flags.FLAGS

# 获取词向量
def load_embedding():
    model = Word2Vec.load('../input/wikimodel/wiki_char_model/wiki_char_model')
    embedding = model.wv.syn0
    # embedding = model.wv.vectors
    word2index = {v:k for k,v in enumerate(model.wv.index2word)}
    return embedding, word2index

# 句子转成索引
def sent_to_idx(sent, word2index, sequence_len):
    sent2idx = [word2index.get(word, 0) for word in sent[:sequence_len]]
    num = sequence_len - len(sent2idx)
    for _ in range(num):
        sent2idx.append(0)
    return sent2idx

# 将所有句子转换成索引
def load_train_data(filename, word2index, sequence_len):
    x1, x2, y = [], [], []
    with open(filename, 'r') as file_data:
        reader = csv.reader(file_data)
        for line in reader:
            x1.append(sent_to_idx(line[0], word2index, sequence_len))
            x2.append(sent_to_idx(line[1], word2index, sequence_len))
            y.append(line[2])
    return x1, x2, y

def load_test_data(filename, word2index, sequence_len):
    x1, x2, data = [], [], []
    with open(filename, 'r') as file_data:
        reader = csv.reader(file_data)
        for line in reader:
            x1.append(sent_to_idx(line[0], word2index, sequence_len))
            x2.append(sent_to_idx(line[1], word2index, sequence_len))
            data.append(line)
    return x1, x2, data

print('开始加载数据...')
embedding, word2index = load_embedding()
train_x1, train_x2, train_y = load_train_data(FLAGS.train_file, word2index, FLAGS.sequence_len)
test_rand, test_stand, test_data = load_test_data(FLAGS.test_file, word2index, FLAGS.sequence_len)

num = random.randint(0, 100)
# random.seed(num)
# random.shuffle(train_x1)
# random.seed(num)
# random.shuffle(train_x2)
# random.seed(num)
# random.shuffle(train_y)

random.seed(num)
random.shuffle(test_rand)
random.seed(num)
random.shuffle(test_stand)
random.seed(num)
random.shuffle(test_data)

test_rand = test_rand[:1000]
test_stand = test_stand[:1000]
test_data = test_data[:1000]

# 提取出所有标准问
labels, standard_data = [], []
with open(FLAGS.standard_file, 'r') as stand:
    reader = csv.reader(stand)
    for line in reader:
        labels.append(sent_to_idx(line[0], word2index, FLAGS.sequence_len))
        standard_data.append(line[0])

print('加载完成')

# 计算有多少个批次
train_total = len(train_x1)
train_batch_num = train_total // FLAGS.batch_size

model = siameseLSTM(FLAGS.rnn_size, FLAGS.layer_size, FLAGS.sequence_len, FLAGS.grad_clip, FLAGS.learning_rate,
                    FLAGS.decay_rate, FLAGS.decay_step, embedding)
saver = tf.train.Saver()

print('开始训练...')
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # saver = tf.train.import_meta_graph('../input/lstm2-v68-epoch2/model.ckpt.meta')
    # ckpt = tf.train.latest_checkpoint('../input/lstm2-v68-epoch2/')
    # if ckpt:
    #     print('加载模型...')
    #     saver.restore(sess, ckpt)

    # 训练数据
    for i in range(FLAGS.num_epochs):
        num = random.randint(0, 100)
        random.seed(num)
        random.shuffle(train_x1)
        random.seed(num)
        random.shuffle(train_x2)
        random.seed(num)
        random.shuffle(train_y)
        for j in range(train_batch_num):
            _, loss, lr = sess.run([model.train_op, model.loss, model.learning_rate],
                                  feed_dict={model.input_x1: train_x1[j*FLAGS.batch_size: (j+1)*FLAGS.batch_size],
                                              model.input_x2: train_x2[j*FLAGS.batch_size: (j+1)*FLAGS.batch_size],
                                              model.y: train_y[j*FLAGS.batch_size: (j+1)*FLAGS.batch_size],
                                              model.dropout_keep_prob: FLAGS.dropout_keep_prob})
            if j % 50 == 0:
                print('迭代次数：', i, ' 训练批次：', j, ' 损失值：', loss, ' 学习率：', lr)

        saver.save(sess, FLAGS.save_path)
    # 查看模型保存内容
    # print_tensors_in_checkpoint_file(FLAGS.save_path, None, True)

    print('训练完成，开始测试...')

    # 计算准确率
    acc = 0
    count = 0
    limit_acc = 0
    limit_count = 0
    err_pre = []

    n = len(labels)

    for rand, stand in zip(test_rand, test_stand):
        sim = sess.run(model.Ew, feed_dict={model.input_x1: [rand]*n,
                                            model.input_x2: labels,
                                            model.dropout_keep_prob: 1.0})
        # print('相似度：', sim)
        max_sim = max(sim)
        print('最大相似度：', max_sim)
        label_index = np.argwhere(sim == max_sim)
        # print('最大相似度索引：', label_index)
        pre_label = labels[label_index[0][0]]
        # print('最大相似度标签：', pre_label)
        print('随机：', test_data[count][0], '     标准：', test_data[count][1], '    预测：', standard_data[label_index[0][0]])

        if pre_label == stand and max_sim > 0.8:
            limit_acc += 1
        if max_sim > 0.8:
            limit_count += 1

        if pre_label == stand:
            acc += 1
        else:
            err_pre.append([test_data[count][0], test_data[count][1], standard_data[label_index[0][0]]])
        count += 1

        if count % 50 == 0:
            accuracy = 1. * acc / count
            limit_accuracy = 1. * limit_acc / limit_count
            recall = 1. * limit_acc / count
            F1 = (2 * limit_accuracy * recall)/(limit_accuracy + recall)

            print('============= ', count, '条数据测试结果 =============')
            print('数据量：', count, '            正确个数：', acc, '            正确率：', accuracy)
            print('数据量(sim>0.8)：', limit_count, '   正确个数(sim>0.8)：', limit_acc, '    正确率：', limit_accuracy)
            print('数据量：', count, '            正确个数(sim>0.8)：', limit_acc, '    召回率:', recall)
            print('F1值：', F1)
            print('预测错误：', err_pre)