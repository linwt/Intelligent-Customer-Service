import tensorflow as tf
import csv
import numpy as np
from gensim.models import Word2Vec
from collections import Counter


class Model:
    def __init__(self, meta_path, ckpt_path):
        self.graph = tf.Graph()

        # 恢复模型
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(meta_path)
            self.session = tf.Session(graph=self.graph)

        with self.session.as_default():
            with self.graph.as_default():
                self.saver.restore(self.session, ckpt_path)
                # 获取输入输出张量
                self.input_x1 = self.graph.get_tensor_by_name(name='input_x1:0')
                self.input_x2 = self.graph.get_tensor_by_name(name='input_x2:0')
                self.dropout = self.graph.get_tensor_by_name(name='dropout_keep_prob:0')
                self.Ew = tf.get_collection('Ew')[0]

    # 预测
    def predict(self, rand, labels):
        result = self.session.run(self.Ew, feed_dict={self.input_x1: [rand]*len(labels),
                                                       self.input_x2: labels,
                                                       self.dropout: 1.0})
        return result


tf.flags.DEFINE_string('train_file', '../input/compete-data/compete_train.csv', 'train file')
tf.flags.DEFINE_string('test_file', '../input/compete-data/compete_test.csv', 'test file')
tf.flags.DEFINE_string('final_model_path', '../input/final-models/model.ckpt', 'final model path')
tf.flags.DEFINE_string('char_model_path', '../input/w2v-model/wiki_char_model', 'character model path')
tf.flags.DEFINE_string('f', '', 'kernel')
FLAGS = tf.flags.FLAGS


# 获取字向量
def load_embedding():
    model = Word2Vec.load(FLAGS.char_model_path)
    word2index = {v:k for k,v in enumerate(model.wv.index2word)}
    return word2index


# 句子转成索引
def sent_to_idx(sent, word2index):
    sent2idx = [word2index.get(word, 0) for word in sent[:30]]
    num = 30 - len(sent2idx)
    for _ in range(num):
        sent2idx.append(0)
    return sent2idx


# 将所有句子转换成索引
def load_test_data(filename, word2index):
    x1, x2, data = [], [], []
    with open(filename, 'r') as file_data:
        reader = csv.reader(file_data)
        for line in reader:
            x1.append(sent_to_idx(line[0], word2index))
            x2.append(sent_to_idx(line[1], word2index))
            data.append(line)
    return x1, x2, data


print('加载数据...')
word2index = load_embedding()
test_rand, test_stand, test_data = load_test_data(FLAGS.test_file, word2index)

# 提取出所有标准问题
labels, standard_data = [], []
with open(FLAGS.train_file, 'r') as train:
    reader = csv.reader(train)
    for line in reader:
        if line[1] not in standard_data:
            labels.append(sent_to_idx(line[1], word2index))
            standard_data.append(line[1])

with open(FLAGS.test_file, 'r') as test:
    reader = csv.reader(test)
    for line in reader:
        if line[1] not in standard_data:
            labels.append(sent_to_idx(line[1], word2index))
            standard_data.append(line[1])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    count = limit_acc = limit_count = 0

    print('加载模型...')
    models = []
    for k in range(1, 6):
        path = FLAGS.final_model_path + str(k)
        model = Model(path + '.meta', path)
        models.append(model)

    print('开始测试...')
    for rand, stand in zip(test_rand, test_stand):
        sims, indexs = [], []
        for model in models:
            sim = model.predict(rand, labels)
            top5_sim = sorted(sim)[::-1][:5]
            sims.extend(top5_sim)
            top5_index = []
            for s in top5_sim:
                label_index = np.argwhere(sim == s)
                top5_index.append(label_index[0][0])
            indexs.extend(top5_index)
        res = Counter(indexs).most_common(5)

        max_sim = res_index = err = pre_count = 0

        print('随机：', test_data[count][0], '     标准：', test_data[count][1])
        for r in res:
            res_index = r[0]
            pre_count += 1
            print('预测' + str(pre_count) + '：', standard_data[res_index])
            pre_label = labels[res_index]
            if pre_label == stand:
                for i in indexs:
                    if i == res_index and sims[indexs.index(res_index)] > max_sim:
                        max_sim = sims[indexs.index(res_index)]
                if max_sim > 0.6:
                    limit_acc += 1
                    limit_count += 1
            else:
                err += 1
                if err == 5:
                    max_sim = sims[indexs.index(res_index)]
                    if max_sim > 0.6:
                        limit_count += 1

        print(' ')
        count += 1

    limit_accuracy = float('%.3f' %(1. * limit_acc / limit_count))
    recall = float('%.3f' %(1. * limit_acc / count))
    F1 = float('%.3f' %((2 * limit_accuracy * recall)/(limit_accuracy + recall)))

    print('============= 相似度大于0.6的条件下，', count, '条数据测试结果 =============')
    print('正确率：', limit_accuracy)
    print('召回率：', recall)
    print('F1值：', F1)