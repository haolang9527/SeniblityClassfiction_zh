# -*- coding: utf-8 -*-
""" 完成数据的预处理, 主要是向模型提供embedding_matrix, train_data, valid_data, test_data
    1. 函数load_embedding()得到embedding_matrix
    2. load_data() 得到train_data, valid_data, test_data.   data结构：[...(sentence, label)...]
    """
import os
import tensorflow as tf
import pynlpir
import cPickle as pickle
import sys
from gensim.models import Word2Vec
import numpy as np
import json

# 这两句大大的有用， 设置整个程序的默认编码为utf-8
reload(sys)
sys.setdefaultencoding("utf-8")

_PAD_ID = 0
POS_LABEL = [[1, 0]]
NEG_LABEL = [[0, 1]]

STOP_PATH = './data/stopword1208.txt'
NEG_PATH = './data/hotelNeg2443.txt'
POS_PATH = './data/hotelPos2322.txt'
MODEL_PATH = './data/pre_trained_embedding/Word60.model'
TRAIN_PATH = "./data/train_data.pickle"
VALID_PATH = "./data/valid_data.pickle"
TEST_PATH = "./data/test_data.pickle"

def _read_words(path, stop_word_flag=False, read_to_sentences=False):
    """ 给定文本的path， 把所有分割好的token读出
    sentences_flag : True  -> 返回[[sentence 0 ]...[sentence 99 ] ... ],
                     False -> 返回[word0, word2,......word999999...] (将句子截断)
    """
    data = []
    pynlpir.open(data_dir='/home/haolang/PycharmProjects/SeniblityClassfiction_zh/')
    with tf.gfile.GFile(path, mode='r') as file:
        for line in file:
            sent = line.strip()  # 砍掉头部和尾部的空字符
            if not stop_word_flag:
                words = pynlpir.segment(sent, pos_tagging=False)
                if read_to_sentences:
                    data.append(words)
                else:
                    for word in words:
                        data.append(unicode(word)) # 把生成器里面的词一个个取出来
            else:
                data.append(sent.decode("utf-8"))
    pynlpir.close()
    return data


def file_to_vocab(pos_path, neg_path, stop_path, model):
    vocab_path = "./data/vocabulary.pickle"
    reversed_vocab_path = "./data/reversed_vocabulary.pickle"
    # 词表存在则直接加载
    if os.path.exists(vocab_path) and os.path.exists(reversed_vocab_path):
        with tf.gfile.GFile(vocab_path) as file:
            with tf.gfile.GFile(reversed_vocab_path) as re_file:
                return pickle.load(file), pickle.load(re_file)

    print "下面开始制作词表"
    pos_words = _read_words(pos_path)
    neg_words = _read_words(neg_path)
    stop_words = _read_words(stop_path, stop_word_flag=True)

    # 开始去停用词
    total_words = pos_words + neg_words
    print "total length", len(total_words)
    total_words = [word.decode("utf-8") for word in total_words if word not in stop_words]

    # collections.Counter 搞不定Unicode码， 自己手动统计词频
    count_pairs = {}
    illegal_words = []  # 不在预训练模型内的词
    for word in total_words:
        if word in model.index2word:
            if word in count_pairs.keys():
                count_pairs[word] += 1
            else:
                count_pairs[word] = 1
                print word, "进入词表"
        else:
            print word, "不在预训练的模型内"
            illegal_words.append(word)

    with open("./data/illegal_words.txt", mode="w") as file:
        for line in illegal_words:
            file.write("".join(line) + "\n")
    # 把统计词频后的dict排序， 砍掉很低频的词和高频的词，留下5000个词
    sorted_pairs = sorted(count_pairs.items(), key=lambda item: item[1])
    vocab_pairs = sorted_pairs[-3050: -50]
    # 这边这么打算： 不采用UNK， 扔掉所有不在词表里面的词，
    # 给每个token按序号分配id PAD（填充词）放进表的头部
    vocab_dic = dict((vocab_pairs[i][0], i+1) for i in xrange(len(vocab_pairs)))
    vocab_dic["PAD"] = _PAD_ID  # 0
    # 将词表字典序列化保存起来。
    with tf.gfile.GFile(vocab_path, mode="w") as file:
        pickle.dump(vocab_dic, file)
    # 把字典放入json
    with open("./data/vocab.json", 'w') as file:
        dict_json = json.dump(vocab_dic, file)
    # 将词典翻转之后保存起来: {词 : id } -> {id : 词}
    reversed_vocab = {id:token for token, id in vocab_dic.items()}
    with tf.gfile.GFile(reversed_vocab_path, mode="w") as file:
        pickle.dump(reversed_vocab, file)
    return vocab_dic, reversed_vocab


def load_embedding_matrix():
    embedding_path = "./data/embeddings.pickle"
    if os.path.exists(embedding_path):
        print "开始从磁盘读取保存的embedding matrix"
        with tf.gfile.GFile(embedding_path) as file:
            return pickle.load(file)
    else:
        print "开始从预训练的词向量模型load embedding matrix"
        embedding_model = Word2Vec.load(MODEL_PATH)
        _, reversed_vocab = file_to_vocab(POS_PATH, NEG_PATH, STOP_PATH, embedding_model)
        embedding_matrix = [[0 for i in xrange(60)]]  # PAD 词向量为 [0,0,...0,0]
        for i in xrange(len(reversed_vocab) - 1):  # PAD 已经写进去了
            id = i + 1 # 从id = 1 到len(reversed_vocab)
            word = reversed_vocab[id]
            embedding = embedding_model[word.decode("utf-8")]
            embedding_matrix.append(embedding)
        embedding_matrix = np.array(embedding_matrix)  # 打包成np.array
        with tf.gfile.GFile(embedding_path, mode='w') as file:
            pickle.dump(embedding_matrix, file)  # 腌好，保存
        return embedding_matrix


def sentence2ids(sentence, vocabulary):
    ids = []
    for word in sentence:
        id = vocabulary.get(word, None)
        if id is not None:
            ids.append(id)
    return ids


def prepare_data(pos_path, neg_path, stop_path, model):
    vocabulary, _ = file_to_vocab(pos_path, neg_path,stop_path, model)
    total_data = []
    # 把预料从文本里面读出来，全部存放到total_data 里面
    for path in pos_path, neg_path:
        word_sentences = _read_words(path, read_to_sentences=True)
        id_sentences = []
        for sentence in word_sentences:
            id_sentence = sentence2ids(sentence, vocabulary)
            id_sentences.append(id_sentence)
        label = NEG_LABEL if 'Neg' in path else POS_LABEL  # 判断这些句子的label
        label = label * len(id_sentences) # label的个数要等于句子总数
        data = list(zip(id_sentences, label)) # 打包成格式： [(sentence, label)]
        print "get %d sentence in %s" % (len(data), path.split('/')[-1])
        total_data += data

    np.random.shuffle(total_data)  # 把原本规整的由POS NGE排列好的数据打乱
    total_data_num = len(total_data) # 统计数据总数，分割成train valid test
    # 按照 8:1:1 的比例分配train_data, valid_data, test_data
    train_data_num = int(total_data_num * 0.8)
    valid_data_num = int(total_data_num * 0.1)
    test_data_num = int(total_data_num * 0.1)
    print "train data： %d, valid data %d,  test data %d" %(train_data_num, valid_data_num, test_data_num)
    train_data = total_data[: train_data_num]
    valid_data = total_data[train_data_num: train_data_num + valid_data_num]
    test_data = total_data[train_data_num + valid_data_num:]
    print "train data： %d, valid data %d,  test data %d" % (len(train_data), len(valid_data), len(test_data))
    # 保存这三个数据
    with tf.gfile.GFile(TRAIN_PATH, mode='w') as file:
        pickle.dump(train_data, file)
    with tf.gfile.GFile(VALID_PATH, mode='w') as file:
        pickle.dump(valid_data, file)
    with tf.gfile.GFile(TEST_PATH, mode='w') as file:
        pickle.dump(test_data, file)

    return train_data, valid_data, test_data


def load_data():
    if os.path.exists(TRAIN_PATH) and os.path.exists(VALID_PATH) and os.path.exists(TEST_PATH):
        print "开始从pickle文件里读取train_data, valid_data, test_data"
        with tf.gfile.GFile(TRAIN_PATH) as file:
            train_data = pickle.load( file)
        with tf.gfile.GFile(VALID_PATH) as file:
            valid_data = pickle.load(file)
        with tf.gfile.GFile(TEST_PATH) as file:
            test_data = pickle.load(file)
        return train_data, valid_data, test_data
    else:
        return prepare_data(POS_PATH, NEG_PATH, STOP_PATH,
                            Word2Vec.load(MODEL_PATH))


def data_producer(data, sentence_length, batch_size):
    """
    输入数据， 稳定的为graph提供数据
    :param data:  数据 [...(sentence, labels)...]
    :param sentence_length: 限定的句子长度
    :param train_batch_size: data不是训练数据时， 每次讲整个数据集提供给graph
    :return:
    """
    _sentences, labels = zip(*data)
    with tf.name_scope(name="data_producer"):
        labels = tf.convert_to_tensor(labels, name='labels')
        sentences = []
        for sentence in _sentences:
            if len(sentence) < sentence_length:
                sentence += ([0] * (sentence_length - len(sentence)))  # PAD 不够长的句子
            else:
                sentence = sentence[:sentence_length] # 截断过长的句子
            sentences.append(sentence)
        sentences = tf.convert_to_tensor(sentences, name="Sentences")
        # 定下这块数据共能够分为几次运行
        # 当data是test 或者valid的时候  会出现range_intput_producer(0)
        if len(data) == batch_size :
            return sentences, labels
        else:
            i = tf.train.range_input_producer((len(data) / batch_size) - 1).dequeue()
            x = sentences[i * batch_size : (i + 1) * batch_size]
            y = labels[i * batch_size : (i + 1) * batch_size]
            return x, y

if __name__ == '__main__':
    train_data, valid_data, test_data = load_data()
    embeddings = load_embedding_matrix()

    test_sentence, test_label = zip(*test_data)
    print "train data： %d, valid data %d,  test data %d" % (len(train_data), len(valid_data), len(test_data))
    assert len(test_sentence) == len(test_label)
    print test_sentence[0]
    print test_label[0]
    print embeddings[0]


