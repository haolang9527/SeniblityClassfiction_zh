# -*- coding: utf-8 -*-
import tensorflow as tf

class CNN(object):
    def __init__(self, keep_prob, filter_list,depth, learning_rate, batch_size, is_train, is_static, sentences, labels,
                 embeddings, sentence_length, l2_lambda):
        """TODO:
                1. 添加 valid test 选项
                2.

        :param keep_prob: dropout 的概率
        :param depth: 滤波器的 output_channel
        :param filter_list:  包含多个滤波器的列表
        :param learning_rate: 模型的学习速度
        :param batch_size: 模型训练时每次读入的句子量
        :param is_train: 是否 dropout
        :param is_static:  静态则embedding不会在学习过程中改变
        :param embeddings: 预训练好的词向量列表 np.array
        :param embedding_dim: 词向量的长度
        :param data:  带标签的所有句子 [...(sentence, label)...]
        :param log_dir: 存放checkpoint 和 tensorboard需要的数据
        :param sentence_length:  限定的句子长度， 句子过长则砍掉多余部分，过短则填充PAD词， 如果没有指定，会赋值为训练集最长的句子长度
        """
        self.keep_prob = keep_prob
        self.filter_size_list = filter_list
        self.depth = depth
        # self.learning_rate = tf.get_variable(name="learning_rate", dtype=tf.float32,shape=[],trainable=False,
        #                                     initializer=tf.ones_initializer(dtype=tf.float32))
        self.learning_rate = tf.Variable(name="learning_rate", initial_value=learning_rate, trainable=False)
        self.batch_size = batch_size
        self.is_train = is_train
        self.is_static = is_static
        self.embeddings = embeddings
        self.embedding_dim = len(embeddings[0])
        self.batch_sents = sentences
        self.batch_labels = labels
        self.sentence_length = sentence_length
        self.l2_lambda = l2_lambda

        self.build_model()

    def build_model(self):
        # 先搞定词向量矩阵
        self.l2_loss = 0
        with tf.name_scope("Embedding_Matrix"):
            self.embedding_matrix = None
            if self.is_static:
                self.embedding_matrix = tf.constant(self.embeddings, name="embedding_matrix")
            else:
                # self.embedding_matrix = tf.get_variable(name="embedding_matrix", shape=[5001, 60], # 5000个词，加上一个PAD
                #                                   initializer=self.embeddings)
                self.embedding_matrix = tf.Variable(self.embeddings, name="embedding_matrix")
                self.l2_loss += tf.nn.l2_loss(self.embedding_matrix)
        # 拿到输入数据

        batch_sentences = tf.nn.embedding_lookup(self.embedding_matrix, self.batch_sents, name="Lookup_embedding_matrix")

        _, sent_len, embedding_dim = batch_sentences.get_shape().as_list()
        new_shape = (self.batch_size, sent_len, embedding_dim, 1)
        self.batch_sentences = tf.reshape(batch_sentences, new_shape)

        # 搭建CNN结构
        with tf.variable_scope("convolution_layer"):
            conv1_outputs = []
            for filter_size in self.filter_size_list:
                # 指定第一层卷积的滤波器  [filter_size, embedding_dim, in_channel(1), out_channels(depth)]
                filter = tf.get_variable(name="filter_size_%d"%filter_size,
                                         shape=[filter_size, self.embedding_dim, 1, self.depth],
                                         initializer=tf.truncated_normal_initializer(seed=666))
                bias = tf.get_variable(name="bias_size_%d"%filter_size, shape=[self.depth],
                                       initializer=tf.ones_initializer())
                # 添加l2 正则项
                self.l2_loss += (tf.nn.l2_loss(filter) + tf.nn.l2_loss(bias))
                # 开始卷积
                conv = tf.nn.conv2d(tf.cast(self.batch_sentences, tf.float32), filter, strides=[1,1,1,1], padding='VALID',
                                    use_cudnn_on_gpu=True, name="conv_size_%d"%filter_size)
                relu = tf.nn.relu(tf.nn.bias_add(conv, bias))  # 加偏置
                pooling_ksize = [1, self.sentence_length - filter_size + 1, 1, 1]
                output = tf.nn.max_pool(relu, pooling_ksize, [1,1,1,1], padding="VALID",
                                        name="pooling_size_%d"%filter_size) # pooling
                # output: [batch_size, 1, 1, depth]
                conv1_outputs.append(output)
        hidden = tf.concat(3, conv1_outputs, name="feature_map_of_concat_filters")
        with tf.variable_scope("Fully_connection_layer"):
            fc_weights = tf.get_variable(name="FC_weights", shape=[len(self.filter_size_list) * self.depth, 2],
                                         initializer=tf.truncated_normal_initializer(seed=666)
                                         )
            fc_bias = tf.get_variable(name="FC_bias", shape=[2], dtype=tf.float32,
                                      initializer=tf.ones_initializer())
            self.l2_loss += (tf.nn.l2_loss(fc_weights) + tf.nn.l2_loss(fc_bias))
            hidden = tf.reshape(hidden, [self.batch_size, -1])
            if self.is_train:
                hidden = tf.nn.dropout(hidden, self.keep_prob, seed=666, name="dropout")
            output = tf.matmul(hidden, fc_weights) + fc_bias  # 加个非线性变换

        # 此次模型预测值
        self.prediction = tf.nn.softmax(output, name="prediction")
        # 此次模型的loss
        self.loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(output, self.batch_labels))
        self.total_loss = self.l2_loss + self.loss

        self.accuracy = self.accuracy(self.prediction, self.batch_labels)
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
        # tensorboard 画图用到的step (横轴)
        self.global_step = tf.Variable(1, trainable=False)
        self.global_step_update = tf.assign_add(self.global_step, 1)

        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_loss)
        self.lr_update = tf.assign(self.learning_rate, tf.multiply(self.learning_rate, 0.96))

        if self.is_train:
            tf.scalar_summary("total_loss", self.total_loss)
            tf.scalar_summary("l2_loss", self.l2_loss)
            tf.scalar_summary("learning_rate", self.learning_rate)
            tf.scalar_summary("global_step", self.global_step)
            tf.histogram_summary("embeddings", self.embedding_matrix)
            tf.scalar_summary("accuracy", self.accuracy)

    def run(self, sess, run_step):
        fetch_dict = {"accuracy":self.accuracy, "loss":self.total_loss}
        if self.is_train:
            fetch_dict["train_op"] = self.train_op

        with sess.as_default():
            # 开整！
            for i in xrange(run_step):
                vals = sess.run(fetch_dict)
                if i % 30 == 0:
                    print_log = "iteration:    {0}\n" \
                                "     loss:    {1}\n" \
                                "  l2_loss:    {3}\n" \
                                " accuracy:    {2}\n" \
                                "       lr:    {4}\n".format(self.global_step.eval(), vals["loss"], vals["accuracy"],
                                                             self.l2_loss.eval(), self.learning_rate.eval())
                    print print_log
                    sess.run([self.global_step_update, self.lr_update])

                    with open("./print_log_dynamic_withFC_32_keep_prob%.3f.txt"%self.keep_prob, "a") as file:
                        if not self.is_train:
                            file.write("*************************************************\n")
                        file.write(print_log)  # 文件存档
                        if not self.is_train:
                            file.write("*************************************************\n")
                            return vals["accuracy"]


    def accuracy(self, prediction, labels):
        pre = tf.arg_max(prediction, 1)
        lab = tf.arg_max(labels, 1)
        accracy = 1.0 - (tf.count_nonzero(pre-lab, dtype=tf.float32) / self.batch_size)
        return accracy






