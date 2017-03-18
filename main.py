# -*- coding:utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
from model import CNN
import data_util


flags = tf.app.flags
flags.DEFINE_float("keep_prob", 0.8,
                   "probability of dropout")
flags.DEFINE_integer("depth", 128,
                     "output channel of filter")
flags.DEFINE_float("learning_rate", 0.007,
                   "initial learning rate of the model")
flags.DEFINE_float("is_static", False,
                   "Mode of CNN to train")
flags.DEFINE_string("logdir", "./logdir/",
                    "dir to save log files.")
flags.DEFINE_integer("sentence_length", 128,
                     "length of each sentence to feed into CNN")
flags.DEFINE_integer("epoch_num", 64,
                     "numbers of epoch to train model")
FLAGS = flags.FLAGS


def main(_):
    if not tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.MakeDirs(FLAGS.logdir)
    # 模型参数：
    filter_list = [2, 4, 8, 16, 32]
    train_batch_size = 64
    l2_lambda = 0.005
    embeddings = data_util.load_embedding_matrix()
    train_data, valid_data, test_data = data_util.load_data()
    initializer = tf.random_uniform_initializer()
    with tf.name_scope("train"):
        train_sentences, train_labels = data_util.data_producer(train_data, FLAGS.sentence_length, train_batch_size)
        with tf.variable_scope("model", initializer=initializer):
            train_model = CNN(
                keep_prob=FLAGS.keep_prob,
                filter_list=filter_list,
                depth=FLAGS.depth,
                learning_rate=FLAGS.learning_rate,
                embeddings=embeddings,
                sentence_length=FLAGS.sentence_length,
                sentences=train_sentences,
                labels=train_labels,
                is_static=True,
                is_train=True,
                batch_size=train_batch_size,
                l2_lambda=l2_lambda
            )
    with tf.name_scope("valid"):
        valid_sentences, valid_labels = data_util.data_producer(valid_data, FLAGS.sentence_length, len(valid_data))
        with tf.variable_scope("model", reuse=True):
            valid_model = CNN(
                keep_prob=FLAGS.keep_prob,
                filter_list=filter_list,
                depth=FLAGS.depth,
                learning_rate=FLAGS.learning_rate,
                embeddings=embeddings,
                sentence_length=FLAGS.sentence_length,
                sentences=valid_sentences,
                labels=valid_labels,
                is_static=True,
                is_train=False,
                batch_size=len(valid_data),
                l2_lambda=l2_lambda
            )
    with tf.name_scope("test"):
        test_sentences, test_labels = data_util.data_producer(test_data, FLAGS.sentence_length, len(test_data))
        with tf.variable_scope("model", reuse=True):
            test_model = CNN(
                keep_prob=FLAGS.keep_prob,
                filter_list=filter_list,
                depth=FLAGS.depth,
                learning_rate=FLAGS.learning_rate,
                embeddings=embeddings,
                sentence_length=FLAGS.sentence_length,
                sentences=test_sentences,
                labels=test_labels,
                is_static=True,
                is_train=False,
                batch_size=len(test_data),
                l2_lambda=l2_lambda
            )
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(logdir=FLAGS.logdir, global_step=train_model.global_step)
    with sv.managed_session(config=sess_config) as sess:
        steps = []
        valid_accuracys = []
        test_accuracys = []
        for i in xrange(FLAGS.epoch_num):
            train_model.run(sess, len(train_data)/train_batch_size)

            print "********************* VALID BEGIN *************************"
            valid_accuracy = valid_model.run(sess, 1)
            print "********************* VALID  END  *************************"
            print "********************* TEST BEGIN *************************"
            test_accuracy = test_model.run(sess, 1)
            print "********************* TEST  END  *************************"
            steps.append(i)
            valid_accuracys.append(valid_accuracy)
            test_accuracys.append(test_accuracy)
        plt.plot(steps, valid_accuracys, 'b-', label='valid accuracy')
        plt.plot(steps, test_accuracys, 'r-', label="test accuracy")
        plt.xlabel("steps")
        plt.ylabel("accuracy")
        plt.show()



if __name__ == "__main__":
    tf.app.run()
