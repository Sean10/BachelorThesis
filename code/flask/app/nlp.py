#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/04/2018 12:30 PM
# @Author  : sean10
# @Site    : 
# @File    : nlp.py
# @Software: PyCharm

import numpy as np
import jieba
import tensorflow as tf
import re
from gensim.models import Word2Vec
import os


class nlp():
    def __init__(self):
        self.batchSize = 24
        self.lstmUnits = 64
        self.numClasses = 2
        self.iterations = 10000
        self.maxSeqLength = 50
        self.numDimensions = 300
        self.stoplist = self.stopwordslist("../stopwords.txt")
        self.model = Word2Vec.load("../model_word2vec")
        self.w2v_list = np.array([self.model.wv[word] for word in self.model.wv.vocab])
        self.word_list = np.array([word for word in self.model.wv.vocab]).tolist()
        # self.init_graph()

        tf.reset_default_graph()

        self.labels = tf.placeholder(tf.float32, [self.batchSize, self.numClasses])
        self.input_data = tf.placeholder(tf.int32, [self.batchSize, self.maxSeqLength])

        self.data = tf.Variable(tf.zeros([self.batchSize, self.maxSeqLength, self.numDimensions]), dtype=tf.float32)
        self.data = tf.nn.embedding_lookup(self.w2v_list, self.input_data)

        self.lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        self.lstmCell = tf.contrib.rnn.DropoutWrapper(cell=self.lstmCell, output_keep_prob=0.75)

        self.input = tf.unstack(self.data, self.maxSeqLength, 1)

        # value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
        self.value, self._ = tf.contrib.rnn.static_rnn(self.lstmCell, self.input, dtype=tf.float32)

        self.weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.numClasses]))
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        # value = tf.transpose(value, [1, 0, 2])
        # last = tf.gather(value, int(value.get_shape()[0]) - 1)
        self.last = self.value[-1]
        # print(value)
        # print(last)
        self.prediction = (tf.matmul(self.last, self.weight) + self.bias)
        self.correctPred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPred, tf.float32))

        self.sess = tf.InteractiveSession()
        # self.sess.run(tf.global_variables_initializer())

        # self.saver = tf.train.Saver()
        self.saver = tf.train.import_meta_graph('../models_lstm/pretrained_lstm.ckpt-9000.meta')
        self.sess.run(tf.global_variables_initializer())

        self.saver.restore(self.sess, tf.train.latest_checkpoint('../models_lstm'))

    def init_graph(self):
        tf.reset_default_graph()


        self.labels = tf.placeholder(tf.float32, [self.batchSize, self.numClasses])
        self.input_data = tf.placeholder(tf.int32, [self.batchSize, self.maxSeqLength])

        self.data = tf.Variable(tf.zeros([self.batchSize, self.maxSeqLength, self.numDimensions]), dtype=tf.float32)
        self.data = tf.nn.embedding_lookup(self.w2v_list, self.input_data)

        self.lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
        self.lstmCell = tf.contrib.rnn.DropoutWrapper(cell=self.lstmCell, output_keep_prob=0.75)

        self.input = tf.unstack(self.data, self.maxSeqLength, 1)

        # value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
        self.value, self._ = tf.contrib.rnn.static_rnn(self.lstmCell, self.input, dtype=tf.float32)

        self.weight = tf.Variable(tf.truncated_normal([self.lstmUnits, self.numClasses]))
        self.bias = tf.Variable(tf.constant(0.1, shape=[self.numClasses]))
        # value = tf.transpose(value, [1, 0, 2])
        # last = tf.gather(value, int(value.get_shape()[0]) - 1)
        self.last = self.value[-1]
        # print(value)
        # print(last)
        self.prediction = (tf.matmul(self.last, self.weight) + self.bias)
        self.correctPred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPred, tf.float32))


        self.sess = tf.InteractiveSession()

        self.saver = tf.train.Saver()
        # saver = tf.train.import_meta_graph('my_test_model-1000.meta')
        self.sess.run(tf.global_variables_initializer())

        self.saver.restore(self.sess, tf.train.latest_checkpoint('../models_lstm'))

    def getSentenceMatrix(self, sentence):
        arr = np.zeros([self.batchSize, self.maxSeqLength])
        sentenceMatrix = np.zeros([self.batchSize, self.maxSeqLength], dtype='int32')
        cleanedSentence = self.cleanSentences(sentence)
        split = jieba.cut(cleanedSentence)

        for indexCounter, word in enumerate(split):
            if word in self.stoplist:
                continue
            try:
                print(word)
                sentenceMatrix[0, indexCounter] = self.word_list.index(word)
            except ValueError:
                sentenceMatrix[0, indexCounter] = 572296  # Vector for unkown
        return sentenceMatrix

    def stopwordslist(self, filepath):
        stopwords = {line.strip() for line in open(filepath, "r", encoding='utf-8').readlines()}
        return stopwords

    def cleanSentences(self, string):
        return re.sub("[a-zA-Z0-9\s+\.\!\/_,$%^*(+\"\']+|[+——！<>《》，。？、~@#￥%……&*（） ]+", "", string)

    def analysis(self, text):

        print(text)

        inputMatrix = self.getSentenceMatrix(text)
        predictedSentiment = self.sess.run(self.prediction, {self.input_data: inputMatrix})
        # print(predictedSentiment)
        if (predictedSentiment[0][0] > predictedSentiment[0][1]):
            return "Positive Sentiment"
        else:
            return "Negative Sentiment"

        # return "positive"




