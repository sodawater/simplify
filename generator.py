import tensorflow as tf
import numpy as np
#from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import data_utils
import random
import os
from tensorflow.python.layers import core as layers_core

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class Generator():
    def __init__(self,hparams, mode):
        self.vocab_size = hparams.to_vocab_size
        self.emb_dim = hparams.emb_dim
        self.num_units = hparams.units
        self.num_layers = hparams.num_layers
        self.learning_rate = tf.Variable(float(hparams.learning_rate), trainable=False)
        self.clip_value = hparams.clip_value
        self.max_seq_length = 50

        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.decoder_input_ids = tf.placeholder(dtype=tf.int32, shape=[None,None])
            self.decoder_input_length = tf.placeholder(dtype=tf.int32, shape=[None])
            self.initial_state = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
            self.batch_size = tf.size(self.decoder_input_length)
        else:
            self.batch_size = 1

        with tf.variable_scope("embedding") as scope:
            self.embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.emb_dim]))

        with tf.variable_scope("projection") as scope:
            self.output_layer = layers_core.Dense(self.vocab_size)

        with tf.variable_scope("decoder") as scope:
            if self.num_layers > 1:
                decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.num_units) for _ in range(self.num_layers)])
            else:
                decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units)
            if mode != tf.contrib.learn.ModeKeys.INFER:
                initial_state = self.initial_state
                with tf.device("/cpu:0"):
                    decoder_inputs = tf.nn.embedding_lookup(self.embeddings, self.decoder_input_ids)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, self.decoder_input_length)
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                             helper=helper,
                                                             initial_state=initial_state,
                                                             output_layer=self.output_layer
                                                             )
                decoder_outputs, decoder_state, decoder_output_len = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                                                       maximum_iterations=self.max_seq_length * 2,
                                                                                                       swap_memory=True,)
                self.sample_id = decoder_outputs.sample_id
                self.logits = decoder_outputs.rnn_output
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings, [hparams.GO_ID], hparams.EOS_ID)
                initial_state = self.initial_state
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                             helper=helper,
                                                             initial_state=initial_state,
                                                             output_layer=self.output_layer)
                decoder_outputs, decoder_state, decoder_output_len = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                                                       maximum_iterations=self.max_seq_length * 2,
                                                                                                       swap_memory=True)
                self.sample_id = tf.unstack(decoder_outputs.sample_id, axis=0)

        with tf.variable_scope("rollout") as scope:
            self.given_decoder_inputs_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.given_decoder_length = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.given_next_ids = tf.placeholder(dtype=tf.int32, shape=[None])
            initial_state = self.initial_state
            with tf.device("/cpu:0"):
                given_decoder_inputs = tf.nn.embedding_lookup(self.embeddings, self.given_decoder_input_ids)
            helper1 = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, self.decoder_input_length)
            my_decoder1 = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                             helper=helper,
                                                             initial_state=initial_state,
                                                             output_layer=self.output_layer
                                                             )
            decoder1_outputs, decoder1_state, decoder1_output_len = tf.contrib.seq2seq.dynamic_decode(my_decoder1,
                                                                                                   maximum_iterations=self.max_seq_length * 2,
                                                                                                   swap_memory=True)

            helper2 = tf.contrib.seq2seq.SampleEmbeddingHelper(self.embeddings, self.given_next_ids, hparams.EOS_ID)
            my_decoder2 = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                          helper=helper,
                                                          initial_state=decoder1_state,
                                                          output_layer=self.output_layer
                                                          )
            decoder2_outputs, decoder2_state, decoder2_output_len = tf.contrib.seq2seq.dynamic_decode(my_decoder2,
                                                                                                      maximum_iterations=self.max_seq_length * 2,
                                                                                                      swap_memory=True)



        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.target_weights = tf.placeholder(dtype=tf.float32, shape=[None, None])
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, None])
            with tf.variable_scope("loss") as scope:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
                self.pretrain_loss = tf.reduce_sum(crossent * self.target_weights) / tf.to_float(self.batch_size)
                self.train_loss = tf.reduce_sum(crossent * self.rewards) / tf.to_float(self.batch_size)

            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.pretrain_global_step = tf.Variable(0, trainable=False)
                self.train_global_step = tf.Variable(0, trainable=False)
                with tf.variable_scope("train_op") as scope:
                    optimizer = tf.train.AdamOptimizer(self.learning_rate)

                    pretrain_gradients, pretrain_v = zip(*optimizer.compute_gradients(self.pretrain_loss))
                    pretrain_gradients, _ = tf.clip_by_global_norm(pretrain_gradients, self.clip_value)
                    self.pretrain_train_op = optimizer.apply_gradients(zip(pretrain_gradients, pretrain_v),
                                                              global_step=self.pretrain_global_step)

                    train_gradients, train_v = zip(*optimizer.compute_gradients(self.train_loss))
                    train_gradients, _ = tf.clip_by_global_norm(train_gradients, self.clip_value)
                    self.train_op = optimizer.apply_gradients(zip(train_gradients, train_v),
                                                              global_step=self.train_global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def generate(self, sess, initial_state, output_length):
        outputs = sess.run(self.sample_id,
                           feed_dic={self.initial_state:initial_state})
        return outputs

    def pretrain_step(self, sess, x, z, target_weights, output_length):
        outputs = sess.run([self.pretrain_updates, self.pretrain_loss, self.g_outputs],
                           feed_dict={self.x: x,
                                      self.z:z,
                                      self.target_weights:target_weights,
                                      self.output_length:output_length})
        #_, _, out = outputs
        #print(out)
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)