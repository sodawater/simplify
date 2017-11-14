import tensorflow as tf
import numpy as np
import random
import data_utils
from tensorflow.python.layers import core as layers_core

class Discriminator_RNN():
    def __init__(self, hparams, mode):
        self.from_vocab_size = hparams.from_vocab_size
        self.to_vocab_size = hparams.to_vocab_size
        self.num_units = hparams.num_units
        self.emb_dim = hparams.emb_dim
        self.num_layers = hparams.num_layers
        self.learning_rate = tf.Variable(float(hparams.learning_rate), trainable=False)
        self.clip_value = hparams.clip_value
        self.max_seq_length = 50
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * hparams.decay_factor)

        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.encoder_input_ids = tf.placeholder(dtype=tf.int32, shape=[None,None])
            self.encoder_input_length = tf.placeholder(dtype=tf.int32, shape=[None])
            self.batch_size = tf.size(self.encoder_input_length)
        else:
            self.encoder_input_ids = tf.placeholder(dtype=tf.int32, shape=[1, None])
            self.encoder_input_length = tf.placeholder(dtype=tf.int32, shape=[1])
            self.batch_size = 1

        with tf.variable_scope("embedding") as scope:
            self.embeddings = tf.Variable(self.init_matrix([self.from_vocab_size, self.emb_dim]))

        with tf.variable_scope("projection") as scope:
            self.output_layer = layers_core.Dense(2)

        with tf.variable_scope("encoder") as scope:
            if self.num_layers > 1:
                encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.num_units) for _ in range(self.num_layers)])
                encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.num_units) for _ in range(self.num_layers)])
            else:
                encoder_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.num_units)
                encoder_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.num_units)
            with tf.device("/cpu:0"):
                self.encoder_inputs = tf.nn.embedding_lookup(self.embeddings, self.encoder_input_ids)
            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                                                             cell_bw=encoder_cell_bw,
                                                                             inputs=self.encoder_inputs,
                                                                             dtype=tf.float32,
                                                                             sequence_length=self.encoder_input_length)
            encoder_outputs_fw, encoder_outputs_bw = encoder_outputs
            if self.num_layers > 1:
                fw_c, fw_h = encoder_state[0][self.num_layers - 1]
                bw_c, bw_h = encoder_state[1][self.num_layers - 1]
                c = (fw_c + bw_c) / 2.0
                h = (fw_h + bw_h) / 2.0
                encoder_state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)
                encoder_state = bw_h
            else:
                fw_c, fw_h = encoder_state[0]
                bw_c, bw_h = encoder_state[1]
                c = (fw_c + bw_c) / 2.0
                h = (fw_h + bw_h) / 2.0
                encoder_state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)
                encoder_state = bw_h

        with tf.variable_scope("decoder") as scope:
            self.logits = self.output_layer(encoder_state)

        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, 2])
            self.ypred_for_auc = tf.nn.softmax(self.logits)
            self.answers = tf.arg_max(self.targets, 1)
            #self.target_weights = tf.placeholder(dtype=tf.float32, shape=[None, None])
            with tf.variable_scope("loss") as scope:
                crossent = tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
                self.loss = tf.reduce_sum(crossent) / tf.to_float(self.batch_size)
                self.predictions = tf.argmax(self.logits, 1)
                self.absolute_diff = tf.losses.absolute_difference(labels=self.answers,
                                                                   predictions=self.predictions)

            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.global_step = tf.Variable(0, trainable=False)
                with tf.variable_scope("train_op") as scope:
                    optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    gradients, v = zip(*optimizer.compute_gradients(self.loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)

                    self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                              global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)


    def get_batch(self, data, buckets, bucket_id, batch_size):
        encoder_size = buckets[bucket_id]
        encoder_inputs = []
        targets = []
        source_sequence_length = []
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(batch_size):
            input, label = random.choice(data[bucket_id])

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            pad_size = encoder_size - len(input)
            encoder_inputs.append(input + [data_utils.PAD_ID] * pad_size)
            targets.append(label)
            source_sequence_length.append(len(input))

        return encoder_inputs, targets, source_sequence_length

    def get_pretrain_batch(self, pos_set, neg_set, buckets, bucket_id, batch_size):
        size = buckets[bucket_id]
        encoder_inputs = []
        targets = []
        source_sequence_length = []
        for _ in range(batch_size):
            _, pos_input = random.choice(pos_set[bucket_id])
            pad_size = size - len(pos_input)
            encoder_inputs.append(pos_input + [data_utils.PAD_ID] * pad_size)
            targets.append([0, 1])
            source_sequence_length.append(len(pos_input))

            neg_input = random.choice(neg_set[bucket_id])
            pad_size = size - len(neg_input)
            encoder_inputs.append(neg_input + [data_utils.PAD_ID] * pad_size)
            targets.append([1, 0])
            source_sequence_length.append(len(neg_input))

        return encoder_inputs, targets, source_sequence_length

    def train_step(self, sess, pos_set, neg_set, buckets, bucket_id, batch_size):
        encoder_inputs, targets, source_sequence_length = self.get_pretrain_batch(pos_set, neg_set, buckets, bucket_id, batch_size)
        feed = {self.encoder_input_ids:encoder_inputs,
                self.encoder_input_length:source_sequence_length,
                self.targets:targets}
        loss, diff, global_step, _ = sess.run([self.loss,
                                               self.absolute_diff,
                                               self.global_step,
                                               self.train_op], feed_dict=feed)
        accuracy = 1 - (diff / (2.0 * batch_size))
        return loss, accuracy, global_step
    #def decode(self, sess, encoder_inputs):

