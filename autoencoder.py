import tensorflow as tf
import numpy as np
import random
import data_utils
from tensorflow.python.layers import core as layers_core

class Autoencoder():
    def __init__(self, hparams, mode):
        self.vocab_size = hparams.from_vocab_size
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
            self.decoder_input_ids = tf.placeholder(dtype=tf.int32, shape=[None,None])
            self.decoder_input_length = tf.placeholder(dtype=tf.int32, shape=[None])
            self.batch_size = tf.size(self.decoder_input_length)
        else:
            self.encoder_input_ids = tf.placeholder(dtype=tf.int32, shape=[1, None])
            self.encoder_input_length = tf.placeholder(dtype=tf.int32, shape=[1])
            self.batch_size = 1

        with tf.variable_scope("embedding") as scope:
            self.embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.emb_dim]))

        with tf.variable_scope("projection") as scope:
            self.output_layer = layers_core.Dense(self.vocab_size)

        with tf.variable_scope("encoder") as scope:
            if self.num_layers > 1:
                encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.num_units) for _ in range(self.num_layers)])
            else:
                encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units)
            with tf.device("/cpu:0"):
                self.encoder_inputs = tf.nn.embedding_lookup(self.embeddings, self.encoder_input_ids)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                               inputs=self.encoder_inputs,
                                                               dtype=tf.float32,
                                                               sequence_length=self.encoder_input_length)

        with tf.variable_scope("decoder") as scope:
            if self.num_layers > 1:
                decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.num_units) for _ in range(self.num_layers)])
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units,
                                                                    memory=encoder_outputs,
                                                                    memory_sequence_length=self.encoder_input_length)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_mechanism,
                                                               attention_layer_size=self.num_units)

            if mode != tf.contrib.learn.ModeKeys.INFER:
                initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_state)
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
                initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_state)
                #initial_state = encoder_state
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                             helper=helper,
                                                             initial_state=initial_state,
                                                             output_layer=self.output_layer)
                decoder_outputs, decoder_state, decoder_output_len = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                                                       maximum_iterations=self.max_seq_length * 2,
                                                                                                       swap_memory=True)
                self.sample_id = tf.unstack(decoder_outputs.sample_id, axis=0)

        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.target_weights = tf.placeholder(dtype=tf.float32, shape=[None, None])
            with tf.variable_scope("loss") as scope:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
                self.loss = tf.reduce_sum(crossent * self.target_weights) / tf.to_float(self.batch_size)

            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.global_step = tf.Variable(0, trainable=False)
                with tf.variable_scope("train_op") as scope:
                    optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    gradients, v = zip(*optimizer.compute_gradients(self.loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)

                    self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                              global_step=self.global_step)
                """
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                gradients, v = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
                self.train_op = optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step)
                """

        self.saver = tf.train.Saver(tf.global_variables())

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)


    def get_batch(self, data, buckets, bucket_id, batch_size):
        encoder_size = buckets[bucket_id]
        encoder_inputs = []
        decoder_inputs = []
        targets = []
        target_weights = []
        source_sequence_length = []
        target_sequence_length = []
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(batch_size):
            encoder_input = random.choice(data[bucket_id])

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            encoder_pad_size = encoder_size - len(encoder_input)
            pad_size = encoder_size - len(encoder_input)
            #print(len(encoder_input))
            encoder_inputs.append(encoder_input +
                                  [data_utils.PAD_ID] * encoder_pad_size)
            targets.append(encoder_input + [data_utils.PAD_ID] * (pad_size))
            decoder_inputs.append([data_utils.GO_ID] + encoder_input[0:len(encoder_input) - 1]
                                  + [data_utils.PAD_ID] * (pad_size))
            target_weights.append([1.0] * (len(encoder_input)) + [0.0] * pad_size)
            source_sequence_length.append(len(encoder_input))
            target_sequence_length.append(encoder_size)
        # Now we create batch-major vectors from the data selected above.
        return encoder_inputs, decoder_inputs, targets, target_weights, source_sequence_length, target_sequence_length

    #def decode(self, sess, encoder_inputs):

