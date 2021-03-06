import tensorflow as tf
import numpy as np
import random
import data_utils
from tensorflow.python.layers import core as layers_core

class BiRNN_seq2seq():
    def __init__(self, hparams, mode):
        self.from_vocab_size = hparams.from_vocab_size
        self.to_vocab_size = hparams.to_vocab_size
        self.num_units = hparams.num_units
        self.emb_dim = hparams.emb_dim
        self.num_layers = hparams.num_layers
        self.learning_rate = tf.Variable(float(hparams.learning_rate), trainable=False)
        self.clip_value = hparams.clip_value
        self.max_seq_length = 80
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * hparams.decay_factor)
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.keep_prob = 0.7
        else:
            self.keep_prob = 1.0

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
            self.from_embeddings = tf.Variable(self.init_matrix([self.from_vocab_size, self.emb_dim]))
            self.to_embeddings = tf.Variable(self.init_matrix([self.to_vocab_size, self.emb_dim]))
            #self.to_embeddings = self.from_embeddings
        with tf.variable_scope("projection") as scope:
            self.output_layer = layers_core.Dense(self.to_vocab_size, use_bias=False)

        with tf.variable_scope("encoder") as scope:
            if self.num_layers > 1:
                encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.num_units),
                                                                                             input_keep_prob=self.keep_prob) for _ in range(self.num_layers)])
                encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.num_units),
                                                                                             input_keep_prob=self.keep_prob) for _ in range(self.num_layers)])
            else:
                encoder_cell_fw = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.num_units),
                                                                output_keep_prob=self.keep_prob)
                encoder_cell_bw = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.num_units),
                                                                output_keep_prob=self.keep_prob)
            with tf.device("/cpu:0"):
                self.encoder_inputs = tf.nn.embedding_lookup(self.from_embeddings, self.encoder_input_ids)
            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                                                             cell_bw=encoder_cell_bw,
                                                                             inputs=self.encoder_inputs,
                                                                             dtype=tf.float32,
                                                                             sequence_length=self.encoder_input_length)
            encoder_outputs_fw, encoder_outputs_bw = encoder_outputs
            if self.num_layers > 1:
                states = []
                for layer_id in range(self.num_layers):
                    fw_c, fw_h = encoder_state[0][layer_id]
                    bw_c, bw_h = encoder_state[1][layer_id]
                    c = (fw_c + bw_c) / 2.0
                    #c = bw_c
                    #h = bw_h
                    h = (fw_h + bw_h) / 2.0
                    state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)
                    states.append(state)
                encoder_state = tuple(states)
            else:
                fw_c, fw_h = encoder_state[0]
                bw_c, bw_h = encoder_state[1]
                c = (fw_c + bw_c) / 2.0
                h = (fw_h + bw_h) / 2.0
                encoder_state = tf.contrib.rnn.LSTMStateTuple(c=c, h=h)

        with tf.variable_scope("decoder") as scope:
            if self.num_layers > 1:
                decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.num_units),
                                                                                          output_keep_prob=self.keep_prob) for _ in range(self.num_layers)])
            memory = tf.concat([encoder_outputs_fw, encoder_outputs_bw], axis=2)
            initial_state = encoder_state
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units,
                                                                    memory=memory,
                                                                    scale=True,
                                                                    memory_sequence_length=self.encoder_input_length)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_mechanism,
                                                               alignment_history=True,
                                                               attention_layer_size=self.num_units)

            if mode != tf.contrib.learn.ModeKeys.INFER:
                initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=initial_state)
                with tf.device("/cpu:0"):
                    decoder_inputs = tf.nn.embedding_lookup(self.to_embeddings, self.decoder_input_ids)
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
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.to_embeddings,
                                                                  tf.fill([self.batch_size], hparams.GO_ID),
                                                                  hparams.EOS_ID)
                initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=initial_state)
                #initial_state = encoder_state
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                             helper=helper,
                                                             initial_state=initial_state,
                                                             output_layer=self.output_layer)
                decoder_outputs, decoder_state, decoder_output_len = tf.contrib.seq2seq.dynamic_decode(my_decoder,
                                                                                                       maximum_iterations=self.max_seq_length * 2,
                                                                                                       swap_memory=True)
                alignment_history = (decoder_state.alignment_history.stack())
                print(decoder_state)
                print(alignment_history)
                self.alignment_history = tf.unstack(tf.argmax(alignment_history, axis=2), axis=1)
                print(self.alignment_history)
                self.tmp = alignment_history
                # self.tmp = tf.summary.image("images", 255 * tf.expand_dims(tf.transpose(alignment_history,
                #                                                                         [1,2,0]),
                #                                                            -1))
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
                    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                    #optimizer = tf.train.AdamOptimizer(self.learning_rate)
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
        encoder_size, decoder_size = buckets[bucket_id]
        encoder_inputs = []
        decoder_inputs = []
        targets = []
        target_weights = []
        source_sequence_length = []
        target_sequence_length = []
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            encoder_pad_size = encoder_size - len(encoder_input)
            decoder_pad_size = decoder_size -len(decoder_input)
            #print(len(encoder_input))
            encoder_inputs.append(encoder_input +
                                  [data_utils.PAD_ID] * encoder_pad_size)
            targets.append(decoder_input + [data_utils.PAD_ID] * (decoder_pad_size))
            decoder_inputs.append([data_utils.GO_ID] + decoder_input[0:len(decoder_input) - 1]
                                  + [data_utils.PAD_ID] * (decoder_pad_size))
            target_weights.append([1.0 / len(decoder_input)] * (len(decoder_input)) + [0.0] * (decoder_pad_size))
            source_sequence_length.append(len(encoder_input))
            target_sequence_length.append(decoder_size)
        # Now we create batch-major vectors from the data selected above.
        return encoder_inputs, decoder_inputs, targets, target_weights, source_sequence_length, target_sequence_length

    #def decode(self, sess, encoder_inputs):

