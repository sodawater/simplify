import tensorflow as tf
import numpy as np
import random
import data_utils
from tensorflow.python.layers import core as layers_core
import my_dynamic_decode
from tensorflow.python.util import nest

class Reconstructor():
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

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.keep_prob = hparams.keep_prob
        else:
            self.keep_prob = 1.0

        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.encoder_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.encoder_input_length = tf.placeholder(dtype=tf.int32, shape=[None])
            self.decoder_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.decoder_input_length = tf.placeholder(dtype=tf.int32, shape=[None])
            self.reconstructor_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.reconstructor_input_length = tf.placeholder(dtype=tf.int32, shape=[None])
            self.batch_size = tf.size(self.reconstructor_input_length)
        else:
            self.encoder_input_ids = tf.placeholder(dtype=tf.int32, shape=[1, None])
            self.encoder_input_length = tf.placeholder(dtype=tf.int32, shape=[1])
            self.batch_size = 1

        with tf.variable_scope("embedding") as scope:
            self.from_embeddings = tf.Variable(self.init_matrix([self.from_vocab_size, self.emb_dim]))
            self.to_embeddings = tf.Variable(self.init_matrix([self.to_vocab_size, self.emb_dim]))

        with tf.variable_scope("projection") as scope:
            self.output_layer = layers_core.Dense(self.to_vocab_size, use_bias=False)
            self.output_layer2 = layers_core.Dense(self.from_vocab_size, use_bias=False)

        def _get_cell(num_units):
            return tf.contrib.rnn.DropoutWrappper(tf.contrib.rnn.BasicLSTMCell(num_units),
                                                  input_keep_prob=self.keep_prob)
        with tf.variable_scope("encoder") as scope:
            if self.num_layers > 1:
                encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([_get_cell(self.num_units) for _ in range(self.num_layers)])
                encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([_get_cell(self.num_units) for _ in range(self.num_layers)])
            else:
                encoder_cell_fw = _get_cell(self.num_units)
                encoder_cell_bw = _get_cell(self.num_units)
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
                decoder_cell = tf.contrib.rnn.MultiRNNCell([_get_cell(self.num_units) for _ in range(self.num_layers)])
            else:
                decoder_cell = _get_cell(self.num_units)
            memory_t = tf.concat([encoder_outputs_fw, encoder_outputs_bw], axis=2)
            initial_state = encoder_state

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units,
                                                                    memory=memory_t,
                                                                    scale=True,
                                                                    memory_sequence_length=self.encoder_input_length)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_mechanism,
                                                               alignment_history=True,
                                                               attention_layer_size=self.num_units)
            #########train without targets ###########
            helper_t = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.to_embeddings, tf.tile([hparams.GO_ID], [self.batch_size]), hparams.EOS_ID)
            initial_state_t = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=initial_state)
            my_decoder_t = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                         helper=helper_t,
                                                         initial_state=initial_state_t,
                                                         output_layer=self.output_layer)
            decoder_outputs_t, decoder_states_t, final_state_t, decoder_output_len_t = tf.contrib.seq2seq.my_dynamic_decode(my_decoder_t,
                                                                                                                    decoder_cell.state_size.cell_state,
                                                                                                   maximum_iterations=self.max_seq_length * 2,
                                                                                                   swap_memory=True)
            #########################################

            if mode != tf.contrib.learn.ModeKeys.INFER:
                with tf.device("/cpu:0"):
                    decoder_inputs = tf.nn.embedding_lookup(self.to_embeddings, self.decoder_input_ids)
                ##################pretain with targets#####################
                helper_pt = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, self.decoder_input_length)
                initial_state_pt = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=initial_state)
                my_decoder_pt = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                helper=helper_pt,
                                                                initial_state=initial_state_pt,
                                                                output_layer=self.output_layer)

                decoder_outputs_pt, decoder_states_pt, final_state_pt, decoder_output_len_pt = tf.contrib.seq2seq.my_dynamic_decode(my_decoder_pt,
                                                                                                                                    decoder_cell.state_size.cell_state,
                                                                                                     maximum_iterations=self.max_seq_length * 2,
                                                                                                     swap_memory=True)
                self.sample_id_pt = decoder_outputs_pt.sample_id
                self.decoder_logits_pt = decoder_outputs_pt.rnn_output
                #############################################################
            else:
                alignment_history = (final_state_t.alignment_history.stack())
                self.a = alignment_history
                self.alignment_history_t = tf.unstack(tf.argmax(alignment_history, axis=2), axis=1)
                self.sample_id_t = decoder_outputs_t.sample_id

        with tf.variable_scope("reconstructor") as scope:
            if self.num_layers > 1:
                reconstructor_cell = tf.contrib.rnn.MultiRNNCell(
                    [_get_cell(self.num_units) for _ in range(self.num_layers)])
            else:
                reconstructor_cell = _get_cell(self.num_units)
            ##############train without targets###################
            memory_t = decoder_states_t[self.num_layers - 1].h
            attention_mechanism_t = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units,
                                                                        memory=memory_t,
                                                                        scale=True,
                                                                        memory_sequence_length=decoder_output_len_t)
            reconstructor_cell_t = tf.contrib.seq2seq.AttentionWrapper(reconstructor_cell,
                                                                         attention_mechanism_t,
                                                                         alignment_history=True,
                                                                         attention_layer_size=self.num_units)
            initial_state_t = reconstructor_cell_t.zero_state(self.batch_size, tf.float32).clone(
                cell_state=final_state_t.cell_state)
            #######################################################

            ##################pretrain with targets################
            if mode != tf.contrib.learn.ModeKeys.INFER:
                memory_pt = decoder_states_pt[self.num_layers - 1].h
                attention_mechanism_pt = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units,
                                                                    memory=memory_pt,
                                                                    scale=True,
                                                                    memory_sequence_length=decoder_output_len_pt)
                reconstructor_cell_pt = tf.contrib.seq2seq.AttentionWrapper(reconstructor_cell,
                                                                     attention_mechanism_pt,
                                                                     alignment_history=True,
                                                                     attention_layer_size=self.num_units)
                initial_state_pt = reconstructor_cell_pt.zero_state(self.batch_size, tf.float32).clone(
                    cell_state=final_state_pt.cell_state)
            #########################################################

            ##############train of pretrain##########################
            if mode != tf.contrib.learn.ModeKeys.INFER:
                with tf.device("/cpu:0"):
                    reconstructor_inputs = tf.nn.embedding_lookup(self.from_embeddings, self.reconstructor_input_ids)

                helper_t = tf.contrib.seq2seq.TrainingHelper(reconstructor_inputs, self.reconstructor_input_length)
                my_reconstructor_t = tf.contrib.seq2seq.BasicDecoder(cell=reconstructor_cell_t,
                                                             helper=helper_t,
                                                             initial_state=initial_state_t,
                                                             output_layer=self.output_layer2
                                                             )
                reconstructor_outputs_t, reconstructor_state_t, reconstructor_output_len = tf.contrib.seq2seq.dynamic_decode(my_reconstructor_t,
                                                                                                       maximum_iterations=self.max_seq_length * 2,
                                                                                                       swap_memory=True)
                self.reconstructor_sample_id_t = reconstructor_outputs_t.sample_id
                self.reconstructor_logits_t = reconstructor_outputs_t.rnn_output


                helper_pt = tf.contrib.seq2seq.TrainingHelper(reconstructor_inputs, self.reconstructor_input_length)
                my_reconstructor_pt = tf.contrib.seq2seq.BasicDecoder(cell=reconstructor_cell_pt,
                                                                      helper=helper_pt,
                                                                      initial_state=initial_state_pt,
                                                                      output_layer=self.output_layer2)
                reconstructor_outputs_pt, reconstructor_state_pt, reconstructor_output_len_pt = tf.contrib.seq2seq.dynamic_decode(my_reconstructor_pt,
                                                                                                                                  maximum_iterations=self.max_seq_length * 2,
                                                                                                                                  swap_memory=True)
                self.reconstruct_sample_id_pt = reconstructor_outputs_pt.sample_id
                self.reconstructor_logits_pt = reconstructor_outputs_pt.rnn_output
            #########################infer sources#################################
            else:
                helper_t = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.from_embeddings,
                                                                  tf.tile([hparams.GO_ID], [self.batch_size]),
                                                                  hparams.EOS_ID)

                my_reconstructor_t = tf.contrib.seq2seq.BasicDecoder(cell=reconstructor_cell_t,
                                                                   helper=helper_t,
                                                                   initial_state=initial_state_t,
                                                                   output_layer=self.output_layer2
                                                                   )
                reconstructor_outputs_t, reconstructor_state_t, reconstructor_output_len = tf.contrib.seq2seq.dynamic_decode(
                    my_reconstructor_t,
                    maximum_iterations=self.max_seq_length * 2,
                    swap_memory=True)
                self.reconstructor_sample_id_t = reconstructor_outputs_t.sample_id



        if mode != tf.contrib.learn.ModeKeys.INFER:
            self.decoder_targets = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.decoder_target_weights = tf.placeholder(dtype=tf.float32, shape=[None, None])
            self.reconstructor_targets = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.reconstructor_target_weights = tf.placeholder(dtype=tf.float32, shape=[None, None])

            with tf.variable_scope("loss") as scope:
                crossent_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.reconstructor_targets, logits=self.reconstructor_logits_t)
                self.loss_t = tf.reduce_sum(crossent_t * self.reconstructor_target_weights) / tf.to_float(self.batch_size)

                decoder_crossent_pt = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_targets, logits=self.decoder_logits_pt)
                reconstructor_crossent_pt = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.reconstructor_targets, logits=self.reconstructor_logits_pt)
                self.loss_pt = (tf.reduce_sum(decoder_crossent_pt * self.decoder_target_weights)) / tf.to_float(self.batch_size)#+
                                #0.0 * tf.reduce_sum(reconstructor_crossent_pt * self.reconstructor_target_weights)) / tf.to_float(self.batch_size)

            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.global_step_t = tf.Variable(0, trainable=False)
                self.global_step_pt = tf.Variable(0, trainable=False)
                with tf.variable_scope("train_op") as scope:
                    optimizer_t = tf.train.AdamOptimizer(self.learning_rate)
                    gradients_t, v_t = zip(*optimizer_t.compute_gradients(self.loss_t))
                    gradients_t, _ = tf.clip_by_global_norm(gradients_t, self.clip_value)

                    #optimizer_pt = tf.train.AdamOptimizer(self.learning_rate)
                    optimizer_pt = tf.train.GradientDescentOptimizer(self.learning_rate)
                    gradients_pt, v_pt = zip(*optimizer_pt.compute_gradients(self.loss_pt))
                    gradients_pt, _ = tf.clip_by_global_norm(gradients_pt, self.clip_value)

                    self.train_op_t = optimizer_t.apply_gradients(zip(gradients_t, v_t),
                                                                  global_step=self.global_step_t)
                    self.train_op_pt = optimizer_pt.apply_gradients(zip(gradients_pt, v_pt),
                                                              global_step=self.global_step_pt)



        self.saver = tf.train.Saver(tf.global_variables())

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)


    def get_batch(self, data, buckets, bucket_id, batch_size):
        encoder_size, decoder_size = buckets[bucket_id]
        reconstructor_size = encoder_size + 1
        encoder_inputs = []
        decoder_inputs = []
        reconstructor_inputs = []
        decoder_targets = []
        decoder_target_weights = []
        reconstructor_targets = []
        reconstructor_target_weights = []
        encoder_sequence_length = []
        decoder_sequence_length = []
        recon_sequence_length = []
        for _ in range(batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            reconstructor_input = encoder_input
            encoder_pad_size = encoder_size - len(encoder_input)
            decoder_pad_size = decoder_size - len(decoder_input)
            reconstructor_pad_size = reconstructor_size - len(reconstructor_input) - 1
            encoder_inputs.append(encoder_input +
                                  [data_utils.PAD_ID] * encoder_pad_size)
            decoder_inputs.append([data_utils.GO_ID] + decoder_input[0:len(decoder_input)-1] + [data_utils.PAD_ID] * decoder_pad_size)
            reconstructor_inputs.append([data_utils.GO_ID] + reconstructor_input + [data_utils.PAD_ID] * reconstructor_pad_size)

            decoder_targets.append(decoder_input + [data_utils.PAD_ID] * decoder_pad_size)
            reconstructor_targets.append(reconstructor_input + [data_utils.EOS_ID] + [data_utils.PAD_ID] * reconstructor_pad_size)

            decoder_target_weights.append([1.0] * len(decoder_input) + [0.0] * decoder_pad_size)
            reconstructor_target_weights.append([1.0] * len(reconstructor_input) + [1.0] + [0.0] * reconstructor_pad_size)

            encoder_sequence_length.append(len(encoder_input))
            decoder_sequence_length.append(decoder_size)
            recon_sequence_length.append(reconstructor_size)
        encoder_in = (encoder_inputs, encoder_sequence_length)
        decoder_in = (decoder_inputs, decoder_targets, decoder_target_weights, decoder_sequence_length)
        reconstructor_in = (reconstructor_inputs, reconstructor_targets, reconstructor_target_weights, recon_sequence_length)
        #print(len(decoder_inputs[0]), len(decoder_targets[0]))
        return encoder_in, decoder_in, reconstructor_in

    #def decode(self, sess, encoder_inputs):

