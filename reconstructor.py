import tensorflow as tf
import numpy as np
import random
import data_utils
from tensorflow.python.layers import core as layers_core
import my_dynamic_decode
from tensorflow.python.util import nest

class Reconstructor():
    def __init__(self, hparams, mode):
        self.hparams = hparams
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
            helper_t = tf.contrib.seq2seq.SampleEmbeddingHelper(self.to_embeddings, tf.tile([hparams.GO_ID], [self.batch_size]), hparams.EOS_ID)
            initial_state_t = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=initial_state)
            my_decoder_t = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                         helper=helper_t,
                                                         initial_state=initial_state_t,
                                                         output_layer=self.output_layer)
            decoder_outputs_t, decoder_states_t, final_state_t, decoder_output_len_t = tf.contrib.seq2seq.my_dynamic_decode(my_decoder_t,
                                                                                                                    decoder_cell.state_size.cell_state,
                                                                                                   maximum_iterations=self.max_seq_length * 2,
                                                                                                   swap_memory=True)
            self.sample_id_t = decoder_outputs_t.sample_id
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

        with tf.variable_scope("rolllout") as scope:
            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.rollout_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
                self.rollout_input_length = tf.placeholder(dtype=tf.int32, shape=[None])
                self.rollout_next_id = tf.placeholder(dtype=tf.int32, shape=[None])
                with tf.device("/cpu:0"):
                    rollout_inputs = tf.nn.embedding_lookup(self.to_embeddings, self.rollout_input_ids)
                rollout_cell = decoder_cell
                memory_rollout = tf.concat([encoder_outputs_fw, encoder_outputs_bw], axis=2)
                initial_state_ro = rollout_cell.zero_sate(self.batch_size, tf.float32).clone(cell_state=encoder_state)
                attention_mechanism_ro = tf.contrib.seq2seq.LuongAttention(num_units=self.num_units,
                                                                           memory=memory_rollout,
                                                                           scale=True,
                                                                           memory_sequence_length=self.encoder_input_length)

                helper_ro = tf.contrib.seq2seq.TrainingHelper(rollout_inputs, self.rollout_input_length)
                rollout_decoder = tf.contrib.seq2seq.BasicDecoder(cell=rollout_cell,
                                                                  helper=helper_ro,
                                                                  initial_state=initial_state_ro,
                                                                  output_layer=self.output_layer)
                _, final_state_ro, _ = tf.contrib.seq2seq.dynamic_decode(rollout_decoder,
                                                                         maximum_iterations=self.max_seq_length,
                                                                         swap_memory=True)

                initial_state_MC = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=final_state_ro.cell_state)
                helper_MC = tf.contrib.seq2seq.SampleEmbeddingHelper(self.to_embeddings, self.rollout_next_id, hparams.EOS_ID)
                rollout_decoder_MC = tf.contrib.seq2seq.BasicDecoder(cell=rollout_cell,
                                                                     helper=helper_MC,
                                                                     initial_state=initial_state_MC,
                                                                     output_layer=self.output_layer)
                decoder_output_MC, _, _ = tf.contrib.seq2seq.dynamic_decode(rollout_decoder_MC,
                                                                            maximum_iterations=self.max_seq_length,
                                                                            swap_memory=True)
                self.sample_id_MC = decoder_output_MC.sample_id

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
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None, None])
            with tf.variable_scope("loss") as scope:
                crossent_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.reconstructor_targets, logits=self.reconstructor_logits_t)
                self.loss_t = tf.reduce_sum(crossent_t * self.reconstructor_target_weights) / tf.to_float(self.batch_size)

                crossent_ro = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_targets, logits=self.decoder_logits_pt)
                self.loss_ro = (tf.reduce_sum(crossent_t * self.reconstructor_target_weights) +
                                tf.reduce_sum(crossent_ro * self.rewards)) / tf.to_float(self.batch_size)

                decoder_crossent_pt = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_targets, logits=self.decoder_logits_pt)
                reconstructor_crossent_pt = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.reconstructor_targets, logits=self.reconstructor_logits_pt)
                self.loss_pt = (tf.reduce_sum(decoder_crossent_pt * self.decoder_target_weights)) / tf.to_float(self.batch_size)#+
                                #0.0 * tf.reduce_sum(reconstructor_crossent_pt * self.reconstructor_target_weights)) / tf.to_float(self.batch_size)

            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.global_step_t = tf.Variable(0, trainable=False)
                self.global_step_pt = tf.Variable(0, trainable=False)
                self.global_step_ro = tf.Variable(0, trainable=False)
                with tf.variable_scope("train_op") as scope:
                    optimizer_t = tf.train.AdamOptimizer(self.learning_rate)
                    gradients_t, v_t = zip(*optimizer_t.compute_gradients(self.loss_t))
                    gradients_t, _ = tf.clip_by_global_norm(gradients_t, self.clip_value)

                    #optimizer_pt = tf.train.AdamOptimizer(self.learning_rate)
                    optimizer_pt = tf.train.GradientDescentOptimizer(self.learning_rate)
                    gradients_pt, v_pt = zip(*optimizer_pt.compute_gradients(self.loss_pt))
                    gradients_pt, _ = tf.clip_by_global_norm(gradients_pt, self.clip_value)

                    optimizer_ro = tf.train.GradientDescentOptimizer(self.learning_rate)
                    gradients_ro, v_ro = zip(*optimizer_pt.compute_gradients(self.loss_ro))
                    gradients_ro, _ = tf.clip_by_global_norm(gradients_ro, self.clip_value)

                    self.train_op_t = optimizer_t.apply_gradients(zip(gradients_t, v_t),
                                                                  global_step=self.global_step_t)
                    self.train_op_pt = optimizer_pt.apply_gradients(zip(gradients_pt, v_pt),
                                                              global_step=self.global_step_pt)
                    self.train_op_ro = optimizer_pt.apply_gradients(zip(gradients_ro, v_ro),
                                                                    global_step=self.global_step_ro)



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

    def train_step(self):
        return

    def adversarial_train_step(self, sess_r, sess_d, data, buckets, bucket_id, batch_size, discriminator, rollout_num):
        encoder_in, decoder_in, reconstructor_in = self.get_batch(self, data, buckets, bucket_id, batch_size)
        encoder_inputs, encoder_sequence_length = encoder_in
        decoder_inputs, decoder_targets, decoder_target_weights, decoder_sequence_length = decoder_in
        reconstructor_inputs, reconstructor_targets, reconstructor_target_weights, recon_sequence_length = reconstructor_in
        feed = {self.encoder_input_ids: encoder_inputs,
                self.encoder_input_length: encoder_sequence_length}
        samples = sess_r.run([self.sample_id_t], feed_dict=feed)
        rewards = self.get_reward(sess_r, sess_d, encoder_in, samples, discriminator, rollout_num)

    def get_reward(self, sess_r, sess_d, encoder_in, samples, discriminator, rollout_num):
        encoder_inputs, encoder_sequence_length = encoder_in
        sample_length = []
        max_sample_length = 0
        samples_list = []
        for batch in range(len(encoder_sequence_length)):
            sample = samples[batch].tolist()
            if self.hparams.EOS_ID in sample:
                sample = sample[:sample.index(self.hparams.EOS_ID) + 1]
            length  = len(sample)
            sample_length.append(length)
            samples_list.append(sample)
            if length > max_sample_length:
                max_sample_length = length

        decoder_targets = []
        decoder_inputs = []
        decoder_target_weights = []
        decoder_sequence_length = sample_length
        for batch in range(len(encoder_sequence_length)):
            length = sample_length[batch]
            pad_size = max_sample_length - length
            sample = samples_list[batch]
            input = [self.hparams.GO_ID] + sample[0:length - 1] + pad_size * [self.hparams.PAD_ID]
            target = sample + pad_size * [self.hparams.PAD_ID]
            weight = [1.0] * length + [0.0] * self.hparams.PAD_ID
            decoder_inputs.append(input)
            decoder_targets.append(target)
            decoder_target_weights.append(weight)
        decoder_inputs = np.array(decoder_inputs)
        rewards = []
        for give_num in range(1, max_sample_length):
            feed = {self.rollout_input_ids:decoder_inputs[:,0:give_num],
                    self.rollout_input_length:tf.tile([give_num], [self.batch_size]),
                    self.rollout_next_id:decoder_inputs[:,give_num]}
            for i in range(rollout_num):
                rollout_part = sess_r.run(self.sample_id_MC, feed_dict=feed)
                rollout_samples, origin_length = self.padding(np.concatenate([decoder_inputs[:,0:give_num], rollout_part], axis=1))
                feed_d = {discriminator.encoder_input_ids:rollout_samples,
                          discriminator.encoder_input_length:origin_length}
                ypred_for_auc = sess_d.run(discriminator.ypred_for_auc, feed_dict=feed_d)
                ypred = np.array(item[1] for item in ypred_for_auc)
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[give_num - 1] += ypred
        feed_d = {discriminator.encoder_input_ids:decoder_inputs,
                  discriminator.encoder_input_length:decoder_sequence_length}
        ypred_for_auc = sess_d.run(discriminator.ypred_for_auc, feed_dict=feed_d)
        ypred = np.array(item[1] for item in ypred_for_auc)
        rewards.append(ypred * rollout_num)
        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)
        return rewards
    
    def padding(self, max_length=0):
        return
    #def decode(self, sess, encoder_inputs):

