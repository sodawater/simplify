import tensorflow as tf
import numpy as np
import random
import data_utils
from tensorflow.python.layers import core as layers_core

class Autoencoder():
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, max_seqlen, start_token, num_layers, use_lstm=True, learning_rate=0.01):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = max_seqlen
        #self.start_token = tf.constant([start_token] * batch_size, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.ae_params = []
        self.temperature = 1.0
        self.clip_value = 5.0
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.seq_length = tf.constant([max_seqlen] * batch_size, dtype=tf.int32)

        def single_cell():
            return tf.contrib.rnn.GRUCell(hidden_dim)
        if use_lstm:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        encode_cell = single_cell()
        decode_cell = single_cell()
        if num_layers > 1:
            encode_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

        self.input_ids = tf.placeholder(tf.int32, shape=[None,
                                                         None])
        self.target_weights = tf.placeholder(tf.float32, shape=[None,
                                                                self.sequence_length + 1])
        self.targets = tf.placeholder(tf.int32, shape=[None,
                                                       self.sequence_length + 1])
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_length + 1])
        self.source_sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        with tf.variable_scope('autoencoder'):
            self.ae_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.ae_params.append(self.ae_embeddings)

        with tf.device("/cpu:0"):
            self.encoder_inputs = tf.transpose(tf.nn.embedding_lookup(self.ae_embeddings, self.input_ids),
                                               perm=[0, 1, 2])  # batch_size x seq_length x emb_dim

        self.output_layer = layers_core.Dense(num_emb)
        encode_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        self.length = tf.placeholder(dtype=tf.int32)
#        source_sequence_length = tf.constant([tf.to_int32(self.length)] * batch_size,dtype=tf.int32)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encode_cell,
                                                      inputs=self.encoder_inputs,
                                                      dtype=tf.float32,
                                                      sequence_length=self.source_sequence_length)
        self.sentence_embedding = encoder_state

        #self.length = tf.placeholder(dtype=tf.int32)

        #print(encoder_outputs.ndims)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(hidden_dim, encoder_outputs)
        #decode_cell = tf.contrib.seq2seq.AttentionWrapper(decode_cell, attention_mechanism, attention_layer_size=hidden_dim)
        if num_layers > 1:
            decode_cell = tf.contrib.rnn.MultiRNNCell([decode_cell() for _ in range(num_layers)])
        decode_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim)
        print(self.decoder_inputs)
        decoder_inputs = tf.nn.embedding_lookup(self.ae_embeddings, self.decoder_inputs)
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, self.seq_length)

        my_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decode_cell,
                                                     helper=helper,
                                                     initial_state=encoder_state)
                                                     #initial_state=decode_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state))
        print(decode_cell,my_decoder)



        decoder_outputs, decoder_state, decoder_output_len = tf.contrib.seq2seq.dynamic_decode(my_decoder)
                                                                                               #maximum_iterations=self.sequence_length * 2,
                                                                                               #swap_memory=True)
        self.sample_id = decoder_outputs.sample_id
        self.logits = self.output_layer(decoder_outputs.rnn_output)

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
        self.loss = tf.reduce_sum(crossent * self.target_weights) / tf.to_float(self.batch_size)


        eval_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.ae_embeddings, [data_utils.GO_ID], data_utils.EOS_ID)
        eval_decoder = tf.contrib.seq2seq.BasicDecoder(decode_cell, eval_helper, encoder_state, output_layer=self.output_layer)
        eval_outputs, eval_state, eval_length = tf.contrib.seq2seq.dynamic_decode(eval_decoder,
                                                                                  maximum_iterations=self.sequence_length * 2,
                                                                                  swap_memory=True)
        self.eval_sample_id = eval_outputs.sample_id

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)


        """
        self.projection_w = tf.get_variable('projection_w', shape=[hidden_dim, num_emb],
                                            initializer=tf.contrib.layers.xavier_initializer())
        self.projection_b = tf.get_variable('projection_b', shape=[num_emb],
                                            initializer=tf.zeros_initializer())
        
        
        
        projection_w_t = tf.transpose(self.projection_w)
        def loss_function(logits, labels):
            labels = tf.reshape(labels, (-1, 1))
            return tf.cast(tf.nn.sampled_softmax_loss(weights=tf.cast(projection_w_t,dtype=tf.float32),
                                              biases=tf.cast(self.projection_b,dtype=tf.float32),
                                              labels=labels,
                                              inputs=tf.cast(logits,tf.float32),
                                              num_sampled=100,
                                              num_classes=self.num_emb), dtype=tf.float32)

        self.decoder_outputs = tf.transpose(tf.stack(decoder_outputs), perm=[1,0,2])
        print(self.decoder_outputs,self.targets,self.target_weights)
        self.loss = tf.contrib.seq2seq.sequence_loss(self.decoder_outputs,
                                                   self.targets,
                                                   self.target_weights,
                                                   softmax_loss_function=loss_function)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)

        self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                  global_step=self.global_step)
        """
        self.saver = tf.train.Saver(tf.global_variables())



    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def get_batch(self, data, buckets, bucket_id):
        encoder_size = buckets[bucket_id]
        encoder_inputs = []
        decoder_inputs = []
        targets = []
        batch_weights = []
        source_sequence_length = []
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input = random.choice(data[bucket_id])

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            encoder_pad_size = encoder_size - len(encoder_input)
            pad_size = self.sequence_length - len(encoder_input)
            encoder_inputs.append(encoder_input +
                                  [data_utils.PAD_ID] * encoder_pad_size)
            targets.append(encoder_input + [data_utils.EOS_ID]
                                  [data_utils.PAD_ID] * (pad_size - 1))
            decoder_inputs.append([data_utils.GO_ID] + encoder_input + [data_utils.PAD_ID] * (pad_size - 1))
            batch_weights.append([1.0] * (len(encoder_input)) + [0.0] * pad_size)
            source_sequence_length.append(len(encoder_input))
        # Now we create batch-major vectors from the data selected above.
        return encoder_inputs, decoder_inputs, targets, batch_weights, source_sequence_length

    #def decode(self, sess, encoder_inputs):

