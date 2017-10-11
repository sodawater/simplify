import tensorflow as tf
import numpy as np

def Autoencoder():
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, max_seqlen, start_token, num_layers, use_lstm=True, learning_rate=0.01):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = max_seqlen
        self.start_token = tf.constant([start_token] * batch_size, dtype=tf.int32)
        self.start_token = tf.reshape(self.start_token, shape=[batch_size, 1])
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.ae_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0
        def single_cell():
            return tf.contrib.rnn.GRUCell(hidden_dim)
        if use_lstm:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        encode_cell = single_cell()
        decode_cell = single_cell()
        if num_layers > 1:
            encode_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size,
                                                 None])
        self.target_weights = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                                None])
        with tf.variable_scope('autoencoder'):
            self.ae_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.ae_params.append(self.ae_embeddings)

        with tf.device("/cpu:0"):
            self.encoder_inputs = tf.transpose(tf.nn.embedding_lookup(self.ae_embeddings, self.x),
                                            perm=[0, 1, 2])  # batch_size x seq_length x emb_dim

        hidden_state, final_state = tf.nn.dynamic_rnn(cell=encode_cell, inputs=self.encoder_inputs)
        self.sentence_embedding = final_state

        decoder_x = tf.concat([self.start_token, self.x], 1)
        decoder_inputs = tf.transpose(tf.nn.embedding_lookup(self.ae_embeddings, self.x),
                                      perm=[1,0,2])
        encoder_outputs = [tf.reshape(x, [self.batch_size, 1, hidden_dim]) for x in hidden_state]
        self.attention_state = tf.concat(axis=1, values=encoder_outputs)
        decoder_outputs, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs,
                                                                                    final_state,
                                                                                    self.attention_state,
                                                                                    decode_cell
                                                                                    )
        self.projection_w = tf.get_variable('projection_w', shape=[hidden_dim, num_emb],
                                            initializer=tf.contrib.layers.xavier_initializer())
        self.projection_b = tf.get_variable('projection_b',
                                            initializer=tf.zeros_initializer((self.vocab_size,)))

        projection_w_t = tf.transpose(self.projection_w)
        def loss_function(inputs, labels):
            labels = tf.reshape(labels, (-1, 1))
            return tf.nn.sampled_softmax_loss(projection_w_t,
                                              self.projection_b,
                                              inputs, labels,
                                              100, self.vocab_size)

        self.loss = tf.nn.seq2seq.sequence_loss(self.decoder_outputs,
                                                   decoder_labels,
                                                   label_weights,
                                                   softmax_loss_function=loss_function)




    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)