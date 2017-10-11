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
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.ae_params = []
        self.temperature = 1.0
        self.grad_clip = 5.0

        def single_cell():
            return tf.contrib.rnn.GRUCell(hidden_dim)
        if use_lstm:
            def single_cell():
                return tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        cell = single_cell()
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size,
                                                 None])
        self.target_weights = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                                None])
        with tf.variable_scope('autoencoder'):
            self.ae_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.ae_params.append(self.g_embeddings)

        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x),
                                            perm=[0, 1, 2])  # batch_size x seq_length x emb_dim

        self.hidden_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.processed_x)


    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)