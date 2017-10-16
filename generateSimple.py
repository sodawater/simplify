from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
#from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from autoencoder import Autoencoder
import argparse
import copy
import collections


tf.app.flags.DEFINE_integer("total_batch", 5000, "Number of batches.")
tf.app.flags.DEFINE_integer("ae_epoch_num", 100, "train epoch num for ae")
tf.app.flags.DEFINE_integer("pretrain_epoch_gen", 50, "Pretrain epoch num for generator.")
tf.app.flags.DEFINE_integer("pretrain_epoch_dis", 50, "Pretrain epoch num for discriminator.")
tf.app.flags.DEFINE_integer("cnn_seq_len", 25, "Length of the sequence fed to CNN.")
tf.app.flags.DEFINE_integer("generated_num", 10000, "Number of samples generated by generator.")

FLAGS = None

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_dir", type=str, default="/data/wtm/data/wikilarge/", help="Data directory")
    parser.add_argument("--train_dir", type=str, default="/data/wtm/data/wikilarge/model/GAN/", help="Training directory")
    parser.add_argument("--from_train_data", type=str, default="/data/wtm/data/wikilarge/wiki.full.aner.ori.train.src",
                        help="Training data_src path")
    parser.add_argument("--to_train_data", type=str, default="/data/wtm/data/wikilarge/wiki.full.aner.ori.train.dst",
                        help="Training data_dst path")
    parser.add_argument("--from_valid_data", type=str, default="/data/wtm/data/wikilarge/wiki.full.aner.ori.valid.src",
                        help="Valid data_src path")
    parser.add_argument("--to_valid_data", type=str, default="/data/wtm/data/wikilarge/wiki.full.aner.ori.valid.dst",
                        help="Valid data_dst path")
    parser.add_argument("--ae_ckpt_path", type=str, default="data/wtm/data/wikilarge/model/ae/", help="ae model checkpoint path")

    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on the size of training data (0: no limit)")
    parser.add_argument("--attention", type=str, default="", help="""\
          luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
          attention\
          """)
    parser.add_argument("--from_vocab_size", type=int, default=50000, help="NormalWiki vocabulary size")
    parser.add_argument("--to_vocab_size", type=int, default=30000, help="SimpleWiki vocabulary size")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in the model")
    parser.add_argument("--num_units", type=int, default=256, help="Size of each model layer")
    parser.add_argument("--emb_dim", type=int, default=256, help="Dimension of word embedding")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use during training")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.99, help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=0.5, help="Learning rate")



# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(10, 5), (15, 10), (25, 20), (50, 40)]

_target_buckets = [5, 10, 20, 40]
_source_buckets = [10, 15, 25, 50]

def read_data_pair(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

def read_data_simple(path, max_size=None):
    data_set = [[] for _ in _target_buckets]
    with tf.gfile.GFile(path, mode="r") as target_file:
        target = target_file.readline()
        counter = 0
        while target and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            target_ids = [int(x) for x in target.split()]
            target_ids.append(data_utils.EOS_ID)
            for bucket_id, (target_size) in enumerate(_target_buckets):
                if len(target_ids) < target_size:
                    data_set[bucket_id].append(target_ids)
                    break
            target = target_file.readline()
    return data_set

def read_data_normal(path, max_size=None):
    data_set = [[] for _ in _source_buckets]
    with tf.gfile.GFile(path, mode="r") as source_file:
        source = source_file.readline()
        counter = 0
        while source and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            source_ids.append(data_utils.EOS_ID)
            for bucket_id, (source_size) in enumerate(_source_buckets):
                if len(source_ids) < source_size:
                    data_set[bucket_id].append(source_ids)
                    break
            source = source_file.readline()
    return data_set

def read_data_gen(path, max_size=None):
    data_set = [[] for _ in _target_buckets]
    with tf.gfile.GFile(path, mode="r") as target_file:
        target = target_file.readline()
        counter = 0
        while target and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            target_ids = [int(x) for x in target.split()]
            target_ids.append(data_utils.EOS_ID)
            for bucket_id, (target_size) in enumerate(_target_buckets):
                if len(target_ids) < target_size:
                    data_set[bucket_id].append([target_ids])
                    break
            target = target_file.readline()
    return data_set

def create_model_generator(session, forward_only):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  generator = Generator(FLAGS.to_vocab_size,
                        FLAGS.batch_size,
                        FLAGS.size,
                        FLAGS.size,
                        50,
                        data_utils.GO_ID,)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + "/generator/")
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      generator.saver.restore(session, ckpt.model_checkpoint_path)
      return generator, False
  else:
      print("Created model with fresh parameters.")
      session.run(tf.global_variables_initializer())
      return generator, True
  return generator

def create_model_discriminator(session, forward_only):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
    dis_dropout_keep_prob = 0.75
    dis_l2_reg_lambda = 0.2
    discriminator = Discriminator(FLAGS.batch_size,
                                  30,
                                  2,
                                  FLAGS.to_vocab_size,
                                  FLAGS.size,
                                  filter_sizes=dis_filter_sizes,
                                  num_filters=dis_num_filters,
                                  l2_reg_lambda=dis_l2_reg_lambda
                                  )
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + "/generator/")
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        discriminator.saver.restore(session, ckpt.model_checkpoint_path)
        return discriminator, False
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return discriminator, True
    return discriminator

class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph", "model"))):
  pass

class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model"))):
  pass

class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model"))):
  pass

def create_model_autoencoder(hparams):
    print("Creating auto-encoder...")
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = Autoencoder(hparams, tf.contrib.learn.ModeKeys.TRAIN)

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_model = Autoencoder(hparams, tf.contrib.learn.ModeKeys.EVAL)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_model = Autoencoder(hparams, tf.contrib.learn.ModeKeys.INFER)
    return TrainModel(graph=train_graph, model=train_model), EvalModel(graph=eval_graph, model=eval_model), InferModel(
        graph=infer_graph, model=infer_model)

    """
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + "/autoencoder/")
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        autoencoder.saver.restore(session, ckpt.model_checkpoint_path)
        return autoencoder, False
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return autoencoder, True
    return autoencoder
    """
def generate_samples(sess, trainable_model, batch_size, buckets_size, buckets, output_file):
    # Generate Samples
    generated_samples = []
    for i in range(buckets):
        for _ in range(buckets_size[i] / batch_size):
            z = np.random.rand(batch_size, trainable_model.hidden_dim) * 2 - [[1] * trainable_model.hidden_dim] * batch_size
            generated_samples.extend(trainable_model.generate(sess, z, buckets[i]))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def decode_sen(outputs):
    to_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.to" % FLAGS.to_vocab_size)
    _, rev_to_vocab = data_utils.initialize_vocabulary(to_vocab_path)
    for sentence in outputs:
        sentence = sentence.tolist()
        if data_utils.EOS_ID in sentence:
            sentence = sentence[:sentence.index(data_utils.EOS_ID)]
        print(" ".join([tf.compat.as_str(rev_to_vocab[word]) for word in sentence]))

def train(from_train, to_train, from_dev, to_dev):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print("Creating generator...")
        generator, if_new_gen = create_model_generator(sess, False)
        print("Creating discriminator...")
        discriminator, if_new_dis = create_model_discriminator(sess, False)
        dev_set = read_data_pair(from_dev, to_dev)
        train_set = read_data_pair(from_train, to_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]
        train_set_sim = read_data_simple(to_train, FLAGS.max_train_data_size)
        pretrain_bucket_sizes_sim = [len(train_set_sim[b]) for b in range(len(_target_buckets))]
        pretrain_total_size_sim = float(sum(pretrain_bucket_sizes_sim))
        pretrain_buckets_scale_sim = [sum(pretrain_bucket_sizes_sim[:i + 1]) / pretrain_total_size_sim
                                      for i in range(len(pretrain_bucket_sizes_sim))]
        if if_new_gen:
            print("Pre-training generator...")
            for epoch in range(FLAGS.pretrain_epoch_gen):
                pretrain_g_loss = []
                for it in range(int(pretrain_total_size_sim / FLAGS.batch_size)):
                    random_number_01 = np.random.random_sample()
                    bucket_id = min([i for i in range(len(pretrain_buckets_scale_sim))
                                     if pretrain_buckets_scale_sim[i] > random_number_01])
                    decoder_inputs, target_weights = generator.get_batch_simple(train_set_sim, _target_buckets, bucket_id)
                    #z = np.random.rand(FLAGS.batch_size, generator.hidden_dim) * 2 - [[1] * generator.hidden_dim] * FLAGS.batch_size
                    z = np.zeros(shape=[FLAGS.batch_size, generator.hidden_dim], dtype=np.int32)
                    output_length = _target_buckets[bucket_id]
                    _, g_loss, outputs = generator.pretrain_step(sess, decoder_inputs, z, target_weights, output_length)
                    pretrain_g_loss.append(g_loss)
                    if it % 50 == 0:
                        decode_sen(outputs)
                        print(np.mean(pretrain_g_loss))
                print(np.mean(pretrain_g_loss))

        if if_new_dis:
            print("Pre-training discriminator...")
            for epoch in range(FLAGS.pretrain_epoch_dis):
                generate_samples(sess, generator, FLAGS.batch_size, pretrain_bucket_sizes_sim, _target_buckets, FLAGS.generated_file)
                train_set_gen = read_data_gen(FLAGS.generated_file, FLAGS.max_train_data_size)
                pretrain_bucket_sizes_gen = [len(train_set_gen[b]) for b in range(len(_target_buckets))]
                pretrain_total_size_gen = float(sum(pretrain_bucket_sizes_gen))
                pretrain_buckets_scale_gen = [sum(pretrain_bucket_sizes_gen[:i + 1]) / pretrain_total_size_gen
                                      for i in range(len(pretrain_bucket_sizes_gen))]
                for _ in range(int(pretrain_total_size_gen / FLAGS.batch_size)):
                    random_number_02 = np.random.random_sample()
                    bucket_id = min([i for i in range(len(pretrain_buckets_scale_gen))
                                     if pretrain_buckets_scale_gen[i] > random_number_02])
                    positive_inputs, positive_labels = discriminator.get_batch_simple(train_set_sim, _target_buckets, bucket_id, [0, 1])
                    negative_inputs, negative_labels = discriminator.get_batch_simple(train_set_gen, _target_buckets, bucket_id, [1, 0])
                    inputs = np.array(positive_inputs + negative_inputs)
                    labels = np.array(positive_labels + negative_labels)
                    shuffle_indices = np.random.permutation(np.arange(len(labels)))
                    inputs = inputs[shuffle_indices]
                    labels = labels[shuffle_indices]
                    feed1 = {discriminator.input_x:inputs[:FLAGS.cnn_seq_len],
                             discriminator.input_y:labels[:FLAGS.cnn_seq_len],
                             discriminator.dropout_keep_prob:FLAGS.dis_dropout_keep_prob}
                    feed2 = {discriminator.input_x:inputs[FLAGS.cnn_seq_len:],
                             discriminator.input_y:labels[FLAGS.cnn_seq_len:],
                             discriminator.dropout_keep_prob:FLAGS.dis_dropout_keep_prob}
                    _ = sess.run(discriminator.train_op, feed1)
                    _ = sess.run(discriminator.train_op, feed2)

        print("Adversarial training...")
        rollout = Rollout(0.8)
        for total_batch in range(FLAGS.total_batch):
            for _ in range(1):
                random_number_03 = np.random.random_sample()
                bucket_id = min([i for i in range(len(pretrain_buckets_scale_sim))
                                 if pretrain_buckets_scale_sim[i] > random_number_03])
                z = np.random.rand(FLAGS.batch_size, generator.hidden_dim) * 2 - [[1] * generator.hidden_dim] * FLAGS.batch_size
                output_length = _target_buckets[bucket_id]
                samples = generator.generate(sess, z, output_length)
                rewards = rollout.get_reward(sess, samples, 16, discriminator)
                feed = {generator.x:samples,
                        generator.reward:rewards,
                        }
                _ = sess.run(generator.g_updates, feed_dict=feed)

            rollout.update_params()

            for _ in range(300):
                random_number_04 = np.random.random_sample()
                bucket_id = min([i for i in range(len(pretrain_buckets_scale_sim))
                                 if pretrain_buckets_scale_sim[i] > random_number_04])
                z = np.random.rand(FLAGS.batch_size, generator.hidden_dim) * 2 - [[1] * generator.hidden_dim] * FLAGS.batch_size
                output_length = _target_buckets[bucket_id]
                samples = generator.generate(sess, z, output_length)
                positive_inputs, positive_labels = discriminator.get_batch_simple(train_set_sim, _target_buckets,
                                                                                  bucket_id, [0, 1])
                negative_inputs = []
                for sample in samples:
                    pad_size = FLAGS.cnn_seq_len - len(input) - 1
                    if pad_size >= 0 :
                        negative_inputs.append([data_utils.GO_ID] + sample + [data_utils.PAD_ID] * pad_size)
                    else:
                        negative_inputs.append([data_utils.GO_ID] + sample[0: FLAGS.cnn_seq_len - 1])
                negative_labels = [[1, 0] for _ in negative_inputs]
                inputs = np.array(positive_inputs + negative_inputs)
                labels = np.array(positive_labels + negative_labels)
                shuffle_indices = np.random.permutation(np.arange(len(labels)))
                inputs = inputs[shuffle_indices]
                labels = labels[shuffle_indices]
                feed1 = {discriminator.input_x: inputs[:FLAGS.cnn_seq_len],
                         discriminator.input_y: labels[:FLAGS.cnn_seq_len],
                         discriminator.dropout_keep_prob: FLAGS.dis_dropout_keep_prob}
                feed2 = {discriminator.input_x: inputs[FLAGS.cnn_seq_len:],
                         discriminator.input_y: labels[FLAGS.cnn_seq_len:],
                         discriminator.dropout_keep_prob: FLAGS.dis_dropout_keep_prob}
                _ = sess.run(discriminator.train_op, feed1)
                _ = sess.run(discriminator.train_op, feed2)

def train_ae(hparams):
    ae_hparams = copy.deepcopy(hparams)
    ae_hparams.add_hparam(name="num_train_epoch", value=100)
    train_model, eval_model, infer_model = create_model_autoencoder(ae_hparams)
    config =tf.ConfigProto()
    config.gpu_options.allow_growth = True
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + "/autoencoder/")
    global_step = 0
    with train_model.graph.as_default():
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            train_model.model.saver.restore(train_sess, ckpt.model_checkpoint_path)
            global_step = train_model.model.global_step.eval(session=train_sess)
        else:
            train_sess.run(tf.global_variables_initializer())

    train_set_nor = read_data_normal(ae_hparams.from_train, ae_hparams.max_train_data_size)
    train_bucket_sizes_nor = [len(train_set_nor[b]) for b in range(len(_source_buckets))]
    train_total_size_nor = float(sum(train_bucket_sizes_nor))
    train_buckets_scale_nor = [sum(train_bucket_sizes_nor[:i + 1]) / train_total_size_nor
                               for i in range(len(train_bucket_sizes_nor))]
    num_train_steps = hparams.num_train_epoch * int(train_total_size_nor / FLAGS.batch_size)
    while global_step < num_train_steps:
        random_number_05 = np.random.random_sample()
        bucket_id = min([i for i in range(len(train_buckets_scale_nor))
                         if train_buckets_scale_nor[i] > random_number_05])
        encoder_inputs, decoder_inputs, targets, target_weights, source_sequence_length, target_sequence_length = train_model.model.get_batch(
            train_set_nor, _source_buckets,
            bucket_id, ae_hparams.batch_size)
        feed = {train_model.model.encoder_input_ids:encoder_inputs,
                train_model.model.encoder_input_length:source_sequence_length,
                train_model.model.decoder_input_ids:decoder_inputs,
                train_model.model.decoder_input_length:target_sequence_length,
                train_model.model.targets:targets,
                train_model.model.target_weights:target_weights}
        loss, _, global_step= train_sess.run([train_model.model.loss, train_model.model.train_op, train_model.model.global_step], feed_dict=feed)
        print(loss)
        #if global_step % hparams.steps_per_eval:
        #    continue


    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    ae_hparams = copy.deepcopy(hparams)
    with tf.Session(config=config) as sess:
        print("Creating auto-encoder...")
        autoencoder, if_new_ae = create_model_autoencoder(sess)
        train_set_nor = read_data_normal(ae_hparams.from_train, FLAGS.max_train_data_size)
        train_bucket_sizes_nor = [len(train_set_nor[b]) for b in range(len(_source_buckets))]
        train_total_size_nor = float(sum(train_bucket_sizes_nor))
        train_buckets_scale_nor = [sum(train_bucket_sizes_nor[:i + 1]) / train_total_size_nor
                                      for i in range(len(train_bucket_sizes_nor))]
        if if_new_ae:
            print("Training auto-encoder...")
            for epoch in range(FLAGS.ae_epoch_num):
                train_loss = []
                for it in range(int(train_total_size_nor / FLAGS.batch_size)):
                    random_number_05 = np.random.random_sample()
                    bucket_id = min([i for i in range(len(train_buckets_scale_nor))
                                     if train_buckets_scale_nor[i] > random_number_05])
                    encoder_inputs, decoder_inputs, targets, target_weights, source_sequence_length = autoencoder.get_batch(train_set_nor, _source_buckets,
                                                                                bucket_id)
                    feed = {autoencoder.input_ids:encoder_inputs,
                            autoencoder.targets:targets,
                            autoencoder.decoder_inputs:decoder_inputs,
                            autoencoder.target_weights:target_weights,
                            autoencoder.source_sequence_length:source_sequence_length,
                            autoencoder.length:_source_buckets[bucket_id]}
                    loss, _ = sess.run([autoencoder.loss, autoencoder.train_op], feed_dict=feed)
                    train_loss.append(loss)

                    if it % 500 == 0:
                        test_in, test_out = autoencoder.decode(sess, encoder_inputs[0])
                        for i in range(len(test_in)):
                            print("test_in:  ", test_in[i])
                            print("test_out:  ", test_out[i])
                        autoencoder.saver.save(sess, FLAGS.ae_ckpt_path, global_step=autoencoder.global_step)


                    print(_source_buckets[bucket_id]," ",loss)
    """
def create_hparams(flags):
    return tf.contrib.training.HParams(
        # dir path
        data_dir=flags.data_dir,
        train_dir=flags.train_dir,
        ae_ckpt_path=flags.ae_ckpt_path,

        # data params
        batch_size=flags.batch_size,
        from_vocab_size=flags.from_vocab_size,
        to_vocab_size=flags.to_vocab_size,
        buckets=_buckets,
        source_buckets=_source_buckets,
        target_buckets=_target_buckets,
        GO_ID=data_utils.GO_ID,
        EOS_ID=data_utils.EOS_ID,
        PAD_ID=data_utils.PAD_ID,
        emb_dim=flags.emb_dim,

        # ae params
        num_units=flags.num_units,
        num_layers=flags.num_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor

    )
def main(_):
    from_train_data = FLAGS.from_train_data
    to_train_data = FLAGS.to_train_data
    from_dev_data = FLAGS.from_dev_data
    to_dev_data = FLAGS.to_dev_data
    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_dir,
        from_train_data,
        to_train_data,
        from_dev_data,
        to_dev_data,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size)
    hparams = create_hparams(FLAGS)
    hparams.add_hparam(name="from_train", value=from_train)
    hparams.add_hparam(name="to_train", value=to_train)
    hparams.add_hparam(name="from_dev", value=from_dev)
    hparams.add_hparam(name="to_dev", value=to_dev)
    from_vocab_path = os.path.join(hparams.data_dir, "vocab%d.from" % hparams.from_vocab_size)
    to_vocab_path = os.path.join(hparams.data_dir, "vocab%d.to" % hparams.to_vocab_size)
    train_ae(hparams)
    #train(from_train, to_train, from_dev, to_dev)


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    tf.app.run()