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
from reconstructor import Reconstructor
from birnn_seq2seq import BiRNN_seq2seq
import argparse
import copy
import collections
from autoencoder_noattention import Autoencoder_noattention
from discriminator_RNN import Discriminator_RNN
tf.app.flags.DEFINE_integer("total_batch", 5000, "Number of batches.")
tf.app.flags.DEFINE_integer("ae_epoch_num", 100, "train epoch num for ae")
tf.app.flags.DEFINE_integer("pretrain_epoch_gen", 50, "Pretrain epoch num for generator.")
tf.app.flags.DEFINE_integer("pretrain_epoch_dis", 50, "Pretrain epoch num for discriminator.")
tf.app.flags.DEFINE_integer("cnn_seq_len", 25, "Length of the sequence fed to CNN.")
tf.app.flags.DEFINE_integer("generated_num", 10000, "Number of samples generated by generator.")

FLAGS = None
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_dir", type=str, default="/data/wtm/data/wikilarge/", help="Data directory")
    parser.add_argument("--train_dir", type=str, default="/data/wtm/data/wikilarge/model/GAN/", help="Training directory")
    parser.add_argument("--from_train_data", type=str, default="/data/wtm/data/wikilarge/wiki.full.aner.train.src",
                        help="Training data_src path")
    parser.add_argument("--to_train_data", type=str, default="/data/wtm/data/wikilarge/wiki.full.aner.train.dst",
                        help="Training data_dst path")
    parser.add_argument("--from_valid_data", type=str, default="/data/wtm/data/wikilarge/wiki.full.aner.valid.src",
                        help="Valid data_src path")
    parser.add_argument("--to_valid_data", type=str, default="/data/wtm/data/wikilarge/wiki.full.aner.valid.dst",
                        help="Valid data_dst path")
    parser.add_argument("--from_test_data", type=str, default="/data/wtm/data/wikilarge/wiki.full.aner.test.src",
                        help="Test data_src path")
    parser.add_argument("--to_test_data", type=str, default="/data/wtm/data/wikilarge/wiki.full.aner.test.dst",
                        help="Test data_dst path")
    parser.add_argument("--ae_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/ae/",
                        help="ae model checkpoint directory")
    parser.add_argument("--nmt_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/nmt/sgd_new/",
                        help="nmt model checkpoint directory")
    parser.add_argument("--gan_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/gan/",
                        help="gan model checkpoint directory")
    parser.add_argument("--dis_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/dis/",
                        help="dis model checkpoint directory")
    parser.add_argument("--recon_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/recon_0.0/",
                        help="reconstructor model checkpoint directory")
    parser.add_argument("--recon_g_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/recon/g/",
                        help="reconstructor model checkpoint directory")
    parser.add_argument("--recon_d_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/recon/d/",
                        help="reconstructor model checkpoint directory")

    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on the size of training data (0: no limit)")
    parser.add_argument("--attention", type=str, default="", help="""\
          luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
          attention\
          """)
    parser.add_argument("--from_vocab_size", type=int, default=30000, help="NormalWiki vocabulary size")
    parser.add_argument("--to_vocab_size", type=int, default=30000, help="SimpleWiki vocabulary size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--num_units", type=int, default=256, help="Size of each model layer")
    parser.add_argument("--emb_dim", type=int, default=256, help="Dimension of word embedding")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use during training")
    parser.add_argument("--max_gradient_norm", type=float, default=1.0, help="Clip gradients to this norm")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.7, help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=1, help="Learning rate")
    parser.add_argument("--keep_prob", type=float, default=0.7, help="dropout keep prob")

    parser.add_argument("--num_train_epoch", type=int, default=100, help="Number of epoch for training")
    parser.add_argument("--steps_per_eval", type=int, default=2000, help="How many training steps to do per eval/checkpoint")


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(15, 10), (25, 20), (35, 30), (50, 45)]
_buckets = [(80, 80)]
_target_buckets = [10, 15, 20, 25, 30, 40, 50]
_source_buckets = [5, 10, 15, 20, 25, 35, 45]
_dis_buckets = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#_dis_buckets = [(2 * x) for x in range(4,26)]

def read_data_all(source_path, target_path, max_size=None):
  data_set = [[] for _ in _dis_buckets]
  neg_set = [[] for _ in _dis_buckets]
  pos_set = [[] for _ in _dis_buckets]
  neg = [0 for _ in _dis_buckets]
  pos = [0 for _ in _dis_buckets]
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
        for bucket_id, (size) in enumerate(_dis_buckets):
            if len(source_ids) < size:
                neg_set[bucket_id].append([source_ids, [1, 0]])
                neg[bucket_id] += 1
                break
        for bucket_id, (size) in enumerate(_dis_buckets):
            if len(target_ids) < size:
                pos_set[bucket_id].append([target_ids, [0, 1]])
                pos[bucket_id] += 1
                break
        source, target = source_file.readline(), target_file.readline()
  for i in range(len(_dis_buckets)):
    print(neg[i], pos[i])
    if neg[i] > pos[i]:
        data_set[i].extend(pos_set[i])
        data_set[i].extend(neg_set[i][0:pos[i]])
    else:
        data_set[i].extend(pos_set[i][0:neg[i]])
        data_set[i].extend(neg_set[i])
  return data_set

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
        if (len(target_ids) >= 3):
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

def split_dataset(dataset, ratio):
    dataset1 = [[] for _ in _buckets]
    dataset2 = [[] for _ in _buckets]
    for i in range(len(_buckets)):
        for pair in dataset[i]:
            random_number = random.random()
            if random_number > ratio:
                dataset2[i].append(pair)
            else:
                dataset1[i].append(pair)
    return dataset1, dataset2

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

def create_model(hparams, model):
    if model == BiRNN_seq2seq:
        print("Creating birnn_seq2seq_model...")
    elif model == Autoencoder:
        print("Creating ae model...")
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = model(hparams, tf.contrib.learn.ModeKeys.TRAIN)

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_model = model(hparams, tf.contrib.learn.ModeKeys.EVAL)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_model = model(hparams, tf.contrib.learn.ModeKeys.INFER)
    return TrainModel(graph=train_graph, model=train_model), EvalModel(graph=eval_graph, model=eval_model), InferModel(
        graph=infer_graph, model=infer_model)



def decode_sen(outputs):
    to_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.to" % FLAGS.to_vocab_size)
    _, rev_to_vocab = data_utils.initialize_vocabulary(to_vocab_path)
    for sentence in outputs:
        sentence = sentence.tolist()
        if data_utils.EOS_ID in sentence:
            sentence = sentence[:sentence.index(data_utils.EOS_ID)]
        print(" ".join([tf.compat.as_str(rev_to_vocab[word]) for word in sentence]))


def train_dis(hparams, train=True, interact=False):
    dis_hparams = copy.deepcopy(hparams)
    dis_hparams.add_hparam(name="dis_ckpt_path", value=os.path.join(hparams.dis_ckpt_dir, "dis.ckpt"))
    train_model, eval_model, infer_model = create_model(dis_hparams, Discriminator_RNN)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)

    ckpt = tf.train.get_checkpoint_state(dis_hparams.dis_ckpt_dir)
    global_step = 0
    with train_model.graph.as_default():
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            train_model.model.saver.restore(train_sess, ckpt.model_checkpoint_path)
            global_step = train_model.model.global_step.eval(session=train_sess)
        else:
            train_sess.run(tf.global_variables_initializer())

    train_set_pair = read_data_all(dis_hparams.from_train, dis_hparams.to_train, dis_hparams.max_train_data_size)
    train_bucket_sizes_pair = [len(train_set_pair[b]) for b in range(len(_dis_buckets))]
    train_total_size_pair = float(sum(train_bucket_sizes_pair))
    train_buckets_scale_pair = [sum(train_bucket_sizes_pair[:i + 1]) / train_total_size_pair
                               for i in range(len(train_bucket_sizes_pair))]

    valid_set_pair = read_data_all(dis_hparams.from_valid, dis_hparams.to_valid, dis_hparams.max_train_data_size)
    valid_bucket_sizes_pair = [len(valid_set_pair[b]) for b in range(len(_dis_buckets))]
    valid_total_size_pair = float(sum(valid_bucket_sizes_pair))
    valid_buckets_scale_pair = [sum(valid_bucket_sizes_pair[:i + 1]) / valid_total_size_pair
                               for i in range(len(valid_bucket_sizes_pair))]

    num_train_steps = dis_hparams.num_train_epoch * int(train_total_size_pair / FLAGS.batch_size)
    #num_train_steps = 2000
    if train:
        while global_step < num_train_steps:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale_pair))
                             if train_buckets_scale_pair[i] > random_number_01])
            encoder_inputs, targets, source_sequence_length = train_model.model.get_batch(
                train_set_pair, _dis_buckets,
                bucket_id, dis_hparams.batch_size)
            feed = {train_model.model.encoder_input_ids: encoder_inputs,
                    train_model.model.encoder_input_length: source_sequence_length,
                    train_model.model.targets: targets}
            absolute_diff, a, b, _, global_step = train_sess.run([train_model.model.absolute_diff,
                                                            train_model.model.targets,
                                                            train_model.model.predictions,
                                                   train_model.model.train_op,
                                                   train_model.model.global_step], feed_dict=feed)

            if global_step % 50 == 0:
                print(1 - absolute_diff)

            if global_step % 1000 == 0:
                train_model.model.saver.save(train_sess, dis_hparams.dis_ckpt_path, global_step=global_step)
                ckpt = tf.train.get_checkpoint_state(dis_hparams.dis_ckpt_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    eval_model.model.saver.restore(eval_sess, ckpt.model_checkpoint_path)
                else:
                    raise ValueError("ckpt file not found.")
                random_number_02 = np.random.random_sample()
                bucket_id = min([i for i in range(len(valid_buckets_scale_pair))
                                 if valid_buckets_scale_pair[i] > random_number_02])
                encoder_inputs, targets, source_sequence_length = train_model.model.get_batch(
                    valid_set_pair, _dis_buckets,
                    bucket_id, dis_hparams.batch_size)
                feed = {eval_model.model.encoder_input_ids: encoder_inputs,
                        eval_model.model.encoder_input_length: source_sequence_length,
                        eval_model.model.targets: targets}
                absolute_diff = eval_sess.run(eval_model.model.absolute_diff, feed_dict=feed)
                # print(loss)
                print("step %d with eval auc %f" % (global_step, 1 - absolute_diff))
    else:
        ckpt = tf.train.get_checkpoint_state(dis_hparams.dis_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
            from_vocab_path = os.path.join(dis_hparams.data_dir,
                                           "vocab_same%d.from" % dis_hparams.from_vocab_size)
            to_vocab_path = os.path.join(dis_hparams.data_dir,
                                         "vocab_same%d.to" % dis_hparams.to_vocab_size)
            from_vocab, _ = data_utils.initialize_vocabulary(from_vocab_path)
            _, rev_to_vocab = data_utils.initialize_vocabulary(to_vocab_path)
            file = open(FLAGS.from_test_file, "r", encoding="utf-8")
            outfile = open("test_out", "w", encoding="utf-8")
            for line in file:
                sentence = line.rstrip("\n")
                token_ids = data_utils.sentence_to_token_ids(sentence, from_vocab)
                encoder_inputs = [token_ids + [dis_hparams.EOS_ID]]
                source_sequence_length = [len(token_ids) + 1]
                feed = {infer_model.model.encoder_input_ids: encoder_inputs,
                        infer_model.model.encoder_input_length: source_sequence_length}
                sample_outputs = infer_sess.run(infer_model.model.sample_id, feed_dict=feed)
                sample_outputs = sample_outputs[0].tolist()
                if dis_hparams.EOS_ID in sample_outputs:
                    sample_outputs = sample_outputs[:sample_outputs.index(data_utils.EOS_ID)]
                outfile(" ".join([tf.compat.as_str(rev_to_vocab[output]) for output in sample_outputs]) + "\n")
            file.close()
            outfile.close()
        else:
            raise ValueError("ckpt file not found.")

def generate_samples(model, sess, generated_num, dataset, buckets, bucket_id, batch_size):
    gen_set = [[]]
    for _ in range(int(generated_num / batch_size)):
        bucket_id = 0
        samples = model.get_random_samples(sess, dataset, buckets, bucket_id, batch_size)
        gen_set[0].extend(samples)
    return gen_set

def train_recon(hparams, pretrain=True, train=True, interact=False):
    recon_hparams = copy.deepcopy(hparams)
    recon_hparams.add_hparam(name="g_ckpt_path", value=os.path.join(hparams.recon_g_ckpt_dir, "g_recon.ckpt"))
    recon_hparams.add_hparam(name="d_ckpt_path", value=os.path.join(hparams.recon_d_ckpt_dir, "d_recon.ckpt"))
    g_train_model, g_eval_model, g_infer_model = create_model(recon_hparams, Reconstructor)
    d_train_model, d_eval_model, d_infer_model = create_model(recon_hparams, Discriminator_RNN)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    g_train_sess = tf.Session(config=config, graph=g_train_model.graph)
    g_eval_sess = tf.Session(config=config, graph=g_eval_model.graph)
    g_infer_sess = tf.Session(config=config, graph=g_infer_model.graph)
    d_train_sess = tf.Session(config=config, graph=d_train_model.graph)
    d_eval_sess = tf.Session(config=config, graph=d_eval_model.graph)
    d_infer_sess = tf.Session(config=config, graph=d_infer_model.graph)

    train_set = read_data_pair(recon_hparams.from_train, recon_hparams.to_train, recon_hparams.max_train_data_size)
    train_set_pair, train_set_backup = split_dataset(train_set, 0.1)
    train_bucket_sizes_pair = [len(train_set_pair[b]) for b in range(len(_buckets))]
    train_total_size_pair = float(sum(train_bucket_sizes_pair))
    train_buckets_scale_pair = [sum(train_bucket_sizes_pair[:i + 1]) / train_total_size_pair
                               for i in range(len(train_bucket_sizes_pair))]

    valid_set_pair = read_data_pair(recon_hparams.from_valid, recon_hparams.to_valid, recon_hparams.max_train_data_size)
    valid_bucket_sizes_pair = [len(valid_set_pair[b]) for b in range(len(_buckets))]
    valid_total_size_pair = float(sum(valid_bucket_sizes_pair))
    valid_buckets_scale_pair = [sum(valid_bucket_sizes_pair[:i + 1]) / valid_total_size_pair
                               for i in range(len(valid_bucket_sizes_pair))]

    num_train_steps = recon_hparams.num_train_epoch * int(train_total_size_pair / FLAGS.batch_size)
    num_pretrain_steps = train_total_size_pair
    if pretrain:

        ckpt = tf.train.get_checkpoint_state(recon_hparams.recon_g_ckpt_dir)
        with g_train_model.graph.as_default():
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                g_train_model.model.saver.restore(g_train_sess, ckpt.model_checkpoint_path)
                global_step_pt = g_train_model.model.global_step_pt.eval(session=g_train_sess)
            else:
                g_train_sess.run(tf.global_variables_initializer())
                global_step_pt = 0
        while global_step_pt < num_pretrain_steps:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale_pair))
                             if train_buckets_scale_pair[i] > random_number_01])
            loss, global_step = g_train_model.model.pretrain_step(g_train_sess,
                                                                train_set_pair,
                                                                _buckets,
                                                                bucket_id,
                                                                recon_hparams.batch_size)
            if global_step % 50 == 0:
                print(loss)
            if global_step % recon_hparams.steps_per_eval == 0:
                g_train_model.model.saver.save(g_train_sess, recon_hparams.recon_g_ckpt_path, global_step=global_step)
                ckpt = tf.train.get_checkpoint_state(recon_hparams.recon_g_ckpt_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    g_eval_model.model.saver.restore(g_eval_sess, ckpt.model_checkpoint_path)
                else:
                    raise ValueError("ckpt file not found.")
                random_number_02 = np.random.random_sample()
                bucket_id = min([i for i in range(len(valid_buckets_scale_pair))
                                 if valid_buckets_scale_pair[i] > random_number_02])
                loss = g_eval_model.model.eval_step(g_eval_sess,
                                                  valid_set_pair,
                                                  _buckets,
                                                  bucket_id,
                                                  recon_hparams.batch_size)
                # print(loss)
                print("step %d with eval loss %f" % (global_step, loss))

        gen_set = generate_samples(g_train_model.model,
                                   g_train_sess,
                                   train_total_size_pair,
                                   train_set_pair,
                                   _buckets,
                                   0,
                                   recon_hparams.batch_size)

        ckpt = tf.train.get_checkpoint_state(recon_hparams.recon_d_ckpt_dir)
        with d_train_model.graph.as_default():
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                d_train_model.model.saver.restore(d_train_sess, ckpt.model_checkpoint_path)
                global_step_pt = d_train_model.model.global_step.eval(session=d_train_sess)
            else:
                g_train_sess.run(tf.global_variables_initializer())
                global_step_pt = 0

        while global_step_pt < num_pretrain_steps:
            random_number_03 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale_pair))
                             if train_buckets_scale_pair[i] > random_number_03])
            loss, accuracy, global_step = d_train_model.model.train_step(d_train_sess,
                                                               train_set_pair,
                                                               gen_set,
                                                               _buckets,
                                                               bucket_id,
                                                               recon_hparams.batch_size)
            print("step %d with accuracy %f" % (global_step, accuracy))

    if train:
        ckpt = tf.train.get_checkpoint_state(recon_hparams.recon_g_ckpt_dir)
        with g_train_model.graph.as_default():
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                g_train_model.model.saver.restore(g_train_sess, ckpt.model_checkpoint_path)
                global_step = g_train_model.model.global_step.eval(session=g_train_sess)
            else:
                g_train_sess.run(tf.global_variables_initializer())
                global_step = 0
        ckpt = tf.train.get_checkpoint_state(recon_hparams.recon_d_ckpt_dir)
        with d_train_model.graph.as_default():
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                d_train_model.model.saver.restore(d_train_sess, ckpt.model_checkpoint_path)
            else:
                g_train_sess.run(tf.global_variables_initializer())

        while global_step < num_train_steps:
            random_number_04 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale_pair))
                             if train_buckets_scale_pair[i] > random_number_04])
            loss, global_step = g_train_model.model.adversarial_train_step(
                g_train_sess,
                d_train_sess,
                train_set_backup,
                _buckets,
                bucket_id,
                recon_hparams.batch_size,
                d_train_model.model,
                30)

            print("step %d with adversarial train loss %f " % (global_step, loss))

            if global_step % 20 == 0:
                gen_set = generate_samples(
                    g_train_model.model,
                    g_train_sess,
                    train_total_size_pair,
                    train_set_pair,
                    _buckets,
                    0,
                    recon_hparams.batch_size)
            if global_step % 3 == 0:
                random_number_05 = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale_pair))
                                 if train_buckets_scale_pair[i] > random_number_05])
                loss, accuracy, global_step = d_train_model.model.train_step(d_train_sess,
                                                                             train_set_pair,
                                                                             gen_set,
                                                                             _buckets,
                                                                             bucket_id)
                print("step %d with accuracy %f " % (global_step, accuracy))





def create_hparams(flags):
    return tf.contrib.training.HParams(
        # dir path
        data_dir=flags.data_dir,
        train_dir=flags.train_dir,
        ae_ckpt_dir=flags.ae_ckpt_dir,
        nmt_ckpt_dir=flags.nmt_ckpt_dir,
        gan_ckpt_dir=flags.gan_ckpt_dir,
        dis_ckpt_dir=flags.dis_ckpt_dir,
        recon_ckpt_dir=flags.recon_ckpt_dir,
        recon_g_ckpt_dir=flags.recon_g_ckpt_dir,
        recon_d_ckpt_dir=flags.recon_d_ckpt_dir,

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
        max_train_data_size = flags.max_train_data_size,
        num_train_epoch = flags.num_train_epoch,
        steps_per_eval = flags.steps_per_eval,

        # recon params
        keep_prob=flags.keep_prob,
        num_units=flags.num_units,
        num_layers=flags.num_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor
    )

def main(_):
    from_train_data = FLAGS.from_train_data
    to_train_data = FLAGS.to_train_data
    from_valid_data = FLAGS.from_valid_data
    to_valid_data = FLAGS.to_valid_data
    from_train, to_train, from_valid, to_valid, _, _ = data_utils.prepare_data(
        FLAGS.data_dir,
        from_train_data,
        to_train_data,
        from_valid_data,
        to_valid_data,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size,
        same_vocab=False
    )
    hparams = create_hparams(FLAGS)
    hparams.add_hparam(name="from_train", value=from_train)
    hparams.add_hparam(name="to_train", value=to_train)
    hparams.add_hparam(name="from_valid", value=from_valid)
    hparams.add_hparam(name="to_valid", value=to_valid)
    from_vocab_path = os.path.join(hparams.data_dir, "vocab%d.from" % hparams.from_vocab_size)
    to_vocab_path = os.path.join(hparams.data_dir, "vocab%d.to" % hparams.to_vocab_size)
    #train_dis(hparams, train=True, interact=False)
    train_recon(hparams, pretrain=True, train=False, interact=False)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    print(FLAGS)
    tf.app.run()