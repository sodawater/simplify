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
    parser.add_argument("--nmt_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/nmt/sgd_t3/",
                        help="nmt model checkpoint directory")
    parser.add_argument("--gan_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/gan/",
                        help="gan model checkpoint directory")
    parser.add_argument("--dis_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/dis/",
                        help="dis model checkpoint directory")
    parser.add_argument("--recon_ckpt_dir", type=str, default="/data/wtm/data/wikilarge/model/recon_0.0/",
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
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.6, help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=1, help="Learning rate")

    parser.add_argument("--num_train_epoch", type=int, default=100, help="Number of epoch for training")
    parser.add_argument("--steps_per_eval", type=int, default=2000, help="How many training steps to do per eval/checkpoint")


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(15, 10), (25, 20), (35, 30), (50, 45)]
_buckets = [(90, 90)]
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
        if (len(target_ids) >= 5):
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



def train_nmt(hparams, train=True, interact=False):
    nmt_hparams = copy.deepcopy(hparams)
    nmt_hparams.add_hparam(name="nmt_ckpt_path", value=os.path.join(hparams.nmt_ckpt_dir, "nmt.ckpt"))
    train_model, eval_model, infer_model = create_model(nmt_hparams, BiRNN_seq2seq)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)

    ckpt = tf.train.get_checkpoint_state(nmt_hparams.nmt_ckpt_dir)
    global_step = 0
    with train_model.graph.as_default():
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            train_model.model.saver.restore(train_sess, ckpt.model_checkpoint_path)
            global_step = train_model.model.global_step.eval(session=train_sess)
        else:
            train_sess.run(tf.global_variables_initializer())

    train_set_pair = read_data_pair(nmt_hparams.from_train, nmt_hparams.to_train, nmt_hparams.max_train_data_size)
    train_bucket_sizes_pair = [len(train_set_pair[b]) for b in range(len(_buckets))]
    train_total_size_pair = float(sum(train_bucket_sizes_pair))
    train_buckets_scale_pair = [sum(train_bucket_sizes_pair[:i + 1]) / train_total_size_pair
                               for i in range(len(train_bucket_sizes_pair))]

    valid_set_pair = read_data_pair(nmt_hparams.from_valid, nmt_hparams.to_valid, nmt_hparams.max_train_data_size)
    valid_bucket_sizes_pair = [len(valid_set_pair[b]) for b in range(len(_buckets))]
    valid_total_size_pair = float(sum(valid_bucket_sizes_pair))
    valid_buckets_scale_pair = [sum(valid_bucket_sizes_pair[:i + 1]) / valid_total_size_pair
                               for i in range(len(valid_bucket_sizes_pair))]

    num_train_steps = nmt_hparams.num_train_epoch * int(train_total_size_pair / FLAGS.batch_size)
    loss_file1 = open("t3train_loss","a",encoding="utf-8")
    loss_file2 = open("t3eval_loss","a",encoding="utf-8")
    if train:
        tot_loss = 0
        while global_step < num_train_steps:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale_pair))
                             if train_buckets_scale_pair[i] > random_number_01])
            encoder_inputs, decoder_inputs, targets, target_weights, source_sequence_length, target_sequence_length = train_model.model.get_batch(
                train_set_pair, _buckets,
                bucket_id, nmt_hparams.batch_size)
            feed = {train_model.model.encoder_input_ids: encoder_inputs,
                    train_model.model.encoder_input_length: source_sequence_length,
                    train_model.model.decoder_input_ids: decoder_inputs,
                    train_model.model.decoder_input_length: target_sequence_length,
                    train_model.model.targets: targets,
                    train_model.model.target_weights: target_weights}
            loss, _, global_step = train_sess.run([train_model.model.loss,
                                                   train_model.model.train_op,
                                                   train_model.model.global_step
                                                   ], feed_dict=feed)

            # print(loss/_source_buckets[bucket_id])
            #print(alignment_history)
            tot_loss += loss
            if global_step % 50 == 0:
                print("global_step ",global_step,":   ",tot_loss / 50.0)
                loss_file1.write(str(tot_loss / 50.0))
                loss_file1.write("\n")
                tot_loss = 0
                #print(loss / _source_buckets[bucket_id])
            if global_step % 5000 == 2000 and global_step > 15000:
                learning_rate = train_sess.run(train_model.model.learning_rate_decay_op)
                print("learning rate is %f now" % learning_rate)
            if global_step % nmt_hparams.steps_per_eval == 0:
                train_model.model.saver.save(train_sess, nmt_hparams.nmt_ckpt_path, global_step=global_step)
                ckpt = tf.train.get_checkpoint_state(nmt_hparams.nmt_ckpt_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    eval_model.model.saver.restore(eval_sess, ckpt.model_checkpoint_path)
                else:
                    raise ValueError("ckpt file not found.")
                mean_loss = 0
                for _ in range(50):
                    random_number_02 = np.random.random_sample()
                    bucket_id = min([i for i in range(len(valid_buckets_scale_pair))
                                     if valid_buckets_scale_pair[i] > random_number_02])
                    encoder_inputs, decoder_inputs, targets, target_weights, source_sequence_length, target_sequence_length = train_model.model.get_batch(
                        valid_set_pair, _buckets,
                        bucket_id, nmt_hparams.batch_size)
                    feed = {eval_model.model.encoder_input_ids: encoder_inputs,
                            eval_model.model.encoder_input_length: source_sequence_length,
                            eval_model.model.decoder_input_ids: decoder_inputs,
                            eval_model.model.decoder_input_length: target_sequence_length,
                            eval_model.model.targets: targets,
                            eval_model.model.target_weights: target_weights}
                    loss = eval_sess.run(eval_model.model.loss, feed_dict=feed)
                    mean_loss += loss
                # print(loss)
                loss_file2.write(str(mean_loss))
                loss_file2.write("\n")
                loss_file2.close()
                loss_file2 = open("t2eval_loss","a",encoding="utf-8")
                print("step %d with eval loss %f" % (global_step, mean_loss))
            if global_step % 2000 == 0:
                ckpt = tf.train.get_checkpoint_state(nmt_hparams.nmt_ckpt_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
                    from_vocab_path = os.path.join(nmt_hparams.data_dir,
                                                   "vocab%d.from" % nmt_hparams.from_vocab_size)
                    to_vocab_path = os.path.join(nmt_hparams.data_dir,
                                                 "vocab%d.to" % nmt_hparams.to_vocab_size)
                    from_vocab, rev_from_vocab = data_utils.initialize_vocabulary(from_vocab_path)
                    _, rev_to_vocab = data_utils.initialize_vocabulary(to_vocab_path)
                    file = open(FLAGS.from_test_data, "r", encoding="utf-8")
                    outfile = open("test_out_sgdt3_" + str(global_step), "w", encoding="utf-8")
                    for line in file:
                        sentence = line.rstrip("\n")
                        token_words = sentence.rstrip("\n").lower().split()
                        token_words.append("")
                        token_ids = data_utils.sentence_to_token_ids(sentence, from_vocab)
                        encoder_inputs = [token_ids]
                        source_sequence_length = [len(token_ids)]
                        feed = {infer_model.model.encoder_input_ids: encoder_inputs,
                                infer_model.model.encoder_input_length: source_sequence_length}
                        sample_outputs, alignment = infer_sess.run([infer_model.model.sample_id,
                                                                    infer_model.model.alignment_history],
                                                                   feed_dict=feed)
                        sample_outputs = sample_outputs[0].tolist()
                        if nmt_hparams.EOS_ID in sample_outputs:
                            sample_outputs = sample_outputs[:sample_outputs.index(data_utils.EOS_ID)]
                        outputs = []
                        alignment = alignment[0].tolist()
                        id = 0
                        for output in sample_outputs:
                            if output != 3:
                                outputs.append(tf.compat.as_str(rev_to_vocab[output]))
                            else:
                                if id + 1 <= len(alignment):
                                    outputs.append(tf.compat.as_str(token_words[alignment[id]]))
                                else:
                                    outputs.append(tf.compat.as_str(rev_to_vocab[output]))
                            id += 1
                        print(" ".join(outputs))
                        outfile.write(" ".join(outputs) + "\n")
                    file.close()
                    outfile.close()
    else:
        ckpt = tf.train.get_checkpoint_state(nmt_hparams.nmt_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
            from_vocab_path = os.path.join(nmt_hparams.data_dir,
                                           "vocab%d.from" % nmt_hparams.from_vocab_size)
            to_vocab_path = os.path.join(nmt_hparams.data_dir,
                                         "vocab%d.to" % nmt_hparams.to_vocab_size)
            from_vocab, rev_from_vocab = data_utils.initialize_vocabulary(from_vocab_path)
            _, rev_to_vocab = data_utils.initialize_vocabulary(to_vocab_path)
            if interact:
                sys.stdout.write("> ")
                sys.stdout.flush()
                sentence = sys.stdin.readline()
                while sentence:
                    token_words = sentence.rstrip("\n").lower().split()
                    token_words.append("")
                    #print(token_words)
                    token_ids = data_utils.sentence_to_token_ids(sentence, from_vocab)
                    encoder_inputs = [token_ids]
                    source_sequence_length = [len(token_ids)]
                    feed = {infer_model.model.encoder_input_ids: encoder_inputs,
                            infer_model.model.encoder_input_length: source_sequence_length}
                    sample_outputs, alignment, tmp = infer_sess.run([infer_model.model.sample_id,
                                                     infer_model.model.alignment_history,
                                                                infer_model.model.tmp],
                                                    feed_dict=feed)
                    sample_outputs = sample_outputs[0].tolist()
                    #if nmt_hparams.EOS_ID in sample_outputs:
                    #    sample_outputs = sample_outputs[:sample_outputs.index(data_utils.EOS_ID)]
                    print(tmp)
                    outputs = []
                    alignment = alignment[0].tolist()
                    print(source_sequence_length)
                    print(alignment)
                    id = 0
                    for output in sample_outputs:
                        if output != 3:
                            outputs.append(tf.compat.as_str(rev_to_vocab[output]))
                        else:
                            outputs.append(tf.compat.as_str(token_words[alignment[id]]))
                        id += 1
                    print(" ".join(outputs))
                    print(" ".join([tf.compat.as_str(rev_to_vocab[output]) for output in sample_outputs]))
                    print("> ", end="")
                    sys.stdout.flush()
                    sentence = sys.stdin.readline()
            else:
                file = open(FLAGS.from_test_data, "r", encoding="utf-8")
                outfile = open("test_out_sgd", "w", encoding="utf-8")
                for line in file:
                    sentence = line.rstrip("\n")
                    token_words = sentence.rstrip("\n").lower().split()
                    token_words.append("")
                    token_ids = data_utils.sentence_to_token_ids(sentence, from_vocab)
                    encoder_inputs = [token_ids + [nmt_hparams.EOS_ID]]
                    source_sequence_length = [len(token_ids) + 1]
                    feed = {infer_model.model.encoder_input_ids: encoder_inputs,
                            infer_model.model.encoder_input_length: source_sequence_length}
                    sample_outputs, alignment = infer_sess.run([infer_model.model.sample_id,
                                                                infer_model.model.alignment_history],
                                                               feed_dict=feed)
                    sample_outputs = sample_outputs[0].tolist()
                    if nmt_hparams.EOS_ID in sample_outputs:
                        sample_outputs = sample_outputs[:sample_outputs.index(data_utils.EOS_ID)]
                    outputs = []
                    alignment = alignment[0].tolist()
                    id = 0
                    for output in sample_outputs:
                        if output != 3:
                            outputs.append(tf.compat.as_str(rev_to_vocab[output]))
                        else:
                            outputs.append(tf.compat.as_str(token_words[alignment[id]]))
                        id += 1
                    print(" ".join(outputs))
                    outfile.write(" ".join(outputs) + "\n")
                file.close()
                outfile.close()
        else:
            raise ValueError("ckpt file not found.")


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
    #train_ae(hparams, train=True, interact=True)
    train_nmt(hparams, train=True, interact=True)
    #train(from_train, to_train, from_dev, to_dev)
    #train_dis(hparams, train=True, interact=False)
    #train_recon(hparams, pretrain=True, train=False, interact=False)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    print(FLAGS)
    tf.app.run()