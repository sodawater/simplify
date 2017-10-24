from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
#_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  #words = []
  #for space_separated_fragment in sentence.strip('\n').strip().split():
  #  words.extend(_WORD_SPLIT.split(space_separated_fragment))
  words = sentence.strip('\n').strip(" ").split(" ")
  return words


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(r"0", w) if normalize_digits else w.lower()
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          #print(w + "\n")
          vocab_file.write(w + "\n")

def create_vocabulary_samevocab(from_vocab_path, to_vocab_path, from_data_path, to_data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  if not gfile.Exists(from_vocab_path):
    print("Creating vocabulary %s from data %s" % (from_vocab_path, from_data_path))
    vocab = {}
    with gfile.GFile(from_data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(r"0", w) if normalize_digits else w.lower()
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
    with gfile.GFile(to_data_path, mode="r") as f:
          counter = 0
          for line in f:
              counter += 1
              if counter % 100000 == 0:
                  print("  processing line %d" % counter)
              tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
              for w in tokens:
                  word = _DIGIT_RE.sub(r"0", w) if normalize_digits else w.lower()
                  if word in vocab:
                      vocab[word] += 1
                  else:
                      vocab[word] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(from_vocab_path, mode="w") as vocab_file:
        for w in vocab_list:
            # print(w + "\n")
            vocab_file.write(w + "\n")
    with gfile.GFile(to_vocab_path, mode="w") as vocab_file:
        for w in vocab_list:
            # print(w + "\n")
            vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w.lower(), UNK_ID) for w in words]
  return [vocabulary.get(_DIGIT_RE.sub("0", w.lower()), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False  ):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab,
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, from_vocabulary_size,
                 to_vocabulary_size, same_vocab=False, tokenizer=None):
    if same_vocab == False:
        to_vocab_path = os.path.join(data_dir, "vocab%d.to" % to_vocabulary_size)
        from_vocab_path = os.path.join(data_dir, "vocab%d.from" % from_vocabulary_size)
        to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
        from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
        to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
        from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
    else:
        to_vocab_path = os.path.join(data_dir, "vocab_same%d.to" % to_vocabulary_size)
        from_vocab_path = os.path.join(data_dir, "vocab_same%d.from" % from_vocabulary_size)
        to_train_ids_path = to_train_path + (".ids_same%d" % to_vocabulary_size)
        from_train_ids_path = from_train_path + (".ids_same%d" % from_vocabulary_size)
        to_dev_ids_path = to_dev_path + (".ids_same%d" % to_vocabulary_size)
        from_dev_ids_path = from_dev_path + (".ids_same%d" % from_vocabulary_size)
    if same_vocab == False:
        create_vocabulary(to_vocab_path, to_train_path, to_vocabulary_size, tokenizer)
        create_vocabulary(from_vocab_path, from_train_path, from_vocabulary_size, tokenizer)
    else:
        create_vocabulary_samevocab(from_vocab_path, to_vocab_path, from_train_path, to_train_path, from_vocabulary_size, tokenizer)
    data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path, tokenizer)
    data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path, tokenizer)

    data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
    data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path, tokenizer)

    return (from_train_ids_path, to_train_ids_path,
          from_dev_ids_path, to_dev_ids_path,
          from_vocab_path, to_vocab_path)