import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys

model_path = '/language-style-transfer/code'

if model_path not in sys.path:
    sys.path.append(model_path)
    
from vocab import Vocabulary, build_vocab
from accumulator import Accumulator
from options import load_arguments
from file_io import load_sent, write_sent
from utils import *
from nn import *
import beam_search, greedy_decoding

from style_transfer import Model, transfer, create_model

from itertools import groupby
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import re

from collections import namedtuple

args = {
    'batch_size': 32,
    'beam': 1,
    'dev': '/storage/data3/ods_dota_dev',
    'dim_emb': 100,
    'dim_y': 200,
    'dim_z': 500,
    'dropout_keep_prob': 0.5,
    'embedding': '',
    'filter_sizes': '1,2,3,4,5',
    'gamma_decay': 1,
    'gamma_init': 0.1,
    'gamma_min': 0.1,
    'learning_rate': 0.0005,
    'load_model': True,
    'max_epochs': 20,
    'max_seq_length': 10,
    'max_train_size': -1,
    'model': '/storage/tmp/model',
    'n_filters': 128,
    'n_layers': 1,
    'online_testing': True,
    'output': '/storage/tmp/ods_dota.dev',
    'rho': 1,
    'steps_per_checkpoint': 1000,
    'test': '',
    'train': '/storage/data3/ods_dota',
    'vocab': '/storage/tmp/ods_dota.vocab'
}

args = namedtuple('args', args.keys())(*args.values())

vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)

tf.reset_default_graph()

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)

model = create_model(sess, args, vocab)
    
if args.beam > 1:
    decoder = beam_search.Decoder(sess, args, vocab, model)
else:
    decoder = greedy_decoding.Decoder(sess, args, vocab, model)

from collections import OrderedDict

def remove_duplicates(tokens):
    return [g[0] for g in groupby(tokens)]

def decode_sentences(sentences):
    return [' '.join(remove_duplicates(tokens)) for tokens in sentences]

def transfer(model, decoder, sess, args, vocab, inp):
    
    inp = inp.split()

    batch0 = get_batch([inp], [0], vocab.word2id)
    batch1 = get_batch([inp], [1], vocab.word2id)
    ori0, tsf0 = decoder.rewrite(batch0)
    ori1, tsf1 = decoder.rewrite(batch1)

    return ' '.join(w for w in ori0[0]), ' '.join(w for w in tsf0[0]), ' '.join(w for w in ori1[0]), ' '.join(w for w in tsf1[0])

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('msg')

from functools import wraps

def memoize(function):
    memo = {}
    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper

class DOTAnizer(Resource):
    
    @classmethod
    @memoize
    def dotanize(cls, msg):

        ori0, tsf0, ori1, tsf1 = transfer(model, decoder, sess, args, vocab, msg)
        print(ori0, tsf0, ori1, tsf1)
        
        return ori0, tsf0, ori1, tsf1

    def post(self):
        args = parser.parse_args()
        msg = args['msg'].encode('utf8')

        ori_ods, tsf_dota, ori_dota, tsf_ods = self.dotanize(msg)
        ori_ods = re.sub('<unk>|_unk_', '*' * np.random.randint(3,6), ori_ods)
        tsf_ods = re.sub('<unk>|_unk_', '*' * np.random.randint(3,6), tsf_ods)
        ori_dota = re.sub('<unk>|_unk_', '*' * np.random.randint(3,6), ori_dota)
        tsf_dota = re.sub('<unk>|_unk_', '*' * np.random.randint(3,6), tsf_dota)
   
        print("Msg: %s, DotA: %s, ODS: %s" % (msg, tsf_dota, tsf_ods))
        
        return {'dota': tsf_dota, 'ods': tsf_ods, 'orig_dota': ori_dota, 'orig_ods': ori_ods}
    

api.add_resource(DOTAnizer, '/')

    
if __name__ == '__main__':
    app.run()