import sys
sys.path.append("../")
# sys.path.append("../../")
#
# import os
# dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, '../deeplog')


import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor, Trainer
from logdeep.tools.utils import *

import warnings
warnings.filterwarnings('ignore')

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

options["output_dir"] = "../output/bgl/"
options["model_dir"] = options["output_dir"] + "bert/"
options["model_path"] = options["model_dir"] + "best_bert.pth"
options["train_vocab"] = options['output_dir'] + 'train'
options["vocab_path"] = options["output_dir"] + "vocab.pkl"

options["window_size"] = 128
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512 # for position embedding
options["min_len"] = 10

# TODO mask_ratio_bgl
options["mask_ratio"] = 0.5  # original
# options["mask_ratio"] = 0.1
# options["mask_ratio"] = 0.2
# options["mask_ratio"] = 0.3
# options["mask_ratio"] = 0.5
# options["mask_ratio"] = 0.7
# options["mask_ratio"] = 0.9
# options["mask_ratio"] = 0.6
# options["mask_ratio"] = 0.4
# options["mask_ratio"] = 0.45
# options["mask_ratio"] = 0.55
# options["mask_ratio"] = 0.65

options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

# features
options["is_logkey"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = False

options["scale"] = None # MinMaxScaler()
options["scale_path"] = options["model_dir"] + "scale.pkl"

# model
options["hidden"] = 256 # embedding size
# TODO num_layers_bgl
options["layers"] = 4  # original
# options["layers"] = 2
# options["layers"] = 5
# options["layers"] = 3
# options["layers"] = 6
options["attn_heads"] = 4

options["epochs"] = 200
# TODO n_epochs_stop
# options["n_epochs_stop"] = 10
options["n_epochs_stop"] = 20
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
# TODO num_workers_bgl
options["num_workers"] = 5  # original
# options["num_workers"] = 0
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
# TODO num_candidates_bgl
options["num_candidates"] = 15  # original
# options["num_candidates"] = 10
# options["num_candidates"] = 20
# options["num_candidates"] = 30
# options["num_candidates"] = 40
# options["num_candidates"] = 50
# options["num_candidates"] = 60
# options["num_candidates"] = 70
# options["num_candidates"] = 80
# options["num_candidates"] = 90
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)
print("device", options["device"])
print("features logkey:{} time: {}".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std", type=float, default=1)

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')
    vocab_parser.add_argument("-s", "--vocab_size", type=int, default=None)
    vocab_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    vocab_parser.add_argument("-m", "--min_freq", type=int, default=1)

    args = parser.parse_args()
    print("arguments", args)
    # Trainer(options).train()
    # Predictor(options).predict()

    if args.mode == 'train':
        Trainer(options).train()

    elif args.mode == 'predict':
        Predictor(options).predict()

    elif args.mode == 'vocab':
        with open(options["train_vocab"], 'r') as f:
            logs = f.readlines()
        vocab = WordVocab(logs)
        print("vocab_size", len(vocab))
        vocab.save_vocab(options["vocab_path"])





