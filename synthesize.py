import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data import LJspeechDataset, collate_fn, collate_fn_synthesize
from model import WaveFlow
from torch.distributions.normal import Normal
import numpy as np
import librosa
import os
import argparse
import time
import json
import gc

torch.backends.cudnn.benchmark = True
np.set_printoptions(precision=4)
torch.manual_seed(1111)

parser = argparse.ArgumentParser(description='Train WaveFlow of LJSpeech',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='/home/tkdrlf9202/Datasets/LJSpeech-preprocessed/',
                    help='Dataset Path')
parser.add_argument('--output_path', type=str, default='./output')

parser.add_argument('--model_name', type=str, default='waveflow', help='Model Name')
parser.add_argument('--load_step', type=int, default=0, help='Load Step')

parser.add_argument('--cin_channels', type=int, default=80, help='Cin Channels')
parser.add_argument('--res_channels', type=int, default=64, help='residual channels')
parser.add_argument('--n_height', type=int, default=64,
                    help='Number of height for 2D matrix conversion of 1D waveform. notated as h.')
parser.add_argument('--n_layer', type=int, default=8, help='Number of layers')
parser.add_argument('--n_flow', type=int, default=8, help='Number of layers')
parser.add_argument('--n_layer_per_cycle', type=int, default=5, help="number of layers inside a single flow for height dilation cycle."
                                                                     "ex: 3 with --n_layer=8 equals [1 2 4 1 2 4 1 2]"
                                                                     "ex2: 5 with --n_layer=8 equals [1 2 4 8 16 1 2 4]")
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPUs to use. >1 uses DataParallel')

parser.add_argument('--num_samples', type=int, default=10, help='# of audio samples')
parser.add_argument('--temp', type=float, default=1.0, help='Temperature')


args = parser.parse_args()

# auto-complete additional args for output subfolders
args.sample_path = os.path.join(args.output_path, 'samples')
args.param_path = os.path.join(args.output_path, 'params')
args.log_path = os.path.join(args.output_path, 'log')
args.loss_path = os.path.join(args.output_path, 'loss')

# Init logger
if not os.path.isdir(args.log_path):
    os.makedirs(args.log_path)
if not os.path.isdir(os.path.join(args.log_path, args.model_name)):
    os.makedirs(os.path.join(args.log_path, args.model_name))

# Checkpoint dir
if not os.path.isdir(args.param_path):
    os.makedirs(args.param_path)
if not os.path.isdir(args.loss_path):
    os.makedirs(args.loss_path)
if not os.path.isdir(args.sample_path):
    os.makedirs(args.sample_path)
if not os.path.isdir(os.path.join(args.sample_path, args.model_name)):
    os.makedirs(os.path.join(args.sample_path, args.model_name))
if not os.path.isdir(os.path.join(args.param_path, args.model_name)):
    os.makedirs(os.path.join(args.param_path, args.model_name))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# LOAD DATASETS
train_dataset = LJspeechDataset(args.data_path, True, 0.1)
test_dataset = LJspeechDataset(args.data_path, False, 0.1)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                          num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                         num_workers=args.num_workers, pin_memory=True)
synth_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_synthesize,
                          num_workers=args.num_workers, pin_memory=True)


def build_model():
    model = WaveFlow(in_channel=1,
                     cin_channel=args.cin_channels,
                     res_channel=args.res_channels,
                     n_height=args.n_height,
                     n_flow=args.n_flow,
                     n_layer=args.n_layer,
                     layers_per_dilation_h_cycle=args.n_layer_per_cycle,
                     )
    return model


def synthesize(model):
    global global_step
    model.eval()
    for batch_idx, (x, c) in enumerate(synth_loader):
        if batch_idx < args.num_samples:
            x, c = x.to(device), c.to(device)

            start_time = time.time()
            with torch.no_grad():
                y_gen = model.reverse(c, args.temp).squeeze()

            wav = y_gen.to(torch.device("cpu")).data.numpy()
            wav_name = '{}/{}/generate_{}_{}.wav'.format(args.sample_path, args.model_name, global_step, batch_idx)
            print('{} seconds'.format(time.time() - start_time))
            librosa.output.write_wav(wav_name, wav, sr=22050)
            print('{} Saved!'.format(wav_name))


def load_checkpoint(step, model):
    checkpoint_path = os.path.join(args.load, args.model_name, "checkpoint_step{:09d}.pth".format(step))
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    # generalized load procedure for both single-gpu and DataParallel models
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    except RuntimeError:
        print("INFO: this model is trained with DataParallel. Creating new state_dict without module...")
        state_dict = checkpoint["state_dict"]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model


if __name__ == "__main__":
    step = args.load_step
    global_step = step
    model = build_model()
    model = load_checkpoint(step, model)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        synthesize(model)
