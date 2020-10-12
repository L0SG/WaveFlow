## WaveFlow: A Compact Flow-based Model for Raw Audio

This is an unofficial PyTorch implementation of [WaveFlow] (Ping et al, ICML 2020) model.

The aim for this repo is to provide easy-to-use PyTorch version of WaveFlow as a drop-in alternative to various neural vocoder models used with NVIDIA's [Tacotron2] audio processing backend.

Please refer to the [official implementation] written in PaddlePaddle for the official results.

## Setup

1. Clone this repo and install requirements

   ```command
   git clone https://github.com/L0SG/WaveFlow.git
   cd WaveFlow
   pip install -r requirements.txt
   ```

2. Install [Apex] for mixed-precision training


## Train your model

1. Download [LJ Speech Data]. In this example it's in `data/`

2. Make a list of the file names to use for training/testing.

   ```command
   ls data/*.wav | tail -n+10 > train_files.txt
   ls data/*.wav | head -n10 > test_files.txt
   ```
    `-n+10` and `-n10` indicates that this example reserves the first 10 audio clips for model testing.

3. Edit the configuration file and train the model.

    Below are the example commands using `waveflow-h16-r64-bipartize.json`

   ```command
   nano configs/waveflow-h16-r64-bipartize.json
   python train.py -c configs/waveflow-h16-r64-bipartize.json
   ```
   Single-node multi-GPU training is automatically enabled with [DataParallel] (instead of [DistributedDataParallel] for simplicity).

   For mixed precision training, set `"fp16_run": true` on the configuration file.

   You can load the trained weights from saved checkpoints by providing the path to `checkpoint_path` variable in the config file.

   `checkpoint_path` accepts either explicit path, or the parent directory if resuming from averaged weights over multiple checkpoints.

   ### Examples
   insert `checkpoint_path: "experiments/waveflow-h16-r64-bipartize/waveflow_5000"` in the config file then run
   ```command
   python train.py -c configs/waveflow-h16-r64-bipartize.json
   ```

   for loading averaged weights over 10 recent checkpoints, insert `checkpoint_path: "experiments/waveflow-h16-r64-bipartize"` in the config file then run
   ```command
   python train.py -a 10 -c configs/waveflow-h16-r64-bipartize.json
   ```

   you can reset the optimizer and training scheduler (and keep the weights) by providing `--warm_start`
   ```command
   python train.py --warm_start -c configs/waveflow-h16-r64-bipartize.json
   ```
   
4. Synthesize waveform from the trained model.

   insert `checkpoint_path` in the config file and use `--synthesize` to `train.py`. The model generates waveform by looping over `test_files.txt`.
   ```command
   python train.py --synthesize -c configs/waveflow-h16-r64-bipartize.json
   ```
   if `fp16_run: true`, the model uses FP16 (half-precision) arithmetic for faster performance (on GPUs equipped with Tensor Cores).
   
   
## Reference
NVIDIA Tacotron2: https://github.com/NVIDIA/waveglow

NVIDIA WaveGlow: https://github.com/NVIDIA/waveglow

r9y9 wavenet-vocoder: https://github.com/r9y9/wavenet_vocoder

FloWaveNet: https://github.com/ksw0306/FloWaveNet

Parakeet: https://github.com/PaddlePaddle/Parakeet

[Tacotron2]: https://github.com/NVIDIA/tacotron2
[DataParallel]: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
[DistributedDataParallel]: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
[WaveFlow]: https://arxiv.org/abs/1912.01219
[LJ Speech Data]: https://keithito.com/LJ-Speech-Dataset
[Apex]: https://github.com/nvidia/apex
[official implementation]: https://github.com/PaddlePaddle/Parakeet
