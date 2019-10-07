# WaveFlow : A Compact Flow-based Model for Raw Audio

This is an unofficial PyTorch implementation of a paper "WaveFlow : A Compact Flow-based Model for Raw Audio".

Currently WIP. The implementation details may not be faithful.

# Requirements

PyTorch 1.1.0 or later (tested on 1.2.0) & python 3.6 & Librosa


# Examples

#### Step 1. Download Dataset

- LJSpeech : [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)

#### Step 2. Preprocessing (Preparing Mel Spectrogram)

`python preprocessing.py --in_dir /path/to/ljspeech/data/root --out_dir ./ljspeech_data`

#### Step 3. Train the Model

`python train.py --model_name waveflow_h8_r64 --n_height 8 --res_channels 64 --n_layer_per_cycle 1`

`python train.py --model_name waveflow_h64_r64 --n_height 64 --res_channels 64 --n_layer_per_cycle 5`

`python train.py --model_name waveflow_h32_r128 --n_height 32 --res_channels 128 --n_layer_per_cycle 3`

#### Step 4. Synthesize

Specify `--load_step` and `--num_samples` that looks like: 

`python synthesize.py --model_name waveflow_h8_r64 --n_height 8 --res_channels 64 --n_layer_per_cycle 1 --load_step 100000 --num_samples 5`

# References

- WaveNet vocoder : [https://github.com/r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)
- FloWaveNet : [https://github.com/ksw0306/FloWaveNet](https://github.com/ksw0306/FloWaveNet)
- WaveFlow: A Compact Flow-based Model for Raw Audio : [https://openreview.net/forum?id=Skeh1krtvH](https://openreview.net/forum?id=Skeh1krtvH)