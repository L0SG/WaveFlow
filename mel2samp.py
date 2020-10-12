# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from scipy.io.wavfile import read

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2_custom')
from tacotron2_custom.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, data_type, synth_mode, deterministic_mode, training_files, test_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, rescale=False, use_dbmel=False):
        self.data_type = data_type
        self.synth_mode = synth_mode  # get full waveform & mel instead of chunk
        self.deterministic_mode = deterministic_mode  # no random chunk. only get the first x samples. useful for eval loop

        assert self.data_type in ["train", "test"], "unknown data_type"
        if self.data_type == "train":
            self.audio_files = files_to_list(training_files)
            random.seed(1234)
            random.shuffle(self.audio_files)
        elif self.data_type == "test":
            self.audio_files = files_to_list(test_files)

        self.hop_length = hop_length
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

        self.rescale = rescale
        if self.rescale:
            print("INFO: audio rescaling is on. the audio is normalized to have range (-1, 1).")
        self.use_dbmel = use_dbmel
        if self.use_dbmel:
            print("INFO: using db-scale normalized mel-spec with range (0, 1).")

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE

        if self.rescale:
            audio_norm = audio_norm / audio_norm.abs().max() * 0.999

        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)

        if self.use_dbmel:
            melspec = self.stft.mel_spectrogram_dbver(audio_norm)
        else:
            melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        if not self.synth_mode:
        # Take segment
            if audio.size(0) >= self.segment_length:
                if not self.deterministic_mode:
                    max_audio_start = audio.size(0) - self.segment_length
                    audio_start = random.randint(0, max_audio_start)
                else:
                    audio_start = 0  # always get the first chunk
                audio = audio[audio_start:audio_start+self.segment_length]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        else:
            # full audio segment but with multiples of hop length for full-clip eval loop
            cut_len = audio.size(0) % self.hop_length
            audio = audio[:-cut_len]

        mel = self.get_mel(audio)
        audio = audio / MAX_WAV_VALUE

        if self.rescale:
            audio = audio / audio.abs().max() * 0.999

        if self.synth_mode: # also return waveform filename
            return(mel, audio, filename)
        else:
            return (mel, audio)


    def __len__(self):
        return len(self.audio_files)


class Mel2SampSeqDst(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, data_type, synth_mode, deterministic_mode, training_files, test_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, p_seqdst):
        self.data_type = data_type
        self.synth_mode = synth_mode  # get full waveform & mel instead of chunk
        self.deterministic_mode = deterministic_mode  # no random chunk. only get the first x samples. useful for eval loop

        assert self.data_type in ["train", "test"], "unknown data_type"
        if self.data_type == "train":
            self.audio_files = files_to_list(training_files)
            random.seed(1234)
            random.shuffle(self.audio_files)
        elif self.data_type == "test":
            self.audio_files = files_to_list(test_files)


        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

        from torch.distributions import Bernoulli
        self.p_seqdst = p_seqdst
        self.bernoulli = Bernoulli(self.p_seqdst)

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def get_fake_audio(self, filename):
        # replace filename with fake audio and load it
        rng = random.randint(0, 9)
        filename_fake = filename.replace('wavs', 'wavs-synth').replace('.wav', '-synth{}.wav'.format(rng))
        audio_fake, sampling_rate = load_wav_to_torch(filename_fake)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        return audio_fake

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        fake_mode = False
        if not self.synth_mode:
        # Take segment
            if self.data_type == "train" and self.bernoulli.sample().item():
                audio_fake = self.get_fake_audio(filename)
                fake_mode = True
            else:
                audio_fake = audio.clone()
            if audio.size(0) >= self.segment_length:
                if not self.deterministic_mode:
                    max_audio_start = audio.size(0) - self.segment_length
                    audio_start = random.randint(0, max_audio_start)
                else:
                    audio_start = 0  # always get the first chunk
                audio = audio[audio_start:audio_start+self.segment_length]
                audio_fake = audio_fake[audio_start:audio_start+self.segment_length]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
                audio_fake = torch.nn.functional.pad(audio_fake, (0, self.segment_length - audio_fake.size(0)), 'constant').data
            audio_fake = audio_fake / MAX_WAV_VALUE

        mel = self.get_mel(audio)
        audio = audio / MAX_WAV_VALUE

        if self.synth_mode: # also return waveform filename
            return(mel, audio, filename)
        else:
            if fake_mode:
                return (mel, audio_fake)
            else:
                return (mel, audio)


    def __len__(self):
        return len(self.audio_files)

# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp = Mel2Samp(**data_config)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio, sr = load_wav_to_torch(filepath)
        melspectrogram = mel2samp.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)
