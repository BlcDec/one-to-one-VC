# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob
import random
import os

import librosa
import numpy as np
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow import PrefetchData
from audio import read_wav, preemphasis, amp2db
from hparam import hparam as hp
from utils import normalize_0_1
from dtw import dtw
from numpy.linalg import norm


class DataFlow(RNGDataFlow):

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        self.wav_files = glob.glob(data_path)

    def __call__(self, n_prefetch=1000, n_thread=1):
        df = self
        df = BatchData(df, self.batch_size)
        df = PrefetchData(df, n_prefetch, n_thread)
        return df


class NetDataFlow(DataFlow):

    def get_data(self):
        while True:
            wav_file = random.choice(self.wav_files)
            yield get_parallel_mceps(wav_file)


def load_data(mode):
    wav_files = glob.glob(getattr(hp, mode).data_path)

    return wav_files


def wav_random_crop(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def get_mfccs_and_spectrogram(wav_file, trim=True, random_crop=False):
    '''This is applied in `train2`, `test2` or `convert` phase.
    '''

    # Load
    wav, _ = librosa.load(wav_file, sr=hp.default.sr)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=hp.default.win_length, hop_length=hp.default.hop_length)

    if random_crop:
        wav = wav_random_crop(wav, hp.default.sr, hp.default.duration)

    # Padding or crop
    length = hp.default.sr * hp.default.duration
    wav = librosa.util.fix_length(wav, length)

    return _get_mfcc_and_spec(wav, hp.default.preemphasis, hp.default.n_fft, hp.default.win_length,
                              hp.default.hop_length)


# TODO refactoring
def _get_mfcc_and_spec(wav, preemphasis_coeff, n_fft, win_length, hop_length):
    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.default.sr, hp.default.n_fft, hp.default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)

    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.default.max_db, hp.default.min_db)
    mel_db = normalize_0_1(mel_db, hp.default.max_db, hp.default.min_db)

    return mag_db.T, mel_db.T  # (t, 1+n_fft/2), (t, n_mels)


def get_parallel_mceps(wav1):
    hp.set_hparam_yaml('')
    _, file_name = os.path.split(wav1)
    wav2 = hp.train.p_path + file_name
    spec1, mceps1 = get_mfccs_and_spectrogram(wav1)
    spec2, mceps2 = get_mfccs_and_spectrogram(wav2)
    # D, wp = librosa.core.dtw(mceps1, mceps2, subseq=True)
    # last = wp.shape[0] - 1
    # paired_mceps1 = np.array([mceps1[wp[last][0]]])
    # paired_mceps2 = np.array([mceps2[wp[last][1]]])
    # for i in range(last - 1, -1, -1):
    #     paired_mceps1 = np.append(paired_mceps1, [mceps2[wp[i][1]]], axis=0)
    #     paired_mceps2 = np.append(paired_mceps2, [mceps2[wp[i][1]]], axis=0)
    # paired_mceps1 = paired_mceps1[0:80]
    # paired_mceps2 = paired_mceps2[0:80]
    # spec2 = spec2[0:80]
    _, _, _, path = dtw(mceps1, mceps2, dist=lambda x, y: norm(x - y, ord=1))
    last = path[0].size
    paired_mceps1 = np.array([mceps1[path[0][0]]])
    paired_mceps2 = np.array([mceps2[path[1][0]]])
    for i in range(1, last):
        paired_mceps1 = np.append(paired_mceps1, [mceps2[path[0][i]]], axis=0)
        paired_mceps2 = np.append(paired_mceps2, [mceps2[path[1][i]]], axis=0)
    paired_mceps1 = paired_mceps1[0:401]
    paired_mceps2 = paired_mceps2[0:401]
    return paired_mceps1, spec2, paired_mceps2, spec1
