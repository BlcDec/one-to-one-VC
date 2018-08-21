import sys
import requests
from pydub import AudioSegment
import librosa

import math
import numpy as np
import scipy.io.wavfile

import os

def get_int16(input_sound):
    return np.int16(input_sound/np.max(np.abs(input_sound)) * 32767)

def write_wav(wave_name, sigs, rate=16000):
    scipy.io.wavfile.write(wave_name, rate, get_int16(sigs))

def read_wav(filename):
    rate, data = scipy.io.wavfile.read(filename)
    #only use the 1st channel if stereo
    if len(data.shape) > 1:
        data =  data[:,0]
    data = data.astype(np.float32)
    data = data / 32768 #convert PCM int16 to float
    return data, rate

def wav_segmentation(in_sig, framesamp=320, hopsamp=160):
    sigLength = in_sig.shape[0]
    increment = framesamp/hopsamp
    M = int(math.floor(sigLength/hopsamp))
    a = np.zeros((M, framesamp), dtype=np.float32)
    for m in range(M):
        if m < increment -1:
            seg = in_sig[0:(m+1)*hopsamp]
            print(seg.shape)
            seg = seg * scipy.hamming(seg.shape[0])
            a[m,-len(seg):] = seg
        else:
            startpoint = (m + 1 - increment)*hopsamp;
            seg = in_sig[startpoint:startpoint+framesamp]
            # print seg.shape
            seg = seg * scipy.hamming(seg.shape[0])

            a[m,:] = seg
    return a


dir = 'datasets/guo/'
i=0
j=0
# #file = 'datasets/guo/1.wav'
new_dir = 'datasets/wang/'
save1 = 'datasets/new1/'
save2 = 'datasets/new2/'
for file in os.listdir(dir):
     sound, sr = librosa.load(dir+file, sr=None)
     time = librosa.get_duration(sound)
     if time >= 2:
         i = i+1
         sound1, sr1= librosa.load(new_dir+file, sr=None)
         time1 =librosa.get_duration(sound1)
         if time1 >= 2:
             j=j+1
             write_wav(save1+file, sound)
             write_wav(save2+file, sound1)
print(i)
print(j)
# print('end')

