from collections import defaultdict
import copy
import csv
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from os import path
from pydub import AudioSegment
from scipy.io import wavfile
import time
import torch
import torchvision


AudioSegment.converter = "ffmpeg"
allowed_languages = ["arabic","dutch","english","french","german","korean","mandarin","russian","spanish"]

def readAudio(mp3_path):
    '''
    Takes path of an .mp3 file, makes a .wav file if it
    doesn't already exist, and returns: frequencies
    and times (useful for plots), and required spectrogram 
    along with the sample rate
    '''
    
    # files             
    dst = mp3_path[:-4] + ".wav"

    # convert mp3 to wav   
    if not os.path.exists(dst):                                                         
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(dst, format="wav")

    # read wav and convert to spectrogram
    sr, data = wavfile.read(dst)
    #plt.plot(data)
    
    nperseg = int(sr * 0.001 * 20)
    frequencies, times, spectrogram = scipy.signal.spectrogram(data, sr, nperseg=nperseg, window=scipy.signal.hann(nperseg))

    return frequencies,times,spectrogram, sr
    
def writeAudio(spectrogram, sample_rate, output_file_name):
    '''
    For given spectrogram and sample_rate, write .wav audiofile
    with the name output_file_name + '.wav'
    '''
    audio_signal = librosa.core.spectrum.griffinlim(spectrogram)
    #print(audio_signal, audio_signal.shape)
    # write output
    file_name = output_file_name + '.wav'
    scipy.io.wavfile.write(file_name, sample_rate, np.array(audio_signal, dtype=np.int16))

class AccentDataset(torch.utils.data.Dataset):
    def __init__(self, path, csv_path):
        self.data_names = []
        self.classes = []
        self.encod_data = []

        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile,delimiter=',')
            for row in csv_reader:
                if (row[8] == "FALSE" and row[4] in allowed_languages):
                    mp3_path = os.path.join(path,row[3]+'.mp3')
                    self.data_names.append((mp3_path, row[4]))
                    if (row[4] not in self.classes):
                        self.classes.append(row[4])

            for data in self.data_names:
                label = torch.zeros(len(self.classes)) 
                label[self.classes.index(data[1])] = 1
                self.encod_data.append((data[0],label))

    def __getitem__(self,index):
        freq, time, dat, s_rate = readAudio(self.data_names[index][0])
        label = self.encod_data[index][1]
        return (freq, time, dat), label

    def __len__(self):
        return len(self.data_names)
    